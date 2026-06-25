"""Elastic parallelism and congestion control for Demeter V2."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

from .models import FunctionSpec, FunctionStatus, JobSpec, Observation


@dataclass
class ElasticityConfig:
    enabled: bool = True
    min_parallelism: int = 1
    max_parallelism: int = 256


@dataclass(frozen=True)
class DoPDecision:
    job_id: str
    stage_id: str
    old_parallelism: int
    new_parallelism: int
    alpha: float
    dc_name: str


class DoPTuner:
    """Balance triggered stages so their total DoP fits the orchestration wave."""

    def __init__(self, config: ElasticityConfig | None = None) -> None:
        self.config = config or ElasticityConfig()

    def tune_observation(self, observation: Observation, wave_size: int) -> List[DoPDecision]:
        if not self.config.enabled:
            return []
        all_decisions: List[DoPDecision] = []
        for dc_name, pending in observation.pending_by_dc.items():
            if len(pending) <= wave_size:
                continue
            stage_groups = self._group_pending_by_stage(pending)
            if len(stage_groups) > wave_size:
                stage_groups = dict(list(stage_groups.items())[:wave_size])
            decisions = self._decide_for_groups(stage_groups, wave_size, dc_name)
            all_decisions.extend(decisions)
        return self._synchronize_by_lowest_dop(all_decisions)

    def _group_pending_by_stage(
        self,
        functions: Sequence[FunctionSpec],
    ) -> Dict[Tuple[str, str], List[FunctionSpec]]:
        groups: Dict[Tuple[str, str], List[FunctionSpec]] = defaultdict(list)
        for fn in functions:
            if fn.status == FunctionStatus.PENDING:
                groups[(fn.job_id, fn.stage_id)].append(fn)
        return dict(groups)

    def _decide_for_groups(
        self,
        groups: Mapping[Tuple[str, str], Sequence[FunctionSpec]],
        wave_size: int,
        dc_name: str,
    ) -> List[DoPDecision]:
        if not groups:
            return []
        alphas = {
            key: max(sum(fn.input_mb for fn in fns), float(len(fns)))
            for key, fns in groups.items()
        }
        total_alpha = sum(alphas.values()) or 1.0
        remaining = max(wave_size, len(groups))
        raw: Dict[Tuple[str, str], float] = {
            key: max(self.config.min_parallelism, wave_size * alpha / total_alpha)
            for key, alpha in alphas.items()
        }
        floors = {
            key: min(self.config.max_parallelism, max(self.config.min_parallelism, int(math.floor(value))))
            for key, value in raw.items()
        }
        remaining -= sum(floors.values())

        fractions = sorted(
            raw,
            key=lambda key: raw[key] - math.floor(raw[key]),
            reverse=True,
        )
        idx = 0
        while remaining > 0 and fractions:
            key = fractions[idx % len(fractions)]
            if floors[key] < self.config.max_parallelism:
                floors[key] += 1
                remaining -= 1
            idx += 1

        decisions = []
        for key, fns in groups.items():
            job_id, stage_id = key
            old_parallelism = len(fns)
            new_parallelism = min(old_parallelism, floors[key])
            decisions.append(
                DoPDecision(
                    job_id=job_id,
                    stage_id=stage_id,
                    old_parallelism=old_parallelism,
                    new_parallelism=max(self.config.min_parallelism, new_parallelism),
                    alpha=alphas[key],
                    dc_name=dc_name,
                )
            )
        return decisions

    def _synchronize_by_lowest_dop(self, decisions: Sequence[DoPDecision]) -> List[DoPDecision]:
        by_stage: Dict[Tuple[str, str], DoPDecision] = {}
        for decision in decisions:
            key = (decision.job_id, decision.stage_id)
            current = by_stage.get(key)
            if current is None or decision.new_parallelism < current.new_parallelism:
                by_stage[key] = decision
        return list(by_stage.values())


def apply_dop_decisions(
    jobs: Mapping[str, JobSpec],
    decisions: Sequence[DoPDecision],
    now: float = 0.0,
) -> int:
    """Rewrite pending stage function graphs to match the synchronized DoP plan.

    The runtime Pheromone path would repartition buckets before invocation.
    In the simulator we model that before execution by retaining the target
    number of function nodes, redistributing stage data across them, and
    deleting overflow function nodes from the execution graph.
    """

    changed = 0
    for decision in decisions:
        job = jobs.get(decision.job_id)
        if job is None or decision.new_parallelism >= decision.old_parallelism:
            continue
        changed += _rewrite_stage_functions(job, decision, now)
        for stage in job.stages:
            if stage.stage_id == decision.stage_id:
                stage.parallelism = decision.new_parallelism
                stage.metadata["tuned_parallelism"] = float(decision.new_parallelism)
                break
    return changed


def _rewrite_stage_functions(job: JobSpec, decision: DoPDecision, now: float) -> int:
    stage_fns = sorted(
        [
            fn
            for fn in job.functions.values()
            if fn.stage_id == decision.stage_id and fn.status == FunctionStatus.PENDING
        ],
        key=lambda fn: fn.function_id,
    )
    new_parallelism = max(1, min(decision.new_parallelism, len(stage_fns)))
    kept = stage_fns[:new_parallelism]
    overflow = stage_fns[new_parallelism:]
    if not kept or not overflow:
        return 0

    original_input = sum(fn.input_mb for fn in stage_fns)
    original_output = sum(fn.output_mb for fn in stage_fns)
    original_sources = _sum_sources(stage_fns)
    predecessors = sorted({pred for fn in stage_fns for pred in fn.predecessors})
    successors = sorted({succ for fn in stage_fns for succ in fn.successors})
    removed_ids = {fn.function_id for fn in overflow}

    per_input = original_input / new_parallelism
    per_output = original_output / new_parallelism
    for idx, fn in enumerate(kept):
        fn.input_mb = per_input
        fn.output_mb = per_output
        fn.source_dcs = {name: mb / new_parallelism for name, mb in original_sources.items()}
        fn.predecessors = _predecessors_for_tuned_stage(predecessors, idx, new_parallelism)
        fn.successors = _successors_for_tuned_stage(successors, idx, new_parallelism)
        fn.metadata["dop_tuned"] = 1.0
        fn.metadata["dop_original_parallelism"] = float(decision.old_parallelism)
        fn.metadata["dop_tuned_at"] = now

    for fn in overflow:
        del job.functions[fn.function_id]

    _rewrite_neighbors(job, removed_ids, kept)
    return len(overflow)


def _sum_sources(functions: Sequence[FunctionSpec]) -> Dict[str, float]:
    sources: Dict[str, float] = {}
    for fn in functions:
        for dc_name, mb in fn.source_dcs.items():
            sources[dc_name] = sources.get(dc_name, 0.0) + mb
    return sources


def _predecessors_for_tuned_stage(
    predecessors: Sequence[str],
    idx: int,
    new_parallelism: int,
) -> List[str]:
    if not predecessors:
        return []
    if len(predecessors) == new_parallelism:
        return [predecessors[idx]]
    return list(predecessors)


def _successors_for_tuned_stage(
    successors: Sequence[str],
    idx: int,
    new_parallelism: int,
) -> List[str]:
    if not successors:
        return []
    if len(successors) == new_parallelism:
        return [successors[idx]]
    return list(successors)


def _rewrite_neighbors(
    job: JobSpec,
    removed_ids: set[str],
    kept: Sequence[FunctionSpec],
) -> None:
    kept_ids = [fn.function_id for fn in kept]
    for fn in job.functions.values():
        fn.predecessors = _rewrite_references(fn.predecessors, removed_ids, kept_ids)
        fn.successors = _rewrite_references(fn.successors, removed_ids, kept_ids)


def _rewrite_references(
    refs: Sequence[str],
    removed_ids: set[str],
    kept_ids: Sequence[str],
) -> List[str]:
    if not refs:
        return []
    out: List[str] = []
    had_removed = False
    for ref in refs:
        if ref in removed_ids:
            had_removed = True
            continue
        if ref not in out:
            out.append(ref)
    if had_removed:
        for kept_id in kept_ids:
            if kept_id not in out:
                out.append(kept_id)
    return out
