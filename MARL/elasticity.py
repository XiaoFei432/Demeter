"""Elastic parallelism and congestion control for Demeter V2."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

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
    """Coalesce pending function requests to the synchronized DoP plan.

    The runtime Pheromone path would repartition buckets before invocation.
    In the simulator we model that by merging the data assigned to overflow
    functions into the retained functions and marking the overflow nodes as
    already represented by the tuned stage.
    """

    changed = 0
    for decision in decisions:
        job = jobs.get(decision.job_id)
        if job is None or decision.new_parallelism >= decision.old_parallelism:
            continue
        pending = sorted(
            [
                fn
                for fn in job.functions.values()
                if fn.stage_id == decision.stage_id and fn.status == FunctionStatus.PENDING
            ],
            key=lambda fn: fn.function_id,
        )
        kept = pending[: decision.new_parallelism]
        overflow = pending[decision.new_parallelism :]
        if not kept:
            continue
        for idx, fn in enumerate(overflow):
            target = kept[idx % len(kept)]
            target.input_mb += fn.input_mb
            target.output_mb += fn.output_mb
            for dc_name, mb in fn.source_dcs.items():
                target.source_dcs[dc_name] = target.source_dcs.get(dc_name, 0.0) + mb
            for successor_id in list(fn.successors):
                successor = job.functions.get(successor_id)
                if successor is None:
                    continue
                successor.predecessors = [
                    target.function_id if pred == fn.function_id else pred
                    for pred in successor.predecessors
                ]
                if target.function_id not in successor.predecessors:
                    successor.predecessors.append(target.function_id)
                if successor_id not in target.successors:
                    target.successors.append(successor_id)
            fn.successors.clear()
            fn.status = FunctionStatus.FINISHED
            fn.start_time = now
            fn.finish_time = now
            fn.observed_duration = 0.0
            fn.metadata["dop_coalesced"] = 1.0
            fn.metadata["dop_target"] = float(pending.index(target))
            changed += 1
        for stage in job.stages:
            if stage.stage_id == decision.stage_id:
                stage.metadata["tuned_parallelism"] = float(decision.new_parallelism)
                break
    return changed
