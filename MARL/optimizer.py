"""DAG optimizer for expanding Pheromone-style job profiles."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .models import FunctionSpec, FunctionStatus, JobSpec, StageSpec


class DAGOptimizer:
    """Generate Demeter's function-level execution graph.

    Pheromone exposes stages and trigger primitives. Demeter needs a finer
    function-level graph so that placement and resource actions can be taken
    per parallel function. The optimizer keeps stage metadata, expands
    parallelism, and derives one-to-one or all-to-all links between functions.
    """

    def __init__(self, default_parallelism: int = 1) -> None:
        self.default_parallelism = default_parallelism

    def build_job(self, raw: Mapping) -> JobSpec:
        stages: List[StageSpec] = []
        for item in raw.get("stages", []):
            stage = StageSpec(
                stage_id=str(item["id"]),
                function_type=str(item.get("function", item.get("type", item["id"]))),
                parallelism=int(item.get("parallelism", self.default_parallelism)),
                trigger=str(item.get("trigger", "direct")),
                input_mb=float(item.get("input_mb", 1.0)),
                output_mb=float(item.get("output_mb", item.get("input_mb", 1.0))),
                candidate_dcs=list(item.get("candidate_dcs", raw.get("candidate_dcs", []))),
                source_dcs=dict(item.get("source_dcs", raw.get("source_dcs", {}))),
                metadata=dict(item.get("metadata", {})),
            )
            stages.append(stage)

        job = JobSpec(
            job_id=str(raw["job_id"]),
            slo_seconds=float(raw.get("slo_seconds", raw.get("slo", 60.0))),
            arrival_time=float(raw.get("arrival_time", 0.0)),
            priority=float(raw.get("priority", 1.0)),
            stages=stages,
            edges=[(str(a), str(b)) for a, b in raw.get("edges", [])],
        )
        self.expand_execution_graph(job)
        return job

    def expand_execution_graph(self, job: JobSpec) -> JobSpec:
        stage_map = {stage.stage_id: stage for stage in job.stages}
        funcs: Dict[str, FunctionSpec] = {}

        for stage in job.stages:
            per_fn_input = stage.input_mb / max(stage.parallelism, 1)
            per_fn_output = stage.output_mb / max(stage.parallelism, 1)
            for idx in range(stage.parallelism):
                function_id = f"{job.job_id}:{stage.stage_id}:{idx}"
                funcs[function_id] = FunctionSpec(
                    function_id=function_id,
                    job_id=job.job_id,
                    stage_id=stage.stage_id,
                    function_type=stage.function_type,
                    input_mb=per_fn_input,
                    output_mb=per_fn_output,
                    status=FunctionStatus.PENDING if self._is_source_stage(job, stage.stage_id) else FunctionStatus.WAITING,
                    candidate_dcs=list(stage.candidate_dcs),
                    source_dcs=dict(stage.source_dcs),
                    arrival_time=job.arrival_time,
                    metadata=dict(stage.metadata),
                )

        for src_stage, dst_stage in job.edges:
            src_fns = [fn for fn in funcs.values() if fn.stage_id == src_stage]
            dst_fns = [fn for fn in funcs.values() if fn.stage_id == dst_stage]
            trigger = stage_map[dst_stage].trigger if dst_stage in stage_map else "direct"
            if trigger in ("direct", "one_to_one") and len(src_fns) == len(dst_fns):
                pairs = zip(sorted(src_fns, key=lambda x: x.function_id), sorted(dst_fns, key=lambda x: x.function_id))
            else:
                pairs = ((src, dst) for src in src_fns for dst in dst_fns)
            for src, dst in pairs:
                src.successors.append(dst.function_id)
                dst.predecessors.append(src.function_id)

        job.functions = funcs
        return job

    def release_ready_functions(self, job: JobSpec, now: float) -> List[FunctionSpec]:
        released: List[FunctionSpec] = []
        for fn in job.functions.values():
            if fn.status != FunctionStatus.WAITING:
                continue
            if all(job.functions[p].status == FunctionStatus.FINISHED for p in fn.predecessors):
                fn.status = FunctionStatus.PENDING
                fn.arrival_time = now
                released.append(fn)
        return released

    def topological_batches(self, job: JobSpec) -> List[List[str]]:
        indegree = {fid: len(fn.predecessors) for fid, fn in job.functions.items()}
        children: Dict[str, List[str]] = defaultdict(list)
        for fid, fn in job.functions.items():
            for succ in fn.successors:
                children[fid].append(succ)

        queue = deque(sorted([fid for fid, degree in indegree.items() if degree == 0]))
        batches: List[List[str]] = []
        while queue:
            batch = list(queue)
            queue.clear()
            batches.append(batch)
            for fid in batch:
                for succ in children[fid]:
                    indegree[succ] -= 1
                    if indegree[succ] == 0:
                        queue.append(succ)
        return batches

    def _is_source_stage(self, job: JobSpec, stage_id: str) -> bool:
        return stage_id not in {dst for _, dst in job.edges}
