"""Configuration pruning for the TON 2025 Demeter policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .cost import valid_allocation
from .history import InvocationHistory, InvocationRecord
from .models import AllocationType, FunctionSpec, JobSpec


@dataclass
class PruningConfig:
    enabled: bool = True
    overload_threshold: float = 0.8
    input_bucket_ratio: float = 0.25
    history_limit: int = 8
    min_pool_size: int = 1


class ConfigurationPruner:
    """Paper Algorithm 1 implemented over a sorted allocation grid."""

    def __init__(
        self,
        allocations: Sequence[AllocationType],
        config: PruningConfig | None = None,
        memory_loss_factor: float = 0.8,
    ) -> None:
        self.allocations = sorted(allocations, key=lambda item: (item.memory_mb, item.cpu_cores))
        self.config = config or PruningConfig()
        self.memory_loss_factor = memory_loss_factor

    def prune(
        self,
        fn: FunctionSpec,
        job: JobSpec | None,
        *,
        affinity: float,
        now: float,
        history: InvocationHistory,
    ) -> List[AllocationType]:
        valid = [alloc for alloc in self.allocations if valid_allocation(fn, alloc, self.memory_loss_factor)]
        if not self.config.enabled or not valid:
            return valid

        pool = list(valid)
        progress = self._job_progress(fn, job, now)
        current_score = max(0.0, affinity) * (1.0 - progress)
        records = history.query(
            fn.stage_id,
            fn.function_type,
            fn.input_mb,
            window_ratio=self.config.input_bucket_ratio,
            limit=self.config.history_limit,
        )

        for record in records:
            next_pool = self._apply_record(pool, record, current_score)
            if len(next_pool) < self.config.min_pool_size:
                break
            pool = next_pool
        return pool

    def _apply_record(
        self,
        pool: Sequence[AllocationType],
        record: InvocationRecord,
        current_score: float,
    ) -> List[AllocationType]:
        baseline = record.allocation
        if record.memory_pressure >= self.config.overload_threshold:
            return self._upper_segment(pool, baseline)
        if record.urgency_score > current_score:
            return self._upper_segment(pool, baseline)
        return self._lower_segment(pool, baseline)

    def _upper_segment(
        self,
        pool: Sequence[AllocationType],
        baseline: AllocationType,
    ) -> List[AllocationType]:
        out = [
            alloc
            for alloc in pool
            if (alloc.memory_mb, alloc.cpu_cores) >= (baseline.memory_mb, baseline.cpu_cores)
        ]
        return out or list(pool)

    def _lower_segment(
        self,
        pool: Sequence[AllocationType],
        baseline: AllocationType,
    ) -> List[AllocationType]:
        out = [
            alloc
            for alloc in pool
            if (alloc.memory_mb, alloc.cpu_cores) <= (baseline.memory_mb, baseline.cpu_cores)
        ]
        return out or list(pool)

    def _job_progress(self, fn: FunctionSpec, job: JobSpec | None, now: float) -> float:
        if job is None:
            return 0.0
        elapsed = max(0.0, now - job.arrival_time)
        return min(1.0, elapsed / max(job.slo_seconds, 1e-6))
