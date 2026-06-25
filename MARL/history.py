"""Invocation history used by the TON 2025 Demeter extensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from .models import AllocationType, FunctionSpec


@dataclass(frozen=True)
class InvocationRecord:
    """Compact record of one finished function invocation."""

    stage_id: str
    function_type: str
    input_mb: float
    allocation: AllocationType
    peak_memory_mb: float
    peak_cpu: float
    duration: float
    affinity: float
    progress: float
    placement: str = ""

    @property
    def memory_pressure(self) -> float:
        return self.peak_memory_mb / max(float(self.allocation.memory_mb), 1.0)

    @property
    def urgency_score(self) -> float:
        return self.affinity * (1.0 - self.progress)


class InvocationHistory:
    """In-memory history store grouped by stage/function and input-size class."""

    def __init__(self, bucket_ratio: float = 0.25, max_records: int = 4096) -> None:
        self.bucket_ratio = max(bucket_ratio, 1e-6)
        self.max_records = max(1, int(max_records))
        self._records: List[InvocationRecord] = []

    def __len__(self) -> int:
        return len(self._records)

    def add(self, record: InvocationRecord) -> None:
        self._records.append(record)
        overflow = len(self._records) - self.max_records
        if overflow > 0:
            del self._records[:overflow]

    def add_from_function(
        self,
        fn: FunctionSpec,
        *,
        job_progress: float,
        affinity: Optional[float] = None,
    ) -> None:
        if fn.allocation is None or fn.observed_duration is None:
            return
        self.add(
            InvocationRecord(
                stage_id=fn.stage_id,
                function_type=fn.function_type,
                input_mb=fn.input_mb,
                allocation=fn.allocation,
                peak_memory_mb=fn.resource_peak_memory_mb,
                peak_cpu=fn.resource_peak_cpu,
                duration=fn.observed_duration,
                affinity=float(affinity if affinity is not None else fn.metadata.get("affinity", 0.5)),
                progress=max(0.0, min(1.0, job_progress)),
                placement=fn.placement or "",
            )
        )

    def query(
        self,
        stage_id: str,
        function_type: str,
        input_mb: float,
        *,
        window_ratio: Optional[float] = None,
        limit: int = 8,
    ) -> List[InvocationRecord]:
        ratio = self.bucket_ratio if window_ratio is None else max(float(window_ratio), 1e-6)
        lower = input_mb * (1.0 - ratio)
        upper = input_mb * (1.0 + ratio)
        matches = [
            rec
            for rec in reversed(self._records)
            if rec.stage_id == stage_id
            and rec.function_type == function_type
            and lower <= rec.input_mb <= upper
        ]
        return matches[: max(0, int(limit))]

    def records(self) -> Iterable[InvocationRecord]:
        return tuple(self._records)
