"""Shared data structures used by the Demeter controller and simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class FunctionStatus(str, Enum):
    """Lifecycle state of a function in a serverless analytics job."""

    WAITING = "waiting"
    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass(frozen=True)
class AllocationType:
    """CPU-memory configuration for one function invocation."""

    name: str
    cpu_cores: float
    memory_mb: int

    @property
    def memory_gb(self) -> float:
        return self.memory_mb / 1024.0


@dataclass
class DataCenter:
    """Runtime and price information for a geo-distributed worker region."""

    name: str
    node_ip: str
    region: str = ""
    running_functions: int = 0
    warm_containers: int = 0
    available_executors: int = 0
    price_core_second: float = 0.00001667
    price_gb_second: float = 0.00000965
    request_price: float = 0.0000002
    storage_gb_second: float = 0.0000000008
    bandwidth_mbps: Dict[str, float] = field(default_factory=dict)
    transfer_price_gb: Dict[str, float] = field(default_factory=dict)

    def bandwidth_to(self, other: str) -> float:
        if other == self.name:
            return float("inf")
        return max(self.bandwidth_mbps.get(other, 100.0), 1e-6)

    def transfer_price_to(self, other: str) -> float:
        if other == self.name:
            return 0.0
        return self.transfer_price_gb.get(other, 0.02)


@dataclass
class FunctionSpec:
    """Function-level node in Demeter's execution graph."""

    function_id: str
    job_id: str
    stage_id: str
    function_type: str
    input_mb: float
    output_mb: float = 0.0
    status: FunctionStatus = FunctionStatus.WAITING
    candidate_dcs: List[str] = field(default_factory=list)
    source_dcs: Dict[str, float] = field(default_factory=dict)
    placement: Optional[str] = None
    allocation: Optional[AllocationType] = None
    arrival_time: float = 0.0
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    observed_duration: Optional[float] = None
    resource_peak_memory_mb: float = 0.0
    resource_peak_cpu: float = 0.0
    cold_start_penalty: float = 0.0
    predecessors: List[str] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)
    metadata: Dict[str, float] = field(default_factory=dict)

    def clone_for_wave(self) -> "FunctionSpec":
        return FunctionSpec(
            function_id=self.function_id,
            job_id=self.job_id,
            stage_id=self.stage_id,
            function_type=self.function_type,
            input_mb=self.input_mb,
            output_mb=self.output_mb,
            status=self.status,
            candidate_dcs=list(self.candidate_dcs),
            source_dcs=dict(self.source_dcs),
            placement=self.placement,
            allocation=self.allocation,
            arrival_time=self.arrival_time,
            start_time=self.start_time,
            finish_time=self.finish_time,
            observed_duration=self.observed_duration,
            resource_peak_memory_mb=self.resource_peak_memory_mb,
            resource_peak_cpu=self.resource_peak_cpu,
            cold_start_penalty=self.cold_start_penalty,
            predecessors=list(self.predecessors),
            successors=list(self.successors),
            metadata=dict(self.metadata),
        )


@dataclass
class StageSpec:
    """Stage-level node before expansion into parallel functions."""

    stage_id: str
    function_type: str
    parallelism: int
    trigger: str = "direct"
    input_mb: float = 0.0
    output_mb: float = 0.0
    candidate_dcs: List[str] = field(default_factory=list)
    source_dcs: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class JobSpec:
    """A geo-distributed serverless analytics job."""

    job_id: str
    slo_seconds: float
    arrival_time: float = 0.0
    priority: float = 1.0
    stages: List[StageSpec] = field(default_factory=list)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    functions: Dict[str, FunctionSpec] = field(default_factory=dict)

    def active_functions(self) -> List[FunctionSpec]:
        return [
            fn
            for fn in self.functions.values()
            if fn.status in (FunctionStatus.RUNNING, FunctionStatus.PENDING, FunctionStatus.WAITING)
        ]

    def pending_functions(self) -> List[FunctionSpec]:
        return [fn for fn in self.functions.values() if fn.status == FunctionStatus.PENDING]

    def finished(self) -> bool:
        return bool(self.functions) and all(
            fn.status == FunctionStatus.FINISHED for fn in self.functions.values()
        )

    def jct(self) -> float:
        if not self.functions:
            return 0.0
        finish = max((fn.finish_time or self.arrival_time) for fn in self.functions.values())
        return max(0.0, finish - self.arrival_time)


@dataclass(frozen=True)
class OrchestrationAction:
    """One function placement and allocation decision."""

    function_id: str
    dc_name: str
    allocation: AllocationType
    posterior: float = 0.0
    allocation_score: float = 0.0


@dataclass
class Observation:
    """Snapshot consumed by the policy at one orchestration interval."""

    now: float
    jobs: List[JobSpec]
    data_centers: List[DataCenter]
    pending_by_dc: Dict[str, List[FunctionSpec]]
    running: List[FunctionSpec] = field(default_factory=list)
    history: List[Mapping[str, float]] = field(default_factory=list)

    @property
    def data_center_map(self) -> Dict[str, DataCenter]:
        return {dc.name: dc for dc in self.data_centers}

    def all_pending_functions(self) -> List[FunctionSpec]:
        seen = set()
        pending: List[FunctionSpec] = []
        for funcs in self.pending_by_dc.values():
            for fn in funcs:
                if fn.function_id not in seen:
                    seen.add(fn.function_id)
                    pending.append(fn)
        return pending


def allocation_grid(
    cpu_values: Sequence[float] = (1, 2, 4, 8),
    memory_values_mb: Sequence[int] = (128, 512, 1024, 2048, 4096, 8192, 10240),
) -> List[AllocationType]:
    """Build the decoupled CPU-memory action space used by Demeter."""

    grid = []
    for cpu in cpu_values:
        for memory in memory_values_mb:
            name = f"{cpu:g}c-{memory}m"
            grid.append(AllocationType(name=name, cpu_cores=float(cpu), memory_mb=int(memory)))
    return grid


def functions_by_id(functions: Iterable[FunctionSpec]) -> Dict[str, FunctionSpec]:
    return {fn.function_id: fn for fn in functions}
