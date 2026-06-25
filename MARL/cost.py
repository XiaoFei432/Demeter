"""Duration and cost model used by Demeter's simulator and policy heuristics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

from .models import AllocationType, DataCenter, FunctionSpec


@dataclass
class FunctionMetrics:
    waiting_time: float
    compute_time: float
    transfer_time: float
    cold_start_time: float
    duration: float
    operating_cost: float
    transfer_cost: float
    storage_cost: float

    @property
    def total_cost(self) -> float:
        return self.operating_cost + self.transfer_cost + self.storage_cost


def valid_allocation(
    fn: FunctionSpec,
    allocation: AllocationType,
    memory_loss_factor: float = 0.8,
) -> bool:
    """Return whether the allocation satisfies the paper's memory constraint."""

    required = fn.input_mb / max(memory_loss_factor, 1e-6)
    return allocation.memory_mb >= max(required, 1.0)


def estimate_transfer_time(
    fn: FunctionSpec,
    dc: DataCenter,
    data_centers: Mapping[str, DataCenter],
) -> float:
    seconds = 0.0
    for source, mb in fn.source_dcs.items():
        if source == dc.name:
            continue
        bandwidth = data_centers.get(source, dc).bandwidth_to(dc.name)
        seconds += (mb * 8.0) / max(bandwidth, 1e-6)
    return seconds


def estimate_transfer_cost(
    fn: FunctionSpec,
    dc: DataCenter,
    data_centers: Mapping[str, DataCenter],
) -> float:
    cost = 0.0
    for source, mb in fn.source_dcs.items():
        if source == dc.name:
            continue
        src_dc = data_centers.get(source, dc)
        cost += (mb / 1024.0) * src_dc.transfer_price_to(dc.name)
    return cost


def estimate_compute_time(fn: FunctionSpec, allocation: AllocationType) -> float:
    """Black-box-inspired duration estimate for analytics functions.

    The function intentionally stays simple and observable. It captures the
    relation used throughout the paper: data size, CPU, memory, and function
    type shape the duration, while the learned model can correct it online.
    """

    base = float(fn.metadata.get("base_seconds", 0.05))
    cpu_sensitivity = float(fn.metadata.get("cpu_sensitivity", 0.65))
    memory_sensitivity = float(fn.metadata.get("memory_sensitivity", 0.35))
    type_weight = float(fn.metadata.get("type_weight", 1.0))
    input_mb = max(fn.input_mb, 1.0)
    memory_term = math.sqrt(max(allocation.memory_mb, 1) / 1024.0)
    cpu_term = max(allocation.cpu_cores, 0.1) ** cpu_sensitivity
    return type_weight * (base + input_mb / (45.0 * cpu_term * max(memory_term, 0.1) ** memory_sensitivity))


def estimate_metrics(
    fn: FunctionSpec,
    dc: DataCenter,
    allocation: AllocationType,
    data_centers: Mapping[str, DataCenter],
    now: float,
    timeout_seconds: float = 120.0,
) -> FunctionMetrics:
    waiting = max(0.0, now - fn.arrival_time)
    transfer_time = estimate_transfer_time(fn, dc, data_centers)
    compute_time = estimate_compute_time(fn, allocation)
    cold_start = 0.0 if dc.warm_containers > 0 else float(fn.metadata.get("cold_start_seconds", 0.4))
    cold_start += fn.cold_start_penalty
    duration = waiting + transfer_time + compute_time + cold_start
    duration = min(duration, timeout_seconds * 1.5)

    operating = duration * (
        dc.price_core_second * allocation.cpu_cores + dc.price_gb_second * allocation.memory_gb
    ) + dc.request_price
    transfer = estimate_transfer_cost(fn, dc, data_centers)
    storage = max(fn.output_mb, 0.0) / 1024.0 * dc.storage_gb_second * max(compute_time, 0.001)
    return FunctionMetrics(
        waiting_time=waiting,
        compute_time=compute_time,
        transfer_time=transfer_time,
        cold_start_time=cold_start,
        duration=duration,
        operating_cost=operating,
        transfer_cost=transfer,
        storage_cost=storage,
    )
