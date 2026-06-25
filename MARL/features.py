"""Feature extraction for DAGNN and HTGNN-style encoders."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .models import DataCenter, FunctionSpec, FunctionStatus, JobSpec, Observation


STATUS_INDEX = {
    FunctionStatus.RUNNING: 0,
    FunctionStatus.PENDING: 1,
    FunctionStatus.WAITING: 2,
    FunctionStatus.FINISHED: 3,
}


class FeatureEncoder:
    """Build compact numeric features from jobs and data centers."""

    def __init__(self, function_types: Sequence[str] | None = None) -> None:
        self.function_types = {name: idx for idx, name in enumerate(function_types or [])}

    @property
    def function_feature_dim(self) -> int:
        return 15

    @property
    def dc_feature_dim(self) -> int:
        return 8

    def encode_function(self, fn: FunctionSpec, job: JobSpec | None = None) -> np.ndarray:
        status = [0.0] * 4
        status[STATUS_INDEX[fn.status]] = 1.0
        type_id = self.function_types.setdefault(fn.function_type, len(self.function_types))
        alloc_cpu = fn.allocation.cpu_cores if fn.allocation else 0.0
        alloc_mem = fn.allocation.memory_mb if fn.allocation else 0.0
        elapsed = 0.0
        if fn.start_time is not None and fn.finish_time is None:
            elapsed = max(0.0, (job.arrival_time if job else 0.0) - fn.start_time)
        return np.array(
            [
                type_id / 32.0,
                fn.input_mb / 10240.0,
                fn.output_mb / 10240.0,
                fn.resource_peak_cpu / 8.0,
                fn.resource_peak_memory_mb / 10240.0,
                alloc_cpu / 8.0,
                alloc_mem / 10240.0,
                elapsed / 120.0,
                float(len(fn.predecessors)) / 64.0,
                float(len(fn.successors)) / 64.0,
                fn.cold_start_penalty / 10.0,
            ]
            + status,
            dtype=np.float32,
        )

    def encode_dc(self, dc: DataCenter, data_centers: Sequence[DataCenter]) -> np.ndarray:
        bandwidths = [dc.bandwidth_to(other.name) for other in data_centers if other.name != dc.name]
        avg_bw = 0.0 if not bandwidths else float(np.mean([min(v, 10000.0) for v in bandwidths]))
        return np.array(
            [
                dc.running_functions / 256.0,
                dc.warm_containers / 256.0,
                dc.available_executors / 256.0,
                avg_bw / 10000.0,
                dc.price_core_second * 100000.0,
                dc.price_gb_second * 100000.0,
                dc.request_price * 1000000.0,
                dc.storage_gb_second * 1000000000.0,
            ],
            dtype=np.float32,
        )

    def build_observation_tensors(self, obs: Observation) -> Dict[str, np.ndarray]:
        job_by_id = {job.job_id: job for job in obs.jobs}
        pending = obs.all_pending_functions()
        function_features = []
        for fn in pending:
            function_features.append(self.encode_function(fn, job_by_id.get(fn.job_id)))
        if function_features:
            fn_tensor = np.stack(function_features, axis=0)
        else:
            fn_tensor = np.zeros((0, self.function_feature_dim), dtype=np.float32)

        dc_features = [self.encode_dc(dc, obs.data_centers) for dc in obs.data_centers]
        dc_tensor = np.stack(dc_features, axis=0) if dc_features else np.zeros((0, self.dc_feature_dim), dtype=np.float32)
        return {"functions": fn_tensor, "data_centers": dc_tensor}
