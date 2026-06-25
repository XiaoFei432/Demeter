"""Multi-process Python function driver used by Demeter functions."""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence


def _run_chunk(args):
    func, chunk = args
    return func(chunk)


@dataclass
class MultiProcessFunctionDriver:
    """Split inputs according to allocated CPU cores and fork workers."""

    cpu_cores: int

    def run(self, func: Callable[[Sequence], object], data: Sequence) -> List[object]:
        workers = max(1, int(self.cpu_cores))
        if workers == 1 or len(data) <= 1:
            return [func(data)]
        chunks = [data[i::workers] for i in range(workers)]
        with mp.Pool(processes=workers) as pool:
            return pool.map(_run_chunk, [(func, chunk) for chunk in chunks])


def cpu_cores_from_env(default: int = 1) -> int:
    raw = os.environ.get("DEMETER_CPU_CORES") or os.environ.get("CPU_CORES")
    try:
        return max(1, int(float(raw))) if raw else default
    except ValueError:
        return default
