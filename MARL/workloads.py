"""Synthetic workload generation following the paper's evaluation setup."""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Mapping, Sequence


TPC_DS_Q94 = {
    "stages": [
        {"id": "scan_store_sales", "function": "scan", "parallelism": 12, "input_mb": 640, "output_mb": 180},
        {"id": "filter_join", "function": "join", "parallelism": 8, "trigger": "many_to_one", "input_mb": 180, "output_mb": 96},
        {"id": "aggregate", "function": "aggregate", "parallelism": 4, "trigger": "many_to_one", "input_mb": 96, "output_mb": 8},
        {"id": "finalize", "function": "finalize", "parallelism": 1, "trigger": "many_to_one", "input_mb": 8, "output_mb": 1},
    ],
    "edges": [
        ("scan_store_sales", "filter_join"),
        ("filter_join", "aggregate"),
        ("aggregate", "finalize"),
    ],
}

BIGDATA_Q3 = {
    "stages": [
        {"id": "read_logs", "function": "scan", "parallelism": 16, "input_mb": 900, "output_mb": 250},
        {"id": "map_tokens", "function": "map", "parallelism": 16, "trigger": "direct", "input_mb": 250, "output_mb": 160},
        {"id": "reduce_tokens", "function": "reduce", "parallelism": 4, "trigger": "many_to_one", "input_mb": 160, "output_mb": 16},
    ],
    "edges": [("read_logs", "map_tokens"), ("map_tokens", "reduce_tokens")],
}


def generate_job_profiles(
    count: int,
    dc_names: Sequence[str],
    mode: str = "normal",
    seed: int = 1,
) -> List[Mapping]:
    rnd = random.Random(seed)
    profiles = []
    interval = {"slow": 72.0, "normal": 22.5, "burst": 9.0}.get(mode, 22.5)
    templates = [TPC_DS_Q94, BIGDATA_Q3]
    for idx in range(count):
        template = templates[idx % len(templates)]
        scale = rnd.uniform(0.65, 1.5)
        stages = []
        source_dc = rnd.sample(list(dc_names), k=min(max(1, len(dc_names) // 2), len(dc_names)))
        source_split = {name: 1.0 / len(source_dc) for name in source_dc}
        for stage in template["stages"]:
            item = dict(stage)
            item["input_mb"] = max(1.0, stage["input_mb"] * scale)
            item["output_mb"] = max(0.5, stage["output_mb"] * scale)
            item["candidate_dcs"] = list(dc_names)
            item["source_dcs"] = {
                name: item["input_mb"] * share for name, share in source_split.items()
            }
            item["metadata"] = {
                "type_weight": {"scan": 1.0, "map": 0.8, "join": 1.25, "reduce": 1.1, "aggregate": 0.9, "finalize": 0.45}.get(
                    item["function"], 1.0
                ),
                "cold_start_seconds": rnd.uniform(0.2, 0.8),
            }
            stages.append(item)
        slo = rnd.uniform(40.0, 140.0) if idx % 2 == 0 else rnd.uniform(40.0, 80.0)
        profiles.append(
            {
                "job_id": f"job-{idx}",
                "slo_seconds": slo,
                "arrival_time": idx * interval,
                "priority": 1.0,
                "stages": stages,
                "edges": template["edges"],
                "candidate_dcs": list(dc_names),
            }
        )
    return profiles
