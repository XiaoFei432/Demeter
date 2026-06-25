"""Configuration loading helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime by CLI messages
    yaml = None

from .models import AllocationType, DataCenter, allocation_grid


@dataclass
class ModelConfig:
    embedding_dim: int = 64
    gnn_layers: int = 3
    temporal_window: int = 5
    transformer_layers: int = 3
    transformer_heads: int = 8
    score_hidden: List[int] = field(default_factory=lambda: [32, 16])
    lr: float = 3e-4
    gamma: float = 1.0
    clip_epsilon: float = 0.2
    penalty_beta: float = 0.3
    timeout_seconds: float = 120.0


@dataclass
class RuntimeConfig:
    interval_seconds: float = 5.0
    max_wave_size: int = 20
    memory_loss_factor: float = 0.8
    enable_cluster_rpc: bool = False
    policy_path: Optional[str] = None


@dataclass
class DemeterConfig:
    data_centers: List[DataCenter]
    allocations: List[AllocationType]
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _require_yaml() -> None:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required. Install dependencies with "
            "`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt`."
        )


def load_config(path: str | Path) -> DemeterConfig:
    """Load a Demeter YAML configuration file."""

    _require_yaml()
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    dcs = [
        DataCenter(
            name=item["name"],
            node_ip=item.get("node_ip", item.get("ip", item["name"])),
            region=item.get("region", item.get("name", "")),
            running_functions=int(item.get("running_functions", 0)),
            warm_containers=int(item.get("warm_containers", 0)),
            available_executors=int(item.get("available_executors", 0)),
            price_core_second=float(item.get("price_core_second", 0.00001667)),
            price_gb_second=float(item.get("price_gb_second", 0.00000965)),
            request_price=float(item.get("request_price", 0.0000002)),
            storage_gb_second=float(item.get("storage_gb_second", 0.0000000008)),
            bandwidth_mbps=dict(item.get("bandwidth_mbps", {})),
            transfer_price_gb=dict(item.get("transfer_price_gb", {})),
        )
        for item in raw.get("data_centers", [])
    ]
    if not dcs:
        dcs = [DataCenter(name="local", node_ip="127.0.0.1", available_executors=8)]

    alloc_raw = raw.get("allocations")
    if alloc_raw:
        allocations = [
            AllocationType(
                name=item.get("name", f"{item['cpu_cores']}c-{item['memory_mb']}m"),
                cpu_cores=float(item["cpu_cores"]),
                memory_mb=int(item["memory_mb"]),
            )
            for item in alloc_raw
        ]
    else:
        grid = raw.get("allocation_grid", {})
        allocations = allocation_grid(
            cpu_values=grid.get("cpu_values", (1, 2, 4, 8)),
            memory_values_mb=grid.get(
                "memory_values_mb", (128, 512, 1024, 2048, 4096, 8192, 10240)
            ),
        )

    model_raw = raw.get("model", {})
    runtime_raw = raw.get("runtime", {})
    model = ModelConfig(
        embedding_dim=int(model_raw.get("embedding_dim", 64)),
        gnn_layers=int(model_raw.get("gnn_layers", 3)),
        temporal_window=int(model_raw.get("temporal_window", 5)),
        transformer_layers=int(model_raw.get("transformer_layers", 3)),
        transformer_heads=int(model_raw.get("transformer_heads", 8)),
        score_hidden=list(model_raw.get("score_hidden", [32, 16])),
        lr=float(model_raw.get("lr", 3e-4)),
        gamma=float(model_raw.get("gamma", 1.0)),
        clip_epsilon=float(model_raw.get("clip_epsilon", 0.2)),
        penalty_beta=float(model_raw.get("penalty_beta", 0.3)),
        timeout_seconds=float(model_raw.get("timeout_seconds", 120.0)),
    )
    runtime = RuntimeConfig(
        interval_seconds=float(runtime_raw.get("interval_seconds", 5.0)),
        max_wave_size=int(runtime_raw.get("max_wave_size", 20)),
        memory_loss_factor=float(runtime_raw.get("memory_loss_factor", 0.8)),
        enable_cluster_rpc=bool(runtime_raw.get("enable_cluster_rpc", False)),
        policy_path=runtime_raw.get("policy_path"),
    )
    return DemeterConfig(data_centers=dcs, allocations=allocations, model=model, runtime=runtime)
