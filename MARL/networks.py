"""PyTorch modules for Demeter's hierarchical GNN and pointer-score actor."""

from __future__ import annotations

from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for training. Install with a domestic mirror, e.g. "
            "`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch`."
        )


class DAGNN(nn.Module):
    """Topological-batching-friendly DAG message passing layer stack."""

    def __init__(self, input_dim: int, hidden_dim: int, layers: int = 3):
        _require_torch()
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(layers))

    def forward(self, x, adjacency):
        h = torch.tanh(self.input(x))
        for layer in self.layers:
            pred_msg = adjacency.transpose(-1, -2).matmul(h)
            h = torch.tanh(layer(h + pred_msg))
        graph = h.mean(dim=-2)
        return h, graph


class HTGNN(nn.Module):
    """Compact heterogeneous temporal graph encoder.

    The implementation keeps the paper's two aggregation axes: spatial
    function/DC edges inside a graph slice and temporal aggregation over recent
    DC states. It avoids external graph packages so the project remains easy to
    build inside the Pheromone repository.
    """

    def __init__(self, function_dim: int, dc_dim: int, hidden_dim: int, layers: int = 3):
        _require_torch()
        super().__init__()
        self.fn_in = nn.Linear(function_dim, hidden_dim)
        self.dc_in = nn.Linear(dc_dim, hidden_dim)
        self.fn_layers = nn.ModuleList(nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(layers))
        self.dc_layers = nn.ModuleList(nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(layers))
        self.temporal_gate = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, fn_x, dc_x, fn_dc_edges, dc_history=None):
        fn_h = torch.tanh(self.fn_in(fn_x))
        dc_h = torch.tanh(self.dc_in(dc_x))
        for fn_layer, dc_layer in zip(self.fn_layers, self.dc_layers):
            fn_msg = fn_dc_edges.matmul(dc_h)
            dc_msg = fn_dc_edges.transpose(-1, -2).matmul(fn_h)
            fn_h = torch.tanh(fn_layer(torch.cat([fn_h, fn_msg], dim=-1)))
            dc_h = torch.tanh(dc_layer(torch.cat([dc_h, dc_msg], dim=-1)))
        if dc_history is not None and dc_history.numel() > 0:
            _, h_last = self.temporal_gate(dc_history)
            dc_h = dc_h + h_last.squeeze(0)
        return fn_h, dc_h


class PointerScoreActor(nn.Module):
    """Decoupled placement pointer and allocation score network."""

    def __init__(
        self,
        function_dim: int,
        dc_dim: int,
        allocation_dim: int,
        hidden_dim: int = 64,
        transformer_layers: int = 3,
        transformer_heads: int = 8,
    ):
        _require_torch()
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=function_dim,
            nhead=max(1, min(transformer_heads, function_dim)),
            dim_feedforward=max(hidden_dim * 2, function_dim),
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.query = nn.Linear(dc_dim, function_dim)
        self.pointer_v = nn.Linear(function_dim, 1, bias=False)
        self.score = nn.Sequential(
            nn.Linear(function_dim + dc_dim + allocation_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

    def placement_logits(self, fn_h, dc_h, mask=None):
        encoded = self.transformer(fn_h)
        query = self.query(dc_h).unsqueeze(-2)
        logits = self.pointer_v(torch.tanh(encoded + query)).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return logits

    def allocation_scores(self, fn_h, dc_h, alloc_x):
        dc_expand = dc_h.unsqueeze(-2).expand(*alloc_x.shape[:-1], dc_h.shape[-1])
        fn_expand = fn_h.unsqueeze(-2).expand(*alloc_x.shape[:-1], fn_h.shape[-1])
        x = torch.cat([fn_expand, dc_expand, alloc_x], dim=-1)
        return self.score(x).squeeze(-1)


class CentralCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        _require_torch()
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state)
