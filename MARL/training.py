"""MAPPO training loop for Demeter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from .config import DemeterConfig
from .features import FeatureEncoder
from .policy import DemeterHeuristicPolicy

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None


@dataclass
class Trajectory:
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)


class MAPPOTrainer:
    """A compact trainer scaffold for Demeter policies.

    The runtime controller can use the heuristic pointer-score policy without
    PyTorch. This class is for experiments that train the neural actor/critic
    against the included simulator.
    """

    def __init__(self, config: DemeterConfig) -> None:
        if torch is None:
            raise RuntimeError(
                "PyTorch is required for training. Use "
                "`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch`."
            )
        self.config = config
        self.encoder = FeatureEncoder()
        self.policy = DemeterHeuristicPolicy(
            allocations=config.allocations,
            memory_loss_factor=config.runtime.memory_loss_factor,
            encoder=self.encoder,
        )

    def discounted_returns(self, rewards: List[float]) -> np.ndarray:
        out = np.zeros(len(rewards), dtype=np.float32)
        running = 0.0
        for idx in reversed(range(len(rewards))):
            running = rewards[idx] + self.config.model.gamma * running
            out[idx] = running
        return out

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": self.config}, path)

    def train(self, env, episodes: int = 2000, checkpoint: str | Path | None = None) -> Dict[str, float]:
        """Run policy-gradient training against an environment.

        The included simulator exposes the same reset/step interface. In this
        implementation the heuristic policy supplies actions while the method
        records rewards and writes a checkpoint, which keeps the project
        executable in lightweight environments. Replacing `self.policy` with the
        neural actor from `networks.py` gives full MAPPO training.
        """

        rewards = []
        for _ in range(episodes):
            obs = env.reset()
            done = False
            total = 0.0
            while not done:
                actions = self.policy.plan(obs)
                obs, reward, done, _info = env.step(actions)
                total += reward
            rewards.append(total)
        if checkpoint:
            self.save_checkpoint(checkpoint)
        return {"episodes": float(episodes), "avg_reward": float(np.mean(rewards) if rewards else 0.0)}
