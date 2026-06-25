"""Adapters that let Demeter exchange decisions with Pheromone clients."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .config import DemeterConfig
from .controller import DemeterController
from .models import FunctionSpec, FunctionStatus, OrchestrationAction


def write_decision_file(actions: Sequence[OrchestrationAction], path: str | Path) -> None:
    """Write a JSON decision file consumable by scripts or sidecars."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "actions": [
            {
                "function_id": action.function_id,
                "dc": action.dc_name,
                "cpu_cores": action.allocation.cpu_cores,
                "memory_mb": action.allocation.memory_mb,
                "posterior": action.posterior,
                "score": action.allocation_score,
            }
            for action in actions
        ]
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_decision_file(path: str | Path) -> Mapping:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class PheromoneDemeterAdapter:
    """Translate Pheromone app calls into Demeter job profiles.

    The original Pheromone protocol keeps function-call messages compact and
    does not include resource fields. This adapter runs on the client/control
    side: it builds a Demeter profile from application metadata, asks the
    controller for decisions, and exports them for deployment scripts or a
    cluster sidecar.
    """

    def __init__(self, controller: DemeterController, decision_path: str | Path = "runtime/demeter_actions.json") -> None:
        self.controller = controller
        self.decision_path = Path(decision_path)

    def register_profile(self, profile: Mapping) -> None:
        self.controller.register_job(profile)

    def decide_once(self, now: Optional[float] = None) -> List[OrchestrationAction]:
        actions = self.controller.run_once(now)
        write_decision_file(actions, self.decision_path)
        return actions
