"""Demeter controller that turns observations into orchestration decisions."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .config import DemeterConfig, load_config
from .models import FunctionSpec, FunctionStatus, JobSpec, Observation, OrchestrationAction
from .optimizer import DAGOptimizer
from .policy import BasePolicy, DemeterHeuristicPolicy, PheromonePolicy, RandomPolicy


class DemeterController:
    """Policy-facing orchestration loop.

    In cluster mode the controller is fed observations from Pheromone and emits
    placement/resource decisions. In simulation mode it is used directly by the
    experiment driver.
    """

    def __init__(self, config: DemeterConfig | str | Path, policy: Optional[BasePolicy] = None) -> None:
        if isinstance(config, (str, Path)):
            config = load_config(config)
        self.config = config
        self.optimizer = DAGOptimizer()
        self.policy = policy or DemeterHeuristicPolicy(
            allocations=config.allocations,
            memory_loss_factor=config.runtime.memory_loss_factor,
        )
        self.jobs: Dict[str, JobSpec] = {}
        self.history: List[Mapping[str, float]] = []

    def register_job(self, raw_profile: Mapping) -> JobSpec:
        job = self.optimizer.build_job(raw_profile)
        self.jobs[job.job_id] = job
        return job

    def observe(self, now: Optional[float] = None) -> Observation:
        now = time.time() if now is None else now
        pending_by_dc = {dc.name: [] for dc in self.config.data_centers}
        running: List[FunctionSpec] = []
        for job in self.jobs.values():
            self.optimizer.release_ready_functions(job, now)
            for fn in job.functions.values():
                if fn.status == FunctionStatus.RUNNING:
                    running.append(fn)
                if fn.status == FunctionStatus.PENDING:
                    candidates = fn.candidate_dcs or list(pending_by_dc.keys())
                    for dc_name in candidates:
                        if dc_name in pending_by_dc:
                            pending_by_dc[dc_name].append(fn)

        max_wave = self.config.runtime.max_wave_size
        for dc_name, fns in pending_by_dc.items():
            pending_by_dc[dc_name] = fns[:max_wave]

        return Observation(
            now=now,
            jobs=list(self.jobs.values()),
            data_centers=self.config.data_centers,
            pending_by_dc=pending_by_dc,
            running=running,
            history=self.history[-self.config.model.temporal_window :],
        )

    def plan(self, observation: Optional[Observation] = None) -> List[OrchestrationAction]:
        observation = observation or self.observe()
        return self.policy.plan(observation)

    def apply_actions(self, actions: Sequence[OrchestrationAction], now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        fn_by_id = {
            fn.function_id: fn for job in self.jobs.values() for fn in job.functions.values()
        }
        dc_by_name = {dc.name: dc for dc in self.config.data_centers}
        for action in actions:
            fn = fn_by_id.get(action.function_id)
            dc = dc_by_name.get(action.dc_name)
            if fn is None or dc is None:
                continue
            fn.placement = action.dc_name
            fn.allocation = action.allocation
            fn.start_time = now
            fn.status = FunctionStatus.RUNNING
            dc.running_functions += 1
            dc.available_executors = max(0, dc.available_executors - 1)

    def export_actions(self, actions: Sequence[OrchestrationAction], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "function_id": action.function_id,
                "dc_name": action.dc_name,
                "allocation": asdict(action.allocation),
                "posterior": action.posterior,
                "allocation_score": action.allocation_score,
            }
            for action in actions
        ]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def run_once(self, now: Optional[float] = None) -> List[OrchestrationAction]:
        observation = self.observe(now)
        actions = self.plan(observation)
        self.apply_actions(actions, observation.now)
        return actions
