"""Discrete-event simulator for Demeter experiments."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import DemeterConfig
from .controller import DemeterController
from .cost import estimate_metrics
from .models import FunctionStatus, JobSpec, Observation, OrchestrationAction


@dataclass
class EpisodeResult:
    jobs: int
    functions: int
    total_cost: float
    avg_jct: float
    avg_slo_bias: float
    slo_violations: float
    memory_usage_gb_seconds: float
    dop_tuning_events: int = 0
    history_records: int = 0


class DemeterEnv:
    """Lightweight environment matching Demeter's observation/action loop."""

    def __init__(self, config: DemeterConfig, job_profiles: Sequence[Mapping], seed: int = 1) -> None:
        self.base_config = config
        self.job_profiles = list(job_profiles)
        self.random = random.Random(seed)
        self.seed = seed
        self.controller: Optional[DemeterController] = None
        self.now = 0.0
        self.total_cost = 0.0
        self.memory_usage_gb_seconds = 0.0
        self.finished_functions = 0
        self.dop_tuning_events = 0

    def reset(self) -> Observation:
        self.now = 0.0
        self.total_cost = 0.0
        self.memory_usage_gb_seconds = 0.0
        self.finished_functions = 0
        self.dop_tuning_events = 0
        config = copy.deepcopy(self.base_config)
        self.controller = DemeterController(config)
        for raw in self.job_profiles:
            self.controller.register_job(raw)
        return self.controller.observe(self.now)

    def step(self, actions: Sequence[OrchestrationAction]):
        assert self.controller is not None, "reset() must be called before step()."
        dc_map = {dc.name: dc for dc in self.controller.config.data_centers}
        fn_by_id = {
            fn.function_id: fn
            for job in self.controller.jobs.values()
            for fn in job.functions.values()
        }
        reward = 0.0
        max_finish = self.now
        for action in actions:
            fn = fn_by_id.get(action.function_id)
            dc = dc_map.get(action.dc_name)
            if fn is None or dc is None or fn.status != FunctionStatus.PENDING:
                continue
            metrics = estimate_metrics(
                fn,
                dc,
                action.allocation,
                dc_map,
                self.now,
                timeout_seconds=self.controller.config.model.timeout_seconds,
            )
            fn.placement = action.dc_name
            fn.allocation = action.allocation
            fn.start_time = self.now
            jitter = self.random.uniform(0.92, 1.08)
            fn.observed_duration = metrics.duration * jitter
            fn.finish_time = self.now + fn.observed_duration
            fn.status = FunctionStatus.FINISHED
            fn.resource_peak_cpu = action.allocation.cpu_cores
            fn.resource_peak_memory_mb = min(action.allocation.memory_mb, fn.input_mb * 1.1)
            fn.metadata["affinity"] = action.posterior
            dc.running_functions = max(0, dc.running_functions - 1)
            dc.available_executors += 1
            dc.warm_containers += 1
            self.total_cost += metrics.total_cost
            self.memory_usage_gb_seconds += action.allocation.memory_gb * fn.observed_duration
            self.finished_functions += 1
            job = self.controller.jobs.get(fn.job_id)
            if job is not None:
                elapsed = max(0.0, fn.finish_time - job.arrival_time)
                progress = min(1.0, elapsed / max(job.slo_seconds, 1e-6))
                self.controller.invocation_history.add_from_function(
                    fn,
                    job_progress=progress,
                    affinity=action.posterior,
                )
            reward -= metrics.total_cost
            max_finish = max(max_finish, fn.finish_time)

        self.now = max_finish + self.controller.config.runtime.interval_seconds
        for job in self.controller.jobs.values():
            self.controller.optimizer.release_ready_functions(job, self.now)
        previous_dop_events = len(self.controller.dop_decisions)
        obs = self.controller.observe(self.now)
        self.dop_tuning_events += len(self.controller.dop_decisions) - previous_dop_events
        done = all(job.finished() for job in self.controller.jobs.values())
        if done:
            reward += self._terminal_reward()
        return obs, reward, done, {}

    def result(self) -> EpisodeResult:
        assert self.controller is not None
        jobs = list(self.controller.jobs.values())
        jcts = [job.jct() for job in jobs]
        slo_bias = [job.jct() / max(job.slo_seconds, 1e-6) for job in jobs]
        violations = [1.0 for value in slo_bias if value > 1.0]
        return EpisodeResult(
            jobs=len(jobs),
            functions=sum(len(job.functions) for job in jobs),
            total_cost=self.total_cost,
            avg_jct=float(np.mean(jcts) if jcts else 0.0),
            avg_slo_bias=float(np.mean(slo_bias) if slo_bias else 0.0),
            slo_violations=float(sum(violations) / max(len(jobs), 1)),
            memory_usage_gb_seconds=self.memory_usage_gb_seconds,
            dop_tuning_events=self.dop_tuning_events,
            history_records=len(self.controller.invocation_history),
        )

    def _terminal_reward(self) -> float:
        assert self.controller is not None
        reward = 0.0
        beta = self.controller.config.model.penalty_beta
        for job in self.controller.jobs.values():
            remaining = job.slo_seconds - job.jct()
            reward += beta * remaining / max(job.slo_seconds, 1e-6)
        return reward
