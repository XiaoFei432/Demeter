"""Policy implementations for Demeter and baselines."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .cost import estimate_metrics, valid_allocation
from .features import FeatureEncoder
from .models import AllocationType, DataCenter, FunctionSpec, Observation, OrchestrationAction


class BasePolicy(ABC):
    @abstractmethod
    def plan(self, observation: Observation) -> List[OrchestrationAction]:
        raise NotImplementedError


class RandomPolicy(BasePolicy):
    def __init__(self, allocations: Sequence[AllocationType], seed: int = 1) -> None:
        self.allocations = list(allocations)
        self.random = random.Random(seed)

    def plan(self, observation: Observation) -> List[OrchestrationAction]:
        actions = []
        dcs = observation.data_centers
        for fn in observation.all_pending_functions():
            candidates = [dc for dc in dcs if not fn.candidate_dcs or dc.name in fn.candidate_dcs]
            valid = [a for a in self.allocations if valid_allocation(fn, a)]
            if not candidates or not valid:
                continue
            actions.append(OrchestrationAction(fn.function_id, self.random.choice(candidates).name, self.random.choice(valid)))
        return actions


class PheromonePolicy(BasePolicy):
    """Approximate Pheromone baseline: data locality plus minimum valid resources."""

    def __init__(self, allocations: Sequence[AllocationType]) -> None:
        self.allocations = sorted(allocations, key=lambda a: (a.memory_mb, a.cpu_cores))

    def plan(self, observation: Observation) -> List[OrchestrationAction]:
        dcs = observation.data_center_map
        actions = []
        for fn in observation.all_pending_functions():
            candidates = [dcs[name] for name in (fn.candidate_dcs or dcs.keys()) if name in dcs]
            if not candidates:
                continue
            dc = max(candidates, key=lambda item: fn.source_dcs.get(item.name, 0.0))
            alloc = next((a for a in self.allocations if valid_allocation(fn, a)), self.allocations[-1])
            actions.append(OrchestrationAction(fn.function_id, dc.name, alloc))
        return actions


class GreedyCostPolicy(BasePolicy):
    """Cost-aware baseline used for Astrea/Aquatope-style comparisons."""

    def __init__(self, allocations: Sequence[AllocationType], memory_loss_factor: float = 0.8) -> None:
        self.allocations = list(allocations)
        self.memory_loss_factor = memory_loss_factor

    def plan(self, observation: Observation) -> List[OrchestrationAction]:
        dc_map = observation.data_center_map
        actions = []
        for fn in observation.all_pending_functions():
            best = None
            for dc in observation.data_centers:
                if fn.candidate_dcs and dc.name not in fn.candidate_dcs:
                    continue
                for alloc in self.allocations:
                    if not valid_allocation(fn, alloc, self.memory_loss_factor):
                        continue
                    metrics = estimate_metrics(fn, dc, alloc, dc_map, observation.now)
                    objective = metrics.total_cost + 0.001 * metrics.duration
                    if best is None or objective < best[0]:
                        best = (objective, dc.name, alloc)
            if best is not None:
                actions.append(OrchestrationAction(fn.function_id, best[1], best[2], allocation_score=-best[0]))
        return actions


class DemeterHeuristicPolicy(BasePolicy):
    """Deployment-friendly fallback for the learned pointer-score policy.

    It mirrors Demeter's action decoupling: estimate function/DC affinity first,
    then pick the best valid resource allocation for the chosen placement.
    """

    def __init__(
        self,
        allocations: Sequence[AllocationType],
        memory_loss_factor: float = 0.8,
        encoder: Optional[FeatureEncoder] = None,
    ) -> None:
        self.allocations = list(allocations)
        self.memory_loss_factor = memory_loss_factor
        self.encoder = encoder or FeatureEncoder()

    def plan(self, observation: Observation) -> List[OrchestrationAction]:
        dc_map = observation.data_center_map
        pending = observation.all_pending_functions()
        if not pending:
            return []

        # Pr(n): lightly favors less loaded DCs with warm containers.
        priors = {}
        for dc in observation.data_centers:
            priors[dc.name] = (1.0 + dc.warm_containers + dc.available_executors) / (
                1.0 + dc.running_functions
            )
        prior_sum = sum(priors.values()) or 1.0
        priors = {name: val / prior_sum for name, val in priors.items()}

        actions = []
        for fn in pending:
            affinities: Dict[str, float] = {}
            candidates = [dc for dc in observation.data_centers if not fn.candidate_dcs or dc.name in fn.candidate_dcs]
            for dc in candidates:
                locality_mb = fn.source_dcs.get(dc.name, 0.0)
                remote_mb = max(sum(fn.source_dcs.values()) - locality_mb, 0.0)
                bw_penalty = 0.0
                for source, mb in fn.source_dcs.items():
                    if source != dc.name:
                        bw_penalty += mb / dc.bandwidth_to(source)
                load_penalty = dc.running_functions / max(dc.available_executors + dc.running_functions, 1)
                warm_bonus = 0.15 if dc.warm_containers > 0 else 0.0
                affinities[dc.name] = (
                    1.0
                    + locality_mb / max(fn.input_mb, 1.0)
                    - 0.25 * remote_mb / max(fn.input_mb, 1.0)
                    - 0.35 * bw_penalty
                    - 0.2 * load_penalty
                    + warm_bonus
                )
                affinities[dc.name] *= priors.get(dc.name, 1e-6)

            if not affinities:
                continue
            posterior_total = sum(max(v, 1e-9) for v in affinities.values())
            dc_name = max(affinities, key=affinities.get)
            posterior = max(affinities[dc_name], 1e-9) / posterior_total
            dc = dc_map[dc_name]

            best_alloc = None
            best_score = float("-inf")
            for alloc in self.allocations:
                if not valid_allocation(fn, alloc, self.memory_loss_factor):
                    continue
                metrics = estimate_metrics(fn, dc, alloc, dc_map, observation.now)
                # Lower affinity functions need earlier and slightly stronger resources.
                urgency = 1.0 - posterior
                score = -metrics.total_cost - 0.0005 * metrics.duration + urgency * alloc.cpu_cores * 0.00001
                if score > best_score:
                    best_score = score
                    best_alloc = alloc
            if best_alloc is not None:
                actions.append(
                    OrchestrationAction(
                        function_id=fn.function_id,
                        dc_name=dc_name,
                        allocation=best_alloc,
                        posterior=posterior,
                        allocation_score=best_score,
                    )
                )
        actions.sort(key=lambda a: (a.posterior, -a.allocation.cpu_cores))
        return actions
