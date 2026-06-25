"""Command-line entry points for Demeter."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .config import load_config
from .controller import DemeterController
from .policy import DemeterHeuristicPolicy, GreedyCostPolicy, PheromonePolicy, RandomPolicy
from .simulator import DemeterEnv
from .workloads import generate_job_profiles


def _policy(name, config):
    if name == "demeter":
        return DemeterHeuristicPolicy(config.allocations, config.runtime.memory_loss_factor)
    if name == "pheromone":
        return PheromonePolicy(config.allocations)
    if name in ("astrea", "aquatope", "greedy"):
        return GreedyCostPolicy(config.allocations, config.runtime.memory_loss_factor)
    if name == "random":
        return RandomPolicy(config.allocations)
    raise ValueError(f"unknown policy: {name}")


def cmd_simulate(args) -> None:
    config = load_config(args.config)
    dc_names = [dc.name for dc in config.data_centers]
    jobs = generate_job_profiles(args.jobs, dc_names, mode=args.mode, seed=args.seed)
    rows = []
    for policy_name in args.policies:
        env = DemeterEnv(config, jobs, seed=args.seed)
        obs = env.reset()
        policy = _policy(policy_name, config)
        done = False
        while not done:
            actions = policy.plan(obs)
            obs, _reward, done, _info = env.step(actions)
        result = env.result()
        rows.append(
            {
                "policy": policy_name,
                "jobs": result.jobs,
                "functions": result.functions,
                "total_cost": result.total_cost,
                "avg_jct": result.avg_jct,
                "avg_slo_bias": result.avg_slo_bias,
                "slo_violations": result.slo_violations,
                "memory_usage_gb_seconds": result.memory_usage_gb_seconds,
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['policy']}: cost={row['total_cost']:.6f}, "
            f"avg_jct={row['avg_jct']:.3f}, slo_violation={row['slo_violations']:.3f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="demeter")
    sub = parser.add_subparsers(dest="command", required=True)
    sim = sub.add_parser("simulate", help="run synthetic Demeter experiments")
    sim.add_argument("--config", default="configs/demeter.yml")
    sim.add_argument("--jobs", type=int, default=64)
    sim.add_argument("--mode", choices=["slow", "normal", "burst"], default="normal")
    sim.add_argument("--seed", type=int, default=1)
    sim.add_argument(
        "--policies",
        nargs="+",
        default=["demeter", "pheromone", "astrea", "aquatope", "random"],
    )
    sim.add_argument("--output", default="results/simulation.csv")
    sim.set_defaults(func=cmd_simulate)
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
