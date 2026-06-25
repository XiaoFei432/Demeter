"""Example profile and Pheromone registration flow for Demeter."""

from MARL.config import load_config
from MARL.controller import DemeterController
from MARL.pheromone import PheromoneDemeterAdapter
from MARL.workloads import TPC_DS_Q94


def build_profile():
    stages = []
    for stage in TPC_DS_Q94["stages"]:
        item = dict(stage)
        item["candidate_dcs"] = ["paris", "ohio", "oregon", "london"]
        item["source_dcs"] = {"paris": item["input_mb"] * 0.5, "ohio": item["input_mb"] * 0.5}
        stages.append(item)
    return {
        "job_id": "demo-q94",
        "slo_seconds": 80,
        "arrival_time": 0,
        "stages": stages,
        "edges": TPC_DS_Q94["edges"],
    }


def main():
    config = load_config("configs/demeter.yml")
    controller = DemeterController(config)
    adapter = PheromoneDemeterAdapter(controller)
    adapter.register_profile(build_profile())
    actions = adapter.decide_once(now=0)
    for action in actions[:10]:
        print(action)


if __name__ == "__main__":
    main()
