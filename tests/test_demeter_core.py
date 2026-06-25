from MARL.config import load_config
from MARL.controller import DemeterController
from MARL.policy import DemeterHeuristicPolicy, PheromonePolicy
from MARL.simulator import DemeterEnv
from MARL.workloads import generate_job_profiles


def test_controller_generates_actions():
    config = load_config("configs/demeter.yml")
    jobs = generate_job_profiles(2, [dc.name for dc in config.data_centers], seed=7)
    controller = DemeterController(config)
    for job in jobs:
        controller.register_job(job)
    actions = controller.plan(controller.observe(0.0))
    assert actions
    assert all(action.dc_name for action in actions)
    assert all(action.allocation.memory_mb >= 128 for action in actions)


def test_simulator_runs_to_completion():
    config = load_config("configs/demeter.yml")
    jobs = generate_job_profiles(2, [dc.name for dc in config.data_centers], seed=9)
    env = DemeterEnv(config, jobs, seed=9)
    obs = env.reset()
    policy = DemeterHeuristicPolicy(config.allocations)
    done = False
    steps = 0
    while not done and steps < 20:
        obs, _reward, done, _info = env.step(policy.plan(obs))
        steps += 1
    assert done
    result = env.result()
    assert result.functions > 0
    assert result.total_cost >= 0
