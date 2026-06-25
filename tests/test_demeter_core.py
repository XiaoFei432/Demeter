from MARL.config import load_config
from MARL.controller import DemeterController
from MARL.history import InvocationHistory, InvocationRecord
from MARL.partitioner import BucketPartitioner, jump_consistent_hash
from MARL.policy import DemeterHeuristicPolicy, PheromonePolicy
from MARL.pruning import ConfigurationPruner
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
    assert result.history_records > 0


def test_pruner_uses_history_to_narrow_allocation_pool():
    config = load_config("configs/demeter.yml")
    job = DemeterController(config).register_job(
        generate_job_profiles(1, [dc.name for dc in config.data_centers], seed=3)[0]
    )
    fn = next(iter(job.functions.values()))
    history = InvocationHistory()
    baseline = config.allocations[-2]
    history.add(
        InvocationRecord(
            stage_id=fn.stage_id,
            function_type=fn.function_type,
            input_mb=fn.input_mb,
            allocation=baseline,
            peak_memory_mb=baseline.memory_mb * 0.95,
            peak_cpu=baseline.cpu_cores,
            duration=1.0,
            affinity=0.5,
            progress=0.5,
        )
    )
    pruner = ConfigurationPruner(config.allocations, memory_loss_factor=config.runtime.memory_loss_factor)
    pool = pruner.prune(fn, job, affinity=0.2, now=0.0, history=history)
    assert pool
    assert all(alloc.memory_mb >= baseline.memory_mb for alloc in pool)


def test_dop_tuning_coalesces_congested_wave():
    config = load_config("configs/demeter.yml")
    config.runtime.max_wave_size = 4
    profile = generate_job_profiles(1, [dc.name for dc in config.data_centers], seed=11)[0]
    profile["stages"][0]["parallelism"] = 12
    controller = DemeterController(config)
    controller.register_job(profile)
    obs = controller.observe(0.0)
    assert controller.dop_decisions
    assert len(obs.all_pending_functions()) <= config.runtime.max_wave_size


def test_jump_hash_partitioner_is_stable_and_bounded():
    keys = [f"bucket-{idx}" for idx in range(64)]
    partitioner = BucketPartitioner()
    old = partitioner.assign(keys, 8)
    new = partitioner.assign(keys, 10)
    moved = partitioner.migration_plan(old, 10)
    assert all(0 <= bucket < 10 for bucket in new.values())
    assert all(jump_consistent_hash(key, 10) == new[key] for key in keys)
    assert 0 < len(moved) < len(keys)
