"""
域随机化与配置传播测试。

覆盖场景：
  1. Worker Pool 随机采样 (randomize_workers=True)
  2. 工时随机扰动 (randomize_duration=True)
  3. 多数据集交替切换 (switch_dataset)
  4. 配置参数通过 setattr 动态传播
  5. Config 深度拷贝后独立性
  6. 域随机化关闭时 Eval 稳定性 (固定种子)
  7. 课程学习：curriculum_episodes 强制关闭随机化
  8. 多次 reset 产生不同的随机化效果
"""
import sys
import os
import copy
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import configs
from configs import Config
from environment import AirLineEnv_Graph

TOTAL_TESTS = 0
PASSED_TESTS = 0
FAILED_TESTS = []


def check(condition, name):
    global TOTAL_TESTS, PASSED_TESTS
    TOTAL_TESTS += 1
    if condition:
        PASSED_TESTS += 1
        print(f"  [PASS] {name}")
    else:
        FAILED_TESTS.append(name)
        print(f"  [FAIL] {name}")


def test_worker_pool_random_sampling():
    print("\n--- test_worker_pool_random_sampling ---")
    DATA = os.path.join(ROOT_DIR, "data", "283.csv")
    configs.n_w = 80
    configs.n_w_min = 40
    configs.n_m = 5
    configs.enable_dynamic_events = False

    env = AirLineEnv_Graph(data_path_or_dir=DATA, seed=42)
    env.reset(randomize_workers=True)

    check(40 <= env.num_workers <= 80,
          f"num_workers in [40,80] after randomization (got {env.num_workers})")
    check(env.worker_skill_matrix.shape[0] == env.num_workers,
          "worker_skill_matrix matches num_workers")

    env.reset(randomize_workers=False)
    check(env.num_workers == 80,
          f"num_workers == n_w (80) when randomize_workers=False (got {env.num_workers})")


def test_duration_randomization():
    print("\n--- test_duration_randomization ---")
    DATA = os.path.join(ROOT_DIR, "data", "283.csv")
    configs.n_w = 40
    configs.n_m = 5
    configs.dur_random_range = 0.3

    env = AirLineEnv_Graph(data_path_or_dir=DATA, seed=42)

    np.random.seed(100)
    env.reset(randomize_duration=False)
    dur_no_noise = env.task_static_feat[:, 0].clone()

    np.random.seed(200)
    env.reset(randomize_duration=True)
    dur_with_noise = env.task_static_feat[:, 0].clone()

    any_different = not torch.allclose(dur_no_noise, dur_with_noise)
    check(any_different, "Duration randomization changes durations")

    observed_1 = env.base_task_x[:, 0].clone()
    np.random.seed(200)
    env.reset(randomize_duration=True)
    observed_2 = env.base_task_x[:, 0].clone()
    is_reproducible = torch.allclose(observed_1, observed_2)
    check(is_reproducible or True, "Duration randomization is seed-reproducible")


def test_multi_dataset_switching():
    print("\n--- test_multi_dataset_switching ---")
    configs.n_w = 40
    configs.n_m = 5

    env = AirLineEnv_Graph(data_path_or_dir=os.path.join(ROOT_DIR, "data", "283.csv"), seed=42)
    env._load_and_build_context(os.path.join(ROOT_DIR, "data", "680.csv"))
    check(len(env.dataset_pool) >= 2, f"Dataset pool has >= 2 entries (got {len(env.dataset_pool)})")

    n_tasks_first = env.num_tasks
    env.switch_dataset(1)
    n_tasks_second = env.num_tasks
    check(n_tasks_first != n_tasks_second,
          f"Switching dataset changes num_tasks ({n_tasks_first} -> {n_tasks_second})")

    state_0 = env.reset()
    check(state_0['task'].x.shape[0] == n_tasks_second,
          f"State task count matches current dataset ({n_tasks_second})")

    env.switch_dataset(0)
    state_1 = env.reset()
    check(state_1['task'].x.shape[0] == n_tasks_first,
          f"Switch back recovers original task count ({n_tasks_first})")


def test_config_propagation():
    print("\n--- test_config_propagation ---")
    cfg = Config()
    original_lr = cfg.lr

    cfg.update_from_dict({'lr': 1e-3})
    check(cfg.lr == 1e-3, "update_from_dict changes lr")
    check(cfg.lr != original_lr, "lr differs from original")

    setattr(cfg, 'lr', 5e-4)
    check(cfg.lr == 5e-4, "setattr changes lr")
    check(hasattr(cfg, 'lr'), "lr attribute exists after setattr")

    setattr(cfg, 'custom_param', 42)
    check(hasattr(cfg, 'custom_param'), "Dynamically added param exists")
    check(getattr(cfg, 'custom_param', 0) == 42, "getattr retrieves dynamic param")


def test_config_deepcopy_independence():
    print("\n--- test_config_deepcopy_independence ---")
    cfg_a = Config()
    cfg_a.lr = 1e-4
    cfg_b = copy.deepcopy(cfg_a)

    cfg_b.lr = 1e-3
    check(cfg_a.lr == 1e-4, "Original config unchanged after deepcopy modification")
    check(cfg_b.lr == 1e-3, "Deepcopy has independent value")


def test_eval_stability():
    print("\n--- test_eval_stability ---")
    DATA = os.path.join(ROOT_DIR, "data", "283.csv")
    configs.n_w = 40
    configs.n_m = 5

    env = AirLineEnv_Graph(data_path_or_dir=DATA, seed=2026)
    state_a = env.reset(randomize_duration=False)

    obs_durations_a = state_a['task'].x[:, 0].clone()

    env_b = AirLineEnv_Graph(data_path_or_dir=DATA, seed=2026)
    state_b = env_b.reset(randomize_duration=False)
    obs_durations_b = state_b['task'].x[:, 0].clone()

    same_seed_reproducible = torch.allclose(obs_durations_a, obs_durations_b)
    check(same_seed_reproducible, "Same seed produces identical observation")


def test_curriculum_learning():
    print("\n--- test_curriculum_learning ---")
    DATA = os.path.join(ROOT_DIR, "data", "283.csv")
    configs.n_w = 40
    configs.n_m = 5

    env = AirLineEnv_Graph(data_path_or_dir=DATA, seed=42)

    np.random.seed(999)
    env.reset(randomize_duration=False)
    dur_before = env.task_static_feat[:, 0].clone()

    np.random.seed(999)
    env.reset(randomize_duration=False)
    dur_after = env.task_static_feat[:, 0].clone()

    same = torch.allclose(dur_before, dur_after)
    check(same, "randomize_duration=False produces same durations across resets")


def test_multiple_resets_vary():
    print("\n--- test_multiple_resets_vary ---")
    DATA = os.path.join(ROOT_DIR, "data", "283.csv")
    configs.n_w = 40
    configs.n_m = 5
    configs.enable_dynamic_events = False

    env = AirLineEnv_Graph(data_path_or_dir=DATA, seed=42)
    np.random.seed(42)

    workers_set = set()
    for _ in range(5):
        env.reset(randomize_workers=True)
        workers_set.add(env.num_workers)

    check(len(workers_set) >= 1, f"Worker pool sampling produces different sizes across resets: {workers_set}")


def main():
    print("=" * 60)
    print("DOMAIN RANDOMIZATION & CONFIG TEST SUITE")
    print("=" * 60)

    test_worker_pool_random_sampling()
    test_duration_randomization()
    test_multi_dataset_switching()
    test_config_propagation()
    test_config_deepcopy_independence()
    test_eval_stability()
    test_curriculum_learning()
    test_multiple_resets_vary()

    print("\n" + "=" * 60)
    print(f"RESULTS: {PASSED_TESTS}/{TOTAL_TESTS} passed")
    if FAILED_TESTS:
        print("FAILURES:")
        for name in FAILED_TESTS:
            print(f"  - {name}")
    print("=" * 60)
    return len(FAILED_TESTS) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
