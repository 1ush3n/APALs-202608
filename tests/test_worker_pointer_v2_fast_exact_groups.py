# -*- coding: utf-8 -*-
"""WorkerPointer v2 Fast-Exact 阶段一：group planner 语义测试。

- 最终 num_envs / group 上限不得超过 16（Linux 正式配置上限）；
- 逻辑 batch 256 内完整 behavior group 不拆分、不重排、不混合；
- 异质组（组内 worker 数不同）仍按样本数打包，绝不拆组；
- 尾批不足 256 时按实际样本数处理。
"""

from __future__ import annotations

import pytest

from configs import Config
from runtime.configuration import validate_runtime_config
from training.worker_pointer_v2_replay import plan_behavior_groups


def _fast_exact_config(**overrides: object) -> Config:
    cfg = Config()
    cfg.team_selection_mode = "autoregressive_pressure_v2_fast_exact"
    cfg.policy_action_scope = "operation_station_worker"
    cfg.actor_context_mode = "attention"
    cfg.worker_pointer_v2_replay_mode = "behavior_group_exact_gpu_template_v2"
    cfg.worker_pointer_v2_rollout_group_upper_bound = 16
    cfg.batch_size = 256
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_fast_exact_rejects_num_envs_above_16() -> None:
    cfg = _fast_exact_config(num_envs=32)
    with pytest.raises(ValueError, match="num_envs"):
        validate_runtime_config(cfg)


def test_fast_exact_rejects_group_upper_bound_above_16() -> None:
    cfg = _fast_exact_config()
    cfg.worker_pointer_v2_rollout_group_upper_bound = 32
    with pytest.raises(ValueError, match="rollout_group_upper_bound"):
        validate_runtime_config(cfg)


def test_fast_exact_accepts_16_envs_and_group_bound_16() -> None:
    cfg = _fast_exact_config(num_envs=16)
    cfg.worker_pointer_v2_rollout_group_upper_bound = 16
    validate_runtime_config(cfg)


def test_planner_packs_full_groups_into_cap256_without_splitting() -> None:
    # 16 组 × 16 样本 = 256，恰好 1 个逻辑 batch；组不拆分且全部容纳。
    group_sizes = [16] * 16
    batches = plan_behavior_groups(
        group_sizes, logical_cap=256, seed=42, current_step=1, epoch=0
    )
    assert len(batches) == 1
    assert sorted(batches[0]) == list(range(16))


def test_planner_tail_batch_uses_actual_samples_with_17_groups() -> None:
    # 前 16 组恰好装满 256，第 17 组成为尾批；每组样本数保持 16（不拆分）。
    group_sizes = [16] * 17
    batches = plan_behavior_groups(
        group_sizes, logical_cap=256, seed=42, current_step=1, epoch=0
    )
    assert len(batches) == 2
    assert sum(group_sizes[i] for i in batches[0]) <= 256
    assert sum(group_sizes[i] for i in batches[1]) == 16
    for batch in batches:
        for group_index in batch:
            assert group_sizes[group_index] == 16


def test_planner_heterogeneous_group_sizes_stay_whole() -> None:
    # 异质组（不同 worker 数导致不同样本数）仍按样本数打包，绝不拆组。
    group_sizes = [16, 4, 16, 7, 4]
    batches = plan_behavior_groups(
        group_sizes, logical_cap=256, seed=7, current_step=0, epoch=0
    )
    assert len(batches) == 1
    assert sorted(batches[0]) == [0, 1, 2, 3, 4]
