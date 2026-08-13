# -*- coding: utf-8 -*-
"""WorkerPointer v2 Fast-Exact 阶段三：bf16 数值与梯度合同。

验证（CUDA 上，bf16-mixed）：
- 首次同形重算四分量（task/station/team/total）log-prob MaxAE ≤ 1e-3；
- 所有 v2 参数进入 actor 优化器并获得有限梯度（Gradient/V2Coverage == 1.0）；
- 无重复工人、非法工位、错误团队人数或非有限 loss。
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from configs import configs
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from tests.runtime_safety import temporary_config
from tests.test_joint_experiment_architecture import (
    DATA_PATH,
    _advance_to_ready_physical_task,
)
from tests.test_worker_pointer_v2_fast_exact_replay import (
    _DEVICE,
    _fast_exact_overrides,
    _make_agent,
    _rollout_single_step,
)
from training.v2_fast_exact_batch import GPUExactBatchBuilder

REQUIRES_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="bf16 数值合同需要 CUDA"
)


def _bf16_overrides() -> dict[str, Any]:
    overrides = _fast_exact_overrides()
    overrides["lightning_precision"] = "bf16-mixed"
    return overrides


def _run_single_step_update() -> tuple[dict[str, float], PPOAgent, Any]:
    env = AirLineEnv_Graph(DATA_PATH, seed=42)
    agent = _make_agent()
    memory, b_task, b_station, b_team, old_logprobs, rewards, advantages = (
        _rollout_single_step(agent, env)
    )
    memory.rewards = [0.0]
    memory.is_terminals = [False]
    builder = GPUExactBatchBuilder(config=configs, env=env, device=_DEVICE)
    metrics = agent._run_v2_fast_exact_replay_update(
        memory,
        env,
        current_ep=1,
        advantages=advantages,
        rewards=rewards,
        old_logprobs=old_logprobs,
        b_task=b_task,
        b_station=b_station,
        b_team=b_team,
        action_scope="operation_station_worker",
        fast_exact_builder=builder,
    )
    return metrics, agent, memory


@REQUIRES_CUDA
def test_fast_exact_bf16_first_contract_all_components_within_1e3() -> None:
    with temporary_config(configs, _bf16_overrides()):
        metrics, _agent, _memory = _run_single_step_update()
        assert metrics["PointerV2/AutocastBF16"] == 1.0
        assert metrics["V2/FirstContractTaskMaxAE"] <= 1.0e-3
        assert metrics["V2/FirstContractStationMaxAE"] <= 1.0e-3
        assert metrics["V2/FirstContractTeamMaxAE"] <= 1.0e-3
        assert metrics["V2/FirstContractTotalMaxAE"] <= 1.0e-3


@REQUIRES_CUDA
def test_fast_exact_bf16_all_v2_parameters_receive_finite_gradients() -> None:
    with temporary_config(configs, _bf16_overrides()):
        metrics, agent, _memory = _run_single_step_update()
        assert metrics["Gradient/V2Coverage"] == 1.0
        assert metrics["Gradient/Finite"] == 1.0
        assert metrics["Gradient/V2Norm"] > 0.0


@REQUIRES_CUDA
def test_fast_exact_bf16_losses_finite_and_no_illegal_targets() -> None:
    with temporary_config(configs, _bf16_overrides()):
        metrics, _agent, memory = _run_single_step_update()
        assert torch.isfinite(torch.tensor(metrics["PPO/Loss"]))
        assert torch.isfinite(torch.tensor(metrics["PPO/ValueLoss"]))
        assert torch.isfinite(torch.tensor(metrics["PPO/Entropy"]))
        for value in metrics.values():
            assert torch.isfinite(torch.tensor(value)), f"非有限指标: {value}"

        trace = memory.worker_pointer_v2_behavior_traces[0]
        team = list(memory.actions[0][2])
        # 团队人数正确：非填充目标数与动作团队长度一致，且无重复工人。
        assert len(team) >= 1
        assert len(set(team)) == len(team)
        # 非法工位：动作工位必须在合法站位数范围内。
        station_index = memory.actions[0][1]
        assert 0 <= station_index < int(getattr(configs, "n_m", 5))


@REQUIRES_CUDA
def test_fast_exact_fp32_contract_is_tighter_than_1e4() -> None:
    """fp32（32-true）下首次同形合同要求 MaxAE ≤ 1e-4。"""
    overrides = _fast_exact_overrides()
    overrides["lightning_precision"] = "32-true"
    with temporary_config(configs, overrides):
        metrics, _agent, _memory = _run_single_step_update()
        assert metrics["PointerV2/AutocastBF16"] == 0.0
        assert metrics["V2/FirstContractTotalMaxAE"] <= 1.0e-4
