# -*- coding: utf-8 -*-
"""
APAL 低显存 GPU 安全验证。

这些测试只覆盖小图、小 batch 和极短轨迹，用于在正式训练前发现 CUDA、AMP、PyG、
HB-GAT-PN、PPO 采样与 PPO 更新中的形状错误和 OOM 风险。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import configs
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from training.memory import Memory
from tests.runtime_safety import (
    get_cuda_info,
    guarded_cuda_test,
    temporary_config,
)


pytestmark = pytest.mark.gpu_safe


DATA_PATH = PROJECT_ROOT / "data" / "283.csv"


def _gpu_required() -> torch.device:
    info = get_cuda_info()
    assert info["available"], f"当前环境未启用 CUDA: {info}"
    return torch.device("cuda")


def _low_memory_overrides() -> dict[str, object]:
    return {
        "n_w": 40,
        "n_m": 5,
        "hidden_dim": 32,
        "num_gat_layers": 1,
        "num_heads": 2,
        "batch_size": 2,
        "accumulation_steps": 2,
        "k_epochs": 1,
        "use_schedule_free": False,
        "use_ema": False,
        "use_shared_trunk": False,
        "use_gradient_checkpointing": False,
        "enable_gpu_batch_rebuild": False,
        "enable_dynamic_events": False,
        "enable_station_breakdown": False,
        "enable_material_delay": False,
        "randomize_durations": False,
    }


def _make_env(seed: int = 42) -> AirLineEnv_Graph:
    return AirLineEnv_Graph(data_path_or_dir=str(DATA_PATH), seed=seed)


def _make_agent(device: torch.device) -> PPOAgent:
    model = HBGATPN(configs).to(device)
    return PPOAgent(
        model=model,
        lr=configs.lr,
        gamma=configs.gamma,
        k_epochs=configs.k_epochs,
        eps_clip=configs.eps_clip,
        device=device,
        batch_size=configs.batch_size,
        total_timesteps=2,
    )


def _first_valid_action(agent: PPOAgent, env: AirLineEnv_Graph, state, device: torch.device):
    task_mask, station_mask, worker_mask = env.get_masks()
    if task_mask.all():
        assert env.try_wait_for_resources(), "初始状态不应无动作且无法推进事件时钟"
        task_mask, station_mask, worker_mask = env.get_masks()
    assert not task_mask.all(), "小图初始状态必须存在合法任务"
    return agent.select_action(
        state.to(device),
        mask_task=task_mask.to(device),
        mask_station_matrix=station_mask.to(device),
        mask_worker=worker_mask.to(device),
        deterministic=True,
    ), (task_mask, station_mask, worker_mask)


def _collect_tiny_memory(agent: PPOAgent, env: AirLineEnv_Graph, device: torch.device, steps: int = 3) -> Memory:
    memory = Memory()
    state = env.reset(randomize_duration=False, randomize_workers=False, seed=123)

    for _ in range(steps):
        action_ret, masks = _first_valid_action(agent, env, state, device)
        action, logprob, value, _, is_invalid = action_ret
        assert action is not None
        assert not is_invalid

        memory.states.append(env.get_state_snapshot())
        memory.actions.append(action)
        memory.logprobs.append(logprob)
        memory.masks.append(masks)
        memory.values.append(value)

        state, reward, done, _ = env.step(action)
        memory.rewards.append(float(reward))
        memory.is_terminals.append(bool(done))
        if done:
            break

    assert len(memory.states) > 0, "极小 PPO 轨迹不能为空"
    assert len(memory.states) == len(memory.rewards) == len(memory.values)
    return memory


def test_cuda_tensor_and_amp_are_available() -> None:
    device = _gpu_required()
    with guarded_cuda_test():
        x = torch.randn(8, 8, device=device)
        with torch.amp.autocast(device_type="cuda"):
            y = x @ x
        assert y.shape == (8, 8)
        assert torch.isfinite(y).all()


def test_hbgatpn_forward_and_value_low_memory() -> None:
    device = _gpu_required()
    with temporary_config(configs, _low_memory_overrides()), guarded_cuda_test():
        env = _make_env()
        state = env.reset(randomize_duration=False, randomize_workers=False, seed=42)
        model = HBGATPN(configs).to(device)
        model.eval()

        with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
            x_dict, global_context = model(state.to(device))
            value = model.get_value(state.to(device), actor_x_dict_encoded=x_dict)

        assert x_dict["task"].shape[-1] == configs.hidden_dim
        assert x_dict["worker"].shape[-1] == configs.hidden_dim
        assert x_dict["station"].shape[-1] == configs.hidden_dim
        assert global_context.shape[0] == 1
        assert value.shape == (1, 1)
        assert torch.isfinite(value).all()


def test_ppo_select_action_single_env_low_memory() -> None:
    device = _gpu_required()
    with temporary_config(configs, _low_memory_overrides()), guarded_cuda_test():
        env = _make_env()
        state = env.reset(randomize_duration=False, randomize_workers=False, seed=42)
        agent = _make_agent(device)

        action_ret, _ = _first_valid_action(agent, env, state, device)
        action, logprob, value, specific_station_mask, is_invalid = action_ret

        assert action is not None
        assert isinstance(action[0], int)
        assert isinstance(action[1], int)
        assert isinstance(action[2], list)
        assert isinstance(logprob, float)
        assert isinstance(value, float)
        assert specific_station_mask is not None
        assert not is_invalid


def test_ppo_select_actions_batch_two_envs_low_memory() -> None:
    device = _gpu_required()
    with temporary_config(configs, _low_memory_overrides()), guarded_cuda_test():
        envs = [_make_env(42), _make_env(43)]
        states = [env.reset(randomize_duration=False, randomize_workers=False, seed=100 + i) for i, env in enumerate(envs)]
        masks = [env.get_masks() for env in envs]
        agent = _make_agent(device)

        results = agent.select_actions_batch(
            obs_list=states,
            mask_task_list=[m[0] for m in masks],
            mask_station_matrix_list=[m[1] for m in masks],
            mask_worker_list=[m[2] for m in masks],
            deterministic=True,
            is_eval=False,
        )

        assert len(results) == 2
        for action, logprob, value, station_mask, is_invalid in results:
            assert action is not None
            assert isinstance(logprob, float)
            assert isinstance(value, float)
            assert station_mask is not None
            assert not is_invalid


def test_ppo_update_tiny_trajectory_low_memory() -> None:
    device = _gpu_required()
    with temporary_config(configs, _low_memory_overrides()), guarded_cuda_test():
        env = _make_env()
        agent = _make_agent(device)
        memory = _collect_tiny_memory(agent, env, device, steps=3)

        metrics = agent.update(memory, env, current_ep=1)

        assert "Loss/Total" in metrics
        assert "Policy/ApproxKL" in metrics
        assert "Memory/Allocated_GB" in metrics
        assert "Memory/Reserved_GB" in metrics
        assert torch.isfinite(torch.tensor(metrics["Loss/Total"]))
        assert torch.isfinite(torch.tensor(metrics["Policy/ApproxKL"]))
        assert metrics["Memory/Allocated_GB"] <= 2.5
        assert metrics["Memory/Reserved_GB"] <= 3.5


def test_ppo_agent_amp_helpers_match_cuda_device() -> None:
    device = _gpu_required()
    with temporary_config(configs, _low_memory_overrides()), guarded_cuda_test():
        agent = _make_agent(device)
        assert agent.amp_device_type == "cuda"
        assert agent.amp_enabled
        assert agent.scaler.is_enabled()

        with agent.autocast_context():
            x = torch.randn(4, 4, device=device)
            y = x @ x
        assert y.is_cuda
        assert torch.isfinite(y).all()
