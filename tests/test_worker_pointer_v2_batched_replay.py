# -*- coding: utf-8 -*-
"""批量向量化 Worker Pointer v2 的 PPO 路由回归测试。"""

from __future__ import annotations

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
    _small_overrides,
)
from training.memory import Memory


def _append_ready_transition(memory: Memory, agent: PPOAgent, env: AirLineEnv_Graph) -> None:
    """向 memory 追加一个可调度物理工序的真实 v2 动作。"""
    observation, (task_mask, station_mask, worker_mask) = _advance_to_ready_physical_task(
        env
    )
    result = agent.select_actions_batch(
        obs_list=[observation],
        mask_task_list=[task_mask],
        mask_station_matrix_list=[station_mask],
        mask_worker_list=[worker_mask],
        deterministic=False,
        temperature=1.0,
        is_eval=False,
    )[0]
    action, logprob, value, _, invalid = result
    assert not invalid
    if isinstance(logprob, (list, tuple)):
        logprob = logprob[0]
    memory.states.append(env.get_state_snapshot())
    memory.actions.append(action)
    memory.logprobs.append(float(logprob))
    memory.values.append(float(value))
    memory.masks.append((task_mask, station_mask, worker_mask))
    memory.rewards.append(0.0)
    memory.is_terminals.append(True)
    memory.is_truncated.append(False)


def test_batched_v2_update_uses_vectorized_path_not_exact_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """batched-v2 必须复用批量 PPO，不能进入两条 exact replay 路径。"""
    overrides = _small_overrides(
        team_selection_mode="autoregressive_pressure_v2",
        policy_action_scope="operation_station_worker",
        actor_context_mode="attention",
        worker_pointer_v2_behavior_replay=False,
        worker_pointer_v2_replay_mode="batched_vectorized_v2",
        batch_size=2,
        accumulation_steps=1,
    )
    with temporary_config(configs, overrides):
        env_a = AirLineEnv_Graph(DATA_PATH, seed=42)
        env_b = AirLineEnv_Graph(DATA_PATH, seed=43)
        agent = PPOAgent(
            HBGATPN(configs),
            lr=1.0e-4,
            gamma=0.99,
            k_epochs=1,
            eps_clip=0.2,
            device=torch.device("cpu"),
            batch_size=2,
            total_timesteps=2,
            config=configs,
        )
        memory = Memory()
        _append_ready_transition(memory, agent, env_a)
        _append_ready_transition(memory, agent, env_b)
        policy_batch_sizes: list[int] = []
        original_forward = agent.policy.forward

        def _record_batch_forward(batch: object):
            policy_batch_sizes.append(int(batch.num_graphs))
            return original_forward(batch)

        monkeypatch.setattr(agent.policy, "forward", _record_batch_forward)

        def _exact_replay_must_not_run(*_args: object, **_kwargs: object) -> None:
            pytest.fail("batched_vectorized_v2 不得调用 exact replay")

        monkeypatch.setattr(agent, "_run_v2_behavior_replay_update", _exact_replay_must_not_run)
        monkeypatch.setattr(agent, "_run_v2_fast_exact_replay_update", _exact_replay_must_not_run)

        metrics = agent.update(memory, env=env_a, current_ep=1)

    assert metrics["PPO/GradientsFinite"] == 1.0
    assert metrics["V2/BatchedReplayUpdateSeconds"] > 0.0
    assert 2 in policy_batch_sizes
