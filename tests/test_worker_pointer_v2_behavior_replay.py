# -*- coding: utf-8 -*-
"""WorkerPointer v2 行为组同形重放集成合同。

覆盖：
- 真实 v2 模型 + 环境：行为（select_actions_batch）与重放（B=1 同形）三部分
  log-prob 一致，首次同形合同 MaxAE 通过；
- 聚合与窗口归一化跑通，返回指标无 NaN/Inf；
- 篡改行为轨迹时首次同形合同 fail-closed（backward 前抛错）。
"""

from __future__ import annotations

import dataclasses

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
from training.worker_pointer_v2_behavior import make_behavior_traces

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _v2_overrides() -> dict:
    return _small_overrides(
        team_selection_mode="autoregressive_pressure_v2",
        policy_action_scope="operation_station_worker",
        actor_context_mode="attention",
        worker_pointer_v2_behavior_replay=True,
        worker_pointer_v2_logical_batch_cap=4,
    )


def _rollout_single_step(
    agent: PPOAgent, env: AirLineEnv_Graph
) -> tuple[Memory, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """执行一次 v2 决策，构造与动作对齐的 memory 与目标张量。"""
    obs, (t_mask, s_mask, w_mask) = _advance_to_ready_physical_task(env)
    results = agent.select_actions_batch(
        obs_list=[obs],
        mask_task_list=[t_mask],
        mask_station_matrix_list=[s_mask],
        mask_worker_list=[w_mask],
        deterministic=False,
        temperature=1.0,
        is_eval=False,
    )
    action, logprob, value, _, is_invalid = results[0]
    assert not is_invalid

    memory = Memory()
    memory.states.append(env.get_state_snapshot())
    memory.actions.append(action)
    # select_actions_batch 返回的 logprob 可能为单元素 list（B=1 stack 语义）。
    if isinstance(logprob, (list, tuple)):
        logprob = float(logprob[0])
    memory.logprobs.append(float(logprob))
    memory.values.append(float(value))
    memory.masks.append((t_mask, s_mask, w_mask))
    traces = make_behavior_traces(
        group_id=(0, 0),
        env_indices=[0],
        behavior_logprobs=agent.last_v2_behavior_logprobs,
    )
    assert len(traces) == 1
    memory.worker_pointer_v2_behavior_traces.append(traces[0])

    team = list(action[2])
    max_team = int(getattr(configs, "max_team_size", 5))
    padded = team + [-1] * (max_team - len(team))
    b_task = torch.tensor([int(action[0])], dtype=torch.long)
    b_station = torch.tensor([int(action[1])], dtype=torch.long)
    b_team = torch.tensor([padded[:max_team]], dtype=torch.long)
    old_logprobs = torch.tensor([float(logprob)], dtype=torch.float32)
    rewards = torch.tensor([0.0], dtype=torch.float32)
    advantages = torch.tensor([1.0], dtype=torch.float32)
    return memory, b_task, b_station, b_team, old_logprobs, rewards, advantages


def test_v2_behavior_replay_first_contract_passes_and_aggregation_runs() -> None:
    with temporary_config(configs, _v2_overrides()):
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
        agent = PPOAgent(
            HBGATPN(configs),
            lr=1.0e-4,
            gamma=0.99,
            k_epochs=1,
            eps_clip=0.2,
            device=_DEVICE,
            batch_size=4,
            total_timesteps=1,
            config=configs,
        )
        (
            memory,
            b_task,
            b_station,
            b_team,
            old_logprobs,
            rewards,
            advantages,
        ) = _rollout_single_step(agent, env)
        metrics = agent._run_v2_behavior_replay_update(
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
        )

        assert metrics["V2/FirstContractTotalMaxAE"] <= 1.0e-4
        assert metrics["V2/BehaviorReplayGroups"] == 1.0
        assert metrics["V2/BehaviorReplaySamples"] == 1.0
        assert metrics["PPO/UpdateSteps"] == 1.0
        assert metrics["PPO/GradientsFinite"] == 1.0
        assert metrics["PointerV2/GradientNorm"] > 0.0
        assert metrics["PointerV2/GradientCoverage"] > 0.0
        assert metrics["PointerV2/PPOFirstRecomputeMaxAE"] <= 1.0e-4
        assert metrics["PointerV2/AutocastEnabled"] == float(agent.amp_enabled)
        assert metrics["PointerV2/GradScalerEnabled"] == float(
            agent.scaler.is_enabled()
        )
        assert metrics["PointerV2/NonFiniteCount"] == 0.0
        for value in metrics.values():
            assert torch.isfinite(torch.tensor(value)), f"非有限指标: {value}"


def test_v2_behavior_snapshot_stores_component_logprobs_and_conditional_values() -> None:
    overrides = _v2_overrides()
    overrides["conditional_head_baseline_mode"] = "diagnostic"
    with temporary_config(configs, overrides):
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
        agent = PPOAgent(
            HBGATPN(configs),
            lr=1.0e-4,
            gamma=0.99,
            k_epochs=1,
            eps_clip=0.2,
            device=_DEVICE,
            batch_size=4,
            total_timesteps=1,
            config=configs,
        )
        obs, (task_mask, station_mask, worker_mask) = _advance_to_ready_physical_task(env)
        result = agent.select_actions_batch(
            [obs],
            [task_mask],
            [station_mask],
            [worker_mask],
            deterministic=False,
            temperature=1.0,
            is_eval=False,
        )

        assert not result[0][-1]
        component_logprobs = agent.last_v2_behavior_logprobs[0]
        conditional_values = agent.last_v2_behavior_values[0]
        assert component_logprobs is not None
        assert conditional_values is not None
        observed_logprob = result[0][1]
        if isinstance(observed_logprob, (list, tuple)):
            observed_logprob = observed_logprob[0]
        assert sum(component_logprobs) == pytest.approx(
            observed_logprob, abs=1.0e-6
        )
        assert all(torch.isfinite(torch.tensor(value)) for value in conditional_values)


def test_v2_behavior_replay_fails_closed_on_contract_break() -> None:
    with temporary_config(configs, _v2_overrides()):
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
        agent = PPOAgent(
            HBGATPN(configs),
            lr=1.0e-4,
            gamma=0.99,
            k_epochs=1,
            eps_clip=0.2,
            device=_DEVICE,
            batch_size=4,
            total_timesteps=1,
            config=configs,
        )
        (
            memory,
            b_task,
            b_station,
            b_team,
            old_logprobs,
            rewards,
            advantages,
        ) = _rollout_single_step(agent, env)
        # 篡改行为轨迹的 station 分量，重放必须 fail-closed。
        trace = memory.worker_pointer_v2_behavior_traces[0]
        corrupted = dataclasses.replace(trace, station_lp=trace.station_lp + 0.5)
        memory.worker_pointer_v2_behavior_traces[0] = corrupted
        before = {
            name: param.detach().cpu().clone()
            for name, param in agent.policy.named_parameters()
        }
        with pytest.raises(ValueError, match="station"):
            agent._run_v2_behavior_replay_update(
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
            )
        # 安全合同：fail-closed 前不得发生参数更新。
        for name, param in agent.policy.named_parameters():
            assert torch.equal(param.detach().cpu(), before[name]), (
                f"fail-closed 后参数被更新: {name}"
            )


def test_v2_behavior_replay_rejects_mismatched_traces() -> None:
    with temporary_config(configs, _v2_overrides()):
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
        agent = PPOAgent(
            HBGATPN(configs),
            lr=1.0e-4,
            gamma=0.99,
            k_epochs=1,
            eps_clip=0.2,
            device=_DEVICE,
            batch_size=4,
            total_timesteps=1,
            config=configs,
        )
        (
            memory,
            b_task,
            b_station,
            b_team,
            old_logprobs,
            rewards,
            advantages,
        ) = _rollout_single_step(agent, env)
        memory.worker_pointer_v2_behavior_traces = []
        with pytest.raises(RuntimeError, match="行为轨迹"):
            agent._run_v2_behavior_replay_update(
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
            )


def test_v2_group_replay_uses_group_encoder_and_single_sample_heads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """编码器/critic 必须保持原 group 形状，动作 head 必须复现 rollout 的 B=1。"""
    with temporary_config(configs, _v2_overrides()):
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
        agent = PPOAgent(
            HBGATPN(configs),
            lr=1.0e-4,
            gamma=0.99,
            k_epochs=1,
            eps_clip=0.2,
            device=_DEVICE,
            batch_size=4,
            total_timesteps=1,
            config=configs,
        )
        (
            memory,
            b_task,
            b_station,
            b_team,
            old_logprobs,
            rewards,
            advantages,
        ) = _rollout_single_step(agent, env)

        # 用同一合法快照构造两成员原组；这里验证的是张量形状合同，不比较采样随机性。
        memory.states.append(memory.states[0])
        memory.actions.append(memory.actions[0])
        memory.logprobs.append(memory.logprobs[0])
        memory.values.append(memory.values[0])
        memory.masks.append(memory.masks[0])
        memory.rewards.extend([0.0, 0.0])
        first_trace = memory.worker_pointer_v2_behavior_traces[0]
        memory.worker_pointer_v2_behavior_traces[0] = dataclasses.replace(
            first_trace, group_size=2, group_position=0, env_index=0
        )
        memory.worker_pointer_v2_behavior_traces.append(
            dataclasses.replace(
                first_trace, group_size=2, group_position=1, env_index=1
            )
        )
        b_task = b_task.repeat(2)
        b_station = b_station.repeat(2)
        b_team = b_team.repeat(2, 1)
        old_logprobs = old_logprobs.repeat(2)
        rewards = torch.zeros(2, dtype=torch.float32)
        advantages = torch.ones(2, dtype=torch.float32)

        encoder_batch_sizes: list[int] = []
        task_head_batch_sizes: list[int] = []
        station_head_batch_sizes: list[int] = []
        worker_head_batch_sizes: list[int] = []
        original_policy_forward = agent.policy.forward
        original_task_forward = agent.policy.task_head.forward
        original_station_forward = agent.policy.station_head.forward
        original_worker_forward = agent.policy.worker_head.forward_choice_v2

        def policy_forward(batch):
            encoder_batch_sizes.append(int(batch.num_graphs))
            return original_policy_forward(batch)

        def task_forward(task_embeddings, global_context, **kwargs):
            # rollout 向 TaskPointer 传 [N,H]，模块内部扩为 [1,N,H]。
            effective_batch = 1 if task_embeddings.ndim == 2 else int(task_embeddings.shape[0])
            task_head_batch_sizes.append(effective_batch)
            assert int(global_context.shape[0]) == 1
            return original_task_forward(task_embeddings, global_context, **kwargs)

        def station_forward(task_embedding, station_embeddings, **kwargs):
            station_head_batch_sizes.append(int(task_embedding.shape[0]))
            return original_station_forward(task_embedding, station_embeddings, **kwargs)

        def worker_forward(*args, **kwargs):
            worker_head_batch_sizes.append(int(kwargs["task_emb"].shape[0]))
            return original_worker_forward(*args, **kwargs)

        monkeypatch.setattr(agent.policy, "forward", policy_forward)
        monkeypatch.setattr(agent.policy.task_head, "forward", task_forward)
        monkeypatch.setattr(agent.policy.station_head, "forward", station_forward)
        monkeypatch.setattr(agent.policy.worker_head, "forward_choice_v2", worker_forward)

        group_batch = agent._build_v2_behavior_group_batch(
            memory=memory,
            env=env,
            memory_indices=[0, 1],
            b_task=b_task,
            b_station=b_station,
            b_team=b_team,
            old_logprobs=old_logprobs,
            rewards=rewards,
            advantages=advantages,
        )
        outputs = agent._replay_v2_behavior_group(group_batch)

        assert len(outputs) == 2
        assert encoder_batch_sizes == [2]
        assert task_head_batch_sizes == [1, 1]
        assert station_head_batch_sizes == [1, 1]
        assert worker_head_batch_sizes
        assert set(worker_head_batch_sizes) == {1}


def test_v2_optimizer_steps_follow_logical_batches_and_tail_window() -> None:
    """5 个单成员组、cap=2、累积 2：3 个逻辑批应产生 2 次 optimizer step。"""
    overrides = _v2_overrides()
    overrides.update(accumulation_steps=2, worker_pointer_v2_logical_batch_cap=2)
    with temporary_config(configs, overrides):
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
        agent = PPOAgent(
            HBGATPN(configs),
            lr=1.0e-4,
            gamma=0.99,
            k_epochs=1,
            eps_clip=0.2,
            device=_DEVICE,
            batch_size=2,
            total_timesteps=1,
            config=configs,
        )
        (
            memory,
            b_task,
            b_station,
            b_team,
            old_logprobs,
            rewards,
            advantages,
        ) = _rollout_single_step(agent, env)
        base_trace = memory.worker_pointer_v2_behavior_traces[0]
        for group_id in range(1, 5):
            memory.states.append(memory.states[0])
            memory.actions.append(memory.actions[0])
            memory.logprobs.append(memory.logprobs[0])
            memory.values.append(memory.values[0])
            memory.masks.append(memory.masks[0])
            memory.worker_pointer_v2_behavior_traces.append(
                dataclasses.replace(
                    base_trace,
                    group_id=(0, group_id),
                    group_size=1,
                    group_position=0,
                )
            )
        b_task = b_task.repeat(5)
        b_station = b_station.repeat(5)
        b_team = b_team.repeat(5, 1)
        old_logprobs = old_logprobs.repeat(5)
        rewards = torch.zeros(5, dtype=torch.float32)
        advantages = torch.ones(5, dtype=torch.float32)

        metrics = agent._run_v2_behavior_replay_update(
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
        )

        assert metrics["V2/ReplayLogicalBatchCount"] == 3.0
        assert metrics["V2/ReplayLogicalBatchMinSize"] == 1.0
        assert metrics["V2/ReplayLogicalBatchMaxSize"] == 2.0
        assert metrics["PPO/UpdateSteps"] == 2.0
        assert metrics["Gradient/Finite"] == 1.0
        assert metrics["Gradient/V2Coverage"] > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA bf16")
def test_v2_cuda_bf16_replays_original_four_environment_group() -> None:
    """CUDA bf16 回归：行为与首次重算均使用原始四图 encoder 形状。"""
    overrides = _v2_overrides()
    overrides.update(
        lightning_precision="bf16-mixed",
        worker_pointer_v2_logical_batch_cap=64,
        worker_pointer_v2_rollout_group_upper_bound=4,
        accumulation_steps=16,
    )
    with temporary_config(configs, overrides):
        envs = [AirLineEnv_Graph(DATA_PATH, seed=42 + index) for index in range(4)]
        prepared = [_advance_to_ready_physical_task(env) for env in envs]
        agent = PPOAgent(
            HBGATPN(configs),
            lr=1.0e-4,
            gamma=0.99,
            k_epochs=1,
            eps_clip=0.2,
            device=torch.device("cuda"),
            batch_size=64,
            total_timesteps=1,
            config=configs,
        )
        results = agent.select_actions_batch(
            obs_list=[item[0] for item in prepared],
            mask_task_list=[item[1][0] for item in prepared],
            mask_station_matrix_list=[item[1][1] for item in prepared],
            mask_worker_list=[item[1][2] for item in prepared],
            deterministic=False,
            temperature=1.0,
            is_eval=False,
        )
        assert all(not result[4] for result in results)

        memory = Memory()
        for env_index, (env, masks, result) in enumerate(
            zip(envs, prepared, results)
        ):
            action, logprob, value, _station_mask, _invalid = result
            if isinstance(logprob, (list, tuple)):
                logprob = logprob[0]
            memory.states.append(env.get_state_snapshot())
            memory.actions.append(action)
            memory.logprobs.append(float(logprob))
            memory.values.append(float(value))
            memory.masks.append(masks[1])
        memory.worker_pointer_v2_behavior_traces.extend(
            make_behavior_traces(
                group_id=(0, 0),
                env_indices=[0, 1, 2, 3],
                behavior_logprobs=agent.last_v2_behavior_logprobs,
            )
        )
        max_team = max(len(action[2]) for action in memory.actions)
        b_task = torch.tensor([action[0] for action in memory.actions], dtype=torch.long)
        b_station = torch.tensor(
            [action[1] for action in memory.actions], dtype=torch.long
        )
        b_team = torch.tensor(
            [
                action[2] + [-1] * (max_team - len(action[2]))
                for action in memory.actions
            ],
            dtype=torch.long,
        )
        old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32)
        rewards = torch.zeros(4, dtype=torch.float32)
        advantages = torch.ones(4, dtype=torch.float32)

        metrics = agent._run_v2_behavior_replay_update(
            memory,
            envs[0],
            current_ep=1,
            advantages=advantages,
            rewards=rewards,
            old_logprobs=old_logprobs,
            b_task=b_task,
            b_station=b_station,
            b_team=b_team,
            action_scope="operation_station_worker",
        )

        assert agent.amp_dtype == torch.bfloat16
        assert not agent.scaler.is_enabled()
        assert metrics["V2/ReplayMaxPhysicalGroup"] == 4.0
        assert metrics["V2/FirstContractTotalMaxAE"] <= 1.0e-3
        assert metrics["Gradient/Finite"] == 1.0
