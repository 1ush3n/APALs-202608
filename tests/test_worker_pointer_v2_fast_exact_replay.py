# -*- coding: utf-8 -*-
"""WorkerPointer v2 Fast-Exact 阶段二C：PPO 同形 GPU group 重放集成合同。

覆盖：
- Fast-Exact 重放更新跑通（GPU builder + 布局元数据 + actor-only 预检复用）；
- 首次同形合同 actor-only 预检（不计算 critic）且输出复用于首次 PPO 计算；
- 首次同形合同 fail-closed（backward 前抛错，参数不被污染）；
- 返回指标全部有限且梯度覆盖正常。
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
from tests.test_worker_pointer_v2_fast_exact_rollout import _fast_exact_config
from training.memory import Memory
from training.v2_fast_exact_batch import GPUExactBatchBuilder
from training.worker_pointer_v2_behavior import make_behavior_traces

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fast_exact_overrides() -> dict:
    return _small_overrides(
        team_selection_mode="autoregressive_pressure_v2_fast_exact",
        policy_action_scope="operation_station_worker",
        actor_context_mode="attention",
        worker_pointer_v2_replay_mode="behavior_group_exact_gpu_template_v2",
        worker_pointer_v2_rollout_group_upper_bound=16,
        worker_pointer_v2_strict_gpu_replay=True,
        num_envs=4,
    )


def _make_agent(*, k_epochs: int = 1) -> PPOAgent:
    return PPOAgent(
        HBGATPN(configs),
        lr=1.0e-4,
        gamma=0.99,
        k_epochs=k_epochs,
        eps_clip=0.2,
        device=_DEVICE,
        batch_size=4,
        total_timesteps=1,
        config=configs,
    )


def _rollout_single_step(agent: PPOAgent, env: AirLineEnv_Graph) -> tuple:
    """执行一次 fast-exact 决策，构造与动作对齐的 memory 与目标张量。"""
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


def test_fast_exact_replay_update_runs_with_gpu_builder() -> None:
    with temporary_config(configs, _fast_exact_overrides()):
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
        agent = _make_agent()
        (
            memory,
            b_task,
            b_station,
            b_team,
            old_logprobs,
            rewards,
            advantages,
        ) = _rollout_single_step(agent, env)
        builder = GPUExactBatchBuilder(
            config=configs, env=env, device=_DEVICE
        )

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

        assert metrics["V2/FastExact/BehaviorReplayGroups"] == 1.0
        assert metrics["V2/FastExact/BehaviorReplaySamples"] == 1.0
        assert metrics["V2/FastExact/PrecheckReusedGroups"] >= 1.0
        assert metrics["PPO/UpdateSteps"] == 1.0
        assert metrics["Gradient/Finite"] == 1.0
        assert metrics["Gradient/V2Coverage"] > 0.0
        assert metrics["V2/FirstContractTotalMaxAE"] <= 1.0e-4
        for value in metrics.values():
            assert torch.isfinite(torch.tensor(value)), f"非有限指标: {value}"


def test_fast_exact_profile_reports_replay_stage_metrics() -> None:
    overrides = _fast_exact_overrides()
    overrides["worker_pointer_v2_fast_exact_profile"] = True
    with temporary_config(configs, overrides):
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
        agent = _make_agent()
        (
            memory,
            b_task,
            b_station,
            b_team,
            old_logprobs,
            rewards,
            advantages,
        ) = _rollout_single_step(agent, env)
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

        expected = (
            "PhysicalGroupCount",
            "PhysicalGroupMeanSize",
            "PhysicalGroupP50Size",
            "PhysicalGroupP95Size",
            "BuilderCalls",
            "BuilderMs",
            "EncoderCalls",
            "EncoderMs",
            "ActionHeadCalls",
            "ActionHeadMs",
            "WorkerPointerCalls",
            "WorkerPointerMs",
            "PrecheckMs",
            "FormalReplayCalls",
            "FormalReplayMs",
            "BackwardMs",
            "OptimizerMs",
            "ReplaySamplesPerSec",
        )
        for suffix in expected:
            key = f"V2/FastExact/Profile/{suffix}"
            assert key in metrics
            assert torch.isfinite(torch.tensor(metrics[key]))


def test_fast_exact_factorized_replay_update_uses_component_contract() -> None:
    overrides = _fast_exact_overrides()
    overrides["conditional_head_baseline_mode"] = "factorized"
    with temporary_config(configs, overrides):
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
        agent = _make_agent()
        (
            memory,
            b_task,
            b_station,
            b_team,
            old_logprobs,
            rewards,
            advantages,
        ) = _rollout_single_step(agent, env)
        component_logprobs = agent.last_v2_behavior_logprobs[0]
        conditional_values = agent.last_v2_behavior_values[0]
        assert component_logprobs is not None
        assert conditional_values is not None
        memory.old_task_logprob.append(float(component_logprobs[0]))
        memory.old_station_logprob.append(float(component_logprobs[1]))
        memory.old_team_logprob.append(float(component_logprobs[2]))
        memory.old_V_task.append(float(conditional_values[0]))
        memory.old_V_station.append(float(conditional_values[1]))
        memory.old_V_worker.append(float(conditional_values[2]))
        builder = GPUExactBatchBuilder(
            config=configs, env=env, device=_DEVICE
        )

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

        assert metrics["PPO/UpdateSteps"] == 1.0
        assert metrics["Gradient/Finite"] == 1.0
        assert metrics["Gradient/V2Coverage"] > 0.0


def test_fast_exact_actor_only_precheck_skips_critic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """首次同形合同为 actor-only：预检阶段不得调用 critic，且结果被复用。"""
    with temporary_config(configs, _fast_exact_overrides()):
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
        agent = _make_agent()
        (
            memory,
            b_task,
            b_station,
            b_team,
            old_logprobs,
            rewards,
            advantages,
        ) = _rollout_single_step(agent, env)
        builder = GPUExactBatchBuilder(
            config=configs, env=env, device=_DEVICE
        )

        original_get_value = agent.policy.get_value
        get_value_calls: list[torch.Tensor] = []

        def counting_get_value(batch, *args, **kwargs):
            get_value_calls.append(batch)
            return original_get_value(batch, *args, **kwargs)

        monkeypatch.setattr(agent.policy, "get_value", counting_get_value)
        # 预检阶段：get_value 不应被调用。
        actor_only = agent._replay_v2_fast_exact_group(
            agent._build_v2_fast_exact_group(
                memory=memory,
                memory_indices=[0],
                b_task=b_task,
                b_station=b_station,
                b_team=b_team,
                old_logprobs=old_logprobs,
                rewards=rewards,
                advantages=advantages,
                fast_exact_builder=builder,
            ),
            actor_only=True,
        )
        assert len(get_value_calls) == 0
        assert len(actor_only) == 1

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
        # 预检不调用 critic；正式带梯度前向调用一次。
        assert metrics["V2/FastExact/PrecheckReusedGroups"] == 1.0
        assert len(get_value_calls) == 1


def test_fast_exact_actor_prepass_is_recomputed_after_optimizer_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """策略参数更新后不得复用首次合同产生的 actor-only 输出。"""
    with temporary_config(configs, _fast_exact_overrides()):
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
        agent = _make_agent(k_epochs=2)
        (
            memory,
            b_task,
            b_station,
            b_team,
            old_logprobs,
            rewards,
            advantages,
        ) = _rollout_single_step(agent, env)
        builder = GPUExactBatchBuilder(
            config=configs, env=env, device=_DEVICE
        )

        original_replay = agent._replay_v2_fast_exact_group
        actor_only_calls = 0

        def counting_replay(fast_batch, *, actor_only=False):
            nonlocal actor_only_calls
            if actor_only:
                actor_only_calls += 1
            return original_replay(fast_batch, actor_only=actor_only)

        monkeypatch.setattr(
            agent,
            "_replay_v2_fast_exact_group",
            counting_replay,
        )
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

        # 第一次来自合同预检；第二次来自 optimizer.step 后的新策略重算。
        assert actor_only_calls == 2
        assert metrics["V2/FastExact/PrecheckGroups"] == 1.0
        assert metrics["V2/FastExact/PrecheckReusedGroups"] == 1.0
        assert metrics["PPO/UpdateSteps"] == 2.0


def test_fast_exact_replay_fails_closed_on_contract_break() -> None:
    with temporary_config(configs, _fast_exact_overrides()):
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
        agent = _make_agent()
        (
            memory,
            b_task,
            b_station,
            b_team,
            old_logprobs,
            rewards,
            advantages,
        ) = _rollout_single_step(agent, env)
        trace = memory.worker_pointer_v2_behavior_traces[0]
        corrupted = dataclasses.replace(trace, station_lp=trace.station_lp + 0.5)
        memory.worker_pointer_v2_behavior_traces[0] = corrupted
        before = {
            name: param.detach().cpu().clone()
            for name, param in agent.policy.named_parameters()
        }
        builder = GPUExactBatchBuilder(
            config=configs, env=env, device=_DEVICE
        )
        with pytest.raises(ValueError, match="station"):
            agent._run_v2_fast_exact_replay_update(
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
        for name, param in agent.policy.named_parameters():
            assert torch.equal(param.detach().cpu(), before[name]), (
                f"fail-closed 后参数被更新: {name}"
            )


def test_fast_exact_update_rethrows_after_oom_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fast-Exact + strict_gpu_replay：OOM 回滚后必须重新抛出，绝不静默跳过。"""
    with temporary_config(configs, _fast_exact_overrides()):
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
        agent = _make_agent()

        def boom(*_args: object, **_kwargs: object) -> dict[str, float]:
            raise RuntimeError(
                "CUDA out of memory. Tried to allocate 128.00 MiB"
            )

        monkeypatch.setattr(agent, "_update_once", boom)
        with pytest.raises(RuntimeError, match="CUDA OOM"):
            agent.update(Memory(), env=env, current_ep=1)


def test_fast_exact_update_dispatches_to_fast_exact_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fast-Exact 配置下 _update_once 必须分派到 _run_v2_fast_exact_replay_update。"""
    with temporary_config(configs, _fast_exact_overrides()):
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
        agent = _make_agent()
        memory, _b_task, _b_station, _b_team, _old_lp, _r, _adv = (
            _rollout_single_step(agent, env)
        )
        memory.rewards = [0.0]
        memory.is_terminals = [False]

        dispatched: list[str] = []

        def fake_fast_exact_replay(*_args: object, **_kwargs: object) -> dict[str, float]:
            dispatched.append("called")
            return {"V2/FastExact/BehaviorReplayGroups": 1.0}

        monkeypatch.setattr(
            agent, "_run_v2_fast_exact_replay_update", fake_fast_exact_replay
        )
        agent._update_once(memory, env, current_ep=1)
        assert dispatched == ["called"]
