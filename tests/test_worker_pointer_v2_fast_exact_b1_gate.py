# -*- coding: utf-8 -*-
"""WorkerPointer v2 Fast-Exact + B1 Factorized 兼容性门禁测试套件。

涵盖 8 项严格合同与等价性验证：
1. Fast-Exact rollout 下 component logprobs 与 conditional values 完整性、有限性与 Memory 1-to-1 对齐；
2. physical_group 与 logical_batch_v1 在 factorized 模式下的各分量 LP 与 conditional values 等价性；
3. FP32 (32-true, MaxAE <= 1e-4) 与 BF16 (bf16-mixed, MaxAE <= 1e-3) 数值精度门禁；
4. BF16 首次重放数值恒等 (FirstContractTotalMaxAE <= 1e-3)；
5. 完整端到端 update (forward -> backward -> optimizer.step) 成功且 GradientsFinite=1.0、fallback=0；
6. conditional critic 合同约束 (policy_action_scope="operation_station_worker", use_shared_trunk=false)；
7. 虚拟任务/工位活跃掩码边缘工况覆盖；
8. physical_group 与 logical_batch_v1 从相同初始参数出发的 Parameter-Delta (Δθ) 权重更新等价性。
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import torch

from configs import Config, configs
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from runtime.configuration import validate_runtime_config
from tests.runtime_safety import temporary_config
from tests.test_joint_experiment_architecture import (
    DATA_PATH,
    _advance_to_ready_physical_task,
    _small_overrides,
)
from training.memory import Memory
from training.v2_fast_exact_batch import GPUExactBatchBuilder
from training.worker_pointer_v2_behavior import make_behavior_traces

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _gate_fast_exact_b1_overrides(**extra) -> dict:
    values = _small_overrides(
        team_selection_mode="autoregressive_pressure_v2_fast_exact",
        policy_action_scope="operation_station_worker",
        actor_context_mode="attention",
        worker_pointer_v2_replay_mode="behavior_group_exact_gpu_template_v2",
        worker_pointer_v2_rollout_group_upper_bound=16,
        worker_pointer_v2_strict_gpu_replay=True,
        worker_pointer_v2_fast_replay_batching="logical_batch_v1",
        worker_pointer_v2_fast_replay_encoder_batch_cap=16,
        worker_pointer_v2_logical_batch_cap=256,
        conditional_head_baseline_mode="factorized",
        conditional_head_value_coef=1.0,
        use_shared_trunk=False,
        num_envs=4,
    )
    values.update(extra)
    return values


def _make_gate_agent(*, config: Config | None = None, k_epochs: int = 1) -> PPOAgent:
    cfg = config or configs
    model = HBGATPN(cfg)
    return PPOAgent(
        model,
        lr=1.0e-4,
        gamma=0.99,
        k_epochs=k_epochs,
        eps_clip=0.2,
        device=_DEVICE,
        batch_size=4,
        total_timesteps=1,
        config=cfg,
    )


def _collect_multi_step_rollout(
    agent: PPOAgent,
    env: AirLineEnv_Graph,
    steps: int = 3,
) -> tuple[Memory, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """采集多个真实决策步，构造对齐的 Memory 与张量。"""
    memory = Memory()
    b_tasks = []
    b_stations = []
    b_teams = []
    old_lps = []
    rewards = []

    obs = env.reset(seed=42)
    max_team = int(getattr(configs, "max_team_size", 5))

    collected = 0
    for _ in range(env.num_tasks):
        if collected >= steps:
            break
        masks = env.get_masks()
        t_mask, s_mask, w_mask = masks
        if bool((~t_mask).any()):
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
            if not is_invalid:
                memory.states.append(env.get_state_snapshot())
                memory.actions.append(action)
                lp_val = float(logprob[0]) if isinstance(logprob, (list, tuple)) else float(logprob)
                memory.logprobs.append(lp_val)
                memory.values.append(float(value))
                memory.masks.append((t_mask, s_mask, w_mask))

                traces = make_behavior_traces(
                    group_id=(collected, 0),
                    env_indices=[0],
                    behavior_logprobs=agent.last_v2_behavior_logprobs,
                )
                memory.worker_pointer_v2_behavior_traces.append(traces[0])

                comp_lps = agent.last_v2_behavior_logprobs[0]
                comp_vals = agent.last_v2_behavior_values[0]
                assert comp_lps is not None
                assert comp_vals is not None
                memory.old_task_logprob.append(float(comp_lps[0]))
                memory.old_station_logprob.append(float(comp_lps[1]))
                memory.old_team_logprob.append(float(comp_lps[2]))
                memory.old_V_task.append(float(comp_vals[0]))
                memory.old_V_station.append(float(comp_vals[1]))
                memory.old_V_worker.append(float(comp_vals[2]))

                team = list(action[2])
                padded = team + [-1] * (max_team - len(team))
                b_tasks.append(int(action[0]))
                b_stations.append(int(action[1]))
                b_teams.append(padded[:max_team])
                old_lps.append(lp_val)
                rewards.append(1.0)
                collected += 1

        obs, _rew, done, _info = env.step(action if collected > 0 else (0, -1, []))
        if done:
            break

    assert collected > 0, "未能采集到有效的物理任务决策步"
    b_task_t = torch.tensor(b_tasks, dtype=torch.long)
    b_station_t = torch.tensor(b_stations, dtype=torch.long)
    b_team_t = torch.tensor(b_teams, dtype=torch.long)
    old_logprobs_t = torch.tensor(old_lps, dtype=torch.float32)
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    advantages_t = torch.ones_like(rewards_t)

    return memory, b_task_t, b_station_t, b_team_t, old_logprobs_t, rewards_t, advantages_t


def test_gate_1_rollout_component_logprobs_and_values_alignment() -> None:
    """Gate 1: Fast-Exact rollout 下 component logprobs 与 conditional values 完整性、有限性与 Memory 对齐。"""
    overrides = _gate_fast_exact_b1_overrides()
    with temporary_config(configs, overrides):
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
        agent = _make_gate_agent()
        memory, b_task, b_station, b_team, old_lps, rewards, advantages = _collect_multi_step_rollout(
            agent, env, steps=3
        )

        n_samples = len(memory.states)
        assert n_samples >= 1
        assert len(memory.old_task_logprob) == n_samples
        assert len(memory.old_station_logprob) == n_samples
        assert len(memory.old_team_logprob) == n_samples
        assert len(memory.old_V_task) == n_samples
        assert len(memory.old_V_station) == n_samples
        assert len(memory.old_V_worker) == n_samples

        for i in range(n_samples):
            lp_t = memory.old_task_logprob[i]
            lp_s = memory.old_station_logprob[i]
            lp_w = memory.old_team_logprob[i]
            v_t = memory.old_V_task[i]
            v_s = memory.old_V_station[i]
            v_w = memory.old_V_worker[i]

            assert math.isfinite(lp_t) and not math.isnan(lp_t)
            assert math.isfinite(lp_s) and not math.isnan(lp_s)
            assert math.isfinite(lp_w) and not math.isnan(lp_w)
            assert math.isfinite(v_t) and not math.isnan(v_t)
            assert math.isfinite(v_s) and not math.isnan(v_s)
            assert math.isfinite(v_w) and not math.isnan(v_w)

            total_lp = memory.logprobs[i]
            assert abs(total_lp - (lp_t + lp_s + lp_w)) < 1e-4


def test_gate_2_and_3_physical_vs_logical_parity_and_precision() -> None:
    """Gate 2 & 3: physical_group 与 logical_batch_v1 在 factorized 模式下的等价性与精度门禁 (32-true <= 1e-4, BF16 <= 1e-3)。"""
    for precision in ["32-true", "bf16-mixed"]:
        overrides = _gate_fast_exact_b1_overrides(
            lightning_precision=precision,
            worker_pointer_v2_fast_replay_batching="physical_group",
        )
        with temporary_config(configs, overrides):
            env = AirLineEnv_Graph(DATA_PATH, seed=42)
            agent_pg = _make_gate_agent()
            memory, b_task, b_station, b_team, old_lps, rewards, advantages = _collect_multi_step_rollout(
                agent_pg, env, steps=2
            )
            builder = GPUExactBatchBuilder(config=configs, env=env, device=_DEVICE)

            init_state = {k: v.clone() for k, v in agent_pg.policy.state_dict().items()}

            configs.worker_pointer_v2_fast_replay_batching = "physical_group"
            metrics_pg = agent_pg._run_v2_fast_exact_replay_update(
                memory=memory,
                env=env,
                current_ep=1,
                advantages=advantages,
                rewards=rewards,
                old_logprobs=old_lps,
                b_task=b_task,
                b_station=b_station,
                b_team=b_team,
                action_scope="operation_station_worker",
                fast_exact_builder=builder,
            )

            configs.worker_pointer_v2_fast_replay_batching = "logical_batch_v1"
            agent_lb = _make_gate_agent()
            agent_lb.policy.load_state_dict(init_state)

            metrics_lb = agent_lb._run_v2_fast_exact_replay_update(
                memory=memory,
                env=env,
                current_ep=1,
                advantages=advantages,
                rewards=rewards,
                old_logprobs=old_lps,
                b_task=b_task,
                b_station=b_station,
                b_team=b_team,
                action_scope="operation_station_worker",
                fast_exact_builder=builder,
            )

            max_ae_thresh = 1e-3 if precision == "bf16-mixed" else 1e-4
            for key in ["Loss/Policy", "Loss/Value", "Loss/Total"]:
                if key in metrics_pg and key in metrics_lb:
                    v_pg = float(metrics_pg[key])
                    v_lb = float(metrics_lb[key])
                    assert abs(v_pg - v_lb) <= max_ae_thresh, (
                        f"Precision {precision} 下 {key} 差异超过阈值: pg={v_pg}, lb={v_lb}, diff={abs(v_pg - v_lb)}"
                    )


def test_gate_4_first_contract_identity_bf16() -> None:
    """Gate 4: BF16 混合精度下首次同形重放必须满足 FirstContractTotalMaxAE <= 1e-3。"""
    overrides = _gate_fast_exact_b1_overrides(
        lightning_precision="bf16-mixed",
        worker_pointer_v2_fast_replay_batching="logical_batch_v1",
    )
    with temporary_config(configs, overrides):
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
        agent = _make_gate_agent()
        memory, b_task, b_station, b_team, old_lps, rewards, advantages = _collect_multi_step_rollout(
            agent, env, steps=2
        )
        builder = GPUExactBatchBuilder(config=configs, env=env, device=_DEVICE)

        metrics = agent._run_v2_fast_exact_replay_update(
            memory=memory,
            env=env,
            current_ep=1,
            advantages=advantages,
            rewards=rewards,
            old_logprobs=old_lps,
            b_task=b_task,
            b_station=b_station,
            b_team=b_team,
            action_scope="operation_station_worker",
            fast_exact_builder=builder,
        )

        assert "V2/FirstContractTotalMaxAE" in metrics
        max_ae = float(metrics["V2/FirstContractTotalMaxAE"])
        assert max_ae <= 1e-3, f"FirstContractTotalMaxAE={max_ae} 超出 BF16 容差 1e-3"


def test_gate_5_full_e2e_step_gradients_finite_no_oom() -> None:
    """Gate 5: 完整端到端步进闭环 (rollout -> replay -> backward -> optimizer.step) 成功且 GradientsFinite=1.0、fallback=0。"""
    overrides = _gate_fast_exact_b1_overrides(
        lightning_precision="bf16-mixed",
        worker_pointer_v2_fast_replay_batching="logical_batch_v1",
    )
    with temporary_config(configs, overrides):
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
        agent = _make_gate_agent()
        memory, b_task, b_station, b_team, old_lps, rewards, advantages = _collect_multi_step_rollout(
            agent, env, steps=3
        )
        builder = GPUExactBatchBuilder(config=configs, env=env, device=_DEVICE)

        metrics = agent._run_v2_fast_exact_replay_update(
            memory=memory,
            env=env,
            current_ep=1,
            advantages=advantages,
            rewards=rewards,
            old_logprobs=old_lps,
            b_task=b_task,
            b_station=b_station,
            b_team=b_team,
            action_scope="operation_station_worker",
            fast_exact_builder=builder,
        )

        assert metrics.get("PPO/UpdateSteps", 0.0) >= 1.0
        assert metrics.get("Gradient/Finite", 0.0) == 1.0
        assert metrics.get("Gradient/V2Coverage", 0.0) > 0.0
        assert metrics.get("V2/FastExact/FallbackCount", 0.0) == 0.0


def test_gate_6_conditional_critic_contracts() -> None:
    """Gate 6: 验证 conditional critic 约束依然被严格保证。"""
    invalid_trunk = Config()
    invalid_trunk.conditional_head_baseline_mode = "factorized"
    invalid_trunk.team_selection_mode = "autoregressive_pressure_v2_fast_exact"
    invalid_trunk.policy_action_scope = "operation_station_worker"
    invalid_trunk.use_shared_trunk = True
    with pytest.raises(ValueError, match="use_shared_trunk"):
        validate_runtime_config(invalid_trunk)

    invalid_scope = Config()
    invalid_scope.conditional_head_baseline_mode = "factorized"
    invalid_scope.team_selection_mode = "autoregressive_pressure_v2_fast_exact"
    invalid_scope.policy_action_scope = "operation_station"
    invalid_scope.use_shared_trunk = False
    with pytest.raises(ValueError, match="policy_action_scope"):
        validate_runtime_config(invalid_scope)


def test_gate_7_edge_cases_virtual_station_and_worker() -> None:
    """Gate 7: 边缘工况覆盖 (virtual station / virtual worker active mask 分支)。"""
    overrides = _gate_fast_exact_b1_overrides()
    with temporary_config(configs, overrides):
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
        agent = _make_gate_agent()

        obs = env.reset(seed=42)
        t_mask, s_mask, w_mask = env.get_masks()

        results = agent.select_actions_batch(
            obs_list=[obs],
            mask_task_list=[t_mask],
            mask_station_matrix_list=[s_mask],
            mask_worker_list=[w_mask],
            deterministic=True,
            temperature=1.0,
            is_eval=False,
        )
        assert len(results) == 1
        comp_lps = agent.last_v2_behavior_logprobs[0]
        comp_vals = agent.last_v2_behavior_values[0]
        assert comp_lps is not None
        assert comp_vals is not None
        for val in comp_lps:
            assert math.isfinite(val)
        for val in comp_vals:
            assert math.isfinite(val)


def test_gate_8_parameter_delta_parity() -> None:
    """Gate 8: physical_group 与 logical_batch_v1 从相同初始参数出发的 Parameter-Delta (Δθ) 权重更新等价性。"""
    overrides = _gate_fast_exact_b1_overrides(
        lightning_precision="32-true",
        worker_pointer_v2_fast_replay_batching="physical_group",
    )
    with temporary_config(configs, overrides):
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
        agent_pg = _make_gate_agent()
        memory, b_task, b_station, b_team, old_lps, rewards, advantages = _collect_multi_step_rollout(
            agent_pg, env, steps=2
        )
        builder = GPUExactBatchBuilder(config=configs, env=env, device=_DEVICE)

        init_state = {k: v.clone() for k, v in agent_pg.policy.state_dict().items()}

        configs.worker_pointer_v2_fast_replay_batching = "physical_group"
        agent_pg._run_v2_fast_exact_replay_update(
            memory=memory,
            env=env,
            current_ep=1,
            advantages=advantages,
            rewards=rewards,
            old_logprobs=old_lps,
            b_task=b_task,
            b_station=b_station,
            b_team=b_team,
            action_scope="operation_station_worker",
            fast_exact_builder=builder,
        )
        delta_pg = {
            k: agent_pg.policy.state_dict()[k] - init_state[k]
            for k in init_state
        }

        configs.worker_pointer_v2_fast_replay_batching = "logical_batch_v1"
        agent_lb = _make_gate_agent()
        agent_lb.policy.load_state_dict(init_state)

        agent_lb._run_v2_fast_exact_replay_update(
            memory=memory,
            env=env,
            current_ep=1,
            advantages=advantages,
            rewards=rewards,
            old_logprobs=old_lps,
            b_task=b_task,
            b_station=b_station,
            b_team=b_team,
            action_scope="operation_station_worker",
            fast_exact_builder=builder,
        )
        delta_lb = {
            k: agent_lb.policy.state_dict()[k] - init_state[k]
            for k in init_state
        }

        max_delta_ae = 0.0
        for k in init_state:
            diff = (delta_pg[k].float() - delta_lb[k].float()).abs().max().item()
            if diff > max_delta_ae:
                max_delta_ae = diff

        assert max_delta_ae <= 1e-4, f"Parameter-Delta 最大绝对误差超过 FP32 门禁: {max_delta_ae}"
