# -*- coding: utf-8 -*-
"""
PPO/APAL 训练闭环修复专项测试。

本文件只放低显存、短路径测试，专门覆盖训练闭环里容易污染 PPO 信号的边界问题。
"""

from __future__ import annotations

import sys
import ast
from pathlib import Path

import torch
import subprocess
from torch_geometric.data import Batch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import configs
from core.event_engine import Event, EventType
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from runtime.paths import PROJECT_ROOT as TRAIN_PROJECT_ROOT
from runtime.paths import resolve_workspace_path
from training.memory import Memory
from training.observation import refresh_env_observation
from tests.runtime_safety import seed_everything, temporary_config


def test_wait_refresh_rebuilds_current_observation_after_event_jump() -> None:
    """
    等待事件推进后，刷新得到的观测必须反映新的 current_time 和站位槽位状态。
    这能防止训练循环使用旧图状态配新 mask 采样动作。
    """

    seed_everything(42)
    overrides = {
        "n_w": 40,
        "n_m": 5,
        "max_slots_per_station": 3,
        "enable_dynamic_events": False,
        "enable_station_breakdown": False,
        "enable_material_delay": True,
    }

    with temporary_config(configs, overrides):
        env = AirLineEnv_Graph(data_path_or_dir=str(PROJECT_ROOT / "data" / "283.csv"), seed=42)
        env.reset(randomize_duration=False, randomize_workers=False, seed=42)

        ready_indices = torch.where(torch.from_numpy(env.task_status) == 1)[0].numpy()
        assert len(ready_indices) > 0
        env.task_material_ready[ready_indices] = 5.0
        stale_state = refresh_env_observation(env)
        stale_task_mask, _, _ = env.get_masks()
        assert stale_task_mask.all(), "所有 Ready 工序物料未到达时，应没有合法任务可选。"
        assert stale_state["task"].x[ready_indices[0], 17].item() > 0.0

        env.event_queue.push(
            Event(
                time=5.0,
                type=EventType.MATERIAL_ARRIVE,
                data={"task_id": int(ready_indices[0])},
            )
        )

        assert env.try_wait_for_resources()
        fresh_state = refresh_env_observation(env)
        fresh_task_mask, fresh_station_mask, _ = env.get_masks()

        assert env.current_time == 5.0
        assert fresh_state["task"].x[ready_indices[0], 17].item() == 0.0
        assert not fresh_task_mask.all(), "事件恢复站位槽位后，应重新出现合法任务。"

        valid_tasks = torch.where(~fresh_task_mask)[0]
        assert len(valid_tasks) > 0
        assert not fresh_station_mask[valid_tasks[0]].all()


def test_gae_resets_at_rollout_boundary_between_env_segments() -> None:
    """
    手造两段轨迹，第一段末尾标为 truncated 后，第二段高 reward 不得泄漏回第一段。
    """

    rewards = [1.0, 1.0, 100.0]
    terminals = [False, False, True]
    truncated = [False, True, False]
    values = [0.0, 0.0, 0.0]

    adv, returns = PPOAgent.compute_gae_returns(
        rewards=rewards,
        terminals=terminals,
        values=values,
        gamma=1.0,
        gae_lambda=1.0,
        truncated=truncated,
    )

    assert torch.allclose(adv, torch.tensor([2.0, 1.0, 100.0]))
    assert torch.allclose(returns, adv)


def test_ppo_ratio_clamps_log_ratio_without_mutating_logprob() -> None:
    """
    极端 logprob 差值只应在 exp 前裁剪 log_ratio，不能直接裁剪 current_logprob。
    """

    current = torch.tensor([-100.0, 5.0, -2.0])
    old = torch.tensor([0.0, -100.0, -2.5])

    log_ratio, safe_log_ratio, ratio = PPOAgent.compute_stable_log_ratio_and_ratio(
        current_logprob=current,
        old_logprob=old,
        clamp_abs=20.0,
    )

    assert torch.allclose(log_ratio, torch.tensor([-100.0, 105.0, 0.5]))
    assert torch.allclose(safe_log_ratio, torch.tensor([-20.0, 20.0, 0.5]))
    assert torch.isfinite(ratio).all()
    assert torch.allclose(ratio[-1], torch.exp(torch.tensor(0.5)))


def test_value_clipping_uses_larger_clipped_loss_and_stays_finite() -> None:
    """
    当 Critic 预测跨越旧 value 太远时，value clipping 应取 unclipped/clipped 中较大者。
    """

    state_values = torch.tensor([10.0, 1.1])
    returns = torch.tensor([0.0, 1.0])
    old_values = torch.tensor([0.0, 1.0])

    clipped_loss = PPOAgent.compute_value_loss(
        state_values=state_values,
        returns=returns,
        old_values=old_values,
        clip_range=0.2,
    )
    mse_loss = PPOAgent.compute_value_loss(
        state_values=state_values,
        returns=returns,
        old_values=None,
        clip_range=0.2,
    )

    assert torch.isfinite(clipped_loss)
    assert torch.isfinite(mse_loss)
    assert clipped_loss >= mse_loss
    assert torch.allclose(mse_loss, torch.tensor(((10.0 - 0.0) ** 2 + (1.1 - 1.0) ** 2) / 2))


def test_train_import_has_no_debug_stdout() -> None:
    """import train 不应产生调试刷屏输出，避免污染测试和冒烟日志。"""

    result = subprocess.run(
        [sys.executable, "-c", "import train"],
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.strip() == ""


def test_legacy_profiler_initializes_sps_before_progress_display() -> None:
    """legacy 进度条读取 SPS 前必须已有初始化，避免首轮 UnboundLocalError。"""
    source = (PROJECT_ROOT / "archive" / "legacy_train.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    train_fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    assignments = [
        node.lineno
        for node in ast.walk(train_fn)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == "steps_per_sec"
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
        )
    ]
    postfix_reads = [
        node.lineno
        for node in ast.walk(train_fn)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == "steps_per_sec"
    ]
    assert assignments
    assert postfix_reads
    assert min(assignments) < min(postfix_reads)


def test_train_resolve_workspace_path_handles_relative_and_absolute() -> None:
    relative = resolve_workspace_path(Path("data") / "283.csv")
    absolute = resolve_workspace_path(relative)

    assert relative == TRAIN_PROJECT_ROOT / "data" / "283.csv"
    assert absolute == relative


def test_gpu_batch_rebuild_none_falls_back_to_cpu_without_y_task_crash(monkeypatch) -> None:
    """GPU fast rebuild 返回 None 时，PPO update 应自动降级，不能在 batch.y_task 处崩溃。"""

    seed_everything(123)
    overrides = {
        "n_w": 40,
        "n_m": 5,
        "hidden_dim": 32,
        "num_gat_layers": 1,
        "num_heads": 2,
        "batch_size": 2,
        "accumulation_steps": 1,
        "k_epochs": 1,
        "use_schedule_free": False,
        "use_ema": False,
        "use_shared_trunk": False,
        "use_gradient_checkpointing": False,
        "enable_gpu_batch_rebuild": True,
        "enable_dynamic_events": False,
        "enable_station_breakdown": False,
        "enable_material_delay": False,
        "randomize_durations": False,
    }

    with temporary_config(configs, overrides):
        device = torch.device("cpu")
        env = AirLineEnv_Graph(data_path_or_dir=str(PROJECT_ROOT / "data" / "283.csv"), seed=123)
        model = HBGATPN(configs).to(device)
        agent = PPOAgent(
            model=model,
            lr=configs.lr,
            gamma=configs.gamma,
            k_epochs=configs.k_epochs,
            eps_clip=configs.eps_clip,
            device=device,
            batch_size=configs.batch_size,
            total_timesteps=2,
        )
        monkeypatch.setattr(agent.gpu_graph_manager, "batched_rebuild_on_gpu", lambda snapshots, env_obj: None)

        memory = Memory()
        state = env.reset(randomize_duration=False, randomize_workers=False, seed=123)
        for _ in range(2):
            masks = env.get_masks()
            if masks[0].all():
                assert env.try_wait_for_resources()
                state = refresh_env_observation(env)
                masks = env.get_masks()
            action, logprob, value, _, is_invalid = agent.select_action(
                state.to(device),
                mask_task=masks[0].to(device),
                mask_station_matrix=masks[1].to(device),
                mask_worker=masks[2].to(device),
                deterministic=True,
            )
            assert action is not None
            assert not is_invalid
            memory.states.append(env.get_state_snapshot())
            memory.actions.append(action)
            memory.logprobs.append(logprob)
            memory.values.append(value)
            memory.masks.append(masks)
            state, reward, done, _ = env.step(action)
            memory.rewards.append(float(reward))
            memory.is_terminals.append(bool(done))
            if done:
                break

        metrics = agent.update(memory, env, current_ep=1)

        assert "Loss/Total" in metrics
        assert metrics["PPO/GPURebuildFallbackCount"] >= 1
        assert torch.isfinite(torch.tensor(metrics["Loss/Total"]))


def test_gpu_batch_rebuild_missing_node_batch_vectors_are_repaired(monkeypatch) -> None:
    """GPU rebuild 返回缺少 node.batch 的 Batch 时，PPO update 应自动补齐。"""

    seed_everything(123)
    overrides = {
        "n_w": 40,
        "n_m": 5,
        "hidden_dim": 32,
        "num_gat_layers": 1,
        "num_heads": 2,
        "batch_size": 2,
        "accumulation_steps": 1,
        "k_epochs": 1,
        "use_schedule_free": False,
        "use_ema": False,
        "use_shared_trunk": False,
        "use_gradient_checkpointing": False,
        "enable_gpu_batch_rebuild": True,
        "enable_dynamic_events": False,
        "enable_station_breakdown": False,
        "enable_material_delay": False,
        "randomize_durations": False,
    }

    with temporary_config(configs, overrides):
        device = torch.device("cpu")
        env = AirLineEnv_Graph(data_path_or_dir=str(PROJECT_ROOT / "data" / "283.csv"), seed=123)
        model = HBGATPN(configs).to(device)
        agent = PPOAgent(
            model=model,
            lr=configs.lr,
            gamma=configs.gamma,
            k_epochs=configs.k_epochs,
            eps_clip=configs.eps_clip,
            device=device,
            batch_size=configs.batch_size,
            total_timesteps=2,
        )

        def rebuild_without_node_batch(snapshots, env_obj):
            batch = Batch.from_data_list([env_obj.rebuild_state_from_snapshot(snap) for snap in snapshots]).to(device)
            for node_type in ("task", "station", "worker"):
                if "batch" in batch[node_type]:
                    del batch[node_type]["batch"]
            return batch

        monkeypatch.setattr(agent.gpu_graph_manager, "batched_rebuild_on_gpu", rebuild_without_node_batch)

        memory = Memory()
        state = env.reset(randomize_duration=False, randomize_workers=False, seed=123)
        for _ in range(2):
            masks = env.get_masks()
            if masks[0].all():
                assert env.try_wait_for_resources()
                state = refresh_env_observation(env)
                masks = env.get_masks()
            action, logprob, value, _, is_invalid = agent.select_action(
                state.to(device),
                mask_task=masks[0].to(device),
                mask_station_matrix=masks[1].to(device),
                mask_worker=masks[2].to(device),
                deterministic=True,
            )
            assert action is not None
            assert not is_invalid
            memory.states.append(env.get_state_snapshot())
            memory.actions.append(action)
            memory.logprobs.append(logprob)
            memory.values.append(value)
            memory.masks.append(masks)
            state, reward, done, _ = env.step(action)
            memory.rewards.append(float(reward))
            memory.is_terminals.append(bool(done))
            if done:
                break

        metrics = agent.update(memory, env, current_ep=1)

        assert "Loss/Total" in metrics
        assert metrics["PPO/GPURebuildFallbackCount"] == 0
        assert metrics["PPO/BatchVectorRepairCount"] >= 3
        assert torch.isfinite(torch.tensor(metrics["Loss/Total"]))
