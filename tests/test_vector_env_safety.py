# -*- coding: utf-8 -*-
"""
VectorEnv 低负载生命周期验证。

目标是在 Windows/Linux 都能用相同测试发现多进程 reset、step、snapshot rebuild 和 close
中的资源泄漏问题。测试固定 2 个子环境，避免本机内存压力。
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import configs
from tests.runtime_safety import seed_everything, temporary_config
from utils.vector_env import EnvCreator, VectorEnv, VectorEnvWorkerError
from worker_feature_layout import resolve_worker_feature_layout


pytestmark = pytest.mark.vector_safe


class _FailingEnvCreator:
    def __call__(self, index: int):
        raise RuntimeError(f"worker-{index}-init-failure")


class _ResetFailureEnvCreator:
    def __call__(self, index: int) -> object:
        env = EnvCreator(
            str(PROJECT_ROOT / "data" / "283.csv"),
            seed_offset=1000,
        )(index)
        if index == 0:
            def fail_reset(**_kwargs: object) -> None:
                raise RuntimeError("worker-0-reset-failure")

            env.reset = fail_reset
        return env



def _pick_simple_actions(obs_list, masks_list):
    worker_layout = resolve_worker_feature_layout(configs)
    task_skill_end = 5 + worker_layout.num_skill_types
    actions = []
    for obs, masks in zip(obs_list, masks_list):
        task_mask, station_mask, worker_mask = masks
        valid_tasks = torch.where(~task_mask)[0]
        if len(valid_tasks) == 0:
            actions.append(None)
            continue

        task_id = int(valid_tasks[0].item())
        valid_stations = torch.where(~station_mask[task_id])[0]
        if len(valid_stations) == 0:
            actions.append(None)
            continue

        station_id = int(valid_stations[0].item())
        demand = max(1, int(obs["task"].x[task_id, -1].item()))
        task_type_idx = int(torch.argmax(obs["task"].x[task_id, 5:task_skill_end]).item())

        worker_feats = obs["worker"].x
        has_skill = worker_feats[:, 1 + task_type_idx] > 0.5
        worker_locks = torch.argmax(worker_feats[:, worker_layout.lock_slice], dim=1)
        lock_mask = (worker_locks != 0) & (worker_locks != station_id + 1)
        combined_mask = worker_mask | (~has_skill) | lock_mask
        valid_workers = torch.where(~combined_mask)[0]

        if len(valid_workers) < demand:
            actions.append(None)
        else:
            actions.append((task_id, station_id, valid_workers[:demand].tolist()))
    return actions


def test_vector_env_two_process_lifecycle_low_memory() -> None:
    seed_everything(42)
    overrides = {
        "n_w": 40,
        "n_m": 5,
        "max_slots_per_station": 3,
        "randomize_durations": False,
        "enable_dynamic_events": False,
        "enable_station_breakdown": False,
        "enable_material_delay": False,
    }

    before_children = len(mp.active_children())
    vec_env = None
    with temporary_config(configs, overrides):
        make_env = EnvCreator(str(PROJECT_ROOT / "data" / "283.csv"), seed_offset=200)
        vec_env = VectorEnv(make_env, num_envs=2)
        try:
            obs_list = vec_env.reset_all(randomize_duration=False, randomize_workers=False)
            assert len(obs_list) == 2
            assert all(env.num_tasks is not None for env in vec_env.envs)

            masks_list = vec_env.get_masks_all()
            actions = _pick_simple_actions(obs_list, masks_list)
            next_states, rewards, dones, infos = vec_env.step_all(actions)

            assert len(next_states) == 2
            assert len(rewards) == 2
            assert len(dones) == 2
            assert all(isinstance(info, dict) for info in infos)

            snapshots = vec_env.get_state_snapshot_all()
            assert len(snapshots) == 2
            for proxy, snapshot in zip(vec_env.envs, snapshots):
                rebuilt = proxy.rebuild_state_from_snapshot(snapshot)
                assert rebuilt["task"].x.shape[0] == proxy.num_tasks
                assert rebuilt["station"].x.shape[0] == configs.n_m
                assert rebuilt["worker"].x.shape[0] == len(snapshot["worker_free_time"])
        finally:
            vec_env.close()


def test_vector_env_initialization_failure_is_reported_without_hanging() -> None:
    before_children = len(mp.active_children())
    with pytest.raises(RuntimeError, match="worker-0-init-failure"):
        VectorEnv(
            _FailingEnvCreator(),
            num_envs=1,
            start_method="spawn",
            worker_threads=1,
            init_timeout_sec=15.0,
            command_timeout_sec=15.0,
        )

    after_children = len(mp.active_children())
    assert after_children <= before_children, (
        f"VectorEnv close 后仍有多余子进程: before={before_children}, after={after_children}"
    )


def test_vector_env_worker_failure_closes_and_forbids_reuse() -> None:
    vector_env = VectorEnv(
        _ResetFailureEnvCreator(),
        num_envs=2,
        start_method="spawn",
        worker_threads=1,
        init_timeout_sec=30.0,
        command_timeout_sec=30.0,
    )
    try:
        with pytest.raises(
            VectorEnvWorkerError,
            match="worker-0-reset-failure",
        ):
            vector_env.reset_all()

        assert vector_env.closed
        assert all(not process.is_alive() for process in vector_env.processes)
        with pytest.raises(
            VectorEnvWorkerError,
            match="不可复用",
        ):
            vector_env.reset_all()
    finally:
        vector_env.close()


def _assert_masks_equal(left, right) -> None:
    for l_mask, r_mask in zip(left, right):
        assert torch.equal(l_mask, r_mask)


def _assert_snapshot_core_equal(left: dict, right: dict) -> None:
    for key in (
        "task_status",
        "worker_free_time",
        "worker_locks",
        "station_loads",
        "station_wall_clock",
    ):
        np.testing.assert_allclose(left[key], right[key])
    assert left["current_time"] == right["current_time"]
    assert left["assigned_tasks"] == right["assigned_tasks"]
    assert left["dataset_idx"] == right["dataset_idx"]


def test_vector_env_fast_rollout_state_matches_legacy_snapshot() -> None:
    seed_everything(43)
    overrides = {
        "n_w": 40,
        "n_m": 5,
        "max_slots_per_station": 3,
        "randomize_durations": False,
        "enable_dynamic_events": False,
        "enable_station_breakdown": False,
        "enable_material_delay": False,
    }

    vec_env = None
    with temporary_config(configs, overrides):
        make_env = EnvCreator(str(PROJECT_ROOT / "data" / "283.csv"), seed_offset=300)
        vec_env = VectorEnv(make_env, num_envs=2)
        try:
            vec_env.reset_all(randomize_duration=False, randomize_workers=False)

            legacy_masks = vec_env.get_masks_all()
            legacy_snapshots = vec_env.get_state_snapshot_all()
            fast_masks, fast_snapshots = vec_env.get_masks_and_snapshots_all()

            for old_masks, new_masks in zip(legacy_masks, fast_masks):
                _assert_masks_equal(old_masks, new_masks)
            for old_snapshot, new_snapshot in zip(legacy_snapshots, fast_snapshots):
                _assert_snapshot_core_equal(old_snapshot, new_snapshot)

            for proxy, snapshot in zip(vec_env.envs, fast_snapshots):
                rebuilt = proxy.rebuild_state_from_snapshot(snapshot)
                assert rebuilt["task"].x.shape[0] == proxy.num_tasks
                assert rebuilt["station"].x.shape[0] == configs.n_m
                assert rebuilt["worker"].x.shape[0] == len(snapshot["worker_free_time"])
        finally:
            if vec_env is not None:
                vec_env.close()


def test_vector_env_step_snapshot_rebuild_shapes_low_memory() -> None:
    seed_everything(44)
    overrides = {
        "n_w": 40,
        "n_m": 5,
        "max_slots_per_station": 3,
        "randomize_durations": False,
        "enable_dynamic_events": False,
        "enable_station_breakdown": False,
        "enable_material_delay": False,
    }

    vec_env = None
    with temporary_config(configs, overrides):
        make_env = EnvCreator(str(PROJECT_ROOT / "data" / "283.csv"), seed_offset=400)
        vec_env = VectorEnv(make_env, num_envs=2)
        try:
            obs_list = vec_env.reset_all(randomize_duration=False, randomize_workers=False)
            masks_list, start_snapshots = vec_env.get_masks_and_snapshots_all()
            actions = _pick_simple_actions(obs_list, masks_list)

            next_snapshots, rewards, dones, infos = vec_env.step_snapshot_all(actions)

            assert len(next_snapshots) == 2
            assert len(rewards) == 2
            assert len(dones) == 2
            assert all(isinstance(info, dict) for info in infos)

            for proxy, start_snapshot, next_snapshot in zip(vec_env.envs, start_snapshots, next_snapshots):
                rebuilt = proxy.rebuild_state_from_snapshot(next_snapshot)
                assert rebuilt["task"].x.shape[0] == proxy.num_tasks
                assert rebuilt["station"].x.shape[0] == configs.n_m
                assert rebuilt["worker"].x.shape[0] == len(next_snapshot["worker_free_time"])
                assert next_snapshot["current_time"] >= start_snapshot["current_time"]
        finally:
            if vec_env is not None:
                vec_env.close()


def test_vector_env_fused_rollout_matches_legacy_state_queries() -> None:
    """融合 IPC 只能减少往返次数，不得改变 reset/step 后的环境状态。"""
    seed_everything(46)
    overrides = {
        "n_w": 40,
        "n_m": 5,
        "max_slots_per_station": 3,
        "randomize_durations": False,
        "enable_dynamic_events": False,
        "enable_station_breakdown": False,
        "enable_material_delay": False,
    }

    vec_env = None
    with temporary_config(configs, overrides):
        make_env = EnvCreator(str(PROJECT_ROOT / "data" / "283.csv"), seed_offset=450)
        vec_env = VectorEnv(make_env, num_envs=2)
        try:
            fused_masks, fused_snapshots = vec_env.reset_rollout_all(
                randomize_duration=False,
                randomize_workers=False,
            )
            legacy_masks, legacy_snapshots = vec_env.get_masks_and_snapshots_all()
            for fused, legacy in zip(fused_masks, legacy_masks):
                _assert_masks_equal(fused, legacy)
            for fused, legacy in zip(fused_snapshots, legacy_snapshots):
                _assert_snapshot_core_equal(fused, legacy)

            observations = [
                proxy.rebuild_state_from_snapshot(snapshot)
                for proxy, snapshot in zip(vec_env.envs, fused_snapshots)
            ]
            actions = _pick_simple_actions(observations, fused_masks)
            action_map = {
                index: action
                for index, action in enumerate(actions)
                if action is not None
            }
            fused_steps = vec_env.step_rollout_indices(action_map)
            queried_states = vec_env.get_rollout_state_indices(list(action_map))

            assert set(fused_steps) == set(action_map)
            for index in action_map:
                next_masks, next_snapshot, reward, done, info = fused_steps[index]
                queried_masks, queried_snapshot = queried_states[index]
                _assert_masks_equal(next_masks, queried_masks)
                _assert_snapshot_core_equal(next_snapshot, queried_snapshot)
                assert isinstance(reward, float)
                assert isinstance(done, bool)
                assert isinstance(info, dict)
        finally:
            if vec_env is not None:
                vec_env.close()


def test_vector_env_snapshot_rebuild_preserves_per_worker_duration_noise() -> None:
    """代理端重建必须使用各子进程快照中的扰动工时，而非共享静态上下文。"""
    seed_everything(461)
    overrides = {
        "n_w": 40,
        "n_m": 5,
        "randomize_durations": True,
        "enable_dynamic_events": False,
        "enable_station_breakdown": False,
        "enable_material_delay": False,
    }

    vec_env = None
    with temporary_config(configs, overrides):
        make_env = EnvCreator(str(PROJECT_ROOT / "data" / "283.csv"), seed_offset=461)
        vec_env = VectorEnv(make_env, num_envs=2)
        try:
            _masks, snapshots = vec_env.reset_rollout_all(
                randomize_duration=True,
                randomize_workers=False,
            )
            assert all("base_task_x" in snapshot for snapshot in snapshots)
            assert not np.array_equal(
                np.asarray(snapshots[0]["base_task_x"])[:, 0],
                np.asarray(snapshots[1]["base_task_x"])[:, 0],
            )
            for proxy, snapshot in zip(vec_env.envs, snapshots, strict=True):
                rebuilt = proxy.rebuild_state_from_snapshot(snapshot)
                np.testing.assert_allclose(
                    rebuilt["task"].x[:, 0].detach().cpu().numpy(),
                    np.asarray(snapshot["base_task_x"])[:, 0],
                )
        finally:
            if vec_env is not None:
                vec_env.close()


def test_vector_env_fused_wait_preserves_event_driven_queue_semantics() -> None:
    """融合等待必须先推进到未来事件，不能把暂时无资源误报为死锁。"""
    seed_everything(47)
    overrides = {
        "n_w": 40,
        "n_m": 5,
        "max_slots_per_station": 3,
        "randomize_durations": False,
        "enable_dynamic_events": True,
        "enable_station_breakdown": False,
        "enable_material_delay": True,
        "prob_material_delay_base": 1.0,
        "material_delay_min": 5.0,
        "material_delay_max": 5.0,
    }

    vec_env = None
    with temporary_config(configs, overrides):
        make_env = EnvCreator(str(PROJECT_ROOT / "data" / "283.csv"), seed_offset=470)
        vec_env = VectorEnv(make_env, num_envs=1)
        try:
            initial_masks, initial_snapshots = vec_env.reset_rollout_all(
                randomize_duration=False,
                randomize_workers=False,
            )
            assert initial_masks[0][0].all()
            assert initial_snapshots[0]["current_time"] == 0.0

            waited, waited_masks, waited_snapshot = vec_env.wait_rollout_indices([0])[0]
            queried_masks, queried_snapshot = vec_env.get_rollout_state_indices([0])[0]

            assert waited
            assert waited_snapshot["current_time"] == 5.0
            assert not waited_masks[0].all()
            _assert_masks_equal(waited_masks, queried_masks)
            _assert_snapshot_core_equal(waited_snapshot, queried_snapshot)
        finally:
            if vec_env is not None:
                vec_env.close()


def test_vector_env_indexed_reset_step_and_switch() -> None:
    seed_everything(45)
    overrides = {
        "n_w": 40,
        "n_m": 5,
        "randomize_durations": False,
        "enable_dynamic_events": False,
        "enable_station_breakdown": False,
        "enable_material_delay": False,
    }
    vec_env = None
    with temporary_config(configs, overrides):
        make_env = EnvCreator(str(PROJECT_ROOT / "data" / "283.csv"), seed_offset=500)
        vec_env = VectorEnv(make_env, num_envs=2)
        try:
            vec_env.switch_dataset_indices({0: 0, 1: 0})
            observations = vec_env.reset_indices(
                {
                    0: {"randomize_duration": False, "randomize_workers": False, "seed": 1001},
                    1: {"randomize_duration": False, "randomize_workers": False, "seed": 1002},
                }
            )
            rollout_states = vec_env.get_rollout_state_indices([0, 1])
            actions = _pick_simple_actions(
                [observations[0], observations[1]],
                [rollout_states[0][0], rollout_states[1][0]],
            )
            action_map = {index: action for index, action in enumerate(actions) if action is not None}
            results = vec_env.step_snapshot_indices(action_map)

            assert set(results) == set(action_map)
            for index, (snapshot, reward, done, info) in results.items():
                assert snapshot["dataset_idx"] == 0
                assert isinstance(reward, float)
                assert isinstance(done, bool)
                assert isinstance(info, dict)
                rebuilt = vec_env.envs[index].rebuild_state_from_snapshot(snapshot)
                assert rebuilt["task"].x.shape[0] == vec_env.envs[index].num_tasks
        finally:
            if vec_env is not None:
                vec_env.close()


def test_snapshot_rebuild_loads_dataset_context_locally() -> None:
    """重建图时不得通过 Pipe 接收包含 Tensor 的数据集上下文。"""
    seed_everything(45)
    overrides = {
        "n_w": 40,
        "n_m": 5,
        "max_slots_per_station": 3,
        "randomize_durations": False,
        "enable_dynamic_events": False,
        "enable_station_breakdown": False,
        "enable_material_delay": False,
    }

    vec_env = None
    with temporary_config(configs, overrides):
        make_env = EnvCreator(str(PROJECT_ROOT / "data" / "283.csv"), seed_offset=425)
        vec_env = VectorEnv(make_env, num_envs=1)
        try:
            _, snapshots = vec_env.reset_rollout_all(
                randomize_duration=False, randomize_workers=False
            )
            proxy = vec_env.envs[0]

            original_recv = proxy._recv

            def reject_dataset_context_ipc(operation: str):
                if operation == "initialize_dataset_context":
                    raise AssertionError("数据集上下文不得通过 Pipe 返回 Tensor")
                return original_recv(operation)

            proxy._recv = reject_dataset_context_ipc
            rebuilt = proxy.rebuild_state_from_snapshot(snapshots[0])
            assert rebuilt["task"].x.shape[0] == proxy.num_tasks
        finally:
            if vec_env is not None:
                vec_env.close()
