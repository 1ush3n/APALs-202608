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
from utils.vector_env import EnvCreator, VectorEnv


pytestmark = pytest.mark.vector_safe


def test_vector_env_worker_thread_auto_resolution() -> None:
    assert VectorEnv._resolve_worker_threads("auto", num_envs=2) >= 1
    assert VectorEnv._resolve_worker_threads(3, num_envs=2) == 3
    assert VectorEnv._resolve_worker_threads("bad", num_envs=2) >= 1


def _pick_simple_actions(obs_list, masks_list):
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
        task_type_idx = int(torch.argmax(obs["task"].x[task_id, 5:15]).item())

        worker_feats = obs["worker"].x
        has_skill = worker_feats[:, 1 + task_type_idx] > 0.5
        worker_locks = torch.argmax(worker_feats[:, 13:21], dim=1)
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
        vec_env = VectorEnv(make_env, num_envs=2, worker_threads=1)
        try:
            assert vec_env.worker_threads == 1
            assert len(vec_env.worker_audits) == 2
            assert all(str(audit.get("omp_num_threads")) == "1" for audit in vec_env.worker_audits)
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

    after_children = len(mp.active_children())
    assert after_children <= before_children, (
        f"VectorEnv close 后仍有多余子进程: before={before_children}, after={after_children}"
    )


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
