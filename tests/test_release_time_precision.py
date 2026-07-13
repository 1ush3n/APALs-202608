from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from configs import configs
from core.action_masker import ActionMasker
from core.time_comparison import (
    release_time_tolerance,
    time_reached_numpy,
    time_reached_scalar,
    time_reached_tensor,
)
from environment import AirLineEnv_Graph
from tests.runtime_safety import temporary_config


DIAGNOSTIC_CURRENT_TIME = 1591.0752541256609
DIAGNOSTIC_RELEASE_TIME = 1591.0753418554243


class _ReleaseOnlyEnv:
    def __init__(self, device: torch.device) -> None:
        self.num_tasks = 1
        self.num_stations = 1
        self.num_workers = 1
        self.current_time = DIAGNOSTIC_CURRENT_TIME
        self.mean_task_time = 1.0
        self.worker_free_time = np.zeros(1, dtype=np.float64)
        self.worker_locks = np.zeros(1, dtype=np.int64)
        self.worker_skill_matrix = torch.ones((1, 1), dtype=torch.float32, device=device)
        self.task_status = np.ones(1, dtype=np.int64)
        self.task_material_ready = np.asarray([DIAGNOSTIC_RELEASE_TIME], dtype=np.float64)


def _first_valid_action(env: AirLineEnv_Graph) -> tuple[int, int, list[int]]:
    task_mask, station_mask, worker_mask = env.get_masks()
    for task_tensor in torch.where(~task_mask)[0]:
        task_id = int(task_tensor.item())
        for station_tensor in torch.where(~station_mask[task_id])[0]:
            station_id = int(station_tensor.item())
            skill_id = int(env.task_static_feat[task_id, 1].item())
            demand = max(1, int(env.task_static_feat[task_id, 2].item()))
            workers = [
                int(worker_id)
                for worker_id in np.where(env.worker_skill_matrix[:, skill_id].cpu().numpy() > 0.5)[0]
                if not bool(worker_mask[worker_id])
                and int(env.worker_locks[worker_id]) in {0, station_id + 1}
            ]
            if len(workers) >= demand:
                return task_id, station_id, workers[:demand]
    raise AssertionError("测试环境没有可执行动作")


@pytest.mark.parametrize(
    ("gap", "expected"),
    [(0.5e-5, True), (8.772976e-5, False)],
)
def test_release_time_helpers_use_identical_float64_boundary(gap: float, expected: bool) -> None:
    tolerance = 1.0e-5
    release = DIAGNOSTIC_CURRENT_TIME + gap
    assert time_reached_scalar(release, DIAGNOSTIC_CURRENT_TIME, tolerance) is expected
    assert bool(time_reached_numpy(np.asarray([release]), DIAGNOSTIC_CURRENT_TIME, tolerance)[0]) is expected
    tensor_result = time_reached_tensor(
        np.asarray([release]),
        DIAGNOSTIC_CURRENT_TIME,
        tolerance,
        device=torch.device("cpu"),
    )
    assert tensor_result.dtype == torch.bool
    assert bool(tensor_result[0].item()) is expected


@pytest.mark.parametrize(
    "device",
    [torch.device("cpu")]
    + ([torch.device("cuda")] if torch.cuda.is_available() else []),
)
def test_large_time_release_is_masked_by_all_paths_and_shadow(device: torch.device) -> None:
    env = _ReleaseOnlyEnv(device)
    masker = ActionMasker(env)
    queue_ok = np.ones(1, dtype=bool)
    overrides = {
        "enable_gpu_tensor_masking": True,
        "enable_shadow_mask_verification": True,
        "release_time_tolerance_hours": 1.0e-5,
    }
    with temporary_config(configs, overrides):
        tensorized = masker._get_masks_tensorized(queue_ok)
        vectorized = masker._get_masks_vectorized(queue_ok)
        legacy = masker._get_masks_legacy(queue_ok)
        shadow_checked = masker.get_masks()

    for result in (tensorized, vectorized, legacy, shadow_checked):
        assert bool(result[0][0].item()) is True


def test_environment_release_boundary_matches_action_mask() -> None:
    overrides = {
        "enable_reschedule_mode": False,
        "enable_dynamic_events": False,
        "enable_gpu_tensor_masking": True,
        "enable_shadow_mask_verification": False,
        "randomize_durations": False,
        "release_time_tolerance_hours": 1.0e-5,
    }
    with temporary_config(configs, overrides):
        blocked_env = AirLineEnv_Graph(data_path_or_dir="data/283.csv", seed=42)
        blocked_env.reset(randomize_duration=False, randomize_workers=False, seed=42)
        blocked_action = _first_valid_action(blocked_env)
        blocked_env.current_time = DIAGNOSTIC_CURRENT_TIME
        blocked_env.task_material_ready[blocked_action[0]] = DIAGNOSTIC_RELEASE_TIME
        task_mask, _, _ = blocked_env.get_masks()
        assert bool(task_mask[blocked_action[0]]) is True
        _state, _reward, _done, info = blocked_env.step(blocked_action)
        assert info["error"] == "task_release_time_not_reached"

        allowed_env = AirLineEnv_Graph(data_path_or_dir="data/283.csv", seed=42)
        allowed_env.reset(randomize_duration=False, randomize_workers=False, seed=42)
        allowed_action = _first_valid_action(allowed_env)
        allowed_env.current_time = DIAGNOSTIC_CURRENT_TIME
        allowed_env.task_material_ready[allowed_action[0]] = (
            DIAGNOSTIC_CURRENT_TIME + 0.5 * release_time_tolerance(configs)
        )
        task_mask, _, _ = allowed_env.get_masks()
        assert bool(task_mask[allowed_action[0]]) is False
        _state, _reward, _done, info = allowed_env.step(allowed_action)
        assert "error" not in info


def test_release_time_config_validation_rejects_invalid_values() -> None:
    from runtime.configuration import validate_runtime_config

    bad_tolerance = SimpleNamespace(**configs.to_flat_dict())
    bad_tolerance.release_time_tolerance_hours = -1.0
    with pytest.raises(ValueError, match="release_time_tolerance_hours"):
        validate_runtime_config(bad_tolerance)

    bad_policy = SimpleNamespace(**configs.to_flat_dict())
    bad_policy.eval_mask_mismatch_policy = "ignore"
    with pytest.raises(ValueError, match="eval_mask_mismatch_policy"):
        validate_runtime_config(bad_policy)
