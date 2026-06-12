# -*- coding: utf-8 -*-
"""验证初始 APAL 调度模式不会注入动态扰动事件。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import Config, configs, load_config_files
from core.event_engine import EventType
from environment import AirLineEnv_Graph
from tests.runtime_safety import temporary_config


def _initial_schedule_overrides() -> dict[str, object]:
    cfg = Config()
    load_config_files([str(PROJECT_ROOT / "conf" / "experiment" / "initial_schedule.yaml")], target=cfg)
    return cfg.to_flat_dict()


def _event_types(env: AirLineEnv_Graph) -> list[EventType]:
    return [item[3].type for item in env.event_queue._queue]


def _first_valid_action(env: AirLineEnv_Graph, require_positive_duration: bool = False) -> tuple[int, int, list[int]]:
    task_mask, station_mask, worker_mask = env.get_masks()
    valid_tasks = torch.where(~task_mask)[0]
    assert len(valid_tasks) > 0

    for task_tensor in valid_tasks:
        task_id = int(task_tensor.item())
        if require_positive_duration and float(env.task_static_feat[task_id, 0].item()) <= 1e-6:
            continue
        valid_stations = torch.where(~station_mask[task_id])[0]
        if len(valid_stations) == 0:
            continue

        station_id = int(valid_stations[0].item())
        skill_id = int(env.task_static_feat[task_id, 1].item())
        demand = max(1, int(env.task_static_feat[task_id, 2].item()))
        candidate_workers = [
            int(w)
            for w in np.where(env.worker_skill_matrix[:, skill_id].numpy() > 0.5)[0]
            if not bool(worker_mask[w])
        ]
        if len(candidate_workers) >= demand:
            return task_id, station_id, candidate_workers[:demand]

    raise AssertionError("初始调度模式应至少存在一个可执行动作")


def test_initial_schedule_reset_does_not_inject_dynamic_disturbance_events() -> None:
    with temporary_config(configs, _initial_schedule_overrides()):
        env = AirLineEnv_Graph(data_path_or_dir=str(PROJECT_ROOT / "data" / "283.csv"), seed=42)
        env.reset(randomize_duration=True, randomize_workers=True, seed=42)

        forbidden = {
            EventType.WORKER_LEAVE,
            EventType.STATION_BREAKDOWN,
            EventType.STATION_RECOVER,
            EventType.MATERIAL_ARRIVE,
            EventType.DURATION_PERTURB,
        }

        assert configs.randomize_durations is True
        assert not (set(_event_types(env)) & forbidden)
        assert np.allclose(env.task_material_ready, 0.0)


def test_initial_schedule_still_allows_task_finish_events_after_normal_step() -> None:
    with temporary_config(configs, _initial_schedule_overrides()):
        env = AirLineEnv_Graph(data_path_or_dir=str(PROJECT_ROOT / "data" / "283.csv"), seed=42)
        env.reset(randomize_duration=True, randomize_workers=True, seed=42)

        for _ in range(5):
            action = _first_valid_action(env)
            env.step(action)
            if EventType.TASK_FINISH in _event_types(env):
                break

        event_types = _event_types(env)
        assert EventType.TASK_FINISH in event_types
        assert EventType.WORKER_LEAVE not in event_types
        assert EventType.STATION_BREAKDOWN not in event_types
        assert EventType.MATERIAL_ARRIVE not in event_types
