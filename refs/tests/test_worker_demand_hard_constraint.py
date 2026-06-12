from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from configs import configs
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent


def _find_positive_demand_action(env: AirLineEnv_Graph) -> Tuple[int, int, list[int], int]:
    task_mask, station_mask, worker_mask = env.get_masks()
    for task_id in torch.where(~task_mask)[0].tolist():
        demand = max(1, int(env.task_static_feat[task_id, 2].item()))
        duration = float(env.task_static_feat[task_id, 0].item())
        if demand <= 1 or duration <= 1e-5:
            continue
        valid_stations = torch.where(~station_mask[task_id])[0].tolist()
        if not valid_stations:
            continue
        station_id = int(valid_stations[0])
        skill_id = int(env.task_static_feat[task_id, 1].item())
        has_skill = env.worker_skill_matrix[:, skill_id] > 0.5
        lock_ok = torch.from_numpy((env.worker_locks == 0) | (env.worker_locks == station_id + 1))
        valid_workers = torch.where((~worker_mask.cpu()) & has_skill.cpu() & lock_ok)[0].tolist()
        if len(valid_workers) >= demand:
            return task_id, station_id, valid_workers, demand

    env.task_status.fill(0)
    if hasattr(env, "task_material_ready"):
        env.task_material_ready.fill(0.0)
    for task_id in range(env.num_tasks):
        demand = max(1, int(env.task_static_feat[task_id, 2].item()))
        duration = float(env.task_static_feat[task_id, 0].item())
        if demand <= 1 or duration <= 1e-5:
            continue
        fixed_station = int(env.fixed_stations[task_id])
        if fixed_station >= 0:
            station_candidates = [fixed_station]
        else:
            station_candidates = list(range(0, int(env.max_allowed_stations[task_id]) + 1))
        skill_id = int(env.task_static_feat[task_id, 1].item())
        has_skill = env.worker_skill_matrix[:, skill_id] > 0.5
        for station_id in station_candidates:
            lock_ok = torch.from_numpy((env.worker_locks == 0) | (env.worker_locks == station_id + 1))
            valid_workers = torch.where(has_skill.cpu() & lock_ok)[0].tolist()
            if len(valid_workers) >= demand:
                env.task_status[task_id] = 1
                task_mask, station_mask, worker_mask = env.get_masks()
                if not bool(task_mask[task_id]) and not bool(station_mask[task_id, station_id]):
                    return task_id, station_id, valid_workers, demand
    raise AssertionError("未找到可用于测试的多人需求工序")


def test_environment_rejects_insufficient_worker_team() -> None:
    env = AirLineEnv_Graph(data_path_or_dir=str(PROJECT_ROOT / "data" / "283.csv"), seed=7)
    env.reset(randomize_duration=False, randomize_workers=False, seed=7)
    task_id, station_id, workers, demand = _find_positive_demand_action(env)

    before_assigned = len(env.assigned_tasks)
    before_status = int(env.task_status[task_id])
    _, reward, done, info = env.step((task_id, station_id, workers[: demand - 1]))

    assert not done
    assert reward <= 0.0
    assert info["invalid_action"] is True
    assert info["error"] == "insufficient_workers"
    assert len(env.assigned_tasks) == before_assigned
    assert int(env.task_status[task_id]) == before_status


def test_agent_single_action_uses_hard_demand_feature_column() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = AirLineEnv_Graph(data_path_or_dir=str(PROJECT_ROOT / "data" / "283.csv"), seed=11)
    obs = env.reset(randomize_duration=False, randomize_workers=False, seed=11)
    task_id, station_id, _, demand = _find_positive_demand_action(env)

    task_mask, station_mask, worker_mask = env.get_masks()
    forced_task_mask = torch.ones_like(task_mask)
    forced_task_mask[task_id] = False
    forced_station_mask = torch.ones_like(station_mask)
    forced_station_mask[task_id, station_id] = False

    model = HBGATPN(configs).to(device)
    agent = PPOAgent(model, configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip, device, configs.batch_size)
    action, *_ = agent.select_action(
        obs.to(device),
        mask_task=forced_task_mask.to(device),
        mask_station_matrix=forced_station_mask.to(device),
        mask_worker=worker_mask.to(device),
        deterministic=True,
    )

    assert action[0] == task_id
    assert action[1] == station_id
    assert len(action[2]) >= demand


def test_agent_batch_action_uses_hard_demand_feature_column() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = AirLineEnv_Graph(data_path_or_dir=str(PROJECT_ROOT / "data" / "283.csv"), seed=13)
    obs = env.reset(randomize_duration=False, randomize_workers=False, seed=13)
    task_id, station_id, _, demand = _find_positive_demand_action(env)

    task_mask, station_mask, worker_mask = env.get_masks()
    forced_task_mask = torch.ones_like(task_mask)
    forced_task_mask[task_id] = False
    forced_station_mask = torch.ones_like(station_mask)
    forced_station_mask[task_id, station_id] = False

    model = HBGATPN(configs).to(device)
    agent = PPOAgent(model, configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip, device, configs.batch_size)
    result = agent.select_actions_batch(
        obs_list=[obs],
        mask_task_list=[forced_task_mask],
        mask_station_matrix_list=[forced_station_mask],
        mask_worker_list=[worker_mask],
        deterministic=True,
    )[0]

    action = result[0]
    assert action[0] == task_id
    assert action[1] == station_id
    assert len(action[2]) >= demand
