from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch_geometric.data import Batch, HeteroData
from torch_geometric.utils import to_dense_batch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ppo_agent import PPOAgent
from configs import configs
from worker_feature_layout import resolve_worker_feature_layout


def _graph(task_skills: list[int], workers: list[tuple[list[int], int]]) -> HeteroData:
    worker_layout = resolve_worker_feature_layout(configs)
    data = HeteroData()
    task_x = torch.zeros((len(task_skills), configs.task_feat_dim), dtype=torch.float32)
    for idx, skill_id in enumerate(task_skills):
        task_x[idx, 5 + skill_id] = 1.0
    worker_x = torch.zeros((len(workers), worker_layout.total_dim), dtype=torch.float32)
    for idx, (skill_ids, locked_station_plus_one) in enumerate(workers):
        for skill_id in skill_ids:
            worker_x[idx, 1 + skill_id] = 1.0
        worker_x[idx, worker_layout.lock_start + locked_station_plus_one] = 1.0
    data["task"].x = task_x
    data["worker"].x = worker_x
    data["station"].x = torch.zeros((2, configs.station_feat_dim), dtype=torch.float32)
    return data


def _old_dense_static_mask(batch: Batch, y_task: torch.Tensor, y_station: torch.Tensor) -> torch.Tensor:
    worker_layout = resolve_worker_feature_layout(configs)
    task_skill_end = 5 + worker_layout.num_skill_types
    task_raw, _ = to_dense_batch(batch["task"].x, batch["task"].batch)
    batch_indices = torch.arange(y_task.size(0))
    sel_task_raw = task_raw[batch_indices, y_task]
    task_type_idx = torch.argmax(sel_task_raw[:, 5:task_skill_end], dim=1)

    worker_raw, _ = to_dense_batch(batch["worker"].x, batch["worker"].batch)
    worker_skills = worker_raw[:, :, worker_layout.skill_slice]

    batch_size, max_workers = worker_skills.shape[0], worker_skills.shape[1]
    b_indices = torch.arange(batch_size).view(-1, 1).expand(-1, max_workers).reshape(-1)
    w_indices = torch.arange(max_workers).view(1, -1).expand(batch_size, -1).reshape(-1)
    t_indices = task_type_idx.view(-1, 1).expand(-1, max_workers).reshape(-1)
    has_skill_flat = worker_skills[b_indices, w_indices, t_indices] > 0.5
    skill_mask = (~has_skill_flat).view(batch_size, max_workers)

    station_action = y_station + 1
    worker_locks = torch.argmax(worker_raw[:, :, worker_layout.lock_slice], dim=2)
    station_action_expanded = station_action.view(batch_size, 1).expand(batch_size, max_workers)
    lock_mask = (worker_locks != 0) & (worker_locks != station_action_expanded)
    return skill_mask | lock_mask


def test_sparse_static_worker_mask_matches_old_dense_formula() -> None:
    batch = Batch.from_data_list(
        [
            _graph(
                task_skills=[0, 2],
                workers=[
                    ([2], 0),
                    ([0], 0),
                ],
            ),
            _graph(
                task_skills=[0],
                workers=[
                    ([0], 1),
                    ([0], 2),
                    ([], 0),
                ],
            ),
        ]
    )
    batch.y_task = torch.tensor([1, 0], dtype=torch.long)
    batch.y_station = torch.tensor([1, 0], dtype=torch.long)

    agent = object.__new__(PPOAgent)
    agent.device = torch.device("cpu")
    agent.config = configs

    expected = _old_dense_static_mask(batch, batch.y_task, batch.y_station)
    actual = agent.compute_static_worker_constraint_mask(
        batch,
        selected_task=batch.y_task,
        selected_station=batch.y_station,
        max_workers=expected.size(1),
    )

    assert torch.equal(actual, expected)
