from __future__ import annotations

from collections.abc import Sequence
import numpy as np
import torch

from utils.reschedule import BaselineSchedule


def build_baseline_team_edge_index(
    baseline: BaselineSchedule | None,
    *,
    num_tasks: int,
) -> np.ndarray:
    """从已有 baseline team 生成 episode-static 的 task-worker 稀疏关系。"""
    if baseline is None:
        return np.empty((2, 0), dtype=np.int64)
    edges: list[tuple[int, int]] = []
    for task_id, task in baseline.tasks.items():
        if not 0 <= int(task_id) < int(num_tasks):
            raise ValueError(f"baseline task_id 越界: {task_id}")
        edges.extend((int(task_id), int(worker_id)) for worker_id in task.team)
    if not edges:
        return np.empty((2, 0), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64).T


def offset_baseline_team_edge_index(
    edge_index: np.ndarray | torch.Tensor,
    *,
    task_offset: int,
    worker_offset: int,
) -> np.ndarray | torch.Tensor:
    """为 Fast-Exact batch 添加 task/worker 全局偏移，不改变 relation metadata。"""
    if isinstance(edge_index, torch.Tensor):
        assert edge_index.ndim == 2 and edge_index.shape[0] == 2
        offsets = torch.tensor(
            [[int(task_offset)], [int(worker_offset)]],
            dtype=edge_index.dtype,
            device=edge_index.device,
        )
        return edge_index + offsets
    edge_array = np.asarray(edge_index, dtype=np.int64)
    assert edge_array.ndim == 2 and edge_array.shape[0] == 2
    result = edge_array.copy()
    result[0] += int(task_offset)
    result[1] += int(worker_offset)
    return result


def _selected_task_tensor(
    selected_tasks: int | Sequence[int] | torch.Tensor,
    *,
    device: torch.device | None,
) -> torch.Tensor:
    if isinstance(selected_tasks, torch.Tensor):
        result = selected_tasks.to(device=device, dtype=torch.long).reshape(-1)
    else:
        result = torch.as_tensor(selected_tasks, dtype=torch.long, device=device).reshape(-1)
    assert result.ndim == 1
    return result


def build_station_baseline_match(
    baseline_stations: Sequence[int] | np.ndarray | torch.Tensor | None,
    *,
    selected_tasks: int | Sequence[int] | torch.Tensor,
    candidate_station_ids: Sequence[int] | np.ndarray | torch.Tensor,
    enabled: bool,
    device: torch.device | None = None,
) -> torch.Tensor:
    """构造 selected-task × candidate-station 的 [B,S,1] 二值 observation。"""
    selected = _selected_task_tensor(selected_tasks, device=device)
    candidates = torch.as_tensor(candidate_station_ids, dtype=torch.long, device=device)
    selected = selected.to(candidates.device)
    if candidates.ndim == 1:
        candidates = candidates.unsqueeze(0).expand(selected.numel(), -1)
    assert candidates.ndim == 2 and candidates.shape[0] == selected.numel()
    flags = torch.zeros(
        (selected.numel(), candidates.shape[1], 1), dtype=torch.float32, device=candidates.device
    )
    if not enabled or baseline_stations is None:
        return flags
    stations = torch.as_tensor(baseline_stations, dtype=torch.long, device=candidates.device).reshape(-1)
    valid = (selected >= 0) & (selected < stations.numel())
    if valid.any():
        selected_safe = selected.clamp(0, max(0, stations.numel() - 1))
        baseline = stations[selected_safe]
        valid = valid & (baseline >= 0)
        flags[..., 0] = (candidates == baseline.unsqueeze(1)).to(torch.float32)
        flags[~valid] = 0.0
    assert flags.shape == (selected.numel(), candidates.shape[1], 1)
    assert torch.isfinite(flags).all()
    assert ((flags == 0.0) | (flags == 1.0)).all()
    return flags


def build_worker_baseline_membership(
    baseline_team_edge_index: np.ndarray | torch.Tensor | None,
    *,
    selected_tasks: int | Sequence[int] | torch.Tensor,
    num_workers: int,
    enabled: bool,
    candidate_worker_offsets: Sequence[int] | torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """构造 selected-task × candidate-worker 的 [B,N,1] 二值 observation。"""
    selected = _selected_task_tensor(selected_tasks, device=device)
    if device is None and isinstance(baseline_team_edge_index, torch.Tensor):
        selected = selected.to(baseline_team_edge_index.device)
    batch_size = selected.numel()
    flags = torch.zeros(
        (batch_size, int(num_workers), 1), dtype=torch.float32, device=selected.device
    )
    if not enabled or baseline_team_edge_index is None:
        return flags
    edge = torch.as_tensor(baseline_team_edge_index, dtype=torch.long, device=selected.device)
    assert edge.ndim == 2 and edge.shape[0] == 2
    if candidate_worker_offsets is None:
        offsets = torch.zeros((batch_size,), dtype=torch.long, device=selected.device)
    else:
        offsets = torch.as_tensor(
            candidate_worker_offsets, dtype=torch.long, device=selected.device
        ).reshape(-1)
        assert offsets.shape == (batch_size,)
    edge_tasks, edge_workers = edge[0], edge[1]
    for batch_index in range(batch_size):
        workers = edge_workers[edge_tasks == selected[batch_index]] - offsets[batch_index]
        workers = workers[(workers >= 0) & (workers < int(num_workers))]
        if workers.numel() > 0:
            flags[batch_index, workers.unique(), 0] = 1.0
    assert flags.shape == (batch_size, int(num_workers), 1)
    assert torch.isfinite(flags).all()
    assert ((flags == 0.0) | (flags == 1.0)).all()
    return flags
