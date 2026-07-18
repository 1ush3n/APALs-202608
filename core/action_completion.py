"""未由策略学习的 APAL 下层动作的确定性可行补全。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import HeteroData

from worker_feature_layout import resolve_worker_feature_layout


@dataclass(frozen=True)
class CompletedResources:
    station_id: int
    team: tuple[int, ...]


class EarliestFinishActionCompleter:
    """使用当前负载和工人有效效率近似最早完工，不读取未来动作。"""

    def __init__(self, config) -> None:
        self.config = config
        self.worker_layout = resolve_worker_feature_layout(config)

    def complete(
        self,
        obs: HeteroData,
        *,
        task_id: int,
        station_mask: torch.Tensor | None,
        worker_mask: torch.Tensor | None,
        selected_station: int | None = None,
    ) -> CompletedResources | None:
        task_x = obs["task"].x
        worker_x = obs["worker"].x
        station_x = obs["station"].x
        assert task_x.ndim == worker_x.ndim == station_x.ndim == 2
        assert worker_x.size(1) == self.worker_layout.total_dim

        skill_end = 5 + int(self.worker_layout.num_skill_types)
        task_skill_vector = task_x[int(task_id), 5:skill_end]
        if float(task_x[int(task_id), 0].item()) <= 1.0e-8 or not bool(
            (task_skill_vector > 0.5).any()
        ):
            return CompletedResources(station_id=-1, team=())

        demand = max(1, int(round(float(task_x[int(task_id), 16].item()))))
        required_skill = int(torch.argmax(task_skill_vector).item())

        if selected_station is None:
            valid = torch.ones(station_x.size(0), dtype=torch.bool, device=station_x.device)
            if station_mask is not None:
                valid &= ~station_mask.to(device=station_x.device, dtype=torch.bool).reshape(-1)
            station_candidates = torch.nonzero(valid, as_tuple=False).reshape(-1).tolist()
            # station_x[:, 0] 是按理想负载归一化的当前工位负载。
            station_candidates.sort(
                key=lambda sid: (float(station_x[int(sid), 0].item()), int(sid))
            )
        else:
            station_candidates = [int(selected_station)]

        base_worker_mask = (
            worker_mask.to(device=worker_x.device, dtype=torch.bool).clone()
            if worker_mask is not None
            else torch.zeros(worker_x.size(0), dtype=torch.bool, device=worker_x.device)
        )
        skills = worker_x[:, self.worker_layout.skill_slice]
        locks = torch.argmax(worker_x[:, self.worker_layout.lock_slice], dim=1)

        best: tuple[tuple[float, float, int, tuple[int, ...]], CompletedResources] | None = None
        for station_id in station_candidates:
            if station_mask is not None and bool(station_mask.reshape(-1)[station_id].item()):
                continue
            legal = ~base_worker_mask
            legal &= skills[:, required_skill] > 0.5
            legal &= (locks == 0) | (locks == station_id + 1)
            worker_ids = torch.nonzero(legal, as_tuple=False).reshape(-1).tolist()
            if len(worker_ids) < demand:
                continue
            # worker_x[:, 0] 是基础效率；当前可用性已经由 worker_mask 保证。
            worker_ids.sort(
                key=lambda wid: (-float(worker_x[int(wid), 0].item()), int(wid))
            )
            team = tuple(int(wid) for wid in worker_ids[:demand])
            efficiency_sum = sum(float(worker_x[wid, 0].item()) for wid in team)
            station_load = float(station_x[station_id, 0].item())
            score = (station_load - efficiency_sum, station_load, station_id, team)
            result = CompletedResources(station_id=station_id, team=team)
            if best is None or score < best[0]:
                best = (score, result)
        return None if best is None else best[1]


__all__ = ["CompletedResources", "EarliestFinishActionCompleter"]
