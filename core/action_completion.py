"""APAL 下层资源动作的确定性可行补全与安全团队候选生成。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import HeteroData

from worker_feature_layout import resolve_worker_feature_layout


@dataclass(frozen=True)
class CompletedResources:
    """一个完整、可直接交给 APAL 环境执行的资源动作。"""

    station_id: int
    team: tuple[int, ...]


@dataclass(frozen=True)
class TeamCandidates:
    """同一工序—工位下的合法团队候选及其状态门控特征。"""

    station_id: int
    teams: tuple[tuple[int, ...], ...]
    gate_features: torch.Tensor
    relative_finish_costs: torch.Tensor


class EarliestFinishActionCompleter:
    """以当前资源状态估算最早完工时间，并始终遵守 APAL 硬约束。"""

    def __init__(self, config) -> None:
        self.config = config
        self.worker_layout = resolve_worker_feature_layout(config)

    def _extract_task_requirements(
        self, task_x: torch.Tensor, task_id: int
    ) -> tuple[int, int, float] | None:
        skill_end = 5 + int(self.worker_layout.num_skill_types)
        task_row = task_x[int(task_id)]
        task_skill_vector = task_row[5:skill_end]
        if float(task_row[0].item()) <= 1.0e-8 or not bool(
            (task_skill_vector > 0.5).any()
        ):
            return None
        demand = max(1, int(round(float(task_row[16].item()))))
        return int(torch.argmax(task_skill_vector).item()), demand, float(task_row[0].item())

    def _legal_worker_ids(
        self,
        worker_x: torch.Tensor,
        *,
        required_skill: int,
        station_id: int,
        worker_mask: torch.Tensor | None,
    ) -> list[int]:
        base_worker_mask = (
            worker_mask.to(device=worker_x.device, dtype=torch.bool).clone()
            if worker_mask is not None
            else torch.zeros(worker_x.size(0), dtype=torch.bool, device=worker_x.device)
        )
        skills = worker_x[:, self.worker_layout.skill_slice]
        locks = torch.argmax(worker_x[:, self.worker_layout.lock_slice], dim=1)
        legal = ~base_worker_mask
        legal &= skills[:, required_skill] > 0.5
        legal &= (locks == 0) | (locks == int(station_id) + 1)
        return torch.nonzero(legal, as_tuple=False).reshape(-1).tolist()

    def _team_score(
        self,
        *,
        team: tuple[int, ...],
        station_id: int,
        task_duration: float,
        demand: int,
        worker_wait: torch.Tensor,
        worker_capacity: torch.Tensor,
        station_wait: torch.Tensor,
        station_x: torch.Tensor,
    ) -> tuple[float, float, int, tuple[int, ...]]:
        team_ready = max(float(worker_wait[wid].item()) for wid in team)
        capacity_sum = sum(float(worker_capacity[wid].item()) for wid in team)
        synergy = 0.95 ** (len(team) - 1)
        estimated_finish = max(team_ready, float(station_wait[station_id].item()))
        estimated_finish += task_duration * demand / max(capacity_sum * synergy, 1.0e-6)
        station_load = float(station_x[station_id, 0].item())
        return estimated_finish, station_load, int(station_id), tuple(team)

    def _complete_for_station(
        self,
        *,
        task_x: torch.Tensor,
        worker_x: torch.Tensor,
        station_x: torch.Tensor,
        task_id: int,
        station_id: int,
        worker_mask: torch.Tensor | None,
    ) -> CompletedResources | None:
        requirements = self._extract_task_requirements(task_x, task_id)
        if requirements is None:
            return CompletedResources(station_id=-1, team=())
        required_skill, demand, task_duration = requirements
        worker_ids = self._legal_worker_ids(
            worker_x,
            required_skill=required_skill,
            station_id=station_id,
            worker_mask=worker_mask,
        )
        if len(worker_ids) < demand:
            return None

        worker_wait = torch.expm1(worker_x[:, self.worker_layout.wait_idx]).clamp_min(0.0)
        station_wait = torch.expm1(station_x[:, 4]).clamp_min(0.0)
        worker_capacity = (
            worker_x[:, self.worker_layout.efficiency_idx]
            * worker_x[:, self.worker_layout.fatigue_idx]
        ).clamp_min(1.0e-6)

        team_list: list[int] = []
        for _ in range(demand):
            remaining = [wid for wid in worker_ids if wid not in team_list]

            def candidate_finish(worker_id: int) -> tuple[float, int]:
                candidate_team = tuple(team_list + [int(worker_id)])
                score = self._team_score(
                    team=candidate_team,
                    station_id=station_id,
                    task_duration=task_duration,
                    demand=demand,
                    worker_wait=worker_wait,
                    worker_capacity=worker_capacity,
                    station_wait=station_wait,
                    station_x=station_x,
                )
                return score[0], int(worker_id)

            team_list.append(min(remaining, key=candidate_finish))
        return CompletedResources(station_id=int(station_id), team=tuple(team_list))

    def complete(
        self,
        obs: HeteroData,
        *,
        task_id: int,
        station_mask: torch.Tensor | None,
        worker_mask: torch.Tensor | None,
        selected_station: int | None = None,
    ) -> CompletedResources | None:
        """返回旧动作范围使用的唯一最早完工可行团队。"""
        task_x = obs["task"].x
        worker_x = obs["worker"].x
        station_x = obs["station"].x
        assert task_x.ndim == worker_x.ndim == station_x.ndim == 2
        assert worker_x.size(1) == self.worker_layout.total_dim

        requirements = self._extract_task_requirements(task_x, task_id)
        if requirements is None:
            return CompletedResources(station_id=-1, team=())

        if selected_station is None:
            valid = torch.ones(station_x.size(0), dtype=torch.bool, device=station_x.device)
            if station_mask is not None:
                valid &= ~station_mask.to(device=station_x.device, dtype=torch.bool).reshape(-1)
            station_candidates = torch.nonzero(valid, as_tuple=False).reshape(-1).tolist()
            station_candidates.sort(
                key=lambda sid: (float(station_x[int(sid), 0].item()), int(sid))
            )
        else:
            station_candidates = [int(selected_station)]

        best: tuple[tuple[float, float, int, tuple[int, ...]], CompletedResources] | None = None
        for station_id in station_candidates:
            if station_mask is not None and bool(station_mask.reshape(-1)[station_id].item()):
                continue
            result = self._complete_for_station(
                task_x=task_x,
                worker_x=worker_x,
                station_x=station_x,
                task_id=task_id,
                station_id=station_id,
                worker_mask=worker_mask,
            )
            if result is None:
                continue
            if result.station_id < 0:
                return result
            required_skill, demand, task_duration = requirements
            del required_skill
            worker_wait = torch.expm1(worker_x[:, self.worker_layout.wait_idx]).clamp_min(0.0)
            station_wait = torch.expm1(station_x[:, 4]).clamp_min(0.0)
            worker_capacity = (
                worker_x[:, self.worker_layout.efficiency_idx]
                * worker_x[:, self.worker_layout.fatigue_idx]
            ).clamp_min(1.0e-6)
            score = self._team_score(
                team=result.team,
                station_id=station_id,
                task_duration=task_duration,
                demand=demand,
                worker_wait=worker_wait,
                worker_capacity=worker_capacity,
                station_wait=station_wait,
                station_x=station_x,
            )
            if best is None or score < best[0]:
                best = (score, result)
        return None if best is None else best[1]

    def enumerate_team_candidates(
        self,
        obs: HeteroData,
        *,
        task_id: int,
        station_id: int,
        worker_mask: torch.Tensor | None,
        max_candidates: int | None = None,
    ) -> TeamCandidates | None:
        """生成候选 0 为旧补全器输出的、确定性且全部合法的团队集合。"""
        task_x = obs["task"].x
        worker_x = obs["worker"].x
        station_x = obs["station"].x
        assert task_x.ndim == worker_x.ndim == station_x.ndim == 2
        assert worker_x.size(1) == self.worker_layout.total_dim
        return self.enumerate_team_candidates_from_features(
            task_x=task_x,
            worker_x=worker_x,
            station_x=station_x,
            task_id=task_id,
            station_id=station_id,
            worker_mask=worker_mask,
            max_candidates=max_candidates,
        )

    def enumerate_team_candidates_from_features(
        self,
        *,
        task_x: torch.Tensor,
        worker_x: torch.Tensor,
        station_x: torch.Tensor,
        task_id: int,
        station_id: int,
        worker_mask: torch.Tensor | None,
        max_candidates: int | None = None,
    ) -> TeamCandidates | None:
        """从原始节点特征重建候选；供 PPO 采样与重算共同使用。"""
        limit = int(
            max_candidates
            if max_candidates is not None
            else getattr(self.config, "conditional_team_max_candidates", 4)
        )
        if limit < 1:
            raise ValueError("conditional_team_max_candidates 必须大于等于 1")
        requirements = self._extract_task_requirements(task_x, task_id)
        if requirements is None:
            return TeamCandidates(
                station_id=-1,
                teams=((),),
                gate_features=torch.zeros(5, dtype=task_x.dtype, device=task_x.device),
                relative_finish_costs=torch.zeros(1, dtype=task_x.dtype, device=task_x.device),
            )
        required_skill, demand, task_duration = requirements
        base = self._complete_for_station(
            task_x=task_x,
            worker_x=worker_x,
            station_x=station_x,
            task_id=task_id,
            station_id=station_id,
            worker_mask=worker_mask,
        )
        if base is None:
            return None

        worker_ids = self._legal_worker_ids(
            worker_x,
            required_skill=required_skill,
            station_id=station_id,
            worker_mask=worker_mask,
        )
        worker_wait = torch.expm1(worker_x[:, self.worker_layout.wait_idx]).clamp_min(0.0)
        station_wait = torch.expm1(station_x[:, 4]).clamp_min(0.0)
        worker_capacity = (
            worker_x[:, self.worker_layout.efficiency_idx]
            * worker_x[:, self.worker_layout.fatigue_idx]
        ).clamp_min(1.0e-6)
        base_score = self._team_score(
            team=base.team,
            station_id=station_id,
            task_duration=task_duration,
            demand=demand,
            worker_wait=worker_wait,
            worker_capacity=worker_capacity,
            station_wait=station_wait,
            station_x=station_x,
        )

        alternatives: list[tuple[tuple[float, float, int, tuple[int, ...]], tuple[int, ...]]] = []
        base_members = set(base.team)
        for replace_at in range(len(base.team)):
            for replacement in worker_ids:
                if replacement in base_members:
                    continue
                candidate = list(base.team)
                candidate[replace_at] = int(replacement)
                candidate_team = tuple(candidate)
                alternatives.append(
                    (
                        self._team_score(
                            team=candidate_team,
                            station_id=station_id,
                            task_duration=task_duration,
                            demand=demand,
                            worker_wait=worker_wait,
                            worker_capacity=worker_capacity,
                            station_wait=station_wait,
                            station_x=station_x,
                        ),
                        candidate_team,
                    )
                )
        alternatives.sort(key=lambda item: item[0])
        teams: list[tuple[int, ...]] = [base.team]
        relative_finish_costs: list[float] = [0.0]
        duration_scale = max(task_duration, 1.0e-6)
        for score, candidate in alternatives:
            if candidate not in teams:
                teams.append(candidate)
                relative_finish_costs.append(
                    max(0.0, (float(score[0]) - float(base_score[0])) / duration_scale)
                )
            if len(teams) >= limit:
                break

        legal_waits = worker_wait[torch.tensor(worker_ids, device=worker_wait.device)]
        gate_features = torch.tensor(
            [
                min(1.0, float(demand) / max(1, int(worker_x.size(0)))),
                min(1.0, float(len(worker_ids)) / max(1, int(worker_x.size(0)))),
                float(len(teams)) / float(limit),
                min(10.0, float(station_wait[station_id].item()) / duration_scale),
                min(10.0, float(legal_waits.std(unbiased=False).item()) / duration_scale),
            ],
            dtype=task_x.dtype,
            device=task_x.device,
        )
        return TeamCandidates(
            station_id=int(station_id),
            teams=tuple(teams),
            gate_features=gate_features,
            relative_finish_costs=torch.tensor(
                relative_finish_costs, dtype=task_x.dtype, device=task_x.device
            ),
        )


__all__ = [
    "CompletedResources",
    "EarliestFinishActionCompleter",
    "TeamCandidates",
]
