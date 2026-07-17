"""面向 APAL 规则与元启发式基线的可行性保持解码器。"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from configs import configs
from core.constraints import ScheduleValidationReport


class PriorityEncoding(Protocol):
    """解码器依赖的最小优先级编码接口。"""

    task_priority: np.ndarray
    station_priority: np.ndarray
    worker_priority: np.ndarray


@dataclass
class FeasibilityDecodeResult:
    """一次确定性解码的结果与安全状态诊断。"""

    fitness: float
    makespan: float
    balance_std: float
    assigned_tasks: list[tuple[int, int, list[int], float, float]]
    complete: bool
    deadlock_count: int
    failure_type: str | None = None
    validation_report: ScheduleValidationReport | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


class FeasibilityPreservingDecoder:
    """在每次永久绑定移动工人前保留剩余工位—技能配额。"""

    def __init__(
        self,
        env: Any,
        balance_weight: float = 1.0,
        max_steps_factor: int = 4,
    ) -> None:
        self.env = env
        self.balance_weight = float(balance_weight)
        self.max_steps_factor = int(max_steps_factor)
        self.num_skills = int(configs.num_skill_types)
        self.preassignment_ratio = float(configs.heuristic_worker_preassignment_ratio)
        self.reserve_enabled = bool(configs.heuristic_enable_mobile_skill_reserve)
        self.team_candidate_pool = int(configs.heuristic_team_candidate_pool)
        self.team_search_limit = int(configs.heuristic_team_search_limit)

        if not 0.0 <= self.preassignment_ratio <= 1.0:
            raise ValueError("heuristic_worker_preassignment_ratio 必须位于 [0, 1]")
        if self.team_candidate_pool < 1 or self.team_search_limit < 1:
            raise ValueError("启发式团队候选参数必须为正整数")

        self._physical_mask = np.empty(0, dtype=bool)
        self._task_features = np.empty((0, 3), dtype=float)
        self._worker_skills = np.empty((0, self.num_skills), dtype=bool)
        self._station_plan = np.empty(0, dtype=np.int64)
        self._quota_hist = np.empty((0, self.num_skills, 1), dtype=np.int64)
        self._reserve_oracle_active = False

    def decode(self, solution: PriorityEncoding, seed: int) -> FeasibilityDecodeResult:
        """将优先级编码解码为一个由环境硬约束执行并复核的排程。"""
        _set_seed(seed)
        env = self.env
        env.skip_obs_building = True
        env.reset(randomize_duration=False, randomize_workers=False, seed=seed)

        self._task_features = _to_numpy(env.task_static_feat).astype(float, copy=False)
        # worker_skill_matrix shape: [W, K] -> [W, num_skills]
        self._worker_skills = _to_numpy(env.worker_skill_matrix)[:, : self.num_skills] > 0.5
        self._physical_mask = np.asarray(env.constraint_engine.physical_mask, dtype=bool)

        self._validate_encoding_shapes(solution)
        self._station_plan = self._build_station_plan(solution)
        self._quota_hist = self._build_quota_histogram(self._station_plan)
        initial_quota = self._current_quota()
        self._reserve_oracle_active = self.reserve_enabled and self._reserve_state_is_safe(
            np.asarray(env.worker_locks, dtype=np.int64), initial_quota
        )
        preassigned_count = self._preassign_workers(initial_quota)

        max_steps = max(1, int(env.num_tasks) * self.max_steps_factor)
        deadlock_count = 0
        failure_type: str | None = None
        failure_details: dict[str, Any] = {}

        for _ in range(max_steps):
            if len(env.assigned_tasks) == env.num_tasks:
                break

            virtual_task = self._select_ready_virtual_task(solution)
            if virtual_task is not None:
                _, _, _, info = env.step((virtual_task, -1, []))
                if info.get("invalid_action"):
                    failure_type = "invalid_action"
                    failure_details = dict(info)
                    break
                continue

            task_mask_raw, station_mask_raw, _ = env.get_masks()
            task_mask = _to_numpy(task_mask_raw).astype(bool, copy=False)
            station_mask = _to_numpy(station_mask_raw).astype(bool, copy=False)
            available_tasks = np.where((~task_mask) & self._physical_mask)[0]
            ordered_tasks = sorted(
                (int(task_id) for task_id in available_tasks),
                key=lambda task_id: float(solution.task_priority[task_id]),
                reverse=True,
            )

            action: tuple[int, int, list[int]] | None = None
            for task_id in ordered_tasks:
                planned_station = int(self._station_plan[task_id])
                if planned_station < 0 or bool(station_mask[task_id, planned_station]):
                    continue
                team = self._select_safe_team(task_id, planned_station, solution)
                if team is not None:
                    action = (task_id, planned_station, team)
                    break

            if action is None:
                if env.try_wait_for_resources():
                    continue
                deadlock_count += 1
                failure_type = "deadlock"
                break

            task_id, _, _ = action
            _, _, _, info = env.step(action)
            if info.get("invalid_action"):
                failure_type = "invalid_action"
                failure_details = dict(info)
                break
            self._remove_completed_task_from_quota(task_id)

        complete = len(env.assigned_tasks) == env.num_tasks
        validation_report: ScheduleValidationReport | None = None
        if complete:
            assigned_tasks = list(env.assigned_tasks)
            validation_report = env.validate_assignments(assigned_tasks)
            if not validation_report.is_legal:
                complete = False
                assigned_tasks = []
                failure_type = "illegal_schedule"
                failure_details = {
                    "violations": {
                        key: int(value)
                        for key, value in validation_report.violations.items()
                        if int(value) > 0
                    },
                    "examples": {
                        key: value
                        for key, value in validation_report.examples.items()
                        if value
                    },
                }

        if complete:
            makespan = float(np.max(env.station_wall_clock)) if env.num_stations else 0.0
            balance_std = float(np.std(env.station_loads))
            fitness = makespan + self.balance_weight * balance_std
        else:
            assigned_tasks = []
            makespan = float(env.ideal_makespan * 3.0)
            balance_std = float(env.ideal_station_load * 3.0)
            # 任何未完成或非法候选都不得参与最优解竞争。
            fitness = float("inf")
            if failure_type is None:
                failure_type = "step_limit"

        return FeasibilityDecodeResult(
            fitness=float(fitness),
            makespan=makespan,
            balance_std=balance_std,
            assigned_tasks=assigned_tasks,
            complete=complete,
            deadlock_count=deadlock_count,
            failure_type=failure_type,
            validation_report=validation_report,
            diagnostics={
                "failure_type": failure_type,
                "failure_details": failure_details,
                "invalid_schedule_count": int(failure_type == "illegal_schedule"),
                "preassignment_target": int(round(env.num_workers * self.preassignment_ratio)),
                "preassigned_workers": int(preassigned_count),
                "mobile_reserve_workers": int(np.sum(np.asarray(env.worker_locks) == 0)),
                "reserve_oracle_active": bool(self._reserve_oracle_active),
                "planned_station_count": int(np.unique(self._station_plan[self._physical_mask]).size),
                "initial_station_skill_quota": initial_quota.tolist(),
                "final_worker_lock_counts": np.bincount(
                    np.asarray(env.worker_locks, dtype=np.int64),
                    minlength=env.num_stations + 1,
                ).tolist(),
            },
        )

    def _validate_encoding_shapes(self, solution: PriorityEncoding) -> None:
        env = self.env
        assert solution.task_priority.shape == (env.num_tasks,), (
            "task_priority shape 必须为 [T]"
        )
        assert solution.station_priority.shape == (env.num_tasks, env.num_stations), (
            "station_priority shape 必须为 [T, S]"
        )
        assert solution.worker_priority.shape == (env.num_tasks, env.num_workers), (
            "worker_priority shape 必须为 [T, W]"
        )

    def _build_station_plan(self, solution: PriorityEncoding) -> np.ndarray:
        """构造满足物理前驱站位单调性和站位上界的静态目标工位。"""
        env = self.env
        plan = np.full(env.num_tasks, -1, dtype=np.int64)
        projected_load = np.zeros(env.num_stations, dtype=float)

        for task_id in env._topological_sort():
            task_id = int(task_id)
            if not self._physical_mask[task_id]:
                continue
            physical_predecessors = env.constraint_engine.physical_predecessors[task_id]
            minimum = max(
                (int(plan[pred]) for pred in physical_predecessors if int(plan[pred]) >= 0),
                default=0,
            )
            maximum = min(env.num_stations - 1, int(env.max_allowed_stations[task_id]))
            fixed = int(env.fixed_stations[task_id])
            candidates = [fixed] if fixed >= 0 else list(range(minimum, maximum + 1))
            candidates = [station for station in candidates if minimum <= station <= maximum]
            if not candidates:
                raise RuntimeError(
                    f"工序 {task_id} 无可行目标工位：minimum={minimum}, maximum={maximum}, fixed={fixed}"
                )

            workload = float(self._task_features[task_id, 0]) * max(
                1, int(self._task_features[task_id, 2])
            )
            station = min(
                candidates,
                key=lambda station_id: (
                    projected_load[station_id],
                    -float(solution.station_priority[task_id, station_id]),
                    station_id,
                ),
            )
            plan[task_id] = int(station)
            projected_load[station] += workload

        return plan

    def _build_quota_histogram(self, station_plan: np.ndarray) -> np.ndarray:
        max_demand = max(
            1,
            int(np.max(self._task_features[self._physical_mask, 2]))
            if bool(np.any(self._physical_mask))
            else 1,
        )
        histogram = np.zeros(
            (self.env.num_stations, self.num_skills, max_demand + 1),
            dtype=np.int64,
        )
        for task_id in np.where(self._physical_mask)[0]:
            station = int(station_plan[task_id])
            skill = int(self._task_features[task_id, 1])
            demand = max(1, int(self._task_features[task_id, 2]))
            if not 0 <= skill < self.num_skills:
                raise ValueError(f"物理工序 {task_id} 的工种 {skill} 越界")
            histogram[station, skill, demand] += 1
        return histogram

    def _current_quota(self) -> np.ndarray:
        # quota_hist shape: [S, K, D+1] -> quota shape: [S, K]
        demand_values = np.arange(self._quota_hist.shape[2], dtype=np.int64)
        return np.max((self._quota_hist > 0) * demand_values[None, None, :], axis=2)

    def _quota_after_task(self, task_id: int) -> np.ndarray:
        quota = self._current_quota()
        station = int(self._station_plan[task_id])
        skill = int(self._task_features[task_id, 1])
        demand = max(1, int(self._task_features[task_id, 2]))
        if self._quota_hist[station, skill, demand] == 1 and quota[station, skill] == demand:
            lower = np.where(self._quota_hist[station, skill, :demand] > 0)[0]
            quota[station, skill] = int(lower[-1]) if lower.size else 0
        return quota

    def _remove_completed_task_from_quota(self, task_id: int) -> None:
        station = int(self._station_plan[task_id])
        skill = int(self._task_features[task_id, 1])
        demand = max(1, int(self._task_features[task_id, 2]))
        assert self._quota_hist[station, skill, demand] > 0, "配额直方图发生下溢"
        self._quota_hist[station, skill, demand] -= 1

    def _reserve_state_is_safe(self, worker_locks: np.ndarray, quota: np.ndarray) -> bool:
        """用保守的多技能贪心覆盖检查剩余移动工人能否填满全部配额。"""
        if not self.reserve_enabled:
            return True

        locks = np.asarray(worker_locks, dtype=np.int64)
        locked_coverage = np.zeros_like(quota, dtype=np.int64)
        for station in range(self.env.num_stations):
            assigned = np.where(locks == station + 1)[0]
            if assigned.size:
                locked_coverage[station] = np.sum(self._worker_skills[assigned], axis=0)
        deficits = np.maximum(np.asarray(quota, dtype=np.int64) - locked_coverage, 0)
        if not bool(np.any(deficits)):
            return True

        mobile = np.where(locks == 0)[0]
        if mobile.size == 0:
            return False
        mobile_skills = self._worker_skills[mobile].astype(np.int64, copy=False)
        available = np.ones(mobile.size, dtype=bool)

        while bool(np.any(deficits)):
            unmet = deficits > 0
            # mobile_skills shape: [M, K], unmet.T shape: [K, S] -> gains shape: [M, S]
            gains = mobile_skills @ unmet.T.astype(np.int64)
            gains[~available, :] = -1
            flat_index = int(np.argmax(gains))
            best_gain = int(gains.flat[flat_index])
            if best_gain <= 0:
                return False
            mobile_index, station = np.unravel_index(flat_index, gains.shape)
            covered_skills = self._worker_skills[mobile[mobile_index]] & (deficits[station] > 0)
            deficits[station, covered_skills] -= 1
            available[mobile_index] = False
        return True

    def _preassign_workers(self, quota: np.ndarray) -> int:
        """在不破坏移动保留池覆盖能力的前提下预绑定目标比例的工人。"""
        if not self._reserve_oracle_active or self.preassignment_ratio <= 0.0:
            return 0

        env = self.env
        target = int(round(env.num_workers * self.preassignment_ratio))
        accepted = 0
        station_headcount = np.zeros(env.num_stations, dtype=np.int64)

        while accepted < target:
            locks = np.asarray(env.worker_locks, dtype=np.int64)
            mobile = np.where(locks == 0)[0]
            if mobile.size == 0:
                break

            coverage = np.zeros_like(quota, dtype=np.int64)
            for station in range(env.num_stations):
                bound = np.where(locks == station + 1)[0]
                if bound.size:
                    coverage[station] = np.sum(self._worker_skills[bound], axis=0)
            deficits = np.maximum(quota - coverage, 0)

            candidates: list[tuple[float, int, int]] = []
            for worker_id in mobile:
                skills = self._worker_skills[worker_id]
                for station in range(env.num_stations):
                    covered_types = int(np.sum(skills & (deficits[station] > 0)))
                    covered_units = int(np.sum(deficits[station, skills]))
                    score = 100.0 * covered_types + covered_units - float(station_headcount[station])
                    candidates.append((score, int(worker_id), station))
            candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

            selected: tuple[int, int] | None = None
            for _, worker_id, station in candidates:
                tentative = locks.copy()
                tentative[worker_id] = station + 1
                if self._reserve_state_is_safe(tentative, quota):
                    selected = (worker_id, station)
                    break
            if selected is None:
                break
            worker_id, station = selected
            env.worker_locks[worker_id] = station + 1
            station_headcount[station] += 1
            accepted += 1

        return accepted

    def _select_ready_virtual_task(self, solution: PriorityEncoding) -> int | None:
        env = self.env
        ready = (np.asarray(env.task_status) == 1) & (~self._physical_mask)
        if hasattr(env, "task_material_ready"):
            ready &= np.asarray(env.task_material_ready) <= float(env.current_time) + 1.0e-9
        candidates = np.where(ready)[0]
        if candidates.size == 0:
            return None
        return int(max(candidates, key=lambda task_id: float(solution.task_priority[int(task_id)])))

    def _select_safe_team(
        self,
        task_id: int,
        station: int,
        solution: PriorityEncoding,
    ) -> list[int] | None:
        env = self.env
        skill = int(self._task_features[task_id, 1])
        demand = max(1, int(self._task_features[task_id, 2]))
        locks = np.asarray(env.worker_locks, dtype=np.int64)
        skilled = np.where(self._worker_skills[:, skill])[0]
        bound = [int(worker) for worker in skilled if locks[worker] == station + 1]
        mobile = [int(worker) for worker in skilled if locks[worker] == 0]
        bound.sort(
            key=lambda worker: float(solution.worker_priority[task_id, worker]),
            reverse=True,
        )
        mobile.sort(
            key=lambda worker: (
                int(np.sum(self._worker_skills[worker])),
                -float(solution.worker_priority[task_id, worker]),
                worker,
            )
        )

        selected_bound = bound[: min(demand, len(bound))]
        mobile_needed = demand - len(selected_bound)
        if mobile_needed < 0 or len(mobile) < mobile_needed:
            return None
        if mobile_needed == 0:
            return selected_bound

        pool_size = max(mobile_needed, min(self.team_candidate_pool, len(mobile)))
        mobile_pool = mobile[:pool_size]
        quota_after = self._quota_after_task(task_id)
        checked = 0
        for combination in itertools.combinations(mobile_pool, mobile_needed):
            checked += 1
            tentative = locks.copy()
            tentative[list(combination)] = station + 1
            if (not self._reserve_oracle_active) or self._reserve_state_is_safe(
                tentative, quota_after
            ):
                return selected_bound + [int(worker) for worker in combination]
            if checked >= self.team_search_limit:
                break
        return None
