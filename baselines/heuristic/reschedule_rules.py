# -*- coding: utf-8 -*-
"""APAL 预测-反应式重调度规则基线。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Type

import math
import numpy as np
import torch

from configs import configs
from environment import AirLineEnv_Graph
from runtime.reschedule_eval import _compute_assignment_utilization, _compute_reschedule_constraint_metrics
from utils.reschedule import RescheduleScenario, calculate_reschedule_composite_score


@dataclass
class RescheduleRuleResult:
    """单个固定重调度场景下的规则求解结果。"""

    method: str
    scenario_id: str
    scenario_level: str
    scenario_start_time: float
    delayed_task_count: int
    makespan: float
    balance_std: float
    reward: float
    duration_sec: float
    assigned_tasks: list[tuple[int, int, list[int], float, float]]
    constraint_metrics: dict[str, float]
    env: AirLineEnv_Graph


@dataclass
class ReschedulePrioritySolution:
    """重调度搜索候选解：任务、工位、工人偏好的连续优先级编码。"""

    task_priority: np.ndarray
    station_priority: np.ndarray
    worker_priority: np.ndarray

    def clone(self) -> "ReschedulePrioritySolution":
        return ReschedulePrioritySolution(
            task_priority=self.task_priority.copy(),
            station_priority=self.station_priority.copy(),
            worker_priority=self.worker_priority.copy(),
        )


@dataclass
class StaticConstraintContext:
    """规则搜索中不随环境 step 改变的静态约束上下文。"""

    durations: np.ndarray
    skills: np.ndarray
    demands: np.ndarray
    worker_skill_matrix: np.ndarray
    baseline_starts: np.ndarray
    baseline_ends: np.ndarray
    baseline_stations: np.ndarray
    baseline_teams: tuple[frozenset[int], ...]
    movable_mask: np.ndarray
    release_times: np.ndarray
    critical_tail: np.ndarray
    capable_workers_by_skill: tuple[np.ndarray, ...]


def _to_numpy_bool(mask: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(mask):
        return mask.detach().cpu().numpy().astype(bool)
    return np.asarray(mask, dtype=bool)


def _worker_skill_matrix_np(env: AirLineEnv_Graph) -> np.ndarray:
    matrix = env.worker_skill_matrix
    if torch.is_tensor(matrix):
        return matrix.detach().cpu().numpy()
    return np.asarray(matrix)


def _duration(env: AirLineEnv_Graph, task_id: int) -> float:
    return float(env.task_static_feat[int(task_id), 0].item())


def _demand(env: AirLineEnv_Graph, task_id: int) -> int:
    return max(1, int(env.task_static_feat[int(task_id), 2].item()))


def _skill_id(env: AirLineEnv_Graph, task_id: int) -> int:
    return int(env.task_static_feat[int(task_id), 1].item())


def _critical_tail_lengths(env: AirLineEnv_Graph) -> np.ndarray:
    """计算每个任务到后继终点的最长剩余工时，用作 CPM 优先级。"""

    durations = np.asarray([_duration(env, task_id) for task_id in range(env.num_tasks)], dtype=float)
    tail = durations.copy()
    for task_id in reversed(env._topological_sort()):
        succs = env.successors.get(int(task_id), [])
        if succs:
            tail[int(task_id)] = durations[int(task_id)] + max(tail[int(succ)] for succ in succs)
    return tail


def _normalize(values: np.ndarray) -> np.ndarray:
    arr = values.astype(float).copy()
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


class RescheduleRuleScheduler:
    """面向 APAL 重调度约束的确定性修复规则基类。"""

    method_name = "Rule"
    makespan_weight = 0.15
    balance_weight = 0.05
    takt_weight = 0.20
    start_stability_weight = 0.05
    station_stability_weight = 0.05
    team_stability_weight = 0.05
    baseline_team_preference = 0.20

    def __init__(
        self,
        *,
        data_path_or_dir: str | Path,
        scenario: RescheduleScenario,
        scenario_id: str,
        scenario_level: str = "",
        seed: int = 42,
        verbose: bool = False,
        use_static_cache: bool = True,
        verify_static_cache: bool = False,
    ) -> None:
        self.data_path_or_dir = Path(data_path_or_dir)
        self.scenario = scenario
        self.scenario_id = str(scenario_id)
        self.scenario_level = str(scenario_level)
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.use_static_cache = bool(use_static_cache)
        self.verify_static_cache = bool(verify_static_cache)
        self.static_context: StaticConstraintContext | None = None
        self.rng = np.random.RandomState(self.seed)
        self.env = AirLineEnv_Graph(data_path_or_dir=str(self.data_path_or_dir), seed=self.seed)
        self.env.skip_obs_building = True
        setattr(self.env, "_forced_reschedule_scenario", scenario)
        self.env.reset(randomize_duration=False, randomize_workers=False, seed=self.seed)
        self._refresh_static_context()
        self.random_priority = self.rng.rand(self.env.num_tasks)

    def _reset_env(self) -> None:
        self.env.skip_obs_building = True
        setattr(self.env, "_forced_reschedule_scenario", self.scenario)
        self.env.reset(randomize_duration=False, randomize_workers=False, seed=self.seed)
        self._refresh_static_context()
        self.random_priority = self.rng.rand(self.env.num_tasks)

    def _refresh_static_context(self) -> None:
        durations = self.env.task_static_feat[:, 0].detach().cpu().numpy().astype(float)
        skills = self.env.task_static_feat[:, 1].detach().cpu().numpy().astype(int)
        demands = np.maximum(1, self.env.task_static_feat[:, 2].detach().cpu().numpy().astype(int))
        worker_skill_matrix = _worker_skill_matrix_np(self.env).astype(float, copy=False)

        baseline_starts = np.full(self.env.num_tasks, np.nan, dtype=float)
        baseline_ends = np.full(self.env.num_tasks, np.nan, dtype=float)
        baseline_stations = np.full(self.env.num_tasks, -1, dtype=int)
        baseline_team_lists: list[frozenset[int]] = [frozenset() for _ in range(self.env.num_tasks)]
        movable_mask = np.ones(self.env.num_tasks, dtype=bool)
        if self.env.baseline_schedule is not None:
            for task_id, base in self.env.baseline_schedule.tasks.items():
                task_idx = int(task_id)
                if task_idx < 0 or task_idx >= self.env.num_tasks:
                    continue
                baseline_starts[task_idx] = float(base.start)
                baseline_ends[task_idx] = float(base.end)
                baseline_stations[task_idx] = int(base.station_id)
                baseline_team_lists[task_idx] = frozenset(int(w) for w in base.team)
                movable_mask[task_idx] = float(base.start) > float(self.scenario.start_time) + 1e-9

        release_times = np.asarray(getattr(self.env, "task_material_ready", np.zeros(self.env.num_tasks)), dtype=float).copy()
        critical_tail = _critical_tail_lengths(self.env)
        max_skill = int(np.max(skills)) if skills.size else 0
        num_skills = max(int(getattr(configs, "num_skill_types", 10)), max_skill + 1)
        capable_workers = []
        for skill_id in range(num_skills):
            if skill_id < worker_skill_matrix.shape[1]:
                capable_workers.append(np.where(worker_skill_matrix[:, skill_id] > 0.5)[0].astype(int))
            else:
                capable_workers.append(np.asarray([], dtype=int))

        self.critical_tail = critical_tail
        self.static_context = StaticConstraintContext(
            durations=durations,
            skills=skills,
            demands=demands,
            worker_skill_matrix=worker_skill_matrix,
            baseline_starts=baseline_starts,
            baseline_ends=baseline_ends,
            baseline_stations=baseline_stations,
            baseline_teams=tuple(baseline_team_lists),
            movable_mask=movable_mask,
            release_times=release_times,
            critical_tail=critical_tail,
            capable_workers_by_skill=tuple(capable_workers),
        )
        if self.verify_static_cache:
            self._verify_static_context()

    def _verify_static_context(self) -> None:
        ctx = self.static_context
        if ctx is None:
            raise AssertionError("静态约束缓存未初始化")
        task_count = min(16, self.env.num_tasks)
        for task_id in range(task_count):
            assert abs(ctx.durations[task_id] - _duration(self.env, task_id)) < 1e-9
            assert int(ctx.skills[task_id]) == _skill_id(self.env, task_id)
            assert int(ctx.demands[task_id]) == _demand(self.env, task_id)
        for skill_id, cached_workers in enumerate(ctx.capable_workers_by_skill[: min(8, len(ctx.capable_workers_by_skill))]):
            if skill_id >= ctx.worker_skill_matrix.shape[1]:
                assert cached_workers.size == 0
                continue
            expected = np.where(ctx.worker_skill_matrix[:, skill_id] > 0.5)[0].astype(int)
            assert np.array_equal(cached_workers, expected)

    def _task_duration(self, task_id: int) -> float:
        if self.use_static_cache and self.static_context is not None:
            return float(self.static_context.durations[int(task_id)])
        return _duration(self.env, task_id)

    def _task_demand(self, task_id: int) -> int:
        if self.use_static_cache and self.static_context is not None:
            return max(1, int(self.static_context.demands[int(task_id)]))
        return _demand(self.env, task_id)

    def _task_skill(self, task_id: int) -> int:
        if self.use_static_cache and self.static_context is not None:
            return int(self.static_context.skills[int(task_id)])
        return _skill_id(self.env, task_id)

    def _baseline_team(self, task_id: int) -> set[int]:
        if self.use_static_cache and self.static_context is not None:
            return set(self.static_context.baseline_teams[int(task_id)])
        base = self.env.baseline_schedule.tasks.get(int(task_id)) if self.env.baseline_schedule else None
        return set(int(w) for w in base.team) if base is not None else set()

    def _has_baseline_task(self, task_id: int) -> bool:
        if self.use_static_cache and self.static_context is not None:
            return not np.isnan(float(self.static_context.baseline_starts[int(task_id)]))
        return bool(self.env.baseline_schedule and int(task_id) in self.env.baseline_schedule.tasks)

    def _baseline_start(self, task_id: int, default: float | None = None) -> float:
        if self.use_static_cache and self.static_context is not None:
            value = float(self.static_context.baseline_starts[int(task_id)])
            if not np.isnan(value):
                return value
        base = self.env.baseline_schedule.tasks.get(int(task_id)) if self.env.baseline_schedule else None
        if base is not None:
            return float(base.start)
        return float(task_id if default is None else default)

    def _baseline_end(self, task_id: int, default: float | None = None) -> float:
        if self.use_static_cache and self.static_context is not None:
            value = float(self.static_context.baseline_ends[int(task_id)])
            if not np.isnan(value):
                return value
        base = self.env.baseline_schedule.tasks.get(int(task_id)) if self.env.baseline_schedule else None
        if base is not None:
            return float(base.end)
        return float(task_id if default is None else default)

    def _baseline_station(self, task_id: int) -> int:
        if self.use_static_cache and self.static_context is not None:
            return int(self.static_context.baseline_stations[int(task_id)])
        base = self.env.baseline_schedule.tasks.get(int(task_id)) if self.env.baseline_schedule else None
        return int(base.station_id) if base is not None else -1

    def task_priority(self, task_id: int) -> float:
        return self._baseline_start(task_id)

    def _valid_workers(self, task_id: int, station_id: int, worker_mask: np.ndarray) -> list[int]:
        skill_id = self._task_skill(task_id)
        locks = np.asarray(self.env.worker_locks, dtype=int)
        if self.use_static_cache and self.static_context is not None and skill_id < len(self.static_context.capable_workers_by_skill):
            worker_iter = self.static_context.capable_workers_by_skill[skill_id].tolist()
        else:
            skill_matrix = _worker_skill_matrix_np(self.env)
            worker_iter = np.where(skill_matrix[:, skill_id] > 0.5)[0].astype(int).tolist()
        candidates = []
        for worker_id in worker_iter:
            if bool(worker_mask[int(worker_id)]):
                continue
            if locks[int(worker_id)] not in {0, int(station_id) + 1}:
                continue
            candidates.append(int(worker_id))
        return candidates

    def _worker_score(self, task_id: int, station_id: int, worker_id: int) -> tuple[float, float, int]:
        in_base_team = int(worker_id) in self._baseline_team(task_id)
        same_station_lock = int(self.env.worker_locks[int(worker_id)]) == int(station_id) + 1
        return (
            0.0 if in_base_team else self.baseline_team_preference,
            float(self.env.worker_free_time[int(worker_id)]) - (0.01 if same_station_lock else 0.0),
            int(worker_id),
        )

    def _select_team(self, task_id: int, station_id: int, worker_mask: np.ndarray) -> list[int] | None:
        candidates = self._valid_workers(task_id, station_id, worker_mask)
        demand = self._task_demand(task_id)
        if len(candidates) < demand:
            return None
        candidates.sort(key=lambda worker_id: self._worker_score(task_id, station_id, worker_id))
        return candidates[:demand]

    def _estimate_action(self, task_id: int, station_id: int, team: list[int]) -> tuple[float, float]:
        team_ready = max([float(self.env.worker_free_time[w]) for w in team], default=float(self.env.current_time))
        pred_ready = float(self.env.current_time)
        for pred in self.env.predecessors.get(int(task_id), []):
            pred_ready = max(pred_ready, float(self.env.task_end_times[int(pred)]))
        min_start = max(float(self.env.current_time), team_ready, pred_ready)
        duration = float(self.env.calculate_duration(int(task_id), team, start_time_est=min_start))
        station_ready = float(self.env._get_station_earliest_available_time(int(station_id), min_start, duration))
        start_time = max(min_start, station_ready)
        return start_time, start_time + duration

    def _action_score(self, task_id: int, station_id: int, team: list[int]) -> float:
        start_time, finish_time = self._estimate_action(task_id, station_id, team)
        baseline = self.env.baseline_schedule
        has_base = self._has_baseline_task(task_id)
        takt = max(1e-6, float(baseline.makespan if baseline else max(1.0, self.env.ideal_makespan)))
        ideal_load = max(1.0, float(self.env.ideal_station_load))
        score = float(self.task_priority(task_id))
        score += self.makespan_weight * (finish_time / takt)
        if station_id >= 0:
            projected_load = float(self.env.station_loads[int(station_id)]) + self._task_duration(task_id) * len(team)
            score += self.balance_weight * (projected_load / ideal_load)
        score += self.takt_weight * max(0.0, finish_time - takt) / takt
        if has_base:
            score += self.start_stability_weight * abs(start_time - self._baseline_start(task_id)) / takt
            score += self.station_stability_weight * float(int(station_id) != self._baseline_station(task_id))
            base_team = self._baseline_team(task_id)
            overlap = len(base_team.intersection(set(int(w) for w in team)))
            denom = max(1, max(len(base_team), len(team)))
            score += self.team_stability_weight * (1.0 - overlap / denom)
        return float(score)

    def _select_action(self) -> tuple[int, int, list[int]] | None:
        task_mask_raw, station_mask_raw, worker_mask_raw = self.env.get_masks()
        task_mask = _to_numpy_bool(task_mask_raw)
        station_mask = _to_numpy_bool(station_mask_raw)
        worker_mask = _to_numpy_bool(worker_mask_raw)
        if task_mask.all():
            return None

        best_action: tuple[int, int, list[int]] | None = None
        best_score = float("inf")
        for task_id in np.where(~task_mask)[0].tolist():
            for station_id in np.where(~station_mask[int(task_id)])[0].tolist():
                team = self._select_team(int(task_id), int(station_id), worker_mask)
                if team is None:
                    continue
                score = self._action_score(int(task_id), int(station_id), team)
                if score < best_score:
                    best_score = score
                    best_action = (int(task_id), int(station_id), [int(w) for w in team])
        return best_action

    def _build_result(self, *, reward: float, invalid_step_count: int, duration_sec: float) -> RescheduleRuleResult:
        complete = len(self.env.assigned_tasks) == self.env.num_tasks
        if complete:
            makespan = float(np.max(self.env.station_wall_clock))
            balance = float(np.std(self.env.station_loads))
        else:
            makespan = float(self.env.ideal_makespan * 3.0)
            balance = float(self.env.ideal_station_load * 3.0)

        constraints = _compute_reschedule_constraint_metrics(self.env)
        constraints["scenario_id"] = self.scenario_id
        constraints["scenario_level"] = self.scenario_level
        constraints["reschedule_start_time"] = float(self.scenario.start_time)
        constraints["delayed_task_count"] = float(len(self.scenario.task_release_times))
        constraints["invalid_step_count"] = float(invalid_step_count)
        constraints["complete"] = float(complete)
        score_result = calculate_reschedule_composite_score(
            makespan=makespan,
            balance_std=balance,
            constraint_metrics=constraints,
            config_obj=configs,
            ideal_station_load=float(getattr(self.env, "ideal_station_load", 1.0)),
        )
        worker_util, station_util = _compute_assignment_utilization(self.env, makespan)
        constraints["eligible"] = float(score_result.eligible)
        constraints["composite_score"] = float(score_result.score)
        constraints["selection_score"] = float(score_result.selection_score)
        constraints["worker_util"] = float(worker_util)
        constraints["station_util"] = float(station_util)
        constraints.update(score_result.terms)
        return RescheduleRuleResult(
            method=self.method_name,
            scenario_id=self.scenario_id,
            scenario_level=self.scenario_level,
            scenario_start_time=float(self.scenario.start_time),
            delayed_task_count=len(self.scenario.task_release_times),
            makespan=makespan,
            balance_std=balance,
            reward=float(reward),
            duration_sec=float(duration_sec),
            assigned_tasks=list(self.env.assigned_tasks),
            constraint_metrics=constraints,
            env=self.env,
        )

    def run(self) -> RescheduleRuleResult:
        self._reset_env()
        start_wall = time.time()
        done = False
        total_reward = 0.0
        invalid_step_count = 0
        max_steps = max(1, self.env.num_tasks * 3)
        for _ in range(max_steps):
            if done:
                break
            action = self._select_action()
            if action is None:
                if self.env.try_wait_for_resources():
                    continue
                break
            _obs, reward, done, info = self.env.step(action)
            total_reward += float(reward)
            if info.get("invalid_action", False):
                invalid_step_count += 1
                if self.verbose:
                    print(f"[{self.method_name}] invalid action: {info}")
                break
        return self._build_result(
            reward=total_reward,
            invalid_step_count=invalid_step_count,
            duration_sec=time.time() - start_wall,
        )


class NoRescheduleRule(RescheduleRuleScheduler):
    method_name = "NoReschedule"

    def run(self) -> RescheduleRuleResult:
        self._reset_env()
        return self._build_result(reward=0.0, invalid_step_count=0, duration_sec=0.0)


class SPTRepairRule(RescheduleRuleScheduler):
    method_name = "SPTRepair"

    def task_priority(self, task_id: int) -> float:
        return self._task_duration(task_id)


class LPTRepairRule(RescheduleRuleScheduler):
    method_name = "LPTRepair"

    def task_priority(self, task_id: int) -> float:
        return -self._task_duration(task_id)


class EDDRepairRule(RescheduleRuleScheduler):
    method_name = "EDDRepair"

    def task_priority(self, task_id: int) -> float:
        return self._baseline_end(task_id)


class MSLRepairRule(RescheduleRuleScheduler):
    method_name = "MSLRepair"

    def task_priority(self, task_id: int) -> float:
        due = self._baseline_end(task_id, default=float(self.env.current_time))
        return due - float(self.env.current_time) - self._task_duration(task_id)


class CPMRepairRule(RescheduleRuleScheduler):
    method_name = "CPMRepair"

    def task_priority(self, task_id: int) -> float:
        return -float(self.critical_tail[int(task_id)])


class RandomRepairRule(RescheduleRuleScheduler):
    method_name = "RandomRepair"

    def task_priority(self, task_id: int) -> float:
        return float(self.random_priority[int(task_id)])


class ReleaseAwareRepairRule(RescheduleRuleScheduler):
    method_name = "ReleaseAwareRepair"

    def task_priority(self, task_id: int) -> float:
        if self.use_static_cache and self.static_context is not None:
            release_time = float(self.static_context.release_times[int(task_id)])
        else:
            release_time = float(self.env.task_material_ready[int(task_id)]) if hasattr(self.env, "task_material_ready") else 0.0
        baseline_start = self._baseline_start(task_id)
        return release_time + 0.01 * baseline_start


class BottleneckSkillRepairRule(RescheduleRuleScheduler):
    method_name = "BottleneckSkillRepair"

    def task_priority(self, task_id: int) -> float:
        skill = self._task_skill(task_id)
        if self.use_static_cache and self.static_context is not None and skill < len(self.static_context.capable_workers_by_skill):
            supply = max(1.0, float(len(self.static_context.capable_workers_by_skill[skill])))
        else:
            supply = max(1.0, float(np.sum(_worker_skill_matrix_np(self.env)[:, skill] > 0.5)))
        scarcity = float(self._task_demand(task_id)) / supply
        return -scarcity


class TaktAwareRepairRule(RescheduleRuleScheduler):
    method_name = "TaktAwareRepair"
    takt_weight = 1.0
    makespan_weight = 0.35
    balance_weight = 0.10

    def task_priority(self, task_id: int) -> float:
        return -float(self.critical_tail[int(task_id)]) * 0.5 + self._task_duration(task_id) * 0.5


class StabilityAwareRepairRule(RescheduleRuleScheduler):
    method_name = "StabilityAwareRepair"
    start_stability_weight = 0.80
    station_stability_weight = 0.60
    team_stability_weight = 0.50
    baseline_team_preference = 0.0

    def task_priority(self, task_id: int) -> float:
        return self._baseline_start(task_id)


class HybridCPMStabilityRepairRule(StabilityAwareRepairRule):
    method_name = "HybridCPMStabilityRepair"
    start_stability_weight = 0.35
    station_stability_weight = 0.30
    team_stability_weight = 0.25
    makespan_weight = 0.30
    takt_weight = 0.50

    def task_priority(self, task_id: int) -> float:
        baseline_start = self._baseline_start(task_id)
        return -float(self.critical_tail[int(task_id)]) + 0.01 * baseline_start


class FullRescheduleCPMRule(CPMRepairRule):
    """对所有可移动任务做 CPM 修复；冻结任务仍按 APAL 重调度定义保持不变。"""

    method_name = "FullRescheduleCPM"
    start_stability_weight = 0.0
    station_stability_weight = 0.0
    team_stability_weight = 0.0
    baseline_team_preference = 0.20


class PrioritySearchRepairRule(RescheduleRuleScheduler):
    """用连续优先级编码驱动 APAL 重调度搜索，所有动作仍由环境执行硬约束校验。"""

    method_name = "PrioritySearchRepair"
    supports_priority_search = True
    search_priority_weight = 1.0

    def __init__(
        self,
        *,
        beam_width: int = 2,
        beam_branch_factor: int = 2,
        beam_levels: int = 2,
        beam_patience: int = 1,
        ig_iterations: int = 3,
        ig_destroy_ratio: float = 0.08,
        ig_noise_sigma: float = 0.20,
        sa_iterations: int = 3,
        sa_initial_temp: float = 0.05,
        sa_cooling: float = 0.96,
        sa_min_temp: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.beam_width = max(1, int(beam_width))
        self.beam_branch_factor = max(1, int(beam_branch_factor))
        self.beam_levels = max(1, int(beam_levels))
        self.beam_patience = max(1, int(beam_patience))
        self.ig_iterations = max(1, int(ig_iterations))
        self.ig_destroy_ratio = min(1.0, max(0.01, float(ig_destroy_ratio)))
        self.ig_noise_sigma = max(1e-6, float(ig_noise_sigma))
        self.sa_iterations = max(1, int(sa_iterations))
        self.sa_initial_temp = max(1e-6, float(sa_initial_temp))
        self.sa_cooling = min(0.999, max(0.50, float(sa_cooling)))
        self.sa_min_temp = max(1e-8, float(sa_min_temp))

    def _make_random_solution(self, rng: np.random.RandomState) -> ReschedulePrioritySolution:
        return ReschedulePrioritySolution(
            task_priority=rng.normal(0.0, 0.10, size=self.env.num_tasks),
            station_priority=rng.normal(0.0, 0.10, size=(self.env.num_tasks, self.env.num_stations)),
            worker_priority=rng.normal(0.0, 0.10, size=(self.env.num_tasks, self.env.num_workers)),
        )

    def _seed_solution(self, kind: str, rng: np.random.RandomState) -> ReschedulePrioritySolution:
        solution = self._make_random_solution(rng)
        if self.use_static_cache and self.static_context is not None:
            durations = self.static_context.durations
            releases = self.static_context.release_times
            baseline_starts = np.asarray(
                [
                    self._baseline_start(task_id)
                    for task_id in range(self.env.num_tasks)
                ],
                dtype=float,
            )
        else:
            durations = np.asarray([_duration(self.env, task_id) for task_id in range(self.env.num_tasks)], dtype=float)
            releases = np.asarray(
                [
                    float(self.env.task_material_ready[task_id]) if hasattr(self.env, "task_material_ready") else 0.0
                    for task_id in range(self.env.num_tasks)
                ],
                dtype=float,
            )
            baseline_starts = np.asarray(
                [
                    float(self.env.baseline_schedule.tasks[task_id].start)
                    if self.env.baseline_schedule and task_id in self.env.baseline_schedule.tasks
                    else float(task_id)
                    for task_id in range(self.env.num_tasks)
                ],
                dtype=float,
            )
        critical = _normalize(self.critical_tail)
        short_first = 1.0 - _normalize(durations)
        long_first = _normalize(durations)
        early_first = 1.0 - _normalize(baseline_starts)
        release_first = 1.0 - _normalize(releases)

        if kind == "cpm":
            solution.task_priority += 1.20 * critical + 0.30 * early_first
        elif kind == "spt":
            solution.task_priority += 1.00 * short_first + 0.20 * release_first
        elif kind == "lpt":
            solution.task_priority += 0.90 * long_first + 0.30 * critical
        elif kind == "stability":
            solution.task_priority += 1.00 * early_first
        elif kind == "takt":
            solution.task_priority += 0.80 * critical + 0.30 * long_first + 0.20 * release_first
        else:
            solution.task_priority += rng.normal(0.0, 0.20, size=self.env.num_tasks)

        if self.env.baseline_schedule:
            for task_idx in range(self.env.num_tasks):
                base_station = self._baseline_station(task_idx)
                if 0 <= base_station < self.env.num_stations:
                    solution.station_priority[task_idx, base_station] += 1.0
                for worker_id in self._baseline_team(task_idx):
                    worker_idx = int(worker_id)
                    if 0 <= worker_idx < self.env.num_workers:
                        solution.worker_priority[task_idx, worker_idx] += 1.0
        return solution

    def _initial_pool(self, count: int) -> list[ReschedulePrioritySolution]:
        seed_kinds = ["cpm", "takt", "stability", "spt", "lpt", "random"]
        pool: list[ReschedulePrioritySolution] = []
        for idx in range(max(1, int(count))):
            kind = seed_kinds[idx % len(seed_kinds)]
            rng = np.random.RandomState(self.seed + 10_000 + idx)
            pool.append(self._seed_solution(kind, rng))
        return pool

    def _mutate_solution(
        self,
        solution: ReschedulePrioritySolution,
        rng: np.random.RandomState,
        *,
        sigma: float = 0.20,
        task_rate: float = 0.05,
        station_rate: float = 0.03,
        worker_rate: float = 0.02,
    ) -> ReschedulePrioritySolution:
        mutated = solution.clone()
        task_mask = rng.rand(self.env.num_tasks) < float(task_rate)
        station_mask = rng.rand(self.env.num_tasks, self.env.num_stations) < float(station_rate)
        worker_mask = rng.rand(self.env.num_tasks, self.env.num_workers) < float(worker_rate)
        if not task_mask.any():
            task_mask[int(rng.randint(0, self.env.num_tasks))] = True
        mutated.task_priority[task_mask] += rng.normal(0.0, sigma, size=int(task_mask.sum()))
        mutated.station_priority[station_mask] += rng.normal(0.0, sigma, size=int(station_mask.sum()))
        mutated.worker_priority[worker_mask] += rng.normal(0.0, sigma, size=int(worker_mask.sum()))
        return mutated

    def _destroy_repair_solution(
        self,
        solution: ReschedulePrioritySolution,
        rng: np.random.RandomState,
    ) -> ReschedulePrioritySolution:
        repaired = solution.clone()
        if self.use_static_cache and self.static_context is not None:
            movable = np.where(self.static_context.movable_mask)[0].astype(int)
        else:
            movable = np.asarray(
                [
                    task_id
                    for task_id in range(self.env.num_tasks)
                    if (
                        not self.env.baseline_schedule
                        or task_id not in self.env.baseline_schedule.tasks
                        or self.env.baseline_schedule.tasks[task_id].start > self.scenario.start_time + 1e-9
                    )
                ],
                dtype=int,
            )
        if movable.size == 0:
            movable = np.arange(self.env.num_tasks, dtype=int)
        destroy_count = max(1, int(math.ceil(movable.size * self.ig_destroy_ratio)))
        weights = _normalize(self.critical_tail[movable]) + 1e-3
        weights = weights / float(np.sum(weights))
        picked = rng.choice(movable, size=min(destroy_count, movable.size), replace=False, p=weights)
        repaired.task_priority[picked] += rng.normal(0.0, self.ig_noise_sigma, size=picked.size)
        repaired.station_priority[picked, :] += rng.normal(
            0.0, self.ig_noise_sigma, size=(picked.size, self.env.num_stations)
        )
        repaired.worker_priority[picked, :] += rng.normal(
            0.0, self.ig_noise_sigma, size=(picked.size, self.env.num_workers)
        )
        return repaired

    def _select_team_for_solution(
        self,
        task_id: int,
        station_id: int,
        worker_mask: np.ndarray,
        solution: ReschedulePrioritySolution,
    ) -> list[int] | None:
        candidates = self._valid_workers(task_id, station_id, worker_mask)
        demand = self._task_demand(task_id)
        if len(candidates) < demand:
            return None
        base_team = self._baseline_team(task_id)
        candidates.sort(
            key=lambda worker_id: (
                0.0 if int(worker_id) in base_team else self.baseline_team_preference,
                -float(solution.worker_priority[int(task_id), int(worker_id)]),
                float(self.env.worker_free_time[int(worker_id)])
                - (0.01 if int(self.env.worker_locks[int(worker_id)]) == int(station_id) + 1 else 0.0),
                int(worker_id),
            )
        )
        return candidates[:demand]

    def _action_score_for_solution(
        self,
        task_id: int,
        station_id: int,
        team: list[int],
        solution: ReschedulePrioritySolution,
    ) -> float:
        start_time, finish_time = self._estimate_action(task_id, station_id, team)
        baseline = self.env.baseline_schedule
        has_base = self._has_baseline_task(task_id)
        takt = max(1e-6, float(baseline.makespan if baseline else max(1.0, self.env.ideal_makespan)))
        ideal_load = max(1.0, float(self.env.ideal_station_load))
        priority = float(solution.task_priority[int(task_id)])
        station_bias = float(solution.station_priority[int(task_id), int(station_id)])
        worker_bias = float(np.mean([solution.worker_priority[int(task_id), int(w)] for w in team])) if team else 0.0
        score = -self.search_priority_weight * priority
        score -= 0.05 * station_bias
        score -= 0.05 * worker_bias
        score += self.makespan_weight * (finish_time / takt)
        if station_id >= 0:
            projected_load = float(self.env.station_loads[int(station_id)]) + self._task_duration(task_id) * len(team)
            score += self.balance_weight * (projected_load / ideal_load)
        score += self.takt_weight * max(0.0, finish_time - takt) / takt
        if has_base:
            score += self.start_stability_weight * abs(start_time - self._baseline_start(task_id)) / takt
            score += self.station_stability_weight * float(int(station_id) != self._baseline_station(task_id))
            base_team = self._baseline_team(task_id)
            overlap = len(base_team.intersection(set(int(w) for w in team)))
            denom = max(1, max(len(base_team), len(team)))
            score += self.team_stability_weight * (1.0 - overlap / denom)
        return float(score)

    def _select_action_for_solution(self, solution: ReschedulePrioritySolution) -> tuple[int, int, list[int]] | None:
        task_mask_raw, station_mask_raw, worker_mask_raw = self.env.get_masks()
        task_mask = _to_numpy_bool(task_mask_raw)
        station_mask = _to_numpy_bool(station_mask_raw)
        worker_mask = _to_numpy_bool(worker_mask_raw)
        if task_mask.all():
            return None

        best_action: tuple[int, int, list[int]] | None = None
        best_score = float("inf")
        for task_id in np.where(~task_mask)[0].tolist():
            for station_id in np.where(~station_mask[int(task_id)])[0].tolist():
                team = self._select_team_for_solution(int(task_id), int(station_id), worker_mask, solution)
                if team is None:
                    continue
                score = self._action_score_for_solution(int(task_id), int(station_id), team, solution)
                if score < best_score:
                    best_score = score
                    best_action = (int(task_id), int(station_id), [int(w) for w in team])
        return best_action

    def _decode_solution(self, solution: ReschedulePrioritySolution) -> RescheduleRuleResult:
        self._reset_env()
        start_wall = time.time()
        done = False
        total_reward = 0.0
        invalid_step_count = 0
        max_steps = max(1, self.env.num_tasks * 3)
        for _ in range(max_steps):
            if done:
                break
            action = self._select_action_for_solution(solution)
            if action is None:
                if self.env.try_wait_for_resources():
                    continue
                break
            _obs, reward, done, info = self.env.step(action)
            total_reward += float(reward)
            if info.get("invalid_action", False):
                invalid_step_count += 1
                if self.verbose:
                    print(f"[{self.method_name}] invalid action: {info}")
                break
        return self._build_result(
            reward=total_reward,
            invalid_step_count=invalid_step_count,
            duration_sec=time.time() - start_wall,
        )

    def _fitness(self, solution: ReschedulePrioritySolution) -> float:
        result = self._decode_solution(solution)
        return float(result.constraint_metrics.get("selection_score", 1.0e12))

    def run(self) -> RescheduleRuleResult:
        solution = self._seed_solution("cpm", np.random.RandomState(self.seed + 99))
        return self._decode_solution(solution)


class BeamSearchRepairRule(PrioritySearchRepairRule):
    method_name = "BeamSearchRepair"
    makespan_weight = 0.35
    balance_weight = 0.08
    takt_weight = 0.35
    start_stability_weight = 0.20
    station_stability_weight = 0.18
    team_stability_weight = 0.15

    def run(self) -> RescheduleRuleResult:
        beam = [(self._fitness(sol), sol) for sol in self._initial_pool(self.beam_width)]
        beam.sort(key=lambda item: item[0])
        best_score, best_solution = beam[0][0], beam[0][1].clone()
        rng = np.random.RandomState(self.seed + 20_000)
        stale_levels = 0

        for level in range(self.beam_levels):
            candidates = list(beam)
            sigma = max(0.04, 0.25 * (0.85**level))
            for _score, solution in beam:
                for _ in range(self.beam_branch_factor):
                    child = self._mutate_solution(
                        solution,
                        rng,
                        sigma=sigma,
                        task_rate=0.05,
                        station_rate=0.03,
                        worker_rate=0.02,
                    )
                    candidates.append((self._fitness(child), child))
            candidates.sort(key=lambda item: item[0])
            beam = [(score, sol.clone()) for score, sol in candidates[: self.beam_width]]
            if beam[0][0] + 1e-12 < best_score:
                best_score, best_solution = beam[0][0], beam[0][1].clone()
                stale_levels = 0
            else:
                stale_levels += 1
            if stale_levels >= self.beam_patience:
                break
        return self._decode_solution(best_solution)


class IteratedGreedyRepairRule(PrioritySearchRepairRule):
    method_name = "IteratedGreedyRepair"
    makespan_weight = 0.30
    balance_weight = 0.08
    takt_weight = 0.35
    start_stability_weight = 0.25
    station_stability_weight = 0.18
    team_stability_weight = 0.15

    def run(self) -> RescheduleRuleResult:
        rng = np.random.RandomState(self.seed + 30_000)
        current = self._seed_solution("cpm", rng)
        current_score = self._fitness(current)
        best_solution = current.clone()
        best_score = current_score

        for idx in range(self.ig_iterations):
            candidate = self._destroy_repair_solution(current, rng)
            if idx % 3 == 0:
                candidate = self._mutate_solution(
                    candidate,
                    rng,
                    sigma=self.ig_noise_sigma * 0.75,
                    task_rate=0.04,
                    station_rate=0.02,
                    worker_rate=0.02,
                )
            candidate_score = self._fitness(candidate)
            if candidate_score <= current_score:
                current, current_score = candidate, candidate_score
            if candidate_score < best_score:
                best_solution, best_score = candidate.clone(), candidate_score
        return self._decode_solution(best_solution)


class SimulatedAnnealingRepairRule(PrioritySearchRepairRule):
    method_name = "SimulatedAnnealingRepair"
    makespan_weight = 0.30
    balance_weight = 0.08
    takt_weight = 0.35
    start_stability_weight = 0.25
    station_stability_weight = 0.18
    team_stability_weight = 0.15

    def run(self) -> RescheduleRuleResult:
        rng = np.random.RandomState(self.seed + 40_000)
        current = self._seed_solution("takt", rng)
        current_score = self._fitness(current)
        best_solution = current.clone()
        best_score = current_score
        temperature = self.sa_initial_temp

        for idx in range(self.sa_iterations):
            sigma = max(0.03, 0.20 * (0.98**idx))
            candidate = self._mutate_solution(
                current,
                rng,
                sigma=sigma,
                task_rate=0.04,
                station_rate=0.02,
                worker_rate=0.02,
            )
            candidate_score = self._fitness(candidate)
            delta = candidate_score - current_score
            if delta <= 0.0:
                accept = True
            elif current_score >= 1.0e8 and candidate_score >= 1.0e8:
                accept = candidate_score < current_score
            else:
                scale = max(1.0, abs(current_score))
                accept = rng.rand() < math.exp(-delta / max(self.sa_min_temp, temperature * scale))
            if accept:
                current, current_score = candidate, candidate_score
            if current_score < best_score:
                best_solution, best_score = current.clone(), current_score
            temperature = max(self.sa_min_temp, temperature * self.sa_cooling)
        return self._decode_solution(best_solution)


def rule_registry() -> dict[str, Type[RescheduleRuleScheduler]]:
    return {
        "NoReschedule": NoRescheduleRule,
        "SPTRepair": SPTRepairRule,
        "LPTRepair": LPTRepairRule,
        "EDDRepair": EDDRepairRule,
        "MSLRepair": MSLRepairRule,
        "CPMRepair": CPMRepairRule,
        "RandomRepair": RandomRepairRule,
        "ReleaseAwareRepair": ReleaseAwareRepairRule,
        "BottleneckSkillRepair": BottleneckSkillRepairRule,
        "TaktAwareRepair": TaktAwareRepairRule,
        "StabilityAwareRepair": StabilityAwareRepairRule,
        "HybridCPMStabilityRepair": HybridCPMStabilityRepairRule,
        "FullRescheduleCPM": FullRescheduleCPMRule,
        "BeamSearchRepair": BeamSearchRepairRule,
        "BeamSearch": BeamSearchRepairRule,
        "Beam": BeamSearchRepairRule,
        "IteratedGreedyRepair": IteratedGreedyRepairRule,
        "IteratedGreedy": IteratedGreedyRepairRule,
        "DestroyRepair": IteratedGreedyRepairRule,
        "IG": IteratedGreedyRepairRule,
        "SimulatedAnnealingRepair": SimulatedAnnealingRepairRule,
        "SimulatedAnnealing": SimulatedAnnealingRepairRule,
        "SA": SimulatedAnnealingRepairRule,
    }


DEFAULT_RULE_METHODS: tuple[str, ...] = (
    "NoReschedule",
    "SPTRepair",
    "LPTRepair",
    "EDDRepair",
    "MSLRepair",
    "CPMRepair",
    "ReleaseAwareRepair",
    "BottleneckSkillRepair",
    "TaktAwareRepair",
    "StabilityAwareRepair",
    "HybridCPMStabilityRepair",
    "FullRescheduleCPM",
    "RandomRepair",
)


__all__ = [
    "DEFAULT_RULE_METHODS",
    "RescheduleRuleResult",
    "RescheduleRuleScheduler",
    "rule_registry",
]
