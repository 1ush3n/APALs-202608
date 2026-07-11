"""APAL 排程硬约束的单一事实来源。

本模块不依赖环境实现，供动作掩码、环境边界、初始排程校验和重调度
评估共同使用。所有站位均采用环境内部的 0-based 编号；虚拟节点站位为 -1。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np


Assignment = tuple[int, int, Sequence[int], float, float]


@dataclass(frozen=True)
class ScheduleValidationReport:
    """统一排程合法性报告。"""

    violations: dict[str, int]
    examples: dict[str, list[dict[str, object]]] = field(default_factory=dict)

    @property
    def is_legal(self) -> bool:
        return all(int(value) == 0 for value in self.violations.values())


@dataclass(frozen=True)
class ConstraintEngine:
    """不可变的 APAL 工艺与站位硬约束模型。"""

    num_tasks: int
    num_stations: int
    predecessors: tuple[tuple[int, ...], ...]
    physical_predecessors: tuple[tuple[int, ...], ...]
    physical_mask: np.ndarray
    fixed_stations: np.ndarray
    max_allowed_stations: np.ndarray

    @classmethod
    def build(
        cls,
        *,
        num_tasks: int,
        num_stations: int,
        edges: np.ndarray,
        durations: Sequence[float],
        fixed_stations: Sequence[int],
        max_allowed_stations: Sequence[int] | None = None,
        tolerance: float = 1e-5,
    ) -> "ConstraintEngine":
        """由完整层级 DAG 构造压缩后的物理工序约束图。"""
        edge_array = np.asarray(edges, dtype=np.int64)
        if edge_array.size == 0:
            edge_array = np.empty((2, 0), dtype=np.int64)
        if edge_array.ndim != 2 or edge_array.shape[0] != 2:
            raise ValueError(f"edges 必须为 [2, E]，实际为 {edge_array.shape}")

        duration_array = np.asarray(durations, dtype=float).reshape(-1)
        fixed_array = np.asarray(fixed_stations, dtype=np.int64).reshape(-1).copy()
        if duration_array.shape[0] != num_tasks or fixed_array.shape[0] != num_tasks:
            raise ValueError("duration/fixed_station 数量必须与 num_tasks 一致")
        physical_mask = duration_array > float(tolerance)

        predecessors: list[list[int]] = [[] for _ in range(num_tasks)]
        successors: list[list[int]] = [[] for _ in range(num_tasks)]
        for src_raw, dst_raw in edge_array.T:
            src, dst = int(src_raw), int(dst_raw)
            if not (0 <= src < num_tasks and 0 <= dst < num_tasks):
                raise ValueError(f"工艺边越界: ({src}, {dst})")
            predecessors[dst].append(src)
            successors[src].append(dst)

        # 虚拟层级节点若声明固定站位，该约束传递到其首层物理后继。
        effective_fixed = fixed_array.copy()
        for virtual_id in np.where((~physical_mask) & (fixed_array >= 0))[0].tolist():
            station_id = int(fixed_array[virtual_id])
            stack = list(successors[virtual_id])
            visited: set[int] = set()
            while stack:
                node = int(stack.pop())
                if node in visited:
                    continue
                visited.add(node)
                if physical_mask[node]:
                    existing = int(effective_fixed[node])
                    if existing >= 0 and existing != station_id:
                        raise ValueError(
                            f"工序 {node} 的固定站位 {existing + 1} 与虚拟节点 "
                            f"{virtual_id} 传播的站位 {station_id + 1} 冲突"
                        )
                    effective_fixed[node] = station_id
                else:
                    stack.extend(successors[node])

        # 对每个物理工序，穿透连续虚拟节点，找到最近的物理前驱边界。
        compressed: list[tuple[int, ...]] = []
        for task_id in range(num_tasks):
            if not physical_mask[task_id]:
                compressed.append(())
                continue
            frontier: set[int] = set()
            stack = list(predecessors[task_id])
            visited: set[int] = set()
            while stack:
                node = int(stack.pop())
                if node in visited:
                    continue
                visited.add(node)
                if physical_mask[node]:
                    frontier.add(node)
                else:
                    stack.extend(predecessors[node])
            compressed.append(tuple(sorted(frontier)))

        if max_allowed_stations is None:
            max_allowed = np.full(num_tasks, num_stations - 1, dtype=np.int64)
        else:
            max_allowed = np.asarray(max_allowed_stations, dtype=np.int64).reshape(-1).copy()
            if max_allowed.shape[0] != num_tasks:
                raise ValueError("max_allowed_stations 数量必须与 num_tasks 一致")

        return cls(
            num_tasks=int(num_tasks),
            num_stations=int(num_stations),
            predecessors=tuple(tuple(sorted(set(items))) for items in predecessors),
            physical_predecessors=tuple(compressed),
            physical_mask=physical_mask.copy(),
            fixed_stations=effective_fixed,
            max_allowed_stations=max_allowed,
        )

    def with_max_allowed_stations(self, values: Sequence[int]) -> "ConstraintEngine":
        max_allowed = np.asarray(values, dtype=np.int64).reshape(-1).copy()
        if max_allowed.shape[0] != self.num_tasks:
            raise ValueError("max_allowed_stations 数量必须与 num_tasks 一致")
        infeasible = np.where(
            self.physical_mask
            & (self.fixed_stations >= 0)
            & (self.fixed_stations > max_allowed)
        )[0]
        if infeasible.size:
            raise ValueError(
                f"固定站位与后继站位上界冲突，工序示例: {infeasible[:5].tolist()}"
            )
        return ConstraintEngine(
            num_tasks=self.num_tasks,
            num_stations=self.num_stations,
            predecessors=self.predecessors,
            physical_predecessors=self.physical_predecessors,
            physical_mask=self.physical_mask.copy(),
            fixed_stations=self.fixed_stations.copy(),
            max_allowed_stations=max_allowed,
        )

    def minimum_station(self, task_id: int, task_station_map: Mapping[int, int]) -> int:
        """返回已完成物理前驱决定的最小合法站位。"""
        stations = [
            int(task_station_map[pred])
            for pred in self.physical_predecessors[int(task_id)]
            if int(task_station_map.get(pred, -1)) >= 0
        ]
        return max(stations, default=0)

    def station_violation(
        self,
        task_id: int,
        station_id: int,
        task_station_map: Mapping[int, int],
    ) -> dict[str, object] | None:
        """校验单个工序的固定站位、范围与物理工艺单调性。"""
        task_id, station_id = int(task_id), int(station_id)
        if not self.physical_mask[task_id]:
            return None if station_id == -1 else {
                "reason": "virtual_task_requires_virtual_station",
                "task_id": task_id,
                "station_id": station_id,
            }
        if station_id < 0 or station_id >= self.num_stations:
            return {"reason": "invalid_station_id", "task_id": task_id, "station_id": station_id}
        fixed = int(self.fixed_stations[task_id])
        if fixed >= 0 and station_id != fixed:
            return {
                "reason": "fixed_station_violation",
                "task_id": task_id,
                "station_id": station_id,
                "required_station": fixed,
            }
        minimum = self.minimum_station(task_id, task_station_map)
        maximum = int(self.max_allowed_stations[task_id])
        if station_id < minimum:
            return {
                "reason": "physical_precedence_station_violation",
                "task_id": task_id,
                "station_id": station_id,
                "minimum_station": minimum,
            }
        if station_id > maximum:
            return {
                "reason": "station_upper_bound_violation",
                "task_id": task_id,
                "station_id": station_id,
                "maximum_station": maximum,
            }
        return None

    def validate_schedule(
        self,
        assignments: Iterable[Assignment],
        *,
        demands: Sequence[int],
        required_skills: Sequence[int],
        worker_skill_matrix: np.ndarray,
        max_slots_per_station: int | Sequence[int],
        tolerance: float = 1e-5,
    ) -> ScheduleValidationReport:
        """对完整排程执行统一的结构、时间、资源与站位合法性校验。"""
        keys = (
            "duplicate_task_count", "missing_task_count", "precedence_violation_count",
            "physical_station_violation_count", "station_range_violation_count",
            "fixed_station_violation_count", "demand_violation_count",
            "worker_range_violation_count", "skill_violation_count",
            "worker_station_binding_violation_count", "worker_overlap_violation_count",
            "station_slot_violation_count", "negative_or_reversed_time_count",
        )
        violations = {key: 0 for key in keys}
        examples = {key: [] for key in keys}

        rows: dict[int, Assignment] = {}
        seen: set[int] = set()
        worker_intervals: dict[int, list[tuple[float, float, int, int]]] = {}
        station_intervals: dict[int, list[tuple[float, float, int]]] = {}
        demand_array = np.asarray(demands, dtype=np.int64).reshape(-1)
        skill_array = np.asarray(required_skills, dtype=np.int64).reshape(-1)
        skills = np.asarray(worker_skill_matrix)

        for raw in assignments:
            task_id, station_id, raw_team, start, end = raw
            task_id, station_id = int(task_id), int(station_id)
            team = tuple(int(worker) for worker in raw_team)
            start, end = float(start), float(end)
            if task_id in seen:
                violations["duplicate_task_count"] += 1
            seen.add(task_id)
            if not 0 <= task_id < self.num_tasks:
                continue
            rows[task_id] = (task_id, station_id, team, start, end)
            if start < -tolerance or end < start - tolerance:
                violations["negative_or_reversed_time_count"] += 1

            if not self.physical_mask[task_id]:
                continue
            fixed = int(self.fixed_stations[task_id])
            if not 0 <= station_id < self.num_stations:
                violations["station_range_violation_count"] += 1
            if fixed >= 0 and station_id != fixed:
                violations["fixed_station_violation_count"] += 1
            if len(team) != max(1, int(demand_array[task_id])) or len(team) != len(set(team)):
                violations["demand_violation_count"] += 1

            required_skill = int(skill_array[task_id])
            for worker_id in team:
                if not 0 <= worker_id < skills.shape[0]:
                    violations["worker_range_violation_count"] += 1
                    continue
                if not 0 <= required_skill < skills.shape[1] or skills[worker_id, required_skill] < 0.5:
                    violations["skill_violation_count"] += 1
                worker_intervals.setdefault(worker_id, []).append((start, end, task_id, station_id))
            if 0 <= station_id < self.num_stations:
                station_intervals.setdefault(station_id, []).append((start, end, task_id))

        violations["missing_task_count"] = len(set(range(self.num_tasks)) - seen)

        for task_id, row in rows.items():
            _task, station_id, _team, start, _end = row
            for pred in self.predecessors[task_id]:
                pred_row = rows.get(pred)
                if pred_row is not None and float(pred_row[4]) > start + tolerance:
                    violations["precedence_violation_count"] += 1
            if self.physical_mask[task_id]:
                for pred in self.physical_predecessors[task_id]:
                    pred_row = rows.get(pred)
                    if pred_row is not None and int(pred_row[1]) > station_id:
                        violations["physical_station_violation_count"] += 1

        for worker_id, intervals in worker_intervals.items():
            positive = sorted(item for item in intervals if item[1] - item[0] > tolerance)
            station_ids = {item[3] for item in positive if item[3] >= 0}
            if len(station_ids) > 1:
                violations["worker_station_binding_violation_count"] += 1
            for previous, current in zip(positive, positive[1:]):
                if previous[1] > current[0] + tolerance:
                    violations["worker_overlap_violation_count"] += 1

        if np.isscalar(max_slots_per_station):
            capacities = np.full(self.num_stations, int(max_slots_per_station), dtype=np.int64)
        else:
            capacities = np.asarray(max_slots_per_station, dtype=np.int64).reshape(-1)
            if capacities.shape[0] != self.num_stations:
                raise ValueError("站位容量数量必须与 num_stations 一致")
        for station_id, intervals in station_intervals.items():
            events: list[tuple[float, int]] = []
            for start, end, _task_id in intervals:
                if end - start > tolerance:
                    events.extend(((start, 1), (end, -1)))
            events.sort(key=lambda item: (item[0], item[1]))
            active = 0
            for _time, delta in events:
                active += delta
                if active > int(capacities[station_id]):
                    violations["station_slot_violation_count"] += 1
                    break

        return ScheduleValidationReport(violations=violations, examples=examples)
