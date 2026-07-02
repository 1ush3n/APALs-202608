from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BaselineTask:
    task_id: int
    station_id: int
    team: tuple[int, ...]
    start: float
    end: float
    duration: float


@dataclass(frozen=True)
class BaselineSchedule:
    tasks: dict[int, BaselineTask]
    makespan: float


@dataclass(frozen=True)
class RescheduleScenario:
    start_time: float
    task_release_times: dict[int, float]


@dataclass(frozen=True)
class RescheduleScore:
    """重调度综合评分结果，score 越小越好。"""

    eligible: bool
    score: float
    selection_score: float
    terms: dict[str, float]


@dataclass(frozen=True)
class RescheduleLoadLevelSpec:
    """重调度任务延迟负载等级。"""

    name: str
    start_min_ratio: float
    start_max_ratio: float
    task_prob: float
    delay_min: float
    delay_max: float


DEFAULT_RESCHEDULE_LOAD_LEVELS: tuple[RescheduleLoadLevelSpec, ...] = (
    RescheduleLoadLevelSpec("low", 0.15, 0.35, 0.05, 5.0, 15.0),
    RescheduleLoadLevelSpec("medium", 0.30, 0.55, 0.10, 10.0, 35.0),
    RescheduleLoadLevelSpec("high", 0.45, 0.70, 0.18, 20.0, 60.0),
)


HARD_CONSTRAINT_KEYS = (
    "frozen_violation_count",
    "release_violation_count",
    "precedence_violation_count",
    "worker_overlap_violation_count",
    "station_slot_violation_count",
    "skill_violation_count",
    "demand_violation_count",
    "duplicate_task_count",
    "missing_task_count",
    "invalid_step_count",
)


def _parse_team(value: Any) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    text = str(value).strip()
    if not text:
        return ()
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = [part.strip() for part in text.strip("[]").split(",") if part.strip()]
    if isinstance(parsed, int):
        return (int(parsed),)
    return tuple(int(v) for v in parsed)


def load_baseline_schedule(path: Path) -> BaselineSchedule:
    if not path.exists():
        raise FileNotFoundError(f"baseline 调度文件不存在: {path}")
    df = pd.read_csv(path)
    required = {"TaskID", "StationID", "Team", "Start", "End", "Duration"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"baseline 调度文件缺少列: {sorted(missing)}")

    tasks: dict[int, BaselineTask] = {}
    for row in df.itertuples(index=False):
        task_id = int(getattr(row, "TaskID"))
        station_1based = int(getattr(row, "StationID"))
        station_id = station_1based - 1 if station_1based > 0 else -1
        start = float(getattr(row, "Start"))
        end = float(getattr(row, "End"))
        duration = float(getattr(row, "Duration"))
        tasks[task_id] = BaselineTask(
            task_id=task_id,
            station_id=station_id,
            team=_parse_team(getattr(row, "Team")),
            start=start,
            end=end,
            duration=duration,
        )

    makespan = max((task.end for task in tasks.values()), default=0.0)
    return BaselineSchedule(tasks=tasks, makespan=float(makespan))


def load_reschedule_scenario(path: Path) -> RescheduleScenario:
    if not path.exists():
        raise FileNotFoundError(f"重调度场景文件不存在: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        start_time = 0.0
        release_times: dict[int, float] = {}
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "reschedule_start_time" in row and row["reschedule_start_time"]:
                    start_time = float(row["reschedule_start_time"])
                task_key = "TaskID" if "TaskID" in row else "task_id"
                release_key = "release_time" if "release_time" in row else "task_release_time"
                release_times[int(row[task_key])] = float(row[release_key])
        return RescheduleScenario(start_time=start_time, task_release_times=release_times)

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("读取 YAML 重调度场景需要安装 PyYAML，或改用 CSV 场景文件。") from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    start_time = float(data.get("reschedule_start_time", data.get("start_time", 0.0)))
    raw_delays = data.get("task_release_times", data.get("operation_release_delay", {})) or {}
    release_times = {int(k): float(v) for k, v in raw_delays.items()}
    return RescheduleScenario(start_time=start_time, task_release_times=release_times)


def load_reschedule_scenarios(path: Path) -> list[tuple[str, RescheduleScenario]]:
    if not path.exists():
        raise FileNotFoundError(f"重调度验证场景文件不存在: {path}")
    if path.suffix.lower() != ".csv":
        return [("scenario_0", load_reschedule_scenario(path))]

    grouped: dict[str, tuple[float, dict[int, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            scenario_id = str(row.get("scenario_id") or row.get("ScenarioID") or "scenario_0")
            start_time = float(row.get("reschedule_start_time") or row.get("start_time") or 0.0)
            task_key = "TaskID" if "TaskID" in row else "task_id"
            release_key = "release_time" if "release_time" in row else "task_release_time"
            if task_key not in row or release_key not in row:
                raise ValueError(f"重调度验证场景第 {row_idx} 行缺少 TaskID/task_id 或 release_time/task_release_time")
            if scenario_id not in grouped:
                grouped[scenario_id] = (start_time, {})
            task_id = int(row[task_key])
            if task_id >= 0:
                grouped[scenario_id][1][task_id] = float(row[release_key])

    return [
        (scenario_id, RescheduleScenario(start_time=start_time, task_release_times=release_times))
        for scenario_id, (start_time, release_times) in grouped.items()
    ]


def save_reschedule_scenarios(path: Path, scenarios: list[tuple[str, RescheduleScenario]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    for scenario_id, scenario in scenarios:
        if scenario.task_release_times:
            for task_id, release_time in sorted(scenario.task_release_times.items()):
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "reschedule_start_time": float(scenario.start_time),
                        "TaskID": int(task_id),
                        "release_time": float(release_time),
                    }
                )
        else:
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "reschedule_start_time": float(scenario.start_time),
                    "TaskID": -1,
                    "release_time": float(scenario.start_time),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def eligible_delay_tasks(baseline: BaselineSchedule, start_time: float) -> list[BaselineTask]:
    """只允许真实可执行工序进入任务延迟扰动，不延迟 0 工时虚拟汇聚节点。"""

    return [
        task
        for task in baseline.tasks.values()
        if task.start > float(start_time) + 1e-9
        and float(task.duration) > 1e-8
        and len(task.team) > 0
    ]


def sample_task_delay_scenario(
    baseline: BaselineSchedule,
    *,
    rng: np.random.RandomState,
    min_start_ratio: float,
    max_start_ratio: float,
    task_prob: float,
    delay_min: float,
    delay_max: float,
) -> RescheduleScenario:
    takt = max(1e-6, baseline.makespan)
    lo = max(0.0, min_start_ratio) * takt
    hi = max(lo, max_start_ratio * takt)
    start_time = float(rng.uniform(lo, hi))

    release_times: dict[int, float] = {}
    for task in eligible_delay_tasks(baseline, start_time):
        if rng.rand() < task_prob:
            delay = float(rng.uniform(delay_min, delay_max))
            release_times[int(task.task_id)] = max(float(task.start), start_time + delay)
    return RescheduleScenario(start_time=start_time, task_release_times=release_times)


def sample_task_delay_load_scenario(
    baseline: BaselineSchedule,
    *,
    rng: np.random.RandomState,
    levels: tuple[RescheduleLoadLevelSpec, ...] = DEFAULT_RESCHEDULE_LOAD_LEVELS,
    level_name: str | None = None,
    ensure_nonempty: bool = True,
) -> tuple[str, RescheduleScenario]:
    """按 low/medium/high 负载等级采样任务延迟场景。"""

    if not levels:
        raise ValueError("至少需要一个重调度负载等级")
    if level_name:
        matches = [level for level in levels if level.name == level_name]
        if not matches:
            raise ValueError(f"未知重调度负载等级: {level_name}")
        spec = matches[0]
    else:
        spec = levels[int(rng.randint(0, len(levels)))]
    scenario = sample_task_delay_scenario(
        baseline,
        rng=rng,
        min_start_ratio=float(spec.start_min_ratio),
        max_start_ratio=float(spec.start_max_ratio),
        task_prob=float(spec.task_prob),
        delay_min=float(spec.delay_min),
        delay_max=float(spec.delay_max),
    )
    if scenario.task_release_times or not ensure_nonempty:
        return spec.name, scenario
    candidates = eligible_delay_tasks(baseline, scenario.start_time)
    if not candidates:
        return spec.name, scenario
    picked = candidates[int(rng.randint(0, len(candidates)))]
    release_time = max(float(picked.start), float(scenario.start_time) + float(rng.uniform(spec.delay_min, spec.delay_max)))
    return spec.name, RescheduleScenario(
        start_time=float(scenario.start_time),
        task_release_times={int(picked.task_id): float(release_time)},
    )


def calculate_reschedule_lower_bound(
    baseline: BaselineSchedule,
    task_status: np.ndarray,
    task_durations: np.ndarray,
    release_times: np.ndarray,
    *,
    current_time: float,
    num_stations: int,
    station_slots: float,
) -> float:
    frozen_finish = max((task.end for task in baseline.tasks.values() if task.start <= current_time + 1e-9), default=current_time)
    remaining_mask = task_status <= 1
    remaining_work = float(np.sum(task_durations[remaining_mask]))
    workload_lb = current_time + remaining_work / max(1.0, num_stations * station_slots)
    release_lb = float(np.max(release_times[remaining_mask])) if np.any(remaining_mask) else current_time
    return max(float(frozen_finish), workload_lb, release_lb)


def calculate_stability_metrics(
    baseline: BaselineSchedule,
    assigned_tasks: Iterable[tuple[int, int, list[int], float, float]],
    *,
    current_time: float,
) -> dict[str, float]:
    start_devs: list[float] = []
    station_changes = 0
    team_changes = 0
    movable_count = 0
    for task_id, station_id, team, start, _end in assigned_tasks:
        base = baseline.tasks.get(int(task_id))
        if base is None or base.start <= current_time + 1e-9:
            continue
        movable_count += 1
        start_devs.append(abs(float(start) - base.start))
        station_changes += int(int(station_id) != base.station_id)
        team_changes += int(set(int(w) for w in team) != set(base.team))

    denom = max(1, movable_count)
    return {
        "start_deviation_mean_h": float(np.mean(start_devs)) if start_devs else 0.0,
        "station_change_rate": float(station_changes / denom),
        "team_change_rate": float(team_changes / denom),
        "movable_count": float(movable_count),
    }


def calculate_reschedule_composite_score(
    *,
    makespan: float,
    balance_std: float,
    constraint_metrics: dict[str, float],
    config_obj: Any,
    ideal_station_load: float,
) -> RescheduleScore:
    """
    计算 APAL 预测-反应式重调度的统一综合评分。

    score 仅表达目标函数分项；selection_score 在硬约束违规时置为极大值，
    用于 PPO best model 和 GA 个体选优时直接淘汰不可行方案。
    """

    takt_h = max(1e-6, float(constraint_metrics.get("takt_h", 0.0)))
    ideal_load = max(1.0, float(ideal_station_load))
    complete = float(constraint_metrics.get("complete", 0.0)) >= 1.0 - 1e-9
    has_violation = any(float(constraint_metrics.get(key, 0.0)) > 0.0 for key in HARD_CONSTRAINT_KEYS)
    eligible = bool(complete and not has_violation)

    terms = {
        "score_makespan": float(makespan) / takt_h,
        "score_balance": float(getattr(config_obj, "r_coef_std", 0.0)) * float(balance_std) / ideal_load,
        "score_takt_violation": float(getattr(config_obj, "reschedule_takt_violation_weight", 1.0))
        * float(constraint_metrics.get("takt_violation_h", 0.0))
        / takt_h,
        "score_start_stability": float(getattr(config_obj, "reschedule_stability_start_weight", 0.20))
        * float(constraint_metrics.get("start_deviation_mean_h", 0.0))
        / takt_h,
        "score_station_change": float(getattr(config_obj, "reschedule_stability_station_weight", 0.10))
        * float(constraint_metrics.get("station_change_rate", 0.0)),
        "score_team_change": float(getattr(config_obj, "reschedule_stability_team_weight", 0.05))
        * float(constraint_metrics.get("team_change_rate", 0.0)),
    }
    score = float(sum(terms.values()))
    selection_score = score if eligible else 1.0e9 + score
    return RescheduleScore(eligible=eligible, score=score, selection_score=selection_score, terms=terms)
