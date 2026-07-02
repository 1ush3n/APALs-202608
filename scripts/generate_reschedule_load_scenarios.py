# -*- coding: utf-8 -*-
"""生成固定种子的 APAL 重调度低/中/高负载扰动场景库。"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import configs
from runtime.hydra_config import (
    ExtraArgument,
    HydraCliError,
    hydra_help,
    initialize_hydra_runtime,
    should_show_help,
)
from runtime.paths import resolve_workspace_path
from runtime.reschedule_eval import ensure_reschedule_baseline_available
from utils.reschedule import (
    BaselineSchedule,
    RescheduleScenario,
    load_baseline_schedule,
)


@dataclass(frozen=True)
class LoadLevelSpec:
    """单个重调度扰动等级的抽样参数。"""

    name: str
    start_min_ratio: float
    start_max_ratio: float
    task_prob: float
    delay_min: float
    delay_max: float


DEFAULT_LEVELS: tuple[LoadLevelSpec, ...] = (
    LoadLevelSpec("low", 0.15, 0.35, 0.05, 5.0, 15.0),
    LoadLevelSpec("medium", 0.30, 0.55, 0.10, 10.0, 35.0),
    LoadLevelSpec("high", 0.45, 0.70, 0.18, 20.0, 60.0),
)


GENERATOR_EXTRA_ARGS = {
    "baseline_path": ExtraArgument(default=None, help="baseline 调度 CSV；缺省使用配置中的重调度 baseline"),
    "output_path": ExtraArgument(default=None, help="输出场景 CSV；缺省写入 data/reschedule_scenarios"),
    "metadata_path": ExtraArgument(default=None, help="输出元数据 JSON；缺省与 CSV 同名"),
    "seed": ExtraArgument(default=20260701, help="固定随机种子"),
    "scenarios_per_level": ExtraArgument(default=20, help="每个扰动等级生成的场景数量"),
}


def _scenario_stats(
    baseline: BaselineSchedule,
    *,
    scenario_id: str,
    level: str,
    scenario: RescheduleScenario,
) -> dict[str, Any]:
    frozen_count = sum(1 for task in baseline.tasks.values() if task.start <= scenario.start_time + 1e-9)
    delayed_values = list(scenario.task_release_times.values())
    delay_offsets = []
    for task_id, release_time in scenario.task_release_times.items():
        base = baseline.tasks[int(task_id)]
        delay_offsets.append(max(0.0, float(release_time) - max(float(base.start), float(scenario.start_time))))
    eligible_delay_count = sum(
        1
        for task in baseline.tasks.values()
        if task.start > scenario.start_time + 1e-9 and task.duration > 1e-8 and len(task.team) > 0
    )
    return {
        "scenario_id": scenario_id,
        "level": level,
        "reschedule_start_time": float(scenario.start_time),
        "frozen_task_count": int(frozen_count),
        "movable_task_count": int(max(0, len(baseline.tasks) - frozen_count)),
        "eligible_delay_task_count": int(eligible_delay_count),
        "delayed_task_count": int(len(scenario.task_release_times)),
        "delay_mean_h": float(np.mean(delay_offsets)) if delay_offsets else 0.0,
        "delay_max_h": float(np.max(delay_offsets)) if delay_offsets else 0.0,
        "release_time_max_h": float(np.max(delayed_values)) if delayed_values else float(scenario.start_time),
    }


def _eligible_delay_tasks(baseline: BaselineSchedule, start_time: float) -> list:
    """只允许真实可执行工序进入物料延迟扰动，不延迟 0 工时虚拟汇聚节点。"""

    return [
        task
        for task in baseline.tasks.values()
        if task.start > float(start_time) + 1e-9
        and float(task.duration) > 1e-8
        and len(task.team) > 0
    ]


def _sample_task_delay_scenario(
    baseline: BaselineSchedule,
    *,
    rng: np.random.RandomState,
    spec: LoadLevelSpec,
) -> RescheduleScenario:
    takt = max(1e-6, baseline.makespan)
    lo = max(0.0, float(spec.start_min_ratio)) * takt
    hi = max(lo, float(spec.start_max_ratio) * takt)
    start_time = float(rng.uniform(lo, hi))

    release_times: dict[int, float] = {}
    for task in _eligible_delay_tasks(baseline, start_time):
        if rng.rand() < float(spec.task_prob):
            delay = float(rng.uniform(spec.delay_min, spec.delay_max))
            release_times[int(task.task_id)] = max(float(task.start), start_time + delay)
    return RescheduleScenario(start_time=start_time, task_release_times=release_times)


def _ensure_nonempty_delay(
    baseline: BaselineSchedule,
    scenario: RescheduleScenario,
    *,
    spec: LoadLevelSpec,
    rng: np.random.RandomState,
) -> RescheduleScenario:
    if scenario.task_release_times:
        return scenario
    movable = _eligible_delay_tasks(baseline, scenario.start_time)
    if not movable:
        return scenario
    picked = movable[int(rng.randint(0, len(movable)))]
    delay = float(rng.uniform(spec.delay_min, spec.delay_max))
    release_time = max(float(picked.start), float(scenario.start_time) + delay)
    return RescheduleScenario(
        start_time=float(scenario.start_time),
        task_release_times={int(picked.task_id): float(release_time)},
    )


def generate_load_level_scenarios(
    baseline: BaselineSchedule,
    *,
    seed: int = 20260701,
    scenarios_per_level: int = 20,
    levels: tuple[LoadLevelSpec, ...] = DEFAULT_LEVELS,
) -> tuple[list[tuple[str, RescheduleScenario]], list[dict[str, Any]]]:
    """按固定种子生成低/中/高三档任务延迟重调度场景。"""

    scenario_count = max(1, int(scenarios_per_level))
    scenarios: list[tuple[str, RescheduleScenario]] = []
    stats: list[dict[str, Any]] = []
    for level_idx, spec in enumerate(levels):
        for idx in range(scenario_count):
            rng = np.random.RandomState(int(seed) + level_idx * 100_000 + idx)
            scenario = _sample_task_delay_scenario(
                baseline,
                rng=rng,
                spec=spec,
            )
            scenario = _ensure_nonempty_delay(baseline, scenario, spec=spec, rng=rng)
            scenario_id = f"{spec.name}_{idx:03d}"
            scenarios.append((scenario_id, scenario))
            stats.append(_scenario_stats(baseline, scenario_id=scenario_id, level=spec.name, scenario=scenario))
    return scenarios, stats


def write_scenario_library(
    *,
    baseline_path: Path,
    output_path: Path,
    metadata_path: Path,
    seed: int = 20260701,
    scenarios_per_level: int = 20,
    levels: tuple[LoadLevelSpec, ...] = DEFAULT_LEVELS,
) -> dict[str, Any]:
    """写出兼容现有重调度读取器的场景 CSV 和可审计元数据。"""

    baseline = load_baseline_schedule(baseline_path)
    scenarios, stats = generate_load_level_scenarios(
        baseline,
        seed=int(seed),
        scenarios_per_level=int(scenarios_per_level),
        levels=levels,
    )

    rows: list[dict[str, Any]] = []
    stats_by_id = {row["scenario_id"]: row for row in stats}
    for scenario_id, scenario in scenarios:
        stat = stats_by_id[scenario_id]
        common = {
            "scenario_id": scenario_id,
            "level": stat["level"],
            "reschedule_start_time": float(scenario.start_time),
            "delayed_task_count": int(stat["delayed_task_count"]),
            "frozen_task_count": int(stat["frozen_task_count"]),
            "movable_task_count": int(stat["movable_task_count"]),
            "eligible_delay_task_count": int(stat["eligible_delay_task_count"]),
        }
        if scenario.task_release_times:
            for task_id, release_time in sorted(scenario.task_release_times.items()):
                rows.append({**common, "TaskID": int(task_id), "release_time": float(release_time)})
        else:
            rows.append({**common, "TaskID": -1, "release_time": float(scenario.start_time)})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)

    aggregate = []
    for spec in levels:
        level_stats = [row for row in stats if row["level"] == spec.name]
        aggregate.append(
            {
                "level": spec.name,
                "scenario_count": int(len(level_stats)),
                "avg_delayed_task_count": float(np.mean([row["delayed_task_count"] for row in level_stats])),
                "avg_frozen_task_count": float(np.mean([row["frozen_task_count"] for row in level_stats])),
                "avg_delay_mean_h": float(np.mean([row["delay_mean_h"] for row in level_stats])),
                "max_delay_h": float(np.max([row["delay_max_h"] for row in level_stats])),
            }
        )

    metadata = {
        "seed": int(seed),
        "scenarios_per_level": int(scenarios_per_level),
        "baseline_path": str(baseline_path.resolve()),
        "output_path": str(output_path.resolve()),
        "levels": [asdict(spec) for spec in levels],
        "scenario_count": int(len(scenarios)),
        "aggregate_by_level": aggregate,
        "scenario_stats": stats,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def _default_output_path(seed: int) -> Path:
    return PROJECT_ROOT / "data" / "reschedule_scenarios" / f"task_delay_load_grid_seed{int(seed)}.csv"


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if should_show_help(raw_args):
        print(hydra_help(GENERATOR_EXTRA_ARGS))
        return 0
    try:
        args = initialize_hydra_runtime(
            raw_args,
            target=configs,
            project_root=PROJECT_ROOT,
            default_experiment="reschedule_task_delay",
            extra_arguments=GENERATOR_EXTRA_ARGS,
            create_run_context=False,
        )
    except (HydraCliError, KeyError, ValueError, RuntimeError) as exc:
        print(f"[CLI] {exc}", file=sys.stderr)
        return 2

    seed = int(args.seed)
    baseline_path = (
        resolve_workspace_path(args.baseline_path)
        if args.baseline_path
        else ensure_reschedule_baseline_available(configs)
    )
    if baseline_path is None:
        print("[Scenario] enable_reschedule_mode=True 时才能生成重调度场景。", file=sys.stderr)
        return 2

    output_path = resolve_workspace_path(args.output_path) if args.output_path else _default_output_path(seed)
    metadata_path = (
        resolve_workspace_path(args.metadata_path)
        if args.metadata_path
        else output_path.with_suffix(".metadata.json")
    )
    metadata = write_scenario_library(
        baseline_path=Path(baseline_path),
        output_path=output_path,
        metadata_path=metadata_path,
        seed=seed,
        scenarios_per_level=int(args.scenarios_per_level),
    )
    print(
        json.dumps(
            {
                "scenario_count": metadata["scenario_count"],
                "output_path": metadata["output_path"],
                "metadata_path": str(metadata_path.resolve()),
                "aggregate_by_level": metadata["aggregate_by_level"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
