from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import configs, load_training_config
from data_loader import load_data
from environment import AirLineEnv_Graph
from runtime.initial_worker_mapping import apply_initial_worker_mapping


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_optional_config(config_path: str | None, data_path: Path, *, config_obj: Any | None = None) -> None:
    if config_obj is not None:
        apply_initial_worker_mapping(config_obj, data_path, explicit_fields=set())
        return
    if config_path:
        load_training_config([str(_resolve_path(config_path))], target=configs)
    else:
        bucket = PROJECT_ROOT / "conf" / "env" / f"initial_bucket_{data_path.stem}.yaml"
        default = PROJECT_ROOT / "conf" / "env" / "apal_default.yaml"
        paths = [str(default)] if default.exists() else []
        if bucket.exists():
            paths.append(str(bucket))
        if paths:
            load_training_config(paths, target=configs)
    # 与 evaluate_model.py / train_lightning.py 的真实数据集工人数口径一致。
    apply_initial_worker_mapping(configs, data_path, explicit_fields=set())


def _parse_team(raw: Any) -> list[int]:
    if isinstance(raw, list):
        return [int(item) for item in raw]
    try:
        value = ast.literal_eval(str(raw))
    except (SyntaxError, ValueError):
        return []
    if not isinstance(value, list):
        return []
    return [int(item) for item in value]


def _resolve_task_ids(schedule: pd.DataFrame, task_df: pd.DataFrame, mode: str) -> tuple[pd.Series, str, list[int]]:
    if "TaskID" not in schedule.columns:
        raise ValueError("schedule 缺少 TaskID 列")
    raw_ids = schedule["TaskID"].astype(int)
    num_tasks = len(task_df)
    internal_ids = set(int(value) for value in task_df["internal_id"].tolist())

    if mode == "internal":
        mapped = raw_ids
        unknown = [int(value) for value in mapped if int(value) not in internal_ids]
        return mapped, "internal", unknown

    seq_col = "序号"
    if mode == "sequence":
        if seq_col not in task_df.columns:
            raise ValueError("数据文件缺少 序号 列，不能使用 sequence TaskID 模式")
        seq_to_internal = {
            int(row[seq_col]): int(row["internal_id"])
            for _, row in task_df.iterrows()
        }
        mapped = raw_ids.map(seq_to_internal)
        unknown = [int(raw_ids.iloc[idx]) for idx, value in enumerate(mapped) if pd.isna(value)]
        return mapped.astype("Int64"), "sequence", unknown

    internal_unknown = [int(value) for value in raw_ids if int(value) not in internal_ids]
    if not internal_unknown:
        return raw_ids, "internal", []

    if seq_col in task_df.columns:
        seq_to_internal = {
            int(row[seq_col]): int(row["internal_id"])
            for _, row in task_df.iterrows()
        }
        mapped = raw_ids.map(seq_to_internal)
        seq_unknown = [int(raw_ids.iloc[idx]) for idx, value in enumerate(mapped) if pd.isna(value)]
        if not seq_unknown:
            return mapped.astype("Int64"), "sequence", []

    return raw_ids, "internal", internal_unknown


def validate_schedule(
    *,
    data_path: Path,
    schedule_path: Path,
    config_path: str | None = None,
    config_obj: Any | None = None,
    task_id_mode: str = "auto",
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    _load_optional_config(config_path, data_path, config_obj=config_obj)
    raw = load_data(data_path)
    task_df = raw["task_df"].copy().sort_values("internal_id").reset_index(drop=True)
    schedule = pd.read_csv(schedule_path)
    mapped_ids, resolved_mode, unknown_task_ids = _resolve_task_ids(schedule, task_df, task_id_mode)

    env = AirLineEnv_Graph(data_path_or_dir=data_path, seed=int(getattr(configs, "seed", 42)))
    env.reset(randomize_duration=False, randomize_workers=False, seed=int(getattr(configs, "seed", 42)))

    required_columns = {"TaskID", "StationID", "Team", "Start", "End", "Duration"}
    violations: dict[str, int] = {
        "missing_columns": len(required_columns - set(schedule.columns)),
        "row_count_mismatch": int(len(schedule) != int(raw["num_tasks"])),
        "duplicate_task_count": 0,
        "unknown_task_count": len(unknown_task_ids),
        "missing_task_count": 0,
        "negative_or_reversed_time_count": 0,
        "csv_duration_mismatch_count": 0,
        "task_duration_mismatch_count": 0,
        "precedence_violation_count": 0,
        "station_range_violation_count": 0,
        "fixed_station_violation_count": 0,
        "physical_station_violation_count": 0,
        "worker_station_binding_violation_count": 0,
        "demand_violation_count": 0,
        "worker_range_violation_count": 0,
        "skill_violation_count": 0,
        "worker_overlap_violation_count": 0,
        "station_slot_violation_count": 0,
    }
    examples: dict[str, list[dict[str, Any]]] = {
        "duration_mismatch": [],
        "precedence": [],
        "fixed_station": [],
        "demand": [],
        "skill": [],
        "worker_overlap": [],
        "station_slot": [],
        "physical_station": [],
        "worker_station_binding": [],
    }

    operation_ids = set(int(value) for value in task_df["internal_id"].tolist())

    rows: dict[int, dict[str, Any]] = {}
    ratios: list[float] = []
    real_task_count = 0
    scheduled_real_task_count = 0
    assignments: list[tuple[int, int, list[int], float, float]] = []

    for row_idx, row in schedule.iterrows():
        mapped_value = mapped_ids.iloc[row_idx]
        if pd.isna(mapped_value):
            continue
        task_id = int(mapped_value)
        if task_id not in operation_ids:
            continue

        station_csv = int(row["StationID"])
        start = float(row["Start"])
        end = float(row["End"])
        csv_duration = float(row["Duration"])
        actual_duration = end - start
        rows[task_id] = {
            "station_csv": station_csv,
            "start": start,
            "end": end,
            "duration": actual_duration,
        }
        if abs(csv_duration - actual_duration) > tolerance:
            violations["csv_duration_mismatch_count"] += 1

        task = task_df.iloc[task_id]
        base_duration = float(task["duration"])
        is_real = base_duration > tolerance
        if is_real:
            real_task_count += 1
        if is_real and actual_duration > tolerance:
            scheduled_real_task_count += 1


        team = _parse_team(row["Team"])
        station_internal = station_csv - 1 if station_csv > 0 else -1
        assignments.append((task_id, station_internal, team, start, end))
        if is_real:
            expected_env_duration = float(env.calculate_duration(task_id, team, start_time_est=start))
            ratios.append(actual_duration / expected_env_duration if expected_env_duration > 0 else 0.0)
            if abs(actual_duration - expected_env_duration) > tolerance:
                violations["task_duration_mismatch_count"] += 1
                if len(examples["duration_mismatch"]) < 10:
                    examples["duration_mismatch"].append(
                        {
                            "task_id": task_id,
                            "base_csv_duration": base_duration,
                            "expected_environment_duration": expected_env_duration,
                            "schedule_duration": actual_duration,
                        }
                    )

    max_slots = int(getattr(configs, "max_slots_per_station", 3))

    central_report = env.validate_assignments(assignments)
    for key, value in central_report.violations.items():
        if key in violations:
            violations[key] = int(value)
    for example_name in examples:
        if example_name == "duration_mismatch":
            continue
        central_key = f"{example_name}_violation_count"
        examples[example_name] = [
            dict(example)
            for example in central_report.examples.get(central_key, [])
        ]

    hard_without_duration = {
        key: value for key, value in violations.items() if key != "task_duration_mismatch_count"
    }
    resource_structurally_legal = all(value == 0 for value in hard_without_duration.values())
    fully_legal = resource_structurally_legal and violations["task_duration_mismatch_count"] == 0
    makespan = max(
        (
            row["end"]
            for task_id, row in rows.items()
            if float(task_df.iloc[task_id]["duration"]) > tolerance
        ),
        default=0.0,
    )
    ratio_array = np.asarray(ratios, dtype=float)
    ratio_stats = {
        "count": int(ratio_array.size),
        "min": float(ratio_array.min()) if ratio_array.size else None,
        "max": float(ratio_array.max()) if ratio_array.size else None,
        "mean": float(ratio_array.mean()) if ratio_array.size else None,
        "std": float(ratio_array.std()) if ratio_array.size else None,
        "within_20_percent": int(np.sum((ratio_array >= 0.8) & (ratio_array <= 1.2))) if ratio_array.size else 0,
        "outside_20_percent": int(np.sum((ratio_array < 0.8) | (ratio_array > 1.2))) if ratio_array.size else 0,
    }
    return {
        "data_path": str(data_path.resolve()),
        "schedule_path": str(schedule_path.resolve()),
        "task_id_mode": resolved_mode,
        "num_schedule_rows": int(len(schedule)),
        "num_dataset_nodes": int(raw["num_tasks"]),
        "num_real_tasks": int(real_task_count),
        "scheduled_real_tasks": int(scheduled_real_task_count),
        "makespan_real_tasks": float(makespan),
        "station_count": int(env.num_stations),
        "worker_count": int(env.num_workers),
        "max_slots_per_station": max_slots,
        "violations": violations,
        "is_resource_structurally_legal": bool(resource_structurally_legal),
        "is_legal_against_environment_duration": bool(fully_legal),
        "is_legal_against_current_data_duration": bool(fully_legal),
        "duration_ratio_vs_environment_duration": ratio_stats,
        "examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="验证 APAL 初始调度 schedule 的公平性与合法性")
    parser.add_argument("--data", required=True, help="APAL 数据 CSV，例如 data/680.csv")
    parser.add_argument("--schedule", required=True, help="待验证 schedule CSV")
    parser.add_argument("--config", default=None, help="可选实验 YAML；缺省按数据规模加载 initial_bucket")
    parser.add_argument("--task-id-mode", choices=("auto", "internal", "sequence"), default="auto")
    parser.add_argument("--output", default=None, help="可选 JSON 输出路径")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    report = validate_schedule(
        data_path=_resolve_path(args.data),
        schedule_path=_resolve_path(args.schedule),
        config_path=args.config,
        task_id_mode=args.task_id_mode,
        tolerance=float(args.tolerance),
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    print(encoded)
    if args.output:
        output_path = _resolve_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["is_legal_against_environment_duration"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
