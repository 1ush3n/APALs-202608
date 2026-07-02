from __future__ import annotations

import ast
import csv
import json
import math
import time
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from configs import configs
from data_loader import load_data


METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "makespan": ("makespan", "avg_makespan", "Mk", "Makespan"),
    "balance_std": ("balance_std", "avg_balance_std", "workload_balance_std", "BalanceStd", "balance"),
    "worker_utilization": ("worker_utilization", "worker_util", "WorkerUtil"),
    "station_utilization": ("station_utilization", "station_util", "StationUtil"),
    "valid": ("valid", "valid_rate", "Valid"),
    "completion_rate": ("completion_rate", "complete", "Complete"),
    "inference_time": ("inference_time", "inference_time_sec", "algorithm_time_sec", "Time(s)"),
    "wall_time": ("duration_sec", "avg_duration_sec", "wall_time_sec", "process_wall_time_sec"),
    "train_time": ("train_time", "train_time_hours", "estimated_full_train_time_h"),
    "seed": ("seed", "Seed"),
    "rollout_sps": ("SPS", "rollout_sps", "steps_per_second"),
    "gpu_memory_peak_mb": ("gpu_memory_peak_mb", "peak_gpu_memory_mb", "max_memory_mb"),
    "deadlock_count": ("deadlock_count", "deadlocks"),
}

CONSTRAINT_KEYS = (
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
    "deadlock_count",
)


@dataclass(frozen=True)
class PaperExperimentConfig:
    """论文实验套件的轻量配置。"""

    experiment_name: str
    output_root: Path
    datasets: dict[str, Path]
    train_datasets: dict[str, Path]
    result_roots: tuple[Path, ...]
    runs_root: Path
    reference_makespans: dict[str, float]
    methods: tuple[str, ...]
    ablations: dict[str, dict[str, Any]]
    checkpoints: dict[str, dict[str, Path]]
    seeds: tuple[int, ...]
    bootstrap_samples: int
    permutation_samples: int
    reference_method: str
    station_count: int
    worker_count: int
    run_id: str


def resolve_path(path_like: str | Path, project_root: Path) -> Path:
    path = Path(path_like).expanduser()
    return path if path.is_absolute() else project_root / path


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def first_value(data: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    for key in aliases:
        value = data.get(key)
        if value not in (None, ""):
            return value
    return None


def infer_dataset_name(path_or_name: Any) -> str:
    if path_or_name in (None, ""):
        return ""
    text = str(path_or_name)
    name = Path(text.replace("\\", "/")).stem
    return name


def infer_method_dataset(path: Path, payload: Mapping[str, Any] | None = None) -> tuple[str, str]:
    payload = payload or {}
    method = str(first_value(payload, ("method", "Method", "algorithm", "Algorithm")) or "")
    dataset = str(first_value(payload, ("dataset", "Dataset")) or "")
    data_path = first_value(payload, ("data_path", "dataset_path", "test_data"))
    if data_path and not dataset:
        dataset = infer_dataset_name(data_path)

    parts = list(path.parts)
    if not dataset:
        parent = path.parent.name
        if parent.replace(".", "", 1).isdigit() or parent in {"283", "680", "2338", "3182"}:
            dataset = parent
    if not method:
        if len(parts) >= 3 and path.parent.name == dataset:
            method = path.parent.parent.name
        elif "eval_logs" in parts:
            idx = parts.index("eval_logs")
            if idx + 1 < len(parts):
                method = parts[idx + 1]
        elif "baselines" in parts and len(parts) >= 3:
            method = path.parent.parent.name
        else:
            method = "full"

    if method in {"eval", "results", "runs", ""}:
        method = "full"
    return method, dataset


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames or _union_fields(rows))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _union_fields(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(str(key))
                seen.add(str(key))
    return fields or ["status"]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_paper_config(config_path: Path, project_root: Path, overrides: Mapping[str, Any] | None = None) -> PaperExperimentConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("读取 paper experiment YAML 需要 PyYAML。") from exc

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"配置根节点必须是 mapping: {config_path}")
    raw = _deep_update(dict(raw), dict(overrides or {}))

    datasets = {
        str(name): resolve_path(path, project_root)
        for name, path in (raw.get("datasets") or {}).items()
    }
    train_datasets = {
        str(name): resolve_path(path, project_root)
        for name, path in (raw.get("train_datasets") or {}).items()
    }
    result_roots = tuple(
        resolve_path(path, project_root)
        for path in raw.get("result_roots", ["results", "runs"])
    )
    checkpoints: dict[str, dict[str, Path]] = {}
    for method, mapping in (raw.get("checkpoints") or {}).items():
        if not isinstance(mapping, Mapping):
            continue
        checkpoints[str(method)] = {
            str(dataset): resolve_path(path, project_root)
            for dataset, path in mapping.items()
        }
    return PaperExperimentConfig(
        experiment_name=str(raw.get("experiment_name", "paper_experiment_suite")),
        output_root=resolve_path(raw.get("output_root", "results/paper_experiments"), project_root),
        datasets=datasets,
        train_datasets=train_datasets,
        result_roots=result_roots,
        runs_root=resolve_path(raw.get("runs_root", "runs"), project_root),
        reference_makespans={str(k): float(v) for k, v in (raw.get("reference_makespans") or {}).items()},
        methods=tuple(str(item) for item in raw.get("methods", [])),
        ablations={
            str(name): dict(value or {})
            for name, value in (raw.get("ablations") or {}).items()
        },
        checkpoints=checkpoints,
        seeds=tuple(int(seed) for seed in raw.get("seeds", [0, 1, 2, 3, 4])),
        bootstrap_samples=int(raw.get("bootstrap_samples", 2000)),
        permutation_samples=int(raw.get("permutation_samples", 5000)),
        reference_method=str(raw.get("reference_method", "full")),
        station_count=int(raw.get("station_count", getattr(configs, "n_m", 5))),
        worker_count=int(raw.get("worker_count", getattr(configs, "n_w", 80))),
        run_id=str(raw.get("run_id", "") or ""),
    )


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if "." in key:
            head, tail = key.split(".", 1)
            child = base.get(head)
            if not isinstance(child, dict):
                child = {}
            base[head] = _deep_update(child, {tail: value})
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = value
    return base


def dataset_profile_rows(datasets: Mapping[str, Path], *, station_count: int, worker_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, path in datasets.items():
        start = time.perf_counter()
        with redirect_stdout(StringIO()):
            raw = load_data(path)
        load_sec = time.perf_counter() - start
        df = raw["task_df"].copy()
        edge_count = int(raw["precedence_edges"].shape[1])
        durations = pd.to_numeric(df.get("duration"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        demand = pd.to_numeric(df.get("demand_workers"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        skill = pd.to_numeric(df.get("skill_type"), errors="coerce")
        real_mask = (durations > 1e-9) | (demand > 0)
        real_duration = durations[real_mask]
        real_demand = demand[real_mask]
        fixed_station = pd.to_numeric(df.get("fixed_station"), errors="coerce")
        skill_entropy = _entropy(skill[real_mask].dropna().astype(int).tolist())
        root_count, sub_count = _count_hierarchy(df["task_id"].astype(str).tolist())
        node_count = int(len(df))
        rows.append(
            {
                "dataset": dataset,
                "path": str(path),
                "status": "ok",
                "node_count": node_count,
                "real_task_count": int(real_mask.sum()),
                "root_count": root_count,
                "sub_count": sub_count,
                "station_count": int(station_count),
                "worker_count": int(worker_count),
                "precedence_edge_count": edge_count,
                "precedence_edge_density": edge_count / max(1, node_count * max(1, node_count - 1)),
                "critical_path_lower_bound": _critical_path_length(raw["precedence_edges"], durations),
                "total_work_hours": float(real_duration.sum()),
                "duration_mean": _mean(real_duration),
                "duration_std": _std(real_duration),
                "duration_p95": _percentile(real_duration, 95),
                "duration_cv": _std(real_duration) / max(1e-9, _mean(real_duration)),
                "demand_mean": _mean(real_demand),
                "demand_p95": _percentile(real_demand, 95),
                "fixed_station_ratio": float((fixed_station.fillna(0.0).to_numpy() > 0).mean()),
                "skill_type_count": int(skill[real_mask].dropna().nunique()),
                "skill_entropy": skill_entropy,
                "load_data_sec": float(load_sec),
            }
        )
    return rows


@contextmanager
def temporary_resource_graph_mode(*, use_skill_hub: bool, bidirectional: bool):
    old_hub = bool(getattr(configs, "use_skill_hub", True))
    old_bidirectional = bool(getattr(configs, "skill_hub_bidirectional", True))
    try:
        configs.use_skill_hub = use_skill_hub
        configs.skill_hub_bidirectional = bidirectional
        yield
    finally:
        configs.use_skill_hub = old_hub
        configs.skill_hub_bidirectional = old_bidirectional


def graph_complexity_rows(datasets: Mapping[str, Path], *, seed: int = 42) -> list[dict[str, Any]]:
    from environment import AirLineEnv_Graph

    rows: list[dict[str, Any]] = []
    for dataset, path in datasets.items():
        with temporary_resource_graph_mode(use_skill_hub=False, bidirectional=False):
            direct_start = time.perf_counter()
            with redirect_stdout(StringIO()):
                direct_env = AirLineEnv_Graph(path, seed=seed)
                direct_obs = direct_env.reset(seed=seed)
            direct_sec = time.perf_counter() - direct_start

        with temporary_resource_graph_mode(use_skill_hub=True, bidirectional=True):
            hub_start = time.perf_counter()
            with redirect_stdout(StringIO()):
                hub_env = AirLineEnv_Graph(path, seed=seed)
                hub_obs = hub_env.reset(seed=seed)
            hub_sec = time.perf_counter() - hub_start

        direct_worker_task = _edge_count(direct_obs, ("worker", "can_do", "task"))
        worker_skill = _edge_count(hub_obs, ("worker", "has_skill", "skill"))
        skill_task = _edge_count(hub_obs, ("skill", "required_by", "task"))
        reverse_task_skill = _edge_count(hub_obs, ("task", "requires", "skill"))
        reverse_skill_worker = _edge_count(hub_obs, ("skill", "provided_by", "worker"))
        hub_forward = worker_skill + skill_task
        hub_bidirectional = hub_forward + reverse_task_skill + reverse_skill_worker
        rows.append(
            {
                "dataset": dataset,
                "path": str(path),
                "status": "ok",
                "task_nodes": _node_count(hub_obs, "task"),
                "worker_nodes": _node_count(hub_obs, "worker"),
                "station_nodes": _node_count(hub_obs, "station"),
                "skill_nodes": _node_count(hub_obs, "skill"),
                "precedence_edges": _edge_count(hub_obs, ("task", "precedes", "task")),
                "direct_worker_task_edges": direct_worker_task,
                "worker_skill_edges": worker_skill,
                "skill_task_edges": skill_task,
                "reverse_task_skill_edges": reverse_task_skill,
                "reverse_skill_worker_edges": reverse_skill_worker,
                "skill_hub_forward_edges": hub_forward,
                "skill_hub_bidirectional_edges": hub_bidirectional,
                "direct_total_edges": _total_edges(direct_obs),
                "skill_hub_total_edges": _total_edges(hub_obs),
                "skill_edge_reduction_ratio": 1.0 - hub_forward / max(1, direct_worker_task),
                "total_edge_reduction_ratio": 1.0 - _total_edges(hub_obs) / max(1, _total_edges(direct_obs)),
                "direct_build_sec": float(direct_sec),
                "skill_hub_build_sec": float(hub_sec),
            }
        )
    return rows


def collect_result_rows(result_roots: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in result_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.json")):
            if _is_generated_paper_output(path):
                continue
            if not _looks_like_metric_json(path):
                continue
            try:
                payload = read_json(path)
            except Exception:
                continue
            rows.extend(_rows_from_json_payload(path, payload))
        for path in sorted(root.rglob("*.csv")):
            if _is_generated_paper_output(path):
                continue
            if not _looks_like_metric_csv(path):
                continue
            rows.extend(_rows_from_csv(path))
    return _dedupe_rows(rows)


def statistical_summary_rows(
    result_rows: Sequence[Mapping[str, Any]],
    *,
    reference_makespans: Mapping[str, float],
    reference_method: str,
    bootstrap_samples: int,
    seed: int = 123,
) -> list[dict[str, Any]]:
    grouped = _group_numeric(result_rows, "makespan")
    reference_means: dict[str, float] = {}
    for (method, dataset), values in grouped.items():
        if method == reference_method and values:
            reference_means[dataset] = float(np.mean(values))

    rank_by_method: dict[str, list[float]] = {}
    for dataset in sorted({dataset for _, dataset in grouped}):
        means = [
            (method, float(np.mean(values)))
            for (method, ds), values in grouped.items()
            if ds == dataset and values
        ]
        means.sort(key=lambda item: item[1])
        for rank, (method, _value) in enumerate(means, start=1):
            rank_by_method.setdefault(method, []).append(float(rank))

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for (method, dataset), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=float)
        ci_low, ci_high = bootstrap_ci(arr, samples=bootstrap_samples, rng=rng)
        reference_mean = reference_means.get(dataset)
        normalized = arr / float(reference_makespans[dataset]) if dataset in reference_makespans else np.full_like(arr, np.nan)
        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "metric": "makespan",
                "status": "ok" if len(arr) else "missing_result",
                "num_runs": int(len(arr)),
                "mean": _mean(arr),
                "std": _std(arr),
                "median": _percentile(arr, 50),
                "iqr": _percentile(arr, 75) - _percentile(arr, 25),
                "min": float(np.min(arr)) if len(arr) else "",
                "max": float(np.max(arr)) if len(arr) else "",
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "normalized_mean": _mean(normalized[~np.isnan(normalized)]) if not np.isnan(normalized).all() else "",
                "relative_gap_to_reference": (
                    _mean(arr) / reference_mean - 1.0
                    if reference_mean not in (None, 0.0)
                    else ""
                ),
                "average_rank": _mean(np.asarray(rank_by_method.get(method, []), dtype=float)),
            }
        )
    return rows


def significance_test_rows(
    result_rows: Sequence[Mapping[str, Any]],
    *,
    reference_method: str,
    permutation_samples: int,
    seed: int = 123,
) -> list[dict[str, Any]]:
    grouped = _group_numeric(result_rows, "makespan")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for dataset in sorted({dataset for _, dataset in grouped}):
        ref = np.asarray(grouped.get((reference_method, dataset), []), dtype=float)
        for (method, ds), values in sorted(grouped.items()):
            if ds != dataset or method == reference_method:
                continue
            other = np.asarray(values, dtype=float)
            if len(ref) < 2 or len(other) < 2:
                rows.append(
                    {
                        "dataset": dataset,
                        "reference_method": reference_method,
                        "method": method,
                        "status": "insufficient_runs",
                        "reference_runs": int(len(ref)),
                        "method_runs": int(len(other)),
                    }
                )
                continue
            p_value = permutation_pvalue(ref, other, samples=permutation_samples, rng=rng)
            rows.append(
                {
                    "dataset": dataset,
                    "reference_method": reference_method,
                    "method": method,
                    "status": "ok",
                    "reference_runs": int(len(ref)),
                    "method_runs": int(len(other)),
                    "reference_mean": _mean(ref),
                    "method_mean": _mean(other),
                    "mean_improvement": _mean(other) - _mean(ref),
                    "relative_improvement": (_mean(other) - _mean(ref)) / max(1e-9, _mean(other)),
                    "p_value": p_value,
                    "cliffs_delta": cliffs_delta(ref, other),
                }
            )
    return apply_holm_correction(rows)


def constraint_diagnostic_rows(result_rows: Sequence[Mapping[str, Any]], result_roots: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in result_rows:
        method = str(row.get("method", ""))
        dataset = str(row.get("dataset", ""))
        diagnostic = {
            "method": method,
            "dataset": dataset,
            "source_path": row.get("source_path", ""),
            "status": row.get("status", "ok"),
            "valid": row.get("valid", ""),
            "completion_rate": row.get("completion_rate", ""),
            "deadlock_count": row.get("deadlock_count", ""),
        }
        if any(diagnostic.get(key, "") not in ("", None) for key in ("valid", "completion_rate", "deadlock_count")):
            rows.append(diagnostic)

    for root in result_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.json")):
            if _is_generated_paper_output(path):
                continue
            try:
                payload = read_json(path)
            except Exception:
                continue
            rows.extend(_constraint_rows_from_payload(path, payload))
        for path in sorted(root.rglob("*.csv")):
            if _is_generated_paper_output(path):
                continue
            if path.stat().st_size > 8_000_000:
                continue
            try:
                frame = pd.read_csv(path)
            except Exception:
                continue
            if not any(key in frame.columns for key in CONSTRAINT_KEYS):
                continue
            method, dataset = infer_method_dataset(path)
            for idx, data in frame.iterrows():
                item = {"method": method, "dataset": dataset, "source_path": str(path), "row_index": int(idx), "status": "ok"}
                for key in CONSTRAINT_KEYS:
                    if key in data:
                        item[key] = data.get(key)
                rows.append(item)
    return _dedupe_rows(rows)


def policy_behavior_rows(result_roots: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in result_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*schedule*.csv")):
            if _is_generated_paper_output(path):
                continue
            if path.stat().st_size > 20_000_000:
                continue
            try:
                frame = pd.read_csv(path)
            except Exception:
                continue
            required = {"TaskID", "StationID", "Team", "Start", "End"}
            if not required.issubset(frame.columns):
                continue
            rows.append(schedule_behavior_metrics(path, frame))
    return rows


def schedule_behavior_metrics(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    method, dataset = infer_method_dataset(path)
    worker_loads: dict[int, float] = {}
    station_loads: dict[int, float] = {}
    team_sizes: list[int] = []
    positive_rows = 0
    makespan = 0.0
    for item in frame.to_dict("records"):
        start = safe_float(item.get("Start")) or 0.0
        end = safe_float(item.get("End")) or start
        duration = max(0.0, end - start)
        if "Duration" in item:
            duration = max(duration, safe_float(item.get("Duration")) or 0.0)
        makespan = max(makespan, end)
        team = parse_team(item.get("Team"))
        team_sizes.append(len(team))
        if duration > 1e-9:
            positive_rows += 1
        for worker_id in team:
            worker_loads[int(worker_id)] = worker_loads.get(int(worker_id), 0.0) + duration
        station = int(safe_float(item.get("StationID")) or 0)
        if station > 0:
            station_loads[station] = station_loads.get(station, 0.0) + duration
    worker_values = np.asarray(list(worker_loads.values()), dtype=float)
    station_values = np.asarray(list(station_loads.values()), dtype=float)
    team_arr = np.asarray(team_sizes, dtype=float)
    return {
        "method": method,
        "dataset": dataset,
        "source_path": str(path),
        "status": "ok",
        "assigned_row_count": int(len(frame)),
        "positive_duration_task_count": int(positive_rows),
        "makespan": float(makespan),
        "active_worker_count": int(len(worker_loads)),
        "active_station_count": int(len(station_loads)),
        "worker_load_mean": _mean(worker_values),
        "worker_load_std": _std(worker_values),
        "worker_load_gini": gini(worker_values),
        "station_load_mean": _mean(station_values),
        "station_load_std": _std(station_values),
        "station_load_gini": gini(station_values),
        "team_size_mean": _mean(team_arr),
        "team_size_std": _std(team_arr),
        "team_size_p95": _percentile(team_arr, 95),
        "single_worker_task_ratio": float(np.mean(team_arr == 1.0)) if len(team_arr) else "",
        "multi_worker_task_ratio": float(np.mean(team_arr > 1.0)) if len(team_arr) else "",
    }


def efficiency_pareto_rows(
    result_rows: Sequence[Mapping[str, Any]],
    *,
    reference_makespans: Mapping[str, float],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in result_rows:
        method = str(row.get("method", ""))
        dataset = str(row.get("dataset", ""))
        makespan = safe_float(row.get("makespan"))
        time_value = safe_float(row.get("inference_time")) or safe_float(row.get("wall_time"))
        if not method or not dataset or makespan is None or time_value is None:
            continue
        grouped.setdefault((method, dataset), []).append(row)

    rows: list[dict[str, Any]] = []
    for (method, dataset), items in sorted(grouped.items()):
        makespans = np.asarray([safe_float(item.get("makespan")) for item in items if safe_float(item.get("makespan")) is not None])
        times = np.asarray([
            safe_float(item.get("inference_time")) or safe_float(item.get("wall_time"))
            for item in items
            if (safe_float(item.get("inference_time")) or safe_float(item.get("wall_time"))) is not None
        ])
        if not len(makespans) or not len(times):
            continue
        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "num_runs": int(min(len(makespans), len(times))),
                "mean_makespan": _mean(makespans),
                "mean_normalized_makespan": (
                    _mean(makespans) / float(reference_makespans[dataset])
                    if dataset in reference_makespans
                    else ""
                ),
                "mean_time_sec": _mean(times),
                "p90_time_sec": _percentile(times, 90),
                "p95_time_sec": _percentile(times, 95),
            }
        )

    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_dataset.setdefault(str(row["dataset"]), []).append(row)
    final_rows: list[dict[str, Any]] = []
    for dataset, dataset_rows in by_dataset.items():
        best_makespan = math.inf
        for row in sorted(dataset_rows, key=lambda item: float(item["mean_time_sec"])):
            current = float(row["mean_makespan"])
            efficient = current < best_makespan - 1e-9
            if efficient:
                best_makespan = current
            updated = dict(row)
            updated["pareto_efficient"] = float(efficient)
            final_rows.append(updated)
    return final_rows


def convergence_stability_rows(result_roots: Iterable[Path]) -> list[dict[str, Any]]:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception:
        return [{"status": "tensorboard_not_available"}]

    rows: list[dict[str, Any]] = []
    for root in result_roots:
        if not root.exists():
            continue
        for event_path in sorted(root.rglob("events.out.tfevents.*")):
            if _is_generated_paper_output(event_path):
                continue
            try:
                accumulator = event_accumulator.EventAccumulator(str(event_path), size_guidance={"scalars": 0})
                accumulator.Reload()
            except Exception:
                continue
            tags = accumulator.Tags().get("scalars", [])
            for tag in tags:
                if not _is_training_scalar_tag(tag):
                    continue
                values = accumulator.Scalars(tag)
                if not values:
                    continue
                scalar_values = np.asarray([float(item.value) for item in values], dtype=float)
                steps = [int(item.step) for item in values]
                tail = scalar_values[-min(10, len(scalar_values)) :]
                rows.append(
                    {
                        "status": "ok",
                        "event_path": str(event_path),
                        "tag": tag,
                        "first_step": steps[0],
                        "last_step": steps[-1],
                        "num_points": int(len(scalar_values)),
                        "first_value": float(scalar_values[0]),
                        "final_value": float(scalar_values[-1]),
                        "best_min_value": float(np.min(scalar_values)),
                        "best_max_value": float(np.max(scalar_values)),
                        "tail_mean": _mean(tail),
                        "tail_std": _std(tail),
                        "tail_slope": _tail_slope(tail),
                    }
                )
    return rows or [{"status": "no_tensorboard_events"}]


def command_plan_rows(config: PaperExperimentConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, spec in config.ablations.items():
        overrides = [str(item) for item in spec.get("overrides", [])]
        experiment = str(spec.get("experiment", "scale_400_800_schedule"))
        for seed in config.seeds:
            command = [
                "python",
                "train.py",
                f"experiment={experiment}",
                f"seed={seed}",
                f"run_id={variant}_seed{seed}",
                *overrides,
            ]
            rows.append(
                {
                    "suite": "ablation",
                    "variant": variant,
                    "seed": int(seed),
                    "command": " ".join(command),
                    "status": "planned",
                }
            )
    for method, datasets in config.checkpoints.items():
        for dataset, checkpoint in datasets.items():
            data_path = config.datasets.get(dataset)
            if data_path is None:
                continue
            command = [
                "python",
                "evaluate_model.py",
                "experiment=initial_schedule",
                f"model_path={checkpoint}",
                f"test_data={data_path}",
                "num_runs=5",
                "temperature=0.0",
                "scenario=standard",
                "no_gantt=true",
            ]
            rows.append(
                {
                    "suite": "generalization",
                    "variant": method,
                    "dataset": dataset,
                    "checkpoint": str(checkpoint),
                    "command": " ".join(command),
                    "status": "planned" if checkpoint.exists() else "missing_checkpoint",
                }
            )
    return rows


def write_summary_markdown(path: Path, outputs: Mapping[str, Path], row_counts: Mapping[str, int]) -> None:
    lines = [
        "# APAL 论文实验可信度证据包",
        "",
        "本报告由 `scripts/paper_experiment_suite.py` 自动生成，面向论文实验表格、附录和复现实验审计。",
        "",
        "## 输出文件",
        "",
        "| 文件 | 行数 | 用途 |",
        "|---|---:|---|",
    ]
    descriptions = {
        "dataset_profile": "数据规模、工时分布、需求分布、关键路径下界",
        "complexity": "Skill Hub 前后节点、边数和构图耗时",
        "statistical_summary": "多种子均值、方差、置信区间和平均排名",
        "significance_tests": "置换检验、Cliff's delta 和 Holm 校正",
        "constraint_diagnostics": "合法性、完成率和硬约束违例诊断",
        "policy_behavior": "负载 Gini、团队规模和资源使用行为",
        "efficiency_pareto": "质量-效率 Pareto 数据",
        "convergence_stability": "TensorBoard 收敛稳定性统计",
        "command_plan": "消融、泛化和评估命令清单",
    }
    for key, output_path in outputs.items():
        lines.append(f"| `{output_path.name}` | {row_counts.get(key, 0)} | {descriptions.get(key, '')} |")
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- APAL 不等同于 FJSP/JSSP；这些指标围绕 APAL 的任务、工位、工人、技能、节拍和重调度约束组织。",
            "- 缺失 checkpoint 或缺失结果不会中断汇总，而是在对应 CSV 中标注 `missing_*` 状态。",
            "- 显著性检验采用无分布假设的 bootstrap/permutation 口径，用于论文证据审计，不替代工程判断。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _looks_like_metric_json(path: Path) -> bool:
    name = path.name.lower()
    if name in {"metrics.json", "summary.json", "run_manifest.json"}:
        return name != "run_manifest.json"
    return "summary" in name or "runtime" in name or "metrics" in name


def _looks_like_metric_csv(path: Path) -> bool:
    name = path.name.lower()
    return any(token in name for token in ("summary", "runtime", "index", "paper")) and path.stat().st_size < 8_000_000


def _rows_from_json_payload(path: Path, payload: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for key in ("rows", "eval_rows", "combined_rows", "train_rows"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, Mapping):
                        rows.append(_normalize_metric_row(path, item))
                return rows
        runs = payload.get("runs")
        if isinstance(runs, list) and runs:
            for item in runs:
                if isinstance(item, Mapping):
                    base = {k: v for k, v in payload.items() if k != "runs"}
                    base.update(item)
                    rows.append(_normalize_metric_row(path, base))
            return rows
        if any(alias in payload for aliases in METRIC_ALIASES.values() for alias in aliases):
            rows.append(_normalize_metric_row(path, payload))
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, Mapping) and _mapping_has_metric_alias(item):
                rows.append(_normalize_metric_row(path, item))
    return rows


def _mapping_has_metric_alias(payload: Mapping[str, Any]) -> bool:
    return any(alias in payload for aliases in METRIC_ALIASES.values() for alias in aliases)


def _is_generated_paper_output(path: Path) -> bool:
    return "paper_experiments" in {part.lower() for part in path.parts}


def _rows_from_csv(path: Path) -> list[dict[str, Any]]:
    try:
        frame = pd.read_csv(path)
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    if not any(alias in frame.columns for aliases in METRIC_ALIASES.values() for alias in aliases):
        return rows
    for item in frame.to_dict("records"):
        rows.append(_normalize_metric_row(path, item))
    return rows


def _normalize_metric_row(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    method, dataset = infer_method_dataset(path, payload)
    row: dict[str, Any] = {
        "method": method,
        "dataset": dataset,
        "source_path": str(path),
        "status": str(payload.get("status", "ok")),
    }
    for target, aliases in METRIC_ALIASES.items():
        value = first_value(payload, aliases)
        if value is not None:
            row[target] = value
    checkpoint = first_value(payload, ("checkpoint", "checkpoint_path", "model_path"))
    data_path = first_value(payload, ("data_path", "dataset_path", "test_data"))
    if checkpoint:
        row["checkpoint"] = str(checkpoint)
    if data_path:
        row["data_path"] = str(data_path)
        if not row["dataset"]:
            row["dataset"] = infer_dataset_name(data_path)
    return row


def _constraint_rows_from_payload(path: Path, payload: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        items: list[Mapping[str, Any]] = []
        for key in ("scenario_rows", "constraint_rows", "last_scenario_metrics", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                items.extend(item for item in value if isinstance(item, Mapping))
        if not items:
            items = [payload]
        for item in items:
            if any(key in item for key in CONSTRAINT_KEYS):
                method, dataset = infer_method_dataset(path, item)
                row = {"method": method, "dataset": dataset, "source_path": str(path), "status": "ok"}
                for key in CONSTRAINT_KEYS:
                    if key in item:
                        row[key] = item[key]
                for key in ("eligible_rate", "selection_score", "composite_score", "scenario_id"):
                    if key in item:
                        row[key] = item[key]
                rows.append(row)
    return rows


def _dedupe_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(row))
    return deduped


def _group_numeric(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[tuple[str, str], list[float]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        method = str(row.get("method", ""))
        dataset = str(row.get("dataset", ""))
        value = safe_float(row.get(metric))
        if not method or not dataset or value is None:
            continue
        grouped.setdefault((method, dataset), []).append(value)
    return grouped


def _count_hierarchy(task_ids: Sequence[str]) -> tuple[int, int]:
    root_count = 0
    sub_count = 0
    for task_id in task_ids:
        tid = str(task_id).strip()
        if "-" not in tid and tid.isalpha():
            root_count += 1
        elif "-" in tid:
            parts = tid.split("-")
            if len(parts) >= 2 and parts[0].isalpha() and parts[1].isdigit():
                sub_count += 1
    return root_count, sub_count


def _critical_path_length(edge_index: Any, durations: np.ndarray) -> float:
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(durations)))
    if hasattr(edge_index, "t"):
        edges = edge_index.t().tolist()
    else:
        arr = np.asarray(edge_index)
        edges = arr.T.tolist() if arr.size else []
    graph.add_edges_from((int(src), int(dst)) for src, dst in edges)
    finish = {node: float(durations[node]) for node in graph.nodes}
    for node in nx.topological_sort(graph):
        pred_finish = [finish[pred] for pred in graph.predecessors(node)]
        finish[node] = float(durations[node]) + (max(pred_finish) if pred_finish else 0.0)
    return float(max(finish.values(), default=0.0))


def _entropy(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    unique, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    probs = counts.astype(float) / max(1, counts.sum())
    return float(-np.sum(probs * np.log2(np.maximum(probs, 1e-12))))


def _edge_count(data: Any, edge_type: tuple[str, str, str]) -> int:
    if edge_type not in data.edge_types:
        return 0
    return int(data[edge_type].edge_index.size(1))


def _node_count(data: Any, node_type: str) -> int:
    if node_type not in data.node_types:
        return 0
    return int(data[node_type].x.size(0))


def _total_edges(data: Any) -> int:
    return int(sum(data[edge_type].edge_index.size(1) for edge_type in data.edge_types))


def _mean(values: np.ndarray | Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)) if len(arr) else 0.0


def _std(values: np.ndarray | Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


def _percentile(values: np.ndarray | Sequence[float], q: float) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.percentile(arr, q)) if len(arr) else 0.0


def bootstrap_ci(values: np.ndarray, *, samples: int, rng: np.random.Generator) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0
    if len(arr) == 1 or samples <= 0:
        value = float(arr[0])
        return value, value
    draws = rng.choice(arr, size=(int(samples), len(arr)), replace=True)
    means = draws.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def permutation_pvalue(a: np.ndarray, b: np.ndarray, *, samples: int, rng: np.random.Generator) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    observed = abs(float(np.mean(a) - np.mean(b)))
    combined = np.concatenate([a, b])
    count = 0
    for _ in range(max(1, int(samples))):
        shuffled = rng.permutation(combined)
        diff = abs(float(np.mean(shuffled[: len(a)]) - np.mean(shuffled[len(a) :])))
        count += int(diff >= observed - 1e-12)
    return float((count + 1) / (max(1, int(samples)) + 1))


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    greater = 0
    less = 0
    for value in a:
        greater += int(np.sum(value > b))
        less += int(np.sum(value < b))
    return float((greater - less) / max(1, len(a) * len(b)))


def apply_holm_correction(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ok_indices = [idx for idx, row in enumerate(rows) if row.get("status") == "ok" and safe_float(row.get("p_value")) is not None]
    ordered = sorted(ok_indices, key=lambda idx: float(rows[idx]["p_value"]))
    adjusted: dict[int, float] = {}
    prev = 0.0
    m = len(ordered)
    for rank, idx in enumerate(ordered):
        raw = float(rows[idx]["p_value"])
        value = min(1.0, max(prev, raw * (m - rank)))
        adjusted[idx] = value
        prev = value
    result: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        item = dict(row)
        if idx in adjusted:
            item["holm_adjusted_p"] = adjusted[idx]
            item["significant_0.05"] = float(adjusted[idx] < 0.05)
        result.append(item)
    return result


def parse_team(value: Any) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    text = str(value or "").strip()
    if not text or text in {"[]", "nan", "None"}:
        return ()
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = [part.strip() for part in text.strip("[]").split(",") if part.strip()]
    if isinstance(parsed, int):
        return (int(parsed),)
    return tuple(int(item) for item in parsed)


def gini(values: np.ndarray | Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return 0.0
    arr = np.sort(np.maximum(arr, 0.0))
    total = float(np.sum(arr))
    if total <= 1e-12:
        return 0.0
    index = np.arange(1, len(arr) + 1, dtype=float)
    return float((2.0 * np.sum(index * arr)) / (len(arr) * total) - (len(arr) + 1.0) / len(arr))


def _is_training_scalar_tag(tag: str) -> bool:
    lowered = tag.lower()
    return any(token in lowered for token in ("loss", "rew", "reward", "mk", "makespan", "sps", "eval"))


def _tail_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    slope, _intercept = np.polyfit(x, values, 1)
    return float(slope)
