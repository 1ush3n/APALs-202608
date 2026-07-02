from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


INDEX_FIELDS = [
    "experiment_name",
    "run_id",
    "run_type",
    "artifact_kind",
    "method",
    "dataset",
    "run_dir",
    "created_at",
    "command",
    "git_commit",
    "config_paths",
    "checkpoint",
    "data_path",
    "makespan",
    "balance_std",
    "reward",
    "worker_utilization",
    "station_utilization",
    "duration_sec",
    "valid_rate",
    "inference_time",
    "train_time",
    "summary_path",
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_resolved_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value)
    return str(value)


def _first_value(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return value
    return None


def _base_row(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    resolved: dict[str, Any],
) -> dict[str, str]:
    return {
        "experiment_name": _as_text(manifest.get("experiment_name") or resolved.get("experiment_name") or run_dir.parent.name),
        "run_id": _as_text(manifest.get("run_id") or run_dir.name),
        "run_type": _as_text(manifest.get("run_type")),
        "artifact_kind": _as_text(manifest.get("artifact_kind")),
        "method": _as_text(manifest.get("method")),
        "dataset": "",
        "run_dir": str(run_dir),
        "created_at": _as_text(manifest.get("created_at")),
        "command": _as_text(manifest.get("command")),
        "git_commit": _as_text(manifest.get("git_commit")),
        "config_paths": _as_text(manifest.get("config_paths")),
        "checkpoint": _as_text(manifest.get("checkpoint")),
        "data_path": _as_text(resolved.get("data_file_path")),
        "makespan": "",
        "balance_std": "",
        "reward": "",
        "worker_utilization": "",
        "station_utilization": "",
        "duration_sec": "",
        "valid_rate": "",
        "inference_time": "",
        "train_time": "",
        "summary_path": "",
    }


def _row_from_metrics(
    base: dict[str, str],
    metrics: dict[str, Any],
    *,
    path: Path,
    artifact_kind: str,
) -> dict[str, str]:
    row = dict(base)
    row["artifact_kind"] = _as_text(metrics.get("artifact_kind") or artifact_kind or base.get("artifact_kind"))
    row["method"] = _as_text(_first_value(metrics, "method", "Method") or row.get("method"))
    row["dataset"] = _as_text(_first_value(metrics, "dataset", "Dataset") or row.get("dataset"))
    row["checkpoint"] = _as_text(_first_value(metrics, "checkpoint", "checkpoint_path", "model_path") or row.get("checkpoint"))
    row["data_path"] = _as_text(_first_value(metrics, "data_path", "dataset_path") or row.get("data_path"))
    row["makespan"] = _as_text(_first_value(metrics, "makespan", "avg_makespan", "Makespan"))
    row["balance_std"] = _as_text(_first_value(metrics, "balance_std", "avg_balance_std", "workload_balance_std", "BalanceStd", "balance"))
    row["reward"] = _as_text(_first_value(metrics, "reward", "avg_reward"))
    row["worker_utilization"] = _as_text(_first_value(metrics, "worker_utilization", "worker_util", "WorkerUtil"))
    row["station_utilization"] = _as_text(_first_value(metrics, "station_utilization", "station_util", "StationUtil"))
    row["duration_sec"] = _as_text(_first_value(metrics, "duration_sec", "avg_duration_sec", "wall_time_sec", "process_wall_time_sec"))
    row["valid_rate"] = _as_text(_first_value(metrics, "valid", "valid_rate", "completion_rate", "eligible_rate", "Valid"))
    row["inference_time"] = _as_text(_first_value(metrics, "inference_time", "inference_time_sec", "algorithm_time_sec", "Time(s)"))
    row["train_time"] = _as_text(_first_value(metrics, "train_time_hours", "train_estimate_h", "scaled_train_time_hours"))
    row["summary_path"] = str(path)
    return row


def _artifact_rows(run_dir: Path, base: dict[str, str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    summary_paths = [
        (run_dir / "eval" / "summary.json", "evaluation"),
        (run_dir / "eval" / "reschedule" / "reschedule_ppo_eval_summary.json", "reschedule_ppo"),
    ]
    for path, kind in summary_paths:
        payload = _read_json(path)
        if payload:
            rows.append(_row_from_metrics(base, payload, path=path, artifact_kind=kind))

    for path in sorted((run_dir / "artifacts" / "baselines").glob("**/metrics.json")):
        payload = _read_json(path)
        if not payload:
            continue
        metrics = dict(payload)
        metrics.setdefault("method", path.parent.parent.name)
        metrics.setdefault("dataset", path.parent.name)
        rows.append(_row_from_metrics(base, metrics, path=path, artifact_kind="baseline"))

    benchmark_jsons = [
        *sorted((run_dir / "artifacts" / "benchmark").glob("**/heuristic_search_budget3_runtime.json")),
        *sorted((run_dir / "artifacts" / "benchmark").glob("**/combined_runtime_summary.json")),
        *sorted((run_dir / "artifacts" / "benchmark").glob("**/initial_failed_runtime_supplement.json")),
    ]
    for path in benchmark_jsons:
        payload = _read_json(path)
        if not payload:
            continue
        expanded = False
        for key in ("rows", "eval_rows", "train_rows", "combined_rows"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        rows.append(_row_from_metrics(base, item, path=path, artifact_kind="benchmark"))
                        expanded = True
        if not expanded:
            rows.append(_row_from_metrics(base, payload, path=path, artifact_kind="benchmark"))
    return rows


def discover_runs(runs_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for manifest_path in sorted(runs_root.glob("*/*/configs/run_manifest.json")):
        configs_dir = manifest_path.parent
        run_dir = configs_dir.parent
        manifest = _read_json(manifest_path)
        resolved = _read_resolved_config(configs_dir / "resolved_config.yaml")
        base = _base_row(run_dir=run_dir, manifest=manifest, resolved=resolved)
        artifact_rows = _artifact_rows(run_dir, base)
        rows.extend(artifact_rows or [base])
    return rows


def write_index(
    runs_root: Path,
    *,
    output_csv: Path | None = None,
    output_json: Path | None = None,
) -> list[dict[str, str]]:
    rows = discover_runs(runs_root)
    csv_path = output_csv or runs_root / "index.csv"
    json_path = output_json or runs_root / "index.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="汇总 APAL runs 目录下的运行结果索引")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output-csv")
    parser.add_argument("--output-json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = write_index(
        Path(args.runs_root),
        output_csv=Path(args.output_csv) if args.output_csv else None,
        output_json=Path(args.output_json) if args.output_json else None,
    )
    print(f"[RunIndex] indexed={len(rows)} root={Path(args.runs_root)}")


if __name__ == "__main__":
    main()
