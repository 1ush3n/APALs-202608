from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


INDEX_FIELDS = [
    "experiment_name",
    "run_id",
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


def discover_runs(runs_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for manifest_path in sorted(runs_root.glob("*/*/configs/run_manifest.json")):
        configs_dir = manifest_path.parent
        run_dir = configs_dir.parent
        manifest = _read_json(manifest_path)
        resolved = _read_resolved_config(configs_dir / "resolved_config.yaml")
        summary_path = run_dir / "eval" / "summary.json"
        summary = _read_json(summary_path)

        row = {
            "experiment_name": _as_text(manifest.get("experiment_name") or resolved.get("experiment_name") or run_dir.parent.name),
            "run_id": _as_text(manifest.get("run_id") or run_dir.name),
            "run_dir": str(run_dir),
            "created_at": _as_text(manifest.get("created_at")),
            "command": _as_text(manifest.get("command")),
            "git_commit": _as_text(manifest.get("git_commit")),
            "config_paths": _as_text(manifest.get("config_paths")),
            "checkpoint": _as_text(summary.get("checkpoint") or manifest.get("checkpoint")),
            "data_path": _as_text(summary.get("data_path") or resolved.get("data_file_path")),
            "makespan": _as_text(summary.get("makespan")),
            "balance_std": _as_text(summary.get("balance_std") or summary.get("balance")),
            "reward": _as_text(summary.get("reward")),
            "worker_utilization": _as_text(summary.get("worker_utilization")),
            "station_utilization": _as_text(summary.get("station_utilization")),
            "duration_sec": _as_text(summary.get("duration_sec")),
            "summary_path": str(summary_path) if summary_path.exists() else "",
        }
        rows.append(row)
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
