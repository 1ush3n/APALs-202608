from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.index_runs import write_index


def test_write_index_collects_run_manifest_and_eval_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "initial_schedule" / "initial_schedule_260630-153000"
    configs_dir = run_dir / "configs"
    eval_dir = run_dir / "eval"
    configs_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    (configs_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "created_at": "2026-06-30T15:30:00",
                "command": "evaluate",
                "experiment_name": "initial_schedule",
                "run_id": "initial_schedule_260630-153000",
                "git_commit": "abc123",
                "config_paths": ["conf/experiment/initial_schedule.yaml"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (configs_dir / "resolved_config.yaml").write_text(
        "experiment_name: initial_schedule\n"
        "data_file_path: data/283.csv\n",
        encoding="utf-8",
    )
    (eval_dir / "summary.json").write_text(
        json.dumps(
            {
                "checkpoint": "runs/initial_schedule/checkpoints/best.ckpt",
                "data_path": "data/283.csv",
                "makespan": 680.0,
                "balance_std": 12.5,
                "reward": 1.0,
                "worker_utilization": 0.8,
                "station_utilization": 0.7,
                "duration_sec": 2.0,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rows = write_index(tmp_path / "runs")

    assert rows[0]["run_id"] == "initial_schedule_260630-153000"
    assert rows[0]["makespan"] == "680.0"
    assert (tmp_path / "runs" / "index.csv").exists()
    assert (tmp_path / "runs" / "index.json").exists()

    with (tmp_path / "runs" / "index.csv").open(encoding="utf-8") as file:
        csv_rows = list(csv.DictReader(file))
    assert csv_rows[0]["experiment_name"] == "initial_schedule"


def test_write_index_expands_baseline_and_benchmark_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "exp" / "exp_260630-160000"
    configs_dir = run_dir / "configs"
    baseline_dir = run_dir / "artifacts" / "baselines" / "heuristic" / "SPT" / "283"
    benchmark_dir = run_dir / "artifacts" / "benchmark" / "heuristic_search_budget3"
    baseline_dir.mkdir(parents=True)
    benchmark_dir.mkdir(parents=True)
    configs_dir.mkdir(parents=True)
    (configs_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "created_at": "2026-06-30T16:00:00",
                "command": "run_all_baselines",
                "experiment_name": "exp",
                "run_id": "exp_260630-160000",
                "run_type": "baseline",
                "artifact_kind": "heuristic_baselines",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (configs_dir / "resolved_config.yaml").write_text("data_file_path: data/283.csv\n", encoding="utf-8")
    (baseline_dir / "metrics.json").write_text(
        json.dumps(
            {
                "makespan": 700.0,
                "workload_balance_std": 10.0,
                "worker_utilization": 0.75,
                "station_utilization": 0.65,
                "valid": 1.0,
                "inference_time": 0.2,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (benchmark_dir / "heuristic_search_budget3_runtime.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "method": "SPT",
                        "dataset": "283",
                        "makespan": 700.0,
                        "valid": 1.0,
                        "algorithm_time_sec": 0.2,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rows = write_index(tmp_path / "runs")

    kinds = {(row["artifact_kind"], row["method"], row["dataset"]) for row in rows}
    assert ("baseline", "SPT", "283") in kinds
    assert ("benchmark", "SPT", "283") in kinds
