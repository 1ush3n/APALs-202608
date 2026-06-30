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
