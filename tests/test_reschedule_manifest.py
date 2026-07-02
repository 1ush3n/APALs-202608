from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from configs import Config, configs, load_config_files
from environment import AirLineEnv_Graph
from runtime.reschedule_manifest import load_reschedule_manifest
from tests.runtime_safety import temporary_config
from tests.test_reschedule_task_delay import PROJECT_ROOT, _reschedule_overrides, _write_greedy_baseline


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": 1, "kind": "reschedule_dataset_manifest", "instances": rows}, ensure_ascii=False),
        encoding="utf-8",
    )


def test_reschedule_manifest_matches_by_instance_and_data_path(tmp_path: Path) -> None:
    data_path = tmp_path / "case.csv"
    baseline_path = tmp_path / "case_schedule.csv"
    scenario_path = tmp_path / "case_scenarios.csv"
    data_path.write_text("dummy", encoding="utf-8")
    baseline_path.write_text("dummy", encoding="utf-8")
    scenario_path.write_text("dummy", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        [
            {
                "instance_id": "real_680",
                "split": "eval",
                "source": "real",
                "data_path": str(data_path),
                "baseline_schedule_path": str(baseline_path),
                "scenario_path": str(scenario_path),
                "status": "ready",
            }
        ],
    )

    manifest = load_reschedule_manifest(manifest_path)
    entry = manifest.get("real_680")
    assert entry.data_path == data_path
    assert entry.baseline_schedule_path == baseline_path
    assert entry.scenario_path == scenario_path
    assert manifest.find_by_data_path(data_path).instance_id == "real_680"


def test_environment_switches_reschedule_baseline_from_manifest(tmp_path: Path) -> None:
    data_dir = tmp_path / "datasets"
    data_dir.mkdir()
    case_a = data_dir / "case_a.csv"
    case_b = data_dir / "case_b.csv"
    shutil.copy2(PROJECT_ROOT / "data" / "283.csv", case_a)
    shutil.copy2(PROJECT_ROOT / "data" / "283.csv", case_b)

    baseline_a = tmp_path / "baseline_a.csv"
    baseline_b = tmp_path / "baseline_b.csv"
    df = _write_greedy_baseline(baseline_a)
    shifted = df.copy()
    shifted["Start"] = shifted["Start"] + 10.0
    shifted["End"] = shifted["End"] + 10.0
    shifted.to_csv(baseline_b, index=False)
    scenario_path = tmp_path / "scenario.csv"
    delayed_row = df[df["Start"] > float(df["Start"].quantile(0.35))].iloc[0]
    pd.DataFrame(
        [
            {
                "reschedule_start_time": float(df["Start"].quantile(0.35)),
                "TaskID": int(delayed_row["TaskID"]),
                "release_time": float(delayed_row["Start"] + 8.0),
            }
        ]
    ).to_csv(scenario_path, index=False)

    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        [
            {
                "instance_id": "train_a",
                "split": "train",
                "source": "generated",
                "data_path": str(case_a),
                "baseline_schedule_path": str(baseline_a),
                "status": "ready",
            },
            {
                "instance_id": "train_b",
                "split": "train",
                "source": "generated",
                "data_path": str(case_b),
                "baseline_schedule_path": str(baseline_b),
                "status": "ready",
            },
        ],
    )

    cfg = Config()
    load_config_files([str(PROJECT_ROOT / "conf" / "experiment" / "reschedule_task_delay.yaml")], target=cfg)
    overrides = cfg.to_flat_dict()
    overrides.update(
        {
            "reschedule_manifest_path": str(manifest_path),
            "reschedule_scenario_path": str(scenario_path),
            "reschedule_eval_scenario_path": str(scenario_path),
            "data_file_path": str(case_a),
            "train_data_path_or_dir": str(data_dir),
            "randomize_durations": False,
            "enable_shadow_mask_verification": False,
        }
    )
    with temporary_config(configs, overrides):
        env = AirLineEnv_Graph(data_path_or_dir=str(data_dir), seed=19)
        env.reset(randomize_duration=False, randomize_workers=False, seed=19)
        assert abs(env.baseline_schedule.makespan - float(df["End"].max())) < 1e-6

        env.switch_dataset(1)
        env.reset(randomize_duration=False, randomize_workers=False, seed=19)
        assert abs(env.baseline_schedule.makespan - float(shifted["End"].max())) < 1e-6
