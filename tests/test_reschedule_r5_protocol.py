from __future__ import annotations

import json
from pathlib import Path

import pytest

from utils.reschedule import BaselineSchedule, BaselineTask
from runtime.reschedule_manifest import (
    RescheduleManifest,
    RescheduleManifestEntry,
    validate_r5_manifest_shape,
)
from scripts.prepare_reschedule_r5_data import split_r5_instance_files
from scripts.evaluate_reschedule_rules import _solver_seed_count
from configs import Config, load_config_files
from utils.reschedule_r5 import (
    generate_r5_scenario_library,
    validate_r5_scenario_library,
    write_r5_scenario_library,
)


def _baseline() -> BaselineSchedule:
    tasks = {
        task_id: BaselineTask(
            task_id=task_id,
            station_id=task_id % 5,
            team=(task_id % 8,),
            start=30.0 + task_id * 2.0,
            end=31.0 + task_id * 2.0,
            duration=1.0,
        )
        for task_id in range(30)
    }
    return BaselineSchedule(tasks=tasks, makespan=100.0)


def _specs() -> tuple[dict[str, float], dict[str, tuple[float, float, float]]]:
    return (
        {"early": 0.225, "middle": 0.400, "late": 0.575},
        {
            "low": (0.05, 5.0, 15.0),
            "medium": (0.10, 10.0, 35.0),
            "high": (0.18, 20.0, 60.0),
        },
    )


def test_r5_generates_three_paired_severities_for_each_stage() -> None:
    stage_ratios, severity_specs = _specs()
    scenarios, metadata, manifest = generate_r5_scenario_library(
        _baseline(),
        instance_id="validation_0001",
        seed=42,
        stage_ratios=stage_ratios,
        severity_specs=severity_specs,
    )

    assert [scenario_id for scenario_id, _scenario in scenarios] == [
        "low_early",
        "medium_early",
        "high_early",
        "low_middle",
        "medium_middle",
        "high_middle",
        "low_late",
        "medium_late",
        "high_late",
    ]
    assert len(metadata) == 9
    assert manifest["protocol"] == "r5_task_delay_v1"

    for stage in stage_ratios:
        rows = {
            row["severity"]: row
            for row in metadata
            if row["stage"] == stage
        }
        assert len({row["reschedule_start_time"] for row in rows.values()}) == 1
        low_ids = set(rows["low"]["delayed_task_ids"])
        medium_ids = set(rows["medium"]["delayed_task_ids"])
        high_ids = set(rows["high"]["delayed_task_ids"])
        assert low_ids <= medium_ids <= high_ids
        assert set(rows["low"]["delay_by_task"]) <= set(rows["medium"]["delay_by_task"])
        assert set(rows["medium"]["delay_by_task"]) <= set(rows["high"]["delay_by_task"])
        for scenario_id, scenario in scenarios:
            if not scenario_id.endswith(f"_{stage}"):
                continue
            assert all(
                release_time > _baseline().tasks[task_id].start
                for task_id, release_time in scenario.task_release_times.items()
            )

        for task_id in low_ids:
            assert rows["low"]["delay_by_task"][str(task_id)] < rows["medium"]["delay_by_task"][str(task_id)]
            assert rows["medium"]["delay_by_task"][str(task_id)] < rows["high"]["delay_by_task"][str(task_id)]


def test_r5_generation_is_reproducible_and_writes_metadata(tmp_path: Path) -> None:
    stage_ratios, severity_specs = _specs()
    baseline_path = tmp_path / "baseline.csv"
    baseline_path.write_text(
        "TaskID,StationID,Team,Start,End,Duration\n"
        + "\n".join(
            f"{task.task_id},{task.station_id + 1},[{task.team[0]}],{task.start},{task.end},{task.duration}"
            for task in _baseline().tasks.values()
        )
        + "\n",
        encoding="utf-8",
    )
    scenario_path = tmp_path / "scenarios.csv"
    metadata_path = tmp_path / "scenarios.metadata.json"

    first = write_r5_scenario_library(
        baseline_path=baseline_path,
        output_path=scenario_path,
        metadata_path=metadata_path,
        instance_id="validation_0001",
        seed=42,
        stage_ratios=stage_ratios,
        severity_specs=severity_specs,
    )
    first_csv = scenario_path.read_bytes()
    first_metadata = metadata_path.read_bytes()
    second = write_r5_scenario_library(
        baseline_path=baseline_path,
        output_path=tmp_path / "scenarios_again.csv",
        metadata_path=tmp_path / "scenarios_again.metadata.json",
        instance_id="validation_0001",
        seed=42,
        stage_ratios=stage_ratios,
        severity_specs=severity_specs,
    )

    assert first["protocol"] == second["protocol"] == "r5_task_delay_v1"
    assert first_csv == (tmp_path / "scenarios_again.csv").read_bytes()
    assert first_metadata == (tmp_path / "scenarios_again.metadata.json").read_bytes()
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["scenario_count"] == 9
    assert payload["baseline_sha256"]


def test_r5_rejects_empty_or_invalid_stage_configuration() -> None:
    with pytest.raises(ValueError, match="stage_ratios"):
        generate_r5_scenario_library(
            _baseline(),
            instance_id="validation_0001",
            seed=42,
            stage_ratios={},
            severity_specs=_specs()[1],
        )


def test_r5_asset_validator_rejects_zero_actual_delay() -> None:
    stage_ratios, severity_specs = _specs()
    _scenarios, metadata, manifest = generate_r5_scenario_library(
        _baseline(),
        instance_id="validation_0001",
        seed=42,
        stage_ratios=stage_ratios,
        severity_specs=severity_specs,
    )
    row = manifest["scenarios"][0]
    task_id = row["delayed_task_ids"][0]
    row["release_time_by_task"][str(task_id)] = row["baseline_start_by_task"][str(task_id)]
    with pytest.raises(ValueError, match="逐任务延迟字段不匹配|非正实际延迟"):
        validate_r5_scenario_library(_baseline(), manifest, instance_id="validation_0001")


def test_r5_manifest_requires_24_train_6_validation_and_4_eval_entries(tmp_path: Path) -> None:
    rows = []
    for index in range(24):
        rows.append((f"train_{index:04d}", "train", "generated"))
    for index in range(6):
        rows.append((f"validation_{index + 1:04d}", "validation", "generated"))
    for instance_id in ("real_283", "real_680", "real_2338", "real_3182"):
        rows.append((instance_id, "eval", "real"))
    entries = tuple(
        RescheduleManifestEntry(
            instance_id=instance_id,
            split=split,
            data_path=tmp_path / f"{instance_id}.csv",
            baseline_schedule_path=tmp_path / f"{instance_id}_baseline.csv",
            source=source,
        )
        for instance_id, split, source in rows
    )
    manifest = RescheduleManifest(
        path=tmp_path / "manifest.json",
        payload={"reschedule_protocol": "r5_task_delay_v1"},
        entries=entries,
    )

    validate_r5_manifest_shape(manifest)

    broken = RescheduleManifest(
        path=manifest.path,
        payload=manifest.payload,
        entries=entries[:-1],
    )
    with pytest.raises(ValueError, match="24 train、6 validation、4 eval"):
        validate_r5_manifest_shape(broken)


def test_r5_split_is_deterministic_and_stratified_by_task_count(tmp_path: Path) -> None:
    files = []
    for index in range(30):
        path = tmp_path / f"sample_{index:02d}.csv"
        path.write_text("x\n" * (index + 2), encoding="utf-8")
        files.append(path)

    train_files, validation_files = split_r5_instance_files(files, validation_count=6)

    assert [path.name for path in validation_files] == [
        "sample_02.csv",
        "sample_07.csv",
        "sample_12.csv",
        "sample_17.csv",
        "sample_22.csv",
        "sample_27.csv",
    ]
    assert len(train_files) == 24


def test_r5_experiment_enables_only_reschedule_fast_async_validation() -> None:
    config = Config()
    load_config_files(
        [str(Path(__file__).resolve().parents[1] / "conf" / "experiment" / "reschedule_task_delay_r5.yaml")],
        target=config,
    )

    assert config.reschedule_async_protocol == "r5_task_delay_v1"
    assert config.async_eval_enabled is True
    assert config.async_eval_device == "cuda"
    assert config.async_eval_worker_count == 3
    assert config.async_eval_submit_every_episodes == 2
    assert config.async_eval_scenario_ids == ["low_early", "medium_early", "high_early"]


def test_r5_solver_seed_policy_expands_only_stochastic_methods() -> None:
    assert _solver_seed_count("Beam", r5=True) == 3
    assert _solver_seed_count("IG", r5=True) == 3
    assert _solver_seed_count("SA", r5=True) == 3
    assert _solver_seed_count("RandomRepair", r5=True) == 3
    assert _solver_seed_count("CPMRepair", r5=True) == 1
    assert _solver_seed_count("IG", r5=False) == 1
