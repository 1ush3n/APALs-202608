from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.evaluate_reschedule_search_small as search_small
from scripts.evaluate_reschedule_search_small import (
    prepare_small_search_protocol,
    select_three_level_scenarios,
)
from tests.test_reschedule_task_delay import PROJECT_ROOT, _write_greedy_baseline
from utils.reschedule import RescheduleScenario, load_reschedule_scenarios, save_reschedule_scenarios


def _scenario_items() -> list[tuple[str, RescheduleScenario]]:
    return [
        (
            f"{level}_{idx:03d}",
            RescheduleScenario(start_time=float(idx + 1), task_release_times={idx: float(idx + 2)}),
        )
        for level in ("low", "medium", "high")
        for idx in range(2)
    ]


def _write_source_manifest(tmp_path: Path) -> Path:
    baseline_path = tmp_path / "baseline.csv"
    _write_greedy_baseline(baseline_path)
    scenario_path = tmp_path / "scenarios.csv"
    save_reschedule_scenarios(scenario_path, _scenario_items())
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "reschedule_dataset_manifest",
                "instances": [
                    {
                        "instance_id": "real_283",
                        "split": "eval",
                        "source": "real",
                        "data_path": str(PROJECT_ROOT / "data" / "283.csv"),
                        "baseline_schedule_path": str(baseline_path),
                        "scenario_path": str(scenario_path),
                        "status": "ready",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_select_three_level_scenarios_uses_same_fixed_index() -> None:
    selected = select_three_level_scenarios(_scenario_items(), scenario_index=1)
    assert [scenario_id for scenario_id, _ in selected] == [
        "low_001",
        "medium_001",
        "high_001",
    ]


def test_select_three_level_scenarios_rejects_missing_level() -> None:
    items = [item for item in _scenario_items() if not item[0].startswith("high_")]
    with pytest.raises(ValueError, match="high_000"):
        select_three_level_scenarios(items, scenario_index=0)


def test_prepare_protocol_does_not_modify_source_and_detects_changes(tmp_path: Path) -> None:
    manifest_path = _write_source_manifest(tmp_path)
    source_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_scenario_path = Path(source_payload["instances"][0]["scenario_path"])
    source_before = source_scenario_path.read_bytes()
    output_dir = tmp_path / "output"

    derived_manifest, protocol = prepare_small_search_protocol(
        manifest_path=manifest_path,
        instance_id="real_283",
        scenario_index=0,
        output_dir=output_dir,
        force_rerun=False,
    )

    assert source_scenario_path.read_bytes() == source_before
    assert protocol["selected_scenario_ids"] == ["low_000", "medium_000", "high_000"]
    derived_payload = json.loads(derived_manifest.read_text(encoding="utf-8"))
    selected_path = Path(derived_payload["instances"][0]["scenario_path"])
    assert [item[0] for item in load_reschedule_scenarios(selected_path)] == [
        "low_000",
        "medium_000",
        "high_000",
    ]

    changed = pd.read_csv(source_scenario_path)
    changed.loc[0, "release_time"] = float(changed.loc[0, "release_time"]) + 1.0
    changed.to_csv(source_scenario_path, index=False)
    with pytest.raises(RuntimeError, match="不同输入"):
        prepare_small_search_protocol(
            manifest_path=manifest_path,
            instance_id="real_283",
            scenario_index=0,
            output_dir=output_dir,
            force_rerun=False,
        )


def test_small_search_delegates_exactly_nine_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = _write_source_manifest(tmp_path)
    captured: dict = {}

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return {"row_count": 9, "summary_by_instance_method": []}

    monkeypatch.setattr(search_small, "evaluate_reschedule_rules_manifest", fake_evaluate)
    summary = search_small.run_small_search(
        manifest_path=manifest_path,
        instance_id="real_283",
        scenario_index=0,
        output_dir=tmp_path / "run",
        seed=42,
        parallel_workers=3,
        beam_width=4,
        beam_branch_factor=4,
        beam_levels=4,
        beam_patience=2,
        ig_iterations=80,
        ig_destroy_ratio=0.10,
        ig_noise_sigma=0.20,
        sa_iterations=120,
        sa_initial_temp=0.05,
        sa_cooling=0.96,
        sa_min_temp=1.0e-4,
        verify_static_cache=False,
        resume_partial=True,
        force_rerun=False,
        flush_every=1,
        progress_interval=30.0,
        quiet=False,
    )
    assert summary["row_count"] == 9
    assert captured["methods"] == ["Beam", "IG", "SA"]
    assert captured["instance_ids"] == ["real_283"]
    assert captured["resume"] is True
    derived = json.loads(Path(captured["manifest_path"]).read_text(encoding="utf-8"))
    selected_path = Path(derived["instances"][0]["scenario_path"])
    assert len(load_reschedule_scenarios(selected_path)) == 3
