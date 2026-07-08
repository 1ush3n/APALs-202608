# -*- coding: utf-8 -*-
"""验证 APAL 重调度规则 baseline 的评估入口。"""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import pytest

from configs import configs
import scripts.evaluate_reschedule_rules as rule_eval_script
from scripts.evaluate_reschedule_rules import evaluate_reschedule_rules, evaluate_reschedule_rules_manifest
from tests.runtime_safety import temporary_config
from tests.test_reschedule_task_delay import PROJECT_ROOT, _reschedule_overrides, _write_greedy_baseline


def _write_single_delay_scenario(baseline_df: pd.DataFrame, path: Path) -> None:
    start_time = float(baseline_df["Start"].quantile(0.30))
    delayed_row = baseline_df[baseline_df["Start"] > start_time].iloc[0]
    pd.DataFrame(
        [
            {
                "scenario_id": "low_000",
                "level": "low",
                "reschedule_start_time": start_time,
                "TaskID": int(delayed_row["TaskID"]),
                "release_time": float(delayed_row["Start"] + 6.0),
            }
        ]
    ).to_csv(path, index=False)


def test_reschedule_rule_eval_writes_outputs_and_reports_methods(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.csv"
    baseline_df = _write_greedy_baseline(baseline_path)
    scenario_path = tmp_path / "scenario.csv"
    _write_single_delay_scenario(baseline_df, scenario_path)
    output_dir = tmp_path / "rule_eval"

    overrides = _reschedule_overrides(baseline_path, scenario_path)
    overrides.update({"enable_shadow_mask_verification": False})
    with temporary_config(configs, overrides):
        summary = evaluate_reschedule_rules(
            data_path_or_dir=PROJECT_ROOT / "data" / "283.csv",
            scenario_path=scenario_path,
            baseline_path=baseline_path,
            methods=["NoReschedule", "SPTRepair", "StabilityAwareRepair"],
            num_runs=1,
            seed=123,
            output_dir=output_dir,
            verbose=False,
        )

    rows = summary["rows"]
    assert {row["method"] for row in rows} == {"NoReschedule", "SPTRepair", "StabilityAwareRepair"}
    assert (output_dir / "reschedule_rule_eval.csv").exists()
    assert (output_dir / "reschedule_rule_summary_by_method.csv").exists()
    assert (output_dir / "reschedule_rule_summary_by_method_level.csv").exists()

    no_reschedule = next(row for row in rows if row["method"] == "NoReschedule")
    assert no_reschedule["complete"] == 0.0
    assert no_reschedule["missing_task_count"] > 0.0

    for row in rows:
        if row["method"] == "NoReschedule":
            continue
        assert row["complete"] == 1.0
        assert row["frozen_violation_count"] == 0.0
        assert row["release_violation_count"] == 0.0
        assert row["precedence_violation_count"] == 0.0
        assert row["worker_overlap_violation_count"] == 0.0
        assert row["station_slot_violation_count"] == 0.0
        assert row["skill_violation_count"] == 0.0
        assert row["demand_violation_count"] == 0.0
        assert row["duplicate_task_count"] == 0.0
        assert row["eligible"] == 1.0


def test_reschedule_search_rules_run_with_small_budgets(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.csv"
    baseline_df = _write_greedy_baseline(baseline_path)
    scenario_path = tmp_path / "scenario.csv"
    _write_single_delay_scenario(baseline_df, scenario_path)
    output_dir = tmp_path / "search_eval"

    overrides = _reschedule_overrides(baseline_path, scenario_path)
    overrides.update({"enable_shadow_mask_verification": False})
    with temporary_config(configs, overrides):
        summary = evaluate_reschedule_rules(
            data_path_or_dir=PROJECT_ROOT / "data" / "283.csv",
            scenario_path=scenario_path,
            baseline_path=baseline_path,
            methods=["Beam", "IG", "SA"],
            num_runs=1,
            seed=321,
            output_dir=output_dir,
            verbose=False,
            beam_width=2,
            beam_branch_factor=2,
            beam_levels=2,
            beam_patience=1,
            ig_iterations=3,
            sa_iterations=3,
        )

    rows = summary["rows"]
    assert {row["method"] for row in rows} == {"Beam", "IG", "SA"}
    assert (output_dir / "reschedule_rule_eval.csv").exists()
    for row in rows:
        assert row["complete"] == 1.0
        assert row["frozen_violation_count"] == 0.0
        assert row["release_violation_count"] == 0.0
        assert row["precedence_violation_count"] == 0.0
        assert row["worker_overlap_violation_count"] == 0.0
        assert row["station_slot_violation_count"] == 0.0
        assert row["skill_violation_count"] == 0.0
        assert row["demand_violation_count"] == 0.0
        assert row["duplicate_task_count"] == 0.0
        assert row["eligible"] == 1.0


def test_reschedule_rule_manifest_eval_uses_manifest_paths(tmp_path: Path) -> None:
    data_path = PROJECT_ROOT / "data" / "283.csv"
    baseline_path = tmp_path / "baseline.csv"
    baseline_df = _write_greedy_baseline(baseline_path)
    scenario_path = tmp_path / "scenario.csv"
    _write_single_delay_scenario(baseline_df, scenario_path)
    manifest_path = tmp_path / "manifest.json"
    output_dir = tmp_path / "manifest_rule_eval"
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
                        "data_path": str(data_path),
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

    overrides = _reschedule_overrides(baseline_path, scenario_path)
    overrides.update({"enable_shadow_mask_verification": False})
    with temporary_config(configs, overrides):
        summary = evaluate_reschedule_rules_manifest(
            manifest_path=manifest_path,
            instance_ids=["real_283"],
            methods=["SPTRepair"],
            num_runs=1,
            seed=11,
            output_dir=output_dir,
            verbose=False,
        )

    rows = summary["rows"]
    assert len(rows) == 1
    assert rows[0]["instance_id"] == "real_283"
    assert Path(rows[0]["data_path"]).resolve() == data_path.resolve()
    assert Path(rows[0]["baseline_path"]).resolve() == baseline_path.resolve()
    assert Path(rows[0]["scenario_path"]).resolve() == scenario_path.resolve()
    assert rows[0]["eligible"] == 1.0
    assert (output_dir / "real_283" / "reschedule_rule_eval.csv").exists()
    assert (output_dir / "reschedule_rule_eval_by_instance.csv").exists()
    assert (output_dir / "reschedule_rule_summary_by_instance_method.csv").exists()


def test_reschedule_rule_eval_resumes_partial_jobs_after_interrupt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    baseline_path = tmp_path / "baseline.csv"
    baseline_df = _write_greedy_baseline(baseline_path)
    scenario_path = tmp_path / "scenario.csv"
    _write_single_delay_scenario(baseline_df, scenario_path)
    output_dir = tmp_path / "resume_eval"

    real_evaluate_job = rule_eval_script._evaluate_rule_job
    calls = {"count": 0}

    def interrupt_after_first_job(job: dict) -> dict:
        calls["count"] += 1
        if calls["count"] > 1:
            raise KeyboardInterrupt()
        return real_evaluate_job(job)

    overrides = _reschedule_overrides(baseline_path, scenario_path)
    overrides.update({"enable_shadow_mask_verification": False})
    with temporary_config(configs, overrides):
        monkeypatch.setattr(rule_eval_script, "_evaluate_rule_job", interrupt_after_first_job)
        with pytest.raises(KeyboardInterrupt):
            evaluate_reschedule_rules(
                data_path_or_dir=PROJECT_ROOT / "data" / "283.csv",
                scenario_path=scenario_path,
                baseline_path=baseline_path,
                methods=["NoReschedule", "SPTRepair"],
                num_runs=1,
                seed=123,
                output_dir=output_dir,
                verbose=False,
                parallel_workers=1,
            )

    partial_path = output_dir / "reschedule_rule_eval_partial.csv"
    state_path = output_dir / "reschedule_rule_resume_state.json"
    assert partial_path.exists()
    assert state_path.exists()
    assert json.loads(state_path.read_text(encoding="utf-8"))["status"] == "interrupted"
    assert len(pd.read_csv(partial_path)) == 1

    monkeypatch.setattr(rule_eval_script, "_evaluate_rule_job", real_evaluate_job)
    with temporary_config(configs, overrides):
        summary = evaluate_reschedule_rules(
            data_path_or_dir=PROJECT_ROOT / "data" / "283.csv",
            scenario_path=scenario_path,
            baseline_path=baseline_path,
            methods=["NoReschedule", "SPTRepair"],
            num_runs=1,
            seed=123,
            output_dir=output_dir,
            verbose=False,
            parallel_workers=1,
        )

    rows = summary["rows"]
    assert [row["method"] for row in rows] == ["NoReschedule", "SPTRepair"]
    assert len(pd.read_csv(partial_path)) == 2
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert state["completed_jobs"] == 2
    assert state["pending_jobs"] == 0


def test_reschedule_rule_eval_rejects_resume_with_different_signature(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.csv"
    baseline_df = _write_greedy_baseline(baseline_path)
    scenario_path = tmp_path / "scenario.csv"
    _write_single_delay_scenario(baseline_df, scenario_path)
    output_dir = tmp_path / "signature_eval"

    overrides = _reschedule_overrides(baseline_path, scenario_path)
    overrides.update({"enable_shadow_mask_verification": False})
    with temporary_config(configs, overrides):
        evaluate_reschedule_rules(
            data_path_or_dir=PROJECT_ROOT / "data" / "283.csv",
            scenario_path=scenario_path,
            baseline_path=baseline_path,
            methods=["NoReschedule"],
            num_runs=1,
            seed=123,
            output_dir=output_dir,
            verbose=False,
            parallel_workers=1,
        )
        with pytest.raises(RuntimeError, match="签名不一致"):
            evaluate_reschedule_rules(
                data_path_or_dir=PROJECT_ROOT / "data" / "283.csv",
                scenario_path=scenario_path,
                baseline_path=baseline_path,
                methods=["SPTRepair"],
                num_runs=1,
                seed=123,
                output_dir=output_dir,
                verbose=False,
                parallel_workers=1,
            )
