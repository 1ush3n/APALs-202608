# -*- coding: utf-8 -*-
"""验证 GA 重调度基线与 PPO 重调度使用完全相同的固定条件和硬约束。"""

from __future__ import annotations

import sys
from pathlib import Path
import py_compile

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.heuristic.reschedule_ga import evaluate_reschedule_ga
from configs import Config, configs
from tests.runtime_safety import temporary_config
from tests.test_reschedule_task_delay import _reschedule_overrides, _write_greedy_baseline
from utils.reschedule import calculate_reschedule_composite_score


def test_reschedule_ga_entrypoints_are_syntax_valid() -> None:
    py_compile.compile(str(PROJECT_ROOT / "baselines" / "heuristic" / "reschedule_ga.py"), doraise=True)
    py_compile.compile(str(PROJECT_ROOT / "scripts" / "evaluate_reschedule_ga.py"), doraise=True)


def test_reschedule_composite_score_uses_normalized_terms() -> None:
    cfg = Config()
    cfg.r_coef_std = 0.5
    cfg.reschedule_takt_violation_weight = 1.0
    cfg.reschedule_stability_start_weight = 0.2
    cfg.reschedule_stability_station_weight = 0.1
    cfg.reschedule_stability_team_weight = 0.05
    metrics = {
        "complete": 1.0,
        "takt_h": 100.0,
        "takt_violation_h": 5.0,
        "start_deviation_mean_h": 10.0,
        "station_change_rate": 0.2,
        "team_change_rate": 0.4,
    }

    score = calculate_reschedule_composite_score(
        makespan=110.0,
        balance_std=20.0,
        constraint_metrics=metrics,
        config_obj=cfg,
        ideal_station_load=50.0,
    )

    assert score.eligible is True
    assert score.terms["score_makespan"] == 1.1
    assert score.terms["score_balance"] == 0.2
    assert score.terms["score_takt_violation"] == 0.05
    assert score.terms["score_start_stability"] == 0.02
    assert score.terms["score_station_change"] == 0.020000000000000004
    assert score.terms["score_team_change"] == 0.020000000000000004
    assert abs(score.score - 1.41) < 1e-9
    assert score.selection_score == score.score


def test_reschedule_composite_score_rejects_hard_violations() -> None:
    cfg = Config()
    metrics = {"complete": 1.0, "takt_h": 100.0, "release_violation_count": 1.0}

    score = calculate_reschedule_composite_score(
        makespan=90.0,
        balance_std=1.0,
        constraint_metrics=metrics,
        config_obj=cfg,
        ideal_station_load=50.0,
    )

    assert score.eligible is False
    assert score.selection_score > 1.0e8


def _write_two_fixed_scenarios(path: Path, baseline_df: pd.DataFrame) -> None:
    rows = []
    for scenario_idx, quantile in enumerate((0.25, 0.35)):
        start_time = float(baseline_df["Start"].quantile(quantile))
        delayed = baseline_df[baseline_df["Start"] > start_time].iloc[scenario_idx]
        rows.append(
            {
                "scenario_id": f"eval_{scenario_idx:03d}",
                "reschedule_start_time": start_time,
                "TaskID": int(delayed["TaskID"]),
                "release_time": float(delayed["Start"] + 4.0 + scenario_idx),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_reschedule_ga_uses_fixed_scenarios_and_writes_summary(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.csv"
    scenario_path = tmp_path / "fixed_scenarios.csv"
    baseline_df = _write_greedy_baseline(baseline_path)
    _write_two_fixed_scenarios(scenario_path, baseline_df)

    overrides = _reschedule_overrides(baseline_path, scenario_path)
    overrides.update({"reschedule_eval_num_scenarios": 2, "n_w": 80})
    output_dir = tmp_path / "ga_eval"

    with temporary_config(configs, overrides):
        summary = evaluate_reschedule_ga(
            pop_size=2,
            max_gen=1,
            num_runs=2,
            seed=123,
            output_dir=output_dir,
            verbose=False,
        )

    result_csv = output_dir / "reschedule_ga_eval.csv"
    assert result_csv.exists()
    assert summary["scenario_count"] == 2
    assert Path(summary["scenario_path"]) == scenario_path.resolve()

    result_df = pd.read_csv(result_csv)
    assert set(result_df["scenario_id"]) == {"eval_000", "eval_001"}
    assert (result_df["frozen_violation_count"] == 0.0).all()
    assert (result_df["release_violation_count"] == 0.0).all()
    assert (result_df["precedence_violation_count"] == 0.0).all()
    assert (result_df["worker_overlap_violation_count"] == 0.0).all()
    assert (result_df["station_slot_violation_count"] == 0.0).all()
    assert (result_df["skill_violation_count"] == 0.0).all()
    assert (result_df["demand_violation_count"] == 0.0).all()
    assert (result_df["duplicate_task_count"] == 0.0).all()
    assert (result_df["takt_h"] == float(baseline_df["End"].max())).all()
    for column in [
        "score",
        "selection_score",
        "eligible",
        "score_makespan",
        "score_balance",
        "score_takt_violation",
        "score_start_stability",
        "score_station_change",
        "score_team_change",
    ]:
        assert column in result_df.columns
