# -*- coding: utf-8 -*-
"""验证 APAL 重调度低/中/高扰动场景库生成。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.generate_reschedule_load_scenarios import write_scenario_library
from tests.test_reschedule_task_delay import _write_greedy_baseline
from utils.reschedule import load_baseline_schedule, load_reschedule_scenarios


def test_reschedule_load_grid_is_seed_reproducible(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.csv"
    _write_greedy_baseline(baseline_path)

    output_a = tmp_path / "scenarios_a.csv"
    meta_a = tmp_path / "scenarios_a.metadata.json"
    output_b = tmp_path / "scenarios_b.csv"
    meta_b = tmp_path / "scenarios_b.metadata.json"

    metadata_a = write_scenario_library(
        baseline_path=baseline_path,
        output_path=output_a,
        metadata_path=meta_a,
        seed=20260701,
        scenarios_per_level=4,
    )
    metadata_b = write_scenario_library(
        baseline_path=baseline_path,
        output_path=output_b,
        metadata_path=meta_b,
        seed=20260701,
        scenarios_per_level=4,
    )

    assert output_a.read_text(encoding="utf-8") == output_b.read_text(encoding="utf-8")
    assert metadata_a["scenario_stats"] == metadata_b["scenario_stats"]
    assert metadata_a["scenario_count"] == 12
    assert {row["level"] for row in metadata_a["scenario_stats"]} == {"low", "medium", "high"}


def test_reschedule_load_grid_is_loader_compatible_and_valid(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.csv"
    _write_greedy_baseline(baseline_path)
    output_path = tmp_path / "scenarios.csv"
    metadata_path = tmp_path / "scenarios.metadata.json"

    metadata = write_scenario_library(
        baseline_path=baseline_path,
        output_path=output_path,
        metadata_path=metadata_path,
        seed=7,
        scenarios_per_level=5,
    )

    scenarios = load_reschedule_scenarios(output_path)
    df = pd.read_csv(output_path)
    baseline = load_baseline_schedule(baseline_path)
    assert len(scenarios) == 15
    assert set(df["level"].unique()) == {"low", "medium", "high"}
    assert {"scenario_id", "reschedule_start_time", "TaskID", "release_time", "eligible_delay_task_count"} <= set(df.columns)

    for row in df.itertuples(index=False):
        task_id = int(row.TaskID)
        if task_id < 0:
            continue
        base = baseline.tasks[task_id]
        assert base.start > float(row.reschedule_start_time)
        assert base.duration > 1e-8
        assert len(base.team) > 0
        assert float(row.release_time) + 1e-8 >= max(float(base.start), float(row.reschedule_start_time))

    aggregate = {row["level"]: row for row in metadata["aggregate_by_level"]}
    assert aggregate["high"]["avg_delayed_task_count"] > aggregate["low"]["avg_delayed_task_count"]
    assert aggregate["high"]["avg_delay_mean_h"] > aggregate["low"]["avg_delay_mean_h"]
