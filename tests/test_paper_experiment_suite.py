from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from runtime.paper_metrics import (
    collect_result_rows,
    command_plan_rows,
    dataset_profile_rows,
    efficiency_pareto_rows,
    load_paper_config,
    policy_behavior_rows,
    schedule_behavior_metrics,
    significance_test_rows,
    statistical_summary_rows,
)


def test_statistical_summary_and_significance_use_seeded_rows() -> None:
    rows = [
        {"method": "full", "dataset": "680", "makespan": 300.0, "seed": 0},
        {"method": "full", "dataset": "680", "makespan": 310.0, "seed": 1},
        {"method": "full", "dataset": "680", "makespan": 305.0, "seed": 2},
        {"method": "baseline", "dataset": "680", "makespan": 400.0, "seed": 0},
        {"method": "baseline", "dataset": "680", "makespan": 410.0, "seed": 1},
        {"method": "baseline", "dataset": "680", "makespan": 405.0, "seed": 2},
    ]

    summary = statistical_summary_rows(
        rows,
        reference_makespans={"680": 300.0},
        reference_method="full",
        bootstrap_samples=50,
    )
    full = next(row for row in summary if row["method"] == "full")
    assert full["num_runs"] == 3
    assert full["mean"] == 305.0
    assert full["normalized_mean"] > 1.0

    tests = significance_test_rows(
        rows,
        reference_method="full",
        permutation_samples=50,
    )
    assert tests[0]["status"] == "ok"
    assert tests[0]["mean_improvement"] > 0.0
    assert "holm_adjusted_p" in tests[0]


def test_schedule_behavior_metrics_compute_resource_gini(tmp_path: Path) -> None:
    schedule_path = tmp_path / "results" / "eval_logs" / "full" / "680" / "schedule.csv"
    schedule_path.parent.mkdir(parents=True)
    frame = pd.DataFrame(
        [
            {"TaskID": 1, "StationID": 1, "Team": "[1, 2]", "Start": 0.0, "End": 2.0, "Duration": 2.0},
            {"TaskID": 2, "StationID": 2, "Team": "[2]", "Start": 2.0, "End": 5.0, "Duration": 3.0},
            {"TaskID": 3, "StationID": 2, "Team": "[]", "Start": 0.0, "End": 0.0, "Duration": 0.0},
        ]
    )
    frame.to_csv(schedule_path, index=False)

    row = schedule_behavior_metrics(schedule_path, frame)
    assert row["method"] == "full"
    assert row["dataset"] == "680"
    assert row["active_worker_count"] == 2
    assert row["active_station_count"] == 2
    assert row["team_size_mean"] == 1.0
    assert row["worker_load_gini"] >= 0.0

    discovered = policy_behavior_rows([tmp_path / "results"])
    assert len(discovered) == 1


def test_dataset_profile_rows_use_apal_loader(tmp_path: Path) -> None:
    data_path = tmp_path / "mini.csv"
    pd.DataFrame(
        [
            {"AO号": "A", "类型": 1, "紧前工序AO号": "", "需求人数": 0, "加工时间/h": 0.0, "限定站位": ""},
            {"AO号": "A-1", "类型": 1, "紧前工序AO号": "", "需求人数": 0, "加工时间/h": 0.0, "限定站位": ""},
            {"AO号": "AAQS00-0010", "类型": 2, "紧前工序AO号": "", "需求人数": 2, "加工时间/h": 1.5, "限定站位": 1},
            {"AO号": "AAQS00-0020", "类型": 3, "紧前工序AO号": "AAQS00-0010", "需求人数": 1, "加工时间/h": 2.5, "限定站位": ""},
        ]
    ).to_csv(data_path, index=False, encoding="utf-8-sig")

    rows = dataset_profile_rows({"mini": data_path}, station_count=5, worker_count=80)
    assert rows[0]["dataset"] == "mini"
    assert rows[0]["node_count"] == 4
    assert rows[0]["real_task_count"] == 2
    assert rows[0]["critical_path_lower_bound"] >= 2.5
    assert rows[0]["skill_entropy"] > 0.0


def test_command_plan_marks_missing_checkpoints(tmp_path: Path) -> None:
    config_path = tmp_path / "paper.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: paper",
                "output_root: results/paper_experiments",
                "datasets:",
                "  '680': data/680.csv",
                "reference_makespans:",
                "  '680': 316.0",
                "ablations:",
                "  full:",
                "    experiment: scale_400_800_schedule",
                "    overrides: []",
                "checkpoints:",
                "  full:",
                "    '680': missing.ckpt",
            ]
        ),
        encoding="utf-8",
    )
    config = load_paper_config(config_path, tmp_path)
    rows = command_plan_rows(config)
    eval_rows = [row for row in rows if row.get("suite") == "generalization"]
    assert eval_rows[0]["status"] == "missing_checkpoint"


def test_efficiency_pareto_marks_fast_non_dominated_points() -> None:
    rows = [
        {"method": "fast", "dataset": "680", "makespan": 500.0, "inference_time": 1.0},
        {"method": "middle", "dataset": "680", "makespan": 420.0, "inference_time": 2.0},
        {"method": "slow_bad", "dataset": "680", "makespan": 430.0, "inference_time": 3.0},
    ]
    pareto = efficiency_pareto_rows(rows, reference_makespans={"680": 316.0})
    by_method = {row["method"]: row for row in pareto}
    assert by_method["fast"]["pareto_efficient"] == 1.0
    assert by_method["middle"]["pareto_efficient"] == 1.0
    assert by_method["slow_bad"]["pareto_efficient"] == 0.0


def test_bare_evaluation_summary_is_treated_as_full_method(tmp_path: Path) -> None:
    summary_path = tmp_path / "results" / "summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        json.dumps(
            {
                "resource_graph_mode": "legacy_direct",
                "data_path": str(tmp_path / "data" / "680.csv"),
                "makespan": 316.0,
                "worker_utilization": 0.2,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rows = collect_result_rows([tmp_path / "results"])
    assert rows[0]["method"] == "full"
    assert rows[0]["dataset"] == "680"
