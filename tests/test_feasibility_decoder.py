from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from baselines.heuristic.advanced_schedulers import (
    AdvancedSchedulerBase,
    IteratedGreedyScheduler,
)
from baselines.heuristic.feasibility_decoder import FeasibilityDecodeResult
from baselines.heuristic.run_all_baselines import compute_cpm_times, select_heuristic_action
from configs import configs
from data_loader import load_data
from environment import AirLineEnv_Graph
from core.constraints import ScheduleValidationReport


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dangling_predecessor_fails_fast(tmp_path: Path) -> None:
    data_path = tmp_path / "dangling.csv"
    data_path.write_text(
        "序号,AO号,类型,专业编码,工种,紧前工序AO号,需求人数,加工时间/h,限定站位,部位容量\n"
        "1,A,1,,-1,,0,0,,\n"
        "2,A-1,1,,-1,A,0,0,,\n"
        "3,ABCT00-0010,2,B,2,NOT-EXISTS,1,1.0,,\n",
        encoding="utf-8-sig",
    )

    with pytest.raises(ValueError, match="悬空前驱 AO"):
        load_data(data_path)


@pytest.fixture(scope="module")
def real_env() -> AirLineEnv_Graph:
    return AirLineEnv_Graph(PROJECT_ROOT / "data" / "283.csv", seed=42)


def test_legacy_selector_emits_resource_free_action_for_virtual_node(
    real_env: AirLineEnv_Graph,
) -> None:
    real_env.skip_obs_building = True
    real_env.reset(randomize_duration=False, randomize_workers=False, seed=42)
    es, ls = compute_cpm_times(real_env)

    action = select_heuristic_action(real_env, "LPT", es, ls)

    assert action is not None
    task_id, station_id, team = action
    assert not bool(real_env.constraint_engine.physical_mask[int(task_id)])
    assert station_id == -1
    assert team == []


@pytest.mark.parametrize("rule", ["SPT", "CPM", "MSL"])
def test_rule_decoder_handles_virtual_nodes_and_returns_legal_schedule(
    real_env: AirLineEnv_Graph,
    rule: str,
) -> None:
    scheduler = AdvancedSchedulerBase(real_env, seed=42)
    solution = scheduler.build_rule_solution(rule, 42)
    result = scheduler.decoder.decode(solution, 42)

    assert result.complete
    assert len(result.assigned_tasks) == real_env.num_tasks
    assert result.deadlock_count == 0
    assert result.validation_report is not None
    assert result.validation_report.is_legal
    assert result.diagnostics["preassigned_workers"] == int(
        round(real_env.num_workers * configs.heuristic_worker_preassignment_ratio)
    )
    assert result.diagnostics["reserve_oracle_active"]

    virtual_ids = set(np.where(~real_env.constraint_engine.physical_mask)[0].tolist())
    virtual_rows = [row for row in result.assigned_tasks if row[0] in virtual_ids]
    assert virtual_rows
    assert all(station == -1 and team == [] and start == end for _, station, team, start, end in virtual_rows)


def test_decoder_penalizes_illegal_complete_candidate_without_raising(
    real_env: AirLineEnv_Graph,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = AdvancedSchedulerBase(real_env, seed=42)
    solution = scheduler.build_rule_solution("SPT", 42)
    forced_report = ScheduleValidationReport(
        violations={"station_slot_violation_count": 1},
        examples={
            "station_slot_violation_count": [
                {
                    "station_id": 0,
                    "time": 1.0,
                    "active_count": 4,
                    "capacity": 3,
                    "active_task_ids": [1, 2, 3, 4],
                }
            ]
        },
    )
    monkeypatch.setattr(real_env, "validate_assignments", lambda _assignments: forced_report)

    result = scheduler.decoder.decode(solution, 42)

    assert not result.complete
    assert result.failure_type == "illegal_schedule"
    assert np.isinf(result.fitness)
    assert result.assigned_tasks == []
    assert result.deadlock_count == 0
    assert result.diagnostics["invalid_schedule_count"] == 1
    assert result.diagnostics["failure_details"]["examples"] == forced_report.examples


def test_iterated_greedy_rejects_illegal_candidate_and_keeps_legal_best(
    real_env: AirLineEnv_Graph,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = IteratedGreedyScheduler(real_env, iterations=1, seed=42)
    legal_schedule = [(0, -1, [], 0.0, 0.0)]
    legal = FeasibilityDecodeResult(
        fitness=10.0,
        makespan=10.0,
        balance_std=0.0,
        assigned_tasks=legal_schedule,
        complete=True,
        deadlock_count=0,
    )
    illegal = FeasibilityDecodeResult(
        fitness=float("inf"),
        makespan=30.0,
        balance_std=3.0,
        assigned_tasks=[],
        complete=False,
        deadlock_count=0,
        failure_type="illegal_schedule",
        diagnostics={
            "failure_details": {"violations": {"station_slot_violation_count": 1}}
        },
    )
    results = iter([legal, illegal])
    monkeypatch.setattr(scheduler.decoder, "decode", lambda _solution, _seed: next(results))

    makespan, balance_std, assigned_tasks = scheduler.run()

    assert makespan == 10.0
    assert balance_std == 0.0
    assert assigned_tasks == legal_schedule
    assert scheduler.search_diagnostics()["illegal_candidate_count"] == 1
