import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
from types import SimpleNamespace

from utils.reschedule import (
    BaselineSchedule,
    BaselineTask,
    calculate_stability_metrics,
    calculate_reschedule_objective_terms,
    calculate_reschedule_composite_score,
)

def test_stability_metrics():
    # Baseline with 3 tasks
    tasks = {
        1: BaselineTask(task_id=1, station_id=0, team=[0, 1], start=10.0, end=20.0, duration=10.0),
        2: BaselineTask(task_id=2, station_id=1, team=[2, 3], start=30.0, end=50.0, duration=20.0),
        3: BaselineTask(task_id=3, station_id=2, team=[4, 5], start=40.0, end=70.0, duration=30.0),
    }
    baseline = BaselineSchedule(tasks=tasks, makespan=70.0)
    
    # Reschedule at current_time = 25.0 (task 1 is frozen, tasks 2 and 3 are movable)
    # Task 2: start=35.0 (+5h), station=1 (same), team=[3, 2] (same set, different order)
    # Task 3: start=42.0 (+2h), station=3 (changed), team=[4, 6] (changed)
    assigned = [
        (1, 0, [0, 1], 10.0, 20.0),
        (2, 1, [3, 2], 35.0, 55.0),
        (3, 3, [4, 6], 42.0, 72.0),
    ]
    
    metrics = calculate_stability_metrics(baseline, assigned, current_time=25.0)
    assert metrics["movable_count"] == 2.0
    # Mean start deviation: (|35-30| + |42-40|) / 2 = 7 / 2 = 3.5
    assert np.isclose(metrics["start_deviation_mean_h"], 3.5)
    # Station change rate: 1 changed / 2 = 0.5
    assert np.isclose(metrics["station_change_rate"], 0.5)
    # Team change rate: task 2 has {2,3}=={3,2} (0 changed), task 3 has {4,6}!={4,5} (1 changed) -> 1/2 = 0.5
    assert np.isclose(metrics["team_change_rate"], 0.5)

def test_reweighted_objective_terms_and_score():
    cfg = SimpleNamespace(
        r_coef_makespan=0.20,
        r_coef_std=0.0,
        reschedule_takt_violation_weight=3.0,
        reschedule_stability_start_weight=4.0,
        reschedule_stability_station_weight=4.0,
        reschedule_stability_team_weight=0.3,
    )
    
    # Takt = 100.0, makespan = 110.0 (violation = 10.0), start_dev = 5.0, station_change = 0.2, team_change = 0.1, balance_std = 15.0, ideal_load = 50.0
    terms = calculate_reschedule_objective_terms(
        makespan=110.0,
        balance_std=15.0,
        takt_h=100.0,
        takt_violation_h=10.0,
        start_deviation_mean_h=5.0,
        station_change_rate=0.2,
        team_change_rate=0.1,
        config_obj=cfg,
        ideal_station_load=50.0,
    )
    
    # Check each term:
    # score_makespan = 0.20 * 110 / 100 = 0.22
    assert np.isclose(terms["score_makespan"], 0.22)
    # score_balance = 0.0 * 15 / 50 = 0.0
    assert np.isclose(terms["score_balance"], 0.0)
    # score_takt_violation = 3.0 * 10 / 100 = 0.30
    assert np.isclose(terms["score_takt_violation"], 0.30)
    # score_start_stability = 4.0 * 5.0 / 100 = 0.20
    assert np.isclose(terms["score_start_stability"], 0.20)
    # score_station_change = 4.0 * 0.2 = 0.80
    assert np.isclose(terms["score_station_change"], 0.80)
    # score_team_change = 0.3 * 0.1 = 0.03
    assert np.isclose(terms["score_team_change"], 0.03)
    
    # Sum: 0.22 + 0.0 + 0.30 + 0.20 + 0.80 + 0.03 = 1.55
    expected_score = 0.22 + 0.30 + 0.20 + 0.80 + 0.03
    
    # Eligible case
    constraint_metrics = {
        "complete": 1.0,
        "takt_h": 100.0,
        "takt_violation_h": 10.0,
        "start_deviation_mean_h": 5.0,
        "station_change_rate": 0.2,
        "team_change_rate": 0.1,
        "precedence_violation_count": 0.0,
        "worker_overlap_violation_count": 0.0,
    }
    score_res = calculate_reschedule_composite_score(
        makespan=110.0,
        balance_std=15.0,
        constraint_metrics=constraint_metrics,
        config_obj=cfg,
        ideal_station_load=50.0,
    )
    assert score_res.eligible is True
    assert np.isclose(score_res.score, expected_score)
    assert np.isclose(score_res.selection_score, expected_score)
    
    # Ineligible case (hard violation)
    constraint_metrics["precedence_violation_count"] = 1.0
    score_res_ineligible = calculate_reschedule_composite_score(
        makespan=110.0,
        balance_std=15.0,
        constraint_metrics=constraint_metrics,
        config_obj=cfg,
        ideal_station_load=50.0,
    )
    assert score_res_ineligible.eligible is False
    assert np.isclose(score_res_ineligible.score, expected_score)
    assert np.isclose(score_res_ineligible.selection_score, 1e9 + expected_score)
