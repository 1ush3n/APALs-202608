from __future__ import annotations

import numpy as np
import pytest

from core.constraints import ConstraintEngine
from environment import AirLineEnv_Graph


def _engine(*, fixed: list[int] | None = None) -> ConstraintEngine:
    # 物理工序 0 -> 虚拟节点 1 -> 物理工序 2。
    return ConstraintEngine.build(
        num_tasks=3,
        num_stations=5,
        edges=np.asarray([[0, 1], [1, 2]], dtype=np.int64),
        durations=[2.0, 0.0, 3.0],
        fixed_stations=fixed or [-1, -1, -1],
        max_allowed_stations=[4, 4, 4],
    )


def test_physical_precedence_is_compressed_through_virtual_nodes() -> None:
    engine = _engine()

    assert engine.physical_predecessors[2] == (0,)
    assert engine.minimum_station(2, {0: 3, 1: -1}) == 3
    violation = engine.station_violation(2, 2, {0: 3, 1: -1})
    assert violation is not None
    assert violation["reason"] == "physical_precedence_station_violation"


def test_virtual_fixed_station_propagates_to_first_physical_successor() -> None:
    engine = _engine(fixed=[-1, 2, -1])

    assert int(engine.fixed_stations[2]) == 2
    violation = engine.station_violation(2, 1, {0: 1})
    assert violation is not None
    assert violation["reason"] == "fixed_station_violation"


def test_conflicting_virtual_and_physical_fixed_stations_fail_fast() -> None:
    with pytest.raises(ValueError, match="固定站位"):
        _engine(fixed=[-1, 2, 1])


def test_complete_schedule_report_detects_spatial_binding_and_exact_demand() -> None:
    engine = _engine()
    report = engine.validate_schedule(
        [
            (0, 3, [0], 0.0, 2.0),
            (1, -1, [], 2.0, 2.0),
            (2, 2, [0], 2.0, 5.0),
        ],
        demands=[1, 1, 2],
        required_skills=[0, 0, 0],
        worker_skill_matrix=np.ones((2, 1), dtype=float),
        max_slots_per_station=1,
    )

    assert report.violations["physical_station_violation_count"] == 1
    assert report.violations["worker_station_binding_violation_count"] == 1
    assert report.violations["demand_violation_count"] == 1
    assert not report.is_legal


def test_complete_schedule_report_accepts_legal_compressed_path() -> None:
    engine = _engine()
    report = engine.validate_schedule(
        [
            (0, 1, [0], 0.0, 2.0),
            (1, -1, [], 2.0, 2.0),
            (2, 2, [1], 2.0, 5.0),
        ],
        demands=[1, 1, 1],
        required_skills=[0, 0, 0],
        worker_skill_matrix=np.ones((2, 1), dtype=float),
        max_slots_per_station=1,
    )

    assert report.is_legal


def test_station_allocator_snaps_tiny_overlap_to_exact_finish_boundary() -> None:
    class _StationState:
        station_available_slots = np.asarray([3], dtype=int)
        assigned_tasks = [
            (0, 0, [0], 0.0, 10.0),
            (1, 0, [1], 0.0, 10.0),
            (2, 0, [2], 0.0, 10.0),
        ]

    state = _StationState()
    proposed_start = 10.0 - 5.0e-6
    accepted_start = AirLineEnv_Graph._get_station_earliest_available_time(
        state,
        0,
        proposed_start,
        1.0,
    )

    assert accepted_start == 10.0

    engine = ConstraintEngine.build(
        num_tasks=4,
        num_stations=1,
        edges=np.empty((2, 0), dtype=np.int64),
        durations=[10.0, 10.0, 10.0, 1.0],
        fixed_stations=[-1, -1, -1, -1],
    )
    report = engine.validate_schedule(
        [*state.assigned_tasks, (3, 0, [3], accepted_start, accepted_start + 1.0)],
        demands=[1, 1, 1, 1],
        required_skills=[0, 0, 0, 0],
        worker_skill_matrix=np.ones((4, 1), dtype=float),
        max_slots_per_station=3,
    )

    assert report.is_legal


def test_station_slot_violation_reports_station_time_and_active_tasks() -> None:
    engine = ConstraintEngine.build(
        num_tasks=4,
        num_stations=1,
        edges=np.empty((2, 0), dtype=np.int64),
        durations=[10.0, 10.0, 10.0, 1.0],
        fixed_stations=[-1, -1, -1, -1],
    )
    report = engine.validate_schedule(
        [
            (0, 0, [0], 0.0, 10.0),
            (1, 0, [1], 0.0, 10.0),
            (2, 0, [2], 0.0, 10.0),
            (3, 0, [3], 9.0, 11.0),
        ],
        demands=[1, 1, 1, 1],
        required_skills=[0, 0, 0, 0],
        worker_skill_matrix=np.ones((4, 1), dtype=float),
        max_slots_per_station=3,
    )

    example = report.examples["station_slot_violation_count"][0]
    assert example == {
        "station_id": 0,
        "time": 9.0,
        "active_count": 4,
        "capacity": 3,
        "active_task_ids": [0, 1, 2, 3],
    }
