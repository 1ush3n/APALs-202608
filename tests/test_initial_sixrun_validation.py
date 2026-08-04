from __future__ import annotations

import json
from pathlib import Path

from scripts import run_initial_sixrun_validation as validation


def _write_completed_run(run_dir: Path, *, legal: bool = True, hard_violation: int = 0) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "schedule.csv").write_text("TaskID,StationID,Team,Start,End,Duration\n0,0,[0],0,1,1\n", encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "makespan": 1.0,
                "reward": 0.5,
                "balance_std": 0.1,
                "duration_sec": 0.01,
                "worker_utilization": 0.6,
                "station_utilization": 0.7,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "legality_audit.json").write_text(
        json.dumps(
            {
                "scheduled_real_tasks": 1,
                "num_real_tasks": 1,
                "is_legal_against_current_data_duration": legal,
                "violations": {"worker_overlap_violation_count": hard_violation},
            }
        ),
        encoding="utf-8",
    )


def test_four_instance_sixrun_protocol_is_exact() -> None:
    assert validation.DATASETS == ("283", "680", "2338", "3182")
    assert validation.RUNS[0] == ("temp0_seed42", 0.0, 42)
    assert [seed for _name, temperature, seed in validation.RUNS if temperature == 0.01] == [42, 43, 44, 45, 46]
    assert validation.EXPECTED_RUN_COUNT == 24


def test_completed_run_requires_full_legal_zero_violation_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "real_283" / "temp0_seed42"
    _write_completed_run(run_dir)
    valid, reason, payload = validation.verify_completed_run(run_dir)
    assert valid, reason
    assert payload is not None
    assert payload["schedule_sha256"] == validation.sha256(run_dir / "schedule.csv")


def test_completed_run_rejects_illegal_or_hard_violation(tmp_path: Path) -> None:
    illegal_dir = tmp_path / "illegal"
    _write_completed_run(illegal_dir, legal=False)
    assert validation.verify_completed_run(illegal_dir)[0] is False

    violation_dir = tmp_path / "violation"
    _write_completed_run(violation_dir, hard_violation=1)
    assert validation.verify_completed_run(violation_dir)[0] is False
