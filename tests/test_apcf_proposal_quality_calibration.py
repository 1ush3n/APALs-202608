# -*- coding: utf-8 -*-
"""APCF 自由 Proposal 质量与评分校准审计契约。"""

from __future__ import annotations

from pathlib import Path

import pytest


def _state_key():
    from scripts.audit_apcf_proposal_quality_calibration import make_full_state_key

    return make_full_state_key(
        csv_sha256="csv",
        decision_count=7,
        task_id=11,
        station_id=2,
        anchor_team=(5, 1),
    )


def test_full_state_key_and_cache_keys_are_scope_complete() -> None:
    from scripts.audit_apcf_proposal_quality_calibration import (
        cache_key_anchor,
        cache_key_proposal,
    )

    state_key = _state_key()
    assert state_key == ("csv", 7, 11, 2, (1, 5))
    assert cache_key_anchor("checkpoint", state_key) == (
        "checkpoint", state_key, -4.0, 0.0
    )
    assert cache_key_proposal("checkpoint", state_key, (9, 3)) == (
        "checkpoint", state_key, (3, 9), -4.0, 0.0
    )
    assert cache_key_proposal("checkpoint", state_key, (9, 4)) != (
        "checkpoint", state_key, (3, 9), -4.0, 0.0
    )


def test_reused_online_row_requires_protocol_and_team_identity() -> None:
    from scripts.audit_apcf_proposal_quality_calibration import (
        validate_reused_online_row,
    )

    row = {
        "csv_sha256": "csv",
        "decision_count": "7",
        "task_id": "11",
        "station_id": "2",
        "anchor_team": "[1, 5]",
        "proposal_team": "[3, 9]",
        "proposal_available": "True",
        "selected": "True",
        "anchor_done": "True",
        "proposal_done": "True",
        "anchor_makespan": "100",
        "proposal_makespan": "90",
        "relative_gain": "0.1",
        "online_cache_id": "cache",
    }
    validate_reused_online_row(
        row,
        expected_state_key=_state_key(),
        expected_anchor_team=(1, 5),
        expected_proposal_team=(3, 9),
        expected_checkpoint_sha256="checkpoint",
        online_protocol={
            "continuation_prior_logit": -4.0,
            "temperature": 0.0,
            "deterministic": True,
            "branch_floor": 0.0,
        },
        expected_asset_sha256={"raw": "raw", "canonical": "canonical", "source": "source"},
        report_asset_sha256={"raw": "raw", "canonical": "canonical", "source": "source"},
    )
    bad = dict(row, proposal_team="[4, 9]")
    with pytest.raises(ValueError, match="proposal_team"):
        validate_reused_online_row(
            bad,
            expected_state_key=_state_key(),
            expected_anchor_team=(1, 5),
            expected_proposal_team=(3, 9),
            expected_checkpoint_sha256="checkpoint",
            online_protocol={
                "continuation_prior_logit": -4.0,
                "temperature": 0.0,
                "deterministic": True,
                "branch_floor": 0.0,
            },
            expected_asset_sha256={"raw": "raw", "canonical": "canonical", "source": "source"},
            report_asset_sha256={"raw": "raw", "canonical": "canonical", "source": "source"},
        )


def test_missing_state_selection_excludes_unavailable_and_reuses_only_uncovered() -> None:
    from scripts.audit_apcf_proposal_quality_calibration import (
        select_missing_available_states,
    )

    rows = [
        {"state_key": "a", "proposal_available": True, "online_cache_id": "x"},
        {"state_key": "b", "proposal_available": True, "online_cache_id": ""},
        {"state_key": "c", "proposal_available": False, "online_cache_id": ""},
        {"state_key": "d", "proposal_available": True, "online_cache_id": None},
    ]
    missing = select_missing_available_states(rows)
    assert [row["state_key"] for row in missing] == ["b", "d"]


def test_graph_bootstrap_is_graph_level_and_reproducible() -> None:
    from scripts.audit_apcf_proposal_quality_calibration import bootstrap_graph_means

    rows = [
        {"graph_id": "a", "relative_gain": 0.1},
        {"graph_id": "a", "relative_gain": 0.3},
        {"graph_id": "b", "relative_gain": -0.1},
    ]
    first = bootstrap_graph_means(rows, bootstrap_reps=200, seed=42)
    second = bootstrap_graph_means(rows, bootstrap_reps=200, seed=42)
    assert first == second
    assert first["graph_count"] == 2
    assert first["state_count"] == 3


def test_spearman_and_binary_metrics_are_null_safe() -> None:
    from scripts.audit_apcf_proposal_quality_calibration import (
        binary_metrics,
        spearman_or_null,
    )

    assert spearman_or_null([1, 2, 3], [1, 4, 9])["value"] == pytest.approx(1.0)
    assert spearman_or_null([1, 1], [1, 2])["value"] is None
    assert spearman_or_null([1, 1], [1, 2])["reason"] == "constant_input"
    metrics = binary_metrics([True, False, True], [True, True, False])
    assert metrics == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "support": 2,
    }
    empty = binary_metrics([], [])
    assert empty["precision"] is None and empty["recall"] is None and empty["f1"] is None


def test_candidate_and_policy_layers_are_explicitly_separate() -> None:
    from scripts.audit_apcf_proposal_quality_calibration import validate_layer_separation

    assert validate_layer_separation(candidate_count=504, policy_count=96)
    with pytest.raises(ValueError, match="candidate-level"):
        validate_layer_separation(candidate_count=96, policy_count=96)


def test_existing_output_directory_is_rejected(tmp_path: Path) -> None:
    from scripts.audit_apcf_proposal_quality_calibration import ensure_output_dir

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        ensure_output_dir(existing, pytest_root=tmp_path)


def test_fail_closed_on_invalid_online_episode() -> None:
    from scripts.audit_apcf_proposal_quality_calibration import validate_online_outcome

    with pytest.raises(ValueError, match="done"):
        validate_online_outcome({"done": False, "makespan": 10.0, "steps": 3})
    with pytest.raises(ValueError, match="有限"):
        validate_online_outcome({"done": True, "makespan": float("nan"), "steps": 3})

def test_policy_row_accepts_validated_anchor_proposal_pair() -> None:
    from scripts.audit_apcf_proposal_quality_calibration import _build_policy_row

    row = {
        "graph_id": "g",
        "csv_sha256": "csv",
        "decision_count": "7",
        "task_id": "11",
        "station_id": "2",
        "anchor_team": "[1, 5]",
        "proposal_team": "[3, 9]",
        "proposal_available": "True",
        "predicted_delta_A": "0.01",
        "gate_value": "0.5",
        "raw_gap": "0.1",
        "raw_branch": "1",
    }
    calibration = {
        "csv_sha256": "csv",
        "decision_count": "7",
        "task_id": "11",
        "station_id": "2",
        "anchor_team": "[1, 5]",
        "proposal_team": "[3, 9]",
        "proposal_available": "True",
        "required_team_size": "2",
        "hamming_distance": "2",
        "predicted_delta_A": "0.01",
        "gate_value": "0.5",
        "residual_term": "0.1",
        "raw_branch_logit_gap": "-3.9",
    }
    result = _build_policy_row(
        row=row,
        calibration_row=calibration,
        outcome={
            "makespan_anchor": 100.0,
            "makespan_proposal": 90.0,
            "anchor_steps": 5,
            "proposal_steps": 5,
        },
        source="supplemented_online",
        selected_by_prior={0.0: False},
    )
    assert result["relative_gain"] == pytest.approx(0.1)