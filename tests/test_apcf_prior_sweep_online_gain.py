# -*- coding: utf-8 -*-
"""APCF prior sweep online gain contracts."""

from __future__ import annotations

import math

import pytest


def test_relative_gain_matches_asset_semantics_and_rejects_nonfinite() -> None:
    from scripts.diagnose_apcf_prior_sweep_online_gain import relative_gain

    assert relative_gain(100.0, 90.0) == pytest.approx(0.1)
    assert relative_gain(100.0, 110.0) == pytest.approx(-0.1)
    with pytest.raises(ValueError, match="有限"):
        relative_gain(float("nan"), 1.0)
    with pytest.raises(ValueError, match="正"):
        relative_gain(0.0, 1.0)


def test_bootstrap_resamples_graph_means_deterministically() -> None:
    from scripts.diagnose_apcf_prior_sweep_online_gain import bootstrap_graph_means

    rows = [
        {"graph_id": "a", "relative_gain": 0.1},
        {"graph_id": "a", "relative_gain": 0.3},
        {"graph_id": "b", "relative_gain": 0.0},
        {"graph_id": "c", "relative_gain": -0.1},
    ]
    first = bootstrap_graph_means(rows, bootstrap_reps=200, seed=42)
    second = bootstrap_graph_means(rows, bootstrap_reps=200, seed=42)
    assert first == second
    assert first["graph_count"] == 3
    assert first["state_count"] == 4
    assert first["point_estimate"] == pytest.approx((0.2 + 0.0 - 0.1) / 3.0)
    assert math.isfinite(float(first["ci_low"]))
    assert math.isfinite(float(first["ci_high"]))


def test_prior_margin_override_restores_on_exception() -> None:
    from scripts.diagnose_apcf_prior_sweep_online_gain import prior_margin_override

    class Gate:
        prior_margin = 4.0

    gate = Gate()
    with pytest.raises(RuntimeError, match="sentinel"):
        with prior_margin_override(gate, prior_logit=-2.0):
            assert gate.prior_margin == pytest.approx(2.0)
            raise RuntimeError("sentinel")
    assert gate.prior_margin == pytest.approx(4.0)


def test_cache_keys_include_checkpoint_state_and_policy_scope() -> None:
    from scripts.diagnose_apcf_prior_sweep_online_gain import (
        make_anchor_cache_key,
        make_proposal_cache_key,
        make_state_key,
    )

    state_key = make_state_key(
        csv_sha256="csv",
        decision_count=7,
        task_id=11,
        station_id=2,
        anchor_team=(5, 1),
    )
    assert state_key == ("csv", 7, 11, 2, (1, 5))
    anchor_key = make_anchor_cache_key(
        checkpoint_sha256="checkpoint",
        state_key=state_key,
        continuation_prior_logit=-4.0,
        temperature=0.0,
    )
    proposal_key = make_proposal_cache_key(
        checkpoint_sha256="checkpoint",
        state_key=state_key,
        proposal_team=(9, 3),
        continuation_prior_logit=-4.0,
        temperature=0.0,
    )
    assert anchor_key == ("checkpoint", state_key, -4.0, 0.0)
    assert proposal_key == ("checkpoint", state_key, (3, 9), -4.0, 0.0)
    assert make_proposal_cache_key(
        checkpoint_sha256="checkpoint",
        state_key=state_key,
        proposal_team=(9, 4),
        continuation_prior_logit=-4.0,
        temperature=0.0,
    ) != proposal_key


def test_prior_sweep_counts_are_checked_against_existing_summary() -> None:
    from scripts.diagnose_apcf_prior_sweep_online_gain import (
        validate_prior_sweep_counts,
    )

    expected = {-4.0: 0, -2.0: 9, -1.0: 33, 0.0: 63}
    assert validate_prior_sweep_counts(
        expected,
        {-4.0: 0, -2.0: 9, -1.0: 33, 0.0: 63},
    )
    with pytest.raises(ValueError, match="prior sweep"):
        validate_prior_sweep_counts(expected, {-4.0: 0, -2.0: 8, -1.0: 33, 0.0: 63})


@pytest.mark.parametrize(
    ("selected_graph_count", "ci_low", "expected"),
    (
        (0, None, "not_selected"),
        (7, 0.5, "insufficient_evidence"),
        (8, 0.0, "rejected"),
        (8, 0.01, "positive_evidence"),
    ),
)
def test_admission_status_has_explicit_evidence_semantics(
    selected_graph_count: int,
    ci_low: float | None,
    expected: str,
) -> None:
    from scripts.diagnose_apcf_prior_sweep_online_gain import classify_admission

    assert classify_admission(
        selected_graph_count=selected_graph_count,
        ci_low=ci_low,
    ) == expected


def test_unavailable_state_fields_are_null_but_available_unselected_is_retained() -> None:
    from scripts.diagnose_apcf_prior_sweep_online_gain import make_state_prior_record

    unavailable = make_state_prior_record(
        state_key=("csv", 1, 2, 3, (4,)),
        prior_logit=-2.0,
        proposal_available=False,
        selected=False,
        anchor_team=(4,),
        proposal_team=None,
        raw_gap=None,
    )
    assert unavailable["selected"] is False
    assert unavailable["proposal_team"] is None
    assert unavailable["raw_gap"] is None
    available = make_state_prior_record(
        state_key=("csv", 1, 2, 3, (4,)),
        prior_logit=-2.0,
        proposal_available=True,
        selected=False,
        anchor_team=(4,),
        proposal_team=(5,),
        raw_gap=-0.2,
    )
    assert available["proposal_team"] == [5]
    assert available["raw_gap"] == pytest.approx(-0.2)
    assert available["relative_gain"] is None

def test_observation_verification_summary_is_fail_closed() -> None:
    from scripts.diagnose_apcf_prior_sweep_online_gain import observation_verification_summary

    snapshots = {
        1: {
            "observation_verification": {
                "state_key_verified": True,
                "recursive_payload_equal": True,
                "worker_mask_equal": True,
                "persisted_obs_pt_sha256_checked": True,
            }
        },
        2: {
            "observation_verification": {
                "state_key_verified": True,
                "recursive_payload_equal": True,
                "worker_mask_equal": True,
                "persisted_obs_pt_sha256_checked": False,
            }
        },
    }
    summary = observation_verification_summary(snapshots)
    assert summary["status"] == "passed"
    assert summary["state_count"] == 2
    assert summary["persisted_obs_pt_sha256_checked_count"] == 1

    snapshots[2]["observation_verification"]["worker_mask_equal"] = False
    with pytest.raises(ValueError, match="观测"):
        observation_verification_summary(snapshots)