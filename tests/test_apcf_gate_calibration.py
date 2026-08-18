# -*- coding: utf-8 -*-
"""APCF gate calibration tests."""

from __future__ import annotations

import math

import pytest
import torch


def test_reconstructed_gap_uses_checkpoint_formula_and_anchor_tie_break() -> None:
    from scripts.diagnose_apcf_gate_calibration import (
        FormulaSpec,
        expected_raw_branch,
        reconstruct_raw_gap,
    )

    spec = FormulaSpec(
        prior_logit=-4.0,
        residual_scale=6.0,
        delta_temperature=0.01,
        anchor_proposal_mode="full_team_v1",
        source="test",
    )
    assert reconstruct_raw_gap(spec, gate_value=2.0 / 3.0, predicted_delta_a=1.0) == pytest.approx(0.0)
    assert expected_raw_branch(0.0) == 0
    assert expected_raw_branch(1.0e-7) == 1


def test_prior_sweep_changes_only_prior_logit() -> None:
    from scripts.diagnose_apcf_gate_calibration import (
        FormulaSpec,
        reconstruct_residual_term,
        sweep_gap,
    )

    spec = FormulaSpec(
        prior_logit=-4.0,
        residual_scale=6.0,
        delta_temperature=0.01,
        anchor_proposal_mode="full_team_v1",
        source="test",
    )
    residual = reconstruct_residual_term(spec, gate_value=1.0, predicted_delta_a=0.01)
    assert sweep_gap(-4.0, residual) == pytest.approx(0.5695649357)
    assert sweep_gap(-2.0, residual) == pytest.approx(2.5695649357)
    assert spec.prior_logit == -4.0


def test_unavailable_policy_fields_are_json_null() -> None:
    from scripts.diagnose_apcf_gate_calibration import policy_state_fields

    row = policy_state_fields(
        proposal_available=False,
        proposal_team=(1, 2),
        hamming_distance=2,
        required_team_size=2,
        gate_logit=0.5,
        gate_value=0.6,
        predicted_delta_a=0.01,
        raw_branch_logit_gap=-1.0,
        production_raw_branch=0,
    )
    assert row["proposal_team"] is None
    assert row["hamming_distance"] is None
    assert row["normalized_hamming_distance"] is None
    assert row["gate_logit"] is None
    assert row["gate_value"] is None
    assert row["predicted_delta_A"] is None
    assert row["raw_branch_logit_gap"] is None
    assert row["production_raw_branch"] is None


def test_normalized_hamming_distance_respects_required_team_size() -> None:
    from scripts.diagnose_apcf_gate_calibration import normalize_hamming_distance

    assert normalize_hamming_distance(1, 1) == pytest.approx(1.0)
    assert normalize_hamming_distance(1, 2) == pytest.approx(0.5)
    assert normalize_hamming_distance(2, 4) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="required_team_size"):
        normalize_hamming_distance(1, 0)


def test_production_branch_is_not_reconstructed_from_diagnostic_branch() -> None:
    from scripts.diagnose_apcf_gate_calibration import validate_production_branch

    assert validate_production_branch(1, 0.25, proposal_available=True)
    assert validate_production_branch(0, -0.25, proposal_available=True)
    assert validate_production_branch(0, 0.0, proposal_available=True)
    with pytest.raises(AssertionError, match="production raw branch"):
        validate_production_branch(0, 0.25, proposal_available=True)


def test_candidate_scores_require_numeric_production_equivalence() -> None:
    from scripts.diagnose_apcf_gate_calibration import assert_candidate_scores_close

    production = {
        "gate_logit": 0.125,
        "predicted_delta_A": 0.002,
        "raw_gap": -2.8,
    }
    candidate = {
        "gate_logit": 0.1250000005,
        "predicted_delta_A": 0.0020000005,
        "raw_gap": -2.800000001,
    }
    assert_candidate_scores_close(production, candidate)
    with pytest.raises(AssertionError, match="candidate scorer"):
        assert_candidate_scores_close(production, {**candidate, "raw_gap": -2.7})


def test_positive_class_metrics_are_finite_when_no_positive_prediction() -> None:
    from scripts.diagnose_apcf_gate_calibration import binary_classification_metrics

    metrics = binary_classification_metrics([True, False, True], [False, False, False])
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert all(math.isfinite(float(value)) for value in metrics.values())


def test_candidate_scorer_matches_production_generated_proposal() -> None:
    from tests.test_apcf_anchor_proposal import (
        DATA_PATH,
        _advance_to_ready_physical_task,
        _make_agent,
    )
    from tests.runtime_safety import seed_everything, temporary_config
    from configs import configs
    from environment import AirLineEnv_Graph
    from scripts.diagnose_apcf_gate_calibration import (
        _score_candidate_via_production,
        assert_candidate_scores_close,
        validate_production_branch,
    )

    seed_everything(42)
    agent, overrides = _make_agent()
    env = AirLineEnv_Graph(DATA_PATH, seed=42)
    obs, masks = _advance_to_ready_physical_task(env)
    with temporary_config(configs, overrides):
        action, _logprob, _value, _station_mask, invalid = agent.select_action(
            obs,
            mask_task=masks[0],
            mask_station_matrix=masks[1],
            mask_worker=masks[2],
            deterministic=True,
            temperature=0.0,
        )
    assert action is not None and not invalid
    trace = agent.last_anchor_proposal_trace
    assert trace is not None and trace.proposal_available

    with torch.inference_mode():
        encoded, _context = agent.policy(obs)
        task_emb = encoded["task"][int(trace.task_id)].unsqueeze(0)
        station_emb = encoded["station"][int(trace.station_id)].unsqueeze(0)
        generated = _score_candidate_via_production(
            agent,
            task_emb=task_emb,
            station_emb=station_emb,
            worker_embs=encoded["worker"],
            anchor_team=tuple(trace.anchor_team),
            candidate_team=tuple(trace.proposal_team),
            gate_features=tuple(trace.gate_features),
        )
    production = {
        "gate_logit": float(
            torch.logit(torch.tensor(trace.gate_value).clamp(1.0e-6, 1.0 - 1.0e-6)).item()
        ),
        "predicted_delta_A": float(trace.predicted_delta_a),
        "raw_gap": float(trace.raw_branch_logit_gap),
    }
    assert_candidate_scores_close(production, generated)
    validate_production_branch(
        int(trace.raw_argmax_branch),
        float(generated["raw_gap"]),
        proposal_available=True,
    )
