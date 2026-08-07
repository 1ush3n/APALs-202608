from __future__ import annotations

from pathlib import Path

from configs import Config, load_config_files
from scripts.audit_initial_team_opportunity import (
    initial_objective_from_info,
    summarize_initial_opportunities,
)


def test_initial_objective_uses_makespan_and_balance_penalties() -> None:
    outcome = initial_objective_from_info(
        -0.4,
        False,
        {"makespan_penalty": 0.30, "std_penalty": 0.10},
    )

    assert outcome == {
        "objective": 0.4,
        "makespan_penalty": 0.3,
        "balance_penalty": 0.1,
        "reward": -0.4,
        "done": 0.0,
    }


def test_initial_opportunity_summary_segments_skill_bottlenecks() -> None:
    summary = summarize_initial_opportunities(
        [
            {
                "candidate_count": 4,
                "skill_bottleneck": True,
                "best_single_objective_gain": 0.3,
                "has_single_swap_improvement": True,
                "best_single_gain_makespan": 0.2,
                "best_single_gain_balance": 0.1,
            },
            {
                "candidate_count": 4,
                "skill_bottleneck": False,
                "best_single_objective_gain": -0.1,
                "has_single_swap_improvement": False,
                "best_single_gain_makespan": -0.1,
                "best_single_gain_balance": 0.0,
            },
        ]
    )

    assert summary["audited_states"] == 2
    assert summary["states_with_better_single_swap"] == 1
    assert summary["better_single_swap_rate"] == 0.5
    assert summary["improvement_rate_skill_bottleneck"] == 1.0
    assert summary["improvement_rate_non_bottleneck"] == 0.0
    assert summary["mean_component_gain_when_improved"] == {
        "makespan": 0.2,
        "balance": 0.1,
    }


def test_margin2_experiment_config_only_relaxes_the_ctg_prior() -> None:
    root = Path(__file__).resolve().parents[1]
    config = Config()

    load_config_files(
        [str(root / "conf" / "experiment" / "initial_conditional_team_gate_prior_margin2.yaml")],
        target=config,
    )

    assert config.policy_action_scope == "operation_station_gated_team"
    assert config.conditional_team_scoring_mode == "relative_heuristic_prior_v1"
    assert config.conditional_team_prior_margin == 2.0
    assert config.conditional_team_gate_bias == -4.0
    assert config.conditional_team_prior_weight == 1.0
