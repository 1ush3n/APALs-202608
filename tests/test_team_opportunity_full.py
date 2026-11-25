from __future__ import annotations

import math

from scripts.audit_initial_team_opportunity_full import (
    _enumerate_one_swaps,
    _final_makespan,
    _spearman_rho,
    summarize_full_episode_audit,
)


class _ClockEnv:
    def __init__(self, clock: list[float]) -> None:
        self.station_wall_clock = clock


def test_final_makespan_uses_station_wall_clock_max() -> None:
    assert _final_makespan(_ClockEnv([10.0, 20.0, 5.0])) == 20.0
    assert math.isnan(_final_makespan(_ClockEnv([])))


def test_spearman_perfect_and_ties() -> None:
    assert _spearman_rho([1.0, 2.0, 3.0], [3.0, 5.0, 9.0]) == 1.0
    assert abs(_spearman_rho([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) + 1.0) < 1.0e-9
    # 并列：单调但非严格 → 秩相关不为 None 且介于 [-1, 1]
    rho = _spearman_rho([1.0, 1.0, 2.0, 3.0], [2.0, 2.0, 4.0, 6.0])
    assert rho is not None and -1.0 <= rho <= 1.0
    assert _spearman_rho([1.0, 2.0], [3.0, 4.0]) is None  # 样本不足


def test_enumerate_one_swaps_covers_all_positions_and_excludes_base() -> None:
    base = (1, 2, 3)
    legal = [4, 5]
    teams = _enumerate_one_swaps(base, legal)
    assert len(teams) == 2 * 3  # 2 名合法工人 × 3 个位置
    assert tuple(sorted(base)) not in teams
    for team in teams:
        assert len(team) == 3
        # 与基准只差一名工人：恰好一个换入、一个换出
        assert len(set(team) - set(base)) == 1
        assert len(set(base) - set(team)) == 1


def test_enumerate_one_swaps_dedups_and_sorts() -> None:
    teams = _enumerate_one_swaps((2, 1), [9, 9, 7])
    assert teams == sorted(teams)
    assert len(teams) == 2 * 2  # 去重后仍为 4 个不同团队
    assert all(t == tuple(sorted(t)) for t in teams)


def test_full_episode_summary_aggregates_generator_coverage() -> None:
    rows = [
        {
            "baseline_done": True,
            "any_team_better_full_episode": True,
            "best_full_gain_pct": 1.2,
            "best_team_in_generator_top4": True,
            "two_swap_beats_best_single": True,
            "heuristic_alignment_spearman": 0.9,
            "capped_episodes": 0,
            "baseline_full_makespan": 100.0,
        },
        {
            "baseline_done": True,
            "any_team_better_full_episode": True,
            "best_full_gain_pct": 0.8,
            "best_team_in_generator_top4": False,
            "two_swap_beats_best_single": None,
            "heuristic_alignment_spearman": None,
            "capped_episodes": 0,
            "baseline_full_makespan": 200.0,
        },
        {
            "baseline_done": True,
            "any_team_better_full_episode": False,
            "best_full_gain_pct": -0.5,
            "best_team_in_generator_top4": False,
            "two_swap_beats_best_single": False,
            "heuristic_alignment_spearman": -0.2,
            "capped_episodes": 1,
            "baseline_full_makespan": 150.0,
        },
        {
            "baseline_done": False,
            "any_team_better_full_episode": False,
            "best_full_gain_pct": 0.0,
            "best_team_in_generator_top4": False,
            "two_swap_beats_best_single": None,
            "heuristic_alignment_spearman": None,
            "capped_episodes": 2,
            "baseline_full_makespan": None,
        },
    ]
    summary = summarize_full_episode_audit(rows)

    assert summary["audited_states"] == 4
    assert summary["valid_states"] == 3
    assert summary["states_with_better_team_full_episode"] == 2
    assert math.isclose(summary["better_team_rate"], 2 / 3)
    assert math.isclose(summary["best_gain_pct_mean_when_improved"], 1.0)
    assert math.isclose(summary["best_gain_pct_max_when_improved"], 1.2)
    assert math.isclose(summary["generator_coverage_of_better_teams"], 0.5)
    assert summary["states_two_swap_beats_best_single"] == 1
    assert summary["states_two_swap_evaluated"] == 2
    assert math.isclose(summary["heuristic_alignment_spearman_mean"], 0.35)
    assert summary["heuristic_alignment_states"] == 2
    assert math.isclose(summary["baseline_makespan_mean"], 150.0)
    assert summary["capped_episode_states"] == 2


def test_full_episode_summary_empty() -> None:
    assert summarize_full_episode_audit([]) == {"audited_states": 0}
