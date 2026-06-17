from __future__ import annotations

from runtime.multiscale import (
    BenchmarkScore,
    apply_scale_profile_to_agent,
    inverse_scale_weight,
    scale_profile_for_task_count,
    scheduled_updates_for_task_count,
    score_multi_benchmark,
)


def test_scale_profile_mapping() -> None:
    assert scale_profile_for_task_count(200).name == "283"
    assert scale_profile_for_task_count(283).name == "283"
    assert scale_profile_for_task_count(284).name == "680"
    assert scale_profile_for_task_count(1000).name == "680"
    assert scale_profile_for_task_count(1001).name == "2338"
    assert scale_profile_for_task_count(2339).name == "3182"
    assert scale_profile_for_task_count(200).batch_size == 512
    assert scale_profile_for_task_count(1000).batch_size == 256
    assert scale_profile_for_task_count(2338).batch_size == 128
    assert scale_profile_for_task_count(3182).batch_size == 64


def test_scheduled_updates_are_sublinear_and_monotonic() -> None:
    lo = scheduled_updates_for_task_count(200)
    mid = scheduled_updates_for_task_count(1000)
    hi = scheduled_updates_for_task_count(3100)

    assert lo == 600
    assert hi == 3300
    assert lo < mid < hi
    assert scheduled_updates_for_task_count(500) < scheduled_updates_for_task_count(2000)


def test_inverse_scale_weight_decreases_with_size() -> None:
    w_small = inverse_scale_weight(200)
    w_large = inverse_scale_weight(3100)

    assert w_small > w_large
    assert w_small > 0.0
    assert w_large > 0.0


def test_composite_score_and_invalid_eligibility() -> None:
    rows = [
        BenchmarkScore("283", "data/283.csv", 130.0, 100.0, 1.3, True, 0, 1.0),
        BenchmarkScore("680", "data/680.csv", 220.0, 200.0, 1.1, True, 0, 2.0),
    ]

    score = score_multi_benchmark(rows)

    assert score.eligible is True
    assert abs(score.composite_score - 1.2) < 1e-9
    assert score.selection_score == score.composite_score

    invalid_score = score_multi_benchmark(
        [
            BenchmarkScore("283", "data/283.csv", 300.0, 100.0, 3.0, False, 1, 1.0),
            BenchmarkScore("680", "data/680.csv", 220.0, 200.0, 1.1, True, 0, 2.0),
        ]
    )
    assert invalid_score.eligible is False
    assert invalid_score.selection_score == float("inf")


def test_apply_scale_profile_to_agent_sets_ppo_epochs_and_batch() -> None:
    class DummyAgent:
        k_epochs = 0
        batch_size = 0

    class DummyConfig:
        ppo_batch_size_cap = 4

    agent = DummyAgent()
    profile = scale_profile_for_task_count(3182)

    apply_scale_profile_to_agent(agent, profile, DummyConfig())

    assert agent.k_epochs == 4
    assert agent.batch_size == 4
