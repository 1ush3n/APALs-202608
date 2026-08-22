from __future__ import annotations

from types import SimpleNamespace

import pytest

from baselines.literature.common import (
    is_better_r5_group,
    r5_group_selection_score,
    validate_r5_learning_protocol,
)


def _valid_config() -> SimpleNamespace:
    return SimpleNamespace(
        reschedule_async_protocol="r5_task_delay_v1",
        reschedule_manifest_path="data/r5_task_delay_v1/manifest.json",
        reschedule_eval_instance_id="validation_0001",
        async_eval_enabled=True,
        async_eval_device="cuda",
        async_eval_worker_count=3,
        async_eval_submit_every_episodes=2,
        async_eval_allow_cpu_fallback=False,
        async_eval_scenario_ids=["low_early", "medium_early", "high_early"],
    )


def test_r5_learning_protocol_requires_three_cuda_scenarios_every_two_episodes() -> None:
    validate_r5_learning_protocol(_valid_config())

    invalid = _valid_config()
    invalid.async_eval_worker_count = 1
    with pytest.raises(ValueError, match="3 个 CUDA worker"):
        validate_r5_learning_protocol(invalid)


def test_r5_group_selection_score_requires_all_scenarios_eligible() -> None:
    eligible = [
        {"eligible": 1.0, "selection_score": 4.0},
        {"eligible": 1.0, "selection_score": 6.0},
        {"eligible": 1.0, "selection_score": 8.0},
    ]
    assert r5_group_selection_score(eligible) == pytest.approx(6.0)

    ineligible = [*eligible[:2], {"eligible": 0.0, "selection_score": 1.0}]
    assert r5_group_selection_score(ineligible) == float("inf")


def test_r5_group_best_selection_uses_score_then_episode_tie_break() -> None:
    assert is_better_r5_group(4.0, 10, {"selection_score": 5.0, "episode": 8})
    assert is_better_r5_group(5.0, 10, {"selection_score": 5.0, "episode": 12})
    assert not is_better_r5_group(5.0, 10, {"selection_score": 5.0, "episode": 8})
