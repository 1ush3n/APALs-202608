from __future__ import annotations

from typing import Any

from runtime.team_opportunity import evaluate_one_step_candidate


class _FakeEnv:
    def __init__(self) -> None:
        self.skip_obs_building = False
        self.actions: list[tuple[int, int, list[int]]] = []

    def step(self, action: tuple[int, int, list[int]]):
        self.actions.append(action)
        return None, -0.25, False, {
            "makespan_penalty": 0.20,
            "std_penalty": 0.05,
        }


def test_one_step_candidate_uses_an_isolated_environment_clone() -> None:
    env = _FakeEnv()

    outcome = evaluate_one_step_candidate(
        env,
        action=(3, 1, (4, 5)),
        metric_extractor=lambda reward, done, info: {
            "objective": float(info["makespan_penalty"] + info["std_penalty"]),
            "reward": float(reward),
            "done": float(done),
        },
    )

    assert env.actions == []
    assert env.skip_obs_building is False
    assert outcome == {"objective": 0.25, "reward": -0.25, "done": 0.0}
