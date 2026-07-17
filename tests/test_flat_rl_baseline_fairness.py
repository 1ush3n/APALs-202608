from __future__ import annotations

import numpy as np
import pytest
import torch

from baselines.evaluate_flat_rl_baseline import evaluate_model


class _AlwaysDeadlockedEnv:
    def __init__(self) -> None:
        self.num_tasks = 2
        self.assigned_tasks: list[tuple[int, int, list[int], float, float]] = []
        self.ideal_makespan = 10.0
        self.ideal_station_load = 5.0
        self.station_wall_clock = np.zeros(1, dtype=float)
        self.station_loads = np.zeros(1, dtype=float)
        self.reset_seeds: list[int] = []

    def reset(self, randomize_duration: bool = False, randomize_workers: bool = False, seed: int | None = None):
        assert randomize_duration is False
        assert randomize_workers is False
        assert seed is not None
        self.assigned_tasks = []
        self.reset_seeds.append(int(seed))
        return None

    def get_masks(self):
        task_mask = torch.ones(self.num_tasks, dtype=torch.bool)
        station_mask = torch.ones((self.num_tasks, 1), dtype=torch.bool)
        worker_mask = torch.ones(1, dtype=torch.bool)
        return task_mask, station_mask, worker_mask

    def try_wait_for_resources(self) -> bool:
        return False


@pytest.mark.parametrize("algorithm", ["l2d_ppo_apal", "graph_ddqn_apal"])
def test_graph_baseline_eval_records_fixed_failed_runs_instead_of_skipping(algorithm: str) -> None:
    env = _AlwaysDeadlockedEnv()
    metrics, schedule, runs = evaluate_model(
        model=torch.nn.Identity(),
        algorithm=algorithm,
        env=env,
        device=torch.device("cpu"),
        seed=100,
        num_runs=3,
        temperature=0.0,
    )

    assert env.reset_seeds == [100, 101, 102]
    assert schedule == []
    assert len(runs) == 3
    assert [run["seed"] for run in runs] == [100, 101, 102]
    assert [run["valid"] for run in runs] == [0.0, 0.0, 0.0]
    assert [run["completion_rate"] for run in runs] == [0.0, 0.0, 0.0]
    assert metrics["valid"] == 0.0
    assert metrics["completion_rate"] == 0.0
