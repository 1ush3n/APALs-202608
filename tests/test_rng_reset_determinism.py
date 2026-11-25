from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np
import pytest

from environment import AirLineEnv_Graph
from utils.vector_env import EnvCreator, VectorEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "283.csv"
RESET_SEED = 20260823
CROSS_PLATFORM_START_METHODS = tuple(
    method
    for method in ("spawn", "forkserver")
    if method in mp.get_all_start_methods()
)


def _task_durations(state: object) -> np.ndarray:
    """提取随机化工时，避免把图中其他动态状态混入确定性断言。"""
    return state["task"].x[:, 0].detach().cpu().numpy().copy()


def _local_reset_sequence() -> tuple[np.ndarray, np.ndarray]:
    env = AirLineEnv_Graph(DATA_PATH, seed=17)
    try:
        first = env.reset(
            randomize_duration=True,
            randomize_workers=False,
            seed=RESET_SEED,
        )
        second = env.reset(
            randomize_duration=True,
            randomize_workers=False,
            seed=None,
        )
        return _task_durations(first), _task_durations(second)
    finally:
        env.close()


def _subprocess_reset_sequence(
    start_method: str,
) -> tuple[np.ndarray, np.ndarray]:
    vector_env = VectorEnv(
        EnvCreator(str(DATA_PATH), seed_offset=900),
        num_envs=1,
        start_method=start_method,
        worker_threads=1,
        init_timeout_sec=30.0,
        command_timeout_sec=30.0,
    )
    try:
        first = vector_env.reset_indices(
            {
                0: {
                    "randomize_duration": True,
                    "randomize_workers": False,
                    "seed": RESET_SEED,
                }
            }
        )[0]
        second = vector_env.reset_indices(
            {
                0: {
                    "randomize_duration": True,
                    "randomize_workers": False,
                    "seed": None,
                }
            }
        )[0]
        return _task_durations(first), _task_durations(second)
    finally:
        vector_env.close()


def test_reset_without_seed_preserves_rng_continuity() -> None:
    first_run = _local_reset_sequence()
    second_run = _local_reset_sequence()

    np.testing.assert_array_equal(first_run[0], second_run[0])
    np.testing.assert_array_equal(first_run[1], second_run[1])
    assert not np.array_equal(first_run[0], first_run[1])


@pytest.mark.parametrize("start_method", CROSS_PLATFORM_START_METHODS)
def test_reset_sequence_is_deterministic_across_processes(
    start_method: str,
) -> None:
    first_process = _subprocess_reset_sequence(start_method)
    second_process = _subprocess_reset_sequence(start_method)

    np.testing.assert_array_equal(first_process[0], second_process[0])
    np.testing.assert_array_equal(first_process[1], second_process[1])
    assert not np.array_equal(first_process[0], first_process[1])
