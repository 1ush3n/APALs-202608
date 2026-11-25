from __future__ import annotations

from configs import Config
from runtime.batch_semantics import (
    resolve_effective_ppo_batch_size,
    resolve_v2_logical_batch_size,
)


def test_v2_explicit_batch_bypasses_platform_cap_and_logical_cap() -> None:
    config = Config()
    config.team_selection_mode = "autoregressive_pressure_v2"
    config.worker_pointer_v2_replay_mode = "behavior_group_exact_v1"
    config.ppo_batch_size_cap = 4
    config.worker_pointer_v2_logical_batch_cap = 64

    effective = resolve_effective_ppo_batch_size(256, config)

    assert effective == 256
    assert resolve_v2_logical_batch_size(effective, config) == 256


def test_batched_v2_obeys_platform_batch_cap() -> None:
    config = Config()
    config.team_selection_mode = "autoregressive_pressure_v2"
    config.worker_pointer_v2_replay_mode = "batched_vectorized_v2"
    config.ppo_batch_size_cap = 16

    assert resolve_effective_ppo_batch_size(256, config) == 16


def test_legacy_batch_keeps_platform_safety_cap() -> None:
    config = Config()
    config.team_selection_mode = "autoregressive"
    config.ppo_batch_size_cap = 4

    assert resolve_effective_ppo_batch_size(256, config) == 4
