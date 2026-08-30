# -*- coding: utf-8 -*-
"""WorkerPointer v2 Fast-Exact 阶段一：配置隔离与模式识别合同测试。

按 TDD 阶段一先写失败测试：实现前 fast_exact 模式尚未被配置白名单、
ModelSpec、training_spec 与统一模式判断函数识别。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from configs import Config
from runtime.configuration import (
    is_fast_exact_mode,
    is_worker_pointer_v2_mode,
    resolve_fast_exact_num_envs,
    validate_runtime_config,
)
from runtime.checkpoints import (
    apply_checkpoint_model_spec,
    build_checkpoint_metadata,
    build_model_spec,
    validate_checkpoint_training_spec,
)
from runtime.hydra_config import (
    ParsedHydraArgs,
    apply_hydra_config,
    compose_hydra_config,
)


CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


FAST_EXACT_MODE = "autoregressive_pressure_v2_fast_exact"
GPU_TEMPLATE_REPLAY_MODE = "behavior_group_exact_gpu_template_v2"


def _fast_exact_config(**overrides: object) -> Config:
    """构造符合 fast_exact 合法语义的最小配置。"""
    cfg = Config()
    cfg.team_selection_mode = FAST_EXACT_MODE
    cfg.policy_action_scope = "operation_station_worker"
    cfg.actor_context_mode = "attention"
    cfg.worker_pointer_v2_replay_mode = GPU_TEMPLATE_REPLAY_MODE
    cfg.worker_pointer_v2_rollout_group_upper_bound = 16
    cfg.batch_size = 256
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _legacy_v2_config() -> Config:
    cfg = Config()
    cfg.team_selection_mode = "autoregressive_pressure_v2"
    cfg.policy_action_scope = "operation_station_worker"
    cfg.actor_context_mode = "attention"
    cfg.worker_pointer_v2_behavior_replay = True
    cfg.worker_pointer_v2_replay_mode = "behavior_group_exact_v1"
    return cfg


def test_runtime_validation_accepts_fast_exact_mode() -> None:
    validate_runtime_config(_fast_exact_config())


def test_fast_exact_mode_requires_operation_station_worker_scope() -> None:
    cfg = _fast_exact_config()
    cfg.policy_action_scope = "operation_station_anchor_proposal_team"
    with pytest.raises(ValueError, match="operation_station_worker"):
        validate_runtime_config(cfg)


def test_fast_exact_mode_requires_attention_context() -> None:
    cfg = _fast_exact_config()
    cfg.actor_context_mode = "mean_max"
    with pytest.raises(ValueError, match="actor_context_mode=attention"):
        validate_runtime_config(cfg)


def test_fast_exact_mode_requires_gpu_template_replay_mode() -> None:
    cfg = _fast_exact_config()
    cfg.worker_pointer_v2_replay_mode = "behavior_group_exact_v1"
    with pytest.raises(ValueError, match="behavior_group_exact_gpu_template_v2"):
        validate_runtime_config(cfg)


def test_model_spec_records_fast_exact_mode_and_pressure_semantics() -> None:
    cfg = _fast_exact_config()
    spec = build_model_spec(cfg)

    assert spec.team_selection_mode == FAST_EXACT_MODE
    assert spec.worker_pointer_context_version == cfg.worker_pointer_context_version
    assert spec.worker_pointer_pressure_temperature == 1.0
    assert spec.worker_pointer_supply_epsilon == 1.0e-6
    assert spec.worker_pointer_wait_discount_mode == "physical_wait_exponential_v1"


def test_fast_exact_encoder_batching_mode_is_structural() -> None:
    cfg = _fast_exact_config(worker_pointer_v2_fast_replay_batching="logical_batch_v1")

    spec = build_model_spec(cfg)
    metadata = build_checkpoint_metadata(cfg)

    assert spec.worker_pointer_v2_fast_replay_batching == "logical_batch_v1"
    assert (
        metadata["training_spec"]["worker_pointer_v2_fast_replay_batching"]
        == "logical_batch_v1"
    )


def test_fast_exact_encoder_batching_mode_rejects_unknown_value() -> None:
    cfg = _fast_exact_config(worker_pointer_v2_fast_replay_batching="unknown")

    with pytest.raises(ValueError, match="worker_pointer_v2_fast_replay_batching"):
        validate_runtime_config(cfg)


def test_checkpoint_metadata_training_spec_records_fast_exact_semantics() -> None:
    cfg = _fast_exact_config()
    metadata = build_checkpoint_metadata(cfg)

    training_spec = metadata["training_spec"]
    assert (
        training_spec["worker_pointer_v2_replay_mode"]
        == GPU_TEMPLATE_REPLAY_MODE
    )
    assert (
        training_spec["worker_pointer_v2_rollout_group_upper_bound"] == 16
    )
    assert training_spec["worker_pointer_v2_logical_batch_cap"] == 256
    assert training_spec["accumulation_steps"] == cfg.accumulation_steps
    assert training_spec["num_envs"] == cfg.num_envs


def test_validate_checkpoint_training_spec_rejects_fast_exact_mode_mismatch() -> None:
    cfg = _fast_exact_config()
    metadata = build_checkpoint_metadata(cfg)

    saved = dict(metadata["training_spec"])
    saved["worker_pointer_v2_replay_mode"] = "behavior_group_exact_v1"
    metadata = dict(metadata)
    metadata["training_spec"] = saved
    with pytest.raises(ValueError, match="training_spec"):
        validate_checkpoint_training_spec(cfg, metadata)


def test_fast_exact_resume_rejects_legacy_v2_checkpoint() -> None:
    """新模式不得恢复任何旧 v2 checkpoint（双向 fail-closed 之一）。"""
    cfg = _fast_exact_config()
    v2_spec = build_model_spec(_legacy_v2_config())
    with pytest.raises(ValueError, match="team_selection_mode"):
        apply_checkpoint_model_spec(cfg, v2_spec)


def test_legacy_v2_resume_rejects_fast_exact_checkpoint() -> None:
    """历史 v2 模式不得恢复 fast_exact checkpoint（双向 fail-closed 之二）。"""
    v2_cfg = _legacy_v2_config()
    fast_spec = build_model_spec(_fast_exact_config())
    with pytest.raises(ValueError, match="team_selection_mode"):
        apply_checkpoint_model_spec(v2_cfg, fast_spec)


def test_is_worker_pointer_v2_mode_matches_legacy_and_fast_exact() -> None:
    assert is_worker_pointer_v2_mode(_legacy_v2_config())
    assert is_worker_pointer_v2_mode(_fast_exact_config())
    assert not is_worker_pointer_v2_mode(Config())


def test_is_fast_exact_mode_isolates_only_fast_exact() -> None:
    assert is_fast_exact_mode(_fast_exact_config())
    assert not is_fast_exact_mode(_legacy_v2_config())
    assert not is_fast_exact_mode(Config())


def test_fast_exact_strict_gpu_replay_flag_defaults_off() -> None:
    cfg = Config()
    assert cfg.worker_pointer_v2_strict_gpu_replay is False
    cfg.worker_pointer_v2_strict_gpu_replay = True
    assert cfg.worker_pointer_v2_strict_gpu_replay is True


def _compose_fast_exact_experiment(
    hardware: str,
    num_envs_cli: int | None = None,
) -> tuple[Config, int]:
    """加载 fast_exact 实验配置 + 平台硬件配置，模拟解析流程返回最终 num_envs。"""
    final_overrides: dict[str, object] = {}
    explicit: set[str] = set()
    if num_envs_cli is not None:
        final_overrides["num_envs"] = num_envs_cli
        explicit.add("num_envs")
    parsed = ParsedHydraArgs(
        experiment="initial_worker_pointer_v2_fast_exact_exploratory",
        hardware=hardware,
        resume=False,
        resume_checkpoint_path=None,
        config_overrides=(),
        final_overrides=final_overrides,
        explicit_fields=explicit,
    )
    hydra_cfg = compose_hydra_config(parsed, config_dir=CONF_DIR)
    target = Config()
    apply_hydra_config(hydra_cfg, target=target, config_paths=(str(CONF_DIR),))
    resolved = resolve_fast_exact_num_envs(
        target,
        cli_explicit_num_envs="num_envs" in parsed.explicit_fields,
    )
    if num_envs_cli is not None:
        resolved = num_envs_cli
    return target, resolved


def test_fast_exact_experiment_linux_resolves_16_envs() -> None:
    target, resolved = _compose_fast_exact_experiment("linux_server")
    assert is_fast_exact_mode(target)
    assert target.worker_pointer_v2_fast_default_num_envs == 16
    assert resolved == 16
    assert target.async_eval_enabled is True
    assert target.async_eval_device == "cuda"
    assert target.async_eval_worker_count == 2
    assert target.async_eval_queue_capacity == 4
    assert target.async_eval_submit_every_episodes == 2
    assert target.async_eval_wait_on_finish is True


def test_fast_exact_experiment_windows_resolves_4_envs() -> None:
    target, resolved = _compose_fast_exact_experiment(
        "windows_4060_low_memory"
    )
    assert target.worker_pointer_v2_fast_default_num_envs == 4
    assert resolved == 4


def test_fast_exact_experiment_cli_num_envs_wins_over_platform() -> None:
    target, resolved = _compose_fast_exact_experiment(
        "linux_server", num_envs_cli=8
    )
    assert target.worker_pointer_v2_fast_default_num_envs == 16
    assert resolved == 8


def test_fast_exact_pilot_keeps_async_validation_and_effective_batch_contract() -> None:
    parsed = ParsedHydraArgs(
        experiment="initial_worker_pointer_v2_fast_exact_pilot_v0",
        hardware="linux_server",
        resume=False,
        resume_checkpoint_path=None,
        config_overrides=(),
        final_overrides={"num_envs": 4},
        explicit_fields={"num_envs"},
    )
    hydra_cfg = compose_hydra_config(parsed, config_dir=CONF_DIR)
    target = Config()
    apply_hydra_config(hydra_cfg, target=target, config_paths=(str(CONF_DIR),))

    assert target.batch_size == 256
    assert target.accumulation_steps == 16
    assert target.async_eval_enabled is True
    assert target.async_eval_worker_count == 2
    assert target.async_eval_submit_every_episodes == 2
