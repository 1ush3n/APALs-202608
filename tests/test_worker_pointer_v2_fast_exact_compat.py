# -*- coding: utf-8 -*-
"""WorkerPointer v2 Fast-Exact 阶段四：兼容性回归。

- legacy autoregressive / static top-q / CTG / APCF 不得创建 Fast-Exact builder；
- 历史 v2 与 Fast-Exact 共用同一 v2 语义族参数键；
- Fast-Exact 不读取历史 replay group 字段（traces 缺失即 fail-closed）；
- 新模式初始化不改变 legacy 参数键与固定权重 logits。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from configs import Config, configs as global_configs
from models.hb_gat_pn import WorkerPointer
from runtime.checkpoints import (
    apply_checkpoint_model_spec,
    build_checkpoint_metadata,
    load_checkpoint,
    load_policy_weights,
    validate_checkpoint_training_spec,
)
from tests.runtime_safety import temporary_config
from training.rollout_service import APALRolloutService


def _fake_vector_env() -> object:
    class _FakeEnv:
        pass

    class _FakeVectorEnv:
        num_envs = 4
        envs = [_FakeEnv()]

    return _FakeVectorEnv()


def _make_service(cfg: Config) -> APALRolloutService:
    return APALRolloutService(
        agent=object(),
        vector_env=_fake_vector_env(),
        eval_env=object(),
        config=cfg,
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize(
    ("mode", "scope", "context"),
    [
        ("autoregressive", "operation_station_worker", "attention"),
        ("static_topq", "operation_station_worker", "attention"),
        ("autoregressive", "operation_station_gated_team", "attention"),
        ("autoregressive", "operation_station_anchor_proposal_team", "attention"),
    ],
)
def test_legacy_and_ctg_apcf_modes_do_not_create_builder(
    mode: str,
    scope: str,
    context: str,
) -> None:
    cfg = Config()
    cfg.team_selection_mode = mode
    cfg.policy_action_scope = scope
    cfg.actor_context_mode = context
    service = _make_service(cfg)
    assert service._fast_exact_builder is None


def test_fast_exact_mode_creates_builder() -> None:
    cfg = Config()
    cfg.team_selection_mode = "autoregressive_pressure_v2_fast_exact"
    cfg.policy_action_scope = "operation_station_worker"
    cfg.actor_context_mode = "attention"
    cfg.worker_pointer_v2_replay_mode = "behavior_group_exact_gpu_template_v2"
    service = _make_service(cfg)
    assert service._fast_exact_builder is not None


def test_fast_exact_parameter_keys_match_legacy_v2() -> None:
    """Fast-Exact 与历史 v2 共享同一 v2 head 结构，参数键一致。"""
    legacy = Config()
    legacy.team_selection_mode = "autoregressive_pressure_v2"
    legacy.policy_action_scope = "operation_station_worker"
    legacy.actor_context_mode = "attention"
    fast = Config()
    fast.team_selection_mode = "autoregressive_pressure_v2_fast_exact"
    fast.policy_action_scope = "operation_station_worker"
    fast.actor_context_mode = "attention"
    fast.worker_pointer_v2_replay_mode = "behavior_group_exact_gpu_template_v2"

    legacy_keys = set(WorkerPointer(legacy).state_dict())
    fast_keys = set(WorkerPointer(fast).state_dict())
    assert fast_keys == legacy_keys


def test_fast_exact_initialization_does_not_change_legacy_keys() -> None:
    torch.manual_seed(123)
    legacy = WorkerPointer(Config())
    legacy_rng = torch.get_rng_state().clone()
    legacy_keys = set(legacy.state_dict())

    torch.manual_seed(123)
    fast_cfg = Config()
    fast_cfg.team_selection_mode = "autoregressive_pressure_v2_fast_exact"
    fast_cfg.policy_action_scope = "operation_station_worker"
    fast_cfg.actor_context_mode = "attention"
    fast_cfg.worker_pointer_v2_replay_mode = "behavior_group_exact_gpu_template_v2"
    fast = WorkerPointer(fast_cfg)
    fast_rng = torch.get_rng_state().clone()

    # 初始化后全局 RNG 必须与仅构建 legacy 模型时一致；
    # fast_exact 参数键为 legacy 参数键的超集（新增 v2 head，不破坏既有键）。
    assert torch.equal(legacy_rng, fast_rng)
    assert legacy_keys.issubset(set(fast.state_dict()))
    assert any(key.startswith("v2_") for key in fast.state_dict())


def test_fast_exact_replay_rejects_memory_without_behavior_traces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """新模式不读取历史 replay group 字段：traces 缺失直接 fail-closed。"""
    from configs import configs as _global_configs
    from environment import AirLineEnv_Graph
    from models.hb_gat_pn import HBGATPN
    from ppo_agent import PPOAgent
    from tests.test_joint_experiment_architecture import (
        DATA_PATH,
        _small_overrides,
    )
    from training.memory import Memory

    overrides = _small_overrides(
        team_selection_mode="autoregressive_pressure_v2_fast_exact",
        policy_action_scope="operation_station_worker",
        actor_context_mode="attention",
        worker_pointer_v2_replay_mode="behavior_group_exact_gpu_template_v2",
        worker_pointer_v2_rollout_group_upper_bound=16,
    )
    with temporary_config(_global_configs, overrides):
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
        env.reset(seed=42)
        agent = PPOAgent(
            HBGATPN(_global_configs),
            lr=1.0e-4,
            gamma=0.99,
            k_epochs=1,
            eps_clip=0.2,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            batch_size=4,
            total_timesteps=1,
            config=_global_configs,
        )
        memory = Memory()
        memory.states.append(env.get_state_snapshot())
        memory.actions.append((0, 0, [0]))
        memory.logprobs.append(0.0)
        memory.values.append(0.0)
        memory.rewards = [0.0]
        memory.is_terminals = [False]
        memory.masks.append(env.get_masks())
        # 不设置 worker_pointer_v2_behavior_traces（模拟历史 replay 结构缺失）。
        with pytest.raises(RuntimeError, match="traces|轨迹"):
            agent._run_v2_fast_exact_replay_update(
                memory,
                env,
                current_ep=1,
                advantages=torch.tensor([1.0]),
                rewards=torch.tensor([0.0]),
                old_logprobs=torch.tensor([0.0]),
                b_task=torch.tensor([0]),
                b_station=torch.tensor([0]),
                b_team=torch.tensor([[0]]),
                action_scope="operation_station_worker",
            )


def _fast_exact_config() -> Config:
    cfg = Config()
    cfg.team_selection_mode = "autoregressive_pressure_v2_fast_exact"
    cfg.policy_action_scope = "operation_station_worker"
    cfg.actor_context_mode = "attention"
    cfg.worker_pointer_v2_replay_mode = "behavior_group_exact_gpu_template_v2"
    return cfg


def test_fast_exact_checkpoint_roundtrip_restore(tmp_path: Path) -> None:
    """Fast-Exact checkpoint 保存后可按同一语义完整恢复（严格权重逐键一致）。"""
    torch.manual_seed(7)
    cfg = _fast_exact_config()
    model = WorkerPointer(cfg)
    metadata = build_checkpoint_metadata(cfg)
    # 保存与恢复同配置时 training_spec 必须无冲突。
    validate_checkpoint_training_spec(cfg, metadata)

    path = tmp_path / "fast_exact.ckpt"
    torch.save(
        {
            "apal_metadata": metadata,
            "state_dict": model.state_dict(),
        },
        path,
    )

    loaded = load_checkpoint(path, map_location="cpu")
    restored_cfg = Config()
    restored_cfg.update_from_dict(loaded.metadata["config"])
    validate_checkpoint_training_spec(restored_cfg, loaded.metadata)
    apply_checkpoint_model_spec(restored_cfg, loaded.model_spec)
    restored = WorkerPointer(restored_cfg)
    load_policy_weights(restored, loaded, strict=True)

    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key]), key


def test_fast_exact_resume_rejects_conflicting_v2_semantics(tmp_path: Path) -> None:
    """Fast-Exact 恢复时 v2 语义字段冲突（上下文版本不同）必须 fail-closed。"""
    torch.manual_seed(7)
    cfg = _fast_exact_config()
    metadata = build_checkpoint_metadata(cfg)
    path = tmp_path / "fast_exact_semantics.ckpt"
    torch.save(
        {
            "apal_metadata": metadata,
            "state_dict": WorkerPointer(cfg).state_dict(),
        },
        path,
    )
    loaded = load_checkpoint(path, map_location="cpu")
    conflicting = Config()
    conflicting.update_from_dict(loaded.metadata["config"])
    conflicting.worker_pointer_context_version = "another_context_v1"
    with pytest.raises(ValueError, match="语义不兼容|WorkerPointer v2 checkpoint"):
        apply_checkpoint_model_spec(conflicting, loaded.model_spec)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("worker_pointer_v2_logical_batch_cap", 128),
        ("num_envs", 8),
    ),
)
def test_fast_exact_resume_rejects_changed_batch_shape_semantics(
    field: str,
    replacement: int,
) -> None:
    """Fast-Exact 不允许以不同 logical batch 或行为组形状续训。"""
    cfg = _fast_exact_config()
    metadata = build_checkpoint_metadata(cfg)
    if field == "worker_pointer_v2_logical_batch_cap":
        cfg.batch_size = replacement
    else:
        cfg.num_envs = replacement

    with pytest.raises(ValueError, match=field):
        validate_checkpoint_training_spec(cfg, metadata)


@pytest.mark.parametrize(
    "field",
    ("worker_pointer_v2_logical_batch_cap", "num_envs"),
)
def test_fast_exact_resume_rejects_missing_batch_shape_semantics(
    field: str,
) -> None:
    """Fast-Exact checkpoint 缺少数值形状字段时必须 fail-closed。"""
    cfg = _fast_exact_config()
    metadata = build_checkpoint_metadata(cfg)
    training_spec = dict(metadata["training_spec"])
    training_spec.pop(field)
    metadata = {**metadata, "training_spec": training_spec}

    with pytest.raises(ValueError, match=field):
        validate_checkpoint_training_spec(cfg, metadata)
