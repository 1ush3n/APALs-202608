from __future__ import annotations

from pathlib import Path

import pytest
import torch

from configs import Config
from models.hb_gat_pn import HBGATPN
from runtime.checkpoints import (
    apply_checkpoint_model_spec,
    build_checkpoint_metadata,
    build_resume_batch_audit,
    infer_model_spec,
    load_checkpoint,
    validate_checkpoint_training_spec,
)


def _state_for(*, use_skill_hub: bool, bidirectional: bool) -> tuple[Config, dict]:
    cfg = Config()
    cfg.use_skill_hub = use_skill_hub
    cfg.skill_hub_bidirectional = bidirectional if use_skill_hub else False
    return cfg, HBGATPN(cfg).state_dict()


@pytest.mark.parametrize(
    ("use_skill_hub", "bidirectional", "expected"),
    [
        (False, False, "legacy_direct"),
        (True, False, "skill_hub_forward"),
        (True, True, "skill_hub_bidirectional"),
    ],
)
def test_infers_resource_graph_mode(use_skill_hub, bidirectional, expected) -> None:
    _cfg, state = _state_for(
        use_skill_hub=use_skill_hub,
        bidirectional=bidirectional,
    )
    assert infer_model_spec(state).resource_graph_mode == expected


def test_loads_lightning_and_legacy_checkpoint_formats(tmp_path: Path) -> None:
    cfg, state = _state_for(use_skill_hub=False, bidirectional=False)
    metadata = build_checkpoint_metadata(cfg)
    lightning_path = tmp_path / "model.ckpt"
    torch.save({
        "state_dict": {f"policy.{key}": value for key, value in state.items()},
        "apal_metadata": metadata,
    }, lightning_path)
    loaded = load_checkpoint(lightning_path)
    assert loaded.format_name == "lightning"
    assert loaded.model_spec.resource_graph_mode == "legacy_direct"
    assert set(loaded.state_dict) == set(state)

    legacy_path = tmp_path / "model.pth"
    torch.save({"model_state_dict": state, "apal_metadata": metadata}, legacy_path)
    assert load_checkpoint(legacy_path).format_name == "legacy_full"


def test_explicit_structural_conflict_is_rejected() -> None:
    cfg = Config()
    cfg.use_skill_hub = True
    _legacy_cfg, state = _state_for(use_skill_hub=False, bidirectional=False)
    spec = infer_model_spec(state)

    with pytest.raises(ValueError, match="checkpoint 冲突"):
        apply_checkpoint_model_spec(cfg, spec, explicit_fields={"use_skill_hub"})

    apply_checkpoint_model_spec(cfg, spec, explicit_fields=set())
    assert cfg.use_skill_hub is False


def _v2_training_config() -> Config:
    cfg = Config()
    cfg.team_selection_mode = "autoregressive_pressure_v2"
    cfg.policy_action_scope = "operation_station_worker"
    cfg.actor_context_mode = "attention"
    cfg.worker_pointer_v2_behavior_replay = True
    cfg.worker_pointer_v2_replay_mode = "behavior_group_exact_v1"
    cfg.batch_size = 64
    cfg.worker_pointer_v2_logical_batch_cap = 64
    cfg.worker_pointer_v2_rollout_group_upper_bound = 4
    cfg.accumulation_steps = 16
    return cfg


def test_v2_checkpoint_records_group_replay_training_semantics() -> None:
    cfg = _v2_training_config()
    metadata = build_checkpoint_metadata(cfg)

    assert metadata["training_spec"] == {
        "worker_pointer_v2_replay_mode": "behavior_group_exact_v1",
        "worker_pointer_v2_logical_batch_cap": 64,
        "worker_pointer_v2_rollout_group_upper_bound": 4,
        "worker_pointer_v2_per_sample_heads": True,
        "num_envs": int(cfg.num_envs),
        "accumulation_steps": 16,
    }


def test_v2_resume_rejects_missing_or_conflicting_training_spec() -> None:
    cfg = _v2_training_config()
    with pytest.raises(ValueError, match="training_spec"):
        validate_checkpoint_training_spec(cfg, {})

    metadata = build_checkpoint_metadata(cfg)
    migrated_batch = dict(metadata)
    migrated_batch["training_spec"] = {
        **metadata["training_spec"],
        "worker_pointer_v2_logical_batch_cap": 256,
    }
    validate_checkpoint_training_spec(cfg, migrated_batch)

    conflicting_group = dict(metadata)
    conflicting_group["training_spec"] = {
        **metadata["training_spec"],
        "worker_pointer_v2_rollout_group_upper_bound": 8,
    }
    with pytest.raises(ValueError, match="rollout_group_upper_bound"):
        validate_checkpoint_training_spec(cfg, conflicting_group)


def test_resume_batch_audit_records_checkpoint_override() -> None:
    audit = build_resume_batch_audit(
        {"apal_agent_state": {"batch_size": 64}},
        current_batch_size=256,
    )

    assert audit == {
        "checkpoint_batch_size": 64,
        "current_batch_size": 256,
        "override_applied": True,
    }


def test_legacy_resume_does_not_require_v2_training_spec() -> None:
    validate_checkpoint_training_spec(Config(), {})
