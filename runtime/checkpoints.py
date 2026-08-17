from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from configs import Config
from runtime.modes import is_fast_exact_mode, is_worker_pointer_v2_mode
from worker_feature_layout import resolve_worker_feature_layout


FORMAT_VERSION = 2
WORKER_FEATURE_LAYOUT_VERSION = "five_skill_v2"


def _sha256_of(path: Path) -> str:
    """计算文件 SHA-256（APCF 反事实 manifest 可追溯校验）。"""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelSpec:
    resource_graph_mode: str
    policy_action_scope: str = "operation_station_worker"
    conditional_team_max_candidates: int = 4
    conditional_team_gate_bias: float = -4.0
    conditional_team_nonbaseline_logit: float = -8.0
    conditional_team_scoring_mode: str = "fixed_prior_v1"
    conditional_team_prior_margin: float = 4.0
    conditional_team_prior_weight: float = 1.0
    workforce_binding_mode: str = "endogenous"
    workforce_preallocation_ratio: float = 1.0
    team_selection_mode: str = "autoregressive"
    worker_pointer_context_version: str | None = None
    worker_pointer_pressure_temperature: float | None = None
    worker_pointer_supply_epsilon: float | None = None
    worker_pointer_wait_discount_mode: str | None = None
    worker_pointer_v2_dynamic_eft_features: bool = False
    worker_pointer_v2_dynamic_eft_feature_clip: float = 10.0
    graph_encoder_mode: str = "hetero_gat"
    actor_context_mode: str = "attention"
    hidden_dim: int | None = None
    num_gat_layers: int | None = None
    task_feat_dim: int | None = None
    worker_feat_dim: int | None = None
    station_feat_dim: int | None = None
    num_skill_types: int | None = None
    worker_skill_feature_slots: int | None = None
    worker_feature_layout_version: str | None = None
    # APCF（锚点条件完整团队提议与反事实门控）语义。
    anchor_proposal_mode: str | None = None
    anchor_proposal_prior_margin: float | None = None
    anchor_proposal_gate_bias: float | None = None
    anchor_proposal_train_branch_floor_start: float | None = None
    anchor_proposal_train_branch_floor_end: float | None = None
    anchor_proposal_branch_floor_decay_fraction: float | None = None
    anchor_proposal_require_difference: bool | None = None
    anchor_proposal_cf_manifest_sha256: str | None = None

    @property
    def use_skill_hub(self) -> bool:
        return self.resource_graph_mode != "legacy_direct"

    @property
    def skill_hub_bidirectional(self) -> bool:
        return self.resource_graph_mode == "skill_hub_bidirectional"


@dataclass(frozen=True)
class LoadedCheckpoint:
    payload: Any
    state_dict: dict[str, torch.Tensor]
    model_spec: ModelSpec
    metadata: dict[str, Any]
    format_name: str


def build_model_spec(config: Config) -> ModelSpec:
    mode = (
        "legacy_direct" if not config.use_skill_hub
        else "skill_hub_bidirectional" if config.skill_hub_bidirectional
        else "skill_hub_forward"
    )
    worker_layout = resolve_worker_feature_layout(config)
    apcf_fields: dict[str, Any] = {}
    if str(config.policy_action_scope) == "operation_station_anchor_proposal_team":
        apcf_fields = {
            "anchor_proposal_mode": str(config.anchor_proposal_mode),
            "anchor_proposal_prior_margin": float(config.anchor_proposal_prior_margin),
            "anchor_proposal_gate_bias": float(config.anchor_proposal_gate_bias),
            "anchor_proposal_train_branch_floor_start": float(
                config.anchor_proposal_train_branch_floor_start
            ),
            "anchor_proposal_train_branch_floor_end": float(
                config.anchor_proposal_train_branch_floor_end
            ),
            "anchor_proposal_branch_floor_decay_fraction": float(
                config.anchor_proposal_branch_floor_decay_fraction
            ),
            "anchor_proposal_require_difference": bool(
                config.anchor_proposal_require_difference
            ),
        }
        manifest_path = str(
            getattr(config, "anchor_proposal_cf_manifest_path", "") or ""
        ).strip()
        if manifest_path:
            path = Path(manifest_path)
            if path.exists():
                apcf_fields["anchor_proposal_cf_manifest_sha256"] = _sha256_of(
                    path
                )
    pointer_v2_fields: dict[str, Any] = {}
    if is_worker_pointer_v2_mode(config):
        pointer_v2_fields = {
            "worker_pointer_context_version": str(config.worker_pointer_context_version),
            "worker_pointer_pressure_temperature": float(
                config.worker_pointer_pressure_temperature
            ),
            "worker_pointer_supply_epsilon": float(config.worker_pointer_supply_epsilon),
            "worker_pointer_wait_discount_mode": str(
                config.worker_pointer_wait_discount_mode
            ),
            "worker_pointer_v2_dynamic_eft_features": bool(
                config.worker_pointer_v2_dynamic_eft_features
            ),
            "worker_pointer_v2_dynamic_eft_feature_clip": float(
                config.worker_pointer_v2_dynamic_eft_feature_clip
            ),
        }
    return ModelSpec(
        resource_graph_mode=mode,
        policy_action_scope=str(config.policy_action_scope),
        conditional_team_max_candidates=int(config.conditional_team_max_candidates),
        conditional_team_gate_bias=float(config.conditional_team_gate_bias),
        conditional_team_nonbaseline_logit=float(config.conditional_team_nonbaseline_logit),
        conditional_team_scoring_mode=str(config.conditional_team_scoring_mode),
        conditional_team_prior_margin=float(config.conditional_team_prior_margin),
        conditional_team_prior_weight=float(config.conditional_team_prior_weight),
        workforce_binding_mode=str(config.workforce_binding_mode),
        workforce_preallocation_ratio=float(config.workforce_preallocation_ratio),
        team_selection_mode=str(config.team_selection_mode),
        graph_encoder_mode=str(config.graph_encoder_mode),
        actor_context_mode=str(config.actor_context_mode),
        hidden_dim=int(config.hidden_dim),
        num_gat_layers=int(config.num_gat_layers),
        task_feat_dim=int(config.task_feat_dim),
        worker_feat_dim=int(config.worker_feat_dim),
        station_feat_dim=int(config.station_feat_dim),
        num_skill_types=worker_layout.num_skill_types,
        worker_skill_feature_slots=worker_layout.skill_slots,
        worker_feature_layout_version=WORKER_FEATURE_LAYOUT_VERSION,
        **pointer_v2_fields,
        **apcf_fields,
    )


def build_checkpoint_metadata(config: Config, **extra: Any) -> dict[str, Any]:
    metadata = {
        "format_version": FORMAT_VERSION,
        "model_type": "HB-GAT-PN",
        "model_spec": asdict(build_model_spec(config)),
        "config": config.to_flat_dict(),
        "experiment_name": config.experiment_name,
    }
    if str(config.policy_action_scope) == "operation_station_anchor_proposal_team":
        source_sha256 = str(
            getattr(config, "anchor_proposal_pretrain_source_sha256", "") or ""
        ).strip()
        if source_sha256:
            metadata["apcf_pretrain_source_sha256"] = source_sha256
            loaded_key_count = int(
                getattr(config, "apcf_pretrain_loaded_model_key_count", 0) or 0
            )
            if loaded_key_count > 0:
                metadata["apcf_pretrain_loaded_model_key_count"] = loaded_key_count
    if is_worker_pointer_v2_mode(config):
        metadata["training_spec"] = {
            "worker_pointer_v2_replay_mode": str(
                config.worker_pointer_v2_replay_mode
            ),
            "worker_pointer_v2_logical_batch_cap": int(
                config.batch_size
            ),
            "worker_pointer_v2_rollout_group_upper_bound": int(
                config.worker_pointer_v2_rollout_group_upper_bound
            ),
            "worker_pointer_v2_per_sample_heads": True,
            "num_envs": int(config.num_envs),
            "accumulation_steps": int(config.accumulation_steps),
        }
    metadata.update(extra)
    return metadata


def validate_checkpoint_training_spec(
    config: Config,
    metadata: Mapping[str, Any],
) -> None:
    """在恢复训练状态前校验 WorkerPointer v2 的数值重放语义。"""
    if not is_worker_pointer_v2_mode(config):
        return
    saved = metadata.get("training_spec")
    if not isinstance(saved, Mapping):
        raise ValueError(
            "WorkerPointer v2 checkpoint 缺少 training_spec；"
            "旧 256/大批形状语义不得恢复为 behavior_group_exact_v1"
        )
    expected = {
        "worker_pointer_v2_replay_mode": str(
            config.worker_pointer_v2_replay_mode
        ),
        "worker_pointer_v2_rollout_group_upper_bound": int(
            config.worker_pointer_v2_rollout_group_upper_bound
        ),
        "worker_pointer_v2_per_sample_heads": True,
        "accumulation_steps": int(config.accumulation_steps),
    }
    if is_fast_exact_mode(config):
        # Fast-Exact 的 bf16 同形合同依赖 logical batch 与原始行为组形状；
        # 普通 v2 继续保留历史 batch 迁移兼容性。
        expected.update(
            {
                "worker_pointer_v2_logical_batch_cap": int(config.batch_size),
                "num_envs": int(config.num_envs),
            }
        )
    conflicts = {
        key: (saved.get(key), expected_value)
        for key, expected_value in expected.items()
        if saved.get(key) != expected_value
    }
    if conflicts:
        raise ValueError(
            f"WorkerPointer v2 checkpoint training_spec 不兼容: {conflicts}"
        )


def build_resume_batch_audit(
    checkpoint_payload: Mapping[str, Any],
    *,
    current_batch_size: int,
) -> dict[str, int | bool | None]:
    """生成 resume 的 batch 迁移审计信息，不修改 checkpoint 内容。"""
    agent_state = checkpoint_payload.get("apal_agent_state")
    checkpoint_batch_size: int | None = None
    if isinstance(agent_state, Mapping) and agent_state.get("batch_size") is not None:
        checkpoint_batch_size = int(agent_state["batch_size"])
    current = max(1, int(current_batch_size))
    return {
        "checkpoint_batch_size": checkpoint_batch_size,
        "current_batch_size": current,
        "override_applied": (
            checkpoint_batch_size is not None and checkpoint_batch_size != current
        ),
    }


def _strip_policy_prefix(state_dict: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    items = {str(key): value for key, value in state_dict.items() if torch.is_tensor(value)}
    if items and all(key.startswith("policy.") for key in items):
        return {key.removeprefix("policy."): value for key, value in items.items()}
    if any(key.startswith("agent.policy.") for key in items):
        return {
            key.removeprefix("agent.policy."): value
            for key, value in items.items() if key.startswith("agent.policy.")
        }
    return items


def extract_state_dict(payload: Any) -> tuple[dict[str, torch.Tensor], str]:
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint 必须是映射或 state_dict")
    if isinstance(payload.get("state_dict"), Mapping):
        return _strip_policy_prefix(payload["state_dict"]), "lightning"
    if isinstance(payload.get("model_state_dict"), Mapping):
        return _strip_policy_prefix(payload["model_state_dict"]), "legacy_full"
    state = _strip_policy_prefix(payload)
    if not state:
        raise ValueError("checkpoint 中未找到模型权重")
    return state, "raw_state_dict"


def infer_model_spec(state_dict: Mapping[str, torch.Tensor]) -> ModelSpec:
    keys = tuple(state_dict)
    inferred_action_scope = (
        "operation_station_gated_team"
        if any(key.startswith("conditional_team_head.") for key in keys)
        else "operation_station_worker"
    )
    has_direct = any("can_do" in key for key in keys)
    has_hub = any(
        marker in key for key in keys
        for marker in ("skill_emb", "has_skill", "required_by")
    )
    has_reverse = any(
        marker in key for key in keys
        for marker in ("requires", "provided_by")
    )
    if has_direct and has_hub:
        raise ValueError("checkpoint 同时包含旧 can_do 和 Skill Hub 权重")
    if has_reverse and not has_hub:
        raise ValueError("checkpoint 的 Skill Hub 关系不完整")
    mode = (
        "legacy_direct" if has_direct and not has_hub
        else "skill_hub_bidirectional" if has_reverse
        else "skill_hub_forward" if has_hub
        else "legacy_direct"
    )
    task = state_dict.get("embedder.task_emb.0.weight")
    worker = state_dict.get("embedder.worker_emb.0.weight")
    station = state_dict.get("embedder.station_emb.0.weight")
    indexes = {
        int(parts[2])
        for key in keys
        if key.startswith("encoder.layers.")
        and len(parts := key.split(".")) > 2 and parts[2].isdigit()
    }
    worker_feat_dim = int(worker.shape[1]) if worker is not None else None
    is_current_layout = worker_feat_dim == 17
    return ModelSpec(
        resource_graph_mode=mode,
        policy_action_scope=inferred_action_scope,
        hidden_dim=int(task.shape[0]) if task is not None else None,
        num_gat_layers=max(indexes) + 1 if indexes else None,
        task_feat_dim=int(task.shape[1]) if task is not None else None,
        worker_feat_dim=worker_feat_dim,
        station_feat_dim=int(station.shape[1]) if station is not None else None,
        num_skill_types=5 if is_current_layout else None,
        worker_skill_feature_slots=5 if is_current_layout else None,
        worker_feature_layout_version=(WORKER_FEATURE_LAYOUT_VERSION if is_current_layout else None),
    )


def load_checkpoint(path: str | Path, map_location: Any = "cpu") -> LoadedCheckpoint:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    state_dict, format_name = extract_state_dict(payload)
    metadata = dict(payload.get("apal_metadata", {})) if isinstance(payload, Mapping) else {}
    if int(metadata.get("format_version", 0)) != FORMAT_VERSION:
        raise ValueError(
            "checkpoint 格式不兼容：当前实验架构仅接受 format_version=2；"
            "旧模型必须按新的动作范围、编码器和池化配置重新训练。"
        )
    saved_spec = metadata.get("model_spec")
    spec = ModelSpec(**saved_spec) if isinstance(saved_spec, Mapping) else infer_model_spec(state_dict)
    return LoadedCheckpoint(payload, state_dict, spec, metadata, format_name)


def apply_checkpoint_model_spec(
    config: Config,
    spec: ModelSpec,
    *,
    explicit_fields: set[str] | None = None,
) -> None:
    current_team_mode = str(getattr(config, "team_selection_mode", "autoregressive"))
    checkpoint_team_mode = str(spec.team_selection_mode)
    if current_team_mode != checkpoint_team_mode:
        raise ValueError(
            "team_selection_mode 与 checkpoint 不兼容："
            f"config={current_team_mode!r}, checkpoint={checkpoint_team_mode!r}"
        )
    if is_worker_pointer_v2_mode(config):
        semantic_fields = (
            "worker_pointer_context_version",
            "worker_pointer_pressure_temperature",
            "worker_pointer_supply_epsilon",
            "worker_pointer_wait_discount_mode",
            "worker_pointer_v2_dynamic_eft_features",
            "worker_pointer_v2_dynamic_eft_feature_clip",
        )
        conflicts = {
            key: (getattr(config, key), getattr(spec, key))
            for key in semantic_fields
            if getattr(spec, key) is None or getattr(config, key) != getattr(spec, key)
        }
        if conflicts:
            raise ValueError(f"WorkerPointer v2 checkpoint 语义不兼容: {conflicts}")
    current_scope = str(getattr(config, "policy_action_scope", ""))
    current_scoring_mode = str(
        getattr(config, "conditional_team_scoring_mode", "fixed_prior_v1")
    )
    if (
        current_scope == "operation_station_gated_team"
        and spec.policy_action_scope == "operation_station_gated_team"
        and current_scoring_mode != spec.conditional_team_scoring_mode
    ):
        raise ValueError(
            "条件式团队评分模式与 checkpoint 不一致："
            f"config={current_scoring_mode!r}, "
            f"checkpoint={spec.conditional_team_scoring_mode!r}"
        )
    current_layout = resolve_worker_feature_layout(config)
    if (
        spec.worker_feat_dim is not None
        and int(spec.worker_feat_dim) != current_layout.total_dim
    ):
        raise ValueError(
            "checkpoint 的工人特征布局与当前五技能布局不兼容："
            f"checkpoint worker_feat_dim={spec.worker_feat_dim}，"
            f"当前要求 {current_layout.total_dim}。"
            "历史 22 维 checkpoint 不能用于当前模型，请使用 17 维五技能 checkpoint 或重新训练。"
        )
    if spec.num_skill_types not in (None, current_layout.num_skill_types):
        raise ValueError(
            "checkpoint 的工种数量与当前配置不兼容："
            f"checkpoint={spec.num_skill_types}，当前={current_layout.num_skill_types}。"
        )
    if spec.worker_skill_feature_slots not in (None, current_layout.skill_slots):
        raise ValueError(
            "checkpoint 的工人技能槽位与当前配置不兼容："
            f"checkpoint={spec.worker_skill_feature_slots}，当前={current_layout.skill_slots}。"
        )
    inferred = {
        "use_skill_hub": spec.use_skill_hub,
        "skill_hub_bidirectional": spec.skill_hub_bidirectional,
        "policy_action_scope": spec.policy_action_scope,
        "conditional_team_max_candidates": spec.conditional_team_max_candidates,
        "conditional_team_gate_bias": spec.conditional_team_gate_bias,
        "conditional_team_nonbaseline_logit": spec.conditional_team_nonbaseline_logit,
        "conditional_team_scoring_mode": spec.conditional_team_scoring_mode,
        "conditional_team_prior_margin": spec.conditional_team_prior_margin,
        "conditional_team_prior_weight": spec.conditional_team_prior_weight,
        "workforce_binding_mode": spec.workforce_binding_mode,
        "workforce_preallocation_ratio": spec.workforce_preallocation_ratio,
        "team_selection_mode": spec.team_selection_mode,
        "graph_encoder_mode": spec.graph_encoder_mode,
        "actor_context_mode": spec.actor_context_mode,
    }
    for key in (
        "worker_pointer_context_version",
        "worker_pointer_pressure_temperature",
        "worker_pointer_supply_epsilon",
        "worker_pointer_wait_discount_mode",
        "worker_pointer_v2_dynamic_eft_features",
        "worker_pointer_v2_dynamic_eft_feature_clip",
    ):
        value = getattr(spec, key)
        if value is not None:
            inferred[key] = value
    if spec.policy_action_scope == "operation_station_anchor_proposal_team":
        for key in (
            "anchor_proposal_mode",
            "anchor_proposal_prior_margin",
            "anchor_proposal_gate_bias",
            "anchor_proposal_train_branch_floor_start",
            "anchor_proposal_train_branch_floor_end",
            "anchor_proposal_branch_floor_decay_fraction",
            "anchor_proposal_require_difference",
        ):
            value = getattr(spec, key)
            if value is not None:
                inferred[key] = value
    for key in (
        "hidden_dim",
        "num_gat_layers",
        "task_feat_dim",
        "worker_feat_dim",
        "station_feat_dim",
        "num_skill_types",
        "worker_skill_feature_slots",
    ):
        value = getattr(spec, key)
        if value is not None:
            inferred[key] = value
    conflicts = {
        key: (getattr(config, key), value)
        for key, value in inferred.items()
        if key in (explicit_fields or set()) and getattr(config, key) != value
    }
    if conflicts:
        details = ", ".join(
            f"{key}: CLI={current!r}, checkpoint={saved!r}"
            for key, (current, saved) in conflicts.items()
        )
        raise ValueError(f"显式模型结构参数与 checkpoint 冲突: {details}")
    config.update_from_dict(inferred)


def load_policy_weights(model: torch.nn.Module, checkpoint: LoadedCheckpoint, *, strict: bool = True) -> Any:
    return model.load_state_dict(checkpoint.state_dict, strict=strict)
