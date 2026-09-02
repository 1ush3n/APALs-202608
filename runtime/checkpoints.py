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
    policy_observation_scope: str = "full"
    graph_input_scope: str = "match_policy"
    critic_observation_scope: str = "match_policy"
    task_feature_scope: str = "full"
    task_mask_mode: str = "resource_aware"
    station_mask_mode: str = "resource_aware"
    action_completion_mode: str = "earliest_finish"
    ablation_protocol: str = "legacy"
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
    worker_pointer_v2_fast_replay_batching: str = "physical_group"
    worker_pointer_v2_explicit_team_state: bool = False
    worker_pointer_v2_marginal_scarcity: bool = False
    worker_pointer_v2_marginal_scarcity_clip: float = 10.0
    worker_pointer_v2_interaction_residual: bool = False
    worker_pointer_v2_next_frontier_pressure: bool = False
    reschedule_baseline_identity_conditioning: bool = False
    conditional_head_baseline_mode: str = "off"
    graph_encoder_mode: str = "hetero_gat"
    actor_context_mode: str = "attention"
    homogeneous_use_type_embedding: bool = True
    homogeneous_shared_input_projection: bool = False
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
            "worker_pointer_v2_fast_replay_batching": str(
                getattr(config, "worker_pointer_v2_fast_replay_batching", "physical_group")
            ),
            "worker_pointer_v2_explicit_team_state": bool(
                config.worker_pointer_v2_explicit_team_state
            ),
            "worker_pointer_v2_marginal_scarcity": bool(
                config.worker_pointer_v2_marginal_scarcity
            ),
            "worker_pointer_v2_marginal_scarcity_clip": float(
                config.worker_pointer_v2_marginal_scarcity_clip
            ),
            "worker_pointer_v2_interaction_residual": bool(
                config.worker_pointer_v2_interaction_residual
            ),
            "worker_pointer_v2_next_frontier_pressure": bool(
                config.worker_pointer_v2_next_frontier_pressure
            ),
        }
    return ModelSpec(
        resource_graph_mode=mode,
        policy_action_scope=str(config.policy_action_scope),
        policy_observation_scope=str(getattr(config, "policy_observation_scope", "full")),
        graph_input_scope=str(getattr(config, "graph_input_scope", "match_policy")),
        critic_observation_scope=str(
            getattr(config, "critic_observation_scope", "match_policy")
        ),
        task_feature_scope=str(getattr(config, "task_feature_scope", "full")),
        task_mask_mode=str(getattr(config, "task_mask_mode", "resource_aware")),
        station_mask_mode=str(
            getattr(config, "station_mask_mode", "resource_aware")
        ),
        action_completion_mode=str(
            getattr(config, "action_completion_mode", "earliest_finish")
        ),
        ablation_protocol=str(getattr(config, "ablation_protocol", "legacy")),
        conditional_team_max_candidates=int(config.conditional_team_max_candidates),
        conditional_team_gate_bias=float(config.conditional_team_gate_bias),
        conditional_team_nonbaseline_logit=float(config.conditional_team_nonbaseline_logit),
        conditional_team_scoring_mode=str(config.conditional_team_scoring_mode),
        conditional_team_prior_margin=float(config.conditional_team_prior_margin),
        conditional_team_prior_weight=float(config.conditional_team_prior_weight),
        workforce_binding_mode=str(config.workforce_binding_mode),
        workforce_preallocation_ratio=float(config.workforce_preallocation_ratio),
        team_selection_mode=str(config.team_selection_mode),
        conditional_head_baseline_mode=str(
            getattr(config, "conditional_head_baseline_mode", "off")
        ),
        graph_encoder_mode=str(config.graph_encoder_mode),
        actor_context_mode=str(config.actor_context_mode),
        homogeneous_use_type_embedding=bool(
            getattr(config, "homogeneous_use_type_embedding", True)
        ),
        homogeneous_shared_input_projection=bool(
            getattr(config, "homogeneous_shared_input_projection", False)
        ),
        hidden_dim=int(config.hidden_dim),
        num_gat_layers=int(config.num_gat_layers),
        task_feat_dim=int(config.task_feat_dim),
        worker_feat_dim=int(config.worker_feat_dim),
        station_feat_dim=int(config.station_feat_dim),
        num_skill_types=worker_layout.num_skill_types,
        worker_skill_feature_slots=worker_layout.skill_slots,
        worker_feature_layout_version=WORKER_FEATURE_LAYOUT_VERSION,
        reschedule_baseline_identity_conditioning=bool(
            getattr(config, "reschedule_baseline_identity_conditioning", False)
        ),
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
            "worker_pointer_v2_fast_replay_batching": str(
                getattr(config, "worker_pointer_v2_fast_replay_batching", "physical_group")
            ),
            "worker_pointer_v2_logical_batch_cap": int(
                config.batch_size
            ),
            "worker_pointer_v2_rollout_group_upper_bound": int(
                config.worker_pointer_v2_rollout_group_upper_bound
            ),
            "worker_pointer_v2_per_sample_heads": True,
            "worker_pointer_v2_fast_replay_encoder_batch_cap": int(
                getattr(config, "worker_pointer_v2_fast_replay_encoder_batch_cap", 16)
            ),
            "num_envs": int(config.num_envs),
            "accumulation_steps": int(config.accumulation_steps),
            "conditional_head_value_coef": float(config.conditional_head_value_coef),
        }
    metadata.update(extra)
    return metadata


def validate_checkpoint_training_spec(
    config: Config,
    metadata: Mapping[str, Any],
) -> None:
    """在恢复训练状态前校验 WorkerPointer v2 的数值重放语义。"""
    current_completion_mode = str(
        getattr(config, "action_completion_mode", "earliest_finish")
    ).lower()
    saved_config = metadata.get("config")
    saved_completion_mode = (
        str(saved_config.get("action_completion_mode", "earliest_finish")).lower()
        if isinstance(saved_config, Mapping)
        else "earliest_finish"
    )
    valid_completion_modes = {"earliest_finish", "earliest_availability"}
    if saved_completion_mode not in valid_completion_modes:
        raise ValueError(
            "checkpoint 的 action_completion_mode 无效："
            f"{saved_completion_mode!r}"
        )
    if current_completion_mode != saved_completion_mode:
        raise ValueError(
            "action_completion_mode 与 checkpoint 不兼容："
            f"config={current_completion_mode!r}, checkpoint={saved_completion_mode!r}"
        )
    if bool(metadata.get("conditional_head_metadata_missing", False)):
        raise ValueError(
            "checkpoint 包含 conditional value head，但缺少 "
            "conditional_head_baseline_mode；禁止用于 training resume，"
            "请显式指定模式后仅用于 evaluation"
        )
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
        "worker_pointer_v2_fast_replay_batching": str(
            getattr(config, "worker_pointer_v2_fast_replay_batching", "physical_group")
        ),
        "worker_pointer_v2_rollout_group_upper_bound": int(
            config.worker_pointer_v2_rollout_group_upper_bound
        ),
        "worker_pointer_v2_per_sample_heads": True,
        "worker_pointer_v2_fast_replay_encoder_batch_cap": int(
            getattr(config, "worker_pointer_v2_fast_replay_encoder_batch_cap", 16)
        ),
        "accumulation_steps": int(config.accumulation_steps),
        "conditional_head_value_coef": float(config.conditional_head_value_coef),
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
    training_defaults = {
        "worker_pointer_v2_fast_replay_batching": "physical_group",
        "worker_pointer_v2_fast_replay_encoder_batch_cap": 16,
    }
    conflicts = {
        key: (saved.get(key), expected_value)
        for key, expected_value in expected.items()
        if saved.get(key, training_defaults.get(key)) != expected_value
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


def _infer_worker_pointer_v2_architecture(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, bool]:
    task = state_dict.get("embedder.task_emb.0.weight")
    query = state_dict.get("worker_head.v2_query_proj.weight")
    hidden_dim = int(task.shape[0]) if task is not None and task.ndim == 2 else None
    query_input_width = (
        int(query.shape[1]) if query is not None and query.ndim == 2 else None
    )
    return {
        "worker_pointer_v2_explicit_team_state": bool(
            hidden_dim is not None
            and query_input_width == hidden_dim * 6 + 19
        ),
        "worker_pointer_v2_marginal_scarcity": any(
            key.startswith("worker_head.v2_marginal_proj.")
            for key in state_dict
        ),
        "worker_pointer_v2_interaction_residual": any(
            key.startswith("worker_head.v2_interaction_mlp.")
            for key in state_dict
        ),
        "worker_pointer_v2_next_frontier_pressure": any(
            key.startswith("worker_head.v2_next_frontier_query_proj.")
            for key in state_dict
        ),
        "reschedule_baseline_identity_conditioning": any(
            key.startswith((
                "station_head.baseline_station_proj.",
                "worker_head.baseline_worker_proj.",
            ))
            for key in state_dict
        ),
    }


def _has_conditional_value_head(state_dict: Mapping[str, torch.Tensor]) -> bool:
    return any(
        key.startswith(("critic_station_cond.", "critic_worker_cond."))
        for key in state_dict
    )


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
    v2_architecture = _infer_worker_pointer_v2_architecture(state_dict)
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
        worker_pointer_v2_dynamic_eft_features=(
            "worker_head.v2_eft_proj.weight" in state_dict
        ),
        **v2_architecture,
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
    eft_enabled = "worker_head.v2_eft_proj.weight" in state_dict
    inferred_architecture = _infer_worker_pointer_v2_architecture(state_dict)
    conditional_head_present = _has_conditional_value_head(state_dict)
    saved_spec = metadata.get("model_spec")
    if isinstance(saved_spec, Mapping):
        spec_values = dict(saved_spec)
        for key, value in inferred_architecture.items():
            spec_values.setdefault(key, value)
        spec_values.setdefault(
            "worker_pointer_v2_dynamic_eft_features",
            eft_enabled,
        )
        if conditional_head_present and "conditional_head_baseline_mode" not in spec_values:
            metadata["conditional_head_metadata_missing"] = True
        if "worker_pointer_v2_dynamic_eft_feature_clip" not in spec_values:
            saved_config = metadata.get("config")
            if isinstance(saved_config, Mapping):
                clip = saved_config.get("worker_pointer_v2_dynamic_eft_feature_clip")
                if clip is not None:
                    spec_values["worker_pointer_v2_dynamic_eft_feature_clip"] = clip
        if (
            eft_enabled
            and spec_values.get("worker_pointer_v2_dynamic_eft_features") is True
            and "worker_pointer_v2_dynamic_eft_feature_clip" not in spec_values
        ):
            raise ValueError(
                "checkpoint 启用了动态 EFT，但缺少可验证的 "
                "worker_pointer_v2_dynamic_eft_feature_clip"
            )
        spec = ModelSpec(**spec_values)
    else:
        spec = infer_model_spec(state_dict)
        if conditional_head_present:
            metadata["conditional_head_metadata_missing"] = True
        if eft_enabled:
            saved_config = metadata.get("config")
            clip = (
                saved_config.get("worker_pointer_v2_dynamic_eft_feature_clip")
                if isinstance(saved_config, Mapping)
                else None
            )
            if clip is None:
                raise ValueError(
                    "checkpoint 启用了动态 EFT，但缺少可验证的 "
                    "worker_pointer_v2_dynamic_eft_feature_clip"
                )
            spec = ModelSpec(
                **{
                    **asdict(spec),
                    "worker_pointer_v2_dynamic_eft_feature_clip": clip,
                }
            )
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
            "worker_pointer_v2_explicit_team_state",
            "worker_pointer_v2_marginal_scarcity",
            "worker_pointer_v2_marginal_scarcity_clip",
            "worker_pointer_v2_interaction_residual",
            "worker_pointer_v2_next_frontier_pressure",
            "worker_pointer_v2_fast_replay_batching",
            "reschedule_baseline_identity_conditioning",
        )
        explicit = explicit_fields or set()
        if not (
            str(spec.conditional_head_baseline_mode) == "off"
            and str(getattr(config, "conditional_head_baseline_mode", "off")) != "off"
            and "conditional_head_baseline_mode" in explicit
        ):
            semantic_fields += ("conditional_head_baseline_mode",)
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
        "policy_observation_scope": getattr(spec, "policy_observation_scope", "full"),
        "graph_input_scope": getattr(spec, "graph_input_scope", "match_policy"),
        "critic_observation_scope": getattr(
            spec, "critic_observation_scope", "match_policy"
        ),
        "task_feature_scope": getattr(spec, "task_feature_scope", "full"),
        "task_mask_mode": getattr(spec, "task_mask_mode", "resource_aware"),
        "station_mask_mode": getattr(spec, "station_mask_mode", "resource_aware"),
        "action_completion_mode": getattr(
            spec, "action_completion_mode", "earliest_finish"
        ),
        "ablation_protocol": getattr(spec, "ablation_protocol", "legacy"),
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
        "homogeneous_use_type_embedding": getattr(
            spec, "homogeneous_use_type_embedding", True
        ),
        "homogeneous_shared_input_projection": getattr(
            spec, "homogeneous_shared_input_projection", False
        ),
        "reschedule_baseline_identity_conditioning": bool(
            getattr(spec, "reschedule_baseline_identity_conditioning", False)
        ),
    }
    explicit_conditional_mode = (
        "conditional_head_baseline_mode" in (explicit_fields or set())
        and str(getattr(config, "conditional_head_baseline_mode", "off")) != "off"
        and str(spec.conditional_head_baseline_mode) == "off"
    )
    if not explicit_conditional_mode:
        inferred["conditional_head_baseline_mode"] = spec.conditional_head_baseline_mode
    for key in (
        "worker_pointer_context_version",
        "worker_pointer_pressure_temperature",
        "worker_pointer_supply_epsilon",
        "worker_pointer_wait_discount_mode",
        "worker_pointer_v2_dynamic_eft_features",
        "worker_pointer_v2_dynamic_eft_feature_clip",
        "worker_pointer_v2_explicit_team_state",
        "worker_pointer_v2_marginal_scarcity",
        "worker_pointer_v2_marginal_scarcity_clip",
        "worker_pointer_v2_interaction_residual",
        "worker_pointer_v2_next_frontier_pressure",
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
