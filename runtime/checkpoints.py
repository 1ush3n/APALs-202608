from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from configs import Config
from worker_feature_layout import resolve_worker_feature_layout


FORMAT_VERSION = 1
WORKER_FEATURE_LAYOUT_VERSION = "five_skill_v2"


@dataclass(frozen=True)
class ModelSpec:
    resource_graph_mode: str
    hidden_dim: int | None = None
    num_gat_layers: int | None = None
    task_feat_dim: int | None = None
    worker_feat_dim: int | None = None
    station_feat_dim: int | None = None
    num_skill_types: int | None = None
    worker_skill_feature_slots: int | None = None
    worker_feature_layout_version: str | None = None

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
    return ModelSpec(
        resource_graph_mode=mode,
        hidden_dim=int(config.hidden_dim),
        num_gat_layers=int(config.num_gat_layers),
        task_feat_dim=int(config.task_feat_dim),
        worker_feat_dim=int(config.worker_feat_dim),
        station_feat_dim=int(config.station_feat_dim),
        num_skill_types=worker_layout.num_skill_types,
        worker_skill_feature_slots=worker_layout.skill_slots,
        worker_feature_layout_version=WORKER_FEATURE_LAYOUT_VERSION,
    )


def build_checkpoint_metadata(config: Config, **extra: Any) -> dict[str, Any]:
    metadata = {
        "format_version": FORMAT_VERSION,
        "model_type": "HB-GAT-PN",
        "model_spec": asdict(build_model_spec(config)),
        "config": config.to_flat_dict(),
        "experiment_name": config.experiment_name,
    }
    metadata.update(extra)
    return metadata


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
    saved_spec = metadata.get("model_spec")
    spec = ModelSpec(**saved_spec) if isinstance(saved_spec, Mapping) else infer_model_spec(state_dict)
    return LoadedCheckpoint(payload, state_dict, spec, metadata, format_name)


def apply_checkpoint_model_spec(
    config: Config,
    spec: ModelSpec,
    *,
    explicit_fields: set[str] | None = None,
) -> None:
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
    }
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
