from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from configs import configs
from runtime.artifacts import (
    checkpoint_paths as resolve_artifact_checkpoint_paths,
    resolve_path as resolve_artifact_path,
    sanitize_name,
    uses_runs_layout,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_workspace_path(path_like, base_dir: Path = PROJECT_ROOT) -> Path:
    """将配置中的路径解析为跨平台绝对路径；绝对路径保持不变。"""
    return resolve_artifact_path(path_like, base_dir)


def sanitize_experiment_name(name: object) -> str:
    """将实验名压缩为安全目录名，避免不同配置的 checkpoint 互相覆盖。"""
    return sanitize_name(name)


def resolve_checkpoint_paths(config_obj=configs) -> dict[str, Path]:
    """按 experiment_name/checkpoint_root 解析当前实验的模型保存路径。"""
    paths = resolve_artifact_checkpoint_paths(config_obj, PROJECT_ROOT)
    model_dir = paths["model_dir"]
    best_model_dir = paths["legacy_best"].parent
    return {
        "model_dir": model_dir,
        "checkpoint_path": paths["legacy_latest"],
        "best_model_dir": best_model_dir,
        "best_model_path": paths["legacy_best"],
        "best_model_meta_path": paths["legacy_best_meta"],
        "lightning_dir": paths["lightning_dir"],
        "lightning_latest": paths["lightning_latest"],
        "lightning_best": paths["lightning_best"],
    }


def resolve_tensorboard_log_root(config_obj=configs) -> Path:
    """严格使用配置中的 TensorBoard 根目录，不按平台隐式改写。"""
    if uses_runs_layout(config_obj) and str(getattr(config_obj, "run_dir", "") or "").strip():
        return Path(config_obj.run_dir) / "logs" / "tensorboard"
    return Path(getattr(config_obj, "log_dir", "/root/tf-logs")).expanduser()


def write_best_model_meta(
    meta_path: Path,
    *,
    episode: int,
    eval_makespan: float,
    selection_metric: str = "eval_makespan",
    best_score: float | None = None,
    score_terms: dict[str, float] | None = None,
    constraint_metrics: dict[str, float] | None = None,
    config_obj=configs,
) -> None:
    """保存 best model 的可追溯元数据，方便服务器和本机定位模型来源。"""
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "episode": int(episode),
        "selection_metric": selection_metric,
        "eval_makespan": float(eval_makespan),
        "best_score": None if best_score is None else float(best_score),
        "score_terms": score_terms or {},
        "constraint_metrics": constraint_metrics or {},
        "config_paths": list(getattr(config_obj, "config_paths", ())),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment_name": sanitize_experiment_name(getattr(config_obj, "experiment_name", "default")),
        "data_file_path": getattr(config_obj, "data_file_path", ""),
        "train_data_path_or_dir": getattr(config_obj, "train_data_path_or_dir", ""),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

__all__ = [
    "PROJECT_ROOT",
    "resolve_checkpoint_paths",
    "resolve_tensorboard_log_root",
    "resolve_workspace_path",
    "sanitize_experiment_name",
    "write_best_model_meta",
]
