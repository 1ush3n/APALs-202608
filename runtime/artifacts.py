from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from configs import Config


def resolve_path(path_like: str | Path, project_root: Path) -> Path:
    path = Path(path_like).expanduser()
    return path if path.is_absolute() else project_root / path


def sanitize_name(value: object) -> str:
    raw = str(value or "default").strip()
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in raw) or "default"


@dataclass(frozen=True)
class RunContext:
    """统一描述一次实验运行的所有输出目录。"""

    experiment_name: str
    run_id: str
    run_dir: Path
    checkpoint_dir: Path
    logs_dir: Path
    tensorboard_dir: Path
    eval_dir: Path
    configs_dir: Path
    artifacts_dir: Path
    reports_dir: Path
    traces_dir: Path


def format_run_timestamp(now: datetime | None = None) -> str:
    """生成紧凑时间戳，例如 260630-153000。"""
    return (now or datetime.now()).strftime("%y%m%d-%H%M%S")


def build_run_id(config: Config, now: datetime | None = None) -> str:
    """按 <experiment>_<YYMMDD-HHMMSS> 生成可读且不覆盖的 run_id。"""
    existing = str(getattr(config, "run_id", "") or "").strip()
    if existing:
        return sanitize_name(existing)
    experiment = sanitize_name(getattr(config, "experiment_name", "default"))
    return f"{experiment}_{format_run_timestamp(now)}"


def uses_runs_layout(config: Config) -> bool:
    return str(getattr(config, "artifact_layout", "runs")).lower() == "runs"


def run_context(
    config: Config,
    project_root: Path,
    *,
    create_dirs: bool = False,
) -> RunContext:
    """解析并可选创建统一 runs 输出目录。"""
    experiment = sanitize_name(getattr(config, "experiment_name", "default"))
    run_id = build_run_id(config)
    setattr(config, "run_id", run_id)
    root = resolve_path(getattr(config, "runs_root", "runs"), project_root)
    run_dir = root / experiment / run_id
    setattr(config, "run_dir", str(run_dir))

    context = RunContext(
        experiment_name=experiment,
        run_id=run_id,
        run_dir=run_dir,
        checkpoint_dir=run_dir / "checkpoints",
        logs_dir=run_dir / "logs",
        tensorboard_dir=run_dir / "logs" / "tensorboard",
        eval_dir=run_dir / "eval",
        configs_dir=run_dir / "configs",
        artifacts_dir=run_dir / "artifacts",
        reports_dir=run_dir / "artifacts" / "reports",
        traces_dir=run_dir / "artifacts" / "traces",
    )
    if create_dirs:
        for directory in (
            context.checkpoint_dir,
            context.tensorboard_dir,
            context.eval_dir,
            context.configs_dir,
            context.reports_dir,
            context.traces_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
    return context


def checkpoint_paths(config: Config, project_root: Path) -> dict[str, Path]:
    if uses_runs_layout(config) and str(getattr(config, "run_id", "") or "").strip():
        context = run_context(config, project_root)
        model_dir = context.checkpoint_dir
        return {
            "model_dir": model_dir,
            "legacy_latest": model_dir / "legacy" / "latest_checkpoint.pth",
            "legacy_best": model_dir / "legacy" / "best_model.pth",
            "legacy_best_meta": model_dir / "legacy" / "best_model_meta.json",
            "lightning_dir": model_dir,
            "lightning_latest": model_dir / "last.ckpt",
            "lightning_best": model_dir / "best.ckpt",
        }

    model_dir = resolve_path(config.checkpoint_root, project_root) / sanitize_name(config.experiment_name)
    return {
        "model_dir": model_dir,
        "legacy_latest": model_dir / "latest_checkpoint.pth",
        "legacy_best": model_dir / "bestmodel" / "best_model.pth",
        "legacy_best_meta": model_dir / "bestmodel" / "best_model_meta.json",
        "lightning_dir": model_dir / "lightning",
        "lightning_latest": model_dir / "lightning" / "last.ckpt",
        "lightning_best": model_dir / "lightning" / "best" / "best.ckpt",
    }


def write_run_manifest(
    directory: Path,
    config: Config,
    *,
    command: str,
    extra: dict[str, Any] | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    import yaml
    (directory / "resolved_config.yaml").write_text(
        yaml.safe_dump(config.to_flat_dict(), allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        git_commit = None
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": command,
        "experiment_name": sanitize_name(config.experiment_name),
        "config_paths": list(config.config_paths),
        "git_commit": git_commit,
        **(extra or {}),
    }
    (directory / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_run_context_files(
    context: RunContext,
    config: Config,
    *,
    command: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """将最终配置与运行清单写入统一 configs 目录。"""
    write_run_manifest(
        context.configs_dir,
        config,
        command=command,
        extra={
            "run_id": context.run_id,
            "run_dir": str(context.run_dir.resolve()),
            **(extra or {}),
        },
    )
