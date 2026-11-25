from __future__ import annotations

import json
import hashlib
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from configs import Config
from runtime.checkpoints import build_model_spec
from runtime.modes import is_worker_pointer_v2_mode


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
    manifest = build_run_manifest_payload(config, command=command, extra=extra)
    (directory / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_run_manifest_payload(
    config: Config,
    *,
    command: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造可测试的运行语义清单；文件写入由调用方单独负责。"""
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError, UnicodeError):
        git_commit = None
    precision = str(getattr(config, "lightning_precision", "16-mixed")).lower()
    autocast_dtype = {
        "16-mixed": "float16",
        "bf16-mixed": "bfloat16",
        "32-true": None,
    }.get(precision)
    try:
        git_status = subprocess.check_output(
            ["git", "status", "--short"],
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError, UnicodeError):
        git_status = []
    training_manifest_raw = str(getattr(config, "training_manifest_path", "") or "").strip()
    training_manifest_path = Path(training_manifest_raw) if training_manifest_raw else None
    training_manifest_sha256 = None
    if training_manifest_path is not None and training_manifest_path.is_file():
        digest = hashlib.sha256()
        with training_manifest_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        training_manifest_sha256 = digest.hexdigest()
    v2_mode = is_worker_pointer_v2_mode(config)
    requested_logical_cap = (
        int(getattr(config, "batch_size", 1))
        if v2_mode
        else None
    )
    effective_logical_cap = (
        int(getattr(config, "batch_size", 1))
        if requested_logical_cap is not None
        else None
    )
    try:
        import torch

        deterministic_algorithms = bool(
            torch.are_deterministic_algorithms_enabled()
        )
        warn_only = bool(
            getattr(
                torch,
                "is_deterministic_algorithms_warn_only_enabled",
                lambda: False,
            )()
        )
        cudnn_deterministic = bool(torch.backends.cudnn.deterministic)
        cudnn_benchmark = bool(torch.backends.cudnn.benchmark)
    except (ImportError, RuntimeError):
        deterministic_algorithms = False
        warn_only = False
        cudnn_deterministic = False
        cudnn_benchmark = False
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": command,
        "experiment_name": sanitize_name(config.experiment_name),
        "config_paths": list(config.config_paths),
        "git_commit": git_commit,
        "git_worktree_dirty": bool(git_status),
        "git_status": git_status,
        "evaluation_protocol": str(getattr(config, "evaluation_protocol", "standard")),
        "model_spec": asdict(build_model_spec(config)),
        "training_manifest_path": training_manifest_raw or None,
        "training_manifest_sha256": training_manifest_sha256,
        "runtime": {
            "num_envs": int(getattr(config, "num_envs", 1)),
            "batch_size": int(getattr(config, "batch_size", 1)),
            "accumulation_steps": int(getattr(config, "accumulation_steps", 1)),
            "lightning_precision": precision,
            "autocast_dtype": autocast_dtype,
            "grad_scaler_enabled": precision == "16-mixed",
            "worker_pointer_v2_replay_mode": (
                str(getattr(config, "worker_pointer_v2_replay_mode", ""))
                if v2_mode
                else None
            ),
            "requested_logical_batch_cap": requested_logical_cap,
            "effective_logical_batch_cap": effective_logical_cap,
            "rollout_group_upper_bound": (
                int(getattr(config, "worker_pointer_v2_rollout_group_upper_bound", 4))
                if v2_mode
                else None
            ),
            "target_max_samples_per_optimizer_step": (
                int(effective_logical_cap)
                * int(getattr(config, "accumulation_steps", 1))
                if effective_logical_cap is not None
                else None
            ),
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
            "deterministic_algorithms_enabled": deterministic_algorithms,
            "deterministic_algorithms_warn_only": warn_only,
            "cudnn_deterministic": cudnn_deterministic,
            "cudnn_benchmark": cudnn_benchmark,
            "worker_pointer_v2_init_seed": (
                int(getattr(config, "seed", 0))
                + int(getattr(config, "worker_pointer_v2_init_seed_offset", 1009))
                if v2_mode
                else None
            ),
            "async_eval_enabled": bool(
                getattr(config, "async_eval_enabled", False)
            ),
            "async_eval_device": str(
                getattr(config, "async_eval_device", "cpu")
            ),
            "async_eval_worker_count": int(
                getattr(config, "async_eval_worker_count", 1)
            ),
            "async_eval_queue_capacity": int(
                getattr(config, "async_eval_queue_capacity", 4)
            ),
            "async_eval_submit_every_episodes": int(
                getattr(config, "async_eval_submit_every_episodes", 1)
            ),
            "async_eval_wait_on_finish": bool(
                getattr(config, "async_eval_wait_on_finish", True)
            ),
        },
        **(extra or {}),
    }
    return manifest


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


def resolve_run_output_dir(
    config: Config,
    project_root: Path,
    *,
    default_legacy_dir: str | Path,
    run_subdir: str | Path,
    explicit_dir: str | Path | None = None,
    section: str = "artifacts",
    create_dirs: bool = True,
) -> tuple[Path, RunContext | None]:
    """解析工具脚本输出目录；默认进入 runs，显式 output-dir 时尊重用户路径。"""
    if explicit_dir:
        output_dir = resolve_path(explicit_dir, project_root)
        if create_dirs:
            output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir, None

    if uses_runs_layout(config):
        context = run_context(config, project_root, create_dirs=create_dirs)
        base = context.eval_dir if section == "eval" else context.artifacts_dir
        output_dir = base / Path(run_subdir)
    else:
        context = None
        output_dir = resolve_path(default_legacy_dir, project_root)

    if create_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, context

def assert_apcf_smoke_output_isolated(
    smoke_root: Path,
    output_paths: list[Path],
) -> None:
    """?? APCF PPO smoke ????????????"""
    root = Path(smoke_root).expanduser().resolve()
    if not output_paths:
        raise ValueError("APCF smoke ????????????")
    for raw_path in output_paths:
        path = Path(raw_path).expanduser().resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"APCF smoke ???????? smoke ?????root={root}, path={path}"
            ) from exc
