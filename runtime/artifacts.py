from __future__ import annotations

import json
import subprocess
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


def checkpoint_paths(config: Config, project_root: Path) -> dict[str, Path]:
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
            ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
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
