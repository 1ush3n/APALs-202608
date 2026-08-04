"""初始调度生产训练池的 manifest 解析与精确文件绑定。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from runtime.five_skill_schema import ALLOWED_PRODUCTION_PROTOCOLS, validate_explicit_five_skill_csv
from runtime.paths import resolve_workspace_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_explicit_five_skill_initial_training_paths(
    manifest_path: str | Path,
    configured_data_path: str | Path,
) -> tuple[Path, ...]:
    """只允许 manifest 已声明、哈希一致的训练 CSV 进入初始调度训练。"""
    path = resolve_workspace_path(manifest_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"初始调度训练 manifest 不存在: {path}")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("protocol", "")).strip() not in ALLOWED_PRODUCTION_PROTOCOLS:
        raise ValueError("初始调度训练只接受 explicit_fiveskill_v1 manifest")
    rows = payload.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("初始调度训练 manifest 的 files 必须是非空列表")
    directory = resolve_workspace_path(configured_data_path).resolve()
    if not directory.is_dir():
        raise ValueError("初始调度训练的 train_data_path_or_dir 必须是 manifest CSV 所在目录")
    declared: list[Path] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("初始调度训练 manifest 的 files 条目必须是对象")
        name = str(row.get("file", "")).strip()
        expected_hash = str(row.get("sha256", "")).strip().lower()
        candidate = (directory / name).resolve()
        if not name or candidate.parent != directory or not candidate.is_file():
            raise ValueError(f"manifest 声明的训练 CSV 不存在或越界: {name!r}")
        if not expected_hash or _sha256(candidate) != expected_hash:
            raise ValueError(f"初始调度训练 CSV 哈希不一致: {candidate}")
        validate_explicit_five_skill_csv(candidate, require_all_skills=True)
        declared.append(candidate)
    if len(set(declared)) != len(declared):
        raise ValueError("初始调度训练 manifest 存在重复 CSV")
    discovered = tuple(sorted(item.resolve() for item in directory.iterdir() if item.suffix.lower() == ".csv"))
    if tuple(sorted(declared)) != discovered:
        extra = sorted(str(item) for item in set(discovered) - set(declared))
        missing = sorted(str(item) for item in set(declared) - set(discovered))
        raise ValueError(f"初始训练目录与 manifest 精确文件列表不一致: extra={extra}; missing={missing}")
    return tuple(declared)


__all__ = ["resolve_explicit_five_skill_initial_training_paths"]
