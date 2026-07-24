"""初始调度真实数据集的旧版工人数映射。"""

from __future__ import annotations

from pathlib import Path
from typing import Any


# 与 conf/env/initial_bucket_{283,680,2338,3182}.yaml 保持一致。
INITIAL_DATASET_WORKERS: dict[str, int] = {
    "283": 80,
    "680": 100,
    "2338": 140,
    "3182": 160,
}


def resolve_initial_worker_count(data_path: str | Path) -> int | None:
    """按真实初始调度数据集文件名解析旧版固定工人数。"""
    stem = Path(data_path).stem.lower()
    if stem.startswith("real_"):
        stem = stem.removeprefix("real_")
    return INITIAL_DATASET_WORKERS.get(stem)


def apply_initial_worker_mapping(
    config: Any,
    data_path: str | Path,
    *,
    explicit_fields: set[str] | None = None,
) -> int | None:
    """将旧版真实数据集工人数应用到配置；显式 CLI 覆盖时保留用户值。"""
    worker_count = resolve_initial_worker_count(data_path)
    if worker_count is None:
        return None
    explicit = explicit_fields or set()
    if "n_w" in explicit or "n_w_min" in explicit or "n_w_max" in explicit:
        return worker_count
    config.n_w = worker_count
    config.n_w_min = worker_count
    config.n_w_max = max(int(getattr(config, "n_w_max", worker_count)), worker_count)
    return worker_count
