# -*- coding: utf-8 -*-
"""WorkerPointer 团队选择模式判断。

独立小模块：供 runtime.configuration 与 runtime.checkpoints 等共同导入，
避免 configuration -> artifacts -> checkpoints -> configuration 的循环导入。
"""

from __future__ import annotations

V2_TEAM_SELECTION_MODES = frozenset(
    {"autoregressive_pressure_v2", "autoregressive_pressure_v2_fast_exact"}
)
FAST_EXACT_TEAM_SELECTION_MODE = "autoregressive_pressure_v2_fast_exact"


def is_worker_pointer_v2_mode(config: object) -> bool:
    """历史 v2 与 Fast-Exact 均属 v2 语义族（rollout 行为组 + 同形重放）。"""
    mode = str(getattr(config, "team_selection_mode", "")).strip().lower()
    return mode in V2_TEAM_SELECTION_MODES


def is_fast_exact_mode(config: object) -> bool:
    """仅 Fast-Exact 专属路径使用 GPU 常驻模板并启用严格失败语义。"""
    return (
        str(getattr(config, "team_selection_mode", "")).strip().lower()
        == FAST_EXACT_TEAM_SELECTION_MODE
    )


__all__ = [
    "V2_TEAM_SELECTION_MODES",
    "FAST_EXACT_TEAM_SELECTION_MODE",
    "is_worker_pointer_v2_mode",
    "is_fast_exact_mode",
]
