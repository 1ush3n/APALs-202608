from __future__ import annotations

from typing import Protocol


class BatchSemanticsConfig(Protocol):
    """PPO 批量语义所需的最小配置接口。"""

    team_selection_mode: str
    ppo_batch_size_cap: int
    worker_pointer_v2_logical_batch_cap: int


def resolve_effective_ppo_batch_size(
    requested_batch_size: int,
    config: BatchSemanticsConfig,
) -> int:
    """解析实际 PPO batch；v2 的同形重放不受平台大图 batch 限幅。"""
    requested = max(1, int(requested_batch_size))
    if str(config.team_selection_mode) == "autoregressive_pressure_v2":
        return requested
    platform_cap = max(0, int(config.ppo_batch_size_cap))
    return min(requested, platform_cap) if platform_cap > 0 else requested


def resolve_v2_logical_batch_size(
    effective_batch_size: int,
    config: BatchSemanticsConfig,
) -> int:
    """v2 最终 batch 是逻辑重放 batch，不再受旧 logical cap 二次截断。"""
    effective = max(1, int(effective_batch_size))
    if str(config.team_selection_mode) != "autoregressive_pressure_v2":
        return effective
    return effective


__all__ = [
    "BatchSemanticsConfig",
    "resolve_effective_ppo_batch_size",
    "resolve_v2_logical_batch_size",
]
