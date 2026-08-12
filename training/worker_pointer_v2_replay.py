# -*- coding: utf-8 -*-
"""WorkerPointer v2 的 group-aware PPO 重放规划器。

确定性地将完整行为组装入逻辑 batch：
- group 不拆分、不重复、不丢弃，组内顺序由调用方保持；
- 逻辑 batch 顺序由 seed/current_step/epoch 重建；
- 下一组放不下时关闭当前 batch；尾部不足 cap 直接使用实际样本数。
"""

from __future__ import annotations

import numpy as np
import torch


def _replay_seed(*, seed: int, current_step: int, epoch: int) -> int:
    """由训练语义量派生确定性重放种子。"""
    return int(seed) * 1000003 + int(current_step) * 1009 + int(epoch)


def plan_behavior_groups(
    group_sizes: list[int],
    *,
    logical_cap: int,
    seed: int,
    current_step: int,
    epoch: int,
) -> list[list[int]]:
    """返回逻辑 batch 列表；每个逻辑 batch 为 group 索引列表。

    group_sizes: 每个行为组的样本数，顺序即 group 索引。
    logical_cap: 每个逻辑 batch 的样本上限。
    """
    cap = int(logical_cap)
    if cap < 1:
        raise ValueError(f"logical_cap 必须大于 0: {cap}")
    sizes = [int(size) for size in group_sizes]
    if any(size < 1 for size in sizes):
        raise ValueError("group size 必须为正")
    if any(size > cap for size in sizes):
        raise ValueError(
            f"group size 超过 logical_cap: cap={cap}, sizes={sorted(sizes)}"
        )

    rng = np.random.default_rng(_replay_seed(seed=seed, current_step=current_step, epoch=epoch))
    order = rng.permutation(len(sizes)).tolist()

    batches: list[list[int]] = []
    current: list[int] = []
    current_samples = 0
    for group_index in order:
        size = sizes[group_index]
        if current and current_samples + size > cap:
            batches.append(current)
            current = []
            current_samples = 0
        current.append(group_index)
        current_samples += size
    if current:
        batches.append(current)
    return batches


def select_validation_groups(
    group_team_sizes: list[list[int]],
    *,
    logical_cap: int,
) -> list[int]:
    """为首次同形合同选择覆盖面优先的完整行为组。

    先最大化尚未覆盖的团队人数类别；含虚拟任务（团队人数 0）的组在
    首轮同分时优先。选中后继续用剩余容量填充完整组，不拆分或补样本。
    """
    cap = int(logical_cap)
    if cap < 1:
        raise ValueError(f"logical_cap 必须大于 0: {cap}")
    normalized = [[int(size) for size in group] for group in group_team_sizes]
    if any(not group for group in normalized):
        raise ValueError("验证行为组不能为空")
    if any(len(group) > cap for group in normalized):
        raise ValueError("验证行为组大小超过 logical_cap")

    selected: list[int] = []
    covered: set[int] = set()
    used_samples = 0
    remaining = set(range(len(normalized)))
    while remaining:
        candidates = [
            index
            for index in remaining
            if used_samples + len(normalized[index]) <= cap
        ]
        if not candidates:
            break
        best = max(
            candidates,
            key=lambda index: (
                len(set(normalized[index]) - covered),
                int(0 in normalized[index] and 0 not in covered),
                -index,
            ),
        )
        selected.append(best)
        used_samples += len(normalized[best])
        covered.update(normalized[best])
        remaining.remove(best)
    return selected


def aggregate_group_batch_losses(
    sample_losses: list[float],
    group_sizes: list[int],
    *,
    logical_batch_actual_size: int,
) -> float:
    """按逻辑 batch 聚合样本损失。

    方案语义：group loss 权重为 ``group_size / logical_batch_actual_size``，
    即 ``Σ_group (mean_group × size_group / B)``；该式与
    ``Σ_sample loss_s / B``（逻辑 batch 内样本均值）数值等价。此处显式
    校验输入后采用等价形式，避免依赖同尺寸 group 的不可辨识分组。
    """
    if len(sample_losses) != len(group_sizes):
        raise ValueError(f"sample_losses 与 group_sizes 长度不一致")
    if len(sample_losses) == 0:
        raise ValueError("逻辑 batch 不允许为空")
    if int(logical_batch_actual_size) <= 0:
        raise ValueError(f"logical_batch_actual_size 必须为正: {logical_batch_actual_size}")
    for size in group_sizes:
        if int(size) < 1:
            raise ValueError(f"group size 必须为正: {size}")
    return float(sum(sample_losses)) / float(logical_batch_actual_size)


def normalize_group_loss_sum(
    group_loss_sum: torch.Tensor,
    *,
    window_sample_count: int,
) -> torch.Tensor:
    """将物理组损失和按当前累积窗口的实际样本总数归一化。

    各物理组分别对返回值执行 backward，与对窗口全部样本损失取均值后
    执行一次 backward 数值等价；尾部不足完整累积窗口时不会被低估。
    """
    sample_count = int(window_sample_count)
    if sample_count < 1:
        raise ValueError(
            f"window_sample_count 必须为正: {window_sample_count}"
        )
    if group_loss_sum.ndim != 0:
        raise ValueError(
            f"group_loss_sum 必须是标量张量: shape={tuple(group_loss_sum.shape)}"
        )
    return group_loss_sum / float(sample_count)


_COMPONENT_INDEXES: dict[str, int] = {"task": 0, "station": 1, "team": 2}


def check_first_recompute_contract(
    behavior: list[tuple[float, float, float]],
    replayed: list[tuple[float, float, float]],
    *,
    max_abs_error: float,
) -> dict[str, object]:
    """首次同形重放合同：按 task/station/team/total 分量校验 MAE/MaxAE。

    任一分量 MaxAE 超过阈值时立即抛错（fail-closed）。
    """
    if len(behavior) != len(replayed):
        raise ValueError(f"behavior 与 replayed 长度不一致")
    if len(behavior) == 0:
        raise ValueError("同形重放验证样本不允许为空")
    if not float(max_abs_error) > 0.0:
        raise ValueError(f"max_abs_error 必须为正: {max_abs_error}")

    threshold = float(max_abs_error)
    report: dict[str, object] = {"passed": True}
    totals_b = [sum(row) for row in behavior]
    totals_r = [sum(row) for row in replayed]
    # 先逐分量校验，最后校验 total，保证超阈值时报告具体分量。
    entries: list[tuple[str, list[float], list[float]]] = []
    for name, index in _COMPONENT_INDEXES.items():
        entries.append((name, [row[index] for row in behavior], [row[index] for row in replayed]))
    entries.append(("total", totals_b, totals_r))

    for name, baseline, actual in entries:
        diffs = [abs(float(a) - float(b)) for a, b in zip(actual, baseline)]
        mae = float(sum(diffs)) / len(diffs)
        max_abs = max(diffs)
        report[name] = {"mae": mae, "max_abs_error": max_abs}
        if max_abs > threshold:
            report["passed"] = False
            raise ValueError(
                f"WorkerPointer v2 首次同形重算 {name} 分量超阈值: "
                f"max_abs_error={max_abs:.6g} threshold={threshold:.6g}"
            )
    return report


__all__ = [
    "plan_behavior_groups",
    "select_validation_groups",
    "aggregate_group_batch_losses",
    "normalize_group_loss_sum",
    "check_first_recompute_contract",
]
