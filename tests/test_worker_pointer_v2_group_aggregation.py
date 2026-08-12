# -*- coding: utf-8 -*-
"""WorkerPointer v2 逻辑 batch 聚合与首次同形合同（纯函数层）。

要求：
- group 加权累积与逻辑 batch 内样本均值数值等价；
- 首次重算合同按 task/station/team/total 四分量校验 MAE/MaxAE，超阈值 fail-closed；
- 空输入与阈值非正时 fail-closed。
"""

from __future__ import annotations

import pytest


def _aggregate(sample_losses, group_sizes, logical_batch_actual_size):
    from training.worker_pointer_v2_replay import aggregate_group_batch_losses

    return aggregate_group_batch_losses(
        sample_losses=list(sample_losses),
        group_sizes=list(group_sizes),
        logical_batch_actual_size=logical_batch_actual_size,
    )


def test_group_weighted_aggregation_equals_batch_sample_mean() -> None:
    losses = [1.0, 2.0, 3.0, 4.0, 5.0]
    # 两个 group：size 2 与 3。
    group_sizes = [2, 2, 3, 3, 3]
    batch_size = 5
    result = _aggregate(losses, group_sizes, batch_size)
    expected = sum(losses) / batch_size
    assert result == pytest.approx(expected)


def test_group_weighted_aggregation_uneven_groups() -> None:
    losses = [0.5, 1.5, 3.0, 4.0]
    # 每样本所属 group 大小：样本 0 单组；样本 1/2 同组(size 2)；样本 3 属 size 4 组。
    group_sizes = [1, 2, 2, 4]
    result = _aggregate(losses, group_sizes, logical_batch_actual_size=7)
    assert result == pytest.approx(sum(losses) / 7)


def test_aggregation_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="长度"):
        _aggregate([1.0, 2.0], [1], logical_batch_actual_size=2)


def test_aggregation_rejects_nonpositive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch"):
        _aggregate([1.0], [1], logical_batch_actual_size=0)


def _check_contract(behavior, replayed, *, max_abs_error):
    from training.worker_pointer_v2_replay import check_first_recompute_contract

    return check_first_recompute_contract(
        behavior=list(behavior),
        replayed=list(replayed),
        max_abs_error=max_abs_error,
    )


def test_contract_computes_component_mae_and_maxae() -> None:
    behavior = [(0.0, 0.0, 0.0), (0.1, 0.2, 0.3)]
    replayed = [(0.001, -0.001, 0.0), (0.101, 0.2005, 0.3)]
    report = _check_contract(behavior, replayed, max_abs_error=0.01)
    assert report["task"]["mae"] == pytest.approx(0.001)
    assert report["task"]["max_abs_error"] == pytest.approx(0.001)
    assert report["station"]["max_abs_error"] == pytest.approx(0.001)
    assert report["team"]["max_abs_error"] == pytest.approx(0.0)
    assert report["total"]["max_abs_error"] == pytest.approx(0.0015)


def test_contract_fails_closed_on_any_component_over_threshold() -> None:
    behavior = [(0.0, 0.0, 0.0)]
    replayed = [(0.0, 0.002, 0.0)]  # station 超 1e-3
    with pytest.raises(ValueError, match="station"):
        _check_contract(behavior, replayed, max_abs_error=1.0e-3)


def test_contract_passes_within_threshold() -> None:
    behavior = [(0.0, 0.0, 0.0), (-1.5, 0.25, -0.75)]
    replayed = [(0.0004, -0.0002, 0.0001), (-1.5008, 0.25, -0.7495)]
    report = _check_contract(behavior, replayed, max_abs_error=1.0e-3)
    assert report["passed"] is True


def test_contract_rejects_mismatched_lengths_and_bad_threshold() -> None:
    with pytest.raises(ValueError, match="长度"):
        _check_contract([(0.0, 0.0, 0.0)], [], max_abs_error=1.0e-3)
    with pytest.raises(ValueError, match="max_abs_error"):
        _check_contract([(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)], max_abs_error=0.0)


def test_window_loss_normalization_matches_all_sample_mean_gradient() -> None:
    """多个物理组分次 backward 后，梯度应等于窗口全部样本的均值梯度。"""
    import torch

    from training.worker_pointer_v2_replay import normalize_group_loss_sum

    weight = torch.tensor(2.0, requires_grad=True)
    # 两个物理组分别含 2、1 个样本；损失和为 (1+2)w 与 3w。
    group_loss_sums = [3.0 * weight, 3.0 * weight]
    for group_loss_sum in group_loss_sums:
        normalize_group_loss_sum(
            group_loss_sum, window_sample_count=3
        ).backward()

    # 全部样本均值损失 ((1+2+3)/3)w 的梯度为 2。
    assert weight.grad is not None
    assert float(weight.grad) == pytest.approx(2.0)


def test_window_loss_normalization_rejects_empty_window() -> None:
    import torch

    from training.worker_pointer_v2_replay import normalize_group_loss_sum

    with pytest.raises(ValueError, match="window_sample_count"):
        normalize_group_loss_sum(torch.tensor(1.0), window_sample_count=0)
