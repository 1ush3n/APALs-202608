# -*- coding: utf-8 -*-
"""WorkerPointer v2 group-aware replay planner 合同。

要求：
- group 不拆分、不重复、不丢弃，组内顺序由调用方保持；
- 逻辑 batch 顺序由 seed/current_step/epoch 确定性重建；
- 下一组放不下时关闭当前 batch；末尾不足 cap 直接使用实际样本数，不补零权重；
- group size 超过 cap 时 fail-closed。
"""

from __future__ import annotations

import pytest


def _plan(group_sizes, *, logical_cap, seed, current_step, epoch):
    from training.worker_pointer_v2_replay import plan_behavior_groups

    return plan_behavior_groups(
        list(group_sizes),
        logical_cap=logical_cap,
        seed=seed,
        current_step=current_step,
        epoch=epoch,
    )


def test_planner_packs_full_groups_without_splitting() -> None:
    batches = _plan([4] * 16, logical_cap=64, seed=42, current_step=0, epoch=0)
    assert len(batches) == 1  # 16 组 × 4 人 = 64 = 恰好一个逻辑 batch
    batch = batches[0]
    assert len(batch) == 16
    assert len(set(batch)) == 16  # 不拆分/不重复


def test_planner_is_deterministic_and_seed_sensitive() -> None:
    a = _plan([4] * 16, logical_cap=64, seed=42, current_step=0, epoch=0)
    b = _plan([4] * 16, logical_cap=64, seed=42, current_step=0, epoch=0)
    assert a == b
    c = _plan([4] * 16, logical_cap=64, seed=43, current_step=0, epoch=0)
    assert a != c
    d = _plan([4] * 16, logical_cap=64, seed=42, current_step=1, epoch=0)
    assert a != d
    e = _plan([4] * 16, logical_cap=64, seed=42, current_step=0, epoch=1)
    assert a != e


def test_planner_uses_all_groups_exactly_once() -> None:
    group_count = 7
    batches = _plan([4] * group_count, logical_cap=64, seed=7, current_step=0, epoch=0)
    seen = [group for batch in batches for group in batch]
    assert sorted(seen) == list(range(group_count))
    assert len(seen) == group_count


def test_planner_closes_batch_when_next_group_does_not_fit() -> None:
    batches = _plan([3, 3, 3, 3, 3], logical_cap=5, seed=1, current_step=0, epoch=0)
    # 每组 3 人，3+3>5，故每组独占一个逻辑 batch。
    assert len(batches) == 5
    assert all(len(batch) == 1 for batch in batches)


def test_planner_tail_batch_uses_actual_sample_count() -> None:
    batches = _plan([4] * 17, logical_cap=64, seed=42, current_step=0, epoch=0)
    assert len(batches) == 2  # 68 = 64 + 4，共两个逻辑批次
    assert sum(len(batch) for batch in batches) == 17
    # 尾部批次样本数 = 68 - 64 = 4，不补零、不丢弃。
    last = batches[-1]
    assert len(last) == 1
    assert sum(4 for _ in last) == 4


def test_planner_rejects_group_larger_than_cap() -> None:
    with pytest.raises(ValueError, match="cap"):
        _plan([5], logical_cap=4, seed=42, current_step=0, epoch=0)


def test_validation_group_selection_prioritizes_virtual_and_team_size_diversity() -> None:
    from training.worker_pointer_v2_replay import select_validation_groups

    selected = select_validation_groups(
        group_team_sizes=[
            [2, 2, 2, 2],
            [1, 1, 1, 1],
            [0, 3, 3, 3],
            [4, 4, 4, 4],
        ],
        logical_cap=8,
    )

    assert selected[0] == 2  # 先覆盖虚拟任务与 team size=3。
    assert len(selected) == 2
    covered = {
        team_size
        for group_index in selected
        for team_size in [[2, 2, 2, 2], [1, 1, 1, 1], [0, 3, 3, 3], [4, 4, 4, 4]][
            group_index
        ]
    }
    assert 0 in covered
    assert len(covered) >= 3


def test_validation_group_selection_never_splits_or_exceeds_cap() -> None:
    from training.worker_pointer_v2_replay import select_validation_groups

    selected = select_validation_groups(
        group_team_sizes=[[0, 1, 2], [3, 3, 3], [4, 4]],
        logical_cap=5,
    )

    assert len(selected) == len(set(selected))
    assert sum(len([[0, 1, 2], [3, 3, 3], [4, 4]][index]) for index in selected) <= 5
