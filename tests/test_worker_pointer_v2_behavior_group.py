# -*- coding: utf-8 -*-
"""WorkerPointer v2 行为组与轻量轨迹合同。

覆盖：
- WorkerPointerV2BehaviorTrace 字段与虚拟任务语义（station/team log-prob 为零）；
- Memory 与动作一一对应存储行为轨迹；
- 多环境/多 episode 合并后，按 group_id 与 position 完整恢复原组（不依赖合并顺序）；
- group 缺失/重复/成员数不符时 fail-closed。
"""

from __future__ import annotations

import pytest


def _make_trace(
    *,
    group_id: tuple[int, int],
    group_size: int,
    group_position: int,
    env_index: int,
    task_lp: float,
    station_lp: float = 0.0,
    team_lp: float = 0.0,
):
    from training.worker_pointer_v2_behavior import WorkerPointerV2BehaviorTrace

    return WorkerPointerV2BehaviorTrace(
        group_id=group_id,
        group_size=group_size,
        group_position=group_position,
        env_index=env_index,
        task_lp=task_lp,
        station_lp=station_lp,
        team_lp=team_lp,
    )


def test_behavior_trace_fields_and_virtual_task_semantics() -> None:
    from training.worker_pointer_v2_behavior import WorkerPointerV2BehaviorTrace

    trace = WorkerPointerV2BehaviorTrace(
        group_id=(0, 3),
        group_size=2,
        group_position=0,
        env_index=1,
        task_lp=-0.5,
        station_lp=0.0,
        team_lp=0.0,
    )
    assert trace.group_id == (0, 3)
    assert trace.group_size == 2
    assert trace.group_position == 0
    assert trace.env_index == 1
    assert trace.task_lp == pytest.approx(-0.5)
    # 虚拟任务的 station/team 分量固定为零。
    assert trace.station_lp == 0.0
    assert trace.team_lp == 0.0


def test_memory_stores_behavior_traces_aligned_with_actions() -> None:
    from training.memory import Memory

    memory = Memory()
    assert memory.worker_pointer_v2_behavior_traces == []
    memory.worker_pointer_v2_behavior_traces.append(
        _make_trace(group_id=(0, 1), group_size=1, group_position=0, env_index=0, task_lp=-0.1)
    )
    memory.actions.append((1, 2, [3]))
    assert len(memory.worker_pointer_v2_behavior_traces) == 1
    assert memory.worker_pointer_v2_behavior_traces[0].group_id == (0, 1)

    memory.clear()
    assert memory.worker_pointer_v2_behavior_traces == []


def test_restore_groups_recovers_members_and_order_after_merge_shuffle() -> None:
    from training.worker_pointer_v2_behavior import restore_behavior_groups

    # 两个环境、两个 episode、交错分组。
    traces = [
        _make_trace(group_id=(0, 1), group_size=2, group_position=0, env_index=0, task_lp=-0.1),
        _make_trace(group_id=(0, 1), group_size=2, group_position=1, env_index=1, task_lp=-0.2),
        _make_trace(group_id=(0, 2), group_size=2, group_position=0, env_index=0, task_lp=-0.3),
        _make_trace(group_id=(0, 2), group_size=2, group_position=1, env_index=1, task_lp=-0.4),
        _make_trace(group_id=(1, 1), group_size=1, group_position=0, env_index=0, task_lp=-0.5),
    ]
    # 模拟多环境 Memory 合并后被打乱的样本顺序。
    import random

    rng = random.Random(7)
    shuffled = traces[:]
    rng.shuffle(shuffled)
    restored = restore_behavior_groups(shuffled)

    assert len(restored) == 3
    assert [group[0].group_id for group in restored] == [(0, 1), (0, 2), (1, 1)]
    # 组内成员按原始 position 顺序恢复，不依赖合并后的样本顺序。
    assert [t.group_position for t in restored[0]] == [0, 1]
    assert [t.env_index for t in restored[0]] == [0, 1]
    assert [t.task_lp for t in restored[0]] == pytest.approx([-0.1, -0.2])
    assert len(restored[2]) == 1


def test_restore_groups_rejects_missing_member() -> None:
    from training.worker_pointer_v2_behavior import restore_behavior_groups

    traces = [
        _make_trace(group_id=(0, 1), group_size=2, group_position=0, env_index=0, task_lp=-0.1),
        # 缺少 position=1 的成员。
    ]
    with pytest.raises(ValueError, match="group"):
        restore_behavior_groups(traces)


def test_restore_groups_rejects_duplicate_position() -> None:
    from training.worker_pointer_v2_behavior import restore_behavior_groups

    traces = [
        _make_trace(group_id=(0, 1), group_size=2, group_position=0, env_index=0, task_lp=-0.1),
        _make_trace(group_id=(0, 1), group_size=2, group_position=0, env_index=1, task_lp=-0.2),
    ]
    with pytest.raises(ValueError, match="group"):
        restore_behavior_groups(traces)


def test_make_behavior_traces_assigns_group_size_and_position() -> None:
    from training.worker_pointer_v2_behavior import make_behavior_traces

    traces = make_behavior_traces(
        group_id=(3, 5),
        env_indices=[0, 1, 2, 3],
        behavior_logprobs=[(-0.1, -0.2, -0.3), (-0.4, -0.5, -0.6), None, (-0.7, -0.8, -0.9)],
    )
    assert len(traces) == 3
    assert all(trace.group_id == (3, 5) for trace in traces)
    assert all(trace.group_size == 3 for trace in traces)
    assert [t.group_position for t in traces] == [0, 1, 2]
    assert [t.env_index for t in traces] == [0, 1, 3]
    assert [t.task_lp for t in traces] == pytest.approx([-0.1, -0.4, -0.7])
    # 非法动作（None）不产生轨迹，且不造成 position 空洞。


def test_make_behavior_traces_empty() -> None:
    from training.worker_pointer_v2_behavior import make_behavior_traces

    assert (
        make_behavior_traces(
            group_id=(0, 0), env_indices=[0, 1], behavior_logprobs=[None, None]
        )
        == []
    )


def test_restore_groups_rejects_size_mismatch() -> None:
    from training.worker_pointer_v2_behavior import restore_behavior_groups

    traces = [
        _make_trace(group_id=(0, 1), group_size=2, group_position=0, env_index=0, task_lp=-0.1),
        _make_trace(group_id=(0, 1), group_size=3, group_position=1, env_index=1, task_lp=-0.2),
    ]
    with pytest.raises(ValueError, match="group_size"):
        restore_behavior_groups(traces)


def test_merge_memories_preserves_behavior_traces() -> None:
    """防止按环境合并轨迹时遗漏 v2 group 元数据。"""
    from training.memory import Memory
    from training.rollout_service import APALRolloutService

    left = Memory()
    right = Memory()
    target = Memory()
    left.worker_pointer_v2_behavior_traces.append(
        _make_trace(
            group_id=(0, 1),
            group_size=2,
            group_position=0,
            env_index=0,
            task_lp=-0.1,
        )
    )
    right.worker_pointer_v2_behavior_traces.append(
        _make_trace(
            group_id=(0, 1),
            group_size=2,
            group_position=1,
            env_index=1,
            task_lp=-0.2,
        )
    )

    APALRolloutService._merge_memories(target, [left, right])

    assert [trace.env_index for trace in target.worker_pointer_v2_behavior_traces] == [0, 1]


def test_restore_groups_rejects_duplicate_environment_member() -> None:
    """同一 rollout group 中同一环境只能出现一次。"""
    from training.worker_pointer_v2_behavior import restore_behavior_groups

    traces = [
        _make_trace(
            group_id=(0, 1),
            group_size=2,
            group_position=0,
            env_index=0,
            task_lp=-0.1,
        ),
        _make_trace(
            group_id=(0, 1),
            group_size=2,
            group_position=1,
            env_index=0,
            task_lp=-0.2,
        ),
    ]
    with pytest.raises(ValueError, match="env"):
        restore_behavior_groups(traces)
