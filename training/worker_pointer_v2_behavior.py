# -*- coding: utf-8 -*-
"""WorkerPointer v2 行为组与轻量轨迹。

行为组：每次 ``select_actions_batch`` 调用（一个 rollout 决策轮）对应的原始环境组。
轻量轨迹：为每个动作记录组归属与三部分行为 log-prob（task/station/team），
使 PPO 重放能按 rollout 原始分组同形重建，规避 bf16 批形状漂移。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class WorkerPointerV2BehaviorTrace:
    """一个动作的行为轨迹记录；虚拟任务的 station/team 分量固定为零。"""

    group_id: tuple[int, int]  # (episode, rollout_call_index)
    group_size: int
    group_position: int
    env_index: int
    task_lp: float
    station_lp: float = 0.0
    team_lp: float = 0.0

    @property
    def total_lp(self) -> float:
        return float(self.task_lp) + float(self.station_lp) + float(self.team_lp)


def make_behavior_traces(
    *,
    group_id: tuple[int, int],
    env_indices: list[int],
    behavior_logprobs: list[tuple[float, float, float] | None],
) -> list[WorkerPointerV2BehaviorTrace]:
    """为一次 select_actions_batch 调用创建行为轨迹。

    behavior_logprobs 与 env_indices 等长；None 表示无效动作（不产生轨迹）。
    group_size 为有效轨迹数，position 按有效成员顺序重新编号（无空洞）。
    """
    if len(env_indices) != len(behavior_logprobs):
        raise ValueError("env_indices 与 behavior_logprobs 长度不一致")
    valid = [
        (int(env_index), (float(task), float(station), float(team)))
        for env_index, logprob in zip(env_indices, behavior_logprobs)
        if logprob is not None
        for (task, station, team) in [tuple(logprob)]
    ]
    size = len(valid)
    return [
        WorkerPointerV2BehaviorTrace(
            group_id=group_id,
            group_size=size,
            group_position=position,
            env_index=env_index,
            task_lp=task_lp,
            station_lp=station_lp,
            team_lp=team_lp,
        )
        for position, (env_index, (task_lp, station_lp, team_lp)) in enumerate(valid)
    ]


def restore_behavior_groups(
    traces: Iterable[WorkerPointerV2BehaviorTrace],
) -> list[list[WorkerPointerV2BehaviorTrace]]:
    """按 group_id 恢复原始组，组内按 group_position 排序。

    不依赖合并后的样本顺序；缺失成员、重复 position 或成员数与
    group_size 不符时 fail-closed。
    """
    groups: dict[tuple[int, int], list[WorkerPointerV2BehaviorTrace]] = {}
    for trace in traces:
        groups.setdefault(trace.group_id, []).append(trace)

    restored: list[list[WorkerPointerV2BehaviorTrace]] = []
    for group_id in sorted(groups):
        members = groups[group_id]
        seen: set[int] = set()
        seen_envs: set[int] = set()
        for member in members:
            if member.group_position in seen:
                raise ValueError(
                    f"group {group_id!r} 存在重复 position: {member.group_position}"
                )
            seen.add(member.group_position)
            if member.env_index in seen_envs:
                raise ValueError(
                    f"group {group_id!r} 存在重复 env_index: {member.env_index}"
                )
            seen_envs.add(member.env_index)
        expected_sizes = {int(member.group_size) for member in members}
        if len(expected_sizes) != 1:
            raise ValueError(
                f"group {group_id!r} group_size 不一致: {sorted(expected_sizes)}"
            )
        expected = next(iter(expected_sizes))
        if len(members) != expected:
            raise ValueError(
                f"group {group_id!r} group_size 不符: "
                f"expected={expected} actual={len(members)}"
            )
        if seen != set(range(expected)):
            raise ValueError(f"group {group_id!r} position 缺失: {sorted(seen)}")
        restored.append(sorted(members, key=lambda item: item.group_position))
    return restored


__all__ = [
    "WorkerPointerV2BehaviorTrace",
    "make_behavior_traces",
    "restore_behavior_groups",
]
