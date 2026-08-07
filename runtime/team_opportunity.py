"""APAL 团队候选的一步反事实执行工具。"""

from __future__ import annotations

import copy
import gc
from collections.abc import Callable, Iterable
from typing import Any


CandidateAction = tuple[int, int, tuple[int, ...]]
MetricExtractor = Callable[[float, bool, dict[str, Any]], dict[str, float]]


def evaluate_one_step_candidate(
    env: Any,
    *,
    action: CandidateAction,
    metric_extractor: MetricExtractor,
    shared_attribute_names: Iterable[str] = (),
) -> dict[str, float]:
    """在隔离环境副本上执行一个 APAL 团队动作并提取可比较指标。"""
    memo = {
        id(getattr(env, name)): getattr(env, name)
        for name in shared_attribute_names
        if hasattr(env, name)
    }
    clone = copy.deepcopy(env, memo=memo)
    try:
        clone.skip_obs_building = True
        task_id, station_id, team = action
        _obs, reward, done, info = clone.step(
            (int(task_id), int(station_id), [int(worker_id) for worker_id in team])
        )
        if "error" in info:
            raise RuntimeError(f"候选团队被 APAL 环境拒绝：{info['error']}")
        result = metric_extractor(float(reward), bool(done), info)
        if not all(isinstance(value, float) for value in result.values()):
            raise TypeError("候选审计指标必须全部为 float")
        return result
    finally:
        del clone
        gc.collect()


__all__ = ["CandidateAction", "MetricExtractor", "evaluate_one_step_candidate"]
