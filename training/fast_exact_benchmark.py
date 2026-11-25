# -*- coding: utf-8 -*-
"""Fast-Exact 性能基准的指标计算工具。

为 WorkerPointer v2 Fast-Exact 验收门提供可测试的指标汇总逻辑：
- GPU 利用率采样统计（均值 / P50 / P90）；
- PPO replay samples/s 与 update 耗时分解；
- GPU 模板命中率。
"""

from __future__ import annotations

import time
from typing import Callable, Mapping, Protocol, Sequence, TypeVar

import numpy as np


_T = TypeVar("_T")


class UtilizationSampler(Protocol):
    """性能测量所需的最小 GPU 利用率采样接口。"""

    def start(self) -> None: ...

    def stop(self) -> list[float]: ...


def summarize_utilization(
    samples: Sequence[float],
) -> dict[str, float | int | bool | None]:
    """汇总 GPU 利用率采样；空采样显式标记为不可用。"""
    values = [float(value) for value in samples]
    if not values:
        return {
            "available": False,
            "sample_count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
        }
    array = np.asarray(values, dtype=float)
    return {
        "available": True,
        "sample_count": len(values),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
    }


def summarize_group_sizes(samples: Sequence[int]) -> dict[str, float]:
    """统计 Fast-Exact physical group 的均值、P50 和 P95。"""
    values = [int(value) for value in samples]
    if not values:
        raise ValueError("group_sizes 不能为空")
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }


def measure_operation(
    operation: Callable[[], _T],
    *,
    sampler: UtilizationSampler,
    synchronize: Callable[[], None],
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[_T, float, list[float]]:
    """在 GPU 同步边界内同时测量操作耗时与利用率。"""
    synchronize()
    sampler.start()
    started = clock()
    try:
        result = operation()
        synchronize()
        elapsed = clock() - started
    finally:
        samples = sampler.stop()
    return result, float(elapsed), list(samples)


def resolve_benchmark_num_envs(
    mode: str,
    *,
    platform_name: str,
    override: int | None,
) -> int:
    """解析三组基准环境数：历史基线固定 4，宽组在 Linux 使用 16。"""
    if override is not None:
        if int(override) < 1:
            raise ValueError("benchmark num_envs override 必须大于 0")
        return int(override)
    if mode == "v2_legacy":
        return 4
    if mode in {"v2_cpu_wide", "v2_fast_exact"}:
        return 16 if str(platform_name) == "Linux" else 4
    raise ValueError(f"未知 benchmark mode: {mode!r}")


def compute_replay_performance(
    update_metrics: Mapping[str, float],
    sample_count: int,
    *,
    k_epochs: int = 1,
) -> dict[str, float]:
    """从单次 PPO update 指标计算 replay 性能契约。

    update_metrics: ``_run_v2_fast_exact_replay_update`` 返回的指标字典
    （键名与 ppo_agent.py 保持一致）。
    sample_count: 该 update 实际重放的样本数（logical batch 实际样本总量）。
    """
    unique_samples = max(0, int(sample_count))
    epochs = max(1, int(k_epochs))
    effective_samples = unique_samples * epochs
    seconds = float(update_metrics.get("V2/FastExact/ReplayUpdateSeconds", 0.0) or 0.0)
    replay_samples_per_sec = effective_samples / seconds if seconds > 0.0 else 0.0
    return {
        "unique_samples": float(unique_samples),
        "effective_replay_samples": float(effective_samples),
        "replay_samples_per_sec": replay_samples_per_sec,
        "update_seconds": seconds,
        "physical_groups": float(
            update_metrics.get("V2/FastExact/PhysicalGroupCount", 0.0) or 0.0
        ),
        "first_contract_total_max_ae": float(
            update_metrics.get("V2/FirstContractTotalMaxAE", 0.0) or 0.0
        ),
        "first_contract_total_mae": float(
            update_metrics.get("V2/FirstContractTotalMAE", 0.0) or 0.0
        ),
    }


def compute_template_hit_rate(hits: int, misses: int) -> float:
    """GPU 模板命中率；无构建请求视为 1.0（全命中语义）。"""
    total = int(hits) + int(misses)
    if total <= 0:
        return 1.0
    return float(int(hits)) / float(total)


__all__ = [
    "UtilizationSampler",
    "compute_replay_performance",
    "compute_template_hit_rate",
    "measure_operation",
    "resolve_benchmark_num_envs",
    "summarize_group_sizes",
    "summarize_utilization",
]
