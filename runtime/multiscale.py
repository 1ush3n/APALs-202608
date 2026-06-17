from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ScaleProfile:
    name: str
    max_ops: int
    batch_size: int
    k_epochs: int


@dataclass(frozen=True)
class DatasetCandidate:
    dataset_idx: int
    file_path: str
    num_tasks: int
    profile: ScaleProfile
    sampling_weight: float
    scheduled_updates: int


@dataclass(frozen=True)
class BenchmarkScore:
    benchmark_name: str
    data_path: str
    makespan: float
    reference_makespan: float
    normalized_score: float
    complete: bool
    invalid_step_count: int
    inference_time: float


@dataclass(frozen=True)
class MultiBenchmarkResult:
    eligible: bool
    composite_score: float
    selection_score: float
    rows: tuple[BenchmarkScore, ...]


def scale_profile_for_task_count(num_tasks: int) -> ScaleProfile:
    """根据工序规模选择上界 profile。"""
    n = int(num_tasks)
    if n <= 283:
        return ScaleProfile(name="283", max_ops=283, batch_size=256, k_epochs=1)
    if n <= 1000:
        return ScaleProfile(name="680", max_ops=1000, batch_size=128, k_epochs=2)
    if n <= 2338:
        return ScaleProfile(name="2338", max_ops=2338, batch_size=64, k_epochs=3)
    return ScaleProfile(name="3182", max_ops=3182, batch_size=32, k_epochs=4)


def scheduled_updates_for_task_count(
    num_tasks: int,
    *,
    min_ops: int = 200,
    max_ops: int = 3100,
    min_updates: int = 600,
    max_updates: int = 3300,
) -> int:
    """次线性更新预算，用于记录和后续扩展采样调度。"""
    if max_ops <= min_ops:
        raise ValueError("max_ops 必须大于 min_ops")
    ratio = (int(num_tasks) - int(min_ops)) / float(int(max_ops) - int(min_ops))
    ratio = min(1.0, max(0.0, ratio))
    return int(round(int(min_updates) + (int(max_updates) - int(min_updates)) * math.sqrt(ratio)))


def inverse_scale_weight(num_tasks: int, exponent: float = 0.5) -> float:
    """反规模采样权重；默认 1/sqrt(n)。"""
    n = max(1, int(num_tasks))
    exp = max(0.0, float(exponent))
    return float(1.0 / (n ** exp))


def build_dataset_candidates(
    dataset_pool: list[dict[str, Any]],
    *,
    min_ops: int,
    max_ops: int,
    sampling_exponent: float,
    min_updates: int,
    max_updates: int,
) -> tuple[DatasetCandidate, ...]:
    candidates: list[DatasetCandidate] = []
    for idx, descriptor in enumerate(dataset_pool):
        num_tasks = descriptor.get("num_tasks")
        if num_tasks is None:
            from data_loader import load_data

            raw = load_data(Path(descriptor["file_path"]))
            num_tasks = int(raw["num_tasks"])
            descriptor["num_tasks"] = num_tasks
        n = int(num_tasks)
        if n < int(min_ops) or n > int(max_ops):
            continue
        profile = scale_profile_for_task_count(n)
        candidates.append(
            DatasetCandidate(
                dataset_idx=int(idx),
                file_path=str(descriptor["file_path"]),
                num_tasks=n,
                profile=profile,
                sampling_weight=inverse_scale_weight(n, sampling_exponent),
                scheduled_updates=scheduled_updates_for_task_count(
                    n,
                    min_ops=int(min_ops),
                    max_ops=int(max_ops),
                    min_updates=int(min_updates),
                    max_updates=int(max_updates),
                ),
            )
        )
    if not candidates:
        raise ValueError(f"未找到规模位于 [{min_ops}, {max_ops}] 的训练数据集")
    return tuple(candidates)


class InverseScaleDatasetSampler:
    """按反规模权重从候选 APAL 实例中抽样。"""

    def __init__(self, candidates: tuple[DatasetCandidate, ...], seed: int) -> None:
        self.candidates = candidates
        weights = np.asarray([item.sampling_weight for item in candidates], dtype=np.float64)
        if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
            raise ValueError("多规模采样权重无效")
        self.probabilities = weights / weights.sum()
        self.rng = np.random.RandomState(int(seed))

    def sample(self) -> DatasetCandidate:
        idx = int(self.rng.choice(len(self.candidates), p=self.probabilities))
        return self.candidates[idx]


def apply_scale_profile_to_agent(agent: Any, profile: ScaleProfile, config_obj: Any) -> None:
    """把当前规模 profile 应用到 PPO 更新参数。"""
    agent.k_epochs = int(profile.k_epochs)
    requested_batch = int(profile.batch_size)
    cap = max(0, int(getattr(config_obj, "ppo_batch_size_cap", 0)))
    agent.batch_size = min(requested_batch, cap) if cap > 0 else requested_batch


def parse_reference_makespans(raw: Any) -> dict[str, float]:
    if isinstance(raw, dict):
        refs = {str(k): float(v) for k, v in raw.items()}
    elif isinstance(raw, str):
        import json

        refs = {str(k): float(v) for k, v in json.loads(raw).items()}
    else:
        refs = {}
    if not refs:
        raise ValueError("multi_benchmark_reference_makespans 不能为空")
    for key, value in refs.items():
        if value <= 0:
            raise ValueError(f"参考 makespan 必须大于 0: {key}={value}")
    return refs


def score_multi_benchmark(rows: list[BenchmarkScore]) -> MultiBenchmarkResult:
    if not rows:
        raise ValueError("多基准评分至少需要一个评估结果")
    eligible = all(row.complete and row.invalid_step_count == 0 for row in rows)
    composite = float(np.mean([row.normalized_score for row in rows]))
    selection = composite if eligible else float("inf")
    return MultiBenchmarkResult(
        eligible=bool(eligible),
        composite_score=composite,
        selection_score=selection,
        rows=tuple(rows),
    )
