"""Full Joint 后续诊断套件的可复现任务定义。"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Iterable


@dataclass(frozen=True)
class DiagnosticJob:
    """一个独立训练任务及其 Hydra 覆盖项。"""

    suite: str
    variant: str
    seed: int
    run_name: str
    overrides: tuple[str, ...]


def _validate_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(seed) for seed in seeds)
    if not values:
        raise ValueError("至少需要一个 seed")
    if len(set(values)) != len(values):
        raise ValueError("seed 不能重复")
    if any(seed < 0 for seed in values):
        raise ValueError("seed 不能为负数")
    return values


def _legacy_overrides(*, context_mode: str = "attention") -> tuple[str, ...]:
    return (
        "model.team_selection_mode=autoregressive",
        "model.policy_action_scope=operation_station_worker",
        "model.policy_observation_scope=full",
        f"model.actor_context_mode={context_mode}",
    )


def _v2_overrides() -> tuple[str, ...]:
    return (
        "model.team_selection_mode=autoregressive_pressure_v2",
        "model.policy_action_scope=operation_station_worker",
        "model.policy_observation_scope=full",
        "model.actor_context_mode=attention",
        "model.worker_pointer_v2_dynamic_eft_features=false",
        "model.worker_pointer_v2_behavior_replay=true",
        "model.worker_pointer_v2_replay_mode=behavior_group_exact_v1",
        "model.worker_pointer_v2_logical_batch_cap=32",
        "model.worker_pointer_v2_rollout_group_upper_bound=4",
    )


def build_jobs(seeds: Iterable[int] = (42, 43, 44)) -> tuple[DiagnosticJob, ...]:
    """构造 T5 六个任务和一个不重复训练 attention 基线的 T6 任务。"""
    values = _validate_seeds(seeds)
    jobs: list[DiagnosticJob] = []
    for seed in values:
        jobs.append(
            DiagnosticJob(
                suite="t5",
                variant="legacy",
                seed=seed,
                run_name=f"full_joint_t5_legacy_seed{seed}",
                overrides=_legacy_overrides(),
            )
        )
    for seed in values:
        jobs.append(
            DiagnosticJob(
                suite="t5",
                variant="v2",
                seed=seed,
                run_name=f"full_joint_t5_v2_seed{seed}",
                overrides=_v2_overrides(),
            )
        )
    jobs.append(
        DiagnosticJob(
            suite="t6",
            variant="local_only",
            seed=values[0],
            run_name=f"full_joint_t6_local_only_seed{values[0]}",
            overrides=_legacy_overrides(context_mode="local_only"),
        )
    )
    return tuple(jobs)


def build_train_command(
    job: DiagnosticJob,
    *,
    max_episodes: int = 20,
    num_envs: int = 4,
    batch_size: int = 32,
) -> list[str]:
    """生成不依赖 shell 语法的 Hydra 训练命令。"""
    if max_episodes < 1 or num_envs < 1 or batch_size < 1:
        raise ValueError("max_episodes、num_envs、batch_size 必须为正数")
    return [
        sys.executable,
        "train.py",
        "experiment=full_joint_diagnostic",
        f"experiment.experiment_name={job.run_name}",
        f"seed={job.seed}",
        f"num_envs={num_envs}",
        f"train.max_episodes={max_episodes}",
        f"train.batch_size={batch_size}",
        "train.accumulation_steps=2",
        "train.async_eval_enabled=false",
        *job.overrides,
    ]


def jobs_for_suite(suite: str, seeds: Iterable[int]) -> tuple[DiagnosticJob, ...]:
    """按套件筛选任务；all 仅用于串行保守执行。"""
    if suite not in {"t5", "t6", "all"}:
        raise ValueError("suite 仅允许 t5、t6 或 all")
    jobs = build_jobs(seeds)
    if suite == "all":
        return jobs
    return tuple(job for job in jobs if job.suite == suite)
