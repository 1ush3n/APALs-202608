"""r5_task_delay_v1 的固定任务延迟场景生成与审计。"""

from __future__ import annotations

import hashlib
import json
import math
import csv
from pathlib import Path
from typing import Any

import numpy as np

from utils.reschedule import (
    BaselineSchedule,
    RescheduleScenario,
    eligible_delay_tasks,
    load_baseline_schedule,
)


R5_PROTOCOL = "r5_task_delay_v1"
R5_STAGES = ("early", "middle", "late")
R5_SEVERITIES = ("low", "medium", "high")


def _stable_seed(instance_id: str, seed: int, stage: str) -> int:
    source = f"{R5_PROTOCOL}|{instance_id}|{int(seed)}|{stage}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(source).digest()[:8], "big") % (2**32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_configuration(
    stage_ratios: dict[str, float],
    severity_specs: dict[str, tuple[float, float, float]],
) -> None:
    if tuple(stage_ratios) != R5_STAGES:
        raise ValueError(f"stage_ratios 必须按 {R5_STAGES} 提供")
    if set(severity_specs) != set(R5_SEVERITIES):
        raise ValueError(f"severity_specs 必须包含 {R5_SEVERITIES}")
    for stage in R5_STAGES:
        ratio = float(stage_ratios[stage])
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"stage_ratios[{stage}] 必须位于 (0, 1)")
    for severity in R5_SEVERITIES:
        spec = tuple(severity_specs[severity])
        if len(spec) != 3:
            raise ValueError(f"severity_specs[{severity}] 必须为 (比例, 最小延迟, 最大延迟)")
        task_ratio, delay_min, delay_max = map(float, spec)
        if not 0.0 < task_ratio < 1.0:
            raise ValueError(f"severity_specs[{severity}] 的任务比例必须位于 (0, 1)")
        if not 0.0 < delay_min <= delay_max:
            raise ValueError(f"severity_specs[{severity}] 的延迟范围无效")


def _scenario_metadata(
    *,
    scenario_id: str,
    stage: str,
    severity: str,
    start_time: float,
    baseline: BaselineSchedule,
    candidates: list[int],
    delayed_ids: list[int],
    delay_by_task: dict[int, float],
) -> dict[str, Any]:
    return {
        "generation_version": R5_PROTOCOL,
        "scenario_id": scenario_id,
        "stage": stage,
        "severity": severity,
        "reschedule_start_time": float(start_time),
        "reschedule_start_ratio": float(start_time / max(1e-12, baseline.makespan)),
        "candidate_task_ids": [int(task_id) for task_id in candidates],
        "delayed_task_ids": [int(task_id) for task_id in delayed_ids],
        "candidate_task_count": len(candidates),
        "delayed_task_count": len(delayed_ids),
        "baseline_start_by_task": {
            str(task_id): float(baseline.tasks[task_id].start) for task_id in delayed_ids
        },
        "release_time_by_task": {
            str(task_id): float(baseline.tasks[task_id].start + delay_by_task[task_id])
            for task_id in delayed_ids
        },
        "delay_by_task": {
            str(task_id): float(delay_by_task[task_id]) for task_id in delayed_ids
        },
        "delay_mean_h": float(np.mean(list(delay_by_task.values()))) if delay_by_task else 0.0,
        "delay_max_h": float(max(delay_by_task.values())) if delay_by_task else 0.0,
        "delay_total_h": float(sum(delay_by_task.values())),
        "nesting_checks": {},
    }


def generate_r5_scenario_library(
    baseline: BaselineSchedule,
    *,
    instance_id: str,
    seed: int,
    stage_ratios: dict[str, float],
    severity_specs: dict[str, tuple[float, float, float]],
) -> tuple[list[tuple[str, RescheduleScenario]], list[dict[str, Any]], dict[str, Any]]:
    """为一个实例生成 3 个阶段 × 3 个强度的固定场景。"""
    _validate_configuration(stage_ratios, severity_specs)
    if baseline.makespan <= 0.0 or not baseline.tasks:
        raise ValueError("baseline 必须包含正 makespan 和任务")

    scenarios: list[tuple[str, RescheduleScenario]] = []
    metadata: list[dict[str, Any]] = []
    for stage in R5_STAGES:
        start_time = float(stage_ratios[stage]) * float(baseline.makespan)
        candidates = eligible_delay_tasks(baseline, start_time)
        if not candidates:
            raise ValueError(f"阶段 {stage} 没有可延迟任务")

        rng = np.random.RandomState(_stable_seed(instance_id, seed, stage))
        permutation = rng.permutation(len(candidates))
        ordered_candidates = [candidates[int(index)] for index in permutation]
        quantiles = {
            int(task.task_id): float(rng.uniform(0.0, 1.0))
            for task in candidates
        }
        selected_by_severity: dict[str, list[int]] = {}
        for severity in R5_SEVERITIES:
            task_ratio, _delay_min, _delay_max = severity_specs[severity]
            count = max(1, int(math.ceil(float(task_ratio) * len(candidates))))
            selected_by_severity[severity] = [
                int(task.task_id) for task in ordered_candidates[:count]
            ]
        if not (
            len(selected_by_severity["low"])
            < len(selected_by_severity["medium"])
            < len(selected_by_severity["high"])
        ):
            raise ValueError(
                f"阶段 {stage} 的 r5 任务数量无法形成严格 low<medium<high 嵌套"
            )

        for severity in R5_SEVERITIES:
            _task_ratio, delay_min, delay_max = severity_specs[severity]
            delayed_ids = selected_by_severity[severity]
            delay_by_task = {
                task_id: float(delay_min) + quantiles[task_id] * (float(delay_max) - float(delay_min))
                for task_id in delayed_ids
            }
            release_times = {
                task_id: float(baseline.tasks[task_id].start) + delay
                for task_id, delay in delay_by_task.items()
            }
            scenario_id = f"{severity}_{stage}"
            scenarios.append(
                (
                    scenario_id,
                    RescheduleScenario(
                        start_time=start_time,
                        task_release_times=release_times,
                    ),
                )
            )
            metadata.append(
                _scenario_metadata(
                    scenario_id=scenario_id,
                    stage=stage,
                    severity=severity,
                    start_time=start_time,
                    baseline=baseline,
                    candidates=[int(task.task_id) for task in candidates],
                    delayed_ids=delayed_ids,
                    delay_by_task=delay_by_task,
                )
            )

    manifest = {
        "protocol": R5_PROTOCOL,
        "protocol_version": 1,
        "instance_id": str(instance_id),
        "seed": int(seed),
        "baseline_makespan": float(baseline.makespan),
        "stage_ratios": {key: float(stage_ratios[key]) for key in R5_STAGES},
        "severity_specs": {
            key: [float(value) for value in severity_specs[key]]
            for key in R5_SEVERITIES
        },
        "scenario_count": len(scenarios),
        "generation_version": R5_PROTOCOL,
        "scenario_ids": [scenario_id for scenario_id, _scenario in scenarios],
        "scenarios": metadata,
    }
    for stage in R5_STAGES:
        rows = [row for row in metadata if row["stage"] == stage]
        by_severity = {row["severity"]: row for row in rows}
        nesting = {
            "low_subset_medium": set(by_severity["low"]["delayed_task_ids"]) <= set(by_severity["medium"]["delayed_task_ids"]),
            "medium_subset_high": set(by_severity["medium"]["delayed_task_ids"]) <= set(by_severity["high"]["delayed_task_ids"]),
        }
        for row in rows:
            row["nesting_checks"] = nesting
    return scenarios, metadata, manifest


def write_r5_scenario_library(
    *,
    baseline_path: Path,
    output_path: Path,
    metadata_path: Path,
    instance_id: str,
    seed: int,
    stage_ratios: dict[str, float],
    severity_specs: dict[str, tuple[float, float, float]],
) -> dict[str, Any]:
    """生成并写入 r5 CSV 与可审计元数据。"""
    baseline_path = Path(baseline_path)
    output_path = Path(output_path)
    metadata_path = Path(metadata_path)
    baseline = load_baseline_schedule(baseline_path)
    scenarios, _metadata, manifest = generate_r5_scenario_library(
        baseline,
        instance_id=instance_id,
        seed=seed,
        stage_ratios=stage_ratios,
        severity_specs=severity_specs,
    )
    manifest["baseline_sha256"] = _sha256(baseline_path)
    metadata_by_id = {
        str(row["scenario_id"]): row for row in manifest["scenarios"]
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "scenario_id",
                "stage",
                "severity",
                "reschedule_start_time",
                "TaskID",
                "baseline_start",
                "release_time",
                "delay_h",
                "eligible_task_count",
                "delayed_task_count",
            ],
        )
        writer.writeheader()
        for scenario_id, scenario in scenarios:
            row = metadata_by_id[str(scenario_id)]
            for task_id in sorted(scenario.task_release_times):
                writer.writerow(
                    {
                        "scenario_id": scenario_id,
                        "stage": row["stage"],
                        "severity": row["severity"],
                        "reschedule_start_time": float(scenario.start_time),
                        "TaskID": int(task_id),
                        "baseline_start": row["baseline_start_by_task"][str(task_id)],
                        "release_time": row["release_time_by_task"][str(task_id)],
                        "delay_h": row["delay_by_task"][str(task_id)],
                        "eligible_task_count": row["candidate_task_count"],
                        "delayed_task_count": row["delayed_task_count"],
                    }
                )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_r5_scenario_library(
    baseline: BaselineSchedule,
    metadata: dict[str, Any],
    *,
    instance_id: str,
) -> None:
    """拒绝零延迟、错误数量或破坏嵌套关系的 r5 场景库。"""
    if str(metadata.get("protocol", "")) != R5_PROTOCOL:
        raise ValueError("场景元数据 protocol 不是 r5_task_delay_v1")
    if str(metadata.get("instance_id", "")) != str(instance_id):
        raise ValueError("场景元数据 instance_id 不匹配")
    metadata_makespan = float(metadata.get("baseline_makespan", 0.0))
    if not math.isclose(metadata_makespan, float(baseline.makespan), rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("场景元数据 baseline_makespan 不匹配")
    stage_ratios = metadata.get("stage_ratios")
    severity_specs = metadata.get("severity_specs")
    if not isinstance(stage_ratios, dict) or not isinstance(severity_specs, dict):
        raise ValueError("r5 场景元数据缺少 stage_ratios 或 severity_specs")
    scenarios = metadata.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != 9:
        raise ValueError("r5 场景库必须包含 9 个场景")
    expected_ids = [
        f"{severity}_{stage}"
        for stage in R5_STAGES
        for severity in R5_SEVERITIES
    ]
    if list(metadata.get("scenario_ids", [])) != expected_ids:
        raise ValueError("r5 场景 ID 顺序或集合不正确")
    by_stage: dict[str, dict[str, dict[str, Any]]] = {}
    for row in scenarios:
        stage = str(row["stage"])
        severity = str(row["severity"])
        if stage not in R5_STAGES or severity not in R5_SEVERITIES:
            raise ValueError("r5 场景元数据包含未知阶段或扰动强度")
        if str(row.get("scenario_id", "")) != f"{severity}_{stage}":
            raise ValueError("r5 场景 ID 与 stage/severity 不一致")
        expected_start = float(stage_ratios[stage]) * float(baseline.makespan)
        if not math.isclose(float(row["reschedule_start_time"]), expected_start, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"阶段 {stage} 的重调度时刻不匹配")
        candidates = eligible_delay_tasks(baseline, expected_start)
        candidate_ids = {int(task.task_id) for task in candidates}
        row_candidate_ids = {int(task_id) for task_id in row["candidate_task_ids"]}
        if row_candidate_ids != candidate_ids or int(row["candidate_task_count"]) != len(candidate_ids):
            raise ValueError(f"阶段 {stage} 的候选工序集合不匹配")
        by_stage.setdefault(stage, {})[severity] = row
        delayed_ids = [int(task_id) for task_id in row["delayed_task_ids"]]
        task_ratio, delay_min, delay_max = map(float, severity_specs[severity])
        expected_count = max(1, int(math.ceil(task_ratio * len(candidate_ids))))
        if len(delayed_ids) != expected_count or int(row["delayed_task_count"]) != expected_count:
            raise ValueError(f"场景 {row['scenario_id']} 的受扰工序数量不匹配")
        if not set(delayed_ids) <= candidate_ids:
            raise ValueError(f"场景 {row['scenario_id']} 包含不可延迟工序")
        if not math.isclose(float(row["reschedule_start_ratio"]), float(stage_ratios[stage]), rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"场景 {row['scenario_id']} 的重调度比例不匹配")
        for task_id in delayed_ids:
            baseline_start = float(baseline.tasks[task_id].start)
            release_time = float(row["release_time_by_task"][str(task_id)])
            delay = float(row["delay_by_task"][str(task_id)])
            if not delay_min <= delay <= delay_max or not math.isclose(release_time - baseline_start, delay, rel_tol=0.0, abs_tol=1e-8):
                raise ValueError(f"场景 {row['scenario_id']} 的逐任务延迟字段不匹配")
            if release_time <= baseline_start:
                raise ValueError(f"场景 {row['scenario_id']} 存在非正实际延迟: task={task_id}")
        delays = [float(row["delay_by_task"][str(task_id)]) for task_id in delayed_ids]
        if not math.isclose(float(row["delay_mean_h"]), float(np.mean(delays)), rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"场景 {row['scenario_id']} 的 delay_mean_h 不匹配")
        if not math.isclose(float(row["delay_total_h"]), float(sum(delays)), rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"场景 {row['scenario_id']} 的 delay_total_h 不匹配")
        if not math.isclose(float(row["delay_max_h"]), float(max(delays)), rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"场景 {row['scenario_id']} 的 delay_max_h 不匹配")

    for stage in R5_STAGES:
        rows = by_stage.get(stage, {})
        if set(rows) != set(R5_SEVERITIES):
            raise ValueError(f"阶段 {stage} 缺少 low/medium/high 场景")
        low = set(int(task_id) for task_id in rows["low"]["delayed_task_ids"])
        medium = set(int(task_id) for task_id in rows["medium"]["delayed_task_ids"])
        high = set(int(task_id) for task_id in rows["high"]["delayed_task_ids"])
        if not low < medium < high:
            raise ValueError(f"阶段 {stage} 的受扰任务集合不是嵌套关系")
        if len(rows) != 3 or len({float(row["reschedule_start_time"]) for row in rows.values()}) != 1:
            raise ValueError(f"阶段 {stage} 的三个扰动强度没有共享重调度时刻")
        for task_id in low:
            low_delay = float(rows["low"]["delay_by_task"][str(task_id)])
            medium_delay = float(rows["medium"]["delay_by_task"][str(task_id)])
            high_delay = float(rows["high"]["delay_by_task"][str(task_id)])
            if not low_delay < medium_delay < high_delay:
                raise ValueError(f"阶段 {stage} 的共同任务延迟不单调: task={task_id}")


def sample_r5_training_scenario(
    baseline: BaselineSchedule,
    *,
    rng: np.random.RandomState,
    stage_ratios: dict[str, float],
    severity_specs: dict[str, tuple[float, float, float]],
) -> tuple[str, RescheduleScenario]:
    """训练期按固定 r5 规则采样一个场景，不改变正式固定场景资产。"""
    stage = R5_STAGES[int(rng.randint(0, len(R5_STAGES)))]
    severity = R5_SEVERITIES[int(rng.randint(0, len(R5_SEVERITIES)))]
    generated, _metadata, _manifest = generate_r5_scenario_library(
        baseline,
        instance_id="training",
        seed=int(rng.randint(0, 2**32 - 1)),
        stage_ratios=stage_ratios,
        severity_specs=severity_specs,
    )
    scenario_map = dict(generated)
    scenario_id = f"{severity}_{stage}"
    return scenario_id, scenario_map[scenario_id]


__all__ = [
    "R5_PROTOCOL",
    "R5_STAGES",
    "R5_SEVERITIES",
    "generate_r5_scenario_library",
    "sample_r5_training_scenario",
    "validate_r5_scenario_library",
    "write_r5_scenario_library",
]
