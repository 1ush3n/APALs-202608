# -*- coding: utf-8 -*-
"""批量评估 APAL 重调度规则基线。"""

from __future__ import annotations

import json
import os
import sys
import hashlib
import contextlib
import io
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.heuristic.reschedule_rules import DEFAULT_RULE_METHODS, rule_registry
from configs import configs
from runtime.artifacts import resolve_run_output_dir, write_run_context_files, write_run_manifest
from runtime.hydra_config import (
    ExtraArgument,
    HydraCliError,
    hydra_help,
    initialize_hydra_runtime,
    should_show_help,
)
from runtime.paths import resolve_workspace_path
from runtime.reschedule_eval import ensure_reschedule_baseline_available, ensure_reschedule_eval_scenarios_available
from runtime.reschedule_manifest import (
    REAL_INSTANCE_IDS,
    load_reschedule_manifest,
    validate_r5_manifest_assets,
)
from utils.reschedule import load_reschedule_scenarios


RULE_EXTRA_ARGS = {
    "beam_width": ExtraArgument(default=2, help="Beam Search 候选解数量"),
    "beam_branch_factor": ExtraArgument(default=2, help="Beam Search 每个候选解的扰动分支数"),
    "beam_levels": ExtraArgument(default=2, help="Beam Search 最大展开层数"),
    "beam_patience": ExtraArgument(default=1, help="Beam Search 连续无改进提前停止层数"),
    "ig_iterations": ExtraArgument(default=3, help="Iterated Greedy / Destroy-Repair 迭代次数"),
    "ig_destroy_ratio": ExtraArgument(default=0.08, help="Iterated Greedy 每次破坏的可移动任务比例"),
    "ig_noise_sigma": ExtraArgument(default=0.20, help="Iterated Greedy 修复优先级扰动强度"),
    "sa_iterations": ExtraArgument(default=3, help="Simulated Annealing 迭代次数"),
    "sa_initial_temp": ExtraArgument(default=0.05, help="Simulated Annealing 初始温度"),
    "sa_cooling": ExtraArgument(default=0.96, help="Simulated Annealing 降温系数"),
    "sa_min_temp": ExtraArgument(default=1e-4, help="Simulated Annealing 最小温度"),
    "methods": ExtraArgument(default=None, help="规则列表，例如 methods=[SPTRepair,CPMRepair]；缺省评估全部规则"),
    "scenario_path": ExtraArgument(default=None, help="固定重调度场景 CSV；缺省使用配置中的 reschedule_eval_scenario_path"),
    "baseline_path": ExtraArgument(default=None, help="baseline 调度 CSV；缺省使用配置中的 baseline 路径"),
    "data_path": ExtraArgument(default=None, help="APAL 数据文件或目录；缺省使用配置中的 data_file_path"),
    "manifest_path": ExtraArgument(default=None, help="可选 manifest；提供后按 instance_ids 自动取 data/baseline/scenario"),
    "instance_ids": ExtraArgument(default=None, help="manifest 实例列表，例如 instance_ids=[real_680]"),
    "num_runs": ExtraArgument(default=None, help="最多评估多少个场景；缺省评估全部场景"),
    "seed": ExtraArgument(default=42, help="规则评估固定种子"),
    "parallel_workers": ExtraArgument(default=0, help="规则评估并行进程数；0 表示自动 min(8, cpu)，1 表示串行"),
    "parallel_backend": ExtraArgument(default="process", help="并行后端；当前支持 process"),
    "verify_static_cache": ExtraArgument(default=False, help="是否抽样校验规则静态约束缓存与原始环境查询一致"),
    "resume_partial": ExtraArgument(default=True, help="是否从同一 output_dir 的断点结果继续运行"),
    "force_rerun": ExtraArgument(default=False, help="是否忽略断点并重新计算全部任务"),
    "flush_every": ExtraArgument(default=1, help="每完成多少个 job 刷新一次断点文件"),
    "progress_interval": ExtraArgument(default=30.0, help="没有新 job 完成时的进度心跳间隔，单位秒；0 表示关闭心跳"),
    "quiet": ExtraArgument(default=False, help="是否关闭逐场景输出"),
    "output_dir": ExtraArgument(default=None, help="输出目录；缺省写入本次 run 的 eval/reschedule_rules"),
}

PARTIAL_FILE_NAME = "reschedule_rule_eval_partial.csv"
RESUME_STATE_FILE_NAME = "reschedule_rule_resume_state.json"
R5_STOCHASTIC_METHODS = frozenset(
    {
        "RandomRepair",
        "Beam",
        "BeamSearch",
        "BeamSearchRepair",
        "IG",
        "DestroyRepair",
        "IteratedGreedy",
        "IteratedGreedyRepair",
        "SA",
        "SimulatedAnnealing",
        "SimulatedAnnealingRepair",
    }
)


def _normalize_methods(raw: Any) -> list[str]:
    registry = rule_registry()
    if raw is None or raw == "":
        methods = list(DEFAULT_RULE_METHODS)
    elif isinstance(raw, str):
        methods = [item.strip() for item in raw.split(",") if item.strip()]
    elif isinstance(raw, (list, tuple)):
        methods = [str(item).strip() for item in raw if str(item).strip()]
    else:
        raise ValueError(f"无法解析 methods 参数: {raw!r}")

    unknown = [method for method in methods if method not in registry]
    if unknown:
        raise ValueError(f"未知重调度规则: {unknown}；可选规则: {sorted(registry)}")
    return methods


def _scenario_level_from_id(scenario_id: str) -> str:
    head = str(scenario_id).split("_", 1)[0]
    return head if head in {"low", "medium", "high"} else "custom"


def _scenario_stage_from_id(scenario_id: str) -> str:
    parts = str(scenario_id).split("_", 1)
    stage = parts[1] if len(parts) == 2 else ""
    return stage if stage in {"early", "middle", "late"} else "custom"


def _solver_seed_count(method_name: str, *, r5: bool) -> int:
    return 3 if r5 and str(method_name) in R5_STOCHASTIC_METHODS else 1


def _as_id_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError(f"无法解析 instance_ids 参数: {value!r}")


def _summarize(df: pd.DataFrame, group_cols: list[str]) -> list[dict[str, Any]]:
    if df.empty:
        return []
    metric_cols = [
        "makespan",
        "balance_std",
        "score",
        "selection_score",
        "eligible",
        "complete",
        "duration_sec",
        "worker_util",
        "station_util",
        "frozen_violation_count",
        "release_violation_count",
        "precedence_violation_count",
        "worker_overlap_violation_count",
        "station_slot_violation_count",
        "skill_violation_count",
        "demand_violation_count",
        "fixed_station_violation_count",
        "station_range_violation_count",
        "physical_station_violation_count",
        "worker_station_binding_violation_count",
        "duplicate_task_count",
        "missing_task_count",
        "takt_violation_h",
        "start_deviation_mean_h",
        "station_change_rate",
        "team_change_rate",
    ]
    available = [col for col in metric_cols if col in df.columns]
    grouped = df.groupby(group_cols, dropna=False)[available].mean(numeric_only=True).reset_index()
    grouped = grouped.rename(columns={"eligible": "eligible_rate", "complete": "complete_rate"})
    return grouped.to_dict(orient="records")


def _summarize_solver_seeds(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty or "solver_seed_index" not in df.columns:
        return []
    group_cols = [
        column
        for column in ("instance_id", "scenario_id", "scenario_level", "scenario_stage", "method")
        if column in df.columns
    ]
    metric_cols = [
        column
        for column in ("selection_score", "score", "makespan", "eligible", "duration_sec")
        if column in df.columns
    ]
    if not group_cols or not metric_cols:
        return []
    grouped = df.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        "_".join(str(part) for part in column if str(part)) if isinstance(column, tuple) else str(column)
        for column in grouped.columns
    ]
    return grouped.fillna(0.0).to_dict(orient="records")


def _resolve_parallel_workers(value: Any) -> int:
    workers = int(value)
    if workers < 0:
        raise ValueError("parallel_workers 不能为负数")
    if workers == 0:
        return max(1, min(8, os.cpu_count() or 1))
    return workers


def _job_key(job: dict[str, Any]) -> str:
    return "|".join(
        [
            str(job.get("instance_id", "")),
            str(job["scenario_id"]),
            str(job["method_name"]),
            str(int(job["seed"])),
            str(int(job.get("solver_seed_index", 0))),
        ]
    )


def _resume_signature(jobs: list[dict[str, Any]], signature_payload: dict[str, Any]) -> dict[str, Any]:
    job_items = [
        {
            "order": int(job["order"]),
            "job_key": str(job["_job_key"]),
            "instance_id": str(job.get("instance_id", "")),
            "scenario_id": str(job["scenario_id"]),
            "scenario_level": str(job["scenario_level"]),
            "scenario_stage": str(job.get("scenario_stage", "custom")),
            "method": str(job["method_name"]),
            "seed": int(job["seed"]),
            "solver_seed_index": int(job.get("solver_seed_index", 0)),
            "data_path": str(job["data_path"]),
            "baseline_path": str(job["baseline_path"]),
            "scenario_path": str(job["scenario_path"]),
        }
        for job in jobs
    ]
    payload = dict(signature_payload)
    payload["jobs"] = job_items
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {"hash": hashlib.sha256(encoded.encode("utf-8")).hexdigest(), "payload": payload}


def _write_resume_state(
    output_dir: Path,
    *,
    signature: dict[str, Any],
    status: str,
    total_jobs: int,
    completed_jobs: int,
) -> None:
    payload = {
        "status": status,
        "signature_hash": signature["hash"],
        "signature": signature["payload"],
        "total_jobs": int(total_jobs),
        "completed_jobs": int(completed_jobs),
        "pending_jobs": int(max(0, total_jobs - completed_jobs)),
        "partial_file": PARTIAL_FILE_NAME,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / RESUME_STATE_FILE_NAME
    tmp_path = state_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(state_path)


def _write_partial_rows(output_dir: Path, rows: list[dict[str, Any]], signature: dict[str, Any], *, status: str, total_jobs: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: int(row["_order"]))
    partial_path = output_dir / PARTIAL_FILE_NAME
    tmp_path = partial_path.with_suffix(".csv.tmp")
    pd.DataFrame(ordered).to_csv(tmp_path, index=False)
    tmp_path.replace(partial_path)
    _write_resume_state(
        output_dir,
        signature=signature,
        status=status,
        total_jobs=total_jobs,
        completed_jobs=len(ordered),
    )


def _load_resume_rows(
    output_dir: Path | None,
    signature: dict[str, Any],
    *,
    resume: bool,
    force_rerun: bool,
) -> dict[str, dict[str, Any]]:
    if output_dir is None or not resume:
        return {}
    partial_path = output_dir / PARTIAL_FILE_NAME
    state_path = output_dir / RESUME_STATE_FILE_NAME
    if force_rerun:
        for path in (partial_path, state_path):
            if path.exists():
                path.unlink()
        return {}
    if not partial_path.exists():
        return {}
    if not state_path.exists():
        raise RuntimeError(f"发现断点明细但缺少状态文件: {partial_path}；请使用 force_rerun=true 重新计算")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("signature_hash") != signature["hash"]:
        raise RuntimeError(
            "output_dir 中已有断点，但运行参数签名不一致；请换一个 output_dir，或使用 force_rerun=true 重新计算"
        )
    df = pd.read_csv(partial_path)
    if df.empty:
        return {}
    if "_job_key" not in df.columns:
        raise RuntimeError(f"断点文件缺少 _job_key 字段: {partial_path}；请使用 force_rerun=true 重新计算")
    rows: dict[str, dict[str, Any]] = {}
    for raw in df.to_dict(orient="records"):
        key = str(raw["_job_key"])
        rows[key] = raw
    return rows


def _format_progress_row(row: dict[str, Any], completed: int, total: int, *, elapsed_sec: float) -> str:
    method = row.get("method", "")
    instance = row.get("_instance_id", "") or "-"
    scenario = row.get("scenario_id", "")
    score = float(row.get("score", 0.0))
    eligible = int(float(row.get("eligible", 0.0)))
    duration = float(row.get("duration_sec", 0.0))
    avg = elapsed_sec / max(1, completed)
    eta = avg * max(0, total - completed)
    return (
        f"[RuleEval][Progress] {completed}/{total} done "
        f"last={instance}/{scenario}/{method} "
        f"score={score:.4f} elig={eligible} job={duration:.2f}s "
        f"elapsed={elapsed_sec:.1f}s eta~{eta:.1f}s"
    )


def _apply_rule_job_config(config_snapshot: dict[str, Any], *, baseline_path: str, scenario_path: str) -> None:
    configs.update_from_dict(config_snapshot)
    configs.enable_dynamic_events = False
    configs.enable_station_breakdown = False
    configs.enable_material_delay = False
    configs.enable_online_duration_perturb = False
    configs.enable_worker_fatigue = False
    configs.randomize_durations = False
    configs.reschedule_manifest_path = ""
    configs.reschedule_eval_instance_id = ""
    configs.reschedule_scenario_path = ""
    configs.reschedule_eval_scenario_path = str(scenario_path)
    configs.reschedule_baseline_schedule_path = str(baseline_path)


def _row_from_rule_result(result: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method": result.method,
        "scenario_id": result.scenario_id,
        "scenario_level": result.scenario_level,
        "scenario_start_time": result.scenario_start_time,
        "delayed_task_count": float(result.delayed_task_count),
        "makespan": result.makespan,
        "balance_std": result.balance_std,
        "reward": result.reward,
        "duration_sec": result.duration_sec,
        "score": float(result.constraint_metrics.get("composite_score", 0.0)),
        "selection_score": float(result.constraint_metrics.get("selection_score", 0.0)),
    }
    row.update(result.constraint_metrics)
    return row


def _evaluate_rule_job(job: dict[str, Any]) -> dict[str, Any]:
    try:
        _apply_rule_job_config(
            dict(job["config_snapshot"]),
            baseline_path=str(job["baseline_path"]),
            scenario_path=str(job["scenario_path"]),
        )
        def run_solver() -> dict[str, Any]:
            registry = rule_registry()
            solver_cls = registry[str(job["method_name"])]
            solver_kwargs: dict[str, Any] = {
                "data_path_or_dir": str(job["data_path"]),
                "scenario": job["scenario"],
                "scenario_id": str(job["scenario_id"]),
                "scenario_level": str(job["scenario_level"]),
                "seed": int(job["seed"]),
                "verbose": bool(job["verbose"]),
                "verify_static_cache": bool(job["verify_static_cache"]),
            }
            if bool(getattr(solver_cls, "supports_priority_search", False)):
                solver_kwargs.update(dict(job["search_kwargs"]))
            solver = solver_cls(**solver_kwargs)
            return _row_from_rule_result(solver.run())

        if bool(job.get("suppress_worker_stdout", True)):
            with contextlib.redirect_stdout(io.StringIO()):
                row = run_solver()
        else:
            row = run_solver()
        row["method"] = str(job["method_name"])
        row["scenario_level"] = str(job["scenario_level"])
        row["scenario_stage"] = str(job.get("scenario_stage", "custom"))
        row["solver_seed_index"] = int(job.get("solver_seed_index", 0))
        row["solver_seed"] = int(job["seed"])
        row["_job_key"] = str(job["_job_key"])
        row["_order"] = int(job["order"])
        row["_instance_id"] = str(job.get("instance_id", ""))
        row["_data_path"] = str(job["data_path"])
        row["_baseline_path"] = str(job["baseline_path"])
        row["_scenario_path"] = str(job["scenario_path"])
        return row
    except Exception as exc:
        raise RuntimeError(
            "规则评估失败: "
            f"instance={job.get('instance_id', '')} "
            f"scenario={job.get('scenario_id', '')} "
            f"method={job.get('method_name', '')} "
            f"seed={job.get('seed', '')}: {exc}"
        ) from exc


def _run_rule_jobs(
    jobs: list[dict[str, Any]],
    *,
    parallel_workers: int,
    parallel_backend: str,
    output_dir: Path | None = None,
    resume: bool = True,
    force_rerun: bool = False,
    flush_every: int = 1,
    signature_payload: dict[str, Any] | None = None,
    show_progress: bool = False,
    progress_interval: float = 30.0,
) -> list[dict[str, Any]]:
    if not jobs:
        return []
    for job in jobs:
        job["_job_key"] = _job_key(job)
    signature = _resume_signature(jobs, signature_payload or {})
    output_path = Path(output_dir) if output_dir is not None else None
    completed_by_key = _load_resume_rows(
        output_path,
        signature,
        resume=bool(resume),
        force_rerun=bool(force_rerun),
    )
    pending_jobs = [job for job in jobs if str(job["_job_key"]) not in completed_by_key]
    rows = list(completed_by_key.values())
    flush_every = max(1, int(flush_every))
    total_jobs = len(jobs)
    progress_interval = max(0.0, float(progress_interval))
    started_at = time.time()
    if output_path is not None:
        _write_partial_rows(output_path, rows, signature, status="running", total_jobs=len(jobs))
    if show_progress:
        print(
            f"[RuleEval][Progress] total={total_jobs} completed={len(rows)} "
            f"resume_hit={len(completed_by_key)} pending={len(pending_jobs)} "
            f"workers={_resolve_parallel_workers(parallel_workers)}",
            flush=True,
        )
    if not pending_jobs:
        rows.sort(key=lambda row: int(row["_order"]))
        if output_path is not None:
            _write_partial_rows(output_path, rows, signature, status="complete", total_jobs=len(jobs))
        if show_progress:
            print(f"[RuleEval][Progress] complete {len(rows)}/{total_jobs}", flush=True)
        return rows

    workers = _resolve_parallel_workers(parallel_workers)
    backend = str(parallel_backend or "process").lower()
    completed_since_flush = 0
    try:
        if workers == 1:
            for job in pending_jobs:
                if show_progress:
                    print(
                        f"[RuleEval][Progress] running {len(rows) + 1}/{total_jobs} "
                        f"{job.get('instance_id', '') or '-'}/{job['scenario_id']}/{job['method_name']}",
                        flush=True,
                    )
                row = _evaluate_rule_job(job)
                rows.append(row)
                completed_since_flush += 1
                if show_progress:
                    print(
                        _format_progress_row(row, len(rows), total_jobs, elapsed_sec=time.time() - started_at),
                        flush=True,
                    )
                if output_path is not None and completed_since_flush >= flush_every:
                    _write_partial_rows(output_path, rows, signature, status="running", total_jobs=len(jobs))
                    completed_since_flush = 0
        else:
            if backend != "process":
                raise ValueError(f"不支持的 parallel_backend={parallel_backend!r}；当前仅支持 process")
            executor = ProcessPoolExecutor(max_workers=workers)
            try:
                future_map = {executor.submit(_evaluate_rule_job, job): job for job in pending_jobs}
                pending = set(future_map)
                last_heartbeat = time.time()
                while pending:
                    done, pending = wait(pending, timeout=max(0.5, progress_interval) if progress_interval > 0 else None, return_when=FIRST_COMPLETED)
                    if not done:
                        if show_progress and progress_interval > 0 and time.time() - last_heartbeat >= progress_interval:
                            print(
                                f"[RuleEval][Progress] waiting completed={len(rows)}/{total_jobs} "
                                f"running<={min(workers, len(pending))} pending={len(pending)}",
                                flush=True,
                            )
                            last_heartbeat = time.time()
                        continue
                    for future in done:
                        row = future.result()
                        rows.append(row)
                        completed_since_flush += 1
                        if show_progress:
                            print(
                                _format_progress_row(row, len(rows), total_jobs, elapsed_sec=time.time() - started_at),
                                flush=True,
                            )
                        if output_path is not None and completed_since_flush >= flush_every:
                            _write_partial_rows(output_path, rows, signature, status="running", total_jobs=len(jobs))
                            completed_since_flush = 0
            except KeyboardInterrupt:
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True)
    except KeyboardInterrupt:
        if output_path is not None:
            _write_partial_rows(output_path, rows, signature, status="interrupted", total_jobs=len(jobs))
        raise
    if output_path is not None:
        _write_partial_rows(output_path, rows, signature, status="complete", total_jobs=len(jobs))
    rows.sort(key=lambda row: int(row["_order"]))
    if show_progress:
        print(f"[RuleEval][Progress] complete {len(rows)}/{total_jobs}", flush=True)
    return rows


def _strip_internal_row_keys(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _console_summary(summary: dict[str, Any]) -> dict[str, Any]:
    compact = {key: value for key, value in summary.items() if key != "rows"}
    if isinstance(compact.get("instances"), dict):
        instances = {}
        for instance_id, payload in compact["instances"].items():
            if isinstance(payload, dict):
                instances[instance_id] = {key: value for key, value in payload.items() if key != "rows"}
            else:
                instances[instance_id] = payload
        compact["instances"] = instances
    return compact


def evaluate_reschedule_rules(
    *,
    data_path_or_dir: str | Path | None = None,
    scenario_path: str | Path | None = None,
    baseline_path: str | Path | None = None,
    methods: Any = None,
    num_runs: int | None = None,
    seed: int = 42,
    output_dir: str | Path | None = None,
    verbose: bool = True,
    beam_width: int = 2,
    beam_branch_factor: int = 2,
    beam_levels: int = 2,
    beam_patience: int = 1,
    ig_iterations: int = 3,
    ig_destroy_ratio: float = 0.08,
    ig_noise_sigma: float = 0.20,
    sa_iterations: int = 3,
    sa_initial_temp: float = 0.05,
    sa_cooling: float = 0.96,
    sa_min_temp: float = 1e-4,
    parallel_workers: int = 1,
    parallel_backend: str = "process",
    verify_static_cache: bool = False,
    resume: bool = True,
    force_rerun: bool = False,
    flush_every: int = 1,
    show_progress: bool = False,
    progress_interval: float = 30.0,
) -> dict[str, Any]:
    """使用固定场景库评估一组重调度规则算法。"""

    registry = rule_registry()
    method_names = _normalize_methods(methods)
    resolved_baseline: Path | None = None
    resolved_scenario: Path | None = None
    data_path: Path | None = None
    scenario_items = []
    rows: list[dict[str, Any]] = []
    backups = {
        "enable_dynamic_events": getattr(configs, "enable_dynamic_events", False),
        "enable_station_breakdown": getattr(configs, "enable_station_breakdown", False),
        "enable_material_delay": getattr(configs, "enable_material_delay", False),
        "enable_online_duration_perturb": getattr(configs, "enable_online_duration_perturb", False),
        "enable_worker_fatigue": getattr(configs, "enable_worker_fatigue", False),
        "randomize_durations": getattr(configs, "randomize_durations", False),
        "reschedule_manifest_path": getattr(configs, "reschedule_manifest_path", ""),
        "reschedule_eval_instance_id": getattr(configs, "reschedule_eval_instance_id", ""),
        "reschedule_scenario_path": getattr(configs, "reschedule_scenario_path", ""),
        "reschedule_eval_scenario_path": getattr(configs, "reschedule_eval_scenario_path", ""),
        "reschedule_baseline_schedule_path": getattr(configs, "reschedule_baseline_schedule_path", ""),
    }
    try:
        configs.enable_dynamic_events = False
        configs.enable_station_breakdown = False
        configs.enable_material_delay = False
        configs.enable_online_duration_perturb = False
        configs.enable_worker_fatigue = False
        configs.randomize_durations = False
        configs.reschedule_manifest_path = ""
        configs.reschedule_eval_instance_id = ""
        if scenario_path is not None:
            configs.reschedule_eval_scenario_path = str(scenario_path)
            configs.reschedule_scenario_path = ""
        if baseline_path is not None:
            configs.reschedule_baseline_schedule_path = str(baseline_path)

        resolved_baseline = ensure_reschedule_baseline_available(configs)
        resolved_scenario = (
            resolve_workspace_path(scenario_path) if scenario_path is not None else ensure_reschedule_eval_scenarios_available(configs)
        )
        if resolved_baseline is None or resolved_scenario is None:
            raise RuntimeError("规则重调度评估需要 enable_reschedule_mode=True、baseline CSV 和固定场景 CSV。")

        data_path = resolve_workspace_path(data_path_or_dir or getattr(configs, "data_file_path", "data/283.csv"))
        scenario_items = load_reschedule_scenarios(Path(resolved_scenario))
        if num_runs is not None:
            scenario_items = scenario_items[: max(1, int(num_runs))]

        config_snapshot = configs.to_flat_dict()
        search_kwargs = {
            "beam_width": int(beam_width),
            "beam_branch_factor": int(beam_branch_factor),
            "beam_levels": int(beam_levels),
            "beam_patience": int(beam_patience),
            "ig_iterations": int(ig_iterations),
            "ig_destroy_ratio": float(ig_destroy_ratio),
            "ig_noise_sigma": float(ig_noise_sigma),
            "sa_iterations": int(sa_iterations),
            "sa_initial_temp": float(sa_initial_temp),
            "sa_cooling": float(sa_cooling),
            "sa_min_temp": float(sa_min_temp),
        }
        jobs: list[dict[str, Any]] = []
        order = 0
        for scenario_idx, (scenario_id, scenario) in enumerate(scenario_items):
            level = _scenario_level_from_id(scenario_id)
            for method_idx, method_name in enumerate(method_names):
                jobs.append(
                    {
                        "order": order,
                        "instance_id": "",
                        "data_path": str(data_path),
                        "baseline_path": str(resolved_baseline),
                        "scenario_path": str(resolved_scenario),
                        "scenario": scenario,
                        "scenario_idx": scenario_idx,
                        "scenario_id": scenario_id,
                        "scenario_level": level,
                        "method_idx": method_idx,
                        "method_name": method_name,
                        "seed": int(seed) + scenario_idx * 1000 + method_idx,
                        "verbose": verbose,
                        "verify_static_cache": bool(verify_static_cache),
                        "suppress_worker_stdout": True,
                        "search_kwargs": search_kwargs,
                        "config_snapshot": config_snapshot,
                    }
                )
                order += 1
        job_rows = _run_rule_jobs(
            jobs,
            parallel_workers=int(parallel_workers),
            parallel_backend=str(parallel_backend),
            output_dir=Path(output_dir) if output_dir is not None else None,
            resume=bool(resume),
            force_rerun=bool(force_rerun),
            flush_every=int(flush_every),
            show_progress=bool(show_progress),
            progress_interval=float(progress_interval),
            signature_payload={
                "mode": "manual",
                "methods": method_names,
                "seed": int(seed),
                "search_kwargs": search_kwargs,
                "verify_static_cache": bool(verify_static_cache),
                "data_path": str(data_path),
                "baseline_path": str(resolved_baseline),
                "scenario_path": str(resolved_scenario),
            },
        )
        rows = [_strip_internal_row_keys(row) for row in job_rows]
        if verbose and not show_progress:
            for row in rows:
                print(
                    f"[RuleEval] {row['method']} {row['scenario_id']} "
                    f"score={row['score']:.4f} elig={int(row.get('eligible', 0.0))} "
                    f"mk={row['makespan']:.2f} dur={row['duration_sec']:.2f}s"
                )
    finally:
        for key, value in backups.items():
            setattr(configs, key, value)

    df = pd.DataFrame(rows)
    summary_by_method = _summarize(df, ["method"])
    summary_by_method_level = _summarize(df, ["method", "scenario_level"])
    summary = {
        "baseline_path": str(Path(resolved_baseline).resolve()) if resolved_baseline is not None else "",
        "scenario_path": str(Path(resolved_scenario).resolve()) if resolved_scenario is not None else "",
        "data_path": str(Path(data_path).resolve()) if data_path is not None else "",
        "methods": method_names,
        "scenario_count": int(len(scenario_items)),
        "row_count": int(len(rows)),
        "summary_by_method": summary_by_method,
        "summary_by_method_level": summary_by_method_level,
        "rows": rows,
    }

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "reschedule_rule_eval.csv", index=False)
        pd.DataFrame(summary_by_method).to_csv(out_dir / "reschedule_rule_summary_by_method.csv", index=False)
        pd.DataFrame(summary_by_method_level).to_csv(out_dir / "reschedule_rule_summary_by_method_level.csv", index=False)
        (out_dir / "reschedule_rule_eval_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return summary


def evaluate_reschedule_rules_manifest(
    *,
    manifest_path: str | Path,
    instance_ids: Any = None,
    methods: Any = None,
    num_runs: int | None = None,
    seed: int = 42,
    output_dir: str | Path | None = None,
    verbose: bool = True,
    beam_width: int = 2,
    beam_branch_factor: int = 2,
    beam_levels: int = 2,
    beam_patience: int = 1,
    ig_iterations: int = 3,
    ig_destroy_ratio: float = 0.08,
    ig_noise_sigma: float = 0.20,
    sa_iterations: int = 3,
    sa_initial_temp: float = 0.05,
    sa_cooling: float = 0.96,
    sa_min_temp: float = 1e-4,
    parallel_workers: int = 1,
    parallel_backend: str = "process",
    verify_static_cache: bool = False,
    resume: bool = True,
    force_rerun: bool = False,
    flush_every: int = 1,
    show_progress: bool = False,
    progress_interval: float = 30.0,
) -> dict[str, Any]:
    """按 manifest 实例批量评估规则，保证与 PPO manifest 评估使用同一数据、baseline 和场景。"""

    manifest = load_reschedule_manifest(manifest_path)
    is_r5 = str(manifest.payload.get("reschedule_protocol", "")).strip() == "r5_task_delay_v1"
    ids = _as_id_list(instance_ids)
    if not ids:
        ids = [entry.instance_id for entry in manifest.filter(split="eval", source="real")]
    if not ids:
        raise ValueError("manifest 中没有可用于评估的 real/eval 实例，请显式传 instance_ids")
    if is_r5:
        validate_r5_manifest_assets(manifest)
        if tuple(ids) != REAL_INSTANCE_IDS:
            raise ValueError(f"r5 正式规则评估必须精确使用四个真实实例: {REAL_INSTANCE_IDS}")
        if num_runs is not None:
            raise ValueError("r5 正式规则评估必须读取每个真实实例的全部 9 个固定场景")

    root = Path(output_dir) if output_dir is not None else None
    method_names = _normalize_methods(methods)
    config_snapshot = configs.to_flat_dict()
    config_snapshot.update(
        {
            "enable_dynamic_events": False,
            "enable_station_breakdown": False,
            "enable_material_delay": False,
            "enable_online_duration_perturb": False,
            "enable_worker_fatigue": False,
            "randomize_durations": False,
            "reschedule_manifest_path": "",
            "reschedule_eval_instance_id": "",
            "reschedule_scenario_path": "",
        }
    )
    search_kwargs = {
        "beam_width": int(beam_width),
        "beam_branch_factor": int(beam_branch_factor),
        "beam_levels": int(beam_levels),
        "beam_patience": int(beam_patience),
        "ig_iterations": int(ig_iterations),
        "ig_destroy_ratio": float(ig_destroy_ratio),
        "ig_noise_sigma": float(ig_noise_sigma),
        "sa_iterations": int(sa_iterations),
        "sa_initial_temp": float(sa_initial_temp),
        "sa_cooling": float(sa_cooling),
        "sa_min_temp": float(sa_min_temp),
    }

    entries_by_id = {instance_id: manifest.get(instance_id) for instance_id in ids}
    scenario_counts: dict[str, int] = {}
    jobs: list[dict[str, Any]] = []
    order = 0
    for instance_order, instance_id in enumerate(ids):
        entry = entries_by_id[instance_id]
        if entry.scenario_path is None:
            raise ValueError(f"{instance_id} 没有固定场景，不能用于重调度规则可比评估")
        scenario_items = load_reschedule_scenarios(entry.scenario_path)
        if num_runs is not None:
            scenario_items = scenario_items[: max(1, int(num_runs))]
        scenario_counts[instance_id] = len(scenario_items)
        if is_r5 and scenario_counts[instance_id] != 9:
            raise ValueError(f"r5 实例 {instance_id} 必须恰好包含 9 个场景")
        for scenario_idx, (scenario_id, scenario) in enumerate(scenario_items):
            level = _scenario_level_from_id(scenario_id)
            stage = _scenario_stage_from_id(scenario_id)
            for method_idx, method_name in enumerate(method_names):
                for solver_seed_index in range(_solver_seed_count(method_name, r5=is_r5)):
                    jobs.append(
                        {
                            "order": order,
                            "instance_order": instance_order,
                            "instance_id": instance_id,
                            "data_path": str(entry.data_path),
                            "baseline_path": str(entry.baseline_schedule_path),
                            "scenario_path": str(entry.scenario_path),
                            "scenario": scenario,
                            "scenario_idx": scenario_idx,
                            "scenario_id": scenario_id,
                            "scenario_level": level,
                            "scenario_stage": stage,
                            "method_idx": method_idx,
                            "method_name": method_name,
                            "seed": int(seed) + instance_order * 10000 + scenario_idx * 1000 + method_idx * 10 + solver_seed_index,
                            "solver_seed_index": solver_seed_index,
                            "verbose": verbose,
                            "verify_static_cache": bool(verify_static_cache),
                            "suppress_worker_stdout": True,
                            "search_kwargs": search_kwargs,
                            "config_snapshot": config_snapshot,
                        }
                    )
                    order += 1

    job_rows = _run_rule_jobs(
        jobs,
        parallel_workers=int(parallel_workers),
        parallel_backend=str(parallel_backend),
        output_dir=root,
        resume=bool(resume),
        force_rerun=bool(force_rerun),
        flush_every=int(flush_every),
        show_progress=bool(show_progress),
        progress_interval=float(progress_interval),
        signature_payload={
            "mode": "manifest",
            "manifest_path": str(resolve_workspace_path(manifest_path).resolve()),
            "instance_ids": ids,
            "methods": method_names,
            "seed": int(seed),
            "search_kwargs": search_kwargs,
            "verify_static_cache": bool(verify_static_cache),
            "solver_seed_policy": "r5_stochastic_methods_three_seeds" if is_r5 else "single_seed",
        },
    )

    instance_summaries: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    for instance_id in ids:
        entry = entries_by_id[instance_id]
        local_job_rows = [row for row in job_rows if row["_instance_id"] == instance_id]
        local_rows = [_strip_internal_row_keys(row) for row in local_job_rows]
        df = pd.DataFrame(local_rows)
        summary_by_method = _summarize(df, ["method"])
        summary_by_method_level = _summarize(df, ["method", "scenario_level"])
        summary = {
            "baseline_path": str(entry.baseline_schedule_path.resolve()),
            "scenario_path": str(entry.scenario_path.resolve()) if entry.scenario_path is not None else "",
            "data_path": str(entry.data_path.resolve()),
            "methods": method_names,
            "scenario_count": int(scenario_counts[instance_id]),
            "row_count": int(len(local_rows)),
            "summary_by_method": summary_by_method,
            "summary_by_method_level": summary_by_method_level,
            "rows": local_rows,
        }
        instance_summaries[instance_id] = summary
        if root is not None:
            subdir = root / instance_id
            subdir.mkdir(parents=True, exist_ok=True)
            df.to_csv(subdir / "reschedule_rule_eval.csv", index=False)
            pd.DataFrame(summary_by_method).to_csv(subdir / "reschedule_rule_summary_by_method.csv", index=False)
            pd.DataFrame(summary_by_method_level).to_csv(subdir / "reschedule_rule_summary_by_method_level.csv", index=False)
            (subdir / "reschedule_rule_eval_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        for row in local_rows:
            enriched = dict(row)
            enriched["instance_id"] = instance_id
            enriched["data_path"] = str(entry.data_path)
            enriched["baseline_path"] = str(entry.baseline_schedule_path)
            enriched["scenario_path"] = str(entry.scenario_path)
            rows.append(enriched)
        for row in summary_by_method:
            enriched = dict(row)
            enriched["instance_id"] = instance_id
            enriched["data_path"] = str(entry.data_path)
            enriched["baseline_path"] = str(entry.baseline_schedule_path)
            enriched["scenario_path"] = str(entry.scenario_path)
            method_rows.append(enriched)
        if verbose and not show_progress:
            for row in local_rows:
                print(
                    f"[RuleEval] {instance_id} {row['method']} {row['scenario_id']} "
                    f"score={row['score']:.4f} elig={int(row.get('eligible', 0.0))} "
                    f"mk={row['makespan']:.2f} dur={row['duration_sec']:.2f}s"
                )

    solver_seed_summary = _summarize_solver_seeds(pd.DataFrame(rows))
    if is_r5 and sum(scenario_counts.values()) != 36:
        raise ValueError("r5 正式规则评估必须恰好包含 36 个场景")
    payload = {
        "manifest_path": str(resolve_workspace_path(manifest_path).resolve()),
        "instance_ids": ids,
        "methods": method_names,
        "scenario_count": int(sum(scenario_counts.values())),
        "row_count": len(rows),
        "solver_run_count": len(rows),
        "solver_seed_policy": "r5_stochastic_methods_three_seeds" if is_r5 else "single_seed",
        "rows": rows,
        "summary_by_instance_method": method_rows,
        "solver_seed_summary": solver_seed_summary,
        "instances": instance_summaries,
    }
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(root / "reschedule_rule_eval_by_instance.csv", index=False)
        pd.DataFrame(method_rows).to_csv(root / "reschedule_rule_summary_by_instance_method.csv", index=False)
        pd.DataFrame(solver_seed_summary).to_csv(root / "reschedule_rule_solver_seed_summary.csv", index=False)
        (root / "reschedule_rule_manifest_summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return payload


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if should_show_help(raw_args):
        print(hydra_help(RULE_EXTRA_ARGS))
        return 0
    try:
        args = initialize_hydra_runtime(
            raw_args,
            target=configs,
            project_root=PROJECT_ROOT,
            default_experiment="reschedule_task_delay",
            extra_arguments=RULE_EXTRA_ARGS,
        )
    except (HydraCliError, KeyError, ValueError, RuntimeError) as exc:
        print(f"[CLI] {exc}", file=sys.stderr)
        return 2

    output_dir, context = resolve_run_output_dir(
        configs,
        PROJECT_ROOT,
        default_legacy_dir="results/reschedule_rules",
        run_subdir="reschedule_rules",
        explicit_dir=args.output_dir,
        section="eval",
    )
    manifest_extra = {
        "run_type": "evaluation",
        "artifact_kind": "reschedule_rules",
        "output_dir": str(output_dir.resolve()),
        "manifest_path": str(args.manifest_path or ""),
        "instance_ids": args.instance_ids,
        "parallel_workers": int(args.parallel_workers),
        "parallel_backend": str(args.parallel_backend),
        "verify_static_cache": bool(args.verify_static_cache),
        "resume_partial": bool(args.resume_partial),
        "force_rerun": bool(args.force_rerun),
        "flush_every": int(args.flush_every),
        "progress_interval": float(args.progress_interval),
    }
    if context is not None:
        write_run_context_files(context, configs, command="evaluate_reschedule_rules", extra=manifest_extra)
    else:
        write_run_manifest(output_dir, configs, command="evaluate_reschedule_rules", extra=manifest_extra)

    common_kwargs = {
        "methods": args.methods,
        "num_runs": args.num_runs,
        "seed": int(args.seed),
        "output_dir": output_dir,
        "verbose": not bool(args.quiet),
        "beam_width": int(args.beam_width),
        "beam_branch_factor": int(args.beam_branch_factor),
        "beam_levels": int(args.beam_levels),
        "beam_patience": int(args.beam_patience),
        "ig_iterations": int(args.ig_iterations),
        "ig_destroy_ratio": float(args.ig_destroy_ratio),
        "ig_noise_sigma": float(args.ig_noise_sigma),
        "sa_iterations": int(args.sa_iterations),
        "sa_initial_temp": float(args.sa_initial_temp),
        "sa_cooling": float(args.sa_cooling),
        "sa_min_temp": float(args.sa_min_temp),
        "parallel_workers": int(args.parallel_workers),
        "parallel_backend": str(args.parallel_backend),
        "verify_static_cache": bool(args.verify_static_cache),
        "resume": bool(args.resume_partial),
        "force_rerun": bool(args.force_rerun),
        "flush_every": int(args.flush_every),
        "show_progress": not bool(args.quiet),
        "progress_interval": float(args.progress_interval),
    }
    try:
        if args.manifest_path:
            summary = evaluate_reschedule_rules_manifest(
                manifest_path=args.manifest_path,
                instance_ids=args.instance_ids,
                **common_kwargs,
            )
        else:
            summary = evaluate_reschedule_rules(
                data_path_or_dir=args.data_path,
                scenario_path=args.scenario_path,
                baseline_path=args.baseline_path,
                **common_kwargs,
            )
    except KeyboardInterrupt:
        print(f"\n[Interrupted] 已保存断点到: {output_dir / PARTIAL_FILE_NAME}", file=sys.stderr)
        return 130
    print(json.dumps(_console_summary(summary), ensure_ascii=False, indent=2))
    detail_name = "reschedule_rule_eval_by_instance.csv" if args.manifest_path else "reschedule_rule_eval.csv"
    print(f"规则重调度评估明细已保存到: {output_dir / detail_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
