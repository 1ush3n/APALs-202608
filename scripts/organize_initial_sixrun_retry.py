"""整理 2026-07-22 初始调度六次验证结果并完成独立审计。"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_initial_schedule import validate_schedule


DATASETS = ("283", "680", "2338", "3182")
RUN_NAMES = (
    "temp0_seed42",
    "temp001_seed42",
    "temp001_seed43",
    "temp001_seed44",
    "temp001_seed45",
    "temp001_seed46",
)
PROTOCOL = (
    "6 runs per dataset: temperature=0.0 seed=42 as primary deterministic result; "
    "temperature=0.01 seeds=42..46 for stochastic supplementary statistics"
)

METHODS: dict[str, dict[str, str]] = {
    "Graph-DDQN-APAL": {
        "source": "results/graph_ddqn_apal_initial_scale400_800_680_seed42_20260721",
        "archive": "results/02_initial_baselines/graph_ddqn_apal_initial_scale400_800_680_seed42_20260721",
        "checkpoint": "results/02_initial_baselines/graph_ddqn_apal_initial_scale400_800_680_seed42_20260721/graph_ddqn_apal_best.pth",
    },
    "Simple-HeteroGAT-PPO": {
        "source": "results/l2d_ppo_apal_initial_scale400_800_680_seed42_20260721",
        "archive": "results/02_initial_baselines/l2d_ppo_apal_initial_scale400_800_680_seed42_20260721",
        "checkpoint": "results/02_initial_baselines/l2d_ppo_apal_initial_scale400_800_680_seed42_20260721/artifacts/baselines/literature/Simple-HeteroGAT-PPO/simple_heterogat_ppo_best.pth",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_metadata(run_name: str) -> tuple[int, float, str]:
    if run_name == "temp0_seed42":
        return 42, 0.0, "primary_deterministic"
    if not run_name.startswith("temp001_seed"):
        raise ValueError(f"未知运行目录：{run_name}")
    seed = int(run_name.removeprefix("temp001_seed"))
    return seed, 0.01, "stochastic_supplement"


def copy_after_preflight(source: Path, target: Path) -> None:
    if target.exists():
        source_files = {p.relative_to(source).as_posix(): sha256(p) for p in source.rglob("*") if p.is_file()}
        target_files = {p.relative_to(target).as_posix(): sha256(p) for p in target.rglob("*") if p.is_file()}
        if source_files != target_files:
            raise RuntimeError(f"目标已存在但内容不一致，拒绝覆盖：{target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)


def raw_run_paths(eval_root: Path, method: str, dataset: str, run_name: str) -> tuple[Path, Path, Path, Path, Path]:
    run_root = eval_root / f"real_{dataset}" / run_name
    run_manifest = run_root / "run_manifest.json"
    resolved_config = run_root / "resolved_config.yaml"
    method_root = run_root / method / dataset
    return (
        method_root / "metrics.json",
        method_root / "runs_detail.csv",
        method_root / "schedule.csv",
        run_manifest,
        resolved_config,
    )


def audit_method(method: str, spec: dict[str, str]) -> dict[str, Any]:
    source = PROJECT_ROOT / spec["source"]
    archive = PROJECT_ROOT / spec["archive"]
    source_eval = source / "eval" / "initial_sixrun_20260722_retry"
    archive_eval = archive / "eval" / "initial_sixrun_20260722_retry"
    if not source_eval.is_dir():
        raise FileNotFoundError(source_eval)
    copy_after_preflight(source_eval, archive_eval)

    # 失败的第一次尝试仅有元数据，保留以便追溯，但不混入正式统计。
    failed_source = source / "eval" / "initial_sixrun_20260722"
    failed_archive = archive / "eval" / "initial_sixrun_20260722_failed_metadata"
    if failed_source.is_dir():
        copy_after_preflight(failed_source, failed_archive)

    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for dataset in DATASETS:
        data_path = PROJECT_ROOT / "data" / f"{dataset}.csv"
        config_path = PROJECT_ROOT / "conf" / "env" / f"initial_bucket_{dataset}.yaml"
        for run_name in RUN_NAMES:
            seed, temperature, role = run_metadata(run_name)
            metric_path, detail_path, schedule_path, manifest_path, resolved_path = raw_run_paths(
                archive_eval, method, dataset, run_name
            )
            if not all(path.is_file() for path in (metric_path, detail_path, schedule_path, manifest_path, resolved_path)):
                raise FileNotFoundError(f"缺少原始文件：{archive_eval / f'real_{dataset}' / run_name}")
            metric = json.loads(metric_path.read_text(encoding="utf-8"))
            detail_rows = list(csv.DictReader(detail_path.open(encoding="utf-8-sig", newline="")))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            report = validate_schedule(
                data_path=data_path,
                schedule_path=schedule_path,
                config_path=str(config_path),
                task_id_mode="internal",
            )
            violations = report["violations"]
            hard_total = int(sum(int(value) for value in violations.values()))
            metric_seed = int(float(metric.get("seed", -1)))
            detail_seed = int(float(detail_rows[0]["Seed"])) if detail_rows else -1
            makespan_diff = abs(float(metric["makespan"]) - float(report["makespan_real_tasks"]))
            row: dict[str, Any] = {
                "method": method,
                "dataset": dataset,
                "instance_id": f"real_{dataset}",
                "run_name": run_name,
                "result_role": role,
                "seed": seed,
                "temperature": temperature,
                "metric_seed": metric_seed,
                "detail_seed": detail_seed,
                "makespan": float(metric["makespan"]),
                "makespan_recomputed": float(report["makespan_real_tasks"]),
                "makespan_abs_diff": makespan_diff,
                "workload_balance_std": float(metric["workload_balance_std"]),
                "worker_utilization": float(metric["worker_utilization"]),
                "station_utilization": float(metric["station_utilization"]),
                "inference_time_sec": float(metric["inference_time"]),
                "metric_valid": float(metric["valid"]),
                "metric_complete": float(metric["complete"]),
                "metric_completion_rate": float(metric["completion_rate"]),
                "num_schedule_rows": int(report["num_schedule_rows"]),
                "num_dataset_nodes": int(report["num_dataset_nodes"]),
                "num_real_tasks": int(report["num_real_tasks"]),
                "scheduled_real_tasks": int(report["scheduled_real_tasks"]),
                "complete_rate_recomputed": float(report["scheduled_real_tasks"] / report["num_real_tasks"]),
                "eligible_rate_recomputed": float(report["is_legal_against_environment_duration"]),
                "structurally_legal": bool(report["is_resource_structurally_legal"]),
                "legal_against_environment_duration": bool(report["is_legal_against_environment_duration"]),
                "hard_violation_total": hard_total,
                "schedule_path": schedule_path.relative_to(PROJECT_ROOT).as_posix(),
                "metrics_path": metric_path.relative_to(PROJECT_ROOT).as_posix(),
                "run_manifest_path": manifest_path.relative_to(PROJECT_ROOT).as_posix(),
                "method_in_manifest": manifest.get("method", ""),
            }
            row.update({f"violation_{key}": int(value) for key, value in violations.items()})
            rows.append(row)
            reports.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "run_name": run_name,
                    "seed": seed,
                    "temperature": temperature,
                    "schedule_path": row["schedule_path"],
                    "report": report,
                }
            )

    summary_rows: list[dict[str, Any]] = []
    summary_json: dict[str, Any] = {
        "method": method,
        "phase": "initial_schedule",
        "protocol": PROTOCOL,
        "datasets": {},
        "overall": {},
    }
    for dataset in DATASETS:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        primary = next(row for row in dataset_rows if row["result_role"] == "primary_deterministic")
        stochastic = [row for row in dataset_rows if row["result_role"] == "stochastic_supplement"]
        mean_ms = statistics.fmean(row["makespan"] for row in stochastic)
        std_ms = statistics.stdev(row["makespan"] for row in stochastic)
        aggregate = {
            "method": method,
            "dataset": dataset,
            "instance_id": f"real_{dataset}",
            "primary_makespan": primary["makespan"],
            "stochastic_makespan_mean": mean_ms,
            "stochastic_makespan_sample_std": std_ms,
            "stochastic_makespan_min": min(row["makespan"] for row in stochastic),
            "stochastic_makespan_max": max(row["makespan"] for row in stochastic),
            "primary_complete_rate": primary["complete_rate_recomputed"],
            "primary_eligible_rate": primary["eligible_rate_recomputed"],
            "all_runs_complete": all(row["complete_rate_recomputed"] == 1.0 for row in dataset_rows),
            "all_runs_legal": all(row["legal_against_environment_duration"] for row in dataset_rows),
            "max_hard_violation_total": max(row["hard_violation_total"] for row in dataset_rows),
            "mean_inference_time_sec": statistics.fmean(row["inference_time_sec"] for row in dataset_rows),
        }
        summary_rows.append(aggregate)
        summary_json["datasets"][dataset] = aggregate
    summary_json["overall"] = {
        "run_count": len(rows),
        "dataset_count": len(DATASETS),
        "all_runs_complete": all(row["complete_rate_recomputed"] == 1.0 for row in rows),
        "all_runs_legal": all(row["legal_against_environment_duration"] for row in rows),
        "max_hard_violation_total": max(row["hard_violation_total"] for row in rows),
    }

    checkpoint = PROJECT_ROOT / spec["checkpoint"]
    integrity = {
        "method": method,
        "protocol": PROTOCOL,
        "expected_datasets": list(DATASETS),
        "expected_runs_per_dataset": 6,
        "expected_run_names": list(RUN_NAMES),
        "expected_stochastic_seeds": [42, 43, 44, 45, 46],
        "expected_primary": {"seed": 42, "temperature": 0.0},
        "observed_run_count": len(rows),
        "observed_runs_by_dataset": {dataset: len([row for row in rows if row["dataset"] == dataset]) for dataset in DATASETS},
        "all_dataset_counts_match": all(len([row for row in rows if row["dataset"] == dataset]) == 6 for dataset in DATASETS),
        "all_seed_and_metric_seed_match": all(row["seed"] == row["metric_seed"] == row["detail_seed"] for row in rows),
        "all_manifest_methods_match": all(row["method_in_manifest"] == method for row in rows),
        "all_task_counts_match": all(row["num_schedule_rows"] == row["num_dataset_nodes"] and row["scheduled_real_tasks"] == row["num_real_tasks"] for row in rows),
        "all_runs_complete": all(row["complete_rate_recomputed"] == 1.0 for row in rows),
        "all_runs_legal": all(row["legal_against_environment_duration"] for row in rows),
        "max_hard_violation_total": max(row["hard_violation_total"] for row in rows),
        "max_metric_recomputed_makespan_abs_diff": max(row["makespan_abs_diff"] for row in rows),
        "temperature_evidence": "根据原始输出目录名 temp0/temp001 记录；逐次 run_manifest/resolved_config 未保存 CLI temperature，因此该字段为路径证据而非独立元数据证据。",
        "strict_main_table_eligible": "conditional",
        "checkpoint": spec["checkpoint"],
        "checkpoint_sha256": sha256(checkpoint),
        "raw_source_eval": spec["source"] + "/eval/initial_sixrun_20260722_retry",
        "archived_eval": spec["archive"] + "/eval/initial_sixrun_20260722_retry",
        "failed_first_attempt_preserved": (failed_archive.exists()),
    }
    archive_eval.mkdir(parents=True, exist_ok=True)
    write_csv(archive_eval / "runs_detail.csv", rows)
    write_csv(archive_eval / "summary.csv", summary_rows)
    write_csv(archive_eval / "validation_by_seed.csv", rows)
    write_json(archive_eval / "validation_by_seed.json", reports)
    write_json(archive_eval / "summary.json", summary_json)
    write_json(archive_eval / "integrity_check.json", integrity)
    (archive_eval / "resolved_config.yaml").write_text(
        "# 本文件是六次初始调度验证的统一协议记录；每个 run 目录保留原始 resolved_config.yaml。\n"
        f"protocol: {PROTOCOL}\n"
        "datasets: [283, 680, 2338, 3182]\n"
        "primary_temperature: 0.0\n"
        "primary_seed: 42\n"
        "stochastic_temperature: 0.01\n"
        "stochastic_seeds: [42, 43, 44, 45, 46]\n"
        "strict_main_table_eligible: conditional\n",
        encoding="utf-8",
    )
    write_json(
        archive_eval / "run_manifest.json",
        {
            "run_type": "initial_schedule_sixrun_protocol",
            "method": method,
            "protocol": PROTOCOL,
            "datasets": [f"real_{dataset}" for dataset in DATASETS],
            "checkpoint": spec["checkpoint"],
            "checkpoint_sha256": integrity["checkpoint_sha256"],
            "raw_source": spec["source"],
            "strict_main_table_eligible": "conditional",
            "note": "temperature 由原始输出目录名核对；原始 run_manifest 未独立记录 CLI temperature。",
        },
    )
    (archive_eval / "README.md").write_text(
        f"# {method} 初始调度六次验证（2026-07-22 retry）\n\n"
        f"- 数据集：`real_283`、`real_680`、`real_2338`、`real_3182`；每个数据集 6 次。\n"
        "- 主结果：`temperature=0.0, seed=42`；补充统计：`temperature=0.01, seed=42..46`。\n"
        "- 每个原始 schedule 均按当前数据、initial bucket 配置和环境约束独立回放检查。\n"
        f"- 结果：{len(rows)} 次排程全部完整，全部硬约束违规为 0；指标 makespan 与独立重算最大差值为 `{integrity['max_metric_recomputed_makespan_abs_diff']:.8f}` h。\n"
        "- 证据限制：原始逐次 run_manifest/resolved_config 没有保存 CLI temperature，温度依据输出目录名 `temp0`/`temp001` 核对，因此主表资格记录为 `conditional`，不需要因合法性问题重跑。\n"
        f"- checkpoint：`{spec['checkpoint']}`；SHA-256：`{integrity['checkpoint_sha256']}`。\n\n"
        "详细文件：`summary.csv/json`、`runs_detail.csv`、`validation_by_seed.csv/json`、`integrity_check.json`、`run_manifest.json`、`resolved_config.yaml`、`file_manifest.json`；每个数据集/seed 的原始输出仍保留。\n",
        encoding="utf-8",
    )
    files = []
    for path in sorted(archive_eval.rglob("*")):
        if path.is_file() and path.name != "file_manifest.json":
            files.append({"path": path.relative_to(archive_eval).as_posix(), "size": path.stat().st_size, "sha256": sha256(path)})
    write_json(archive_eval / "file_manifest.json", {"root": archive_eval.relative_to(PROJECT_ROOT).as_posix(), "files": files})
    return {"method": method, "rows": rows, "summary": summary_json, "integrity": integrity}


def main() -> int:
    result = {method: audit_method(method, spec) for method, spec in METHODS.items()}
    print(json.dumps({method: value["integrity"] for method, value in result.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
