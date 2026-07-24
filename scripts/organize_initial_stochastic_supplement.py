"""整理初始调度 temperature=0.01、seed=42..46 的补充验证结果。"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_initial_schedule import validate_schedule

DATASETS = ("283", "680", "2338", "3182")
SEEDS = (42, 43, 44, 45, 46)

METHODS: dict[str, dict[str, str]] = {
    "full_joint": {
        "source": "results/joint100_full_joint_seed42_20260719",
        "archive": "results/01_initial_main_stochastic_20260723/joint100_full_joint_seed42_20260719",
        "display": "HB-GAT-PPO full-joint",
        "variant": "full_joint",
        "checkpoint": "results/01_initial_main/joint100_full_joint_seed42_20260719/checkpoints/best.ckpt",
    },
    "mean_max_pooling": {
        "source": "results/ablation_joint100_mean_max_pooling_seed42_20260720",
        "archive": "results/01_initial_main_stochastic_20260723/ablation_joint100_mean_max_pooling_seed42_20260720",
        "display": "HB-GAT-PPO 消融：mean-max pooling",
        "variant": "mean_max_pooling",
        "checkpoint": "results/01_initial_main/ablation_joint100_mean_max_pooling_seed42_20260720/checkpoints/best.ckpt",
    },
    "operation_only": {
        "source": "results/ablation_joint100_operation_only_seed42_20260720",
        "archive": "results/01_initial_main_stochastic_20260723/ablation_joint100_operation_only_seed42_20260720",
        "display": "HB-GAT-PPO 消融：operation-only",
        "variant": "operation_only",
        "checkpoint": "results/01_initial_main/ablation_joint100_operation_only_seed42_20260720/checkpoints/best.ckpt",
    },
    "operation_station": {
        "source": "results/ablation_joint100_operation_station_seed42_20260720",
        "archive": "results/01_initial_main_stochastic_20260723/ablation_joint100_operation_station_seed42_20260720",
        "display": "HB-GAT-PPO 消融：operation-station",
        "variant": "operation_station",
        "checkpoint": "results/01_initial_main/ablation_joint100_operation_station_seed42_20260720/checkpoints/best.ckpt",
    },
    "fixed_preallocation": {
        "source": "results/ablation_joint100_fixed_preallocation_seed42_20260720",
        "archive": "results/01_initial_main_stochastic_20260723/ablation_joint100_fixed_preallocation_seed42_20260720",
        "display": "HB-GAT-PPO 消融：fixed-preallocation",
        "variant": "fixed_preallocation",
        "checkpoint": "results/01_initial_main/ablation_joint100_fixed_preallocation_seed42_20260720/checkpoints/best.ckpt",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def copy_and_hash(source: Path, target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    # 允许脚本在中断后续跑：源文件必须逐一同哈希，目标可以包含本脚本生成的审计文件。
    if not target.exists():
        shutil.copytree(source, target)
    else:
        shutil.copytree(source, target, dirs_exist_ok=True)
    source_map = {p.relative_to(source).as_posix(): sha256(p) for p in source.rglob("*") if p.is_file()}
    target_map = {p.relative_to(target).as_posix(): sha256(p) for p in target.rglob("*") if p.is_file()}
    return {
        "source": source.resolve().as_posix(),
        "archive": target.resolve().as_posix(),
        "source_file_count": len(source_map),
        "archive_file_count": len(target_map),
        "all_sha256_equal": all(target_map.get(key) == value for key, value in source_map.items()),
    }


def main() -> int:
    generated_at = datetime.now().astimezone().isoformat()
    all_rows: list[dict[str, Any]] = []
    overall: list[dict[str, Any]] = []
    for name, spec in METHODS.items():
        source = ROOT / spec["source"] / "eval" / "initial_sixrun_20260722_stochastic"
        archive = ROOT / spec["archive"] / "eval" / "initial_sixrun_20260722_stochastic"
        if not source.is_dir():
            raise FileNotFoundError(source)
        copy_audit = copy_and_hash(source, archive)
        rows: list[dict[str, Any]] = []
        for dataset in DATASETS:
            for seed in SEEDS:
                run = archive / f"real_{dataset}" / f"temp001_seed{seed}"
                schedule = run / "schedule.csv"
                summary_path = run / "summary.json"
                if not schedule.is_file() or not summary_path.is_file():
                    raise FileNotFoundError(f"缺少 schedule.csv 或 summary.json：{run}")
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                report = validate_schedule(
                    data_path=ROOT / "data" / f"{dataset}.csv",
                    schedule_path=schedule,
                    config_path=str(ROOT / "conf" / "env" / f"initial_bucket_{dataset}.yaml"),
                    task_id_mode="internal",
                )
                violations = {key: int(value) for key, value in report["violations"].items()}
                row: dict[str, Any] = {
                    "method": spec["display"],
                    "variant": spec["variant"],
                    "dataset": dataset,
                    "instance_id": f"real_{dataset}",
                    "run_name": f"temp001_seed{seed}",
                    "temperature": 0.01,
                    "seed": seed,
                    "scheduled_tasks_reported": int(summary["scheduled_tasks"]),
                    "scheduled_tasks_recomputed": int(report["scheduled_real_tasks"]),
                    "dataset_nodes": int(report["num_dataset_nodes"]),
                    "num_real_tasks": int(report["num_real_tasks"]),
                    "makespan_reported": float(summary["makespan"]),
                    "makespan_recomputed": float(report["makespan_real_tasks"]),
                    "makespan_abs_diff": abs(float(summary["makespan"]) - float(report["makespan_real_tasks"])),
                    "balance_std": float(summary["balance_std"]),
                    "reward": float(summary["reward"]),
                    "worker_utilization": float(summary["worker_utilization"]),
                    "station_utilization": float(summary["station_utilization"]),
                    "duration_sec": float(summary["duration_sec"]),
                    "complete_rate": float(report["scheduled_real_tasks"] / report["num_real_tasks"]),
                    "eligible_rate": float(report["is_legal_against_environment_duration"]),
                    "structurally_legal": bool(report["is_resource_structurally_legal"]),
                    "hard_violation_total": sum(violations.values()),
                    "schedule_path": schedule.relative_to(ROOT).as_posix(),
                    "summary_path": summary_path.relative_to(ROOT).as_posix(),
                }
                row.update({f"violation_{key}": value for key, value in violations.items()})
                rows.append(row)
                all_rows.append(row)
                write_json(run / "legality_audit.json", report)

        summary_rows: list[dict[str, Any]] = []
        for dataset in DATASETS:
            subset = [r for r in rows if r["dataset"] == dataset]
            values = [float(r["makespan_recomputed"]) for r in subset]
            summary_rows.append({
                "method": spec["display"],
                "variant": spec["variant"],
                "dataset": dataset,
                "run_count": len(subset),
                "seed_min": min(r["seed"] for r in subset),
                "seed_max": max(r["seed"] for r in subset),
                "temperature": 0.01,
                "makespan_mean": statistics.fmean(values),
                "makespan_sample_std": statistics.stdev(values),
                "makespan_min": min(values),
                "makespan_max": max(values),
                "complete_rate": min(r["complete_rate"] for r in subset),
                "eligible_rate": min(r["eligible_rate"] for r in subset),
                "max_hard_violation_total": max(r["hard_violation_total"] for r in subset),
                "mean_duration_sec": statistics.fmean(float(r["duration_sec"]) for r in subset),
            })
        write_csv(archive / "runs_detail_stochastic.csv", rows)
        write_csv(archive / "summary_stochastic.csv", summary_rows)
        integrity = {
            "generated_at": generated_at,
            "method": spec["display"],
            "variant": spec["variant"],
            "protocol": "temperature=0.01; seeds=42,43,44,45,46; 4 datasets",
            "expected_dataset_count": 4,
            "expected_runs_per_dataset": 5,
            "observed_run_count": len(rows),
            "observed_runs_by_dataset": {d: sum(r["dataset"] == d for r in rows) for d in DATASETS},
            "all_dataset_counts_match": all(sum(r["dataset"] == d for r in rows) == 5 for d in DATASETS),
            "all_seed_sets_match": all(sorted(r["seed"] for r in rows if r["dataset"] == d) == list(SEEDS) for d in DATASETS),
            "all_complete": all(r["complete_rate"] == 1.0 for r in rows),
            "all_eligible": all(r["eligible_rate"] == 1.0 for r in rows),
            "all_structurally_legal": all(r["structurally_legal"] for r in rows),
            "max_hard_violation_total": max(r["hard_violation_total"] for r in rows),
            "max_makespan_recompute_abs_diff": max(r["makespan_abs_diff"] for r in rows),
            "checkpoint": spec["checkpoint"],
            "checkpoint_sha256": sha256(ROOT / spec["checkpoint"]),
            "copy_audit": copy_audit,
            "temperature_evidence": "原始逐次输出目录名 temp001_seed42..46；summary.json 未单独保存 CLI temperature，因此温度是目录证据而非 run_manifest 独立字段。",
            "strict_main_table_eligible": "conditional",
        }
        write_json(archive / "integrity_check_stochastic.json", integrity)
        write_json(archive / "run_manifest_stochastic.json", {
            "run_type": "initial_schedule_stochastic_supplement",
            "method": spec["display"],
            "variant": spec["variant"],
            "protocol": "temperature=0.01; seeds=42..46",
            "datasets": [f"real_{d}" for d in DATASETS],
            "checkpoint": spec["checkpoint"],
            "checkpoint_sha256": integrity["checkpoint_sha256"],
            "source_root": spec["source"],
            "strict_main_table_eligible": "conditional",
        })
        (archive / "resolved_config_stochastic.yaml").write_text(
            "protocol: initial_schedule_stochastic_supplement\n"
            "temperature: 0.01\nseeds: [42, 43, 44, 45, 46]\n"
            "datasets: [283, 680, 2338, 3182]\n"
            "primary_deterministic_result: results/01_initial_main/unified_eval_parallel_20260720_214219\n"
            "strict_main_table_eligible: conditional\n",
            encoding="utf-8",
        )
        (archive / "README_stochastic.md").write_text(
            f"# {spec['display']} 初始调度随机补充验证\n\n"
            "本目录保存 temperature=0.01、seed=42–46 在 real_283/680/2338/3182 上的 20 个原始逐次结果。"
            "temperature=0.0、seed=42 的主结果仍位于统一确定性验证目录，并在本目录的汇总中作为关联证据。\n\n"
            f"完整性：{len(rows)}/20 个 schedule，complete_rate={integrity['all_complete']}，eligible_rate={integrity['all_eligible']}，"
            f"最大硬约束违规总数={integrity['max_hard_violation_total']}。\n"
            "由于原始 summary 未独立记录 CLI temperature，主表证据等级为 conditional；不得将本批随机补充与确定性主结果混作同一运行。\n",
            encoding="utf-8",
        )
        file_entries = []
        for file in sorted(archive.rglob("*")):
            if file.is_file() and file.name != "file_manifest_stochastic.json":
                file_entries.append({"path": file.relative_to(archive).as_posix(), "size": file.stat().st_size, "sha256": sha256(file)})
        write_json(archive / "file_manifest_stochastic.json", {"generated_at": generated_at, "files": file_entries})
        overall.extend(summary_rows)

    write_csv(ROOT / "results/01_initial_main_stochastic_20260723/initial_stochastic_supplement_summary.csv", overall)
    write_json(ROOT / "results/01_initial_main_stochastic_20260723/initial_stochastic_supplement_summary.json", {
        "generated_at": generated_at,
        "protocol": "temperature=0.01; seeds=42..46; 4 datasets; 5 methods",
        "rows": overall,
        "total_schedules": len(all_rows),
    })
    write_csv(ROOT / "results/01_initial_main_stochastic_20260723/initial_stochastic_supplement_runs_detail.csv", all_rows)
    print(json.dumps({"methods": len(METHODS), "datasets": len(DATASETS), "runs": len(all_rows), "summary_rows": len(overall)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
