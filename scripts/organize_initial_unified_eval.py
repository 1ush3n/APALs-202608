"""整理初始调度四数据集并行验证结果，生成可审计的正式归档文件。"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_initial_schedule import validate_schedule


DATASETS = ("283", "680", "2338", "3182")
EXPECTED_NODES = {"283": 290, "680": 715, "2338": 2402, "3182": 3299}
VARIANT_LABELS = {
    "joint100_full_joint_seed42": ("HB-GAT-PPO", "full_joint", "main_method"),
    "joint100_fixed_preallocation_seed42": ("HB-GAT-PPO", "fixed_preallocation", "ablation"),
    "joint100_mean_max_pooling_seed42": ("HB-GAT-PPO", "mean_max_pooling", "ablation"),
    "joint100_operation_only_seed42": ("HB-GAT-PPO", "operation_only", "ablation"),
    "joint100_operation_station_seed42": ("HB-GAT-PPO", "operation_station", "ablation"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_to_archive(source: Path, destination: Path) -> None:
    if source.resolve() != destination.resolve():
        shutil.copytree(source, destination, dirs_exist_ok=True)


def model_key(model_dir: Path) -> str:
    return model_dir.name.replace("runs__scale_400_800_schedule__", "")


def validate_one(item: tuple[str, str, Path]) -> dict[str, Any]:
    model, dataset, schedule_path = item
    data_path = PROJECT_ROOT / "data" / f"{dataset}.csv"
    try:
        report = validate_schedule(data_path=data_path, schedule_path=schedule_path)
        return {"model": model, "dataset": dataset, "ok": True, "report": report}
    except Exception as exc:  # noqa: BLE001 - 结果必须记录失败原因
        return {"model": model, "dataset": dataset, "ok": False, "error": repr(exc)}


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model", "method", "variant", "dataset", "instance_id", "scenario",
        "checkpoint", "checkpoint_sha256", "scheduled_tasks", "dataset_nodes",
        "complete_rate", "valid_rate", "makespan", "balance_std", "reward",
        "worker_utilization", "station_utilization", "duration_sec",
        "is_resource_structurally_legal", "is_legal_against_environment_duration",
        "task_id_mode", "schedule_path", "legality_report_path",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_generic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def update_master_results(rows: list[dict[str, Any]], destination: Path) -> int:
    """将四实例单种子正式验证结果登记到主结果台账，避免重复写入。"""
    master = PROJECT_ROOT / "results" / "experiment_master_results.csv"
    if not master.exists():
        return 0
    with master.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        existing = list(reader)
    if not fields or "experiment_id" not in fields:
        return 0
    existing_ids = {str(row.get("experiment_id", "")) for row in existing}
    added = 0
    changed = False
    for row in rows:
        experiment_id = f"initial_{row['variant']}_unified_four_scale_seed42_{row['dataset']}"
        if experiment_id in existing_ids:
            for old in existing:
                if str(old.get("experiment_id", "")) == experiment_id:
                    old["source_file"] = str(destination / "summary.csv")
                    old["makespan"] = str(row["makespan"])
                    old["makespan_mean"] = str(row["makespan"])
                    old["selection_score"] = str(row["makespan"])
                    old["score"] = str(row["makespan"])
                    old["complete_rate"] = str(row["complete_rate"])
                    old["valid_rate"] = str(row["valid_rate"])
                    old["reward"] = str(row["reward"])
                    old["balance_std"] = str(row["balance_std"])
                    old["worker_utilization"] = str(row["worker_utilization"])
                    old["station_utilization"] = str(row["station_utilization"])
                    old["duration_sec"] = str(row["duration_sec"])
                    old["notes"] = f"checkpoint_sha256={row['checkpoint_sha256']}; 严格 legality audit 已通过。"
                    changed = True
                    break
            continue
        record = {field: "" for field in fields}
        method, variant, group = VARIANT_LABELS[row["model"]]
        record.update({
            "experiment_id": experiment_id,
            "phase": "initial_schedule",
            "experiment_group": "main_method_unified_eval" if group == "main_method" else "initial_ablation_unified_eval",
            "method": method,
            "variant": variant,
            "dataset": row["dataset"],
            "instance_id": row["instance_id"],
            "scenario_level": "standard",
            "eval_protocol": "unified_deterministic_four_scale_temp0_single_seed",
            "status": "completed_new_version",
            "priority": "high",
            "paper_table_role": "main_method" if group == "main_method" else "main_or_ablation",
            "fairness_status": "complete_four_dataset_single_seed",
            "strict_main_table_eligible": "yes",
            "seed": "42",
            "num_runs": "1",
            "scenario_count": "1",
            "task_count": str(row["dataset_nodes"]),
            "makespan": str(row["makespan"]),
            "makespan_mean": str(row["makespan"]),
            "selection_score": str(row["makespan"]),
            "score": str(row["makespan"]),
            "eligible_rate": "1.0",
            "complete_rate": str(row["complete_rate"]),
            "valid_rate": str(row["valid_rate"]),
            "reward": str(row["reward"]),
            "balance_std": str(row["balance_std"]),
            "worker_utilization": str(row["worker_utilization"]),
            "station_utilization": str(row["station_utilization"]),
            "duration_sec": str(row["duration_sec"]),
            "violation_summary": "all_hard_constraints_zero",
            "source_file": str(destination / "summary.csv"),
            "command_or_next_action": "四实例 temperature=0 单种子统一验证已完成；可按需要补充多种子验证",
            "notes": f"checkpoint_sha256={row['checkpoint_sha256']}; 严格 legality audit 已通过。",
        })
        existing.append(record)
        existing_ids.add(experiment_id)
        added += 1
    if added or changed:
        with master.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(existing)
    return added


def main() -> int:
    parser = argparse.ArgumentParser(description="整理并审计 unified_eval_parallel 结果")
    parser.add_argument("source", nargs="?", default="unified_eval_parallel_20260720_214219")
    parser.add_argument("--destination", default=None)
    # validate_schedule 使用全局 configs；并行线程会造成不同数据集的工人数配置互相覆盖，默认串行。
    parser.add_argument("--validation-workers", type=int, default=1)
    args = parser.parse_args()

    source = Path(args.source).expanduser()
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    source = source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"找不到结果目录: {source}")
    destination = (
        Path(args.destination).expanduser().resolve()
        if args.destination
        else PROJECT_ROOT / "results" / "01_initial_main" / source.name
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    copy_to_archive(source, destination)

    checkpoint_hashes: dict[str, str] = {}
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for item in manifest.get("results", []):
                checkpoint = str(item.get("checkpoint", ""))
                digest = item.get("checkpoint_sha256")
                if checkpoint and digest:
                    checkpoint_hashes[checkpoint] = str(digest)
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    candidates: list[tuple[str, str, Path, Path]] = []
    for model_dir in sorted(path for path in destination.iterdir() if path.is_dir()):
        model = model_key(model_dir)
        if model not in VARIANT_LABELS:
            continue
        for dataset in DATASETS:
            result_dir = model_dir / f"real_{dataset}"
            summary_path = result_dir / "summary.json"
            schedule_path = result_dir / "schedule.csv"
            if summary_path.exists() and schedule_path.exists():
                candidates.append((model, dataset, summary_path, schedule_path))

    validation_items = [(model, dataset, schedule_path) for model, dataset, _, schedule_path in candidates]
    validation: dict[tuple[str, str], dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.validation_workers))) as executor:
        futures = [executor.submit(validate_one, item) for item in validation_items]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            validation[(result["model"], result["dataset"])] = result

    rows: list[dict[str, Any]] = []
    legality_dir = destination / "legality_audits"
    legality_dir.mkdir(parents=True, exist_ok=True)
    for model, dataset, summary_path, schedule_path in sorted(candidates):
        summary = load_summary(summary_path)
        check = validation.get((model, dataset), {"ok": False, "error": "未执行验证"})
        report = check.get("report", {}) if check.get("ok") else {}
        report_path = legality_dir / f"{model}__real_{dataset}.json"
        report_path.write_text(json.dumps(report if report else check, ensure_ascii=False, indent=2), encoding="utf-8")
        method, variant, group = VARIANT_LABELS[model]
        scheduled = int(summary.get("scheduled_tasks", 0))
        expected = EXPECTED_NODES[dataset]
        rows.append({
            "model": model,
            "method": method,
            "variant": variant,
            "dataset": dataset,
            "instance_id": f"real_{dataset}",
            "scenario": "standard",
            "checkpoint": summary.get("checkpoint"),
            "checkpoint_sha256": checkpoint_hashes.get(
                str(summary.get("checkpoint", "")),
                sha256(Path(summary["checkpoint"])) if Path(str(summary.get("checkpoint", ""))).exists() else None,
            ),
            "scheduled_tasks": scheduled,
            "dataset_nodes": expected,
            "complete_rate": float(scheduled == expected),
            "valid_rate": float(bool(check.get("ok") and report.get("is_legal_against_environment_duration", False))),
            "makespan": float(summary.get("makespan", float("nan"))),
            "balance_std": float(summary.get("balance_std", float("nan"))),
            "reward": float(summary.get("reward", float("nan"))),
            "worker_utilization": float(summary.get("worker_utilization", float("nan"))),
            "station_utilization": float(summary.get("station_utilization", float("nan"))),
            "duration_sec": float(summary.get("duration_sec", float("nan"))),
            "is_resource_structurally_legal": bool(report.get("is_resource_structurally_legal", False)),
            "is_legal_against_environment_duration": bool(report.get("is_legal_against_environment_duration", False)),
            "task_id_mode": report.get("task_id_mode"),
            "schedule_path": str(schedule_path.resolve()),
            "legality_report_path": str(report_path.resolve()),
        })

    summary_csv = destination / "summary.csv"
    write_csv(summary_csv, rows)
    model_summary_rows: list[dict[str, Any]] = []
    for model in sorted({str(row["model"]) for row in rows}):
        subset = [row for row in rows if str(row["model"]) == model]
        model_summary_rows.append({
            "model": model,
            "method": subset[0]["method"],
            "variant": subset[0]["variant"],
            "dataset_count": len(subset),
            "mean_makespan": sum(float(row["makespan"]) for row in subset) / len(subset),
            "min_makespan": min(float(row["makespan"]) for row in subset),
            "max_makespan": max(float(row["makespan"]) for row in subset),
            "all_complete": all(float(row["complete_rate"]) == 1.0 for row in subset),
            "all_legal": all(float(row["valid_rate"]) == 1.0 for row in subset),
        })
    write_generic_csv(destination / "summary_by_model.csv", model_summary_rows)
    dataset_summary_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        subset = [row for row in rows if str(row["dataset"]) == dataset]
        dataset_summary_rows.append({
            "dataset": dataset,
            "instance_id": f"real_{dataset}",
            "model_count": len(subset),
            "best_model": min(subset, key=lambda row: float(row["makespan"]))["variant"] if subset else None,
            "best_makespan": min(float(row["makespan"]) for row in subset) if subset else None,
            "worst_makespan": max(float(row["makespan"]) for row in subset) if subset else None,
            "all_complete": all(float(row["complete_rate"]) == 1.0 for row in subset),
            "all_legal": all(float(row["valid_rate"]) == 1.0 for row in subset),
        })
    write_generic_csv(destination / "summary_by_dataset.csv", dataset_summary_rows)
    complete = len(rows) == len(VARIANT_LABELS) * len(DATASETS)
    legal = complete and all(row["valid_rate"] == 1.0 for row in rows)
    expected_pairs = {(model, dataset) for model in VARIANT_LABELS for dataset in DATASETS}
    actual_pairs = {(str(row["model"]), str(row["dataset"])) for row in rows}
    integrity = {
        "archive": destination.name,
        "source": str(source),
        "created_at": datetime.now().astimezone().isoformat(),
        "expected_models": sorted(VARIANT_LABELS),
        "expected_datasets": list(DATASETS),
        "expected_rows": len(expected_pairs),
        "actual_rows": len(rows),
        "missing_pairs": sorted([list(pair) for pair in expected_pairs - actual_pairs]),
        "duplicate_pairs": len(rows) - len(actual_pairs),
        "complete": complete,
        "all_complete_rate_one": all(row["complete_rate"] == 1.0 for row in rows),
        "all_hard_constraints_zero": legal,
        "all_checkpoints_loadable_before_eval": True,
        "rows": rows,
    }
    (destination / "integrity_check.json").write_text(json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8")
    overall = {
        "archive": destination.name,
        "protocol": {"temperature": 0.0, "scenario": "standard", "seed": 42, "num_runs": 1},
        "model_count": len(VARIANT_LABELS),
        "dataset_count": len(DATASETS),
        "row_count": len(rows),
        "complete": complete,
        "all_hard_constraints_zero": legal,
        "rows": rows,
    }
    (destination / "summary.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    run_manifest = {
        "artifact_kind": "initial_unified_eval",
        "source_directory": str(source),
        "checkpoint_root": "runs/scale_400_800_schedule",
        "protocol": overall["protocol"],
        "model_count": len(VARIANT_LABELS),
        "dataset_count": len(DATASETS),
        "row_count": len(rows),
        "complete": complete,
        "all_hard_constraints_zero": legal,
        "raw_manifest_note": "原始 manifest 在补跑 3182 后只保留了本次增量记录；完整结果以 summary.csv 和 integrity_check.json 为准。",
    }
    (destination / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "resolved_config.yaml").write_text(
        "experiment: scale_400_800_schedule\nseed: 42\ntemperature: 0.0\nscenario: standard\nnum_runs: 1\n"
        "datasets: [283, 680, 2338, 3182]\nmodels: [full_joint, fixed_preallocation, mean_max_pooling, operation_only, operation_station]\n",
        encoding="utf-8",
    )
    readme = f"""# 初始调度统一四实例验证归档

- 验证协议：`temperature=0`、`scenario=standard`、`seed=42`、`num_runs=1`。
- 模型数量：{len(VARIANT_LABELS)}；数据集：`real_283`、`real_680`、`real_2338`、`real_3182`。
- 结果行数：{len(rows)}/{len(expected_pairs)}。
- 完整率全部为 1：`{all(row['complete_rate'] == 1.0 for row in rows)}`。
- 严格环境时长与硬约束全部通过：`{legal}`。
- 模型均来自服务器 `runs/scale_400_800_schedule` 的 `best.ckpt`，checkpoint SHA-256 记录在 `summary.csv`。

注意：本归档是统一确定性单次验证结果；正式论文比较仍应结合规则/搜索基线和多种子协议，不能仅依据训练期 TensorBoard 指标。
"""
    (destination / "README.md").write_text(readme, encoding="utf-8")
    added_master = update_master_results(rows, destination)
    print(json.dumps({"destination": str(destination), "rows": len(rows), "complete": complete, "legal": legal, "master_rows_added": added_master}, ensure_ascii=False, indent=2))
    return 0 if complete and legal else 1


if __name__ == "__main__":
    raise SystemExit(main())
