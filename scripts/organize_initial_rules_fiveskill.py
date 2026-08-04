"""核验并归档五技能初始调度规则基线结果。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_initial_schedule import validate_schedule


DATASETS: tuple[str, ...] = ("283", "680", "2338", "3182")
METHODS: tuple[str, ...] = ("SPT", "LPT", "Random", "EDD", "CPM", "MSL")
EXPECTED_REAL_TASKS: dict[str, int] = {"283": 283, "680": 680, "2338": 2338, "3182": 3182}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("拒绝写入空结果表")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def update_master(rows: list[dict[str, Any]], source: Path, *, protocol: str, strict_eligible: bool) -> int:
    path = PROJECT_ROOT / "results" / "experiment_master_results.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        master_rows = list(reader)
    changed = 0
    by_id = {str(row.get("experiment_id", "")): row for row in master_rows}
    for row in rows:
        record = by_id.get(f"initial_baseline_{row['method']}_{row['dataset']}")
        if record is None:
            continue
        legal = str(row["legal"]).lower() == "true"
        eligible = legal and strict_eligible
        record.update(
            {
                "status": "completed_fiveskill_v1" if eligible else "completed_evidence_pending",
                "eval_protocol": protocol,
                "seed": "42",
                "num_runs": "10" if row["method"] == "Random" else "1",
                "scenario_count": "1",
                "task_count": str(row["expected_tasks"]),
                "makespan": str(row["makespan"]),
                "makespan_mean": str(row["makespan"]),
                "selection_score": str(row["makespan"]),
                "score": str(row["makespan"]),
                "balance_std": str(row["balance_std"]),
                "worker_utilization": str(row["worker_utilization"]),
                "station_utilization": str(row["station_utilization"]),
                "duration_sec": str(row["duration_sec"]),
                "complete_rate": "1.0" if row["complete"] else "0.0",
                "valid_rate": "1.0" if legal else "0.0",
                "eligible_rate": "1.0" if legal else "0.0",
                "strict_main_table_eligible": "yes" if eligible else "no",
                "violation_summary": row["violation_summary"],
                "source_file": str(source / "summary.csv"),
                "notes": (
                    f"data_sha256={row['data_sha256']}; schedule_sha256={row['schedule_sha256']}; "
                    "独立 initial-schedule legality audit 已通过。"
                    if eligible
                    else "独立 legality audit 已通过，但协议或搜索预算的可追溯证据不足；不得纳入严格主表。"
                ),
            }
        )
        changed += 1
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(master_rows)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="核验并归档五技能初始调度规则基线")
    parser.add_argument(
        "--source",
        default="results/02_initial_baselines/initial_rules_fiveskill_v1_s42_260803_231101",
    )
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument(
        "--protocol",
        default="initial_rules_fiveskill_v1_temp0_seed42; Random=best_of_10",
    )
    parser.add_argument("--strict-eligible", action="store_true")
    args = parser.parse_args()
    source = (PROJECT_ROOT / args.source).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)

    summary_source = source / "baselines_summary.csv"
    with summary_source.open("r", encoding="utf-8-sig", newline="") as stream:
        original = {(str(row["Method"]), str(row["Dataset"])): row for row in csv.DictReader(stream)}
    methods = tuple(str(method) for method in args.methods)
    expected_pairs = {(method, dataset) for method in methods for dataset in DATASETS}
    if set(original) != expected_pairs:
        raise ValueError(f"规则结果组合不完整或存在额外项: expected={len(expected_pairs)}, actual={len(original)}")

    audit_dir = source / "legality_audits"
    audit_dir.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    for method in methods:
        for dataset in DATASETS:
            schedule = source / method / dataset / "schedule.csv"
            metrics_path = source / method / dataset / "metrics.json"
            data = PROJECT_ROOT / "data" / f"{dataset}.csv"
            if not schedule.is_file() or not metrics_path.is_file() or not data.is_file():
                raise FileNotFoundError(f"缺少规则结果资产: {method}/{dataset}")
            report = validate_schedule(data_path=data, schedule_path=schedule)
            audit_path = audit_dir / f"{method}__real_{dataset}.json"
            audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            source_row = original[(method, dataset)]
            makespan = float(metrics["makespan"])
            if abs(makespan - float(source_row["Makespan"])) > 0.02:
                raise ValueError(f"汇总与原始指标 makespan 不一致: {method}/{dataset}")
            violations = dict(report.get("violations", {}))
            total_violations = sum(int(value) for value in violations.values())
            scheduled = int(report["scheduled_real_tasks"])
            expected = int(report["num_real_tasks"])
            if expected != EXPECTED_REAL_TASKS[dataset]:
                raise ValueError(f"真实工序数不符合实例标识: {dataset} -> {expected}")
            complete = scheduled == expected
            legal = bool(report["is_legal_against_environment_duration"])
            rows.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "instance_id": f"real_{dataset}",
                    "makespan": makespan,
                    "balance_std": float(metrics["workload_balance_std"]),
                    "worker_utilization": float(metrics["worker_utilization"]),
                    "station_utilization": float(metrics["station_utilization"]),
                    "duration_sec": float(metrics["inference_time"]),
                    "expected_tasks": expected,
                    "scheduled_tasks": scheduled,
                    "schedule_rows": int(report["num_schedule_rows"]),
                    "dataset_nodes": int(report["num_dataset_nodes"]),
                    "complete": complete,
                    "legal": legal,
                    "hard_violation_total": total_violations,
                    "violation_summary": json.dumps(violations, ensure_ascii=False, sort_keys=True),
                    "data_sha256": sha256(data),
                    "schedule_sha256": sha256(schedule),
                    "schedule_path": str(schedule.resolve()),
                    "audit_path": str(audit_path.resolve()),
                }
            )

    write_csv(source / "summary.csv", rows)
    by_dataset: list[dict[str, Any]] = []
    for dataset in DATASETS:
        subset = [row for row in rows if row["dataset"] == dataset]
        best = min(subset, key=lambda row: float(row["makespan"]))
        by_dataset.append(
            {
                "dataset": dataset,
                "instance_id": f"real_{dataset}",
                "method_count": len(subset),
                "best_method": best["method"],
                "best_makespan": best["makespan"],
                "all_complete": all(bool(row["complete"]) for row in subset),
                "all_legal": all(bool(row["legal"]) for row in subset),
            }
        )
    write_csv(source / "summary_by_dataset.csv", by_dataset)

    complete = len(rows) == len(expected_pairs) and all(bool(row["complete"]) for row in rows)
    legal = complete and all(bool(row["legal"]) and int(row["hard_violation_total"]) == 0 for row in rows)
    integrity = {
        "protocol": str(args.protocol),
        "expected_rows": len(expected_pairs),
        "actual_rows": len(rows),
        "duplicate_pairs": len(rows) - len({(row["method"], row["dataset"]) for row in rows}),
        "all_complete": complete,
        "all_legal": legal,
        "all_hard_constraints_zero": legal,
        "rows": rows,
    }
    (source / "integrity_check.json").write_text(json.dumps(integrity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (source / "summary.json").write_text(
        json.dumps({"protocol": integrity["protocol"], "rows": rows, "by_dataset": by_dataset}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    (source / "README.md").write_text(
        "# 五技能初始调度规则基线\n\n"
        "- 方法：SPT、LPT、Random、EDD、CPM、MSL。\n"
        "- 实例：real_283、real_680、real_2338、real_3182。\n"
        f"- 协议：{args.protocol}。\n"
        f"- 完整性：24/24={complete}；独立硬约束审核全零={legal}。\n"
        "- 原始 schedule.csv、metrics.json、run.log、独立 legality_audits、输入/输出 SHA-256 均保留。\n"
        "- 该结果不依赖初始学习训练池；因此不受 ctg_fv1 训练 manifest 待核验问题影响。\n"
        f"- 严格主表资格：{bool(args.strict_eligible and legal)}。\n",
        encoding="utf-8",
    )
    manifest_rows = []
    for path in sorted(item for item in source.rglob("*") if item.is_file() and item.name != "file_manifest.json"):
        manifest_rows.append({"path": str(path.relative_to(source)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    (source / "file_manifest.json").write_text(json.dumps({"files": manifest_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    changed = update_master(rows, source, protocol=str(args.protocol), strict_eligible=bool(args.strict_eligible))
    print(json.dumps({"source": str(source), "rows": len(rows), "complete": complete, "legal": legal, "master_rows_updated": changed}, ensure_ascii=False))
    return 0 if legal else 1


if __name__ == "__main__":
    raise SystemExit(main())
