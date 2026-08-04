"""整理手工运行的初始调度四实例六次验证结果。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("283", "680", "2338", "3182")
RUNS = (("temp0_seed42", 0.0, 42),) + tuple(
    (f"temp001_seed{seed}", 0.01, seed) for seed in range(42, 47)
)
PROTOCOL_ID = "initial_real4_sixrun_temp0_s42_temp001_s42_46_v1"


def sha256(path: Path) -> str:
    """计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    """原子写入 JSON。"""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    """读取对象型 JSON。"""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须为对象：{path}")
    return value


def collect_run(run_dir: Path, instance_id: str, run_name: str, temperature: float, seed: int) -> dict[str, Any]:
    """读取并严格核验一次已完成运行。"""
    schedule_path = run_dir / "schedule.csv"
    summary_path = run_dir / "summary.json"
    audit_path = run_dir / "legality_audit.json"
    for path in (schedule_path, summary_path, audit_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"缺少验证产物：{path}")
    summary = read_json(summary_path)
    audit = read_json(audit_path)
    metrics = {
        key: float(summary[key])
        for key in ("makespan", "reward", "balance_std", "duration_sec", "worker_utilization", "station_utilization")
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError(f"存在非有限指标：{run_dir}")
    complete = int(audit["scheduled_real_tasks"]) == int(audit["num_real_tasks"])
    legal = bool(audit["is_legal_against_current_data_duration"])
    violations = audit.get("violations", {})
    if not isinstance(violations, dict):
        raise ValueError(f"violations 格式无效：{audit_path}")
    max_hard = max((int(value) for value in violations.values()), default=0)
    if not complete or not legal or max_hard != 0:
        raise ValueError(f"验证不合格：{run_dir} complete={complete} legal={legal} hard={max_hard}")
    return {
        "instance_id": instance_id,
        "run_name": run_name,
        "temperature": temperature,
        "seed": seed,
        **metrics,
        "complete": complete,
        "legal": legal,
        "max_hard_violation": max_hard,
        "schedule_sha256": sha256(schedule_path),
    }


def update_master(summary_rows: list[dict[str, Any]], args: argparse.Namespace, checkpoint_sha: str) -> None:
    """写入四条可追溯的条件性主表记录。"""
    master_path = PROJECT_ROOT / "results" / "experiment_master_results.csv"
    with master_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    prefix = f"initial_{args.variant}_four_instance_sixrun_{args.tag}_"
    rows = [row for row in rows if not row.get("experiment_id", "").startswith(prefix)]
    for item in summary_rows:
        dataset = item["instance_id"].removeprefix("real_")
        row = {field: "" for field in fieldnames}
        row.update(
            {
                "experiment_id": f"{prefix}{dataset}",
                "phase": "initial_schedule",
                "experiment_group": "ablation",
                "method": args.method,
                "variant": args.variant,
                "dataset": dataset,
                "instance_id": item["instance_id"],
                "scenario_level": "standard",
                "eval_protocol": PROTOCOL_ID,
                "status": "completed_single_checkpoint_eval",
                "priority": "high",
                "paper_table_role": "ablation",
                "fairness_status": "teacher_approved_config_variation",
                "strict_main_table_eligible": "conditional",
                "seed": "42",
                "num_runs": "6",
                "scenario_count": "6",
                "makespan": item["deterministic_makespan_temp0_seed42"],
                "makespan_mean": item["stochastic_makespan_temp001_mean"],
                "makespan_std": item["stochastic_makespan_temp001_std_sample"],
                "eligible_rate": item["eligible_rate"],
                "complete_rate": item["complete_rate"],
                "valid_rate": item["eligible_rate"],
                "violation_summary": "max_hard_violation=0",
                "source_file": str(args.eval_root / "summary.json"),
                "command_or_next_action": "已完成四实例六次验证；保留 checkpoint 哈希和逐次审计。",
                "notes": f"checkpoint_sha256={checkpoint_sha}; 四报告实例参与模型选择，主表资格为 conditional。",
            }
        )
        rows.append(row)
    with master_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="整理手工四实例六次初始调度验证")
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", default="HB-GAT-PPO")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--tag", default="20260804")
    args = parser.parse_args()
    args.eval_root = args.eval_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if not args.checkpoint.is_file() or args.checkpoint.stat().st_size <= 0:
        raise FileNotFoundError(f"checkpoint 不存在或为空：{args.checkpoint}")

    details: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for run_name, temperature, seed in RUNS:
            details.append(collect_run(args.eval_root / f"real_{dataset}" / run_name, f"real_{dataset}", run_name, temperature, seed))
    if len(details) != 24:
        raise RuntimeError(f"预期 24 条验证，实际 {len(details)} 条")

    with (args.eval_root / "runs_detail.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    summary_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        current = [row for row in details if row["instance_id"] == f"real_{dataset}"]
        deterministic = next(row for row in current if row["temperature"] == 0.0)
        stochastic = [row["makespan"] for row in current if row["temperature"] == 0.01]
        summary_rows.append(
            {
                "instance_id": f"real_{dataset}",
                "deterministic_makespan_temp0_seed42": deterministic["makespan"],
                "stochastic_makespan_temp001_mean": mean(stochastic),
                "stochastic_makespan_temp001_std_sample": stdev(stochastic),
                "eligible_rate": mean(float(row["legal"]) for row in current),
                "complete_rate": mean(float(row["complete"]) for row in current),
                "max_hard_violation": max(int(row["max_hard_violation"]) for row in current),
            }
        )
    with (args.eval_root / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    checkpoint_sha = sha256(args.checkpoint)
    summary = {
        "protocol_id": PROTOCOL_ID,
        "method": args.method,
        "variant": args.variant,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "rows": summary_rows,
        "evidence_level": "completed_conditional",
        "strict_main_table_eligible": False,
    }
    write_json(args.eval_root / "summary.json", summary)
    write_json(
        args.eval_root / "integrity_check.json",
        {
            "protocol_id": PROTOCOL_ID,
            "expected_run_count": 24,
            "observed_run_count": len(details),
            "instance_count": len(DATASETS),
            "runs_per_instance": len(RUNS),
            "all_complete": True,
            "all_legal": True,
            "max_hard_violation": 0,
            "checkpoint_sha256": checkpoint_sha,
            "strict_main_table_eligible": False,
        },
    )
    write_json(
        args.eval_root / "run_manifest.json",
        {
            "protocol_id": PROTOCOL_ID,
            "method": args.method,
            "variant": args.variant,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "datasets": [f"real_{dataset}" for dataset in DATASETS],
            "runs_per_dataset": [{"name": name, "temperature": temp, "seed": seed} for name, temp, seed in RUNS],
            "organized_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    (args.eval_root / "README.md").write_text(
        "# 初始调度四实例六次验证\n\n"
        "- 实例：real_283、real_680、real_2338、real_3182。\n"
        "- 每实例：temperature=0、seed42 一次；temperature=0.01、seed42--46 五次。\n"
        "- 24/24 排程均经独立回放，完整、合法，全部硬约束违规为零。\n"
        "- 四个报告实例参与 checkpoint 选择，因此结果为 completed_conditional，不作为独立 held-out 结论。\n",
        encoding="utf-8",
    )
    manifest_rows = [
        {"path": path.relative_to(args.eval_root).as_posix(), "size": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(args.eval_root.rglob("*"))
        if path.is_file() and path.name != "file_manifest.json"
    ]
    write_json(args.eval_root / "file_manifest.json", {"root": str(args.eval_root), "files": manifest_rows})
    update_master(summary_rows, args, checkpoint_sha)
    print(json.dumps({"eval_root": str(args.eval_root), "rows": len(details), "checkpoint_sha256": checkpoint_sha}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
