"""整理初始调度的少量快速验证，不将其登记为正式主表结果。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    """计算文件哈希，确保模型和排程可追溯。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="整理初始调度快速验证结果")
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    root = args.eval_root.resolve()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        directory = root / f"real_{dataset}"
        summary_path = directory / "summary.json"
        audit_path = directory / "legality_audit.json"
        schedule_path = directory / "schedule.csv"
        if not all(path.is_file() for path in (summary_path, audit_path, schedule_path)):
            raise FileNotFoundError(f"快速验证产物不完整：{directory}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        violations = audit.get("violations", {})
        rows.append(
            {
                "instance_id": f"real_{dataset}",
                "dataset": str(dataset),
                "temperature": float(args.temperature),
                "seed": int(args.seed),
                "num_runs": 1,
                "makespan": float(summary["makespan"]),
                "reward": float(summary["reward"]),
                "balance_std": float(summary["balance_std"]),
                "duration_sec": float(summary["duration_sec"]),
                "worker_utilization": float(summary["worker_utilization"]),
                "station_utilization": float(summary["station_utilization"]),
                "scheduled_real_tasks": int(audit["scheduled_real_tasks"]),
                "num_real_tasks": int(audit["num_real_tasks"]),
                "complete": int(audit["scheduled_real_tasks"]) == int(audit["num_real_tasks"]),
                "legal": bool(audit["is_legal_against_current_data_duration"]),
                "max_hard_violation": max((int(value) for value in violations.values()), default=0),
                "schedule_sha256": sha256(schedule_path),
            }
        )
    if not all(row["complete"] and row["legal"] and row["max_hard_violation"] == 0 for row in rows):
        raise RuntimeError("至少一份快速验证排程未完整通过独立合法性审计")

    root.mkdir(parents=True, exist_ok=True)
    with (root / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "method": "HB-GAT-PPO",
        "phase": "initial_schedule_quick_validation",
        "evaluation_role": "quick_deterministic_comparison_only",
        "strict_main_table_eligible": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "temperature": float(args.temperature),
        "seed": int(args.seed),
        "num_runs_per_dataset": 1,
        "datasets": rows,
    }
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    integrity = {
        "dataset_count": len(rows),
        "all_complete": all(row["complete"] for row in rows),
        "all_legal": all(row["legal"] for row in rows),
        "max_hard_violation": max(int(row["max_hard_violation"]) for row in rows),
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "strict_main_table_eligible": False,
    }
    (root / "integrity_check.json").write_text(json.dumps(integrity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_type": "initial_schedule_quick_deterministic_validation",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "datasets": [f"real_{dataset}" for dataset in args.datasets],
                "temperature": float(args.temperature),
                "seed": int(args.seed),
                "num_runs": 1,
                "formal_protocol": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "# HB-GAT-PPO 初始调度快速验证\n\n"
        "- 协议：temperature=0、seed=42、每个数据集单次确定性运行。\n"
        "- 范围：仅 real_283 与 real_680，用于检查新 checkpoint 的可用性和快速对比。\n"
        "- 每份 schedule 均已独立审计；详见 `summary.json`、`summary.csv` 和各实例的 `legality_audit.json`。\n"
        "- 这不是当前六次验证协议，且四实例参与过 checkpoint 选择，因此 `strict_main_table_eligible=no`，不得作为论文主表。\n",
        encoding="utf-8",
    )
    manifest_rows = [
        {"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "file_manifest.json"
    ]
    (root / "file_manifest.json").write_text(json.dumps({"root": str(root), "files": manifest_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
