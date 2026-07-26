"""执行并整理初始调度的六次验证协议。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("283", "680")
RUNS = (("temp0_seed42", 0.0, 42),) + tuple(
    (f"temp001_seed{seed}", 0.01, seed) for seed in range(42, 47)
)


def sha256(path: Path) -> str:
    """计算文件哈希，用于锁定 checkpoint 与全部产物。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_command(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    """执行单次评估；失败立即记录并停止，绝不自动重试。"""
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"命令失败，exit={completed.returncode}，日志：{log_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="运行初始调度 283/680 的六次验证协议")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    output_root = args.output_root.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if output_root.exists():
        raise FileExistsError(f"输出目录已经存在，拒绝覆盖：{output_root}")
    output_root.mkdir(parents=True)
    checkpoint_sha = sha256(checkpoint)
    environment = os.environ.copy()
    environment.update({name: "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    progress_path = output_root / "progress.json"
    progress: dict[str, Any] = {"status": "running", "completed": [], "failed": None}
    write_json(
        output_root / "launch_manifest.json",
        {
            "run_type": "initial_schedule_sixrun_protocol",
            "method": "HB-GAT-PPO",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "datasets": [f"real_{dataset}" for dataset in DATASETS],
            "protocol": "temperature=0 seed42 once; temperature=0.01 seeds42..46 once each",
            "formal_protocol": True,
            "strict_main_table_eligible": False,
            "eligibility_note": "four report instances participated in checkpoint selection; local Windows validation is not held-out testing",
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    shutil.copy2(
        PROJECT_ROOT / "conf" / "experiment" / "initial_main_real4_async_selection.yaml",
        output_root / "resolved_config.yaml",
    )
    try:
        for dataset in DATASETS:
            for run_name, temperature, seed in RUNS:
                run_dir = output_root / f"real_{dataset}" / run_name
                run_dir.mkdir(parents=True)
                eval_log = run_dir / "evaluation.log"
                audit_log = run_dir / "legality_audit.log"
                print(f"[开始] dataset={dataset} run={run_name} temperature={temperature} seed={seed}", flush=True)
                eval_command = [
                    args.python_executable,
                    "-u",
                    "evaluate_model.py",
                    "experiment=initial_main_real4_async_selection",
                    f"model_path={checkpoint}",
                    f"test_data=data/{dataset}.csv",
                    "num_runs=1",
                    f"temperature={temperature}",
                    "scenario=standard",
                    f"seed={seed}",
                    "no_gantt=true",
                    "verbose_eval_progress=true",
                    f"output_dir={run_dir}",
                ]
                run_command(eval_command, eval_log, environment)
                audit_command = [
                    args.python_executable,
                    "-u",
                    "scripts/validate_initial_schedule.py",
                    "--data",
                    f"data/{dataset}.csv",
                    "--schedule",
                    str(run_dir / "schedule.csv"),
                    "--output",
                    str(run_dir / "legality_audit.json"),
                ]
                run_command(audit_command, audit_log, environment)
                progress["completed"].append({"dataset": dataset, "run": run_name})
                write_json(progress_path, progress)
                print(f"[完成] dataset={dataset} run={run_name}", flush=True)
    except Exception as exc:
        progress["status"] = "failed"
        progress["failed"] = f"{type(exc).__name__}: {exc}"
        write_json(progress_path, progress)
        raise

    detail_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for run_name, temperature, seed in RUNS:
            run_dir = output_root / f"real_{dataset}" / run_name
            metrics = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            audit = json.loads((run_dir / "legality_audit.json").read_text(encoding="utf-8"))
            violations = audit.get("violations", {})
            detail_rows.append(
                {
                    "instance_id": f"real_{dataset}",
                    "run_name": run_name,
                    "temperature": temperature,
                    "seed": seed,
                    "makespan": float(metrics["makespan"]),
                    "reward": float(metrics["reward"]),
                    "balance_std": float(metrics["balance_std"]),
                    "duration_sec": float(metrics["duration_sec"]),
                    "worker_utilization": float(metrics["worker_utilization"]),
                    "station_utilization": float(metrics["station_utilization"]),
                    "complete": int(audit["scheduled_real_tasks"]) == int(audit["num_real_tasks"]),
                    "legal": bool(audit["is_legal_against_current_data_duration"]),
                    "max_hard_violation": max((int(value) for value in violations.values()), default=0),
                    "schedule_sha256": sha256(run_dir / "schedule.csv"),
                }
            )
    with (output_root / "runs_detail.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)

    summary_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        current = [row for row in detail_rows if row["instance_id"] == f"real_{dataset}"]
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
    with (output_root / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    write_json(
        output_root / "summary.json",
        {
            "method": "HB-GAT-PPO",
            "checkpoint_sha256": checkpoint_sha,
            "protocol": "6 runs/dataset: temperature=0 seed42 + temperature=0.01 seeds42..46",
            "rows": summary_rows,
            "strict_main_table_eligible": False,
            "note": "checkpoint was selected using four report instances; this is validation evidence, not held-out final test evidence",
        },
    )
    write_json(
        output_root / "integrity_check.json",
        {
            "expected_run_count": len(DATASETS) * len(RUNS),
            "observed_run_count": len(detail_rows),
            "all_complete": all(row["complete"] for row in detail_rows),
            "all_legal": all(row["legal"] for row in detail_rows),
            "max_hard_violation": max(int(row["max_hard_violation"]) for row in detail_rows),
            "checkpoint_sha256": checkpoint_sha,
            "strict_main_table_eligible": False,
        },
    )
    write_json(
        output_root / "run_manifest.json",
        {
            "run_type": "initial_schedule_sixrun_protocol",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "datasets": [f"real_{dataset}" for dataset in DATASETS],
            "temperature0": {"seed": 42, "runs": 1},
            "temperature001": {"seeds": [42, 43, 44, 45, 46], "runs": 5},
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "strict_main_table_eligible": False,
        },
    )
    (output_root / "README.md").write_text(
        "# HB-GAT-PPO 初始调度六次验证\n\n"
        "- 每个实例运行 6 次：temperature=0、seed=42 为确定性结果；temperature=0.01、seed=42–46 用于均值和样本标准差。\n"
        "- 每次均导出任务级 schedule，并用 `validate_initial_schedule.py` 回放审核。\n"
        "- 由于该 checkpoint 的训练期选择使用过四个报告实例，结果不属于 held-out 测试，`strict_main_table_eligible=no`。\n",
        encoding="utf-8",
    )
    manifest_rows = [
        {"path": path.relative_to(output_root).as_posix(), "size": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "file_manifest.json"
    ]
    write_json(output_root / "file_manifest.json", {"root": str(output_root), "files": manifest_rows})
    progress["status"] = "completed"
    write_json(progress_path, progress)
    print("[全部完成] 12/12 次评估与合法性审计完成", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
