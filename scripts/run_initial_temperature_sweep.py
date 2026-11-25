"""对固定初始调度 checkpoint 执行可续跑的单实例温度敏感性验证。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    """计算产物哈希，锁定模型与排程证据。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    """使用 UTF-8 写入结构化实验元数据。"""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_command(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    """运行一次评估或审计；失败立即停止，绝不自动重试。"""
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"命令失败，exit={result.returncode}，日志：{log_path}")


def temperature_label(temperature: float) -> str:
    """将温度转换为稳定、可读且不会冲突的目录名。"""
    return f"temp{temperature:.4f}".replace(".", "")


def read_completed_run(run_dir: Path) -> dict[str, Any] | None:
    """仅接受同时含结果和合法审计的既有运行，作为安全续跑依据。"""
    summary_path = run_dir / "summary.json"
    audit_path = run_dir / "legality_audit.json"
    schedule_path = run_dir / "schedule.csv"
    if not all(path.is_file() for path in (summary_path, audit_path, schedule_path)):
        return None
    metrics = json.loads(summary_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    violations = audit.get("violations", {})
    complete = int(audit["scheduled_real_tasks"]) == int(audit["num_real_tasks"])
    legal = bool(audit["is_legal_against_current_data_duration"])
    max_violation = max((int(value) for value in violations.values()), default=0)
    if not complete or not legal or max_violation != 0:
        raise RuntimeError(f"已有目录包含不合格结果，拒绝静默覆盖：{run_dir}")
    return {
        "makespan": float(metrics["makespan"]),
        "reward": float(metrics["reward"]),
        "balance_std": float(metrics["balance_std"]),
        "duration_sec": float(metrics["duration_sec"]),
        "worker_utilization": float(metrics["worker_utilization"]),
        "station_utilization": float(metrics["station_utilization"]),
        "complete": complete,
        "legal": legal,
        "max_hard_violation": max_violation,
        "schedule_sha256": sha256(schedule_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="运行初始调度单实例温度敏感性验证")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", default="680", help="不含 .csv 后缀的数据集编号")
    parser.add_argument("--temperatures", type=float, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    data_path = (PROJECT_ROOT / "data" / f"{args.dataset}.csv").resolve()
    output_root = args.output_root.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if len(set(args.temperatures)) != len(args.temperatures) or any(value < 0 for value in args.temperatures):
        raise ValueError("温度必须非负且不重复")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("种子必须不重复")
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_sha = sha256(checkpoint)
    environment = os.environ.copy()
    environment.update({name: "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    planned = [(temperature, seed) for temperature in args.temperatures for seed in args.seeds]
    progress_path = output_root / "progress.json"
    progress: dict[str, Any] = {
        "status": "running",
        "completed": [],
        "failed": None,
        "planned_runs": len(planned),
    }
    write_json(
        output_root / "launch_manifest.json",
        {
            "run_type": "initial_schedule_temperature_sensitivity",
            "method": "HB-GAT-PPO",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "dataset": f"real_{args.dataset}",
            "temperatures": args.temperatures,
            "seeds": args.seeds,
            "resume_policy": "仅跳过已有且 complete、合法、零硬约束违规的运行",
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    write_json(progress_path, progress)
    try:
        for temperature, seed in planned:
            run_name = f"{temperature_label(temperature)}_seed{seed}"
            run_dir = output_root / run_name
            previous = read_completed_run(run_dir) if run_dir.exists() else None
            if previous is not None:
                print(f"[跳过] 已审计完成：{run_name}", flush=True)
                progress["completed"].append({"run": run_name, "resumed": True})
                write_json(progress_path, progress)
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"[开始] {run_name} temperature={temperature} seed={seed}", flush=True)
            run_command(
                [
                    args.python_executable,
                    "-u",
                    "evaluate_model.py",
                    "experiment=initial_main_real4_async_selection",
                    f"model_path={checkpoint}",
                    f"test_data={data_path}",
                    "num_runs=1",
                    f"temperature={temperature}",
                    "scenario=standard",
                    f"seed={seed}",
                    "no_gantt=true",
                    "verbose_eval_progress=true",
                    f"output_dir={run_dir}",
                ],
                run_dir / "evaluation.log",
                environment,
            )
            run_command(
                [
                    args.python_executable,
                    "-u",
                    "scripts/validate_initial_schedule.py",
                    "--data",
                    str(data_path),
                    "--schedule",
                    str(run_dir / "schedule.csv"),
                    "--output",
                    str(run_dir / "legality_audit.json"),
                ],
                run_dir / "legality_audit.log",
                environment,
            )
            read_completed_run(run_dir)
            progress["completed"].append({"run": run_name, "resumed": False})
            write_json(progress_path, progress)
            print(f"[完成] {run_name}", flush=True)
    except Exception as exc:
        progress["status"] = "failed"
        progress["failed"] = f"{type(exc).__name__}: {exc}"
        write_json(progress_path, progress)
        raise

    detail_rows: list[dict[str, Any]] = []
    for temperature, seed in planned:
        run_name = f"{temperature_label(temperature)}_seed{seed}"
        result = read_completed_run(output_root / run_name)
        assert result is not None
        detail_rows.append({"temperature": temperature, "seed": seed, "run_name": run_name, **result})
    with (output_root / "runs_detail.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)
    summary_rows: list[dict[str, Any]] = []
    for temperature in args.temperatures:
        values = [row["makespan"] for row in detail_rows if row["temperature"] == temperature]
        summary_rows.append(
            {
                "temperature": temperature,
                "runs": len(values),
                "makespan_mean": mean(values),
                "makespan_std_sample": stdev(values) if len(values) > 1 else 0.0,
                "makespan_min": min(values),
                "makespan_max": max(values),
                "makespan_cv": stdev(values) / mean(values) if len(values) > 1 else 0.0,
                "eligible_rate": mean(float(row["legal"]) for row in detail_rows if row["temperature"] == temperature),
                "complete_rate": mean(float(row["complete"]) for row in detail_rows if row["temperature"] == temperature),
                "max_hard_violation": max(int(row["max_hard_violation"]) for row in detail_rows if row["temperature"] == temperature),
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
            "dataset": f"real_{args.dataset}",
            "checkpoint_sha256": checkpoint_sha,
            "rows": summary_rows,
            "all_complete": all(row["complete"] for row in detail_rows),
            "all_legal": all(row["legal"] for row in detail_rows),
            "max_hard_violation": max(int(row["max_hard_violation"]) for row in detail_rows),
        },
    )
    write_json(
        output_root / "integrity_check.json",
        {
            "expected_run_count": len(planned),
            "observed_run_count": len(detail_rows),
            "all_complete": all(row["complete"] for row in detail_rows),
            "all_legal": all(row["legal"] for row in detail_rows),
            "max_hard_violation": max(int(row["max_hard_violation"]) for row in detail_rows),
            "checkpoint_sha256": checkpoint_sha,
        },
    )
    write_json(
        output_root / "run_manifest.json",
        {
            "run_type": "initial_schedule_temperature_sensitivity",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "dataset": f"real_{args.dataset}",
            "temperatures": args.temperatures,
            "seeds": args.seeds,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    (output_root / "README.md").write_text(
        "# HB-GAT-PPO 初始调度温度敏感性验证\n\n"
        "- 固定 checkpoint、数据集和种子，仅改变解码温度。\n"
        "- 每次均导出任务级 schedule，并用 `validate_initial_schedule.py` 回放审核。\n"
        "- 脚本可续跑，但只会跳过已完成、合法且零硬约束违规的运行。\n",
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
    print(f"[全部完成] {len(detail_rows)}/{len(planned)} 次温度敏感性验证与合法性审计完成", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
