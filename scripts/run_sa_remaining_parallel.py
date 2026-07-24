"""并行运行初始调度 SA 的剩余大规模实例。

每个数据集使用一个独立 Python 子进程；默认两个进程、每进程 4 个数值线程，
从而在 16 核本地电脑上至少提供 8 个可用 CPU 线程。SA 本身的迭代循环仍是
单实例串行的，不能把一个固定 seed 的搜索拆成多个不等价的子任务。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("2338", "3182")


def _complete(output_dir: Path, dataset: str) -> bool:
    """仅在排程和指标都存在时判定该实例已完成。"""
    result_dir = output_dir / "SA" / dataset
    return (result_dir / "metrics.json").is_file() and (result_dir / "schedule.csv").is_file()


def _build_command(dataset: str, output_dir: Path, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-u",
        "baselines/heuristic/run_all_baselines.py",
        "--config",
        args.config,
        "--data_dir",
        "data",
        "--datasets",
        f"{dataset}.csv",
        "--methods",
        "SA",
        "--sa_iterations",
        str(args.sa_iterations),
        "--sa_initial_temp",
        str(args.sa_initial_temp),
        "--sa_cooling",
        str(args.sa_cooling),
        "--sa_min_temp",
        str(args.sa_min_temp),
        "--balance_weight",
        str(args.balance_weight),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(output_dir),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="并行运行 real_2338/real_3182 的初始调度 SA")
    parser.add_argument(
        "--output-root",
        default="results/02_initial_baselines/initial_search_high_budget_fiveskill_seed42_20260717_111949/SA_pending",
        help="两个实例的独立输出根目录",
    )
    parser.add_argument("--threads-per-job", type=int, default=4)
    parser.add_argument("--sa-iterations", type=int, default=120)
    parser.add_argument("--sa-initial-temp", type=float, default=0.05)
    parser.add_argument("--sa-cooling", type=float, default=0.96)
    parser.add_argument("--sa-min-temp", type=float, default=0.0001)
    parser.add_argument("--balance-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="conf/experiment/initial_schedule_680.yaml")
    args = parser.parse_args()

    if args.threads_per_job < 1:
        raise ValueError("--threads-per-job 必须为正数")

    output_root = (PROJECT_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(args.threads_per_job)
    env["PYTHONUNBUFFERED"] = "1"

    processes: dict[str, tuple[subprocess.Popen[str], object]] = {}
    for dataset in DATASETS:
        dataset_output = output_root / dataset
        dataset_output.mkdir(parents=True, exist_ok=True)
        if _complete(dataset_output, dataset):
            print(f"[跳过] {dataset} 已有完整 metrics.json 和 schedule.csv", flush=True)
            continue

        log_path = dataset_output / "process.log"
        log_file = log_path.open("a", encoding="utf-8", buffering=1)
        command = _build_command(dataset, dataset_output, args)
        print(f"[启动] {dataset}: {' '.join(command)}", flush=True)
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes[dataset] = (process, log_file)

    if not processes:
        print("没有需要运行的实例。", flush=True)
        return 0

    print(
        f"[并行状态] 进程数={len(processes)}，每进程数值线程={args.threads_per_job}，"
        f"理论线程数={len(processes) * args.threads_per_job}",
        flush=True,
    )
    while processes:
        for dataset, (process, log_file) in list(processes.items()):
            code = process.poll()
            if code is None:
                continue
            log_file.close()
            print(f"[结束] {dataset}: returncode={code}", flush=True)
            processes.pop(dataset)
        time.sleep(10)

    failed = [dataset for dataset in DATASETS if not _complete(output_root / dataset, dataset)]
    if failed:
        print(f"[未完成] {', '.join(failed)}；请检查对应 process.log 后重新运行。", flush=True)
        return 1
    print("[完成] 两个实例均生成 metrics.json 和 schedule.csv。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
