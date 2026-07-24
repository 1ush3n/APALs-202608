"""并行验证已有初始调度模型的四数据集统一脚本。

每个 checkpoint 使用一个独立验证进程；同一模型的四个数据集按顺序验证，
避免同一 checkpoint 在 GPU 上重复加载。默认只发现当前格式可用的 best.ckpt，
旧 format_version=1 或损坏 checkpoint 会被跳过并记录在 manifest 中。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 服务器训练命令的默认输出位置；归档结果可通过 --roots 显式加入。
DEFAULT_ROOTS = (PROJECT_ROOT / "runs" / "scale_400_800_schedule",)
DATASETS = ("283", "680", "2338", "3182")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_checkpoints(roots: tuple[Path, ...]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        paths.extend(path for path in root.rglob("best.ckpt") if path.is_file())
    return sorted(set(path.resolve() for path in paths))


def checkpoint_status(path: Path) -> tuple[bool, str]:
    """只做格式预检，不构造模型，防止损坏文件进入并行队列。"""
    probe = (
        "import sys; from runtime.checkpoints import load_checkpoint; "
        "c=load_checkpoint(sys.argv[1], map_location='cpu'); "
        "print(c.format_name)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(path)],
        cwd=PROJECT_ROOT,
        env=base_environment(),
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        return True, completed.stdout.strip() or "loadable"
    error = (completed.stderr or completed.stdout).strip().replace("\n", " ")
    return False, error[-500:]


def base_environment() -> dict[str, str]:
    env = os.environ.copy()
    # 防止每个模型进程再创建大量 BLAS 线程，线程数由模型进程级并行控制。
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def model_name(checkpoint: Path) -> str:
    # checkpoints/best.ckpt 的父目录通常是模型目录；保留相对路径避免重名。
    return checkpoint.parent.parent.relative_to(PROJECT_ROOT).as_posix().replace("/", "__")


def run_one_model(
    checkpoint: Path,
    output_root: Path,
    experiment: str,
    timestamp: str,
    datasets: tuple[str, ...],
    no_gantt: bool,
) -> dict[str, Any]:
    name = model_name(checkpoint)
    model_output = output_root / name
    model_output.mkdir(parents=True, exist_ok=True)
    log_path = model_output / "worker.log"
    status: dict[str, Any] = {
        "model": name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "datasets": list(datasets),
        "temperature": 0.0,
        "scenario": "standard",
        "started_at": dt.datetime.now().astimezone().isoformat(),
        "results": [],
    }
    env = base_environment()
    try:
        # 使用追加模式；在同一输出目录补跑单个数据集时不丢失此前日志。
        with log_path.open("a", encoding="utf-8") as log:
            for dataset in datasets:
                data_path = PROJECT_ROOT / "data" / f"{dataset}.csv"
                result_dir = model_output / f"real_{dataset}"
                result_dir.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable,
                    "evaluate_model.py",
                    f"experiment={experiment}",
                    f"model_path={checkpoint}",
                    f"test_data={data_path}",
                    "num_runs=1",
                    "temperature=0.0",
                    "scenario=standard",
                    "seed=42",
                    f"output_dir={result_dir}",
                ]
                if no_gantt:
                    command.append("no_gantt=true")
                log.write(f"\n[START] dataset={dataset} command={' '.join(map(str, command))}\n")
                log.flush()
                started = time.monotonic()
                completed = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                item = {
                    "dataset": dataset,
                    "returncode": int(completed.returncode),
                    "output_dir": str(result_dir),
                    "duration_sec": round(time.monotonic() - started, 3),
                }
                summary_path = result_dir / "summary.json"
                if summary_path.exists():
                    try:
                        item["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError as exc:
                        item["summary_error"] = str(exc)
                status["results"].append(item)
                if completed.returncode != 0:
                    status["failed"] = True
                    break
        status.setdefault("failed", False)
    except Exception as exc:  # noqa: BLE001 - worker 必须将异常写入汇总而不是静默退出
        status["failed"] = True
        status["exception"] = repr(exc)
    status["finished_at"] = dt.datetime.now().astimezone().isoformat()
    old_status_path = model_output / "status.json"
    if old_status_path.exists():
        try:
            old_status = json.loads(old_status_path.read_text(encoding="utf-8"))
            old_results = {
                str(item.get("dataset")): item
                for item in old_status.get("results", [])
            }
            old_results.update({
                str(item.get("dataset")): item
                for item in status.get("results", [])
            })
            status["results"] = [old_results[key] for key in sorted(old_results)]
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    (model_output / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="并行验证已有初始调度 checkpoint 的四实例温度0结果")
    parser.add_argument(
        "--roots",
        nargs="*",
        default=[str(path) for path in DEFAULT_ROOTS],
        help="checkpoint 搜索根目录；默认 runs/scale_400_800_schedule",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="输出根目录；默认写入 results/01_initial_main/unified_eval_parallel_<时间>",
    )
    parser.add_argument("--experiment", default="scale_400_800_schedule")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="只验证指定数据集；默认 283 680 2338 3182",
    )
    parser.add_argument("--max-workers", type=int, default=0, help="并行模型进程数；0 表示等于可用模型数")
    parser.add_argument("--no-gantt", action="store_true", help="不生成 Gantt 图，降低磁盘和绘图开销")
    parser.add_argument("--dry-run", action="store_true", help="只发现和预检 checkpoint，不启动验证")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = tuple(Path(value).resolve() for value in args.roots)
    checkpoints = discover_checkpoints(roots)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else PROJECT_ROOT / "results" / "01_initial_main" / f"unified_eval_parallel_{timestamp}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "project_root": str(PROJECT_ROOT),
        "roots": [str(root) for root in roots],
        "datasets": list(args.datasets),
        "temperature": 0.0,
        "scenario": "standard",
        "experiment": args.experiment,
        "checkpoint_candidates": [],
        "results": [],
    }
    valid: list[Path] = []
    for checkpoint in checkpoints:
        ok, detail = checkpoint_status(checkpoint)
        row = {"checkpoint": str(checkpoint), "loadable": ok, "detail": detail}
        manifest["checkpoint_candidates"].append(row)
        print(f"[预检] {'可用' if ok else '跳过'} {checkpoint} :: {detail}", flush=True)
        if ok:
            valid.append(checkpoint)
    if args.dry_run:
        manifest["dry_run"] = True
        (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    if not valid:
        print("[错误] 没有找到可加载的 best.ckpt。", file=sys.stderr)
        return 2

    max_workers = len(valid) if args.max_workers <= 0 else min(args.max_workers, len(valid))
    manifest["max_workers"] = max_workers
    manifest["output_root"] = str(output_root)
    print(f"[启动] 模型数={len(valid)} 并行进程数={max_workers} 输出={output_root}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_one_model,
                checkpoint,
                output_root,
                args.experiment,
                timestamp,
                tuple(args.datasets),
                args.no_gantt,
            )
            for checkpoint in valid
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            manifest["results"].append(result)
            print(
                f"[完成] {result['model']} failed={result.get('failed', True)} "
                f"datasets={len(result.get('results', []))}",
                flush=True,
            )
    # 增量运行时保留已有数据集结果，只替换本次重新验证的数据集。
    existing_manifest_path = output_root / "manifest.json"
    if existing_manifest_path.exists():
        try:
            existing = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
            old_by_model = {str(row.get("model")): row for row in existing.get("results", [])}
            for row in manifest["results"]:
                key = str(row.get("model"))
                previous = old_by_model.get(key)
                if previous:
                    old_by_dataset = {
                        str(item.get("dataset")): item
                        for item in previous.get("results", [])
                    }
                    old_by_dataset.update({
                        str(item.get("dataset")): item
                        for item in row.get("results", [])
                    })
                    row["results"] = [old_by_dataset[key] for key in sorted(old_by_dataset)]
            merged = dict(existing)
            merged.update(manifest)
            manifest = merged
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    manifest["finished_at"] = dt.datetime.now().astimezone().isoformat()
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = [row for row in manifest["results"] if row.get("failed")]
    print(f"[汇总] 成功模型={len(valid) - len(failed)} 失败模型={len(failed)} manifest={output_root / 'manifest.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
