from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
INITIAL_CKPT_DIR = PROJECT_ROOT / "checkpoints" / "initial_schedule"
DEFAULT_DATASETS = ["283.csv", "680.csv", "2338.csv", "3182.csv"]

HBGAT_TARGETS = {
    "no_gat": {
        "checkpoint": INITIAL_CKPT_DIR / "680_no_gat.ckpt",
        "extra": ["--set", "ablation_no_gat=true"],
        "datasets": DEFAULT_DATASETS,
    },
    "no_pointer": {
        "checkpoint": INITIAL_CKPT_DIR / "680_no_pointer.ckpt",
        "extra": ["--set", "ablation_no_pointer=true"],
        "datasets": DEFAULT_DATASETS,
    },
}

FLAT_TARGETS = {
    "ppo": {
        "algorithm": "basic_ppo",
        "checkpoints": {
            "283": INITIAL_CKPT_DIR / "283_ppo.pth",
            "680": INITIAL_CKPT_DIR / "680_ppo.pth",
            "2338": INITIAL_CKPT_DIR / "2338_ppo.pth",
            "3182": INITIAL_CKPT_DIR / "3182_ppo.pth",
        },
    },
    "dqn": {
        "algorithm": "dqn",
        "checkpoints": {
            "283": INITIAL_CKPT_DIR / "283_dqn.pth",
            "680": INITIAL_CKPT_DIR / "680_dqn.pth",
            "2338": INITIAL_CKPT_DIR / "2338_dqn.pth",
            "3182": INITIAL_CKPT_DIR / "3182_dqn.pth",
        },
    },
}

FULL_MISSING_TARGET = {
    "method": "full",
    "dataset": "2338.csv",
    "preferred_checkpoint": INITIAL_CKPT_DIR / "2338.ckpt",
    "fallback_checkpoint": INITIAL_CKPT_DIR / "680.ckpt",
}

FIELDS = [
    "method",
    "method_family",
    "dataset",
    "status",
    "wall_time_sec",
    "inference_time_sec",
    "makespan",
    "valid",
    "checkpoint_path",
    "checkpoint_note",
    "train_estimate_h",
    "train_plus_eval_wall_h",
    "command",
    "raw_output_dir",
    "error",
]


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    wall_time_sec: float
    stdout_path: Path
    stderr_path: Path


def dataset_stem(dataset: str) -> str:
    return Path(dataset).stem


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return str(timedelta(seconds=int(round(seconds))))


def fmt_num(value: Any, digits: int = 3) -> str:
    if value in ("", None):
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def command_to_string(command: list[str]) -> str:
    return " ".join(command)


def valid_thread_env(env: dict[str, str]) -> dict[str, str]:
    fixed = dict(env)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        value = fixed.get(name)
        if value is None:
            continue
        pieces = [part.strip() for part in str(value).split(",")]
        if not pieces or any(not part.isdigit() or int(part) <= 0 for part in pieces):
            fixed[name] = "1"
    fixed.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    fixed.setdefault("MPLBACKEND", "Agg")
    fixed.setdefault("PYTHONUNBUFFERED", "1")
    return fixed


def tail_text(path: Path, max_chars: int = 3000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_chars), os.SEEK_SET)
        return handle.read().decode("utf-8", errors="replace")


def run_command(command: list[str], raw_dir: Path, timeout_sec: int, progress_interval_sec: float) -> CommandResult:
    raw_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = raw_dir / "stdout.log"
    stderr_path = raw_dir / "stderr.log"
    (raw_dir / "command.json").write_text(
        json.dumps(
            {
                "command": command,
                "command_string": command_to_string(command),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    start = time.perf_counter()
    print(f"  命令: {command_to_string(command)}", flush=True)
    print(f"  日志: {raw_dir}", flush=True)
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_file:
        with stderr_path.open("w", encoding="utf-8", errors="replace") as stderr_file:
            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                env=valid_thread_env(os.environ),
            )
            deadline = None if timeout_sec <= 0 else start + float(timeout_sec)
            next_progress = start + max(1.0, progress_interval_sec)
            while process.poll() is None:
                now = time.perf_counter()
                if deadline is not None and now >= deadline:
                    process.kill()
                    process.wait()
                    stderr_file.write(f"\n[benchmark] command timeout after {timeout_sec} sec\n")
                    break
                if progress_interval_sec > 0 and now >= next_progress:
                    print(f"  运行中: {fmt_duration(now - start)}", flush=True)
                    next_progress = now + max(1.0, progress_interval_sec)
                time.sleep(0.5)
    return CommandResult(
        returncode=int(process.returncode),
        wall_time_sec=float(time.perf_counter() - start),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def add_bucket_config(command: list[str], stem: str) -> None:
    bucket = PROJECT_ROOT / "conf" / "env" / f"initial_bucket_{stem}.yaml"
    if bucket.exists():
        command.extend(["--config", rel_or_abs(bucket)])


def parse_hbgat_summary(raw_dir: Path) -> dict[str, Any]:
    path = raw_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"缺少 summary.json: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    scheduled_tasks = int(data.get("scheduled_tasks", 0) or 0)
    return {
        "inference_time_sec": data.get("duration_sec"),
        "makespan": data.get("makespan"),
        "valid": 1.0 if scheduled_tasks > 0 else 0.0,
    }


def parse_flat_metrics(method: str, stem: str, command_start_wall: float) -> dict[str, Any]:
    output_method = "BasicPPO" if method == "ppo" else "DQN"
    path = PROJECT_ROOT / "results" / "eval_logs" / output_method / stem / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"缺少 metrics.json: {path}")
    if path.stat().st_mtime + 1.0 < command_start_wall:
        raise RuntimeError(f"metrics.json 可能是旧文件: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "inference_time_sec": data.get("inference_time"),
        "makespan": data.get("makespan"),
        "valid": data.get("valid"),
    }


def hbgat_command(
    *,
    method: str,
    dataset: str,
    checkpoint: Path,
    raw_dir: Path,
    args: argparse.Namespace,
    extra: list[str],
) -> list[str]:
    dataset_path = Path(args.data_dir) / dataset
    command = [
        args.python,
        str(PROJECT_ROOT / "evaluate_model.py"),
        "--config",
        args.config,
    ]
    add_bucket_config(command, dataset_stem(dataset))
    command.extend(
        [
            "--model-path",
            str(checkpoint),
            "--test-data",
            str(dataset_path),
            "--output-dir",
            str(raw_dir),
            "--num-runs",
            str(args.num_runs),
            "--temperature",
            str(args.temperature),
            "--scenario",
            "standard",
            "--no-gantt",
            "--seed",
            str(args.seed),
            "--set",
            "enable_multi_benchmark_eval=false",
            "--set",
            "verbose_eval_progress=false",
        ]
    )
    command.extend(extra)
    return command


def flat_command(
    *,
    method: str,
    dataset: str,
    checkpoint: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        args.python,
        str(PROJECT_ROOT / "baselines" / "evaluate_flat_rl_baseline.py"),
        "--algorithm",
        str(FLAT_TARGETS[method]["algorithm"]),
        "--model-path",
        str(checkpoint),
        "--config",
        args.config,
        "--data-dir",
        str(args.data_dir),
        "--datasets",
        dataset,
        "--seed",
        str(args.seed),
    ]
    if args.pass_flat_sampling_args:
        command.extend(["--num-runs", str(args.num_runs), "--temperature", str(args.temperature)])
    return command


def missing_row(method: str, family: str, dataset: str, checkpoint: Path, raw_dir: Path, error: str) -> dict[str, Any]:
    return {
        "method": method,
        "method_family": family,
        "dataset": dataset_stem(dataset),
        "status": "skipped",
        "checkpoint_path": str(checkpoint),
        "raw_output_dir": str(raw_dir),
        "error": error,
    }


def evaluate_hbgat(
    method: str,
    dataset: str,
    checkpoint: Path,
    checkpoint_note: str,
    extra: list[str],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    stem = dataset_stem(dataset)
    raw_dir = output_dir / "raw" / method / stem
    if not checkpoint.exists():
        return missing_row(method, "hbgat", dataset, checkpoint, raw_dir, f"模型不存在: {checkpoint}")
    command = hbgat_command(
        method=method,
        dataset=dataset,
        checkpoint=checkpoint,
        raw_dir=raw_dir,
        args=args,
        extra=extra,
    )
    print(f"\n[HBGAT] {method} @ {dataset}", flush=True)
    result = run_command(command, raw_dir, args.command_timeout_sec, args.progress_interval_sec)
    row: dict[str, Any] = {
        "method": method,
        "method_family": "hbgat",
        "dataset": stem,
        "status": "ok" if result.returncode == 0 else "failed",
        "wall_time_sec": result.wall_time_sec,
        "checkpoint_path": str(checkpoint),
        "checkpoint_note": checkpoint_note,
        "command": command_to_string(command),
        "raw_output_dir": str(raw_dir),
    }
    if result.returncode == 0:
        try:
            row.update(parse_hbgat_summary(raw_dir))
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
    else:
        row["error"] = tail_text(result.stderr_path)
    print_result(row)
    return row


def evaluate_flat(method: str, dataset: str, checkpoint: Path, args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    stem = dataset_stem(dataset)
    raw_dir = output_dir / "raw" / method / stem
    if not checkpoint.exists():
        return missing_row(method, f"flat_{method}", dataset, checkpoint, raw_dir, f"模型不存在: {checkpoint}")
    command = flat_command(method=method, dataset=dataset, checkpoint=checkpoint, args=args)
    print(f"\n[Flat] {method} @ {dataset}", flush=True)
    start_wall = time.time()
    result = run_command(command, raw_dir, args.command_timeout_sec, args.progress_interval_sec)
    row: dict[str, Any] = {
        "method": method,
        "method_family": f"flat_{method}",
        "dataset": stem,
        "status": "ok" if result.returncode == 0 else "failed",
        "wall_time_sec": result.wall_time_sec,
        "checkpoint_path": str(checkpoint),
        "checkpoint_note": "",
        "command": command_to_string(command),
        "raw_output_dir": str(raw_dir),
    }
    if result.returncode == 0:
        try:
            row.update(parse_flat_metrics(method, stem, start_wall))
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
    else:
        row["error"] = tail_text(result.stderr_path)
    print_result(row)
    return row


def print_result(row: dict[str, Any]) -> None:
    print(
        "  结果: "
        f"status={row.get('status')} "
        f"wall={fmt_duration(row.get('wall_time_sec'))} "
        f"infer={fmt_duration(row.get('inference_time_sec'))} "
        f"mk={fmt_num(row.get('makespan'), 2)} "
        f"valid={fmt_num(row.get('valid'), 3)}",
        flush=True,
    )
    if row.get("error"):
        print(f"  错误: {str(row['error']).strip()[:500]}", flush=True)


def load_train_estimates(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    estimates: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            method = str(row.get("method", ""))
            value = row.get("estimated_full_train_time_h")
            if not method or value in ("", None):
                continue
            try:
                estimates[method] = float(value)
            except ValueError:
                continue
    return estimates


def attach_train_estimate(row: dict[str, Any], train_estimates: dict[str, float]) -> None:
    method = str(row.get("method", ""))
    train_h = train_estimates.get(method, "")
    row["train_estimate_h"] = train_h
    wall = row.get("wall_time_sec")
    if train_h == "" or wall in ("", None):
        row["train_plus_eval_wall_h"] = ""
        return
    try:
        row["train_plus_eval_wall_h"] = float(train_h) + float(wall) / 3600.0
    except (TypeError, ValueError):
        row["train_plus_eval_wall_h"] = ""


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def write_markdown(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    failed_rows = [row for row in rows if row.get("status") != "ok"]
    lines = [
        "# 初始调度失败项耗时补测结果",
        "",
        f"- 生成时间: {datetime.now().isoformat(timespec='seconds')}",
        f"- config: `{args.config}`",
        f"- data_dir: `{args.data_dir}`",
        f"- num_runs: `{args.num_runs}`",
        f"- temperature: `{args.temperature}`",
        f"- 说明: 本脚本只补测上次失败或暂时不可用的验证耗时，不重新训练模型。",
        "",
        "## 汇总",
        "",
        "| method | dataset | status | wall(s) | infer(s) | makespan | valid | train(h) | train+eval(h) | checkpoint_note |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.get('method', '')} | {row.get('dataset', '')} | {row.get('status', '')} | "
            f"{fmt_num(row.get('wall_time_sec'))} | {fmt_num(row.get('inference_time_sec'))} | "
            f"{fmt_num(row.get('makespan'), 2)} | {fmt_num(row.get('valid'), 3)} | "
            f"{fmt_num(row.get('train_estimate_h'), 3)} | {fmt_num(row.get('train_plus_eval_wall_h'), 3)} | "
            f"{row.get('checkpoint_note', '')} |"
        )
    lines.extend(["", "## 可用性判断", ""])
    lines.append(f"- 成功: {len(ok_rows)} / {len(rows)}")
    lines.append(f"- 失败或跳过: {len(failed_rows)} / {len(rows)}")
    if failed_rows:
        lines.extend(["", "## 失败或跳过项", ""])
        for row in failed_rows:
            error = str(row.get("error", "")).strip().replace("\n", " ")
            lines.append(
                f"- `{row.get('method')}` @ `{row.get('dataset')}`: "
                f"{row.get('status')}，raw=`{row.get('raw_output_dir')}`，error={error[:300]}"
            )
    lines.extend(
        [
            "",
            "## 解释口径",
            "",
            "- `wall(s)` 是端到端命令耗时，包含 Python 启动、模型加载、数据加载和结果写盘。",
            "- `infer(s)` 是验证脚本内部记录的调度推理耗时，更接近算法运行时间。",
            "- `train_estimate(h)` 来自已有 `train_runtime_summary.csv`，不是本脚本重新训练得到。",
            "- `train+eval(h)` 是已有训练估算时间加本次补测 wall time。",
            "- `full@2338` 若缺少 `2338.ckpt`，默认使用 `680.ckpt` 作为耗时估算 fallback，并在 `checkpoint_note` 中标记。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="补测初始调度上次失败项的验证耗时")
    parser.add_argument("--config", default="conf/experiment/scale_400_800_schedule.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--command-timeout-sec", type=int, default=0)
    parser.add_argument("--progress-interval-sec", type=float, default=30.0)
    parser.add_argument("--pass-flat-sampling-args", action="store_true")
    parser.add_argument("--include-full-2338", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--previous-train-summary",
        default="results/runtime_benchmark/initial_train_eval_temperature0/train_runtime_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/runtime_benchmark/initial_failed_runtime_supplement",
    )
    parser.add_argument(
        "--docs-stem",
        default="initial_failed_runtime_supplement",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    train_summary = Path(args.previous_train_summary)
    if not train_summary.is_absolute():
        train_summary = PROJECT_ROOT / train_summary
    train_estimates = load_train_estimates(train_summary)

    rows: list[dict[str, Any]] = []
    datasets = [str(dataset) for dataset in args.datasets]

    print("=" * 88, flush=True)
    print("初始调度失败项验证耗时补测", flush=True)
    print(f"datasets={datasets}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"docs={DOCS_DIR / (args.docs_stem + '.md')}", flush=True)
    print("=" * 88, flush=True)

    for method, spec in HBGAT_TARGETS.items():
        for dataset in datasets:
            row = evaluate_hbgat(
                method=method,
                dataset=dataset,
                checkpoint=Path(spec["checkpoint"]),
                checkpoint_note="消融模型使用 680 checkpoint 跨规模验证",
                extra=list(spec["extra"]),
                args=args,
                output_dir=output_dir,
            )
            attach_train_estimate(row, train_estimates)
            rows.append(row)

    for method, spec in FLAT_TARGETS.items():
        checkpoints = spec["checkpoints"]
        for dataset in datasets:
            stem = dataset_stem(dataset)
            row = evaluate_flat(method, dataset, Path(checkpoints[stem]), args, output_dir)
            attach_train_estimate(row, train_estimates)
            rows.append(row)

    if args.include_full_2338:
        preferred = Path(FULL_MISSING_TARGET["preferred_checkpoint"])
        fallback = Path(FULL_MISSING_TARGET["fallback_checkpoint"])
        if preferred.exists():
            checkpoint = preferred
            note = "使用 2338 checkpoint"
        else:
            checkpoint = fallback
            note = "2338 checkpoint 缺失，使用 680 checkpoint 估算 2338 验证耗时"
        row = evaluate_hbgat(
            method="full",
            dataset=str(FULL_MISSING_TARGET["dataset"]),
            checkpoint=checkpoint,
            checkpoint_note=note,
            extra=[],
            args=args,
            output_dir=output_dir,
        )
        attach_train_estimate(row, train_estimates)
        rows.append(row)

    csv_path = DOCS_DIR / f"{args.docs_stem}.csv"
    md_path = DOCS_DIR / f"{args.docs_stem}.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows, args)
    print("\n完成", flush=True)
    print(f"CSV: {csv_path}", flush=True)
    print(f"Markdown: {md_path}", flush=True)


if __name__ == "__main__":
    main()
