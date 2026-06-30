from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import configs
from runtime.artifacts import (
    resolve_run_output_dir,
    write_run_context_files,
    write_run_manifest,
)
from runtime.configuration import (
    add_common_config_arguments,
    parse_runtime_args,
    resolve_runtime_config,
)

INITIAL_CKPT_DIR = PROJECT_ROOT / "checkpoints" / "initial_schedule"

DEFAULT_DATASETS = ["283.csv", "680.csv", "2338.csv", "3182.csv"]
DEFAULT_METHODS = [
    "full",
    "no_gat",
    "no_pointer",
    "no_attention_pooling",
    "ppo",
    "dqn",
    "SPT",
    "LPT",
    "Random",
    "EDD",
    "CPM",
    "MSL",
    "GA",
    "Beam",
    "IG",
    "SA",
]

HBGAT_SCALE_CHECKPOINTS = {
    "283": INITIAL_CKPT_DIR / "283.ckpt",
    "680": INITIAL_CKPT_DIR / "680.ckpt",
    "2338": INITIAL_CKPT_DIR / "2338.ckpt",
    "3182": INITIAL_CKPT_DIR / "3182.ckpt",
}

PPO_SCALE_CHECKPOINTS = {
    "283": INITIAL_CKPT_DIR / "283_ppo.pth",
    "680": INITIAL_CKPT_DIR / "680_ppo.pth",
    "2338": INITIAL_CKPT_DIR / "2338_ppo.pth",
    "3182": INITIAL_CKPT_DIR / "3182_ppo.pth",
}

DQN_SCALE_CHECKPOINTS = {
    "283": INITIAL_CKPT_DIR / "283_dqn.pth",
    "680": INITIAL_CKPT_DIR / "680_dqn.pth",
    "2338": INITIAL_CKPT_DIR / "2338_dqn.pth",
    "3182": INITIAL_CKPT_DIR / "3182_dqn.pth",
}

HBGAT_ABLATION_CHECKPOINTS = {
    "no_gat": INITIAL_CKPT_DIR / "680_no_gat.ckpt",
    "no_pointer": INITIAL_CKPT_DIR / "680_no_pointer.ckpt",
    "no_attention_pooling": INITIAL_CKPT_DIR / "680_no-attention-pooling.ckpt",
}

HBGAT_ABLATION_ARGS = {
    "no_gat": ["--ablation-no-gat"],
    "no_pointer": ["--ablation-no-pointer"],
    "no_attention_pooling": [
        "--set",
        "use_attention_critic=false",
        "--set",
        "use_shared_trunk=true",
        "--set",
        "use_autoregressive_worker=false",
    ],
}

HBGAT_TRAINABLE = {"full", "no_gat", "no_pointer", "no_attention_pooling"}
FLAT_TRAINABLE = {"ppo", "dqn"}
HEURISTIC_METHODS = {"SPT", "LPT", "Random", "EDD", "CPM", "MSL"}
SEARCH_METHODS = {"GA", "Beam", "IG", "SA"}

TRAIN_FIELDS = [
    "method",
    "method_family",
    "status",
    "probe_episodes",
    "target_max_episodes",
    "train_batch_size",
    "train_num_envs",
    "probe_wall_time_sec",
    "avg_probe_episode_time_sec",
    "estimated_full_train_time_sec",
    "estimated_full_train_time_h",
    "command",
    "raw_output_dir",
    "error",
]

EVAL_FIELDS = [
    "method",
    "method_family",
    "dataset",
    "dataset_path",
    "status",
    "temperature",
    "num_runs",
    "seed",
    "makespan",
    "balance_std",
    "worker_utilization",
    "station_utilization",
    "valid",
    "completion_rate",
    "inference_time_sec",
    "wall_time_sec",
    "checkpoint_path",
    "command",
    "raw_output_dir",
    "error",
]

COMBINED_FIELDS = [
    "method",
    "method_family",
    "dataset",
    "train_status",
    "estimated_full_train_time_h",
    "eval_status",
    "eval_inference_time_sec",
    "eval_wall_time_sec",
    "makespan",
    "valid",
    "checkpoint_path",
]


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    wall_time_sec: float
    stdout_path: Path
    stderr_path: Path


@dataclass(frozen=True)
class ProgressContext:
    phase: str
    item_label: str
    item_index: int
    total_items: int
    overall_start: float
    raw_dir: Path
    quiet: bool = False
    progress_interval_sec: float = 30.0
    show_log_tail_lines: int = 2


def fmt_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "估算中"
    seconds_float = float(seconds)
    if seconds_float < 0:
        return "估算中"
    if seconds_float < 60:
        return f"{seconds_float:.1f}s"
    return str(timedelta(seconds=int(round(seconds_float))))


def fmt_float(value: Any, digits: int = 2) -> str:
    if value in ("", None):
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_status(row: dict[str, Any]) -> str:
    status = str(row.get("status", "-"))
    if status == "ok":
        return "OK"
    if status == "failed":
        return "FAILED"
    if status == "skipped":
        return "SKIPPED"
    if status == "not_applicable":
        return "N/A"
    return status


def read_tail_lines(path: Path, line_count: int) -> str:
    if line_count <= 0 or not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - 65536), os.SEEK_SET)
        text = handle.read().decode("utf-8", errors="replace")
    lines = text.splitlines()
    return "\n".join(lines[-line_count:])


def overall_eta(completed_items: int, total_items: int, overall_start: float, now: float) -> float | None:
    if completed_items <= 0:
        return None
    avg = (now - overall_start) / float(completed_items)
    return avg * max(0, total_items - completed_items)


def print_header(args: argparse.Namespace, methods: list[str], datasets: list[str], output_dir: Path) -> None:
    train_count = len(methods) if args.mode in {"all", "train_probe"} else 0
    eval_count = len(methods) * len(datasets) if args.mode in {"all", "eval"} else 0
    total = train_count + eval_count
    print("=" * 88, flush=True)
    print("初始调度训练/验证耗时统计", flush=True)
    print(f"模式: {args.mode} | 总任务: {total} | 训练短测: {train_count} | 验证: {eval_count}", flush=True)
    print(f"方法: {', '.join(methods)}", flush=True)
    print(f"数据集: {', '.join(datasets)}", flush=True)
    print(
        "参数: "
        f"temperature={args.temperature} num_runs={args.num_runs} "
        f"train_probe_episodes={args.train_probe_episodes} target_max_episodes={args.target_max_episodes}",
        flush=True,
    )
    print(
        "训练短测安全参数: "
        f"hbgat_batch={args.hbgat_train_batch_size} flat_batch={args.flat_train_batch_size} "
        f"train_num_envs={args.train_num_envs}",
        flush=True,
    )
    print(f"输出目录: {output_dir}", flush=True)
    print("=" * 88, flush=True)


def print_task_start(context: ProgressContext, command: list[str]) -> None:
    if context.quiet:
        return
    now = time.perf_counter()
    eta = overall_eta(context.item_index - 1, context.total_items, context.overall_start, now)
    print(
        f"\n[{context.item_index}/{context.total_items}] 开始 {context.phase}: {context.item_label}",
        flush=True,
    )
    print(
        f"  已用: {fmt_duration(now - context.overall_start)} | 整体ETA: {fmt_duration(eta)} | 日志: {context.raw_dir}",
        flush=True,
    )
    print(f"  命令: {command_to_string(command)}", flush=True)


def print_task_heartbeat(context: ProgressContext, task_start: float, stdout_path: Path, stderr_path: Path) -> None:
    if context.quiet:
        return
    now = time.perf_counter()
    eta = overall_eta(context.item_index - 1, context.total_items, context.overall_start, now)
    print(
        f"[{context.item_index}/{context.total_items}] 运行中 {context.phase}: {context.item_label} | "
        f"本任务 {fmt_duration(now - task_start)} | 总已用 {fmt_duration(now - context.overall_start)} | "
        f"整体ETA {fmt_duration(eta)}",
        flush=True,
    )
    stdout_tail = read_tail_lines(stdout_path, context.show_log_tail_lines)
    stderr_tail = read_tail_lines(stderr_path, context.show_log_tail_lines)
    if stdout_tail:
        print("  stdout 尾部:", flush=True)
        for line in stdout_tail.splitlines():
            print(f"    {line}", flush=True)
    if stderr_tail:
        print("  stderr 尾部:", flush=True)
        for line in stderr_tail.splitlines():
            print(f"    {line}", flush=True)


def print_task_result(context: ProgressContext, row: dict[str, Any]) -> None:
    now = time.perf_counter()
    eta = overall_eta(context.item_index, context.total_items, context.overall_start, now)
    pieces = [
        f"[{context.item_index}/{context.total_items}] 完成 {context.phase}: {context.item_label}",
        f"status={fmt_status(row)}",
    ]
    if "probe_wall_time_sec" in row:
        pieces.append(f"短测耗时={fmt_duration(row.get('probe_wall_time_sec'))}")
        pieces.append(f"估算训练={fmt_float(row.get('estimated_full_train_time_h'), 3)}h")
    if "wall_time_sec" in row:
        pieces.append(f"wall={fmt_duration(row.get('wall_time_sec'))}")
        pieces.append(f"infer={fmt_duration(row.get('inference_time_sec'))}")
        pieces.append(f"Mk={fmt_float(row.get('makespan'), 2)}")
        pieces.append(f"valid={fmt_float(row.get('valid'), 3)}")
    print(" | ".join(pieces), flush=True)
    if row.get("error"):
        error = str(row.get("error", "")).strip().replace("\n", " | ")
        print(f"  错误摘要: {error[:500]}", flush=True)
    print(
        f"  总已用: {fmt_duration(now - context.overall_start)} | 整体ETA: {fmt_duration(eta)} | raw: {row.get('raw_output_dir', context.raw_dir)}",
        flush=True,
    )


def method_family(method: str) -> str:
    if method in HBGAT_TRAINABLE:
        return "hbgat"
    if method == "ppo":
        return "flat_ppo"
    if method == "dqn":
        return "flat_dqn"
    if method in HEURISTIC_METHODS:
        return "heuristic"
    if method in SEARCH_METHODS:
        return "search"
    return "unknown"


def dataset_stem(dataset: str) -> str:
    return Path(dataset).stem


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def resolve_data_path(data_dir: Path, dataset: str) -> Path:
    path = Path(dataset)
    if path.is_absolute():
        return path
    return data_dir / path


def add_bucket_config(command: list[str], stem: str) -> None:
    bucket = PROJECT_ROOT / "conf" / "env" / f"initial_bucket_{stem}.yaml"
    if bucket.exists():
        command.extend(["--config", rel_or_abs(bucket)])


def valid_thread_env(env: dict[str, str]) -> dict[str, str]:
    fixed = dict(env)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        value = fixed.get(name)
        if value is None:
            continue
        pieces = [part.strip() for part in str(value).split(",")]
        valid = bool(pieces) and all(part.isdigit() and int(part) > 0 for part in pieces)
        if not valid:
            fixed[name] = "1"
    fixed.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    fixed.setdefault("MPLBACKEND", "Agg")
    return fixed


def command_to_string(command: list[str]) -> str:
    return " ".join(command)


def tail_text(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def run_command(
    command: list[str],
    raw_dir: Path,
    timeout_sec: int = 0,
    progress: ProgressContext | None = None,
) -> CommandResult:
    raw_dir.mkdir(parents=True, exist_ok=True)
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
    stdout_path = raw_dir / "stdout.log"
    stderr_path = raw_dir / "stderr.log"
    start = time.perf_counter()
    if progress is not None:
        print_task_start(progress, command)
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_file:
        with stderr_path.open("w", encoding="utf-8", errors="replace") as stderr_file:
            env = valid_thread_env(os.environ)
            env.setdefault("PYTHONUNBUFFERED", "1")
            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                env=env,
            )
            deadline = None if timeout_sec <= 0 else start + float(timeout_sec)
            next_progress = start + max(1.0, float(progress.progress_interval_sec)) if progress else float("inf")
            while process.poll() is None:
                now = time.perf_counter()
                if deadline is not None and now >= deadline:
                    process.kill()
                    process.wait()
                    stderr_file.write(f"\n[benchmark] command timeout after {timeout_sec} sec\n")
                    break
                if (
                    progress is not None
                    and progress.progress_interval_sec > 0
                    and now >= next_progress
                ):
                    stdout_file.flush()
                    stderr_file.flush()
                    print_task_heartbeat(progress, start, stdout_path, stderr_path)
                    next_progress = now + max(1.0, float(progress.progress_interval_sec))
                time.sleep(0.5)
    wall_time = time.perf_counter() - start
    return CommandResult(
        command=command,
        returncode=int(process.returncode),
        wall_time_sec=float(wall_time),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def pivot_rows(
    rows: list[dict[str, Any]],
    *,
    row_key: str,
    col_key: str,
    value_key: str,
) -> list[dict[str, Any]]:
    row_names = sorted({str(row.get(row_key, "")) for row in rows if row.get(row_key, "") != ""})
    col_names = sorted({str(row.get(col_key, "")) for row in rows if row.get(col_key, "") != ""}, key=dataset_sort_key)
    by_key = {
        (str(row.get(row_key, "")), str(row.get(col_key, ""))): row.get(value_key, "")
        for row in rows
    }
    output: list[dict[str, Any]] = []
    for name in row_names:
        record: dict[str, Any] = {row_key: name}
        for col in col_names:
            record[col] = by_key.get((name, col), "")
        output.append(record)
    return output


def dataset_sort_key(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):08d}")
    except ValueError:
        return (1, value)


def write_dynamic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_paper_tables(output_dir: Path, train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> None:
    train_table = [
        {
            "method": row.get("method", ""),
            "method_family": row.get("method_family", ""),
            "status": row.get("status", ""),
            "probe_episodes": row.get("probe_episodes", ""),
            "train_batch_size": row.get("train_batch_size", ""),
            "train_num_envs": row.get("train_num_envs", ""),
            "probe_wall_time_sec": row.get("probe_wall_time_sec", ""),
            "avg_probe_episode_time_sec": row.get("avg_probe_episode_time_sec", ""),
            "target_max_episodes": row.get("target_max_episodes", ""),
            "estimated_full_train_time_h": row.get("estimated_full_train_time_h", ""),
        }
        for row in train_rows
    ]
    write_csv(
        output_dir / "paper_train_time_hours.csv",
        train_table,
        [
            "method",
            "method_family",
            "status",
            "probe_episodes",
            "train_batch_size",
            "train_num_envs",
            "probe_wall_time_sec",
            "avg_probe_episode_time_sec",
            "target_max_episodes",
            "estimated_full_train_time_h",
        ],
    )
    write_dynamic_csv(
        output_dir / "paper_eval_inference_time_seconds.csv",
        pivot_rows(eval_rows, row_key="method", col_key="dataset", value_key="inference_time_sec"),
    )
    write_dynamic_csv(
        output_dir / "paper_eval_wall_time_seconds.csv",
        pivot_rows(eval_rows, row_key="method", col_key="dataset", value_key="wall_time_sec"),
    )
    write_dynamic_csv(
        output_dir / "paper_makespan.csv",
        pivot_rows(eval_rows, row_key="method", col_key="dataset", value_key="makespan"),
    )
    write_dynamic_csv(
        output_dir / "paper_valid.csv",
        pivot_rows(eval_rows, row_key="method", col_key="dataset", value_key="valid"),
    )
    aggregate_rows: list[dict[str, Any]] = []
    methods = sorted({str(row.get("method", "")) for row in eval_rows if row.get("method", "")})
    for method in methods:
        method_eval = [row for row in eval_rows if row.get("method") == method]
        ok_eval = [row for row in method_eval if row.get("status") == "ok"]
        train_row = next((row for row in train_rows if row.get("method") == method), {})
        aggregate_rows.append(
            {
                "method": method,
                "method_family": method_family(method),
                "eval_ok": len(ok_eval),
                "eval_total": len(method_eval),
                "avg_eval_wall_time_sec": mean_numeric(ok_eval, "wall_time_sec"),
                "avg_eval_inference_time_sec": mean_numeric(ok_eval, "inference_time_sec"),
                "avg_makespan": mean_numeric(ok_eval, "makespan"),
                "estimated_full_train_time_h": train_row.get("estimated_full_train_time_h", ""),
                "train_status": train_row.get("status", ""),
            }
        )
    write_csv(
        output_dir / "method_runtime_aggregate.csv",
        aggregate_rows,
        [
            "method",
            "method_family",
            "eval_ok",
            "eval_total",
            "avg_eval_wall_time_sec",
            "avg_eval_inference_time_sec",
            "avg_makespan",
            "estimated_full_train_time_h",
            "train_status",
        ],
    )


def mean_numeric(rows: list[dict[str, Any]], key: str) -> float | str:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value in ("", None):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return ""
    return float(sum(values) / len(values))


def print_table(title: str, rows: list[dict[str, Any]], fields: list[str], max_rows: int = 40) -> None:
    print(f"\n{title}", flush=True)
    if not rows:
        print("  无数据", flush=True)
        return
    visible = rows[:max_rows]
    widths = {
        field: min(
            28,
            max(len(str(field)), *(len(fmt_cell(row.get(field, ""))) for row in visible)),
        )
        for field in fields
    }
    header = "  " + " | ".join(str(field).ljust(widths[field]) for field in fields)
    print(header, flush=True)
    print("  " + "-+-".join("-" * widths[field] for field in fields), flush=True)
    for row in visible:
        print(
            "  " + " | ".join(fmt_cell(row.get(field, "")).ljust(widths[field]) for field in fields),
            flush=True,
        )
    if len(rows) > max_rows:
        print(f"  ... 还有 {len(rows) - max_rows} 行，详见 CSV 文件", flush=True)


def fmt_cell(value: Any) -> str:
    if value in ("", None):
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_final_summary(
    output_dir: Path,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    total_wall_time_sec: float,
) -> None:
    all_rows = train_rows + eval_rows
    ok_count = sum(1 for row in all_rows if row.get("status") == "ok")
    failed_count = sum(1 for row in all_rows if row.get("status") == "failed")
    skipped_count = sum(1 for row in all_rows if row.get("status") in {"skipped", "not_applicable"})
    print("\n" + "=" * 88, flush=True)
    print("运行完成", flush=True)
    print(
        f"总耗时: {fmt_duration(total_wall_time_sec)} | OK={ok_count} | FAILED={failed_count} | SKIPPED/N/A={skipped_count}",
        flush=True,
    )
    print(f"汇总目录: {output_dir}", flush=True)

    print_table(
        "训练短测与完整训练耗时估算",
        train_rows,
        [
            "method",
            "status",
            "train_batch_size",
            "train_num_envs",
            "probe_wall_time_sec",
            "avg_probe_episode_time_sec",
            "estimated_full_train_time_h",
        ],
    )

    aggregate_rows: list[dict[str, Any]] = []
    methods = sorted({str(row.get("method", "")) for row in eval_rows if row.get("method", "")})
    for method in methods:
        method_rows = [row for row in eval_rows if row.get("method") == method]
        ok_rows = [row for row in method_rows if row.get("status") == "ok"]
        aggregate_rows.append(
            {
                "method": method,
                "ok/total": f"{len(ok_rows)}/{len(method_rows)}",
                "avg_wall_sec": mean_numeric(ok_rows, "wall_time_sec"),
                "avg_infer_sec": mean_numeric(ok_rows, "inference_time_sec"),
                "avg_makespan": mean_numeric(ok_rows, "makespan"),
            }
        )
    print_table("验证耗时与效果概览", aggregate_rows, ["method", "ok/total", "avg_wall_sec", "avg_infer_sec", "avg_makespan"])

    failed_rows = [
        {
            "method": row.get("method", ""),
            "dataset": row.get("dataset", ""),
            "status": row.get("status", ""),
            "raw_output_dir": row.get("raw_output_dir", ""),
            "error": str(row.get("error", ""))[:160],
        }
        for row in all_rows
        if row.get("status") in {"failed", "skipped"}
    ]
    print_table("失败或跳过任务", failed_rows, ["method", "dataset", "status", "raw_output_dir", "error"], max_rows=30)

    print("\n已写入关键文件:", flush=True)
    for name in (
        "train_runtime_summary.csv",
        "eval_runtime_summary.csv",
        "combined_runtime_summary.csv",
        "paper_train_time_hours.csv",
        "paper_eval_inference_time_seconds.csv",
        "paper_eval_wall_time_seconds.csv",
        "paper_makespan.csv",
        "paper_valid.csv",
        "method_runtime_aggregate.csv",
    ):
        print(f"  {output_dir / name}", flush=True)
    print("=" * 88, flush=True)


def hbgat_eval_checkpoint(method: str, stem: str) -> Path:
    if method == "full":
        return HBGAT_SCALE_CHECKPOINTS[stem]
    return HBGAT_ABLATION_CHECKPOINTS[method]


def flat_eval_checkpoint(method: str, stem: str) -> Path:
    if method == "ppo":
        return PPO_SCALE_CHECKPOINTS[stem]
    if method == "dqn":
        return DQN_SCALE_CHECKPOINTS[stem]
    raise ValueError(f"未知 flat baseline 方法: {method}")


def missing_row(
    *,
    method: str,
    dataset: str,
    dataset_path: Path,
    checkpoint_path: Path,
    args: argparse.Namespace,
    raw_dir: Path,
    error: str,
) -> dict[str, Any]:
    return {
        "method": method,
        "method_family": method_family(method),
        "dataset": dataset_stem(dataset),
        "dataset_path": str(dataset_path),
        "status": "skipped",
        "temperature": float(args.temperature),
        "num_runs": int(args.num_runs),
        "seed": int(args.seed),
        "checkpoint_path": str(checkpoint_path),
        "raw_output_dir": str(raw_dir),
        "error": error,
    }


def parse_hbgat_summary(raw_dir: Path) -> dict[str, Any]:
    summary_path = raw_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"缺少 summary.json: {summary_path}")
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    scheduled_tasks = int(data.get("scheduled_tasks", 0) or 0)
    return {
        "makespan": data.get("makespan"),
        "balance_std": data.get("balance_std"),
        "worker_utilization": data.get("worker_utilization"),
        "station_utilization": data.get("station_utilization"),
        "valid": 1.0 if scheduled_tasks > 0 else 0.0,
        "completion_rate": 1.0 if scheduled_tasks > 0 else 0.0,
        "inference_time_sec": data.get("duration_sec"),
    }


def parse_metrics_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"缺少 metrics.json: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "makespan": data.get("makespan"),
        "balance_std": data.get("workload_balance_std"),
        "worker_utilization": data.get("worker_utilization"),
        "station_utilization": data.get("station_utilization"),
        "valid": data.get("valid"),
        "completion_rate": data.get("completion_rate"),
        "inference_time_sec": data.get("inference_time"),
    }


def build_hbgat_eval_command(
    method: str,
    checkpoint_path: Path,
    dataset_path: Path,
    raw_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    stem = dataset_path.stem
    command = [
        args.python,
        str(PROJECT_ROOT / "evaluate_model.py"),
        "--config",
        args.config,
    ]
    add_bucket_config(command, stem)
    command.extend(
        [
            "--model-path",
            str(checkpoint_path),
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
    command.extend(HBGAT_ABLATION_ARGS.get(method, []))
    return command


def build_flat_eval_command(
    method: str,
    checkpoint_path: Path,
    dataset: str,
    args: argparse.Namespace,
    artifact_root: Path,
) -> list[str]:
    algorithm = "basic_ppo" if method == "ppo" else "dqn"
    return [
        args.python,
        str(PROJECT_ROOT / "baselines" / "evaluate_flat_rl_baseline.py"),
        "--algorithm",
        algorithm,
        "--model-path",
        str(checkpoint_path),
        "--config",
        args.config,
        "--data-dir",
        str(args.data_dir),
        "--datasets",
        dataset,
        "--num-runs",
        str(args.num_runs),
        "--temperature",
        str(args.temperature),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(artifact_root),
    ]


def build_heuristic_eval_command(
    method: str,
    dataset: str,
    args: argparse.Namespace,
    artifact_root: Path,
) -> list[str]:
    return [
        args.python,
        str(PROJECT_ROOT / "baselines" / "heuristic" / "run_all_baselines.py"),
        "--config",
        args.config,
        "--data_dir",
        str(args.data_dir),
        "--datasets",
        dataset,
        "--methods",
        method,
        "--random_runs",
        "1",
        "--ga_pop_size",
        str(args.ga_pop_size),
        "--ga_max_gen",
        str(args.ga_max_gen),
        "--beam_width",
        str(args.beam_width),
        "--beam_branch_factor",
        str(args.beam_branch_factor),
        "--beam_levels",
        str(args.beam_levels),
        "--beam_patience",
        str(args.beam_patience),
        "--ig_iterations",
        str(args.ig_iterations),
        "--ig_destroy_ratio",
        str(args.ig_destroy_ratio),
        "--ig_noise_sigma",
        str(args.ig_noise_sigma),
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
        str(artifact_root),
    ]


def run_hbgat_eval(
    method: str,
    dataset: str,
    args: argparse.Namespace,
    output_dir: Path,
    progress: ProgressContext | None = None,
) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    dataset_path = resolve_data_path(data_dir, dataset)
    stem = dataset_path.stem
    raw_dir = output_dir / "raw" / "eval" / method / stem
    checkpoint_path = hbgat_eval_checkpoint(method, stem)
    if not checkpoint_path.exists():
        error = f"模型不存在: {checkpoint_path}"
        if args.fail_on_missing:
            raise FileNotFoundError(error)
        return missing_row(
            method=method,
            dataset=dataset,
            dataset_path=dataset_path,
            checkpoint_path=checkpoint_path,
            args=args,
            raw_dir=raw_dir,
            error=error,
        )
    command = build_hbgat_eval_command(method, checkpoint_path, dataset_path, raw_dir, args)
    result = run_command(command, raw_dir, args.command_timeout_sec, progress)
    row = {
        "method": method,
        "method_family": method_family(method),
        "dataset": stem,
        "dataset_path": str(dataset_path),
        "temperature": float(args.temperature),
        "num_runs": int(args.num_runs),
        "seed": int(args.seed),
        "wall_time_sec": result.wall_time_sec,
        "checkpoint_path": str(checkpoint_path),
        "command": command_to_string(command),
        "raw_output_dir": str(raw_dir),
    }
    if result.returncode == 0:
        try:
            row.update(parse_hbgat_summary(raw_dir))
            row["status"] = "ok"
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
    else:
        row["status"] = "failed"
        row["error"] = tail_text(result.stderr_path)
    return row


def run_flat_eval(
    method: str,
    dataset: str,
    args: argparse.Namespace,
    output_dir: Path,
    progress: ProgressContext | None = None,
) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    dataset_path = resolve_data_path(data_dir, dataset)
    stem = dataset_path.stem
    raw_dir = output_dir / "raw" / "eval" / method / stem
    artifact_root = raw_dir / "artifacts"
    checkpoint_path = flat_eval_checkpoint(method, stem)
    if not checkpoint_path.exists():
        error = f"模型不存在: {checkpoint_path}"
        if args.fail_on_missing:
            raise FileNotFoundError(error)
        return missing_row(
            method=method,
            dataset=dataset,
            dataset_path=dataset_path,
            checkpoint_path=checkpoint_path,
            args=args,
            raw_dir=raw_dir,
            error=error,
        )
    command = build_flat_eval_command(method, checkpoint_path, dataset, args, artifact_root)
    result = run_command(command, raw_dir, args.command_timeout_sec, progress)
    output_method = "BasicPPO" if method == "ppo" else "DQN"
    artifact_dir = artifact_root / output_method / stem
    row = {
        "method": method,
        "method_family": method_family(method),
        "dataset": stem,
        "dataset_path": str(dataset_path),
        "temperature": float(args.temperature),
        "num_runs": int(args.num_runs),
        "seed": int(args.seed),
        "wall_time_sec": result.wall_time_sec,
        "checkpoint_path": str(checkpoint_path),
        "command": command_to_string(command),
        "raw_output_dir": str(raw_dir),
    }
    if result.returncode == 0:
        try:
            row.update(parse_metrics_json(artifact_dir / "metrics.json"))
            row["status"] = "ok"
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
    else:
        row["status"] = "failed"
        row["error"] = tail_text(result.stderr_path)
    return row


def run_heuristic_eval(
    method: str,
    dataset: str,
    args: argparse.Namespace,
    output_dir: Path,
    progress: ProgressContext | None = None,
) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    dataset_path = resolve_data_path(data_dir, dataset)
    stem = dataset_path.stem
    raw_dir = output_dir / "raw" / "eval" / method / stem
    artifact_root = raw_dir / "artifacts"
    command = build_heuristic_eval_command(method, dataset, args, artifact_root)
    result = run_command(command, raw_dir, args.command_timeout_sec, progress)
    artifact_dir = artifact_root / method / stem
    row = {
        "method": method,
        "method_family": method_family(method),
        "dataset": stem,
        "dataset_path": str(dataset_path),
        "status": "ok" if result.returncode == 0 else "failed",
        "temperature": "",
        "num_runs": 1,
        "seed": int(args.seed),
        "wall_time_sec": result.wall_time_sec,
        "checkpoint_path": "",
        "command": command_to_string(command),
        "raw_output_dir": str(raw_dir),
    }
    if result.returncode == 0:
        try:
            row.update(parse_metrics_json(artifact_dir / "metrics.json"))
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
    else:
        row["error"] = tail_text(result.stderr_path)
    return row


def build_hbgat_train_command(method: str, raw_dir: Path, args: argparse.Namespace) -> list[str]:
    checkpoint_root = raw_dir / "checkpoints"
    result_dir = raw_dir / "results"
    log_dir = raw_dir / "tf_logs"
    command = [
        args.python,
        str(PROJECT_ROOT / "train.py"),
        "--trainer",
        "lightning",
        "--config",
        args.config,
        "--max-episodes",
        str(args.train_probe_episodes),
        "--num-envs",
        str(args.train_num_envs),
        "--batch-size",
        str(args.hbgat_train_batch_size),
        "--eval-freq",
        "999999",
        "--data-path",
        str(args.train_data_path),
        "--seed",
        str(args.seed),
        "--log-dir",
        str(log_dir),
        "--output-dir",
        str(result_dir),
        "--set",
        "enable_multi_benchmark_eval=false",
        "--set",
        f"experiment_name=runtime_probe_{method}",
        "--set",
        f"checkpoint_root={checkpoint_root}",
    ]
    command.extend(HBGAT_ABLATION_ARGS.get(method, []))
    return command


def build_flat_train_command(method: str, raw_dir: Path, args: argparse.Namespace) -> list[str]:
    script = "train_basic.py" if method == "ppo" else "train_dqn.py"
    subdir = "basic_ppo" if method == "ppo" else "dqn"
    return [
        args.python,
        str(PROJECT_ROOT / "baselines" / subdir / script),
        "--config",
        args.config,
        "--max-episodes",
        str(args.train_probe_episodes),
        "--batch-size",
        str(args.flat_train_batch_size),
        "--data-path",
        str(args.train_data_path),
        "--output-dir",
        str(raw_dir / "results"),
        "--seed",
        str(args.seed),
    ]


def estimate_train_row(method: str, command: list[str], result: CommandResult, args: argparse.Namespace, raw_dir: Path) -> dict[str, Any]:
    avg_episode = result.wall_time_sec / max(1, int(args.train_probe_episodes))
    estimate_sec = avg_episode * int(args.target_max_episodes)
    batch_size = int(args.hbgat_train_batch_size) if method in HBGAT_TRAINABLE else int(args.flat_train_batch_size)
    num_envs = int(args.train_num_envs) if method in HBGAT_TRAINABLE else 1
    row = {
        "method": method,
        "method_family": method_family(method),
        "status": "ok" if result.returncode == 0 else "failed",
        "probe_episodes": int(args.train_probe_episodes),
        "target_max_episodes": int(args.target_max_episodes),
        "train_batch_size": batch_size,
        "train_num_envs": num_envs,
        "probe_wall_time_sec": float(result.wall_time_sec),
        "avg_probe_episode_time_sec": float(avg_episode),
        "estimated_full_train_time_sec": float(estimate_sec),
        "estimated_full_train_time_h": float(estimate_sec / 3600.0),
        "command": command_to_string(command),
        "raw_output_dir": str(raw_dir),
    }
    if result.returncode != 0:
        row["error"] = tail_text(result.stderr_path)
    return row


def run_train_probe(
    methods: list[str],
    args: argparse.Namespace,
    output_dir: Path,
    *,
    overall_start: float,
    total_items: int,
    start_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset, method in enumerate(methods):
        item_index = start_index + offset
        raw_dir = output_dir / "raw" / "train" / method
        progress = ProgressContext(
            phase="训练短测",
            item_label=method,
            item_index=item_index,
            total_items=total_items,
            overall_start=overall_start,
            raw_dir=raw_dir,
            quiet=bool(args.quiet_progress),
            progress_interval_sec=float(args.progress_interval_sec),
            show_log_tail_lines=int(args.show_log_tail_lines),
        )
        if method in HBGAT_TRAINABLE:
            command = build_hbgat_train_command(method, raw_dir, args)
            result = run_command(command, raw_dir, args.command_timeout_sec, progress)
            row = estimate_train_row(method, command, result, args, raw_dir)
        elif method in FLAT_TRAINABLE:
            command = build_flat_train_command(method, raw_dir, args)
            result = run_command(command, raw_dir, args.command_timeout_sec, progress)
            row = estimate_train_row(method, command, result, args, raw_dir)
        else:
            row = {
                "method": method,
                "method_family": method_family(method),
                "status": "not_applicable",
                "probe_episodes": "",
                "target_max_episodes": "",
                "train_batch_size": "",
                "train_num_envs": "",
                "raw_output_dir": str(raw_dir),
                "error": "该方法没有训练阶段，仅统计验证/搜索耗时。",
            }
        rows.append(row)
        print_task_result(progress, row)
    return rows


def run_eval(
    methods: list[str],
    datasets: list[str],
    args: argparse.Namespace,
    output_dir: Path,
    *,
    overall_start: float,
    total_items: int,
    start_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    item_offset = 0
    for dataset in datasets:
        for method in methods:
            item_index = start_index + item_offset
            item_offset += 1
            stem = dataset_stem(dataset)
            raw_dir = output_dir / "raw" / "eval" / method / stem
            progress = ProgressContext(
                phase="验证",
                item_label=f"{method} @ {dataset}",
                item_index=item_index,
                total_items=total_items,
                overall_start=overall_start,
                raw_dir=raw_dir,
                quiet=bool(args.quiet_progress),
                progress_interval_sec=float(args.progress_interval_sec),
                show_log_tail_lines=int(args.show_log_tail_lines),
            )
            if method in HBGAT_TRAINABLE:
                row = run_hbgat_eval(method, dataset, args, output_dir, progress)
            elif method in FLAT_TRAINABLE:
                row = run_flat_eval(method, dataset, args, output_dir, progress)
            elif method in HEURISTIC_METHODS or method in SEARCH_METHODS:
                row = run_heuristic_eval(method, dataset, args, output_dir, progress)
            else:
                row = {
                    "method": method,
                    "method_family": method_family(method),
                    "dataset": stem,
                    "dataset_path": str(resolve_data_path(Path(args.data_dir), dataset)),
                    "status": "skipped",
                    "raw_output_dir": str(raw_dir),
                    "error": f"未知方法: {method}",
                }
            rows.append(row)
            print_task_result(progress, row)
    return rows
def build_combined_rows(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    train_by_method = {str(row.get("method")): row for row in train_rows}
    combined: list[dict[str, Any]] = []
    for eval_row in eval_rows:
        method = str(eval_row.get("method", ""))
        train_row = train_by_method.get(method, {})
        combined.append(
            {
                "method": method,
                "method_family": eval_row.get("method_family", method_family(method)),
                "dataset": eval_row.get("dataset", ""),
                "train_status": train_row.get("status", ""),
                "estimated_full_train_time_h": train_row.get("estimated_full_train_time_h", ""),
                "eval_status": eval_row.get("status", ""),
                "eval_inference_time_sec": eval_row.get("inference_time_sec", ""),
                "eval_wall_time_sec": eval_row.get("wall_time_sec", ""),
                "makespan": eval_row.get("makespan", ""),
                "valid": eval_row.get("valid", ""),
                "checkpoint_path": eval_row.get("checkpoint_path", ""),
            }
        )
    if not eval_rows:
        for train_row in train_rows:
            combined.append(
                {
                    "method": train_row.get("method", ""),
                    "method_family": train_row.get("method_family", ""),
                    "dataset": "",
                    "train_status": train_row.get("status", ""),
                    "estimated_full_train_time_h": train_row.get("estimated_full_train_time_h", ""),
                    "eval_status": "",
                    "eval_inference_time_sec": "",
                    "eval_wall_time_sec": "",
                    "makespan": "",
                    "valid": "",
                    "checkpoint_path": "",
                }
            )
    return combined


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="统计初始调度方法的短训耗时和温度 0 验证耗时")
    parser.add_argument("--mode", choices=("all", "train_probe", "eval"), default="all")
    parser.add_argument("--config", default="conf/experiment/scale_400_800_schedule.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--train-probe-episodes", type=int, default=1)
    parser.add_argument("--target-max-episodes", type=int, default=300)
    parser.add_argument("--hbgat-train-batch-size", type=int, default=4)
    parser.add_argument("--flat-train-batch-size", type=int, default=16)
    parser.add_argument("--train-num-envs", type=int, default=2)
    parser.add_argument("--train-data-path", default="data/680.csv")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--command-timeout-sec", type=int, default=0)
    parser.add_argument("--progress-interval-sec", type=float, default=30.0)
    parser.add_argument("--show-log-tail-lines", type=int, default=1)
    parser.add_argument("--quiet-progress", action="store_true")
    parser.add_argument("--skip-missing", dest="fail_on_missing", action="store_false", default=False)
    parser.add_argument("--fail-on-missing", dest="fail_on_missing", action="store_true")
    parser.add_argument("--ga-pop-size", type=int, default=10)
    parser.add_argument("--ga-max-gen", type=int, default=5)
    parser.add_argument("--balance-weight", type=float, default=1.0)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--beam-branch-factor", type=int, default=2)
    parser.add_argument("--beam-levels", type=int, default=4)
    parser.add_argument("--beam-patience", type=int, default=2)
    parser.add_argument("--ig-iterations", type=int, default=20)
    parser.add_argument("--ig-destroy-ratio", type=float, default=0.10)
    parser.add_argument("--ig-noise-sigma", type=float, default=0.25)
    parser.add_argument("--sa-iterations", type=int, default=30)
    parser.add_argument("--sa-initial-temp", type=float, default=0.05)
    parser.add_argument("--sa-cooling", type=float, default=0.96)
    parser.add_argument("--sa-min-temp", type=float, default=1e-4)
    add_common_config_arguments(parser)
    return parser


def main() -> None:
    args = parse_runtime_args(build_parser())
    if args.train_probe_episodes < 1:
        raise ValueError("--train-probe-episodes 必须大于等于 1")
    if args.hbgat_train_batch_size < 1 or args.flat_train_batch_size < 1:
        raise ValueError("训练 batch size 必须大于等于 1")
    if args.train_num_envs < 1:
        raise ValueError("--train-num-envs 必须大于等于 1")
    script_start = time.perf_counter()
    resolve_runtime_config(args, target=configs)
    output_dir, context = resolve_run_output_dir(
        configs,
        PROJECT_ROOT,
        default_legacy_dir="results/runtime_benchmark/initial_train_eval_temperature0",
        run_subdir="benchmark/initial_train_eval_temperature0",
        explicit_dir=args.output_dir,
        section="artifacts",
    )
    manifest_extra = {
        "run_type": "benchmark",
        "artifact_kind": "initial_train_eval_runtime",
        "mode": str(args.mode),
        "methods": list(args.methods),
        "datasets": list(args.datasets),
        "output_dir": str(output_dir.resolve()),
    }
    if context is not None:
        write_run_context_files(context, configs, command="benchmark_initial_train_eval_runtime", extra=manifest_extra)
    else:
        write_run_manifest(output_dir, configs, command="benchmark_initial_train_eval_runtime", extra=manifest_extra)

    methods = [str(method) for method in args.methods]
    datasets = [str(dataset) for dataset in args.datasets]
    train_count = len(methods) if args.mode in {"all", "train_probe"} else 0
    eval_count = len(methods) * len(datasets) if args.mode in {"all", "eval"} else 0
    total_items = train_count + eval_count
    print_header(args, methods, datasets, output_dir)

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    if args.mode in {"all", "train_probe"}:
        train_rows = run_train_probe(
            methods,
            args,
            output_dir,
            overall_start=script_start,
            total_items=total_items,
            start_index=1,
        )
    if args.mode in {"all", "eval"}:
        eval_rows = run_eval(
            methods,
            datasets,
            args,
            output_dir,
            overall_start=script_start,
            total_items=total_items,
            start_index=train_count + 1,
        )

    write_csv(output_dir / "train_runtime_summary.csv", train_rows, TRAIN_FIELDS)
    write_csv(output_dir / "eval_runtime_summary.csv", eval_rows, EVAL_FIELDS)
    combined_rows = build_combined_rows(train_rows, eval_rows)
    write_csv(output_dir / "combined_runtime_summary.csv", combined_rows, COMBINED_FIELDS)
    write_json(
        output_dir / "combined_runtime_summary.json",
        {
            "mode": args.mode,
            "methods": methods,
            "datasets": datasets,
            "train_rows": train_rows,
            "eval_rows": eval_rows,
            "combined_rows": combined_rows,
        },
    )
    write_paper_tables(output_dir, train_rows, eval_rows)
    print_final_summary(output_dir, train_rows, eval_rows, time.perf_counter() - script_start)


if __name__ == "__main__":
    main()
