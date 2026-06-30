from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = ["283.csv", "680.csv", "2338.csv", "3182.csv"]
DEFAULT_METHODS = ["SPT", "LPT", "Random", "EDD", "CPM", "MSL", "Beam", "IG", "SA"]
HEURISTIC_METHODS = {"SPT", "LPT", "Random", "EDD", "CPM", "MSL"}
SEARCH_METHODS = {"Beam", "IG", "SA"}

SUMMARY_FIELDS = [
    "method",
    "method_family",
    "dataset",
    "status",
    "search_budget",
    "algorithm_time_sec",
    "process_wall_time_sec",
    "makespan",
    "balance_std",
    "worker_utilization",
    "station_utilization",
    "valid",
    "completion_rate",
    "deadlock_count",
    "command",
    "raw_output_dir",
    "metrics_path",
    "error",
]


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    wall_time_sec: float
    stdout_path: Path
    stderr_path: Path


def method_family(method: str) -> str:
    if method in HEURISTIC_METHODS:
        return "heuristic"
    if method in SEARCH_METHODS:
        return "search"
    return "unknown"


def dataset_stem(dataset: str) -> str:
    return Path(dataset).stem


def dataset_sort_key(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):08d}")
    except ValueError:
        return (1, value)


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def command_to_string(command: list[str]) -> str:
    return " ".join(command)


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return str(timedelta(seconds=int(round(seconds))))


def fmt_num(value: Any, digits: int = 2) -> str:
    if value in ("", None):
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def tail_text(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-max_chars:]


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
    fixed.setdefault("PYTHONUNBUFFERED", "1")
    return fixed


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


def pivot_rows(rows: list[dict[str, Any]], *, value_key: str) -> list[dict[str, Any]]:
    methods = [method for method in DEFAULT_METHODS if any(row.get("method") == method for row in rows)]
    extra_methods = sorted({str(row.get("method", "")) for row in rows if row.get("method") not in methods})
    datasets = sorted({str(row.get("dataset", "")) for row in rows if row.get("dataset", "")}, key=dataset_sort_key)
    by_key = {
        (str(row.get("method", "")), str(row.get("dataset", ""))): row.get(value_key, "")
        for row in rows
    }
    output: list[dict[str, Any]] = []
    for method in methods + extra_methods:
        record: dict[str, Any] = {"method": method}
        for dataset in datasets:
            record[dataset] = by_key.get((method, dataset), "")
        output.append(record)
    return output


def build_baseline_command(method: str, dataset: str, args: argparse.Namespace) -> list[str]:
    budget = int(args.search_budget)
    return [
        args.python,
        str(PROJECT_ROOT / "baselines" / "heuristic" / "run_all_baselines.py"),
        "--config",
        args.config,
        "--data_dir",
        args.data_dir,
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
        str(budget),
        "--beam_patience",
        str(args.beam_patience),
        "--ig_iterations",
        str(budget),
        "--ig_destroy_ratio",
        str(args.ig_destroy_ratio),
        "--ig_noise_sigma",
        str(args.ig_noise_sigma),
        "--sa_iterations",
        str(budget),
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
    ]


def run_command(
    command: list[str],
    raw_dir: Path,
    *,
    timeout_sec: int,
    progress_interval_sec: float,
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
    deadline = None if timeout_sec <= 0 else start + float(timeout_sec)
    next_progress = start + max(1.0, progress_interval_sec)

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
            while process.poll() is None:
                now = time.perf_counter()
                if deadline is not None and now >= deadline:
                    process.kill()
                    process.wait()
                    stderr_file.write(f"\n[benchmark] command timeout after {timeout_sec} sec\n")
                    break
                if progress_interval_sec > 0 and now >= next_progress:
                    elapsed = now - start
                    print(f"  运行中: {fmt_duration(elapsed)}", flush=True)
                    next_progress = now + max(1.0, progress_interval_sec)
                time.sleep(0.5)

    return CommandResult(
        returncode=int(process.returncode),
        wall_time_sec=float(time.perf_counter() - start),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def parse_metrics(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"缺少 metrics.json: {metrics_path}")
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "makespan": data.get("makespan"),
        "balance_std": data.get("workload_balance_std"),
        "worker_utilization": data.get("worker_utilization"),
        "station_utilization": data.get("station_utilization"),
        "valid": data.get("valid"),
        "completion_rate": data.get("completion_rate"),
        "deadlock_count": data.get("deadlock_count"),
        "algorithm_time_sec": data.get("inference_time"),
    }


def copy_artifacts(source_dir: Path, target_dir: Path) -> None:
    if source_dir.exists():
        shutil.copytree(source_dir, target_dir / "artifacts", dirs_exist_ok=True)


def estimate_eta(done_count: int, total_count: int, overall_start: float) -> float | None:
    if done_count <= 0:
        return None
    elapsed = time.perf_counter() - overall_start
    remaining = total_count - done_count
    return (elapsed / done_count) * remaining


def run_one(method: str, dataset: str, args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    stem = dataset_stem(dataset)
    raw_dir = output_dir / "raw" / method / stem
    command = build_baseline_command(method, dataset, args)
    print(f"  命令: {command_to_string(command)}", flush=True)
    result = run_command(
        command,
        raw_dir,
        timeout_sec=int(args.command_timeout_sec),
        progress_interval_sec=float(args.progress_interval_sec),
    )

    artifact_dir = PROJECT_ROOT / "results" / "eval_logs" / method / stem
    metrics_path = artifact_dir / "metrics.json"
    row: dict[str, Any] = {
        "method": method,
        "method_family": method_family(method),
        "dataset": stem,
        "status": "ok" if result.returncode == 0 else "failed",
        "search_budget": int(args.search_budget) if method in SEARCH_METHODS else "",
        "process_wall_time_sec": result.wall_time_sec,
        "command": command_to_string(command),
        "raw_output_dir": str(raw_dir),
        "metrics_path": str(metrics_path),
    }

    if result.returncode != 0:
        row["error"] = tail_text(result.stderr_path)
        return row

    try:
        copy_artifacts(artifact_dir, raw_dir)
        row.update(parse_metrics(metrics_path))
    except Exception as exc:
        row["status"] = "missing_metrics"
        row["error"] = str(exc)
    return row


def write_paper_tables(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_dynamic_csv(output_dir / "paper_algorithm_time_seconds.csv", pivot_rows(rows, value_key="algorithm_time_sec"))
    write_dynamic_csv(output_dir / "paper_wall_time_seconds.csv", pivot_rows(rows, value_key="process_wall_time_sec"))
    write_dynamic_csv(output_dir / "paper_makespan.csv", pivot_rows(rows, value_key="makespan"))
    write_dynamic_csv(output_dir / "paper_valid.csv", pivot_rows(rows, value_key="valid"))


def markdown_table(rows: list[dict[str, Any]], *, value_key: str, digits: int = 2) -> str:
    pivoted = pivot_rows(rows, value_key=value_key)
    datasets = [key for key in pivoted[0].keys() if key != "method"] if pivoted else []
    header = "| Method | " + " | ".join(datasets) + " |"
    sep = "|---|" + "|".join("---:" for _ in datasets) + "|"
    body = []
    for row in pivoted:
        cells = [str(row.get("method", ""))]
        for dataset in datasets:
            cells.append(fmt_num(row.get(dataset), digits))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body])


def write_docs_markdown(docs_path: Path, rows: list[dict[str, Any]], args: argparse.Namespace, output_dir: Path) -> None:
    ok_count = sum(1 for row in rows if row.get("status") == "ok")
    failed_count = len(rows) - ok_count
    lines = [
        "# 规则与元启发搜索预算 3 真实运行时间补测",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 数据集：{', '.join(args.datasets)}",
        f"- 方法：{', '.join(args.methods)}",
        f"- 元启发搜索预算：Beam levels = IG iterations = SA iterations = {int(args.search_budget)}",
        f"- 输出目录：`{rel_or_abs(output_dir)}`",
        f"- 完成情况：OK={ok_count}, FAILED/MISSING={failed_count}",
        "",
        "## 论文建议采用的时间口径",
        "",
        "`algorithm_time_sec` 来自真实 baseline 脚本写出的 `metrics.json/inference_time`，表示规则调度或搜索过程本身的运行时间。该口径与当前论文表格中的 scheduling/search runtime 一致。",
        "",
        markdown_table(rows, value_key="algorithm_time_sec", digits=2),
        "",
        "## 端到端进程耗时",
        "",
        "`process_wall_time_sec` 由外层补测脚本计时，包含 Python 进程启动、配置加载、数据加载、结果写入等开销，仅作为复现实验审计依据。",
        "",
        markdown_table(rows, value_key="process_wall_time_sec", digits=2),
        "",
        "## Makespan",
        "",
        markdown_table(rows, value_key="makespan", digits=2),
        "",
        "## 有效性",
        "",
        markdown_table(rows, value_key="valid", digits=3),
        "",
    ]

    failed_rows = [row for row in rows if row.get("status") != "ok"]
    if failed_rows:
        lines.extend(["## 失败或缺失项", ""])
        lines.append("| Method | Dataset | Status | Error |")
        lines.append("|---|---|---|---|")
        for row in failed_rows:
            error = str(row.get("error", "")).replace("\n", " ")[:240]
            lines.append(f"| {row.get('method', '')} | {row.get('dataset', '')} | {row.get('status', '')} | {error} |")
        lines.append("")

    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text("\n".join(lines), encoding="utf-8")


def print_header(args: argparse.Namespace, output_dir: Path, docs_path: Path) -> None:
    print("=" * 88, flush=True)
    print("规则与元启发真实运行时间补测：搜索预算 3", flush=True)
    print(f"datasets={args.datasets}", flush=True)
    print(f"methods={args.methods}", flush=True)
    print(f"search_budget={args.search_budget}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"docs={docs_path}", flush=True)
    print("=" * 88, flush=True)


def print_final_summary(output_dir: Path, docs_path: Path, rows: list[dict[str, Any]], total_wall_sec: float) -> None:
    ok_count = sum(1 for row in rows if row.get("status") == "ok")
    failed_count = len(rows) - ok_count
    print("\n" + "=" * 88, flush=True)
    print("运行完成", flush=True)
    print(f"总耗时: {fmt_duration(total_wall_sec)} | OK={ok_count} | FAILED/MISSING={failed_count}", flush=True)
    print(f"汇总目录: {output_dir}", flush=True)
    print(f"Markdown: {docs_path}", flush=True)
    print("\n算法内部耗时概览 algorithm_time_sec:", flush=True)
    print(markdown_table(rows, value_key="algorithm_time_sec", digits=2), flush=True)
    print("=" * 88, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="补测规则与元启发 baseline 在搜索预算 3 下的真实运行时间")
    parser.add_argument("--config", default="conf/experiment/scale_400_800_schedule.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--search-budget", type=int, default=3)
    parser.add_argument("--output-dir", default="results/runtime_benchmark/heuristic_search_budget3")
    parser.add_argument("--docs-path", default="docs/heuristic_search_budget3_runtime.md")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--command-timeout-sec", type=int, default=0)
    parser.add_argument("--progress-interval-sec", type=float, default=30.0)
    parser.add_argument("--ga-pop-size", type=int, default=10)
    parser.add_argument("--ga-max-gen", type=int, default=3)
    parser.add_argument("--balance-weight", type=float, default=1.0)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--beam-branch-factor", type=int, default=4)
    parser.add_argument("--beam-patience", type=int, default=4)
    parser.add_argument("--ig-destroy-ratio", type=float, default=0.10)
    parser.add_argument("--ig-noise-sigma", type=float, default=0.25)
    parser.add_argument("--sa-initial-temp", type=float, default=0.05)
    parser.add_argument("--sa-cooling", type=float, default=0.96)
    parser.add_argument("--sa-min-temp", type=float, default=1e-4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if int(args.search_budget) < 1:
        raise ValueError("--search-budget 必须大于等于 1")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    docs_path = Path(args.docs_path)
    if not docs_path.is_absolute():
        docs_path = PROJECT_ROOT / docs_path

    args.datasets = [str(dataset) for dataset in args.datasets]
    args.methods = [str(method) for method in args.methods]

    total = len(args.datasets) * len(args.methods)
    start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    print_header(args, output_dir, docs_path)

    item_index = 0
    for dataset in args.datasets:
        for method in args.methods:
            item_index += 1
            print(f"\n[{item_index}/{total}] {method} @ {dataset}", flush=True)
            row = run_one(method, dataset, args, output_dir)
            rows.append(row)
            eta = estimate_eta(item_index, total, start)
            print(
                "  结果: "
                f"status={row.get('status')} "
                f"algorithm={fmt_duration(row.get('algorithm_time_sec'))} "
                f"wall={fmt_duration(row.get('process_wall_time_sec'))} "
                f"makespan={fmt_num(row.get('makespan'), 2)} "
                f"valid={fmt_num(row.get('valid'), 3)}",
                flush=True,
            )
            print(
                f"  总已用: {fmt_duration(time.perf_counter() - start)} | ETA: {fmt_duration(eta)} | raw: {row.get('raw_output_dir')}",
                flush=True,
            )

    write_csv(output_dir / "heuristic_search_budget3_runtime.csv", rows, SUMMARY_FIELDS)
    write_json(
        output_dir / "heuristic_search_budget3_runtime.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "config": args.config,
            "data_dir": args.data_dir,
            "datasets": args.datasets,
            "methods": args.methods,
            "search_budget": int(args.search_budget),
            "rows": rows,
        },
    )
    write_paper_tables(output_dir, rows)
    write_docs_markdown(docs_path, rows, args, output_dir)
    print_final_summary(output_dir, docs_path, rows, time.perf_counter() - start)


if __name__ == "__main__":
    main()
