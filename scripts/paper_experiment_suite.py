from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.paper_metrics import (
    PaperExperimentConfig,
    collect_result_rows,
    command_plan_rows,
    constraint_diagnostic_rows,
    convergence_stability_rows,
    dataset_profile_rows,
    efficiency_pareto_rows,
    graph_complexity_rows,
    load_paper_config,
    policy_behavior_rows,
    significance_test_rows,
    statistical_summary_rows,
    write_csv,
    write_json,
    write_summary_markdown,
)


SUITES = {
    "dataset_profile",
    "complexity",
    "ablation",
    "generalization",
    "reschedule",
    "statistics",
    "convergence",
    "constraints",
    "behavior",
    "efficiency",
    "all",
}

COLLECT_ALL_ORDER = (
    "dataset_profile",
    "complexity",
    "statistics",
    "constraints",
    "behavior",
    "efficiency",
    "convergence",
    "ablation",
    "generalization",
    "reschedule",
)


def parse_set_values(items: list[str] | None) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        yaml = None
    overrides: dict[str, Any] = {}
    for item in items or []:
        key, separator, raw_value = item.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"--set 必须使用 key=value 格式: {item!r}")
        if yaml is not None:
            value = yaml.safe_load(raw_value)
        else:
            value = raw_value
        overrides[key.strip()] = value
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成 APAL 论文实验表格、统计指标和命令清单")
    parser.add_argument(
        "--config",
        default="conf/experiment/paper_experiment_suite.yaml",
        help="论文实验套件 YAML 配置",
    )
    parser.add_argument(
        "--suite",
        default="all",
        choices=sorted(SUITES),
        help="要执行的实验套件",
    )
    parser.add_argument(
        "--mode",
        default="collect",
        choices=("collect", "plan"),
        help="collect 读取现有结果并生成表格；plan 只生成命令清单",
    )
    parser.add_argument("--output-dir", default=None, help="覆盖输出目录")
    parser.add_argument("--run-id", default=None, help="覆盖本次实验套件 run_id")
    parser.add_argument("--set", dest="set_values", action="append", default=[], help="覆盖 YAML 字段，例如 --set seeds=[0,1,2]")
    return parser


def resolve_run_dir(config: PaperExperimentConfig, *, output_dir: str | None, run_id: str | None) -> Path:
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    chosen_run_id = run_id or config.run_id or f"{config.experiment_name}_{datetime.now().strftime('%y%m%d-%H%M%S')}"
    return (config.output_root / chosen_run_id).resolve()


def selected_suites(name: str) -> tuple[str, ...]:
    if name == "all":
        return COLLECT_ALL_ORDER
    return (name,)


def needs_result_rows(suites: tuple[str, ...]) -> bool:
    return any(item in suites for item in ("statistics", "constraints", "efficiency", "all"))


def run(args: argparse.Namespace) -> Path:
    overrides = parse_set_values(args.set_values)
    if args.output_dir:
        overrides["output_root"] = args.output_dir
    if args.run_id:
        overrides["run_id"] = args.run_id
    config = load_paper_config(Path(args.config), PROJECT_ROOT, overrides=overrides)
    run_dir = resolve_run_dir(config, output_dir=args.output_dir, run_id=args.run_id)
    csv_dir = run_dir / "csv"
    json_dir = run_dir / "json"
    markdown_dir = run_dir / "markdown"
    commands_dir = run_dir / "commands"
    for directory in (csv_dir, json_dir, markdown_dir, commands_dir):
        directory.mkdir(parents=True, exist_ok=True)

    suites = selected_suites(args.suite)
    outputs: dict[str, Path] = {}
    row_counts: dict[str, int] = {}

    result_rows: list[dict[str, Any]] = []
    if args.mode == "collect" and needs_result_rows(suites):
        result_rows = collect_result_rows(config.result_roots)
        result_path = csv_dir / "raw_result_rows.csv"
        write_csv(result_path, result_rows)
        write_json(json_dir / "raw_result_rows.json", result_rows)
        outputs["raw_result_rows"] = result_path
        row_counts["raw_result_rows"] = len(result_rows)

    command_rows = command_plan_rows(config)
    command_csv = csv_dir / "command_plan.csv"
    write_csv(command_csv, command_rows)
    write_json(json_dir / "command_plan.json", command_rows)
    write_command_script(commands_dir / "paper_experiment_commands.sh", command_rows)
    outputs["command_plan"] = command_csv
    row_counts["command_plan"] = len(command_rows)

    if args.mode == "plan":
        write_summary_markdown(markdown_dir / "paper_experiment_summary.md", outputs, row_counts)
        print(f"[PaperSuite] mode=plan output={run_dir}")
        return run_dir

    for suite in suites:
        if suite == "dataset_profile":
            rows = dataset_profile_rows(
                config.datasets,
                station_count=config.station_count,
                worker_count=config.worker_count,
            )
            path = csv_dir / "dataset_profile.csv"
            write_csv(path, rows)
            write_json(json_dir / "dataset_profile.json", rows)
            outputs["dataset_profile"] = path
            row_counts["dataset_profile"] = len(rows)
            continue

        if suite == "complexity":
            rows = graph_complexity_rows(config.datasets)
            path = csv_dir / "graph_complexity.csv"
            write_csv(path, rows)
            write_json(json_dir / "graph_complexity.json", rows)
            outputs["complexity"] = path
            row_counts["complexity"] = len(rows)
            continue

        if suite in {"ablation", "generalization", "reschedule"}:
            # 这三类 suite 的长训练/验证由命令清单驱动；现有结果已在 statistics/efficiency 中汇总。
            continue

        if suite == "statistics":
            rows = statistical_summary_rows(
                result_rows,
                reference_makespans=config.reference_makespans,
                reference_method=config.reference_method,
                bootstrap_samples=config.bootstrap_samples,
            )
            path = csv_dir / "statistical_summary.csv"
            write_csv(path, rows)
            write_json(json_dir / "statistical_summary.json", rows)
            outputs["statistical_summary"] = path
            row_counts["statistical_summary"] = len(rows)

            sig_rows = significance_test_rows(
                result_rows,
                reference_method=config.reference_method,
                permutation_samples=config.permutation_samples,
            )
            sig_path = csv_dir / "significance_tests.csv"
            write_csv(sig_path, sig_rows)
            write_json(json_dir / "significance_tests.json", sig_rows)
            outputs["significance_tests"] = sig_path
            row_counts["significance_tests"] = len(sig_rows)
            continue

        if suite == "constraints":
            rows = constraint_diagnostic_rows(result_rows, config.result_roots)
            path = csv_dir / "constraint_diagnostics.csv"
            write_csv(path, rows)
            write_json(json_dir / "constraint_diagnostics.json", rows)
            outputs["constraint_diagnostics"] = path
            row_counts["constraint_diagnostics"] = len(rows)
            continue

        if suite == "behavior":
            rows = policy_behavior_rows(config.result_roots)
            path = csv_dir / "policy_behavior.csv"
            write_csv(path, rows)
            write_json(json_dir / "policy_behavior.json", rows)
            outputs["policy_behavior"] = path
            row_counts["policy_behavior"] = len(rows)
            continue

        if suite == "efficiency":
            rows = efficiency_pareto_rows(
                result_rows,
                reference_makespans=config.reference_makespans,
            )
            path = csv_dir / "efficiency_pareto.csv"
            write_csv(path, rows)
            write_json(json_dir / "efficiency_pareto.json", rows)
            outputs["efficiency_pareto"] = path
            row_counts["efficiency_pareto"] = len(rows)
            continue

        if suite == "convergence":
            rows = convergence_stability_rows(config.result_roots)
            path = csv_dir / "convergence_stability.csv"
            write_csv(path, rows)
            write_json(json_dir / "convergence_stability.json", rows)
            outputs["convergence_stability"] = path
            row_counts["convergence_stability"] = len(rows)
            continue

    summary_path = markdown_dir / "paper_experiment_summary.md"
    write_summary_markdown(summary_path, outputs, row_counts)
    print(f"[PaperSuite] mode=collect suite={args.suite} output={run_dir}")
    for key, path in outputs.items():
        print(f"[PaperSuite] {key}: {path} rows={row_counts.get(key, 0)}")
    return run_dir


def write_command_script(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# APAL 论文实验命令清单。默认不由脚本自动执行，逐条确认后再运行。",
    ]
    for row in rows:
        command = str(row.get("command", "")).strip()
        if not command:
            continue
        lines.append("")
        lines.append(f"# suite={row.get('suite', '')} variant={row.get('variant', '')} status={row.get('status', '')}")
        lines.append(command)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
