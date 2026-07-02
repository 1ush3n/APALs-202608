# -*- coding: utf-8 -*-
"""批量评估 APAL 重调度规则基线。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.heuristic.reschedule_rules import DEFAULT_RULE_METHODS, rule_registry
from configs import configs
from runtime.artifacts import resolve_run_output_dir, write_run_context_files, write_run_manifest
from runtime.hydra_config import (
    ExtraArgument,
    HydraCliError,
    hydra_help,
    initialize_hydra_runtime,
    should_show_help,
)
from runtime.paths import resolve_workspace_path
from runtime.reschedule_eval import ensure_reschedule_baseline_available, ensure_reschedule_eval_scenarios_available
from runtime.reschedule_manifest import load_reschedule_manifest
from utils.reschedule import load_reschedule_scenarios


RULE_EXTRA_ARGS = {
    "beam_width": ExtraArgument(default=2, help="Beam Search 候选解数量"),
    "beam_branch_factor": ExtraArgument(default=2, help="Beam Search 每个候选解的扰动分支数"),
    "beam_levels": ExtraArgument(default=2, help="Beam Search 最大展开层数"),
    "beam_patience": ExtraArgument(default=1, help="Beam Search 连续无改进提前停止层数"),
    "ig_iterations": ExtraArgument(default=3, help="Iterated Greedy / Destroy-Repair 迭代次数"),
    "ig_destroy_ratio": ExtraArgument(default=0.08, help="Iterated Greedy 每次破坏的可移动任务比例"),
    "ig_noise_sigma": ExtraArgument(default=0.20, help="Iterated Greedy 修复优先级扰动强度"),
    "sa_iterations": ExtraArgument(default=3, help="Simulated Annealing 迭代次数"),
    "sa_initial_temp": ExtraArgument(default=0.05, help="Simulated Annealing 初始温度"),
    "sa_cooling": ExtraArgument(default=0.96, help="Simulated Annealing 降温系数"),
    "sa_min_temp": ExtraArgument(default=1e-4, help="Simulated Annealing 最小温度"),
    "methods": ExtraArgument(default=None, help="规则列表，例如 methods=[SPTRepair,CPMRepair]；缺省评估全部规则"),
    "scenario_path": ExtraArgument(default=None, help="固定重调度场景 CSV；缺省使用配置中的 reschedule_eval_scenario_path"),
    "baseline_path": ExtraArgument(default=None, help="baseline 调度 CSV；缺省使用配置中的 baseline 路径"),
    "data_path": ExtraArgument(default=None, help="APAL 数据文件或目录；缺省使用配置中的 data_file_path"),
    "manifest_path": ExtraArgument(default=None, help="可选 manifest；提供后按 instance_ids 自动取 data/baseline/scenario"),
    "instance_ids": ExtraArgument(default=None, help="manifest 实例列表，例如 instance_ids=[real_680]"),
    "num_runs": ExtraArgument(default=None, help="最多评估多少个场景；缺省评估全部场景"),
    "seed": ExtraArgument(default=42, help="规则评估固定种子"),
    "quiet": ExtraArgument(default=False, help="是否关闭逐场景输出"),
    "output_dir": ExtraArgument(default=None, help="输出目录；缺省写入本次 run 的 eval/reschedule_rules"),
}


def _normalize_methods(raw: Any) -> list[str]:
    registry = rule_registry()
    if raw is None or raw == "":
        methods = list(DEFAULT_RULE_METHODS)
    elif isinstance(raw, str):
        methods = [item.strip() for item in raw.split(",") if item.strip()]
    elif isinstance(raw, (list, tuple)):
        methods = [str(item).strip() for item in raw if str(item).strip()]
    else:
        raise ValueError(f"无法解析 methods 参数: {raw!r}")

    unknown = [method for method in methods if method not in registry]
    if unknown:
        raise ValueError(f"未知重调度规则: {unknown}；可选规则: {sorted(registry)}")
    return methods


def _scenario_level_from_id(scenario_id: str) -> str:
    head = str(scenario_id).split("_", 1)[0]
    return head if head in {"low", "medium", "high"} else "custom"


def _as_id_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError(f"无法解析 instance_ids 参数: {value!r}")


def _summarize(df: pd.DataFrame, group_cols: list[str]) -> list[dict[str, Any]]:
    if df.empty:
        return []
    metric_cols = [
        "makespan",
        "balance_std",
        "score",
        "selection_score",
        "eligible",
        "complete",
        "duration_sec",
        "worker_util",
        "station_util",
        "frozen_violation_count",
        "release_violation_count",
        "precedence_violation_count",
        "worker_overlap_violation_count",
        "station_slot_violation_count",
        "skill_violation_count",
        "demand_violation_count",
        "duplicate_task_count",
        "missing_task_count",
        "takt_violation_h",
        "start_deviation_mean_h",
        "station_change_rate",
        "team_change_rate",
    ]
    available = [col for col in metric_cols if col in df.columns]
    grouped = df.groupby(group_cols, dropna=False)[available].mean(numeric_only=True).reset_index()
    grouped = grouped.rename(columns={"eligible": "eligible_rate", "complete": "complete_rate"})
    return grouped.to_dict(orient="records")


def evaluate_reschedule_rules(
    *,
    data_path_or_dir: str | Path | None = None,
    scenario_path: str | Path | None = None,
    baseline_path: str | Path | None = None,
    methods: Any = None,
    num_runs: int | None = None,
    seed: int = 42,
    output_dir: str | Path | None = None,
    verbose: bool = True,
    beam_width: int = 2,
    beam_branch_factor: int = 2,
    beam_levels: int = 2,
    beam_patience: int = 1,
    ig_iterations: int = 3,
    ig_destroy_ratio: float = 0.08,
    ig_noise_sigma: float = 0.20,
    sa_iterations: int = 3,
    sa_initial_temp: float = 0.05,
    sa_cooling: float = 0.96,
    sa_min_temp: float = 1e-4,
) -> dict[str, Any]:
    """使用固定场景库评估一组重调度规则算法。"""

    registry = rule_registry()
    method_names = _normalize_methods(methods)
    resolved_baseline: Path | None = None
    resolved_scenario: Path | None = None
    data_path: Path | None = None
    scenario_items = []
    rows: list[dict[str, Any]] = []
    backups = {
        "enable_dynamic_events": getattr(configs, "enable_dynamic_events", False),
        "enable_station_breakdown": getattr(configs, "enable_station_breakdown", False),
        "enable_material_delay": getattr(configs, "enable_material_delay", False),
        "enable_online_duration_perturb": getattr(configs, "enable_online_duration_perturb", False),
        "enable_worker_fatigue": getattr(configs, "enable_worker_fatigue", False),
        "randomize_durations": getattr(configs, "randomize_durations", False),
        "reschedule_manifest_path": getattr(configs, "reschedule_manifest_path", ""),
        "reschedule_eval_instance_id": getattr(configs, "reschedule_eval_instance_id", ""),
        "reschedule_scenario_path": getattr(configs, "reschedule_scenario_path", ""),
        "reschedule_eval_scenario_path": getattr(configs, "reschedule_eval_scenario_path", ""),
        "reschedule_baseline_schedule_path": getattr(configs, "reschedule_baseline_schedule_path", ""),
    }
    try:
        configs.enable_dynamic_events = False
        configs.enable_station_breakdown = False
        configs.enable_material_delay = False
        configs.enable_online_duration_perturb = False
        configs.enable_worker_fatigue = False
        configs.randomize_durations = False
        configs.reschedule_manifest_path = ""
        configs.reschedule_eval_instance_id = ""
        if scenario_path is not None:
            configs.reschedule_eval_scenario_path = str(scenario_path)
            configs.reschedule_scenario_path = ""
        if baseline_path is not None:
            configs.reschedule_baseline_schedule_path = str(baseline_path)

        resolved_baseline = ensure_reschedule_baseline_available(configs)
        resolved_scenario = (
            resolve_workspace_path(scenario_path) if scenario_path is not None else ensure_reschedule_eval_scenarios_available(configs)
        )
        if resolved_baseline is None or resolved_scenario is None:
            raise RuntimeError("规则重调度评估需要 enable_reschedule_mode=True、baseline CSV 和固定场景 CSV。")

        data_path = resolve_workspace_path(data_path_or_dir or getattr(configs, "data_file_path", "data/283.csv"))
        scenario_items = load_reschedule_scenarios(Path(resolved_scenario))
        if num_runs is not None:
            scenario_items = scenario_items[: max(1, int(num_runs))]

        for scenario_idx, (scenario_id, scenario) in enumerate(scenario_items):
            level = _scenario_level_from_id(scenario_id)
            for method_idx, method_name in enumerate(method_names):
                solver_cls = registry[method_name]
                solver_kwargs: dict[str, Any] = {
                    "data_path_or_dir": data_path,
                    "scenario": scenario,
                    "scenario_id": scenario_id,
                    "scenario_level": level,
                    "seed": int(seed) + scenario_idx * 1000 + method_idx,
                    "verbose": verbose,
                }
                if bool(getattr(solver_cls, "supports_priority_search", False)):
                    solver_kwargs.update(
                        {
                            "beam_width": int(beam_width),
                            "beam_branch_factor": int(beam_branch_factor),
                            "beam_levels": int(beam_levels),
                            "beam_patience": int(beam_patience),
                            "ig_iterations": int(ig_iterations),
                            "ig_destroy_ratio": float(ig_destroy_ratio),
                            "ig_noise_sigma": float(ig_noise_sigma),
                            "sa_iterations": int(sa_iterations),
                            "sa_initial_temp": float(sa_initial_temp),
                            "sa_cooling": float(sa_cooling),
                            "sa_min_temp": float(sa_min_temp),
                        }
                    )
                solver = solver_cls(**solver_kwargs)
                result = solver.run()
                row: dict[str, Any] = {
                    "method": method_name,
                    "scenario_id": scenario_id,
                    "scenario_level": level,
                    "scenario_start_time": result.scenario_start_time,
                    "delayed_task_count": float(result.delayed_task_count),
                    "makespan": result.makespan,
                    "balance_std": result.balance_std,
                    "reward": result.reward,
                    "duration_sec": result.duration_sec,
                    "score": float(result.constraint_metrics.get("composite_score", 0.0)),
                    "selection_score": float(result.constraint_metrics.get("selection_score", 0.0)),
                }
                row.update(result.constraint_metrics)
                rows.append(row)
                if verbose:
                    print(
                        f"[RuleEval] {method_name} {scenario_id} "
                        f"score={row['score']:.4f} elig={int(row.get('eligible', 0.0))} "
                        f"mk={row['makespan']:.2f} dur={row['duration_sec']:.2f}s"
                    )
    finally:
        for key, value in backups.items():
            setattr(configs, key, value)

    df = pd.DataFrame(rows)
    summary_by_method = _summarize(df, ["method"])
    summary_by_method_level = _summarize(df, ["method", "scenario_level"])
    summary = {
        "baseline_path": str(Path(resolved_baseline).resolve()) if resolved_baseline is not None else "",
        "scenario_path": str(Path(resolved_scenario).resolve()) if resolved_scenario is not None else "",
        "data_path": str(Path(data_path).resolve()) if data_path is not None else "",
        "methods": method_names,
        "scenario_count": int(len(scenario_items)),
        "row_count": int(len(rows)),
        "summary_by_method": summary_by_method,
        "summary_by_method_level": summary_by_method_level,
        "rows": rows,
    }

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "reschedule_rule_eval.csv", index=False)
        pd.DataFrame(summary_by_method).to_csv(out_dir / "reschedule_rule_summary_by_method.csv", index=False)
        pd.DataFrame(summary_by_method_level).to_csv(out_dir / "reschedule_rule_summary_by_method_level.csv", index=False)
        (out_dir / "reschedule_rule_eval_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return summary


def evaluate_reschedule_rules_manifest(
    *,
    manifest_path: str | Path,
    instance_ids: Any = None,
    methods: Any = None,
    num_runs: int | None = None,
    seed: int = 42,
    output_dir: str | Path | None = None,
    verbose: bool = True,
    beam_width: int = 2,
    beam_branch_factor: int = 2,
    beam_levels: int = 2,
    beam_patience: int = 1,
    ig_iterations: int = 3,
    ig_destroy_ratio: float = 0.08,
    ig_noise_sigma: float = 0.20,
    sa_iterations: int = 3,
    sa_initial_temp: float = 0.05,
    sa_cooling: float = 0.96,
    sa_min_temp: float = 1e-4,
) -> dict[str, Any]:
    """按 manifest 实例批量评估规则，保证与 PPO manifest 评估使用同一数据、baseline 和场景。"""

    manifest = load_reschedule_manifest(manifest_path)
    ids = _as_id_list(instance_ids)
    if not ids:
        ids = [entry.instance_id for entry in manifest.filter(split="eval", source="real")]
    if not ids:
        raise ValueError("manifest 中没有可用于评估的 real/eval 实例，请显式传 instance_ids")

    root = Path(output_dir) if output_dir is not None else None
    instance_summaries: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []

    for instance_id in ids:
        entry = manifest.get(instance_id)
        if entry.scenario_path is None:
            raise ValueError(f"{instance_id} 没有固定场景，不能用于重调度规则可比评估")
        subdir = root / instance_id if root is not None else None
        summary = evaluate_reschedule_rules(
            data_path_or_dir=entry.data_path,
            scenario_path=entry.scenario_path,
            baseline_path=entry.baseline_schedule_path,
            methods=methods,
            num_runs=num_runs,
            seed=seed,
            output_dir=subdir,
            verbose=verbose,
            beam_width=beam_width,
            beam_branch_factor=beam_branch_factor,
            beam_levels=beam_levels,
            beam_patience=beam_patience,
            ig_iterations=ig_iterations,
            ig_destroy_ratio=ig_destroy_ratio,
            ig_noise_sigma=ig_noise_sigma,
            sa_iterations=sa_iterations,
            sa_initial_temp=sa_initial_temp,
            sa_cooling=sa_cooling,
            sa_min_temp=sa_min_temp,
        )
        instance_summaries[instance_id] = summary
        for row in summary["rows"]:
            enriched = dict(row)
            enriched["instance_id"] = instance_id
            enriched["data_path"] = str(entry.data_path)
            enriched["baseline_path"] = str(entry.baseline_schedule_path)
            enriched["scenario_path"] = str(entry.scenario_path)
            rows.append(enriched)
        for row in summary["summary_by_method"]:
            enriched = dict(row)
            enriched["instance_id"] = instance_id
            enriched["data_path"] = str(entry.data_path)
            enriched["baseline_path"] = str(entry.baseline_schedule_path)
            enriched["scenario_path"] = str(entry.scenario_path)
            method_rows.append(enriched)

    payload = {
        "manifest_path": str(resolve_workspace_path(manifest_path).resolve()),
        "instance_ids": ids,
        "methods": _normalize_methods(methods),
        "row_count": len(rows),
        "rows": rows,
        "summary_by_instance_method": method_rows,
        "instances": instance_summaries,
    }
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(root / "reschedule_rule_eval_by_instance.csv", index=False)
        pd.DataFrame(method_rows).to_csv(root / "reschedule_rule_summary_by_instance_method.csv", index=False)
        (root / "reschedule_rule_manifest_summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return payload


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if should_show_help(raw_args):
        print(hydra_help(RULE_EXTRA_ARGS))
        return 0
    try:
        args = initialize_hydra_runtime(
            raw_args,
            target=configs,
            project_root=PROJECT_ROOT,
            default_experiment="reschedule_task_delay",
            extra_arguments=RULE_EXTRA_ARGS,
        )
    except (HydraCliError, KeyError, ValueError, RuntimeError) as exc:
        print(f"[CLI] {exc}", file=sys.stderr)
        return 2

    output_dir, context = resolve_run_output_dir(
        configs,
        PROJECT_ROOT,
        default_legacy_dir="results/reschedule_rules",
        run_subdir="reschedule_rules",
        explicit_dir=args.output_dir,
        section="eval",
    )
    manifest_extra = {
        "run_type": "evaluation",
        "artifact_kind": "reschedule_rules",
        "output_dir": str(output_dir.resolve()),
        "manifest_path": str(args.manifest_path or ""),
        "instance_ids": args.instance_ids,
    }
    if context is not None:
        write_run_context_files(context, configs, command="evaluate_reschedule_rules", extra=manifest_extra)
    else:
        write_run_manifest(output_dir, configs, command="evaluate_reschedule_rules", extra=manifest_extra)

    common_kwargs = {
        "methods": args.methods,
        "num_runs": args.num_runs,
        "seed": int(args.seed),
        "output_dir": output_dir,
        "verbose": not bool(args.quiet),
        "beam_width": int(args.beam_width),
        "beam_branch_factor": int(args.beam_branch_factor),
        "beam_levels": int(args.beam_levels),
        "beam_patience": int(args.beam_patience),
        "ig_iterations": int(args.ig_iterations),
        "ig_destroy_ratio": float(args.ig_destroy_ratio),
        "ig_noise_sigma": float(args.ig_noise_sigma),
        "sa_iterations": int(args.sa_iterations),
        "sa_initial_temp": float(args.sa_initial_temp),
        "sa_cooling": float(args.sa_cooling),
        "sa_min_temp": float(args.sa_min_temp),
    }
    if args.manifest_path:
        summary = evaluate_reschedule_rules_manifest(
            manifest_path=args.manifest_path,
            instance_ids=args.instance_ids,
            **common_kwargs,
        )
    else:
        summary = evaluate_reschedule_rules(
            data_path_or_dir=args.data_path,
            scenario_path=args.scenario_path,
            baseline_path=args.baseline_path,
            **common_kwargs,
        )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, ensure_ascii=False, indent=2))
    print(f"规则重调度评估明细已保存到: {output_dir / 'reschedule_rule_eval.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
