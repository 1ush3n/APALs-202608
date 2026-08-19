from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def _sanitize_thread_env() -> None:
    """修正非法线程数环境变量，避免 libgomp 在导入计算库时告警或异常。"""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        value = os.environ.get(name)
        if value is None:
            continue
        parts = [part.strip() for part in str(value).split(",")]
        valid = bool(parts) and all(part.isdigit() and int(part) > 0 for part in parts)
        if not valid:
            os.environ[name] = "1"


_sanitize_thread_env()

import pandas as pd
import torch

from configs import configs
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from runtime.artifacts import resolve_path, write_run_manifest
from runtime.artifacts import run_context as create_run_context, uses_runs_layout, write_run_context_files
from runtime.checkpoints import (
    apply_checkpoint_model_spec,
    load_checkpoint,
    load_policy_weights,
)
from runtime.initial_worker_mapping import apply_initial_worker_mapping
from runtime.hydra_config import (
    ExtraArgument,
    HydraCliError,
    hydra_help,
    initialize_hydra_runtime,
    should_show_help,
)
from runtime.evaluation import evaluate_model as run_evaluation
from runtime.seed import set_seed
from utils.visualization import plot_gantt


PROJECT_ROOT = Path(__file__).resolve().parent

EVAL_EXTRA_ARGS = {
    "model_path": ExtraArgument(required=True, help="待评估 checkpoint 路径"),
    "test_data": ExtraArgument(default=None, help="可选测试数据路径；缺省使用实验配置"),
    "num_runs": ExtraArgument(default=1, help="重复评估次数"),
    "temperature": ExtraArgument(default=0.0, help="动作采样温度，0 表示确定性"),
    "scenario": ExtraArgument(default=None, help="单个评估场景，例如 scenario=standard"),
    "scenarios": ExtraArgument(default=None, help="场景列表，例如 scenarios=[standard,mb]"),
    "no_gantt": ExtraArgument(default=False, help="是否跳过甘特图输出"),
    "output_dir": ExtraArgument(default=None, help="可选输出目录；缺省写入本次 run 的 eval 目录"),
}


def _resolve_scenarios(args: Any) -> list[str] | tuple[str, ...]:
    raw = getattr(args, "scenarios", None)
    single = getattr(args, "scenario", None)
    if raw is None and single is None:
        return tuple(configs.eval_scenarios)
    values = raw if raw is not None else single
    if isinstance(values, str):
        return [values]
    if isinstance(values, (list, tuple)):
        return [str(item) for item in values]
    raise ValueError(f"无法解析评估场景参数: {values!r}")


def main(args: Any) -> dict[str, object]:
    explicit_fields = set(getattr(args, "explicit_config_fields", set()))
    # 评估温度大于 0 时仍会进行动作采样；必须在加载模型和创建环境前锁定全局随机流。
    set_seed(int(getattr(configs, "seed", 42)))
    context = None
    if uses_runs_layout(configs):
        context = create_run_context(configs, PROJECT_ROOT, create_dirs=True)
    if "verbose_eval_progress" not in explicit_fields:
        configs.verbose_eval_progress = True
    checkpoint_path = resolve_path(args.model_path, PROJECT_ROOT)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    # 关键：与训练中异步验证（async_eval_worker）保持一致——先恢复训练时的完整配置，
    # 再应用 checkpoint model_spec 与 CLI 显式覆盖。否则评估会沿用 CLI 实验 YAML 的
    # 默认值（如 lightning_precision=16-mixed、randomize_durations=true），与训练/异步
    # 验证（bf16-mixed、false）不一致，导致相同权重在两条评估路径上排出不同排程。
    saved_config = checkpoint.metadata.get("config")
    if isinstance(saved_config, dict):
        for key, value in saved_config.items():
            if key in explicit_fields:
                continue
            if hasattr(configs, key):
                setattr(configs, key, value)
    apply_checkpoint_model_spec(
        configs,
        checkpoint.model_spec,
        explicit_fields=explicit_fields,
    )
    if args.test_data:
        configs.data_file_path = args.test_data
    data_path = resolve_path(configs.data_file_path, PROJECT_ROOT)
    mapped_workers = apply_initial_worker_mapping(
        configs,
        data_path,
        explicit_fields=explicit_fields,
    )
    if mapped_workers is not None:
        print(
            f"[InitialEnv] dataset={data_path.name} legacy_worker_count={mapped_workers} "
            f"max_slots_per_station={int(configs.max_slots_per_station)}",
            flush=True,
        )
    if args.output_dir:
        output_dir = resolve_path(args.output_dir, PROJECT_ROOT)
    elif context is not None and "result_dir" not in explicit_fields:
        output_dir = context.eval_dir
    else:
        output_dir = resolve_path(configs.result_dir, PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = _resolve_scenarios(args)
    print(
        "[Eval] "
        f"checkpoint={checkpoint_path} data={data_path} runs={int(args.num_runs)} "
        f"temperature={float(args.temperature)} scenarios={scenarios} "
        f"output_dir={output_dir} no_gantt={bool(args.no_gantt)}",
        flush=True,
    )
    manifest_extra = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_format": checkpoint.format_name,
        "resource_graph_mode": checkpoint.model_spec.resource_graph_mode,
    }
    if context is not None:
        write_run_context_files(context, configs, command="evaluate", extra=manifest_extra)
    else:
        write_run_manifest(output_dir, configs, command="evaluate", extra=manifest_extra)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HBGATPN(configs).to(device)
    load_policy_weights(model, checkpoint, strict=True)
    agent = PPOAgent(
        model, configs.lr, configs.gamma, configs.k_epochs,
        configs.eps_clip, device, configs.batch_size, config=configs,
    )
    env = AirLineEnv_Graph(data_path_or_dir=data_path, seed=int(configs.seed))
    result = run_evaluation(
        env,
        agent,
        num_runs=int(args.num_runs),
        temperature=float(args.temperature),
        scenario_names=scenarios,
    )
    makespan, balance, reward, schedule, duration, worker_util, station_util = result
    rows = [
        {
            "TaskID": int(task_id),
            "StationID": int(station_id) + 1,
            "Team": str(list(team)),
            "Start": float(start),
            "End": float(end),
            "Duration": float(end - start),
        }
        for task_id, station_id, team, start, end in schedule
    ]
    schedule_path = output_dir / "schedule.csv"
    pd.DataFrame(rows).to_csv(schedule_path, index=False)
    if schedule and not args.no_gantt:
        plot_gantt(schedule, output_dir / "gantt.png")
    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_format": checkpoint.format_name,
        "resource_graph_mode": checkpoint.model_spec.resource_graph_mode,
        "data_path": str(data_path.resolve()),
        "scheduled_tasks": len(schedule),
        "makespan": float(makespan),
        "balance_std": float(balance),
        "reward": float(reward),
        "duration_sec": float(duration),
        "worker_utilization": float(worker_util),
        "station_utilization": float(station_util),
    }
    print(
        "[Eval][Result] "
        f"Tasks={len(schedule)} Mk={float(makespan):.2f} Bal={float(balance):.2f} "
        f"Reward={float(reward):.2f} Time={float(duration):.2f}s "
        f"WUtil={float(worker_util) * 100:.1f}% SUtil={float(station_util) * 100:.1f}% "
        f"summary={output_dir / 'summary.json'} schedule={schedule_path}",
        flush=True,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def cli_main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if should_show_help(raw_args):
        print(hydra_help(EVAL_EXTRA_ARGS))
        return 0
    try:
        runtime_args = initialize_hydra_runtime(
            raw_args,
            target=configs,
            project_root=PROJECT_ROOT,
            default_experiment="initial_schedule_283",
            extra_arguments=EVAL_EXTRA_ARGS,
        )
        main(runtime_args)
    except (HydraCliError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"[CLI] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
