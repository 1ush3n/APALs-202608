from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from configs import configs
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from runtime.artifacts import resolve_path, write_run_manifest
from runtime.checkpoints import (
    apply_checkpoint_model_spec,
    load_checkpoint,
    load_policy_weights,
)
from runtime.configuration import add_common_config_arguments, resolve_runtime_config
from train import evaluate_model as run_evaluation
from utils.visualization import plot_gantt


PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="评估 APAL PPO checkpoint")
    add_common_config_arguments(parser)
    parser.add_argument("--model-path", "--model_path", dest="model_path", required=True)
    parser.add_argument("--test-data", "--test_data", dest="test_data")
    parser.add_argument("--num-runs", "--num_runs", dest="num_runs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--scenario", action="append", dest="scenarios")
    parser.add_argument("--no-gantt", action="store_true")
    return parser


def main(args: argparse.Namespace) -> dict[str, object]:
    _, _, explicit_fields = resolve_runtime_config(args, target=configs)
    checkpoint_path = resolve_path(args.model_path, PROJECT_ROOT)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    apply_checkpoint_model_spec(
        configs,
        checkpoint.model_spec,
        explicit_fields=explicit_fields,
    )
    if args.test_data:
        configs.data_file_path = args.test_data
    data_path = resolve_path(configs.data_file_path, PROJECT_ROOT)
    output_dir = resolve_path(configs.result_dir, PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(
        output_dir,
        configs,
        command="evaluate",
        extra={
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_format": checkpoint.format_name,
            "resource_graph_mode": checkpoint.model_spec.resource_graph_mode,
        },
    )

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
        scenario_names=args.scenarios or tuple(configs.eval_scenarios),
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
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main(build_parser().parse_args())
