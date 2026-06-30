from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import configs
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from runtime.artifacts import (
    resolve_path,
    run_context as create_run_context,
    uses_runs_layout,
    write_run_context_files,
    write_run_manifest,
)
from runtime.checkpoints import apply_checkpoint_model_spec, load_checkpoint, load_policy_weights
from runtime.configuration import add_common_config_arguments, resolve_runtime_config


def generate_schedule(
    model_path: str,
    *,
    explicit_fields: set[str] | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    checkpoint_path = resolve_path(model_path, PROJECT_ROOT)
    checkpoint = load_checkpoint(checkpoint_path)
    apply_checkpoint_model_spec(
        configs, checkpoint.model_spec, explicit_fields=explicit_fields,
    )
    data_path = resolve_path(configs.data_file_path, PROJECT_ROOT)
    context = create_run_context(configs, PROJECT_ROOT, create_dirs=True) if uses_runs_layout(configs) else None
    default_output = (context.eval_dir / "final_schedule.csv") if context is not None else Path(configs.result_dir) / "final_schedule.csv"
    target = resolve_path(output_path or default_output, PROJECT_ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest_extra = {
        "checkpoint": str(checkpoint_path.resolve()),
        "resource_graph_mode": checkpoint.model_spec.resource_graph_mode,
    }
    if context is not None:
        write_run_context_files(context, configs, command="generate_schedule", extra=manifest_extra)
    else:
        write_run_manifest(target.parent, configs, command="generate_schedule", extra=manifest_extra)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = AirLineEnv_Graph(data_path_or_dir=data_path, seed=int(configs.seed))
    model = HBGATPN(configs).to(device)
    load_policy_weights(model, checkpoint)
    agent = PPOAgent(
        model, configs.lr, configs.gamma, configs.k_epochs,
        configs.eps_clip, device, configs.batch_size, config=configs,
    )
    state = env.reset()
    done = False
    while not done:
        task_mask, station_mask, worker_mask = env.get_masks()
        action, _, _, _, _ = agent.select_action(
            state.to(device),
            mask_task=task_mask.to(device),
            mask_station_matrix=station_mask.to(device),
            mask_worker=worker_mask.to(device),
            deterministic=True,
            temperature=0.0,
        )
        if action is None:
            raise RuntimeError("模型未能产生有效动作，排程生成中止")
        state, _, done, info = env.step(action)
        if "error" in info:
            raise RuntimeError(str(info["error"]))

    rows = [
        {
            "TaskID": int(task_id),
            "StationID": int(station_id) + 1,
            "Team": str(list(team)),
            "Start": float(start),
            "End": float(end),
            "Duration": float(end - start),
        }
        for task_id, station_id, team, start, end in env.assigned_tasks
    ]
    frame = pd.DataFrame(rows).sort_values(["Start", "TaskID"])
    frame.to_csv(target, index=False)
    print(f"排程已保存: {target}")
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用 checkpoint 生成确定性 APAL 排程")
    add_common_config_arguments(parser)
    parser.add_argument("--model-path", "--model_path", dest="model_path", required=True)
    parser.add_argument("--output-path")
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    _, _, explicit = resolve_runtime_config(parsed, target=configs)
    generate_schedule(
        parsed.model_path,
        explicit_fields=explicit,
        output_path=parsed.output_path,
    )
