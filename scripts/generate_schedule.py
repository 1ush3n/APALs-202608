from __future__ import annotations

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
from runtime.initial_worker_mapping import apply_initial_worker_mapping
from runtime.hydra_config import (
    ExtraArgument,
    HydraCliError,
    hydra_help,
    initialize_hydra_runtime,
    should_show_help,
)


GENERATE_EXTRA_ARGS = {
    "model_path": ExtraArgument(required=True, help="用于生成排程的 checkpoint 路径"),
    "data_path": ExtraArgument(default=None, help="可选数据集路径；缺省使用配置 data_file_path"),
    "output_path": ExtraArgument(default=None, help="可选输出 CSV 路径；缺省写入本次 run 的 eval 目录"),
}


def generate_schedule(
    model_path: str,
    *,
    explicit_fields: set[str] | None = None,
    data_path: str | Path | None = None,
    output_path: str | Path | None = None,
    write_context: bool = True,
) -> pd.DataFrame:
    checkpoint_path = resolve_path(model_path, PROJECT_ROOT)
    checkpoint = load_checkpoint(checkpoint_path)
    apply_checkpoint_model_spec(
        configs, checkpoint.model_spec, explicit_fields=explicit_fields,
    )
    resolved_data_path = resolve_path(data_path or configs.data_file_path, PROJECT_ROOT)
    apply_initial_worker_mapping(
        configs,
        resolved_data_path,
        explicit_fields=explicit_fields,
    )
    context = create_run_context(configs, PROJECT_ROOT, create_dirs=True) if write_context and uses_runs_layout(configs) else None
    default_output = (context.eval_dir / "final_schedule.csv") if context is not None else Path(configs.result_dir) / "final_schedule.csv"
    target = resolve_path(output_path or default_output, PROJECT_ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest_extra = {
        "checkpoint": str(checkpoint_path.resolve()),
        "resource_graph_mode": checkpoint.model_spec.resource_graph_mode,
    }
    if write_context and context is not None:
        write_run_context_files(context, configs, command="generate_schedule", extra=manifest_extra)
    elif write_context:
        write_run_manifest(target.parent, configs, command="generate_schedule", extra=manifest_extra)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = AirLineEnv_Graph(data_path_or_dir=resolved_data_path, seed=int(configs.seed))
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


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if should_show_help(raw_args):
        print(hydra_help(GENERATE_EXTRA_ARGS))
        return 0
    try:
        parsed = initialize_hydra_runtime(
            raw_args,
            target=configs,
            project_root=PROJECT_ROOT,
            default_experiment="initial_schedule_283",
            extra_arguments=GENERATE_EXTRA_ARGS,
        )
        generate_schedule(
            parsed.model_path,
            explicit_fields=set(getattr(parsed, "explicit_config_fields", set())),
            data_path=parsed.data_path,
            output_path=parsed.output_path,
        )
    except (HydraCliError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"[CLI] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
