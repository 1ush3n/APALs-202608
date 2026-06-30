from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import configs
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from train import (
    ensure_reschedule_baseline_available,
    ensure_reschedule_eval_scenarios_available,
    evaluate_reschedule_model,
    load_warm_start_weights_with_input_expansion,
    resolve_workspace_path,
)
from runtime.checkpoints import (
    apply_checkpoint_model_spec,
    load_checkpoint,
    load_policy_weights,
)
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


def _load_policy_weights(model: torch.nn.Module, model_path: Path, device: torch.device) -> dict[str, int | str]:
    """加载 PPO 重调度模型；若输入维度扩展，复用训练时的兼容加载逻辑。"""

    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型权重文件: {model_path}")
    try:
        checkpoint = load_checkpoint(model_path, map_location=device)
        load_policy_weights(model, checkpoint, strict=True)
        return {
            "mode": "exact",
            "loaded_exact": len(checkpoint.state_dict),
            "loaded_expanded": 0,
            "skipped": 0,
        }
    except RuntimeError:
        stats = load_warm_start_weights_with_input_expansion(model, model_path, device)
        return {"mode": "expanded", **stats}


def evaluate_saved_reschedule_model(
    *,
    model_path: Path,
    num_runs: int | None,
    temperature: float,
    output_dir: Path,
) -> dict[str, object]:
    """按固定重调度验证场景评估已保存 PPO 模型，并导出逐场景 CSV。"""

    baseline_path = ensure_reschedule_baseline_available(configs)
    scenario_path = ensure_reschedule_eval_scenarios_available(configs)
    if baseline_path is None or scenario_path is None:
        raise RuntimeError("重调度验证需要 enable_reschedule_mode=True，并且 baseline/固定场景必须可用。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = AirLineEnv_Graph(data_path_or_dir=str(resolve_workspace_path(configs.data_file_path)), seed=int(configs.seed))
    model = HBGATPN(configs).to(device)
    load_stats = _load_policy_weights(model, model_path, device)
    agent = PPOAgent(
        model,
        configs.lr,
        configs.gamma,
        configs.k_epochs,
        configs.eps_clip,
        device,
        batch_size=configs.batch_size,
        total_timesteps=1,
        config=configs,
    )

    start_time = time.time()
    makespan, balance, reward, _best_schedule, duration, worker_util, station_util = evaluate_reschedule_model(
        env,
        agent,
        num_runs=num_runs,
        temperature=temperature,
    )
    elapsed = time.time() - start_time
    rows = list(getattr(evaluate_reschedule_model, "last_scenario_metrics", []))
    avg_metrics = dict(getattr(evaluate_reschedule_model, "last_metrics", {}))

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "reschedule_ppo_eval.csv"
    json_path = output_dir / "reschedule_ppo_eval_summary.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    summary: dict[str, object] = {
        "model_path": str(model_path.resolve()),
        "baseline_path": str(Path(baseline_path).resolve()),
        "scenario_path": str(Path(scenario_path).resolve()),
        "data_path": str(resolve_workspace_path(configs.data_file_path).resolve()),
        "scenario_count": int(len(rows)),
        "makespan": float(makespan),
        "balance_std": float(balance),
        "reward": float(reward),
        "avg_makespan": float(makespan),
        "avg_balance_std": float(balance),
        "avg_reward": float(reward),
        "avg_score": float(avg_metrics.get("composite_score", 0.0)),
        "avg_selection_score": float(avg_metrics.get("selection_score", 0.0)),
        "eligible_rate": float(avg_metrics.get("eligible_rate", 0.0)),
        "avg_duration_sec": float(duration),
        "wall_time_sec": float(elapsed),
        "worker_util": float(worker_util),
        "station_util": float(station_util),
        "load_stats": load_stats,
        "rows": rows,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="评估已保存的 APAL PPO 重调度模型")
    add_common_config_arguments(parser)
    parser.set_defaults(config=["conf/experiment/reschedule_task_delay.yaml"])
    parser.add_argument("--model-path", "--model_path", dest="model_path", default="checkpoints/reschedule_task_delay/bestmodel/best_model.pth")
    parser.add_argument("--num_runs", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parse_runtime_args(parser)

    _, _, explicit_fields = resolve_runtime_config(args, target=configs)
    model_path = resolve_workspace_path(args.model_path)
    checkpoint = load_checkpoint(model_path)
    apply_checkpoint_model_spec(
        configs,
        checkpoint.model_spec,
        explicit_fields=explicit_fields,
    )
    output_dir, context = resolve_run_output_dir(
        configs,
        PROJECT_ROOT,
        default_legacy_dir="results/reschedule_ppo_eval",
        run_subdir="reschedule",
        explicit_dir=getattr(args, "output_dir", None),
        section="eval",
    )
    manifest_extra = {
        "run_type": "evaluation",
        "artifact_kind": "reschedule_ppo",
        "checkpoint": str(model_path.resolve()),
        "model_format": checkpoint.format_name,
        "output_dir": str(output_dir.resolve()),
    }
    if context is not None:
        write_run_context_files(context, configs, command="evaluate_reschedule_model", extra=manifest_extra)
    else:
        write_run_manifest(output_dir, configs, command="evaluate_reschedule_model", extra=manifest_extra)
    summary = evaluate_saved_reschedule_model(
        model_path=model_path,
        num_runs=args.num_runs,
        temperature=float(args.temperature),
        output_dir=output_dir,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, ensure_ascii=False, indent=2))
    print(f"PPO 重调度逐场景明细已保存到: {output_dir / 'reschedule_ppo_eval.csv'}")


if __name__ == "__main__":
    main()
