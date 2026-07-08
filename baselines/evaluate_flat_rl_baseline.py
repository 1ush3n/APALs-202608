from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from baselines.basic_ppo.train_basic import BasicPPO
from baselines.dqn.train_dqn import DQN
from baselines.graph_baseline import GRAPH_BASELINE_FEATURE_MODE, select_graph_action
from baselines.heuristic.advanced_schedulers import build_metrics
from configs import configs, load_config_files
from environment import AirLineEnv_Graph
from runtime.artifacts import resolve_run_output_dir, write_run_context_files, write_run_manifest
from runtime.hydra_config import ExtraArgument, HydraCliError, hydra_help, initialize_hydra_runtime, should_show_help
from runtime.seed import set_seed
from training.observation import refresh_env_observation


GRAPH_EVAL_EXTRA_ARGS = {
    "algorithm": ExtraArgument(required=True, help="basic_ppo 或 dqn"),
    "model_path": ExtraArgument(required=True, help="待评估 graph baseline checkpoint"),
    "data_dir": ExtraArgument(default="data", help="数据文件所在目录"),
    "datasets": ExtraArgument(default=["283.csv"], help="数据集列表，例如 datasets=[283.csv,680.csv]"),
    "num_runs": ExtraArgument(default=1, help="重复评估次数"),
    "temperature": ExtraArgument(default=0.0, help="动作采样温度，0 表示确定性"),
    "output_dir": ExtraArgument(default=None, help="可选输出目录；缺省写入本次 run 的 artifacts 目录"),
}


def _as_dataset_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise ValueError(f"无法解析 datasets 参数: {value!r}")


def _load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint 缺少 model_state_dict，疑似旧 flat-state checkpoint，不能用于图泛化 baseline")
    feature_mode = checkpoint.get("feature_mode")
    if feature_mode != GRAPH_BASELINE_FEATURE_MODE:
        raise ValueError(
            f"checkpoint feature_mode={feature_mode!r}，当前只接受 {GRAPH_BASELINE_FEATURE_MODE!r}。"
            "旧 flat-state BasicPPO/DQN checkpoint 不能跨规模泛化，请重新训练图版 baseline。"
        )
    return checkpoint


def _build_model(algorithm: str, checkpoint: dict[str, Any], device: torch.device) -> torch.nn.Module:
    expected = "GraphBasicPPO" if algorithm == "basic_ppo" else "GraphDQN"
    if checkpoint.get("model_type") != expected:
        raise ValueError(f"checkpoint model_type={checkpoint.get('model_type')!r} 与 algorithm={algorithm!r} 不匹配，应为 {expected}")
    for name in ("hidden_dim", "num_gat_layers", "num_heads"):
        if name in checkpoint and checkpoint[name] is not None:
            setattr(configs, name, int(checkpoint[name]))
    for name in ("use_skill_hub", "skill_hub_bidirectional"):
        if name in checkpoint and checkpoint[name] is not None:
            setattr(configs, name, bool(checkpoint[name]))
    model = BasicPPO(configs) if algorithm == "basic_ppo" else DQN(configs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate_model(
    model: torch.nn.Module,
    env: AirLineEnv_Graph,
    device: torch.device,
    *,
    algorithm: str | None = None,
    seed: int,
    num_runs: int,
    temperature: float,
) -> tuple[dict[str, Any], list[Any], list[dict[str, Any]]]:
    run_metrics_list: list[dict[str, Any]] = []
    run_schedules_list: list[list[Any]] = []
    run_makespans_list: list[float] = []
    run_count = max(1, int(num_runs))

    for run_idx in range(run_count):
        run_seed = int(seed) + run_idx
        set_seed(run_seed)
        state = env.reset(randomize_duration=False, randomize_workers=False, seed=run_seed)
        done = False
        invalid_count = 0
        start_time = time.time()

        for _step in range(max(1, int(env.num_tasks) * 4)):
            if done or len(env.assigned_tasks) == env.num_tasks:
                break
            masks = env.get_masks()
            while bool(masks[0].all()):
                if not env.try_wait_for_resources():
                    invalid_count += 1
                    break
                state = refresh_env_observation(env)
                masks = env.get_masks()
            if bool(masks[0].all()):
                break

            with torch.no_grad():
                result = select_graph_action(
                    model,
                    state,
                    masks=masks,
                    device=device,
                    deterministic=float(temperature) <= 0.0,
                    temperature=float(temperature),
                    need_value=False,
                )
            if result.action is None:
                invalid_count += 1
                break

            state, _reward, done, info = env.step(result.action)
            if info.get("invalid_action", False):
                invalid_count += 1
                break

        elapsed = time.time() - start_time
        complete = len(env.assigned_tasks) == env.num_tasks
        is_valid = complete and invalid_count == 0
        schedule = list(env.assigned_tasks) if is_valid else []
        if is_valid:
            makespan = float(np.max(env.station_wall_clock))
            balance = float(np.std(env.station_loads))
        else:
            makespan = float(env.ideal_makespan * 3.0)
            balance = float(env.ideal_station_load * 3.0)

        metrics = build_metrics(env, makespan, balance, schedule, elapsed)
        metrics["deadlock_count"] = int(max(metrics["deadlock_count"], invalid_count))
        metrics["valid"] = 1.0 if is_valid else 0.0
        metrics["completion_rate"] = 1.0 if is_valid else 0.0
        metrics["complete"] = 1.0 if complete else 0.0
        metrics["invalid_action_count"] = float(invalid_count)
        metrics["seed"] = run_seed
        run_metrics_list.append(metrics)
        run_schedules_list.append(schedule)
        run_makespans_list.append(makespan)

    best_idx = int(np.argmin(run_makespans_list))
    best_schedule = run_schedules_list[best_idx]
    avg_metrics: dict[str, Any] = {}
    for key in run_metrics_list[0].keys():
        vals = [m[key] for m in run_metrics_list if key in m]
        if vals and all(isinstance(v, (int, float)) for v in vals):
            avg_metrics[key] = float(np.mean(vals))
        else:
            avg_metrics[key] = run_metrics_list[best_idx].get(key)
    return avg_metrics, best_schedule, run_metrics_list


def save_eval_results(
    method: str,
    dataset_name: str,
    metrics: dict[str, Any],
    assigned_tasks: list[Any],
    run_metrics_list: list[dict[str, Any]],
    output_root: Path,
) -> None:
    output_dir = Path(output_root) / method / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    full_metrics = dict(metrics)
    full_metrics["runs"] = run_metrics_list
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=4, ensure_ascii=False)

    rows = [
        {
            "TaskID": int(task_id),
            "StationID": int(station_id) + 1,
            "Team": str([int(w) for w in team]),
            "Start": float(start),
            "End": float(end),
            "Duration": float(end) - float(start),
        }
        for task_id, station_id, team, start, end in assigned_tasks
    ]
    pd.DataFrame(rows).to_csv(output_dir / "schedule.csv", index=False)
    detail_rows = []
    for r_idx, r_metrics in enumerate(run_metrics_list):
        row = {"RunIdx": r_idx + 1, "Seed": int(r_metrics.get("seed", int(getattr(configs, "seed", 42)) + r_idx))}
        row.update({k: v for k, v in r_metrics.items() if isinstance(v, (int, float)) and k != "seed"})
        detail_rows.append(row)
    pd.DataFrame(detail_rows).to_csv(output_dir / "runs_detail.csv", index=False)
    print(f"[Export Complete] {output_dir}")


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if should_show_help(raw_args):
        print(hydra_help(GRAPH_EVAL_EXTRA_ARGS))
        return 0
    try:
        args = initialize_hydra_runtime(
            raw_args,
            target=configs,
            project_root=PROJECT_ROOT,
            default_experiment="initial_schedule_283",
            extra_arguments=GRAPH_EVAL_EXTRA_ARGS,
        )
    except (HydraCliError, KeyError, ValueError, RuntimeError) as exc:
        print(f"[CLI] {exc}", file=sys.stderr)
        return 2

    if args.algorithm not in {"basic_ppo", "dqn"}:
        print("[CLI] algorithm 必须是 basic_ppo 或 dqn", file=sys.stderr)
        return 2

    args.datasets = _as_dataset_list(args.datasets)
    method_name = "BasicPPO" if args.algorithm == "basic_ppo" else "DQN"
    output_root, context = resolve_run_output_dir(
        configs,
        PROJECT_ROOT,
        default_legacy_dir="results/eval_logs",
        run_subdir="baselines/graph_state",
        explicit_dir=getattr(args, "output_dir", None),
        section="artifacts",
    )
    checkpoint_path = Path(args.model_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    manifest_extra = {
        "run_type": "baseline",
        "artifact_kind": "graph_hetero_baseline",
        "method": method_name,
        "checkpoint": str(checkpoint_path.resolve()),
        "datasets": list(args.datasets),
        "output_dir": str(output_root.resolve()),
    }
    if context is not None:
        write_run_context_files(context, configs, command="evaluate_graph_rl_baseline", extra=manifest_extra)
    else:
        write_run_manifest(output_root, configs, command="evaluate_graph_rl_baseline", extra=manifest_extra)

    checkpoint = _load_checkpoint(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(args.algorithm, checkpoint, device)

    summary_rows = []
    for dataset_file in args.datasets:
        set_seed(int(getattr(configs, "seed", 42)))
        dataset_path = Path(args.data_dir) / dataset_file
        if not dataset_path.is_absolute():
            dataset_path = PROJECT_ROOT / dataset_path
        scale_yaml = PROJECT_ROOT / "conf" / "env" / f"initial_bucket_{dataset_path.stem}.yaml"
        if scale_yaml.exists():
            load_config_files([str(scale_yaml)])
            print(f"[Auto-Config] {dataset_file}: {scale_yaml.name} (n_w={configs.n_w})")

        env = AirLineEnv_Graph(data_path_or_dir=str(dataset_path), seed=int(configs.seed))
        env.reset(randomize_duration=False, randomize_workers=False, seed=int(configs.seed))
        metrics, schedule, run_metrics_list = evaluate_model(
            model,
            env,
            device,
            seed=int(configs.seed),
            num_runs=int(args.num_runs),
            temperature=float(args.temperature),
        )
        save_eval_results(method_name, dataset_path.stem, metrics, schedule, run_metrics_list, output_root)
        for r_idx, r_metrics in enumerate(run_metrics_list):
            summary_rows.append(
                {
                    "Dataset": dataset_path.stem,
                    "Method": method_name,
                    "Run": r_idx + 1,
                    "Seed": int(r_metrics.get("seed", int(configs.seed) + r_idx)),
                    "Makespan": r_metrics["makespan"],
                    "BalanceStd": r_metrics["workload_balance_std"],
                    "WorkerUtil": r_metrics["worker_utilization"],
                    "StationUtil": r_metrics["station_utilization"],
                    "Time(s)": r_metrics["inference_time"],
                    "Valid": r_metrics["valid"],
                }
            )

    summary_path = output_root / f"{method_name}_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"[*] 汇总结果已导出: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
