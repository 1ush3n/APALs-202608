from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from baselines.basic_ppo.train_basic import BasicPPO
from baselines.dqn.train_dqn import DQN
from baselines.heuristic.advanced_schedulers import build_metrics
from configs import configs, load_config_files
from env_wrapper import extract_flat_state_for_baselines
from runtime.artifacts import (
    resolve_run_output_dir,
    write_run_context_files,
    write_run_manifest,
)
from runtime.hydra_config import (
    ExtraArgument,
    HydraCliError,
    hydra_help,
    initialize_hydra_runtime,
    should_show_help,
)
from train import set_seed
from environment import AirLineEnv_Graph


FLAT_EVAL_EXTRA_ARGS = {
    "algorithm": ExtraArgument(required=True, help="basic_ppo 或 dqn"),
    "model_path": ExtraArgument(required=True, help="待评估 flat-state baseline checkpoint"),
    "data_dir": ExtraArgument(default="data", help="数据文件所在目录"),
    "datasets": ExtraArgument(default=["283.csv"], help="数据集列表，例如 datasets=[283.csv,680.csv]"),
    "num_runs": ExtraArgument(default=1, help="重复评估次数"),
    "temperature": ExtraArgument(default=0.0, help="动作采样温度，0 表示确定性"),
    "output_dir": ExtraArgument(default=None, help="可选输出目录；缺省写入本次 run 的 artifacts 目录"),
}


def _as_dataset_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise ValueError(f"无法解析 datasets 参数: {value!r}")


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint
    return {"model_state_dict": checkpoint}


def _build_model(algorithm: str, checkpoint: dict[str, Any], state_dim: int, action_dim_list: list[int], device: torch.device):
    ckpt_state_dim = int(checkpoint.get("state_dim", state_dim))
    ckpt_action_dim = [int(v) for v in checkpoint.get("action_dim_list", action_dim_list)]
    if ckpt_state_dim != state_dim:
        raise ValueError(f"checkpoint state_dim={ckpt_state_dim} 与当前环境 state_dim={state_dim} 不一致")
    if ckpt_action_dim != action_dim_list:
        raise ValueError(f"checkpoint action_dim_list={ckpt_action_dim} 与当前环境 action_dim_list={action_dim_list} 不一致")

    if algorithm == "basic_ppo":
        model = BasicPPO(state_dim, action_dim_list)
    elif algorithm == "dqn":
        model = DQN(state_dim, action_dim_list)
    else:
        raise ValueError(f"未知算法: {algorithm}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _select_action_basic_ppo(model: BasicPPO, state: np.ndarray, env: AirLineEnv_Graph, device: torch.device, temperature: float = 0.0) -> Tuple[int, int, List[int]] | None:
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    task_logits, station_logits, worker_logits, _ = model(state_tensor)
    return _select_action_from_logits(task_logits, station_logits, worker_logits, env, device, temperature)


def _select_action_dqn(model: DQN, state: np.ndarray, env: AirLineEnv_Graph, device: torch.device, temperature: float = 0.0) -> Tuple[int, int, List[int]] | None:
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    task_logits, station_logits, worker_logits = model(state_tensor)
    return _select_action_from_logits(task_logits, station_logits, worker_logits, env, device, temperature)


def _select_action_from_logits(
    task_logits: torch.Tensor, 
    station_logits: torch.Tensor, 
    worker_logits: torch.Tensor, 
    env: AirLineEnv_Graph, 
    device: torch.device, 
    temperature: float = 0.0
) -> Tuple[int, int, List[int]] | None:
    task_mask_raw, station_mask_raw, worker_mask_raw = env.get_masks()
    task_mask = task_mask_raw.to(device).bool().unsqueeze(0)
    if bool(task_mask.all()):
        return None

    # Task 选择
    masked_task_logits = task_logits.masked_fill(task_mask, -1e9)
    if temperature > 0.0:
        task_dist = Categorical(logits=masked_task_logits / temperature)
        task_id = int(task_dist.sample().item())
    else:
        task_id = int(torch.argmax(masked_task_logits, dim=-1).item())

    # Station 选择
    station_mask = station_mask_raw[task_id].to(device).bool().unsqueeze(0)
    if bool(station_mask.all()):
        return None
    masked_station_logits = station_logits.masked_fill(station_mask, -1e9)
    if temperature > 0.0:
        station_dist = Categorical(logits=masked_station_logits / temperature)
        station_id = int(station_dist.sample().item())
    else:
        station_id = int(torch.argmax(masked_station_logits, dim=-1).item())

    # Worker 选择
    task_static = _to_numpy(env.task_static_feat)
    worker_skills = _to_numpy(env.worker_skill_matrix)
    worker_locks = _to_numpy(env.worker_locks)
    worker_mask = _to_numpy(worker_mask_raw).astype(bool)
    req_skill = int(task_static[task_id, 1])
    demand = max(1, int(task_static[task_id, 2]))
    has_skill = worker_skills[:, req_skill] > 0.5
    valid_lock = (worker_locks == 0) | (worker_locks == station_id + 1)
    final_worker_mask = worker_mask | (~has_skill) | (~valid_lock)
    valid_workers = np.where(~final_worker_mask)[0].tolist()
    if len(valid_workers) < demand:
        return None

    worker_scores = worker_logits.detach().cpu().numpy()[0]
    if temperature > 0.0:
        # 对有效工人的得分根据温度进行 Softmax 采样（无放回采样选择 demand 个工人）
        valid_worker_logits = torch.tensor([worker_scores[w] for w in valid_workers], dtype=torch.float32, device=device)
        probs = F.softmax(valid_worker_logits / temperature, dim=-1).cpu().numpy()
        probs = probs + 1e-12  # 防御性平滑，防止由于极低温度导致大部分概率下溢为 0.0 导致 np.random.choice 报错
        probs = probs / np.sum(probs)  # 重新归一化
        chosen_indices = np.random.choice(len(valid_workers), size=demand, replace=False, p=probs)
        chosen_workers = [valid_workers[idx] for idx in chosen_indices]
    else:
        valid_workers.sort(key=lambda worker_id: worker_scores[worker_id], reverse=True)
        chosen_workers = valid_workers[:demand]

    return task_id, station_id, chosen_workers


def evaluate_model(
    model: torch.nn.Module, 
    algorithm: str, 
    env: AirLineEnv_Graph, 
    device: torch.device, 
    seed: int, 
    num_runs: int = 1, 
    temperature: float = 0.0
) -> Tuple[Dict[str, Any], List[Any], List[Dict[str, Any]]]:
    run_metrics_list = []
    run_schedules_list = []
    run_makespans_list = []

    valid_count_collected = 0
    attempt_cnt = 0
    max_attempts = num_runs * 10  # 防御性最大重试次数，防止在极差参数下死循环

    while valid_count_collected < num_runs and attempt_cnt < max_attempts:
        run_seed = seed + attempt_cnt
        attempt_cnt += 1
        
        state = env.reset(randomize_duration=False, randomize_workers=False, seed=run_seed)
        _ = state
        done = False
        invalid_count = 0
        start_time = time.time()

        for _step in range(max(1, env.num_tasks * 4)):
            if done or len(env.assigned_tasks) == env.num_tasks:
                break
            task_mask, _, _ = env.get_masks()
            while bool(task_mask.all()):
                if not env.try_wait_for_resources():
                    invalid_count += 1
                    break
                task_mask, _, _ = env.get_masks()
            if bool(task_mask.all()):
                break

            flat_state = extract_flat_state_for_baselines(env)
            with torch.no_grad():
                if algorithm == "basic_ppo":
                    action = _select_action_basic_ppo(model, flat_state, env, device, temperature)
                else:
                    action = _select_action_dqn(model, flat_state, env, device, temperature)
            if action is None:
                invalid_count += 1
                break

            _, _, done, info = env.step(action)
            if info.get("invalid_action", False):
                invalid_count += 1
                break

        elapsed = time.time() - start_time
        complete = len(env.assigned_tasks) == env.num_tasks
        is_valid = complete and invalid_count == 0

        if is_valid:
            valid_count_collected += 1
            makespan = float(np.max(env.station_wall_clock))
            balance = float(np.std(env.station_loads))
            schedule = list(env.assigned_tasks)

            metrics = build_metrics(env, makespan, balance, schedule, elapsed)
            metrics["deadlock_count"] = int(max(metrics["deadlock_count"], invalid_count))
            metrics["valid"] = 1.0
            metrics["completion_rate"] = 1.0
            metrics["seed"] = run_seed

            run_metrics_list.append(metrics)
            run_schedules_list.append(schedule)
            run_makespans_list.append(makespan)
        else:
            print(f"[Warning] Seed {run_seed} resulted in an invalid schedule. Skipping and trying next seed...")

    # 兜底机制：如果确实没有收集到任何一个 valid 的结果，强行保存最后一个结果以防程序崩溃
    if not run_metrics_list and attempt_cnt > 0:
        print("[Warning] No valid schedule collected. Recording the last attempt (even though it's invalid).")
        makespan = float(env.ideal_makespan * 3.0)
        balance = float(env.ideal_station_load * 3.0)
        schedule = []
        metrics = build_metrics(env, makespan, balance, schedule, elapsed)
        metrics["deadlock_count"] = int(max(metrics["deadlock_count"], invalid_count))
        metrics["valid"] = 0.0
        metrics["completion_rate"] = 0.0
        metrics["seed"] = seed + attempt_cnt - 1
        
        run_metrics_list.append(metrics)
        run_schedules_list.append(schedule)
        run_makespans_list.append(makespan)

    # 找到 makespan 最小的那次运行
    best_idx = int(np.argmin(run_makespans_list))
    best_schedule = run_schedules_list[best_idx]

    # 对所有 runs 的 metrics 字典中数值型的 keys 求平均
    avg_metrics = {}
    if run_metrics_list:
        keys = run_metrics_list[0].keys()
        for k in keys:
            vals = [m[k] for m in run_metrics_list if k in m]
            if vals and all(isinstance(v, (int, float)) for v in vals):
                avg_metrics[k] = float(np.mean(vals))
            else:
                avg_metrics[k] = run_metrics_list[best_idx].get(k)
    else:
        avg_metrics = {}

    return avg_metrics, best_schedule, run_metrics_list


def save_eval_results(
    method: str, 
    dataset_name: str, 
    metrics: dict[str, Any], 
    assigned_tasks: list[Any], 
    env: AirLineEnv_Graph, 
    run_metrics_list: List[Dict[str, Any]] | None = None,
    output_root: Path | None = None,
) -> None:
    root = output_root or PROJECT_ROOT / "results" / "eval_logs"
    output_dir = Path(root) / method / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    full_metrics = dict(metrics)
    if run_metrics_list is not None:
        full_metrics["runs"] = run_metrics_list
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=4, ensure_ascii=False)

    rows = []
    for task_id, station_id, team, start, end in assigned_tasks:
        rows.append({
            "TaskID": int(task_id),
            "StationID": int(station_id) + 1,
            "Team": str([int(w) for w in team]),
            "Start": float(start),
            "End": float(end),
            "Duration": float(end) - float(start),
        })
    pd.DataFrame(rows).to_csv(output_dir / "schedule.csv", index=False)
    
    if run_metrics_list is not None:
        detail_rows = []
        for r_idx, r_metrics in enumerate(run_metrics_list):
            row = {
                "RunIdx": r_idx + 1,
                "Seed": int(r_metrics.get("seed", int(getattr(configs, "seed", 42)) + r_idx)),
            }
            for k, v in r_metrics.items():
                if isinstance(v, (int, float)) and k != "seed":
                    row[k] = v
            detail_rows.append(row)
        pd.DataFrame(detail_rows).to_csv(output_dir / "runs_detail.csv", index=False)

    with open(output_dir / "run.log", "w", encoding="utf-8") as f:
        f.write(f"Method: {method}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Metrics:\n{json.dumps(full_metrics, indent=2, ensure_ascii=False)}\n")
    print(f"[Export Complete] {output_dir}")

def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if should_show_help(raw_args):
        print(hydra_help(FLAT_EVAL_EXTRA_ARGS))
        return 0
    try:
        args = initialize_hydra_runtime(
            raw_args,
            target=configs,
            project_root=PROJECT_ROOT,
            default_experiment="initial_schedule_283",
            extra_arguments=FLAT_EVAL_EXTRA_ARGS,
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
        run_subdir="baselines/flat_state",
        explicit_dir=getattr(args, "output_dir", None),
        section="artifacts",
    )
    manifest_extra = {
        "run_type": "baseline",
        "artifact_kind": "flat_state_baseline",
        "method": method_name,
        "checkpoint": str((PROJECT_ROOT / args.model_path).resolve() if not Path(args.model_path).is_absolute() else Path(args.model_path).resolve()),
        "datasets": list(args.datasets),
        "output_dir": str(output_root.resolve()),
    }
    if context is not None:
        write_run_context_files(context, configs, command="evaluate_flat_rl_baseline", extra=manifest_extra)
    else:
        write_run_manifest(output_root, configs, command="evaluate_flat_rl_baseline", extra=manifest_extra)
    checkpoint_path = Path(args.model_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    checkpoint = _load_checkpoint(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        state_dim = int(extract_flat_state_for_baselines(env).shape[0])
        action_dim_list = [int(env.num_tasks), int(env.num_stations), int(env.num_workers)]
        model = _build_model(args.algorithm, checkpoint, state_dim, action_dim_list, device)
        metrics, schedule, run_metrics_list = evaluate_model(
            model, 
            args.algorithm, 
            env, 
            device, 
            seed=int(configs.seed), 
            num_runs=int(args.num_runs), 
            temperature=float(args.temperature)
        )
        save_eval_results(
            method_name,
            dataset_path.stem,
            metrics,
            schedule,
            env,
            run_metrics_list,
            output_root=output_root,
        )
        for r_idx, r_metrics in enumerate(run_metrics_list):
            summary_rows.append({
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
            })

    summary_path = output_root / f"{method_name}_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"[*] 汇总结果已导出: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
