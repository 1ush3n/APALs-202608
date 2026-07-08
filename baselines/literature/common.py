from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from baselines.graph_baseline import select_graph_action
from baselines.heuristic.advanced_schedulers import build_metrics
from configs import configs
from environment import AirLineEnv_Graph
from env_wrapper import standardize_env_step
from runtime.artifacts import resolve_run_output_dir, write_run_context_files, write_run_manifest
from runtime.seed import set_seed
from training.memory import Memory
from training.observation import refresh_env_observation
from utils.visualization import plot_gantt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LITERATURE_FEATURE_MODE = "apal_hetero_graph"


def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def training_data_source(args: Any) -> Path:
    raw = (
        getattr(args, "train_data_path_or_dir", None)
        or getattr(configs, "train_data_path_or_dir", None)
        or getattr(args, "data_path", None)
        or getattr(configs, "data_file_path", None)
        or Path("data") / "283.csv"
    )
    path = resolve_project_path(raw)
    if not path.exists():
        raise FileNotFoundError(f"训练数据不存在: {path}")
    return path


def eval_data_source(args: Any) -> Path:
    raw = getattr(args, "data_path", None) or getattr(configs, "data_file_path", None) or Path("data") / "283.csv"
    path = resolve_project_path(raw)
    if not path.exists():
        raise FileNotFoundError(f"验证数据不存在: {path}")
    return path


def make_training_env(args: Any, *, seed: int) -> AirLineEnv_Graph:
    return AirLineEnv_Graph(data_path_or_dir=str(training_data_source(args)), seed=int(seed))


def make_eval_env(args: Any, *, seed: int) -> AirLineEnv_Graph:
    return AirLineEnv_Graph(data_path_or_dir=str(eval_data_source(args)), seed=int(seed))


def select_episode_dataset(env: AirLineEnv_Graph, episode_index: int, seed: int) -> int:
    count = max(1, int(getattr(env, "dataset_count", 1)))
    if count == 1:
        dataset_idx = 0
    elif bool(getattr(configs, "random_sample_dataset", True)):
        rng = np.random.RandomState(int(seed) + int(episode_index) * 9973)
        dataset_idx = int(rng.randint(0, count))
    else:
        dataset_idx = int(episode_index) % count
    env.switch_dataset(dataset_idx)
    return dataset_idx


def append_ppo_transition(
    memory: Memory,
    *,
    snapshot: dict[str, Any],
    action: tuple[int, int, list[int]],
    logprob: float,
    value: float,
    masks: tuple[Any, Any, Any],
    reward: float,
    done: bool,
) -> None:
    memory.states.append(snapshot)
    memory.actions.append(action)
    memory.logprobs.append(float(logprob))
    memory.values.append(float(value))
    memory.masks.append(masks)
    memory.rewards.append(float(reward))
    memory.is_terminals.append(bool(done))
    memory.is_truncated.append(False)


def collect_ppo_episode(
    env: AirLineEnv_Graph,
    agent: Any,
    device: torch.device,
    *,
    episode_seed: int,
) -> tuple[Memory, dict[str, float]]:
    state = env.reset(randomize_duration=False, randomize_workers=False, seed=int(episode_seed))
    memory = Memory()
    done = False
    total_reward = 0.0
    invalid_count = 0
    start_time = time.time()
    max_steps = max(1, int(env.num_tasks) * 2)

    for _ in range(max_steps):
        if done or len(env.assigned_tasks) == env.num_tasks:
            break
        task_mask, station_mask, worker_mask = env.get_masks()
        while bool(task_mask.all()):
            if not env.try_wait_for_resources():
                invalid_count += 1
                done = True
                break
            state = refresh_env_observation(env)
            task_mask, station_mask, worker_mask = env.get_masks()
        if done or bool(task_mask.all()):
            break

        snapshot = env.get_state_snapshot()
        action_ret = agent.select_action(
            state.to(device),
            mask_task=task_mask.to(device),
            mask_station_matrix=station_mask.to(device),
            mask_worker=worker_mask.to(device),
            deterministic=False,
            temperature=float(getattr(configs, "sample_temperature", 1.0)),
            is_eval=False,
        )
        action, logprob, value, _specific_station_mask, is_invalid = action_ret
        if action is None or is_invalid:
            invalid_count += 1
            total_reward -= 100.0
            done = True
            break

        state, reward, done, info = standardize_env_step(env, action)
        if bool(info.get("invalid_action", False)):
            invalid_count += 1
            done = True
        append_ppo_transition(
            memory,
            snapshot=snapshot,
            action=action,
            logprob=logprob,
            value=value,
            masks=(task_mask, station_mask, worker_mask),
            reward=reward,
            done=done,
        )
        total_reward += float(reward)

    if memory.is_terminals and not memory.is_terminals[-1]:
        memory.is_truncated[-1] = True
    complete = len(env.assigned_tasks) == env.num_tasks
    makespan = float(np.max(env.station_wall_clock)) if complete else float(env.ideal_makespan * 3.0)
    metrics = {
        "reward": float(total_reward),
        "makespan": float(makespan),
        "assigned": float(len(env.assigned_tasks)),
        "complete": 1.0 if complete else 0.0,
        "invalid_count": float(invalid_count),
        "duration_sec": float(time.time() - start_time),
    }
    return memory, metrics


def evaluate_graph_policy(
    model: torch.nn.Module,
    env: AirLineEnv_Graph,
    device: torch.device,
    *,
    seed: int,
    num_runs: int = 1,
    temperature: float = 0.0,
) -> tuple[dict[str, Any], list[Any], list[dict[str, Any]]]:
    run_count = max(1, int(num_runs))
    run_metrics: list[dict[str, Any]] = []
    run_schedules: list[list[Any]] = []
    run_makespans: list[float] = []
    was_training = model.training
    model.eval()

    try:
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
                        done = True
                        break
                    state = refresh_env_observation(env)
                    masks = env.get_masks()
                if done or bool(masks[0].all()):
                    break

                with torch.inference_mode():
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
                state, _reward, done, info = standardize_env_step(env, result.action)
                if bool(info.get("invalid_action", False)):
                    invalid_count += 1
                    break

            elapsed = time.time() - start_time
            complete = len(env.assigned_tasks) == env.num_tasks
            valid = complete and invalid_count == 0
            schedule = list(env.assigned_tasks) if valid else []
            if valid:
                makespan = float(np.max(env.station_wall_clock))
                balance = float(np.std(env.station_loads))
            else:
                makespan = float(env.ideal_makespan * 3.0)
                balance = float(env.ideal_station_load * 3.0)
            metrics = build_metrics(env, makespan, balance, schedule, elapsed)
            metrics["deadlock_count"] = int(max(metrics["deadlock_count"], invalid_count))
            metrics["valid"] = 1.0 if valid else 0.0
            metrics["completion_rate"] = 1.0 if valid else 0.0
            metrics["complete"] = 1.0 if complete else 0.0
            metrics["invalid_action_count"] = float(invalid_count)
            metrics["seed"] = run_seed
            run_metrics.append(metrics)
            run_schedules.append(schedule)
            run_makespans.append(float(makespan))
    finally:
        model.train(was_training)

    best_idx = int(np.argmin(run_makespans))
    avg_metrics: dict[str, Any] = {}
    for key in run_metrics[0].keys():
        vals = [row[key] for row in run_metrics if key in row]
        if vals and all(isinstance(value, (int, float)) for value in vals):
            avg_metrics[key] = float(np.mean(vals))
        else:
            avg_metrics[key] = run_metrics[best_idx].get(key)
    return avg_metrics, run_schedules[best_idx], run_metrics


def prepare_literature_output(args: Any, *, method_name: str, entrypoint: str) -> Path:
    output_root, context = resolve_run_output_dir(
        configs,
        PROJECT_ROOT,
        default_legacy_dir=getattr(configs, "result_dir", "results"),
        run_subdir=Path("baselines") / "literature" / method_name,
        explicit_dir=getattr(args, "output_dir", None),
        section="artifacts",
    )
    setattr(args, "output_dir", str(output_root))
    extra = {
        "baseline": method_name,
        "entrypoint": entrypoint,
        "feature_mode": LITERATURE_FEATURE_MODE,
        "train_data_path_or_dir": str(training_data_source(args)),
        "data_file_path": str(eval_data_source(args)),
    }
    if context is not None:
        write_run_context_files(context, configs, command=f"{method_name}_train", extra=extra)
    else:
        write_run_manifest(output_root, configs, command=f"{method_name}_train", extra=extra)
    return output_root


def save_literature_checkpoint(
    path: Path,
    *,
    algorithm: str,
    literature_family: str,
    model: torch.nn.Module,
    best_makespan: float,
    args: Any,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "algorithm": algorithm,
        "literature_family": literature_family,
        "feature_mode": LITERATURE_FEATURE_MODE,
        "model_state_dict": model.state_dict(),
        "seed": int(getattr(configs, "seed", 42)),
        "train_data_path_or_dir": str(training_data_source(args)),
        "data_file_path": str(eval_data_source(args)),
        "config_paths": list(getattr(configs, "config_paths", ())),
        "use_skill_hub": bool(getattr(configs, "use_skill_hub", False)),
        "skill_hub_bidirectional": bool(getattr(configs, "skill_hub_bidirectional", False)),
        "hidden_dim": int(getattr(configs, "hidden_dim", 128)),
        "num_gat_layers": int(getattr(configs, "num_gat_layers", 1)),
        "num_heads": int(getattr(configs, "num_heads", 1)),
        "best_makespan": float(best_makespan) if np.isfinite(best_makespan) else None,
        **(extra or {}),
    }
    torch.save(payload, path)
    meta = {key: value for key, value in payload.items() if not key.endswith("state_dict")}
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def export_best_schedule(path: Path, schedule: list[Any], *, title: str) -> None:
    rows = [
        {
            "TaskID": int(task_id),
            "StationID": int(station_id) + 1,
            "Team": str([int(worker) for worker in team]),
            "Start": float(start),
            "End": float(end),
            "Duration": float(end) - float(start),
        }
        for task_id, station_id, team, start, end in schedule
    ]
    pd.DataFrame(rows).to_csv(path / f"{title}_schedule.csv", index=False)
    if schedule:
        plot_gantt(schedule, str(path / f"{title}_gantt.png"))


def write_training_metrics(output_dir: Path, rows: list[dict[str, Any]], filename: str = "train_metrics.csv") -> None:
    pd.DataFrame(rows).to_csv(Path(output_dir) / filename, index=False)


def load_training_metrics(output_dir: Path, *, before_episode: int, filename: str = "train_metrics.csv") -> list[dict[str, Any]]:
    path = Path(output_dir) / filename
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if "episode" in frame.columns:
        frame = frame[frame["episode"].astype(int) < int(before_episode)]
    return frame.to_dict("records")
