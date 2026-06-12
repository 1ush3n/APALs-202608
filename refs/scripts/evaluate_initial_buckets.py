from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import Config, configs, load_config_files
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from train import refresh_env_observation, set_seed
from utils.verify_schedule import verify_schedule


EXPERIMENTS = {
    "283": "conf/experiment/initial_schedule_283.yaml",
    "680": "conf/experiment/initial_schedule_680.yaml",
    "2338": "conf/experiment/initial_schedule_2338.yaml",
    "3182": "conf/experiment/initial_schedule_3182.yaml",
}


def _reset_global_config(config_path: Path) -> None:
    defaults = Config()
    configs.__dict__.clear()
    configs.__dict__.update(defaults.__dict__)
    load_config_files([str(config_path)], target=configs)


def _load_model(model_path: Path, device: torch.device) -> PPOAgent:
    model = HBGATPN(configs).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=True)
    return PPOAgent(
        model,
        configs.lr,
        configs.gamma,
        configs.k_epochs,
        configs.eps_clip,
        device,
        batch_size=configs.batch_size,
        total_timesteps=1,
    )


def _run_standard_episode(
    env: AirLineEnv_Graph,
    agent: PPOAgent,
    *,
    seed: int,
) -> tuple[float, float, float, list[Any]]:
    state = env.reset(
        randomize_duration=False,
        randomize_workers=False,
        seed=seed,
    )
    done = False
    total_reward = 0.0
    start_time = time.perf_counter()
    agent.policy.eval()

    while not done:
        task_mask, station_mask, worker_mask = env.get_masks()
        if bool(task_mask.all()):
            if env.try_wait_for_resources():
                state = refresh_env_observation(env)
                continue
            break
        action, _, _, _, invalid = agent.select_action(
            state.to(agent.device),
            mask_task=task_mask.to(agent.device),
            mask_station_matrix=station_mask.to(agent.device),
            mask_worker=worker_mask.to(agent.device),
            deterministic=True,
            temperature=0.0,
            is_eval=True,
        )
        if action is None or invalid:
            break
        state, reward, done, _ = env.step(action)
        total_reward += float(reward)

    if len(env.assigned_tasks) != env.num_tasks:
        raise RuntimeError(
            f"排程未完成: assigned={len(env.assigned_tasks)}, tasks={env.num_tasks}"
        )
    makespan = float(np.max(env.station_wall_clock))
    duration = time.perf_counter() - start_time
    return makespan, total_reward, duration, list(env.assigned_tasks)


def _write_schedule(schedule: list[Any], output_path: Path) -> None:
    rows = [
        {
            "TaskID": int(task_id),
            "StationID": int(station_id) + 1,
            "Team": str(list(team)),
            "Start": float(start),
            "End": float(end),
            "Duration": float(end) - float(start),
        }
        for task_id, station_id, team, start, end in schedule
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _utilization(
    env: AirLineEnv_Graph,
    schedule: list[Any],
    makespan: float,
) -> tuple[float, float]:
    worker_busy = 0.0
    station_busy = np.zeros(env.num_stations, dtype=float)
    for _, station_id, team, start, end in schedule:
        duration = max(0.0, float(end) - float(start))
        worker_busy += duration * len(team)
        if station_id >= 0:
            station_busy[int(station_id)] += duration
    worker_util = worker_busy / (env.num_workers * makespan)
    station_util = station_busy.sum() / (
        env.num_stations * int(configs.max_slots_per_station) * makespan
    )
    return float(worker_util), float(station_util)


def main() -> None:
    parser = argparse.ArgumentParser(description="统一评估四个窄规模 APAL 初始调度模型")
    parser.add_argument("--models_root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/initial_bucket_eval"))
    parser.add_argument("--seed", type=int, default=20260)
    args = parser.parse_args()

    models_root = args.models_root if args.models_root.is_absolute() else PROJECT_ROOT / args.models_root
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records: list[dict[str, Any]] = []

    for name, config_rel in EXPERIMENTS.items():
        config_path = PROJECT_ROOT / config_rel
        _reset_global_config(config_path)
        set_seed(args.seed)
        model_path = models_root / f"initial_schedule_{name}" / "bestmodel" / "best_model.pth"
        if not model_path.exists() and name == "283":
            compatibility_path = models_root / "initial_schedule" / "bestmodel" / "best_model.pth"
            if compatibility_path.exists():
                model_path = compatibility_path
        if not model_path.exists():
            raise FileNotFoundError(f"[{name}] 找不到模型: {model_path}")

        env = AirLineEnv_Graph(
            data_path_or_dir=str(PROJECT_ROOT / configs.data_file_path),
            seed=args.seed,
        )
        agent = _load_model(model_path, device)
        makespan, reward, duration, schedule = _run_standard_episode(
            env,
            agent,
            seed=args.seed,
        )
        schedule_path = output_dir / f"initial_{name}_schedule.csv"
        _write_schedule(schedule, schedule_path)
        verify_output = io.StringIO()
        with contextlib.redirect_stdout(verify_output):
            eligible = bool(verify_schedule(PROJECT_ROOT / configs.data_file_path, schedule_path))
        if not eligible:
            (output_dir / f"initial_{name}_verification.txt").write_text(
                verify_output.getvalue(),
                encoding="utf-8",
            )

        worker_util, station_util = _utilization(env, schedule, makespan)
        normalized = makespan / max(float(env.ideal_makespan), 1e-8)
        records.append(
            {
                "dataset": name,
                "config_path": str(config_path),
                "model_path": str(model_path),
                "data_path": str(PROJECT_ROOT / configs.data_file_path),
                "makespan": makespan,
                "ideal_makespan": float(env.ideal_makespan),
                "normalized_makespan": normalized,
                "eligible": eligible,
                "worker_util": worker_util,
                "station_util": station_util,
                "reward": reward,
                "duration_sec": duration,
                "schedule_path": str(schedule_path),
            }
        )
        print(
            f"[{name}] makespan={makespan:.3f}, normalized={normalized:.4f}, "
            f"eligible={int(eligible)}, W={worker_util:.3f}, S={station_util:.3f}"
        )
        del agent, env
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    eligible_records = [record for record in records if record["eligible"]]
    summary = {
        "device": str(device),
        "dataset_count": len(records),
        "eligible_rate": len(eligible_records) / len(records),
        "average_normalized_makespan": (
            float(np.mean([record["normalized_makespan"] for record in eligible_records]))
            if eligible_records
            else None
        ),
        "datasets": records,
    }
    pd.DataFrame(records).to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"汇总: eligible_rate={summary['eligible_rate']:.2f}, "
        f"average_normalized_makespan={summary['average_normalized_makespan']}"
    )


if __name__ == "__main__":
    main()
