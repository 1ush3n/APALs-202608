from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from baselines.graph_baseline import GRAPH_BASELINE_FEATURE_MODE, GraphBaselineActorCritic
from configs import configs
from env_wrapper import init_env, standardize_env_step
from ppo_agent import PPOAgent
from runtime.artifacts import resolve_run_output_dir, write_run_context_files, write_run_manifest
from runtime.hydra_config import ExtraArgument, HydraCliError, hydra_help, initialize_hydra_runtime, should_show_help
from runtime.seed import set_seed
from training.memory import Memory
from training.observation import refresh_env_observation
from utils.device_utils import clear_torch_cache, get_available_device
from utils.logger import init_logger, record_experiment_time
from utils.visualization import plot_gantt


BASELINE_NAME = "basic_ppo_baseline"
BASELINE_EXTRA_ARGS = {
    "output_dir": ExtraArgument(default=None, help="可选输出目录；缺省写入本次 run 的 artifacts/baselines 目录"),
}


class BasicPPO(GraphBaselineActorCritic):
    """图观测版 BasicPPO 网络，保留类名用于 checkpoint 与评估入口兼容。"""


def _prepare_output_root(args: Any) -> None:
    output_root, context = resolve_run_output_dir(
        configs,
        PROJECT_ROOT,
        default_legacy_dir=getattr(configs, "result_dir", "results"),
        run_subdir=Path("baselines") / "graph_training" / "BasicPPO",
        explicit_dir=getattr(args, "output_dir", None),
        section="artifacts",
    )
    setattr(args, "output_dir", str(output_root))
    extra = {"baseline": "BasicPPO", "entrypoint": "baselines/basic_ppo/train_basic.py", "feature_mode": GRAPH_BASELINE_FEATURE_MODE}
    if context is not None:
        write_run_context_files(context, configs, command="basic_ppo_train", extra=extra)
    else:
        write_run_manifest(output_root, configs, command="basic_ppo_train", extra=extra)


def _save_basic_ppo_checkpoint(path: Path, agent: PPOAgent, best_makespan: float, exp_dir: Path) -> None:
    path = Path(path)
    payload = {
        "algorithm": "BasicPPO",
        "model_type": "GraphBasicPPO",
        "feature_mode": GRAPH_BASELINE_FEATURE_MODE,
        "model_state_dict": agent.policy.state_dict(),
        "seed": int(getattr(configs, "seed", 42)),
        "data_file_path": str(getattr(configs, "data_file_path", "")),
        "config_paths": list(getattr(configs, "config_paths", ())),
        "use_skill_hub": bool(getattr(configs, "use_skill_hub", False)),
        "skill_hub_bidirectional": bool(getattr(configs, "skill_hub_bidirectional", False)),
        "hidden_dim": int(getattr(configs, "hidden_dim", 128)),
        "num_gat_layers": int(getattr(configs, "num_gat_layers", 1)),
        "num_heads": int(getattr(configs, "num_heads", 1)),
        "best_makespan": float(best_makespan) if np.isfinite(best_makespan) else None,
    }
    torch.save(payload, path)
    metadata = {k: v for k, v in payload.items() if k != "model_state_dict"}
    with open(Path(exp_dir) / f"{path.stem}_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def _append_transition(memory: Memory, *, snapshot: dict[str, Any], action: Any, logprob: float, value: float, masks: tuple[Any, Any, Any], reward: float, done: bool) -> None:
    memory.states.append(snapshot)
    memory.actions.append(action)
    memory.logprobs.append(float(logprob))
    memory.values.append(float(value))
    memory.masks.append(masks)
    memory.rewards.append(float(reward))
    memory.is_terminals.append(bool(done))
    memory.is_truncated.append(False)


def _collect_episode(env: Any, agent: PPOAgent, device: torch.device, episode_seed: int) -> tuple[Memory, float, float, int]:
    state = env.reset(randomize_duration=False, randomize_workers=False, seed=int(episode_seed))
    memory = Memory()
    done = False
    total_reward = 0.0
    max_steps = max(1, int(env.num_tasks) * 2)

    for _ in range(max_steps):
        if done or len(env.assigned_tasks) == env.num_tasks:
            break
        task_mask, station_mask, worker_mask = env.get_masks()
        while bool(task_mask.all()):
            if not env.try_wait_for_resources():
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
            done = True
            total_reward -= 100.0
            break

        state, reward, done, info = standardize_env_step(env, action)
        if bool(info.get("invalid_action", False)):
            done = True
        _append_transition(
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
    makespan = float(np.max(env.station_wall_clock)) if len(env.assigned_tasks) == env.num_tasks else float(env.ideal_makespan * 3.0)
    return memory, total_reward, makespan, len(env.assigned_tasks)


def train_basic_ppo(args: Any) -> None:
    set_seed(int(getattr(configs, "seed", 42)))
    _prepare_output_root(args)
    logger, exp_dir_raw = init_logger(args, BASELINE_NAME)
    exp_dir = Path(exp_dir_raw)
    start_time = time.time()

    try:
        device = get_available_device()
        env = init_env(args, seed=getattr(args, "seed", None))
        model = BasicPPO(configs).to(device)
        lr = float(getattr(args, "lr", None) or getattr(configs, "lr", 3e-4))
        gamma = float(getattr(args, "gamma", None) or getattr(configs, "gamma", 0.99))
        k_epochs = int(getattr(args, "k_epochs", None) or getattr(configs, "k_epochs", 4))
        eps_clip = float(getattr(args, "clip_epsilon", None) or getattr(configs, "eps_clip", 0.2))
        batch_size = int(getattr(args, "batch_size", None) or getattr(configs, "batch_size", 64))
        max_episodes = int(getattr(args, "max_episodes", None) or getattr(configs, "max_episodes", 300))
        agent = PPOAgent(model, lr, gamma, k_epochs, eps_clip, device, batch_size=batch_size, total_timesteps=max_episodes, config=configs)

        episode_rewards: list[float] = []
        episode_losses: list[float] = []
        episode_makespans: list[float] = []
        best_makespan = float("inf")

        logger.info(
            "开始图观测 BasicPPO 训练，使用 HeteroData/mask/env.step reward，"
            f"max_episodes={max_episodes}, batch_size={batch_size}"
        )
        for ep in range(max_episodes):
            memory, ep_reward, makespan, assigned_count = _collect_episode(env, agent, device, int(configs.seed) + ep)
            metrics = agent.update(memory, env, current_ep=ep + 1) if memory.rewards else {"Loss/Total": 0.0}
            memory.clear()

            episode_rewards.append(float(ep_reward))
            episode_losses.append(float(metrics.get("Loss/Total", 0.0)))
            episode_makespans.append(float(makespan))

            if makespan < best_makespan and assigned_count == env.num_tasks:
                best_makespan = makespan
                best_sch = env.assigned_tasks.copy()
                rows = [
                    {"TaskID": tid, "StationID": sid + 1, "Team": str(team), "Start": start, "End": end, "Duration": end - start}
                    for (tid, sid, team, start, end) in best_sch
                ]
                pd.DataFrame(rows).to_csv(exp_dir / "Best_Schedule_BasicPPO.csv", index=False)
                plot_gantt(best_sch, str(exp_dir / "Best_Gantt_BasicPPO.png"))
                _save_basic_ppo_checkpoint(exp_dir / "basic_ppo_model_best.pth", agent, best_makespan, exp_dir)
                logger.info(f"新的最佳图 BasicPPO 调度已保存，Makespan={best_makespan:.2f}")

            if (ep + 1) % 10 == 0:
                logger.info(
                    f"Episode {ep+1:04d}/{max_episodes} | "
                    f"Reward={ep_reward:.3f} Avg10={np.mean(episode_rewards[-10:]):.3f} | "
                    f"Loss={np.mean(episode_losses[-10:]):.6f} | "
                    f"Makespan={makespan:.2f} Avg10={np.mean(episode_makespans[-10:]):.2f}"
                )
            if (ep + 1) % 100 == 0:
                clear_torch_cache()
                _save_basic_ppo_checkpoint(exp_dir / f"basic_ppo_model_ep{ep+1}.pth", agent, best_makespan, exp_dir)

        _save_basic_ppo_checkpoint(exp_dir / "basic_ppo_model_final.pth", agent, best_makespan, exp_dir)
        results = pd.DataFrame({"episode": range(1, max_episodes + 1), "reward": episode_rewards, "loss": episode_losses, "makespan": episode_makespans})
        results["avg_makespan_10"] = results["makespan"].rolling(window=10).mean()
        results.to_csv(exp_dir / "basic_ppo_results.csv", index=False)
    except Exception as exc:
        logger.error(f"图 BasicPPO 训练失败: {exc}", exc_info=True)
        raise
    finally:
        record_experiment_time(logger, start_time)
        clear_torch_cache()


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if should_show_help(raw_args):
        print(hydra_help(BASELINE_EXTRA_ARGS))
        return 0
    try:
        args = initialize_hydra_runtime(
            raw_args,
            target=configs,
            project_root=PROJECT_ROOT,
            default_experiment="initial_schedule_283",
            extra_arguments=BASELINE_EXTRA_ARGS,
        )
        train_basic_ppo(args)
    except (HydraCliError, KeyError, ValueError, RuntimeError) as exc:
        print(f"[CLI] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
