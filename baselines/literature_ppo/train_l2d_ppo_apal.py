from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("APAL_QUIET_DATALOADER", "1")
for _thread_env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _thread_env_value = os.environ.get(_thread_env_name, "")
    if _thread_env_value and not str(_thread_env_value).isdigit():
        os.environ[_thread_env_name] = "1"

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from baselines.graph_baseline import GraphBaselineActorCritic
from baselines.literature.common import (
    collect_ppo_episode,
    evaluate_graph_policy,
    export_best_schedule,
    load_training_metrics,
    make_eval_env,
    make_training_env,
    prepare_literature_output,
    save_literature_checkpoint,
    select_episode_dataset,
    write_training_metrics,
)
from configs import configs
from ppo_agent import PPOAgent
from runtime.hydra_config import ExtraArgument, HydraCliError, hydra_help, initialize_hydra_runtime, should_show_help
from runtime.seed import set_seed
from utils.device_utils import clear_torch_cache, get_available_device


METHOD_NAME = "Simple-HeteroGAT-PPO"
ENTRYPOINT = "baselines/literature_ppo/train_l2d_ppo_apal.py"
EXTRA_ARGS = {
    "output_dir": ExtraArgument(default=None, help="可选输出目录；缺省写入 runs artifacts"),
}


class SimpleHeteroGATPPO(GraphBaselineActorCritic):
    """历史联合动作异构图 PPO；不再将其错误标记为 L2D。"""


def _save_checkpoint(path: Path, agent: PPOAgent, best_makespan: float, args: Any, *, episode: int) -> None:
    save_literature_checkpoint(
        path,
        algorithm=METHOD_NAME,
        literature_family="learned_dispatching_rule_ppo",
        model=agent.policy,
        best_makespan=best_makespan,
        args=args,
        extra={
            "optimizer": "PPO",
            "model_type": "SimpleHeteroGATPPO",
            "batch_size": int(agent.batch_size),
            "k_epochs": int(agent.k_epochs),
            "eps_clip": float(agent.eps_clip),
            "episode": int(episode),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "scaler_state_dict": agent.scaler.state_dict(),
        },
    )


def _load_resume_checkpoint(path: Path, agent: PPOAgent) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=agent.device, weights_only=False)
    if checkpoint.get("algorithm") != METHOD_NAME:
        raise ValueError(f"checkpoint algorithm={checkpoint.get('algorithm')!r}，不是 {METHOD_NAME}")
    agent.policy.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scaler_state_dict" in checkpoint:
        agent.scaler.load_state_dict(checkpoint["scaler_state_dict"])
    episode = int(checkpoint.get("episode", 0))
    best_makespan = checkpoint.get("best_makespan")
    return episode + 1, float(best_makespan) if best_makespan is not None else float("inf")


def train(args: Any) -> None:
    seed = int(getattr(configs, "seed", 42))
    set_seed(seed)
    output_dir = prepare_literature_output(args, method_name=METHOD_NAME, entrypoint=ENTRYPOINT)
    start_time = time.time()
    device = get_available_device()

    train_env = make_training_env(args, seed=seed)
    eval_env = make_eval_env(args, seed=seed)
    model = SimpleHeteroGATPPO(configs).to(device)
    max_episodes = int(getattr(configs, "max_episodes", 300))
    batch_size = int(getattr(configs, "batch_size", 64))
    agent = PPOAgent(
        model,
        float(getattr(configs, "lr", 3e-4)),
        float(getattr(configs, "gamma", 0.99)),
        int(getattr(configs, "k_epochs", 2)),
        float(getattr(configs, "eps_clip", 0.2)),
        device,
        batch_size=batch_size,
        total_timesteps=max_episodes,
        config=configs,
    )

    latest_path = output_dir / "simple_heterogat_ppo_latest.pth"
    best_path = output_dir / "simple_heterogat_ppo_best.pth"
    final_path = output_dir / "simple_heterogat_ppo_final.pth"
    start_episode = 1
    rows: list[dict[str, Any]] = []
    best_makespan = float("inf")
    best_eval_schedule: list[Any] = []

    if bool(getattr(args, "resume", False)):
        if not latest_path.exists():
            raise FileNotFoundError(f"找不到可恢复的 {METHOD_NAME} checkpoint: {latest_path}")
        start_episode, best_makespan = _load_resume_checkpoint(latest_path, agent)
        rows = load_training_metrics(output_dir, before_episode=start_episode)
        print(
            f"[{METHOD_NAME}] resume checkpoint={latest_path} "
            f"start_episode={start_episode} best={best_makespan:.2f}",
            flush=True,
        )

    print(
        f"[{METHOD_NAME}] start episodes={max_episodes} batch_size={batch_size} "
        f"train_datasets={train_env.dataset_count} eval_data={getattr(configs, 'data_file_path', '')}",
        flush=True,
    )

    for episode in range(start_episode, max_episodes + 1):
        dataset_idx = select_episode_dataset(train_env, episode, seed)
        episode_seed = seed + episode
        agent.policy.train()
        memory, train_metrics = collect_ppo_episode(train_env, agent, device, episode_seed=episode_seed)
        print(
            f"[{METHOD_NAME}][Rollout] ep={episode}/{max_episodes} ds={dataset_idx} "
            f"R={train_metrics['reward']:.3f} Mk={train_metrics['makespan']:.2f} "
            f"Done={train_metrics['complete'] * 100:.1f}% "
            f"Assigned={int(train_metrics['assigned'])}/{int(train_env.num_tasks)} "
            f"T={train_metrics['duration_sec']:.1f}s",
            flush=True,
        )
        if memory.rewards:
            update_metrics = agent.update(memory, train_env, current_ep=episode)
        else:
            update_metrics = {"Loss/Total": 0.0}
        memory.clear()

        row: dict[str, Any] = {
            "episode": episode,
            "dataset_idx": dataset_idx,
            **train_metrics,
            "loss_total": float(update_metrics.get("Loss/Total", 0.0)),
            "oom_skipped": float(update_metrics.get("OOM/SkippedUpdate", 0.0)),
        }

        if episode % int(getattr(configs, "eval_freq", 1)) == 0:
            eval_metrics, eval_schedule, _eval_runs = evaluate_graph_policy(
                agent.policy,
                eval_env,
                device,
                seed=seed,
                num_runs=1,
                temperature=float(getattr(configs, "eval_temperature", 0.0)),
            )
            row.update(
                {
                    "eval_makespan": float(eval_metrics["makespan"]),
                    "eval_valid": float(eval_metrics["valid"]),
                    "eval_complete": float(eval_metrics["complete"]),
                    "eval_inference_time": float(eval_metrics["inference_time"]),
                }
            )
            if eval_metrics["valid"] >= 1.0 and float(eval_metrics["makespan"]) < best_makespan:
                best_makespan = float(eval_metrics["makespan"])
                best_eval_schedule = list(eval_schedule)
                _save_checkpoint(best_path, agent, best_makespan, args, episode=episode)
                export_best_schedule(output_dir, best_eval_schedule, title="simple_heterogat_ppo_best")
                print(
                    f"[{METHOD_NAME}][Checkpoint] ep={episode} 保存最优模型 "
                    f"Mk={best_makespan:.2f} path={best_path}",
                    flush=True,
                )

        rows.append(row)
        print(
            f"[{METHOD_NAME}][Train] ep={episode}/{max_episodes} "
            f"loss={row['loss_total']:.6f} eval_mk={row.get('eval_makespan', np.nan):.2f} "
            f"valid={row.get('eval_valid', np.nan):.0f} best={best_makespan:.2f}",
            flush=True,
        )
        _save_checkpoint(latest_path, agent, best_makespan, args, episode=episode)
        if episode == 1 or episode % 10 == 0:
            print(f"[{METHOD_NAME}][Checkpoint] ep={episode} 保存最新模型 path={latest_path}", flush=True)
        if episode % 50 == 0:
            write_training_metrics(output_dir, rows)
            clear_torch_cache()

    _save_checkpoint(final_path, agent, best_makespan, args, episode=max_episodes)
    write_training_metrics(output_dir, rows)
    print(f"[{METHOD_NAME}] done elapsed={time.time() - start_time:.1f}s best={best_makespan:.2f}", flush=True)
    clear_torch_cache()


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if should_show_help(raw_args):
        print(hydra_help(EXTRA_ARGS))
        return 0
    try:
        args = initialize_hydra_runtime(
            raw_args,
            target=configs,
            project_root=PROJECT_ROOT,
            default_experiment="initial_schedule_283",
            extra_arguments=EXTRA_ARGS,
        )
        train(args)
    except (HydraCliError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"[CLI] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
