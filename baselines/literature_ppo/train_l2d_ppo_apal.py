from __future__ import annotations

import os
import platform
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
    evaluate_graph_policy,
    export_best_schedule,
    is_r5_learning_protocol,
    load_training_metrics,
    make_eval_env,
    LiteratureCheckpointSaver,
    prepare_literature_output,
    save_literature_checkpoint,
    training_data_source,
    write_training_metrics,
)
from configs import configs
from ppo_agent import PPOAgent
from runtime.hydra_config import ExtraArgument, HydraCliError, hydra_help, initialize_hydra_runtime, should_show_help
from runtime.seed import set_seed
from training.async_evaluation import AsyncEvaluationManager
from training.rollout_service import APALRolloutService
from utils.device_utils import clear_torch_cache, get_available_device
from utils.vector_env import EnvCreator, VectorEnv


METHOD_NAME = "Simple-HeteroGAT-PPO"
ENTRYPOINT = "baselines/literature_ppo/train_l2d_ppo_apal.py"
EXTRA_ARGS = {
    "output_dir": ExtraArgument(default=None, help="可选输出目录；缺省写入 runs artifacts"),
}


class SimpleHeteroGATPPO(GraphBaselineActorCritic):
    """历史联合动作异构图 PPO；不再将其错误标记为 L2D。"""


def _save_checkpoint(
    path: Path,
    agent: PPOAgent,
    best_makespan: float,
    args: Any,
    *,
    episode: int,
    dataset_selector_state: dict[str, Any] | None = None,
) -> None:
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
            "dataset_selector_state": dataset_selector_state,
        },
    )


def _load_resume_checkpoint(
    path: Path,
    agent: PPOAgent,
) -> tuple[int, float, dict[str, Any] | None]:
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
    return (
        episode + 1,
        float(best_makespan) if best_makespan is not None else float("inf"),
        checkpoint.get("dataset_selector_state"),
    )


def _build_training_vector_env(args: Any, *, seed: int) -> VectorEnv:
    """按主方法硬件配置创建 L2D rollout 环境；关闭向量开关时保留单 worker。"""
    requested_envs = (
        int(getattr(configs, "num_envs", 1))
        if bool(getattr(configs, "literature_ppo_enable_vector_env", True))
        else 1
    )
    requested_envs = max(1, requested_envs)
    start_method = str(getattr(configs, "vector_env_start_method", "auto"))
    if start_method == "auto":
        start_method = "forkserver" if platform.system() == "Linux" else "spawn"
    return VectorEnv(
        EnvCreator(str(training_data_source(args)), seed_offset=int(seed)),
        num_envs=requested_envs,
        start_method=start_method,
        worker_threads=getattr(configs, "vector_env_worker_threads", "auto"),
        init_timeout_sec=float(getattr(configs, "vector_env_init_timeout_sec", 120.0)),
        command_timeout_sec=float(getattr(configs, "vector_env_command_timeout_sec", 120.0)),
    )


def _service_dataset_state(service: APALRolloutService) -> dict[str, Any]:
    """在 L2D 自身 checkpoint 中保存 service 的数据集选择状态。"""
    state = service._rng.get_state()  # noqa: SLF001 - 仅限 L2D resume 兼容层
    return {
        "rng_name": str(state[0]),
        "rng_keys": [int(value) for value in state[1].tolist()],
        "rng_pos": int(state[2]),
        "rng_has_gauss": int(state[3]),
        "rng_cached_gaussian": float(state[4]),
        "last_dataset_idx": (
            None
            if service._last_dataset_idx is None  # noqa: SLF001
            else int(service._last_dataset_idx)  # noqa: SLF001
        ),
    }


def _restore_service_dataset_state(
    service: APALRolloutService,
    state: dict[str, Any] | None,
) -> None:
    if not state:
        return
    service._rng.set_state(  # noqa: SLF001 - 仅限 L2D resume 兼容层
        (
            str(state.get("rng_name", "MT19937")),
            np.asarray(state["rng_keys"], dtype=np.uint32),
            int(state["rng_pos"]),
            int(state.get("rng_has_gauss", 0)),
            float(state.get("rng_cached_gaussian", 0.0)),
        )
    )
    raw_dataset_idx = state.get("last_dataset_idx")
    service._last_dataset_idx = (  # noqa: SLF001
        None if raw_dataset_idx is None else int(raw_dataset_idx)
    )


def train(args: Any) -> None:
    seed = int(getattr(configs, "seed", 42))
    set_seed(seed)
    output_dir = prepare_literature_output(args, method_name=METHOD_NAME, entrypoint=ENTRYPOINT)
    start_time = time.time()
    device = get_available_device()

    r5_protocol = is_r5_learning_protocol(configs)
    if r5_protocol and device.type != "cuda":
        raise RuntimeError("r5 literature PPO 训练期异步验证必须使用 CUDA")

    train_vector_env = _build_training_vector_env(args, seed=seed)
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
    service = APALRolloutService(
        agent=agent,
        vector_env=train_vector_env,
        eval_env=eval_env,
        config=configs,
        device=device,
    )

    async_manager = (
        AsyncEvaluationManager(config=configs, latest_path=latest_path, best_path=best_path, project_root=PROJECT_ROOT)
        if r5_protocol
        else None
    )

    if bool(getattr(args, "resume", False)):
        if not latest_path.exists():
            raise FileNotFoundError(f"找不到可恢复的 {METHOD_NAME} checkpoint: {latest_path}")
        start_episode, best_makespan, selector_state = _load_resume_checkpoint(latest_path, agent)
        if selector_state:
            _restore_service_dataset_state(service, selector_state)
        rows = load_training_metrics(output_dir, before_episode=start_episode)
        print(
            f"[{METHOD_NAME}] resume checkpoint={latest_path} "
            f"start_episode={start_episode} best={best_makespan:.2f}",
            flush=True,
        )

    print(
        f"[{METHOD_NAME}] start episodes={max_episodes} batch_size={batch_size} "
        f"train_datasets={train_vector_env.envs[0].dataset_count} "
        f"vector_env={train_vector_env.num_envs > 1} num_envs={train_vector_env.num_envs} "
        f"episode_semantics=one_ppo_update_with_{train_vector_env.num_envs}_trajectories "
        f"eval_data={getattr(configs, 'data_file_path', '')}",
        flush=True,
    )

    for episode in range(start_episode, max_episodes + 1):
        rollout_update = service.collect(episode)
        rollout_metrics = rollout_update.rollout_metrics
        dataset_idx = int(service._last_dataset_idx or 0)  # noqa: SLF001
        trajectory_count = int(service.num_envs * service.episodes_per_update)
        vector_num_envs = int(service.num_envs)
        rollout_mode = "vector" if vector_num_envs > 1 else "serial"
        average_reward = float(np.mean([item.average_reward for item in rollout_metrics]))
        average_makespan = float(np.mean([item.average_makespan for item in rollout_metrics]))
        completion_rate = float(np.mean([item.completion_rate for item in rollout_metrics]))
        environment_steps = int(sum(item.environment_steps for item in rollout_metrics))
        duration_sec = float(sum(item.total_seconds for item in rollout_metrics))
        print(
            f"[{METHOD_NAME}][Rollout] ep={episode}/{max_episodes} "
            f"ds={dataset_idx} trajectories={trajectory_count} "
            f"R={average_reward:.3f} Mk={average_makespan:.2f} "
            f"Done={completion_rate * 100:.1f}% steps={environment_steps} "
            f"T={duration_sec:.1f}s",
            flush=True,
        )
        if rollout_update.memory.rewards:
            update_metrics = agent.update(
                rollout_update.memory,
                rollout_update.env,
                current_ep=episode,
            )
        else:
            update_metrics = {"Loss/Total": 0.0}
        rollout_update.memory.clear()

        row: dict[str, Any] = {
            "episode": episode,
            "dataset_idx": dataset_idx,
            "trajectory_count": trajectory_count,
            "vector_num_envs": vector_num_envs,
            "rollout_mode": rollout_mode,
            "reward": average_reward,
            "makespan": average_makespan,
            "complete": completion_rate,
            "environment_steps": environment_steps,
            "duration_sec": duration_sec,
            "loss_total": float(update_metrics.get("Loss/Total", 0.0)),
            "oom_skipped": float(update_metrics.get("OOM/SkippedUpdate", 0.0)),
        }

        if episode % int(getattr(configs, "eval_freq", 1)) == 0:
            if r5_protocol:
                if episode % int(getattr(configs, "async_eval_submit_every_episodes", 2)) == 0:
                    assert async_manager is not None
                    async_manager.submit(
                        LiteratureCheckpointSaver(
                            lambda path: _save_checkpoint(path, agent, best_makespan, args, episode=episode, dataset_selector_state=_service_dataset_state(service))
                        ),
                        episode=episode,
                    )
            else:
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
                    _save_checkpoint(
                        best_path,
                        agent,
                        best_makespan,
                        args,
                        episode=episode,
                        dataset_selector_state=_service_dataset_state(service),
                    )
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
        _save_checkpoint(
            latest_path,
            agent,
            best_makespan,
            args,
            episode=episode,
            dataset_selector_state=_service_dataset_state(service),
        )
        write_training_metrics(output_dir, rows)
        if episode == 1 or episode % 10 == 0:
            print(f"[{METHOD_NAME}][Checkpoint] ep={episode} 保存最新模型 path={latest_path}", flush=True)
        if episode % 50 == 0:
            clear_torch_cache()

    _save_checkpoint(
        final_path,
        agent,
        best_makespan,
        args,
        episode=max_episodes,
        dataset_selector_state=_service_dataset_state(service),
    )
    if async_manager is not None:
        async_manager.finalize(wait=True)
    write_training_metrics(output_dir, rows)
    print(f"[{METHOD_NAME}] done elapsed={time.time() - start_time:.1f}s best={best_makespan:.2f}", flush=True)
    clear_torch_cache()
    service.close()


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
