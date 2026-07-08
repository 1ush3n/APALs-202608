from __future__ import annotations

import os
import random
import sys
import time
from collections import deque
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
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch, HeteroData

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from baselines.graph_baseline import (
    GraphBaselineActorCritic,
    select_graph_action,
    task_demand_from_obs,
    worker_static_mask_from_obs,
)
from baselines.literature.common import (
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
from env_wrapper import standardize_env_step
from runtime.hydra_config import ExtraArgument, HydraCliError, hydra_help, initialize_hydra_runtime, should_show_help
from runtime.seed import set_seed
from training.observation import refresh_env_observation
from utils.device_utils import clear_torch_cache, get_available_device


METHOD_NAME = "Graph-DDQN-APAL"
ENTRYPOINT = "baselines/literature_dqn/train_graph_ddqn_apal.py"
EXTRA_ARGS = {
    "output_dir": ExtraArgument(default=None, help="可选输出目录；缺省写入 runs artifacts"),
    "epsilon": ExtraArgument(default=1.0, help="epsilon-greedy 初始探索率"),
    "epsilon_min": ExtraArgument(default=0.05, help="epsilon-greedy 最小探索率"),
    "epsilon_decay": ExtraArgument(default=0.995, help="每次 replay 后 epsilon 衰减"),
    "memory_size": ExtraArgument(default=20000, help="CPU snapshot replay buffer 容量"),
    "replay_start_size": ExtraArgument(default=256, help="开始 replay 前的最小样本数"),
    "target_update_episodes": ExtraArgument(default=20, help="target network 同步周期"),
}


class GraphDDQNAPAL(GraphBaselineActorCritic):
    """参考图状态 value-based 调度方法的 APAL Double-DQN baseline。"""


class GraphDDQNAgent:
    def __init__(self, args: Any, device: torch.device) -> None:
        self.device = device
        self.gamma = float(getattr(configs, "gamma", 0.99))
        self.epsilon = float(getattr(args, "epsilon", 1.0))
        self.epsilon_min = float(getattr(args, "epsilon_min", 0.05))
        self.epsilon_decay = float(getattr(args, "epsilon_decay", 0.995))
        self.replay_start_size = int(getattr(args, "replay_start_size", 256))
        self.model = GraphDDQNAPAL(configs).to(device)
        self.target_model = GraphDDQNAPAL(configs).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(getattr(configs, "lr", 3e-4)),
            weight_decay=1e-4,
        )
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = deque(maxlen=int(getattr(args, "memory_size", 20000)))
        self.amp_enabled = device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.amp_enabled)
        self.oom_skipped_updates = 0

    def _autocast(self):
        return torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled)

    def random_action(self, obs: HeteroData, masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[int, int, list[int]] | None:
        task_mask, station_mask_matrix, worker_mask = masks
        valid_tasks = torch.where(~task_mask.bool())[0].detach().cpu().numpy()
        if len(valid_tasks) == 0:
            return None
        task_idx = int(np.random.choice(valid_tasks))
        valid_stations = torch.where(~station_mask_matrix[task_idx].bool())[0].detach().cpu().numpy()
        if len(valid_stations) == 0:
            return None
        station_idx = int(np.random.choice(valid_stations))
        worker_mask_final = worker_static_mask_from_obs(
            obs,
            task_idx=task_idx,
            station_idx=station_idx,
            worker_mask=worker_mask,
            device=self.device,
        )
        valid_workers = torch.where(~worker_mask_final)[0].detach().cpu().numpy().tolist()
        demand = task_demand_from_obs(obs, task_idx)
        if len(valid_workers) < demand:
            return None
        return task_idx, station_idx, random.sample(valid_workers, demand)

    def select_action(self, obs: HeteroData, masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor], *, deterministic: bool = False) -> tuple[int, int, list[int]] | None:
        if (not deterministic) and np.random.rand() <= self.epsilon:
            return self.random_action(obs, masks)
        with torch.inference_mode():
            result = select_graph_action(
                self.model,
                obs,
                masks=masks,
                device=self.device,
                deterministic=True,
                temperature=0.0,
                need_value=False,
            )
        return result.action

    def remember(
        self,
        state_snapshot: dict[str, Any],
        action: tuple[int, int, list[int]],
        reward: float,
        next_snapshot: dict[str, Any],
        done: bool,
        masks: tuple[Any, Any, Any],
        next_masks: tuple[Any, Any, Any],
    ) -> None:
        self.memory.append((state_snapshot, action, float(reward), next_snapshot, bool(done), masks, next_masks))

    def _q_for_actions_batched(
        self,
        model: GraphDDQNAPAL,
        states: list[HeteroData],
        actions: list[tuple[int, int, list[int]]],
        masks_list: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        batch_obs = Batch.from_data_list(states)
        task_ptr = batch_obs["task"].ptr.tolist()
        station_ptr = batch_obs["station"].ptr.tolist()
        worker_ptr = batch_obs["worker"].ptr.tolist()
        batch_obs = batch_obs.to(self.device)
        with self._autocast():
            x_dict, context = model(batch_obs)
            q_values: list[torch.Tensor] = []
            for idx, action in enumerate(actions):
                task_idx, station_idx, team = action
                task_start, task_end = task_ptr[idx], task_ptr[idx + 1]
                station_start, station_end = station_ptr[idx], station_ptr[idx + 1]
                worker_start, worker_end = worker_ptr[idx], worker_ptr[idx + 1]
                task_embs = x_dict["task"][task_start:task_end]
                station_embs = x_dict["station"][station_start:station_end]
                worker_embs = x_dict["worker"][worker_start:worker_end]
                context_i = context[idx].unsqueeze(0)
                task_mask, station_mask_matrix, worker_mask = masks_list[idx]

                task_scores = model.task_head(
                    task_embs,
                    context_i,
                    mask=task_mask.to(self.device, dtype=torch.bool),
                )
                task_q = task_scores.view(-1)[int(task_idx)]

                selected_task_emb = task_embs[int(task_idx)].unsqueeze(0)
                station_scores = model.station_head(
                    selected_task_emb,
                    station_embs.unsqueeze(0),
                    mask=station_mask_matrix[int(task_idx)].to(self.device, dtype=torch.bool).unsqueeze(0),
                )
                station_q = station_scores.view(-1)[int(station_idx)]

                current_worker_mask = worker_static_mask_from_obs(
                    states[idx],
                    task_idx=int(task_idx),
                    station_idx=int(station_idx),
                    worker_mask=worker_mask,
                    device=self.device,
                ).unsqueeze(0)
                worker_embs_batched = worker_embs.unsqueeze(0)
                worker_qs: list[torch.Tensor] = []
                current_team_emb = None
                for worker_idx in team:
                    worker_scores = model.worker_head.forward_choice(
                        selected_task_emb,
                        worker_embs_batched,
                        mask=current_worker_mask,
                        current_team_emb=current_team_emb,
                    )
                    worker_qs.append(worker_scores.view(-1)[int(worker_idx)])
                    current_worker_mask = current_worker_mask.clone()
                    current_worker_mask[0, int(worker_idx)] = True
                    selected = worker_embs_batched[0, [int(w) for w in team[: len(worker_qs)]], :]
                    current_team_emb = selected.mean(dim=0, keepdim=True)
                worker_q = torch.stack(worker_qs).mean() if worker_qs else torch.tensor(0.0, device=self.device)
                q_values.append((task_q + station_q + worker_q) / 3.0)
        return torch.stack(q_values)

    def replay(self, env: Any, batch_size: int) -> float:
        try:
            if len(self.memory) < max(int(batch_size), self.replay_start_size):
                return 0.0
            active_dataset_idx = int(getattr(env, "active_dataset_idx", 0))
            candidates = [
                item
                for item in self.memory
                if int(item[0].get("dataset_idx", 0)) == active_dataset_idx
            ]
            if len(candidates) < max(int(batch_size), self.replay_start_size):
                return 0.0
            samples = random.sample(candidates, int(batch_size))
            states = [env.rebuild_state_from_snapshot(item[0]) for item in samples]
            actions = [item[1] for item in samples]
            rewards = torch.tensor([float(item[2]) for item in samples], dtype=torch.float32, device=self.device)
            dones = torch.tensor([float(item[4]) for item in samples], dtype=torch.float32, device=self.device)
            masks = [item[5] for item in samples]
            next_states = [env.rebuild_state_from_snapshot(item[3]) for item in samples]
            next_masks = [item[6] for item in samples]

            current_q = self._q_for_actions_batched(self.model, states, actions, masks)
            next_actions: list[tuple[int, int, list[int]] | None] = []
            with torch.inference_mode():
                for next_state, mask in zip(next_states, next_masks):
                    result = select_graph_action(
                        self.model,
                        next_state,
                        masks=mask,
                        device=self.device,
                        deterministic=True,
                        temperature=0.0,
                        need_value=False,
                    )
                    next_actions.append(result.action)

            target_q = torch.zeros_like(current_q)
            valid_indices = [idx for idx, action in enumerate(next_actions) if action is not None]
            if valid_indices:
                valid_states = [next_states[idx] for idx in valid_indices]
                valid_actions = [next_actions[idx] for idx in valid_indices]
                valid_masks = [next_masks[idx] for idx in valid_indices]
                assert all(action is not None for action in valid_actions)
                with torch.no_grad():
                    target_q_valid = self._q_for_actions_batched(
                        self.target_model,
                        valid_states,
                        [action for action in valid_actions if action is not None],
                        valid_masks,
                    )
                target_q[torch.tensor(valid_indices, dtype=torch.long, device=self.device)] = target_q_valid

            q_target = rewards + self.gamma * target_q.detach() * (1.0 - dones)
            loss = self.loss_fn(current_q.float(), q_target.float())
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.epsilon > self.epsilon_min:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            return float(loss.detach().cpu().item())
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or self.device.type != "cuda":
                raise
            self.oom_skipped_updates += 1
            self.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if self.oom_skipped_updates <= 3 or self.oom_skipped_updates % 20 == 0:
                print(
                    f"WARNING: {METHOD_NAME} replay CUDA OOM，已跳过本次 replay update；"
                    f"batch_size={batch_size}, skipped={self.oom_skipped_updates}",
                    flush=True,
                )
            return 0.0


def _save_checkpoint(path: Path, agent: GraphDDQNAgent, best_makespan: float, args: Any, *, episode: int) -> None:
    save_literature_checkpoint(
        path,
        algorithm=METHOD_NAME,
        literature_family="graph_double_dqn",
        model=agent.model,
        best_makespan=best_makespan,
        args=args,
        extra={
            "model_type": "GraphDDQNAPAL",
            "target_model_state_dict": agent.target_model.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "scaler_state_dict": agent.scaler.state_dict(),
            "epsilon": float(agent.epsilon),
            "oom_skipped_updates": int(agent.oom_skipped_updates),
            "episode": int(episode),
            "replay_buffer_restored": False,
            "optimizer": "DoubleDQN",
        },
    )


def _load_resume_checkpoint(path: Path, agent: GraphDDQNAgent) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=agent.device, weights_only=False)
    if checkpoint.get("algorithm") != METHOD_NAME:
        raise ValueError(f"checkpoint algorithm={checkpoint.get('algorithm')!r}，不是 {METHOD_NAME}")
    agent.model.load_state_dict(checkpoint["model_state_dict"])
    if "target_model_state_dict" in checkpoint:
        agent.target_model.load_state_dict(checkpoint["target_model_state_dict"])
    else:
        agent.target_model.load_state_dict(agent.model.state_dict())
    if "optimizer_state_dict" in checkpoint:
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scaler_state_dict" in checkpoint:
        agent.scaler.load_state_dict(checkpoint["scaler_state_dict"])
    agent.epsilon = float(checkpoint.get("epsilon", agent.epsilon))
    agent.oom_skipped_updates = int(checkpoint.get("oom_skipped_updates", 0))
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
    agent = GraphDDQNAgent(args, device)
    batch_size = int(getattr(configs, "batch_size", 64))
    max_episodes = int(getattr(configs, "max_episodes", 300))
    target_update_episodes = int(getattr(args, "target_update_episodes", 20))

    rows: list[dict[str, Any]] = []
    latest_path = output_dir / "graph_ddqn_apal_latest.pth"
    best_path = output_dir / "graph_ddqn_apal_best.pth"
    final_path = output_dir / "graph_ddqn_apal_final.pth"
    start_episode = 1
    best_makespan = float("inf")

    if bool(getattr(args, "resume", False)):
        if not latest_path.exists():
            raise FileNotFoundError(f"找不到可恢复的 {METHOD_NAME} checkpoint: {latest_path}")
        start_episode, best_makespan = _load_resume_checkpoint(latest_path, agent)
        rows = load_training_metrics(output_dir, before_episode=start_episode)
        print(
            f"[{METHOD_NAME}] resume checkpoint={latest_path} "
            f"start_episode={start_episode} best={best_makespan:.2f} "
            "replay_buffer=重新预热",
            flush=True,
        )

    print(
        f"[{METHOD_NAME}] start episodes={max_episodes} batch_size={batch_size} "
        f"train_datasets={train_env.dataset_count} replay_start={agent.replay_start_size}",
        flush=True,
    )

    for episode in range(start_episode, max_episodes + 1):
        dataset_idx = select_episode_dataset(train_env, episode, seed)
        episode_seed = seed + episode
        state = train_env.reset(randomize_duration=False, randomize_workers=False, seed=episode_seed)
        done = False
        total_reward = 0.0
        total_loss = 0.0
        invalid_count = 0
        step_count = 0
        episode_start = time.time()

        while not done and step_count < max(1, int(train_env.num_tasks) * 2) and len(train_env.assigned_tasks) < train_env.num_tasks:
            step_count += 1
            masks = train_env.get_masks()
            while bool(masks[0].all()):
                if not train_env.try_wait_for_resources():
                    invalid_count += 1
                    done = True
                    break
                state = refresh_env_observation(train_env)
                masks = train_env.get_masks()
            if done or bool(masks[0].all()):
                break

            snapshot = train_env.get_state_snapshot()
            action = agent.select_action(state, masks, deterministic=False)
            if action is None:
                invalid_count += 1
                total_reward -= 100.0
                done = True
                break
            state, reward, done, info = standardize_env_step(train_env, action)
            if bool(info.get("invalid_action", False)):
                invalid_count += 1
                done = True
            next_snapshot = train_env.get_state_snapshot()
            next_masks = train_env.get_masks()
            agent.remember(snapshot, action, reward, next_snapshot, done, masks, next_masks)
            total_reward += float(reward)
            total_loss += agent.replay(train_env, batch_size)

        complete = len(train_env.assigned_tasks) == train_env.num_tasks
        makespan = float(np.max(train_env.station_wall_clock)) if complete else float(train_env.ideal_makespan * 3.0)
        print(
            f"[{METHOD_NAME}][Rollout] ep={episode}/{max_episodes} ds={dataset_idx} "
            f"R={total_reward:.3f} Mk={makespan:.2f} "
            f"Done={(1.0 if complete else 0.0) * 100:.1f}% "
            f"Assigned={len(train_env.assigned_tasks)}/{int(train_env.num_tasks)} "
            f"steps={step_count} replay={len(agent.memory)} T={time.time() - episode_start:.1f}s",
            flush=True,
        )
        row: dict[str, Any] = {
            "episode": episode,
            "dataset_idx": dataset_idx,
            "reward": float(total_reward),
            "loss": float(total_loss / max(1, step_count)),
            "makespan": float(makespan),
            "assigned": float(len(train_env.assigned_tasks)),
            "complete": 1.0 if complete else 0.0,
            "invalid_count": float(invalid_count),
            "epsilon": float(agent.epsilon),
            "oom_skipped_updates": float(agent.oom_skipped_updates),
            "duration_sec": float(time.time() - episode_start),
        }

        if episode % target_update_episodes == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())

        if episode % int(getattr(configs, "eval_freq", 1)) == 0:
            eval_metrics, eval_schedule, _eval_runs = evaluate_graph_policy(
                agent.model,
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
                _save_checkpoint(best_path, agent, best_makespan, args, episode=episode)
                export_best_schedule(output_dir, list(eval_schedule), title="graph_ddqn_apal_best")
                print(
                    f"[{METHOD_NAME}][Checkpoint] ep={episode} 保存最优模型 "
                    f"Mk={best_makespan:.2f} path={best_path}",
                    flush=True,
                )

        rows.append(row)
        print(
            f"[{METHOD_NAME}][Train] ep={episode}/{max_episodes} "
            f"loss={row['loss']:.6f} eval_mk={row.get('eval_makespan', np.nan):.2f} "
            f"valid={row.get('eval_valid', np.nan):.0f} eps={agent.epsilon:.3f} "
            f"oom_skip={agent.oom_skipped_updates} best={best_makespan:.2f}",
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
