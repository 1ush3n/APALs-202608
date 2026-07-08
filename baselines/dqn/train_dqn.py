from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch, HeteroData

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from baselines.graph_baseline import (
    GRAPH_BASELINE_FEATURE_MODE,
    GraphBaselineActorCritic,
    select_graph_action,
    task_demand_from_obs,
    worker_static_mask_from_obs,
)
from configs import configs
from env_wrapper import init_env, standardize_env_step
from runtime.artifacts import resolve_run_output_dir, write_run_context_files, write_run_manifest
from runtime.hydra_config import ExtraArgument, HydraCliError, hydra_help, initialize_hydra_runtime, should_show_help
from runtime.seed import set_seed
from training.observation import refresh_env_observation
from utils.device_utils import clear_torch_cache, get_available_device
from utils.logger import init_logger, record_experiment_time
from utils.visualization import plot_gantt


BASELINE_NAME = "dqn_baseline"
BASELINE_EXTRA_ARGS = {
    "output_dir": ExtraArgument(default=None, help="可选输出目录；缺省写入本次 run 的 artifacts/baselines 目录"),
}


class DQN(GraphBaselineActorCritic):
    """图观测版 factorized DQN，保留类名用于 checkpoint 与评估入口兼容。"""


def _prepare_output_root(args: Any) -> None:
    output_root, context = resolve_run_output_dir(
        configs,
        PROJECT_ROOT,
        default_legacy_dir=getattr(configs, "result_dir", "results"),
        run_subdir=Path("baselines") / "graph_training" / "DQN",
        explicit_dir=getattr(args, "output_dir", None),
        section="artifacts",
    )
    setattr(args, "output_dir", str(output_root))
    extra = {"baseline": "DQN", "entrypoint": "baselines/dqn/train_dqn.py", "feature_mode": GRAPH_BASELINE_FEATURE_MODE}
    if context is not None:
        write_run_context_files(context, configs, command="dqn_train", extra=extra)
    else:
        write_run_manifest(output_root, configs, command="dqn_train", extra=extra)


def _save_dqn_checkpoint(path: Path, agent: "GraphDQNAgent", best_makespan: float, exp_dir: Path) -> None:
    path = Path(path)
    payload = {
        "algorithm": "DQN",
        "model_type": "GraphDQN",
        "feature_mode": GRAPH_BASELINE_FEATURE_MODE,
        "model_state_dict": agent.model.state_dict(),
        "target_model_state_dict": agent.target_model.state_dict(),
        "seed": int(getattr(configs, "seed", 42)),
        "data_file_path": str(getattr(configs, "data_file_path", "")),
        "config_paths": list(getattr(configs, "config_paths", ())),
        "use_skill_hub": bool(getattr(configs, "use_skill_hub", False)),
        "skill_hub_bidirectional": bool(getattr(configs, "skill_hub_bidirectional", False)),
        "hidden_dim": int(getattr(configs, "hidden_dim", 128)),
        "num_gat_layers": int(getattr(configs, "num_gat_layers", 1)),
        "num_heads": int(getattr(configs, "num_heads", 1)),
        "best_makespan": float(best_makespan) if np.isfinite(best_makespan) else None,
        "epsilon": float(agent.epsilon),
    }
    torch.save(payload, path)
    metadata = {k: v for k, v in payload.items() if not k.endswith("state_dict")}
    with open(Path(exp_dir) / f"{path.stem}_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


class GraphDQNAgent:
    def __init__(self, args: Any, device: torch.device) -> None:
        self.device = device
        self.gamma = float(getattr(args, "gamma", None) or getattr(configs, "gamma", 0.99))
        self.epsilon = float(getattr(args, "epsilon", None) or 1.0)
        self.epsilon_min = float(getattr(args, "epsilon_min", None) or 0.01)
        self.epsilon_decay = float(getattr(args, "epsilon_decay", None) or 0.995)
        self.model = DQN(configs).to(device)
        self.target_model = DQN(configs).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        lr = float(getattr(args, "lr", None) or getattr(configs, "lr", 3e-4))
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = deque(maxlen=int(getattr(args, "memory_size", None) or 10000))
        self.amp_enabled = device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.amp_enabled)

    def _autocast(self):
        return torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled)

    def _random_action(self, obs: HeteroData, masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[int, int, list[int]] | None:
        task_mask, station_mask_matrix, worker_mask = masks
        valid_tasks = torch.where(~task_mask.bool())[0].cpu().numpy()
        if len(valid_tasks) == 0:
            return None
        task_idx = int(np.random.choice(valid_tasks))
        valid_stations = torch.where(~station_mask_matrix[task_idx].bool())[0].cpu().numpy()
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
        team = random.sample(valid_workers, demand)
        return task_idx, station_idx, team

    def select_action(self, obs: HeteroData, masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor], *, deterministic: bool = False) -> tuple[int, int, list[int]] | None:
        if (not deterministic) and np.random.rand() <= self.epsilon:
            return self._random_action(obs, masks)
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

    def _q_for_action(self, model: DQN, obs: HeteroData, action: tuple[int, int, list[int]], masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        batch_obs = Batch.from_data_list([obs]).to(self.device)
        with self._autocast():
            x_dict, context = model(batch_obs)
            task_idx, station_idx, team = action
            task_mask = masks[0].to(self.device).bool().unsqueeze(0) if masks is not None else None
            task_scores = model.task_head(x_dict["task"], context, mask=task_mask)
            task_q = task_scores[0, int(task_idx)]
            station_mask = masks[1][int(task_idx)].to(self.device).bool().unsqueeze(0) if masks is not None else None
            selected_task_emb = x_dict["task"][int(task_idx)].unsqueeze(0)
            station_scores = model.station_head(selected_task_emb, x_dict["station"].unsqueeze(0), mask=station_mask)
            station_q = station_scores[0, int(station_idx)]

            worker_mask = (
                worker_static_mask_from_obs(obs, task_idx=int(task_idx), station_idx=int(station_idx), worker_mask=masks[2], device=self.device).unsqueeze(0)
                if masks is not None
                else torch.zeros(1, x_dict["worker"].size(0), dtype=torch.bool, device=self.device)
            )
            worker_embs = x_dict["worker"].unsqueeze(0)
            current_team_emb = None
            worker_q_values: list[torch.Tensor] = []
            for worker_idx in team:
                worker_scores = model.worker_head.forward_choice(selected_task_emb, worker_embs, mask=worker_mask, current_team_emb=current_team_emb)
                worker_q_values.append(worker_scores[0, int(worker_idx)])
                worker_mask = worker_mask.clone()
                worker_mask[0, int(worker_idx)] = True
                selected = worker_embs[0, [int(w) for w in team[: len(worker_q_values)]], :]
                current_team_emb = selected.mean(dim=0, keepdim=True)
            worker_q = torch.stack(worker_q_values).mean() if worker_q_values else torch.tensor(0.0, device=self.device)
            return (task_q + station_q + worker_q) / 3.0

    def _best_q(self, model: DQN, obs: HeteroData, masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        action = select_graph_action(model, obs, masks=masks, device=self.device, deterministic=True, temperature=0.0).action
        if action is None:
            return torch.tensor(0.0, device=self.device)
        return self._q_for_action(model, obs, action, masks)

    def remember(self, state_snapshot: dict[str, Any], action: tuple[int, int, list[int]], reward: float, next_snapshot: dict[str, Any], done: bool, masks: tuple[Any, Any, Any], next_masks: tuple[Any, Any, Any]) -> None:
        self.memory.append((state_snapshot, action, float(reward), next_snapshot, bool(done), masks, next_masks))

    def replay(self, env: Any, batch_size: int) -> float:
        if len(self.memory) < int(batch_size):
            return 0.0
        samples = random.sample(self.memory, int(batch_size))
        current_qs: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for state_snapshot, action, reward, next_snapshot, done, masks, next_masks in samples:
            state = env.rebuild_state_from_snapshot(state_snapshot)
            current_qs.append(self._q_for_action(self.model, state, action, masks))
            with torch.no_grad():
                if done:
                    next_q = torch.tensor(0.0, device=self.device)
                else:
                    next_state = env.rebuild_state_from_snapshot(next_snapshot)
                    next_q = self._best_q(self.target_model, next_state, next_masks)
                targets.append(torch.tensor(float(reward), device=self.device) + self.gamma * next_q * (0.0 if done else 1.0))

        q_current = torch.stack(current_qs)
        q_target = torch.stack(targets).detach()
        loss = self.loss_fn(q_current.float(), q_target.float())
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(loss.detach().cpu().item())


def train_dqn(args: Any) -> None:
    set_seed(int(getattr(configs, "seed", 42)))
    _prepare_output_root(args)
    logger, exp_dir_raw = init_logger(args, BASELINE_NAME)
    exp_dir = Path(exp_dir_raw)
    start_time = time.time()

    try:
        device = get_available_device()
        env = init_env(args, seed=getattr(args, "seed", None))
        agent = GraphDQNAgent(args, device)
        batch_size = int(getattr(args, "batch_size", None) or getattr(configs, "batch_size", 32))
        max_episodes = int(getattr(args, "max_episodes", None) or getattr(configs, "max_episodes", 300))

        episode_rewards: list[float] = []
        episode_losses: list[float] = []
        episode_makespans: list[float] = []
        best_makespan = float("inf")

        logger.info(
            "开始图观测 factorized DQN 训练，使用 HeteroData/mask/env.step reward，"
            f"max_episodes={max_episodes}, batch_size={batch_size}"
        )
        for ep in range(max_episodes):
            state = env.reset(randomize_duration=False, randomize_workers=False, seed=int(configs.seed) + ep)
            done = False
            ep_reward = 0.0
            ep_loss = 0.0
            step_count = 0
            max_steps = max(1, int(env.num_tasks) * 2)

            while not done and step_count < max_steps and len(env.assigned_tasks) < env.num_tasks:
                step_count += 1
                masks = env.get_masks()
                while bool(masks[0].all()):
                    if not env.try_wait_for_resources():
                        done = True
                        break
                    state = refresh_env_observation(env)
                    masks = env.get_masks()
                if done or bool(masks[0].all()):
                    break

                snapshot = env.get_state_snapshot()
                action = agent.select_action(state, masks, deterministic=False)
                if action is None:
                    done = True
                    ep_reward -= 100.0
                    break
                state, reward, done, info = standardize_env_step(env, action)
                if bool(info.get("invalid_action", False)):
                    done = True
                next_snapshot = env.get_state_snapshot()
                next_masks = env.get_masks()
                agent.remember(snapshot, action, reward, next_snapshot, done, masks, next_masks)
                ep_reward += float(reward)
                ep_loss += agent.replay(env, batch_size)

            makespan = float(np.max(env.station_wall_clock)) if len(env.assigned_tasks) == env.num_tasks else float(env.ideal_makespan * 3.0)
            episode_rewards.append(ep_reward)
            episode_losses.append(ep_loss / max(1, step_count))
            episode_makespans.append(makespan)

            if makespan < best_makespan and len(env.assigned_tasks) == env.num_tasks:
                best_makespan = makespan
                best_sch = env.assigned_tasks.copy()
                rows = [
                    {"TaskID": tid, "StationID": sid + 1, "Team": str(team), "Start": start, "End": end, "Duration": end - start}
                    for (tid, sid, team, start, end) in best_sch
                ]
                pd.DataFrame(rows).to_csv(exp_dir / "Best_Schedule_DQN.csv", index=False)
                plot_gantt(best_sch, str(exp_dir / "Best_Gantt_DQN.png"))
                _save_dqn_checkpoint(exp_dir / "dqn_model_best.pth", agent, best_makespan, exp_dir)
                logger.info(f"新的最佳图 DQN 调度已保存，Makespan={best_makespan:.2f}")

            if (ep + 1) % 10 == 0:
                logger.info(
                    f"Episode {ep+1:04d}/{max_episodes} | "
                    f"AvgReward={np.mean(episode_rewards[-10:]):.3f} | "
                    f"AvgLoss={np.mean(episode_losses[-10:]):.6f} | "
                    f"AvgMakespan={np.mean(episode_makespans[-10:]):.2f} | "
                    f"Epsilon={agent.epsilon:.4f}"
                )
            if (ep + 1) % 50 == 0:
                agent.target_model.load_state_dict(agent.model.state_dict())
                clear_torch_cache()

        _save_dqn_checkpoint(exp_dir / "dqn_model.pth", agent, best_makespan, exp_dir)
        results = pd.DataFrame({"episode": range(1, max_episodes + 1), "reward": episode_rewards, "loss": episode_losses, "makespan": episode_makespans})
        results["avg_reward_10"] = results["reward"].rolling(window=10).mean()
        results["avg_makespan_10"] = results["makespan"].rolling(window=10).mean()
        results.to_csv(exp_dir / "dqn_results.csv", index=False)
    except Exception as exc:
        logger.error(f"图 DQN 训练失败: {exc}", exc_info=True)
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
        train_dqn(args)
    except (HydraCliError, KeyError, ValueError, RuntimeError) as exc:
        print(f"[CLI] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
