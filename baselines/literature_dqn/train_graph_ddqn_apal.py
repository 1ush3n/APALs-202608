from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import random
import sys
import time
import traceback
from dataclasses import replace
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
from torch_geometric.data import HeteroData

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from baselines.graph_baseline import (
    EncodedGraphBatch,
    GraphBaselineActorCritic,
    decode_graph_actions_batch,
    encode_graph_batch,
    select_graph_action,
    select_graph_actions_batch,
    task_demand_from_obs,
    worker_static_mask_from_features,
    worker_static_mask_from_obs,
)
from baselines.literature_dqn.replay import DatasetReplayBuffer, DatasetUTDScheduler
from baselines.literature.common import (
    evaluate_graph_policy,
    export_best_schedule,
    is_r5_learning_protocol,
    load_training_metrics,
    LiteratureCheckpointSaver,
    make_eval_env,
    make_training_env,
    prepare_literature_output,
    rollout_step_limit,
    save_literature_checkpoint,
    select_episode_dataset,
    training_data_source,
    write_training_metrics,
)
from configs import configs
from env_wrapper import standardize_env_step
from runtime.hydra_config import ExtraArgument, HydraCliError, hydra_help, initialize_hydra_runtime, should_show_help
from runtime.seed import set_seed
from training.async_evaluation import AsyncEvaluationManager
from training.observation import refresh_env_observation
from utils.device_utils import clear_torch_cache, get_available_device
from utils.gpu_graph_manager import GPUBatchGraphManager
from utils.vector_env import EnvCreator, VectorEnv


METHOD_NAME = "Graph-DDQN-APAL"
ENTRYPOINT = "baselines/literature_dqn/train_graph_ddqn_apal.py"
EXTRA_ARGS = {
    "output_dir": ExtraArgument(default=None, help="可选输出目录；缺省写入 runs artifacts"),
    "epsilon": ExtraArgument(default=1.0, help="epsilon-greedy 初始探索率"),
    "epsilon_min": ExtraArgument(default=0.05, help="epsilon-greedy 最小探索率"),
    "epsilon_decay": ExtraArgument(default=0.995, help="每次 replay 后 epsilon 衰减"),
    "memory_size": ExtraArgument(default=20000, help="CPU snapshot replay buffer 容量"),
    "replay_start_size": ExtraArgument(default=256, help="开始 replay 前的最小样本数"),
    "replay_every_steps": ExtraArgument(default=None, help="已弃用；映射为 1/N 的 UTD"),
    "max_replay_updates_per_episode": ExtraArgument(default=0, help="可选安全上限；0 表示不限制"),
    "progress_interval_steps": ExtraArgument(default=50, help="episode 内进度打印间隔 step；0 表示关闭"),
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
        seed = int(getattr(configs, "seed", 42))
        self.action_np_rng = np.random.default_rng(seed + 1701)
        self.action_py_rng = random.Random(seed + 1703)
        # DDQN 保持 FP32 主权重，训练/推理由 autocast 临时使用 FP16；
        # 这样不会因服务器 checkpoint 或全局 AMP 设置把 Linear 权重永久留在 Half。
        self.model = GraphDDQNAPAL(configs).to(device).float()
        self.target_model = GraphDDQNAPAL(configs).to(device).float()
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.float()
        self.target_model.eval()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(getattr(configs, "lr", 3e-4)),
            weight_decay=1e-4,
        )
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = DatasetReplayBuffer(
            int(getattr(args, "memory_size", 20000)),
            seed=seed + 1709,
        )
        self.amp_enabled = device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.amp_enabled)
        self.oom_skipped_updates = 0
        self.enable_batched_replay = bool(getattr(args, "ddqn_enable_batched_replay", True))
        self.enable_profiler = bool(getattr(args, "ddqn_enable_profiler", True))
        self.profile_interval_updates = max(
            1,
            int(getattr(args, "ddqn_profile_interval_updates", 20)),
        )
        self.replay_attempts = 0
        self.successful_updates = 0
        self.last_replay_profile: dict[str, float] = {}
        self.gpu_batch_rebuild_enabled = bool(
            getattr(args, "ddqn_enable_gpu_batch_rebuild", False)
            and device.type == "cuda"
        )
        self.current_graph_manager = (
            GPUBatchGraphManager(device, config=configs)
            if self.gpu_batch_rebuild_enabled
            else None
        )
        self.next_graph_manager = (
            GPUBatchGraphManager(device, config=configs)
            if self.gpu_batch_rebuild_enabled
            else None
        )

    def _autocast(self):
        return torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled)

    def random_action(self, obs: HeteroData, masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[int, int, list[int]] | None:
        task_mask, station_mask_matrix, worker_mask = masks
        valid_tasks = torch.where(~task_mask.bool())[0].detach().cpu().numpy()
        if len(valid_tasks) == 0:
            return None
        task_idx = int(self.action_np_rng.choice(valid_tasks))
        valid_stations = torch.where(~station_mask_matrix[task_idx].bool())[0].detach().cpu().numpy()
        if len(valid_stations) == 0:
            return None
        station_idx = int(self.action_np_rng.choice(valid_stations))
        worker_mask_final = worker_static_mask_from_obs(
            obs,
            task_idx=task_idx,
            station_idx=station_idx,
            worker_mask=worker_mask,
            device=torch.device("cpu"),
        )
        valid_workers = torch.where(~worker_mask_final)[0].detach().cpu().numpy().tolist()
        demand = task_demand_from_obs(obs, task_idx)
        if len(valid_workers) < demand:
            return None
        return task_idx, station_idx, self.action_py_rng.sample(valid_workers, demand)

    def select_action(self, obs: HeteroData, masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor], *, deterministic: bool = False) -> tuple[int, int, list[int]] | None:
        if (not deterministic) and self.action_np_rng.random() <= self.epsilon:
            return self.random_action(obs, masks)
        # 动作选择也必须与 replay/update 使用同一 AMP 上下文；否则服务器
        # 半精度模型权重会接收到 float32 图特征，在线性层处触发 Half/Float 错误。
        with torch.inference_mode(), self._autocast():
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

    def select_actions_batch(
        self,
        observations: list[HeteroData],
        masks_list: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        *,
        deterministic: bool = False,
    ) -> list[tuple[int, int, list[int]] | None]:
        """批量选择 rollout 动作；探索动作在 CPU，利用动作共享一次 GNN 前向。"""
        if len(observations) != len(masks_list):
            raise ValueError("observations/masks 数量不一致")
        actions: list[tuple[int, int, list[int]] | None] = [None] * len(observations)
        exploit_indices: list[int] = []
        for index, (observation, masks) in enumerate(zip(observations, masks_list)):
            explore = (
                not deterministic
                and self.action_np_rng.random() <= self.epsilon
            )
            if explore:
                actions[index] = self.random_action(observation, masks)
            else:
                exploit_indices.append(index)
        if exploit_indices:
            # 向量 rollout 的批量 GNN 前向同样置于 autocast，保证输入/权重
            # dtype 由 PyTorch AMP 统一协调。
            with torch.inference_mode(), self._autocast():
                results = select_graph_actions_batch(
                    self.model,
                    [observations[index] for index in exploit_indices],
                    masks_list=[masks_list[index] for index in exploit_indices],
                    device=self.device,
                    deterministic=True,
                    temperature=0.0,
                    need_value=False,
                )
            for result_index, observation_index in enumerate(exploit_indices):
                actions[observation_index] = results[result_index].action
        return actions

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
        dataset_idx = int(state_snapshot.get("dataset_idx", 0))
        self.memory.append(
            dataset_idx=dataset_idx,
            state_snapshot=state_snapshot,
            action=action,
            reward=reward,
            next_snapshot=next_snapshot,
            done=done,
            masks=masks,
            next_masks=next_masks,
        )

    def replay_candidate_count(self, dataset_idx: int) -> int:
        return self.memory.count(int(dataset_idx))

    def can_replay(self, dataset_idx: int, batch_size: int) -> bool:
        required = max(int(batch_size), self.replay_start_size)
        return self.replay_candidate_count(dataset_idx) >= required

    def _q_from_encoded(
        self,
        model: GraphDDQNAPAL,
        encoded: EncodedGraphBatch,
        actions: list[tuple[int, int, list[int]]],
        masks_list: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        *,
        sample_indices: list[int] | None = None,
    ) -> torch.Tensor:
        indices = sample_indices or list(range(len(actions)))
        if len(indices) != len(actions) or len(actions) != len(masks_list):
            raise ValueError("Q 估值的 indices/actions/masks 数量不一致")
        with self._autocast():
            q_values: list[torch.Tensor] = []
            for local_idx, (batch_idx, action) in enumerate(zip(indices, actions)):
                task_idx, station_idx, team = action
                task_start = encoded.task_ptr[batch_idx]
                task_end = encoded.task_ptr[batch_idx + 1]
                station_start = encoded.station_ptr[batch_idx]
                station_end = encoded.station_ptr[batch_idx + 1]
                worker_start = encoded.worker_ptr[batch_idx]
                worker_end = encoded.worker_ptr[batch_idx + 1]
                task_embs = encoded.x_dict["task"][task_start:task_end]
                station_embs = encoded.x_dict["station"][station_start:station_end]
                worker_embs = encoded.x_dict["worker"][worker_start:worker_end]
                context_i = encoded.context[batch_idx].unsqueeze(0)
                task_mask, station_mask_matrix, worker_mask = masks_list[local_idx]

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

                current_worker_mask = worker_static_mask_from_features(
                    encoded.batch["task"].x[task_start:task_end],
                    encoded.batch["worker"].x[worker_start:worker_end],
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

    def _q_for_actions_batched(
        self,
        model: GraphDDQNAPAL,
        states: list[HeteroData],
        actions: list[tuple[int, int, list[int]]],
        masks_list: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        with self._autocast():
            encoded = encode_graph_batch(model, states, device=self.device)
        return self._q_from_encoded(model, encoded, actions, masks_list)

    def _encode_snapshots_on_gpu(
        self,
        model: GraphDDQNAPAL,
        snapshots: list[dict[str, Any]],
        env: Any,
        manager: GPUBatchGraphManager,
    ) -> EncodedGraphBatch:
        batch = manager.batched_rebuild_on_gpu(snapshots, env)
        task_ptr = batch["task"].ptr.detach().cpu().tolist()
        station_ptr = batch["station"].ptr.detach().cpu().tolist()
        worker_ptr = batch["worker"].ptr.detach().cpu().tolist()
        x_dict, context = model(batch)
        return EncodedGraphBatch(
            observations=[],
            batch=batch,
            task_ptr=task_ptr,
            station_ptr=station_ptr,
            worker_ptr=worker_ptr,
            x_dict=x_dict,
            context=context,
        )

    def _disable_gpu_batch_rebuild(self, reason: BaseException) -> None:
        if not self.gpu_batch_rebuild_enabled:
            return
        self.gpu_batch_rebuild_enabled = False
        if self.current_graph_manager is not None:
            self.current_graph_manager.clear()
        if self.next_graph_manager is not None:
            self.next_graph_manager.clear()
        self.current_graph_manager = None
        self.next_graph_manager = None
        clear_torch_cache()
        print(
            f"WARNING: {METHOD_NAME} GPU batch rebuild 已关闭并回退 CPU: "
            f"{type(reason).__name__}: {reason}",
            flush=True,
        )

    def replay(self, env: Any, batch_size: int, *, dataset_idx: int) -> float:
        try:
            self.replay_attempts += 1
            if not self.can_replay(dataset_idx, batch_size):
                return 0.0
            profile_this_update = (
                self.enable_profiler
                and self.replay_attempts % self.profile_interval_updates == 0
            )

            def stage_start() -> float:
                if profile_this_update and self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                return time.perf_counter()

            def stage_end(started: float) -> float:
                if profile_this_update and self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                return time.perf_counter() - started

            profile: dict[str, float] = {}
            started = stage_start()
            samples = self.memory.sample(dataset_idx, int(batch_size))
            state_snapshots = [item.state_snapshot for item in samples]
            next_snapshots = [item.next_snapshot for item in samples]
            states: list[HeteroData] | None = None
            next_states: list[HeteroData] | None = None
            if not self.gpu_batch_rebuild_enabled:
                states = [env.rebuild_state_from_snapshot(snapshot) for snapshot in state_snapshots]
                next_states = [env.rebuild_state_from_snapshot(snapshot) for snapshot in next_snapshots]
            actions = [item.action for item in samples]
            masks = [item.masks for item in samples]
            next_masks = [item.next_masks for item in samples]
            rewards = torch.tensor(
                [item.reward for item in samples],
                dtype=torch.float32,
                device=self.device,
            )
            dones = torch.tensor(
                [float(item.done) for item in samples],
                dtype=torch.float32,
                device=self.device,
            )
            profile["rebuild_sec"] = stage_end(started)

            started = stage_start()
            with self._autocast():
                if self.gpu_batch_rebuild_enabled:
                    assert self.current_graph_manager is not None
                    try:
                        current_encoded = self._encode_snapshots_on_gpu(
                            self.model,
                            state_snapshots,
                            env,
                            self.current_graph_manager,
                        )
                    except Exception as exc:
                        self._disable_gpu_batch_rebuild(exc)
                        states = [
                            env.rebuild_state_from_snapshot(snapshot)
                            for snapshot in state_snapshots
                        ]
                        next_states = [
                            env.rebuild_state_from_snapshot(snapshot)
                            for snapshot in next_snapshots
                        ]
                        current_encoded = encode_graph_batch(
                            self.model,
                            states,
                            device=self.device,
                        )
                else:
                    assert states is not None
                    current_encoded = encode_graph_batch(
                        self.model,
                        states,
                        device=self.device,
                    )
            current_q = self._q_from_encoded(
                self.model,
                current_encoded,
                actions,
                masks,
            )
            profile["current_q_sec"] = stage_end(started)

            started = stage_start()
            with torch.inference_mode(), self._autocast():
                if self.enable_batched_replay:
                    if self.gpu_batch_rebuild_enabled:
                        assert self.next_graph_manager is not None
                        try:
                            next_encoded_online = self._encode_snapshots_on_gpu(
                                self.model,
                                next_snapshots,
                                env,
                                self.next_graph_manager,
                            )
                        except Exception as exc:
                            self._disable_gpu_batch_rebuild(exc)
                            next_states = [
                                env.rebuild_state_from_snapshot(snapshot)
                                for snapshot in next_snapshots
                            ]
                            next_encoded_online = encode_graph_batch(
                                self.model,
                                next_states,
                                device=self.device,
                            )
                    else:
                        if next_states is None:
                            next_states = [
                                env.rebuild_state_from_snapshot(snapshot)
                                for snapshot in next_snapshots
                            ]
                        next_encoded_online = encode_graph_batch(
                            self.model,
                            next_states,
                            device=self.device,
                        )
                    next_results = decode_graph_actions_batch(
                        self.model,
                        next_encoded_online,
                        masks_list=next_masks,
                        device=self.device,
                        deterministic=True,
                    )
                    next_actions = [result.action for result in next_results]
                else:
                    next_encoded_online = None
                    if next_states is None:
                        next_states = [
                            env.rebuild_state_from_snapshot(snapshot)
                            for snapshot in next_snapshots
                        ]
                    next_actions = [
                        select_graph_action(
                            self.model,
                            next_state,
                            masks=mask,
                            device=self.device,
                            deterministic=True,
                            temperature=0.0,
                            need_value=False,
                        ).action
                        for next_state, mask in zip(next_states, next_masks)
                    ]
            profile["next_action_sec"] = stage_end(started)

            # current_q 在 AMP 下可能是 Half；target_q 用 FP32 保存，避免后续
            # 对索引位置写入 FP32 target_q_valid 时触发 Half/Float 冲突。
            target_q = torch.zeros_like(current_q, dtype=torch.float32)
            valid_indices = [idx for idx, action in enumerate(next_actions) if action is not None]
            if valid_indices:
                valid_actions = [next_actions[idx] for idx in valid_indices]
                valid_masks = [next_masks[idx] for idx in valid_indices]
                assert all(action is not None for action in valid_actions)
                started = stage_start()
                with torch.no_grad(), self._autocast():
                    if next_encoded_online is None:
                        assert next_states is not None
                        target_encoded = encode_graph_batch(
                            self.target_model,
                            next_states,
                            device=self.device,
                        )
                    else:
                        target_x, target_context = self.target_model(
                            next_encoded_online.batch
                        )
                        target_encoded = replace(
                            next_encoded_online,
                            x_dict=target_x,
                            context=target_context,
                        )
                    target_q_valid = self._q_from_encoded(
                        self.target_model,
                        target_encoded,
                        [action for action in valid_actions if action is not None],
                        valid_masks,
                        sample_indices=valid_indices,
                    )
                target_q[torch.tensor(valid_indices, dtype=torch.long, device=self.device)] = (
                    target_q_valid.float()
                )
                profile["target_q_sec"] = stage_end(started)
            else:
                profile["target_q_sec"] = 0.0

            started = stage_start()
            q_target = rewards + self.gamma * target_q.detach() * (1.0 - dones)
            loss = self.loss_fn(current_q.float(), q_target.float())
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            profile["backward_sec"] = stage_end(started)
            if self.epsilon > self.epsilon_min:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.successful_updates += 1
            if profile_this_update:
                profile["total_sec"] = sum(profile.values())
                self.last_replay_profile = profile
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
            "replay_buffer_in_checkpoint": False,
            "exact_resume_sidecar_supported": True,
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_exact_resume_state(
    output_dir: Path,
    agent: GraphDDQNAgent,
    scheduler: DatasetUTDScheduler,
    best_makespan: float,
    args: Any,
    *,
    episode: int,
) -> None:
    checkpoint_path = output_dir / "graph_ddqn_apal_exact_resume.pth"
    sidecar_path = output_dir / "graph_ddqn_apal_exact_resume.replay.pt.gz"
    manifest_path = output_dir / "graph_ddqn_apal_exact_resume.json"
    _save_checkpoint(checkpoint_path, agent, best_makespan, args, episode=episode)

    state = {
        "format_version": 1,
        "episode": int(episode),
        "best_makespan": float(best_makespan),
        "memory": agent.memory.state_dict(),
        "utd_scheduler": scheduler.state_dict(),
        "action_np_rng_state": agent.action_np_rng.bit_generator.state,
        "action_py_rng_state": agent.action_py_rng.getstate(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "replay_attempts": int(agent.replay_attempts),
        "successful_updates": int(agent.successful_updates),
    }
    temporary = sidecar_path.with_suffix(sidecar_path.suffix + ".tmp")
    with gzip.open(temporary, "wb", compresslevel=3) as handle:
        torch.save(state, handle)
    temporary.replace(sidecar_path)
    _atomic_json(
        manifest_path,
        {
            "format_version": 1,
            "episode": int(episode),
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": _sha256(checkpoint_path),
            "sidecar": sidecar_path.name,
            "sidecar_sha256": _sha256(sidecar_path),
        },
    )
    print(
        f"[{METHOD_NAME}][ExactResume] ep={episode} checkpoint={checkpoint_path} "
        f"sidecar={sidecar_path}",
        flush=True,
    )


def _load_exact_resume_state(
    output_dir: Path,
    agent: GraphDDQNAgent,
    scheduler: DatasetUTDScheduler,
) -> tuple[int, float] | None:
    manifest_path = output_dir / "graph_ddqn_apal_exact_resume.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_path = output_dir / str(manifest["checkpoint"])
    sidecar_path = output_dir / str(manifest["sidecar"])
    for path in (checkpoint_path, sidecar_path):
        if not path.exists():
            raise FileNotFoundError(f"精确续训文件缺失: {path}")
    if _sha256(checkpoint_path) != str(manifest["checkpoint_sha256"]):
        raise ValueError(f"精确续训 checkpoint 哈希不匹配: {checkpoint_path}")
    if _sha256(sidecar_path) != str(manifest["sidecar_sha256"]):
        raise ValueError(f"精确续训 replay sidecar 哈希不匹配: {sidecar_path}")

    start_episode, best_makespan = _load_resume_checkpoint(checkpoint_path, agent)
    with gzip.open(sidecar_path, "rb") as handle:
        state = torch.load(handle, map_location="cpu", weights_only=False)
    agent.memory.load_state_dict(state["memory"])
    scheduler.load_state_dict(state["utd_scheduler"])
    agent.action_np_rng.bit_generator.state = state["action_np_rng_state"]
    agent.action_py_rng.setstate(state["action_py_rng_state"])
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state.get("torch_cuda_rng_states"):
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_states"])
    agent.replay_attempts = int(state.get("replay_attempts", 0))
    agent.successful_updates = int(state.get("successful_updates", 0))
    expected_episode = int(state["episode"]) + 1
    if start_episode != expected_episode:
        raise ValueError(
            f"精确续训 checkpoint/sidecar episode 不一致: "
            f"{start_episode} != {expected_episode}"
        )
    print(
        f"[{METHOD_NAME}][ExactResume] restored episode={start_episode} "
        f"replay={len(agent.memory)} updates={agent.successful_updates}",
        flush=True,
    )
    return start_episode, best_makespan


def _resolve_updates_per_transition(args: Any) -> float:
    configured = float(getattr(args, "ddqn_updates_per_transition", 0.125))
    legacy_every = getattr(args, "replay_every_steps", None)
    explicitly_set = set(getattr(args, "explicit_config_fields", set()))
    if legacy_every is not None and "ddqn_updates_per_transition" in explicitly_set:
        raise ValueError(
            "ddqn_updates_per_transition 与已弃用 replay_every_steps 不能同时设置"
        )
    if legacy_every is not None:
        legacy_every = int(legacy_every)
        if legacy_every <= 0:
            raise ValueError("replay_every_steps 必须大于 0")
        ratio = 1.0 / float(legacy_every)
        print(
            "WARNING: replay_every_steps 已弃用，已映射为 "
            f"ddqn_updates_per_transition={ratio:.6f}",
            flush=True,
        )
        return ratio
    ratio = float(configured)
    if ratio < 0.0:
        raise ValueError("ddqn_updates_per_transition 不能小于 0")
    return ratio


def _dataset_index_for_episode(dataset_count: int, episode: int, seed: int) -> int:
    if int(dataset_count) <= 1:
        return 0
    if bool(getattr(configs, "random_sample_dataset", True)):
        rng = np.random.RandomState(int(seed) + int(episode) * 9973)
        return int(rng.randint(0, int(dataset_count)))
    return int(episode) % int(dataset_count)


def _training_randomization_flags(episode: int) -> tuple[bool, bool]:
    """复用主方法训练阶段的随机化开关，不改变评估逻辑。"""
    enabled = bool(
        getattr(configs, "randomize_durations", True)
        and int(episode) > int(getattr(configs, "curriculum_episodes", 0))
    )
    return enabled, enabled


def _realized_utd(agent: GraphDDQNAgent, scheduler: DatasetUTDScheduler) -> float:
    if scheduler.transitions_after_warmup <= 0:
        return 0.0
    return float(agent.successful_updates) / float(scheduler.transitions_after_warmup)


def _train_vectorized(
    *,
    args: Any,
    output_dir: Path,
    train_env: Any,
    eval_env: Any,
    agent: GraphDDQNAgent,
    scheduler: DatasetUTDScheduler,
    device: torch.device,
    batch_size: int,
    max_episodes: int,
    start_episode: int,
    best_makespan: float,
    rows: list[dict[str, Any]],
    latest_path: Path,
    best_path: Path,
    target_update_episodes: int,
    max_replay_updates_per_episode: int,
    progress_interval_steps: int,
    exact_resume_enabled: bool,
    exact_checkpoint_interval: int,
    async_manager: AsyncEvaluationManager | None,
) -> tuple[list[dict[str, Any]], float]:
    seed = int(getattr(configs, "seed", 42))
    requested_envs = max(1, int(getattr(args, "ddqn_num_envs", 1)))
    num_envs = min(requested_envs, max(1, max_episodes - start_episode + 1))
    start_method = str(getattr(configs, "vector_env_start_method", "auto"))
    if start_method == "auto":
        start_method = "forkserver" if platform.system() == "Linux" else "spawn"
    vector_env = VectorEnv(
        EnvCreator(str(training_data_source(args)), seed_offset=seed),
        num_envs=num_envs,
        start_method=start_method,
        worker_threads=getattr(configs, "vector_env_worker_threads", 1),
        init_timeout_sec=float(getattr(configs, "vector_env_init_timeout_sec", 120.0)),
        command_timeout_sec=float(getattr(configs, "vector_env_command_timeout_sec", 120.0)),
    )
    use_ipc_fusion = bool(getattr(configs, "ddqn_enable_ipc_fusion", True))
    eval_freq = max(1, int(getattr(configs, "eval_freq", 1)))
    vector_round = 0
    episode_cursor = int(start_episode)
    last_exact_episode = max(0, start_episode - 1)
    try:
        while episode_cursor <= max_episodes:
            vector_round += 1
            wave_episodes = list(
                range(
                    episode_cursor,
                    min(max_episodes + 1, episode_cursor + num_envs),
                )
            )
            active_indices = list(range(len(wave_episodes)))
            episode_by_index = dict(zip(active_indices, wave_episodes))
            # 与主方法保持公平：一次向量 wave 共享一个训练数据集，不能让不同
            # worker 各自抽样导致每轮有效训练分布不一致。
            shared_dataset_idx = _dataset_index_for_episode(
                int(train_env.dataset_count),
                wave_episodes[0],
                seed,
            )
            dataset_by_index = {
                index: shared_dataset_idx for index in active_indices
            }
            vector_env.switch_dataset_all(shared_dataset_idx)
            states = vector_env.reset_indices(
                {
                    index: {
                        "randomize_duration": _training_randomization_flags(
                            episode_by_index[index]
                        )[0],
                        "randomize_workers": _training_randomization_flags(
                            episode_by_index[index]
                        )[1],
                        "seed": seed + episode_by_index[index],
                    }
                    for index in active_indices
                }
            )
            dones = {index: False for index in active_indices}
            rollout_states = vector_env.get_rollout_state_indices(active_indices)
            for index in active_indices:
                states[index] = vector_env.envs[index].rebuild_state_from_snapshot(
                    rollout_states[index][1]
                )
            rewards_sum = {index: 0.0 for index in active_indices}
            invalid_counts = {index: 0 for index in active_indices}
            step_counts = {index: 0 for index in active_indices}
            wave_loss = 0.0
            wave_updates = 0
            wave_attempts = 0
            wave_start = time.time()
            vector_step = 0

            while not all(dones.values()):
                vector_step += 1
                running = [index for index in active_indices if not dones[index]]
                if not running:
                    break

                while True:
                    waiting = [
                        index
                        for index in running
                        if bool(rollout_states[index][0][0].all())
                    ]
                    if not waiting:
                        break
                    if use_ipc_fusion:
                        fused_wait = vector_env.wait_rollout_indices(waiting)
                        wait_results = {
                            index: fused_wait[index][0] for index in waiting
                        }
                    else:
                        fused_wait = {}
                        wait_results = vector_env.try_wait_for_resources_indices(waiting)
                    failed = [index for index in waiting if not wait_results[index]]
                    for index in failed:
                        invalid_counts[index] += 1
                        dones[index] = True
                    refreshed = [index for index in waiting if wait_results[index]]
                    if refreshed:
                        if use_ipc_fusion:
                            rollout_states.update(
                                {
                                    index: (fused_wait[index][1], fused_wait[index][2])
                                    for index in refreshed
                                }
                            )
                        else:
                            rollout_states.update(
                                vector_env.get_rollout_state_indices(refreshed)
                            )
                        for index in refreshed:
                            states[index] = vector_env.envs[index].rebuild_state_from_snapshot(
                                rollout_states[index][1]
                            )
                    running = [index for index in running if not dones[index]]
                    if not running:
                        break
                if not running:
                    break

                ready = []
                for index in running:
                    step_counts[index] += 1
                    max_steps = rollout_step_limit(vector_env.envs[index].num_tasks)
                    if step_counts[index] > max_steps:
                        # 受控冒烟截断不是非法动作；默认配置的上限为自然完整 rollout。
                        dones[index] = True
                    else:
                        ready.append(index)
                if not ready:
                    continue

                selected_actions = agent.select_actions_batch(
                    [states[index] for index in ready],
                    [rollout_states[index][0] for index in ready],
                    deterministic=False,
                )
                actions = {
                    index: selected_actions[position]
                    for position, index in enumerate(ready)
                    if selected_actions[position] is not None
                }
                for position, index in enumerate(ready):
                    if selected_actions[position] is None:
                        invalid_counts[index] += 1
                        rewards_sum[index] -= 100.0
                        dones[index] = True
                if not actions:
                    continue

                if use_ipc_fusion:
                    fused_steps = vector_env.step_rollout_indices(actions)
                    step_results = {
                        index: (
                            fused_steps[index][1],
                            fused_steps[index][2],
                            fused_steps[index][3],
                            fused_steps[index][4],
                        )
                        for index in sorted(actions)
                    }
                    next_rollout_states = {
                        index: (fused_steps[index][0], fused_steps[index][1])
                        for index in sorted(actions)
                    }
                else:
                    step_results = vector_env.step_snapshot_indices(actions)
                    next_rollout_states = vector_env.get_rollout_state_indices(
                        sorted(actions)
                    )
                for index in sorted(actions):
                    next_snapshot, reward, step_done, info = step_results[index]
                    next_masks, authoritative_next_snapshot = next_rollout_states[index]
                    if bool(info.get("invalid_action", False)):
                        invalid_counts[index] += 1
                        step_done = True
                    agent.remember(
                        rollout_states[index][1],
                        actions[index],
                        reward,
                        authoritative_next_snapshot,
                        step_done,
                        rollout_states[index][0],
                        next_masks,
                    )
                    rewards_sum[index] += float(reward)
                    dataset_idx = dataset_by_index[index]
                    updates_due = scheduler.record_transition(
                        dataset_idx,
                        replay_ready=agent.can_replay(dataset_idx, batch_size),
                    )
                    if max_replay_updates_per_episode > 0:
                        updates_due = min(
                            updates_due,
                            max(
                                0,
                                max_replay_updates_per_episode * len(wave_episodes)
                                - wave_attempts,
                            ),
                        )
                    for _ in range(updates_due):
                        wave_attempts += 1
                        before_updates = agent.successful_updates
                        loss_value = agent.replay(
                            train_env,
                            batch_size,
                            dataset_idx=dataset_idx,
                        )
                        if agent.successful_updates > before_updates:
                            wave_loss += loss_value
                            wave_updates += 1
                    dones[index] = bool(step_done)
                    if not dones[index]:
                        states[index] = vector_env.envs[index].rebuild_state_from_snapshot(
                            authoritative_next_snapshot
                        )
                        rollout_states[index] = (
                            next_masks,
                            authoritative_next_snapshot,
                        )

                if (
                    progress_interval_steps > 0
                    and vector_step % progress_interval_steps == 0
                ):
                    completed = sum(1 for value in dones.values() if value)
                    print(
                        f"[{METHOD_NAME}][VectorProgress] round={vector_round} "
                        f"step={vector_step} completed={completed}/{len(wave_episodes)} "
                        f"updates={wave_updates} utd={_realized_utd(agent, scheduler):.3f} "
                        f"elapsed={time.time() - wave_start:.1f}s",
                        flush=True,
                    )

            wave_duration = time.time() - wave_start
            wave_rows: list[dict[str, Any]] = []
            for index in active_indices:
                proxy = vector_env.envs[index]
                complete = len(proxy.assigned_tasks) == int(proxy.num_tasks)
                makespan = (
                    float(np.max(proxy.station_wall_clock))
                    if complete and proxy.station_wall_clock is not None
                    else float(proxy.ideal_makespan) * 3.0
                )
                wave_rows.append(
                    {
                        "episode": episode_by_index[index],
                        "dataset_idx": dataset_by_index[index],
                        "vector_round": vector_round,
                        "reward": float(rewards_sum[index]),
                        "loss": float(wave_loss / max(1, wave_updates)),
                        "makespan": makespan,
                        "assigned": float(len(proxy.assigned_tasks)),
                        "complete": 1.0 if complete else 0.0,
                        "invalid_count": float(invalid_counts[index]),
                        "epsilon": float(agent.epsilon),
                        "oom_skipped_updates": float(agent.oom_skipped_updates),
                        "duration_sec": float(wave_duration),
                        "environment_steps": float(step_counts[index]),
                        "replay_updates": float(wave_updates),
                        "replay_attempts": float(wave_attempts),
                        "updates_per_transition": float(scheduler.updates_per_transition),
                        "effective_utd": float(_realized_utd(agent, scheduler)),
                        "vector_num_envs": float(num_envs),
                        "ipc_fusion": float(use_ipc_fusion),
                    }
                )

            last_episode = wave_episodes[-1]
            if any(episode % target_update_episodes == 0 for episode in wave_episodes):
                agent.target_model.load_state_dict(agent.model.state_dict())

            eval_boundaries = [episode for episode in wave_episodes if episode % eval_freq == 0]
            if eval_boundaries and async_manager is not None:
                episode = int(eval_boundaries[-1])
                async_manager.submit(
                    LiteratureCheckpointSaver(
                        lambda path: _save_checkpoint(path, agent, best_makespan, args, episode=episode)
                    ),
                    episode=episode,
                )
            if eval_boundaries and async_manager is None:
                eval_metrics, eval_schedule, _eval_runs = evaluate_graph_policy(
                    agent.model,
                    eval_env,
                    device,
                    seed=seed,
                    num_runs=1,
                    temperature=float(getattr(configs, "eval_temperature", 0.0)),
                )
                target_row = wave_rows[wave_episodes.index(eval_boundaries[-1])]
                target_row.update(
                    {
                        "eval_makespan": float(eval_metrics["makespan"]),
                        "eval_valid": float(eval_metrics["valid"]),
                        "eval_complete": float(eval_metrics["complete"]),
                        "eval_inference_time": float(eval_metrics["inference_time"]),
                    }
                )
                if eval_metrics["valid"] >= 1.0 and float(eval_metrics["makespan"]) < best_makespan:
                    best_makespan = float(eval_metrics["makespan"])
                    _save_checkpoint(best_path, agent, best_makespan, args, episode=last_episode)
                    export_best_schedule(output_dir, list(eval_schedule), title="graph_ddqn_apal_best")

            rows.extend(wave_rows)
            print(
                f"[{METHOD_NAME}][VectorTrain] episodes={wave_episodes[0]}-{last_episode}/{max_episodes} "
                f"updates={wave_updates} loss={wave_loss / max(1, wave_updates):.6f} "
                f"utd={_realized_utd(agent, scheduler):.3f} best={best_makespan:.2f} "
                f"T={wave_duration:.1f}s",
                flush=True,
            )
            _save_checkpoint(latest_path, agent, best_makespan, args, episode=last_episode)
            if (
                exact_resume_enabled
                and last_episode - last_exact_episode >= exact_checkpoint_interval
            ):
                _save_exact_resume_state(
                    output_dir,
                    agent,
                    scheduler,
                    best_makespan,
                    args,
                    episode=last_episode,
                )
                last_exact_episode = last_episode
            if last_episode % 50 < len(wave_episodes):
                write_training_metrics(output_dir, rows)
                clear_torch_cache()
            episode_cursor = last_episode + 1
    finally:
        vector_env.close()
    return rows, best_makespan


def train(args: Any) -> None:
    seed = int(getattr(configs, "seed", 42))
    set_seed(seed)
    output_dir = prepare_literature_output(args, method_name=METHOD_NAME, entrypoint=ENTRYPOINT)
    start_time = time.time()
    device = get_available_device()
    r5_protocol = is_r5_learning_protocol(configs)
    if r5_protocol and device.type != "cuda":
        raise RuntimeError("r5 Graph-DDQN 训练期异步验证必须使用 CUDA")
    train_env = make_training_env(args, seed=seed)
    eval_env = make_eval_env(args, seed=seed)
    agent = GraphDDQNAgent(args, device)
    batch_size = int(getattr(configs, "batch_size", 64))
    max_episodes = int(getattr(configs, "max_episodes", 300))
    target_update_episodes = max(1, int(getattr(args, "target_update_episodes", 20)))
    updates_per_transition = _resolve_updates_per_transition(args)
    utd_scheduler = DatasetUTDScheduler(updates_per_transition)
    max_replay_updates_per_episode = max(
        0,
        int(getattr(args, "max_replay_updates_per_episode", 0)),
    )
    progress_interval_steps = max(0, int(getattr(args, "progress_interval_steps", 50)))
    exact_resume_enabled = bool(getattr(args, "ddqn_exact_resume", True))
    exact_checkpoint_interval = max(
        1,
        int(getattr(args, "ddqn_exact_checkpoint_interval_trajectories", 10)),
    )

    rows: list[dict[str, Any]] = []
    latest_path = output_dir / "graph_ddqn_apal_latest.pth"
    best_path = output_dir / "graph_ddqn_apal_best.pth"
    final_path = output_dir / "graph_ddqn_apal_final.pth"
    start_episode = 1
    best_makespan = float("inf")
    async_manager = (
        AsyncEvaluationManager(config=configs, latest_path=latest_path, best_path=best_path, project_root=PROJECT_ROOT)
        if r5_protocol
        else None
    )

    if bool(getattr(args, "resume", False)):
        exact_state = (
            _load_exact_resume_state(output_dir, agent, utd_scheduler)
            if exact_resume_enabled
            else None
        )
        if exact_state is not None:
            start_episode, best_makespan = exact_state
            resume_mode = "exact"
            resume_source = output_dir / "graph_ddqn_apal_exact_resume.json"
        else:
            if not latest_path.exists():
                raise FileNotFoundError(
                    f"找不到可恢复的 {METHOD_NAME} checkpoint: {latest_path}"
                )
            start_episode, best_makespan = _load_resume_checkpoint(latest_path, agent)
            resume_mode = "warm_replay_rebuild"
            resume_source = latest_path
        rows = load_training_metrics(output_dir, before_episode=start_episode)
        print(
            f"[{METHOD_NAME}] resume source={resume_source} "
            f"start_episode={start_episode} best={best_makespan:.2f} "
            f"mode={resume_mode} replay_buffer={len(agent.memory)}",
            flush=True,
        )

    print(
        f"[{METHOD_NAME}] start episodes={max_episodes} batch_size={batch_size} "
        f"train_datasets={train_env.dataset_count} replay_start={agent.replay_start_size} "
        f"updates_per_transition={updates_per_transition:.6f} "
        f"max_replay_updates_per_episode={max_replay_updates_per_episode} "
        f"batched_replay={agent.enable_batched_replay} "
        f"progress_interval_steps={progress_interval_steps}",
        flush=True,
    )
    if max_replay_updates_per_episode > 0:
        print(
            f"WARNING: {METHOD_NAME} max_replay_updates_per_episode="
            f"{max_replay_updates_per_episode} 会截断 UTD 调度；"
            "论文实验应同时报告 effective_utd。",
            flush=True,
        )

    vector_enabled = bool(getattr(args, "ddqn_enable_vector_env", False))
    if vector_enabled and int(getattr(args, "ddqn_num_envs", 1)) > 1:
        print(
            f"[{METHOD_NAME}] vector_env=true num_envs={int(args.ddqn_num_envs)} "
            f"max_episodes_semantics=completed_trajectories",
            flush=True,
        )
        rows, best_makespan = _train_vectorized(
            args=args,
            output_dir=output_dir,
            train_env=train_env,
            eval_env=eval_env,
            agent=agent,
            scheduler=utd_scheduler,
            device=device,
            batch_size=batch_size,
            max_episodes=max_episodes,
            start_episode=start_episode,
            best_makespan=best_makespan,
            rows=rows,
            latest_path=latest_path,
            best_path=best_path,
            target_update_episodes=target_update_episodes,
            max_replay_updates_per_episode=max_replay_updates_per_episode,
            progress_interval_steps=progress_interval_steps,
            exact_resume_enabled=exact_resume_enabled,
            exact_checkpoint_interval=exact_checkpoint_interval,
            async_manager=async_manager,
        )
        _save_checkpoint(final_path, agent, best_makespan, args, episode=max_episodes)
        if exact_resume_enabled:
            _save_exact_resume_state(
                output_dir,
                agent,
                utd_scheduler,
                best_makespan,
                args,
                episode=max_episodes,
            )
        if async_manager is not None:
            async_manager.finalize(wait=True)
        write_training_metrics(output_dir, rows)
        print(
            f"[{METHOD_NAME}] done elapsed={time.time() - start_time:.1f}s "
            f"best={best_makespan:.2f}",
            flush=True,
        )
        clear_torch_cache()
        return

    for episode in range(start_episode, max_episodes + 1):
        dataset_idx = select_episode_dataset(train_env, episode, seed)
        episode_seed = seed + episode
        randomize_duration, randomize_workers = _training_randomization_flags(episode)
        state = train_env.reset(
            randomize_duration=randomize_duration,
            randomize_workers=randomize_workers,
            seed=episode_seed,
        )
        done = False
        total_reward = 0.0
        total_loss = 0.0
        invalid_count = 0
        step_count = 0
        replay_updates = 0
        replay_attempts = 0
        episode_start = time.time()

        max_steps = rollout_step_limit(train_env.num_tasks)
        while not done and step_count < max_steps and len(train_env.assigned_tasks) < train_env.num_tasks:
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
            replay_ready = agent.can_replay(dataset_idx, batch_size)
            updates_due = utd_scheduler.record_transition(
                dataset_idx,
                replay_ready=replay_ready,
            )
            if max_replay_updates_per_episode > 0:
                updates_due = min(
                    updates_due,
                    max(0, max_replay_updates_per_episode - replay_attempts),
                )
            for _ in range(updates_due):
                replay_attempts += 1
                before_updates = agent.successful_updates
                loss_value = agent.replay(
                    train_env,
                    batch_size,
                    dataset_idx=dataset_idx,
                )
                if agent.successful_updates > before_updates:
                    total_loss += loss_value
                    replay_updates += 1
            if progress_interval_steps > 0 and step_count % progress_interval_steps == 0:
                print(
                    f"[{METHOD_NAME}][Progress] ep={episode}/{max_episodes} "
                    f"step={step_count}/{max_steps} "
                    f"assigned={len(train_env.assigned_tasks)}/{int(train_env.num_tasks)} "
                    f"replay_updates={replay_updates} utd={_realized_utd(agent, utd_scheduler):.3f} "
                    f"replay_buffer={len(agent.memory)} "
                    f"elapsed={time.time() - episode_start:.1f}s",
                    flush=True,
                )

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
            "loss": float(total_loss / max(1, replay_updates)),
            "makespan": float(makespan),
            "assigned": float(len(train_env.assigned_tasks)),
            "complete": 1.0 if complete else 0.0,
            "invalid_count": float(invalid_count),
            "epsilon": float(agent.epsilon),
            "oom_skipped_updates": float(agent.oom_skipped_updates),
            "duration_sec": float(time.time() - episode_start),
            "replay_updates": float(replay_updates),
            "replay_attempts": float(replay_attempts),
            "environment_steps": float(step_count),
            "updates_per_transition": float(updates_per_transition),
            "effective_utd": float(_realized_utd(agent, utd_scheduler)),
            "max_replay_updates_per_episode": float(max_replay_updates_per_episode),
            "progress_interval_steps": float(progress_interval_steps),
        }
        if agent.last_replay_profile:
            row.update(
                {
                    f"replay_{key}": float(value)
                    for key, value in agent.last_replay_profile.items()
                }
            )

        if episode % target_update_episodes == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())

        if async_manager is not None and episode % int(getattr(configs, "async_eval_submit_every_episodes", 2)) == 0:
            async_manager.submit(
                LiteratureCheckpointSaver(
                    lambda path: _save_checkpoint(path, agent, best_makespan, args, episode=episode)
                ),
                episode=episode,
            )
        if episode % int(getattr(configs, "eval_freq", 1)) == 0 and async_manager is None:
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
        if exact_resume_enabled and episode % exact_checkpoint_interval == 0:
            _save_exact_resume_state(
                output_dir,
                agent,
                utd_scheduler,
                best_makespan,
                args,
                episode=episode,
            )
        if episode == 1 or episode % 10 == 0:
            print(f"[{METHOD_NAME}][Checkpoint] ep={episode} 保存最新模型 path={latest_path}", flush=True)
        if episode % 50 == 0:
            write_training_metrics(output_dir, rows)
            clear_torch_cache()

    _save_checkpoint(final_path, agent, best_makespan, args, episode=max_episodes)
    if async_manager is not None:
        async_manager.finalize(wait=True)
    if exact_resume_enabled and max_episodes >= start_episode:
        _save_exact_resume_state(
            output_dir,
            agent,
            utd_scheduler,
            best_makespan,
            args,
            episode=max_episodes,
        )
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
        # 训练服务器上保留完整 traceback，避免 GPU dtype/IPC 错误只剩一句摘要。
        traceback.print_exc(file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
