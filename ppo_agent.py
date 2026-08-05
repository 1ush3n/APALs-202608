import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import math
import copy
import numpy as np
import gc
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Tuple, List, Dict, Optional, Any
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_dense_batch
from configs import configs
from core.action_completion import EarliestFinishActionCompleter, TeamCandidates
from worker_feature_layout import resolve_worker_feature_layout
from training.best_anchor_teacher import BestAnchorTeacherManager
try:
    from schedulefree import AdamWScheduleFree
except ImportError:
    AdamWScheduleFree = None


@dataclass(frozen=True)
class FrozenGatedTeamTrace:
    """采样时冻结的 APAL 团队候选动作空间。

    PPO 的重要性比率必须在与采样时相同的离散动作空间上计算。这里仅保存
    少量候选团队及其门控特征，不保存图张量，从而避免事后重建观测时的动态
    特征差异改变候选集合。
    """

    task_id: int
    station_id: int
    teams: tuple[tuple[int, ...], ...]
    selected_index: int
    gate_features: tuple[float, ...]
    relative_finish_costs: tuple[float, ...]
    gate_value: float
    alternative_probability: float
    best_alternative_logit_gap: float


class PPOAgent:
    """
    PPO (Proximal Policy Optimization) 鏅鸿兘浣撱€?
    璐熻矗涓?Environment 浜や簰锛屾敹闆嗚建杩癸紝骞舵洿鏂?Strategy Network銆?
    """
    def __init__(
        self,
        model,
        lr,
        gamma,
        k_epochs,
        eps_clip,
        device,
        batch_size=4,
        total_timesteps=0,
        config=None,
        *,
        teacher_model_factory: Callable[[], torch.nn.Module] | None = None,
        teacher_checkpoint_dir: Any | None = None,
    ):
        self.config = config if config is not None else configs
        self.action_completer = EarliestFinishActionCompleter(self.config)
        self.policy = model.to(device)
        # 仅供 rollout_service 在同一次批量解码后立即写入 Memory；不参与 checkpoint。
        self.last_gated_team_trace: FrozenGatedTeamTrace | None = None
        self.last_gated_team_traces: list[FrozenGatedTeamTrace | None] = []
        
        from utils.gpu_graph_manager import GPUBatchGraphManager
        self.gpu_graph_manager = GPUBatchGraphManager(device, config=self.config)
        
        self.use_schedule_free = getattr(self.config, 'use_schedule_free', False)
        
        # [SF Enhancement] 鑻ユ湭瀹夎 schedulefree 搴擄紝寮鸿闄嶇骇涓烘櫘閫?AdamW 浼樺寲妯″紡锛岄槻姝㈠悗缁彂鐢?train()/eval() 璋冪敤宕╂簝
        if self.use_schedule_free and AdamWScheduleFree is None:
            print("WARNING: ScheduleFree is requested but the 'schedulefree' package is not installed. Falling back to default AdamW.")
            self.use_schedule_free = False
        
        actor_lr_multiplier = float(getattr(self.config, 'actor_lr_multiplier', 1.0))
        critic_lr_multiplier = float(getattr(self.config, 'critic_lr_multiplier', 1.0))
        actor_params = []
        critic_params = []
        for name, param in self.policy.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("critic") or ".critic" in name:
                critic_params.append(param)
            else:
                actor_params.append(param)
        self.actor_parameters = tuple(actor_params)
        self.critic_parameters = tuple(critic_params)
        optimizer_params = [
            {'params': actor_params, 'lr': lr * actor_lr_multiplier, 'name': 'actor'},
            {'params': critic_params, 'lr': lr * critic_lr_multiplier, 'name': 'critic'},
        ]

        if self.use_schedule_free and AdamWScheduleFree is not None:
            # [SF Enhancement] 鍔ㄦ€佽皟鏁撮鐑湡锛岃瀹氫负鎬绘洿鏂版鏁扮殑 5% (鏈€灏?100)
            warmup = getattr(self.config, 'sf_warmup_steps', max(100, int(max(1, total_timesteps) * 0.05)))
            self.optimizer = AdamWScheduleFree(optimizer_params, lr=lr, weight_decay=1e-4, warmup_steps=warmup)
        else:
            self.optimizer = torch.optim.AdamW(optimizer_params, lr=lr, weight_decay=1e-4)
            
        self.use_ema = getattr(self.config, 'use_ema', False)
        self.ema_decay = getattr(self.config, 'ema_decay', 0.995)
        
        # [SF Enhancement] 鑷姩瑕嗙洊 EMA 闃叉鎺у埗鏉冨啿绐佷笌鍐椾綑璁＄畻
        if self.use_schedule_free and self.use_ema:
            print("INFO: ScheduleFree optimizer enabled; disabling EMA to avoid duplicate parameter averaging.")
            self.use_ema = False
            
        if self.use_ema:
            self.ema_policy = copy.deepcopy(self.policy).to(device)
            # 鍐荤粨 EMA 妯″瀷鐨勬搴﹁繍绠楋紝瀹冨彧浣滀负鏃佽鑰呮帴鏀舵寚娲?
            for param in self.ema_policy.parameters():
                param.requires_grad = False
                
        self.lr = lr
        self.gamma = gamma          # 鎶樻墸鍥犲瓙
        self.k_epochs = k_epochs    # 姣忔 Update 鐨勮凯浠ｈ疆鏁?
        self.eps_clip = eps_clip    # PPO Clip鍙傛暟 (e.g., 0.2)
        self.device = device
        requested_batch_size = max(1, int(batch_size))
        batch_size_cap = max(0, int(getattr(self.config, "ppo_batch_size_cap", 0)))
        self.batch_size = (
            min(requested_batch_size, batch_size_cap)
            if batch_size_cap > 0
            else requested_batch_size
        )
        if self.batch_size != requested_batch_size:
            print(
                f"PPO Batch 平台限幅: requested={requested_batch_size}, "
                f"effective={self.batch_size}, cap={batch_size_cap}"
            )
        self.accumulation_steps = self.config.accumulation_steps
        self.gae_lambda = self.config.gae_lambda
        
        self.MseLoss = nn.MSELoss() 
        
        self.kl_early_stop = self.config.kl_early_stop
        
        self.initial_lr = lr
        
        self.total_timesteps = max(1, total_timesteps)
        self.current_step = 0
        self.amp_device_type = self.device.type if isinstance(self.device, torch.device) else str(self.device)
        self.amp_enabled = self.amp_device_type == "cuda"
        self.scaler = torch.amp.GradScaler(self.amp_device_type, enabled=self.amp_enabled)
        self.best_anchor_teacher: BestAnchorTeacherManager | None = None
        if bool(getattr(self.config, "best_anchor_distill_enabled", False)):
            if teacher_model_factory is None or teacher_checkpoint_dir is None:
                raise ValueError("启用 best-anchor 蒸馏时必须提供模型工厂和当前 run 的 checkpoint 目录")
            self.best_anchor_teacher = BestAnchorTeacherManager(
                config=self.config,
                device=self.device,
                model_factory=teacher_model_factory,
                checkpoint_dir=teacher_checkpoint_dir,
                make_schedulefree_optimizer=self._make_schedulefree_teacher_optimizer,
                use_schedule_free=self.use_schedule_free,
            )
        
        # 鑷€傚簲璇勪及鏂版棫绛栫暐宸窛 (KL鏁ｅ害) 鐨勬柟娉曞湪 update 灏鹃儴鍙樺姩 LR銆?

    def autocast_context(self):
        """返回与当前设备匹配的 AMP 上下文，CPU 路径默认禁用混合精度。"""
        if self.amp_enabled:
            return torch.amp.autocast(device_type=self.amp_device_type)
        return nullcontext()

    def _make_schedulefree_teacher_optimizer(self, model: torch.nn.Module) -> Any:
        """为冻结教师恢复 ScheduleFree 的评估态，不加入学生优化图。"""
        if AdamWScheduleFree is None:
            raise RuntimeError("未安装 schedulefree，无法恢复 ScheduleFree 教师")
        actor_params = []
        critic_params = []
        for name, parameter in model.named_parameters():
            if name.startswith("critic") or ".critic" in name:
                critic_params.append(parameter)
            else:
                actor_params.append(parameter)
        groups = [
            {
                "params": actor_params,
                "lr": self.lr * float(getattr(self.config, "actor_lr_multiplier", 1.0)),
                "name": "actor",
            },
            {
                "params": critic_params,
                "lr": self.lr * float(getattr(self.config, "critic_lr_multiplier", 1.0)),
                "name": "critic",
            },
        ]
        warmup = int(
            getattr(
                self.config,
                "sf_warmup_steps",
                max(100, int(max(1, self.total_timesteps) * 0.05)),
            )
        )
        return AdamWScheduleFree(groups, lr=self.lr, weight_decay=1e-4, warmup_steps=warmup)

    @staticmethod
    def _masked_kl(
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        invalid_mask: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """计算共享动作掩码上的 KL(teacher || student)。"""
        assert teacher_logits.shape == student_logits.shape == invalid_mask.shape
        scale = max(float(temperature), 1.0e-6)
        teacher_log_probs = F.log_softmax(teacher_logits.float() / scale, dim=-1)
        student_log_probs = F.log_softmax(student_logits.float() / scale, dim=-1)
        teacher_probs = teacher_log_probs.exp()
        kl_terms = teacher_probs * (teacher_log_probs - student_log_probs)
        kl_terms = torch.where(invalid_mask, torch.zeros_like(kl_terms), kl_terms)
        return kl_terms.sum(dim=-1) * (scale * scale)

    def _teacher_task_station_logits(
        self,
        batch: HeteroData,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """在学生前向前计算冻结教师的任务/工位 logits，降低显存峰值。"""
        manager = self.best_anchor_teacher
        if manager is None or not manager.active or manager.teacher is None:
            return None
        batch_indices = torch.arange(batch.y_task.size(0), device=self.device)
        # 原始节点特征仅用于构造 [B, N] 的有效节点掩码。
        _, task_present = to_dense_batch(batch["task"].x[:, :1], batch["task"].batch)
        _, station_present = to_dense_batch(batch["station"].x[:, :1], batch["station"].batch)
        if hasattr(batch, "y_task_mask"):
            logical_task_mask, _ = to_dense_batch(batch.y_task_mask, batch["task"].batch)
            task_mask = logical_task_mask | (~task_present)
        else:
            task_mask = ~task_present
        if hasattr(batch, "y_station_mask"):
            dense_station_mask, _ = to_dense_batch(batch.y_station_mask, batch["task"].batch)
            station_mask = dense_station_mask[batch_indices, batch.y_task] | (~station_present)
        else:
            station_mask = ~station_present
        with torch.inference_mode(), self.autocast_context():
            teacher_x, teacher_global = manager.teacher(batch)
            teacher_task_x, _ = to_dense_batch(teacher_x["task"], batch["task"].batch)
            teacher_station_x, _ = to_dense_batch(teacher_x["station"], batch["station"].batch)
            teacher_task_logits = manager.teacher.task_head(
                teacher_task_x, teacher_global, mask=task_mask
            ).float()
            teacher_selected_task = teacher_task_x[batch_indices, batch.y_task]
            teacher_station_logits = manager.teacher.station_head(
                teacher_selected_task, teacher_station_x, mask=station_mask
            ).float()
        return teacher_task_logits, teacher_station_logits, task_mask, station_mask

    def best_anchor_checkpoint_state(self) -> dict[str, Any] | None:
        if self.best_anchor_teacher is None:
            return None
        return self.best_anchor_teacher.checkpoint_state()

    def restore_best_anchor_checkpoint_state(self, raw: object) -> None:
        if self.best_anchor_teacher is None:
            if raw is not None:
                raise RuntimeError("checkpoint 包含 best-anchor 教师，但当前训练未启用该开关")
            return
        self.best_anchor_teacher.restore_checkpoint_state(raw)

    def clear_device_cache(self) -> None:
        """按当前设备清理缓存，降低连续 PPO 更新后的显存或内存残留风险。"""
        gc.collect()
        if self.amp_device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def validate_snapshot_homogeneity(states: List[Any]) -> None:
        """一次 PPO 更新只允许包含同一图和相同工人数的 snapshot。"""
        snapshots = [state for state in states if isinstance(state, dict)]
        if not snapshots:
            return
        if len(snapshots) != len(states):
            raise RuntimeError("PPO memory 混合了 snapshot 与 HeteroData 状态")
        dataset_ids = {int(state.get("dataset_idx", 0)) for state in snapshots}
        worker_counts = {len(state["worker_free_time"]) for state in snapshots}
        if len(dataset_ids) != 1 or len(worker_counts) != 1:
            raise RuntimeError(
                "PPO update 要求同质窄池轨迹，"
                f"dataset_ids={sorted(dataset_ids)}, worker_counts={sorted(worker_counts)}"
            )

    @staticmethod
    def get_task_demand(task_x: torch.Tensor, task_idx: int) -> int:
        """从 task 特征第 16 列读取 APAL 工序硬性需求人数。"""
        assert task_x.dim() == 2 and task_x.size(1) > 16, f"task_x 形状异常: {tuple(task_x.shape)}"
        demand = int(task_x[task_idx, 16].item())
        return max(1, demand)

    def _gated_team_logits(
        self,
        policy: Any,
        *,
        task_emb: torch.Tensor,
        station_emb: torch.Tensor,
        worker_embs: torch.Tensor,
        candidates: TeamCandidates,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """对一个状态的合法团队候选计算条件式门控 logits。"""
        if policy.conditional_team_head is None:
            raise RuntimeError("operation_station_gated_team 缺少 conditional_team_head")
        assert task_emb.shape == station_emb.shape
        assert task_emb.ndim == 2 and task_emb.size(0) == 1
        assert worker_embs.ndim == 2
        team_embeddings = []
        for team in candidates.teams:
            if not team:
                raise RuntimeError("物理工序的门控团队候选不能为空")
            member_ids = torch.tensor(team, dtype=torch.long, device=worker_embs.device)
            # [团队人数, H] -> [H]
            team_embeddings.append(worker_embs.index_select(0, member_ids).mean(dim=0))
        candidate_team_emb = torch.stack(team_embeddings, dim=0).unsqueeze(0)
        gate_features = candidates.gate_features.to(
            device=worker_embs.device, dtype=worker_embs.dtype
        ).unsqueeze(0)
        candidate_prior_costs = candidates.relative_finish_costs.to(
            device=worker_embs.device, dtype=worker_embs.dtype
        ).unsqueeze(0)
        return policy.conditional_team_head(
            task_emb,
            station_emb,
            candidate_team_emb,
            gate_features,
            candidate_prior_costs=candidate_prior_costs,
        )

    def _select_gated_team(
        self,
        policy: Any,
        *,
        obs: HeteroData,
        task_id: int,
        station_id: int,
        worker_mask: torch.Tensor | None,
        task_emb: torch.Tensor,
        station_emb: torch.Tensor,
        worker_embs: torch.Tensor,
        deterministic: bool,
        temperature: float,
    ) -> tuple[list[int], torch.Tensor, FrozenGatedTeamTrace] | None:
        """从同一确定性候选生成器中采样团队，供单环境和批量路径共用。"""
        candidates = self.action_completer.enumerate_team_candidates(
            obs,
            task_id=int(task_id),
            station_id=int(station_id),
            worker_mask=worker_mask,
        )
        if candidates is None:
            return None
        logits, _gate = self._gated_team_logits(
            policy,
            task_emb=task_emb,
            station_emb=station_emb,
            worker_embs=worker_embs,
            candidates=candidates,
        )
        if torch.isnan(logits).any():
            logits = torch.nan_to_num(logits, nan=-1.0e4)
        if deterministic or temperature <= 0.0:
            selected_index = int(torch.argmax(logits, dim=1).item())
            team_logprob = torch.zeros((), device=logits.device)
            selection_logits = logits
        else:
            selection_logits = logits / max(float(temperature), 1.0e-5)
            dist = Categorical(logits=selection_logits.float())
            sampled = dist.sample()
            selected_index = int(sampled.item())
            team_logprob = dist.log_prob(sampled)
        if len(candidates.teams) > 1:
            selection_probs = torch.softmax(selection_logits, dim=1)
            alternative_probability = selection_probs[:, 1:].sum(dim=1)
            best_alternative_gap = logits[:, 1:].max(dim=1).values - logits[:, 0]
        else:
            alternative_probability = logits.new_zeros(1)
            best_alternative_gap = logits.new_zeros(1)
        frozen_values = torch.cat(
            [
                candidates.gate_features,
                candidates.relative_finish_costs,
                _gate.reshape(-1),
                alternative_probability.reshape(-1),
                best_alternative_gap.reshape(-1),
            ],
            dim=0,
        ).detach().to("cpu", torch.float32).tolist()
        frozen_trace = FrozenGatedTeamTrace(
            task_id=int(task_id),
            station_id=int(station_id),
            teams=tuple(tuple(int(worker_id) for worker_id in team) for team in candidates.teams),
            selected_index=selected_index,
            gate_features=tuple(float(value) for value in frozen_values[:5]),
            relative_finish_costs=tuple(
                float(value) for value in frozen_values[5 : 5 + len(candidates.teams)]
            ),
            gate_value=float(frozen_values[-3]),
            alternative_probability=float(frozen_values[-2]),
            best_alternative_logit_gap=float(frozen_values[-1]),
        )
        return list(candidates.teams[selected_index]), team_logprob, frozen_trace

    @staticmethod
    def _gated_team_rollout_metrics(memory: Any) -> dict[str, float]:
        """从冻结轨迹汇总采样期团队决策诊断，不重建候选集。"""
        traces = [
            trace for trace in getattr(memory, "gated_team_traces", [])
            if isinstance(trace, FrozenGatedTeamTrace)
        ]
        if not traces:
            return {}
        multi_candidate = [trace for trace in traces if len(trace.teams) > 1]
        metrics = {
            "CTG/RolloutDecisionCount": float(len(traces)),
            "CTG/RolloutCandidateCountMean": float(
                sum(len(trace.teams) for trace in traces) / len(traces)
            ),
            "CTG/RolloutMultiCandidateDecisionCount": float(len(multi_candidate)),
            "CTG/RolloutMultiCandidateRate": float(len(multi_candidate) / len(traces)),
            "CTG/RolloutGateMean": float(
                sum(trace.gate_value for trace in traces) / len(traces)
            ),
            "CTG/RolloutGateStd": float(
                torch.tensor([trace.gate_value for trace in traces], dtype=torch.float32)
                .std(unbiased=False)
                .item()
            ),
            "CTG/RolloutNonBaselineSelectRate": float(
                sum(trace.selected_index > 0 for trace in traces) / len(traces)
            ),
        }
        if multi_candidate:
            metrics.update(
                {
                    "CTG/RolloutNonBaselineProbMean": float(
                        sum(trace.alternative_probability for trace in multi_candidate)
                        / len(multi_candidate)
                    ),
                    "CTG/RolloutAltVsBaseLogitGapMean": float(
                        sum(trace.best_alternative_logit_gap for trace in multi_candidate)
                        / len(multi_candidate)
                    ),
                    "CTG/RolloutAltBeatsBaseRate": float(
                        sum(
                            trace.best_alternative_logit_gap > 0.0
                            for trace in multi_candidate
                        )
                        / len(multi_candidate)
                    ),
                }
            )
        else:
            metrics.update(
                {
                    "CTG/RolloutNonBaselineProbMean": 0.0,
                    "CTG/RolloutAltVsBaseLogitGapMean": 0.0,
                    "CTG/RolloutAltBeatsBaseRate": 0.0,
                }
            )
        return metrics

    def _recompute_gated_team_logprobs(
        self,
        *,
        batch: HeteroData,
        task_embeddings: torch.Tensor,
        station_embeddings: torch.Tensor,
        worker_embeddings: torch.Tensor,
        raw_task_x: torch.Tensor,
        raw_station_x: torch.Tensor,
        raw_worker_x: torch.Tensor,
        worker_masks: torch.Tensor,
        selected_task: torch.Tensor,
        selected_station: torch.Tensor,
        selected_teams: torch.Tensor,
        frozen_traces: list[FrozenGatedTeamTrace | None],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """使用采样时冻结的团队候选动作空间重算 PPO 对数概率。"""
        batch_size = int(selected_task.numel())
        if len(frozen_traces) != batch_size:
            raise RuntimeError(
                "门控团队 PPO 轨迹缺少冻结候选序列："
                f"traces={len(frozen_traces)} batch={batch_size}"
            )
        candidate_rows: list[TeamCandidates] = []
        chosen_indices: list[int] = []
        for batch_idx in range(batch_size):
            station_id = int(selected_station[batch_idx].item())
            if station_id < 0:
                candidate_rows.append(
                    TeamCandidates(
                        station_id=-1,
                        teams=((0,),),
                        gate_features=torch.zeros(5, device=raw_task_x.device),
                        relative_finish_costs=torch.zeros(1, device=raw_task_x.device),
                    )
                )
                chosen_indices.append(0)
                continue
            trace = frozen_traces[batch_idx]
            if trace is None:
                raise RuntimeError(
                    "门控团队 PPO 轨迹缺少采样时冻结的候选序列；"
                    "拒绝基于事后重枚举的动作空间更新"
                )
            if trace.task_id != int(selected_task[batch_idx].item()) or trace.station_id != station_id:
                raise RuntimeError(
                    "门控团队冻结轨迹与采样动作的工序/工位不一致："
                    f"trace=({trace.task_id},{trace.station_id}) "
                    f"action=({int(selected_task[batch_idx].item())},{station_id})"
                )
            if not trace.teams or not (0 <= trace.selected_index < len(trace.teams)):
                raise RuntimeError("门控团队冻结轨迹的候选索引或候选集合非法")
            selected_team = tuple(
                int(worker_id)
                for worker_id in selected_teams[batch_idx].tolist()
                if int(worker_id) >= 0
            )
            if selected_team != trace.teams[trace.selected_index]:
                raise RuntimeError(
                    "门控团队冻结轨迹与采样动作团队不一致；"
                    "拒绝使用错误的对数概率更新"
                )
            max_worker_count = int(worker_embeddings.size(1))
            if any(
                not team or any(worker_id < 0 or worker_id >= max_worker_count for worker_id in team)
                for team in trace.teams
            ):
                raise RuntimeError("门控团队冻结轨迹包含越界或空团队候选")
            if len(trace.gate_features) != 5:
                raise RuntimeError("门控团队冻结轨迹的门控特征维度必须为 5")
            if len(trace.relative_finish_costs) != len(trace.teams):
                raise RuntimeError("门控团队冻结先验长度与候选数不一致")
            if (
                not all(math.isfinite(value) for value in trace.relative_finish_costs)
                or trace.relative_finish_costs[0] != 0.0
                or any(value < 0.0 for value in trace.relative_finish_costs[1:])
            ):
                raise RuntimeError("门控团队冻结先验不满足相对完工代价约束")
            candidates = TeamCandidates(
                station_id=station_id,
                teams=trace.teams,
                gate_features=torch.tensor(
                    trace.gate_features,
                    dtype=raw_task_x.dtype,
                    device=raw_task_x.device,
                ),
                relative_finish_costs=torch.tensor(
                    trace.relative_finish_costs,
                    dtype=raw_task_x.dtype,
                    device=raw_task_x.device,
                ),
            )
            candidate_rows.append(candidates)
            chosen_indices.append(trace.selected_index)

        max_candidates = max(len(item.teams) for item in candidate_rows)
        hidden_dim = int(worker_embeddings.size(-1))
        candidate_emb_rows = []
        candidate_mask_rows = []
        feature_rows = []
        prior_cost_rows = []
        for batch_idx, candidates in enumerate(candidate_rows):
            row_embeddings = []
            for team in candidates.teams:
                member_ids = torch.tensor(team, dtype=torch.long, device=worker_embeddings.device)
                # [团队人数, H] -> [H]
                row_embeddings.append(
                    worker_embeddings[batch_idx].index_select(0, member_ids).mean(dim=0)
                )
            padding = max_candidates - len(row_embeddings)
            if padding:
                row_embeddings.extend(
                    [worker_embeddings.new_zeros(hidden_dim) for _ in range(padding)]
                )
            candidate_emb_rows.append(torch.stack(row_embeddings, dim=0))
            candidate_mask_rows.append(
                [False] * len(candidates.teams) + [True] * padding
            )
            feature_rows.append(
                candidates.gate_features.to(
                    device=worker_embeddings.device, dtype=worker_embeddings.dtype
                )
            )
            prior_cost_rows.append(
                torch.nn.functional.pad(
                    candidates.relative_finish_costs.to(
                        device=worker_embeddings.device,
                        dtype=worker_embeddings.dtype,
                    ),
                    (0, padding),
                    value=0.0,
                )
            )

        if self.policy.conditional_team_head is None:
            raise RuntimeError("operation_station_gated_team 缺少 conditional_team_head")
        candidate_team_emb = torch.stack(candidate_emb_rows, dim=0)
        candidate_mask = torch.tensor(
            candidate_mask_rows, dtype=torch.bool, device=worker_embeddings.device
        )
        gate_features = torch.stack(feature_rows, dim=0)
        candidate_prior_costs = torch.stack(prior_cost_rows, dim=0)
        logits, _gate = self.policy.conditional_team_head(
            task_embeddings,
            station_embeddings,
            candidate_team_emb,
            gate_features,
            candidate_mask,
            candidate_prior_costs,
        )
        dist = Categorical(logits=logits.float())
        selected_indices = torch.tensor(
            chosen_indices, dtype=torch.long, device=worker_embeddings.device
        )
        active = selected_station >= 0
        team_lp = dist.log_prob(selected_indices) * active.to(logits.dtype)
        team_entropy = dist.entropy() * active.to(logits.dtype)
        if max_candidates > 1:
            has_alternative = (~candidate_mask[:, 1:]).any(dim=1) & active
            best_alternative = logits[:, 1:].masked_fill(
                candidate_mask[:, 1:], -1.0e4
            ).max(dim=1).values
            alternative_probability = (dist.probs[:, 1:] * (~candidate_mask[:, 1:])).sum(dim=1)
            best_alternative_gap = best_alternative - logits[:, 0]
        else:
            has_alternative = torch.zeros_like(active)
            alternative_probability = logits.new_zeros(batch_size)
            best_alternative_gap = logits.new_zeros(batch_size)
        diagnostics = {
            "gate": _gate.squeeze(-1),
            "multi": has_alternative,
            "alternative_probability": alternative_probability,
            "best_alternative_gap": best_alternative_gap,
        }
        return team_lp, team_entropy, diagnostics

    def get_memory_snapshot(self) -> Dict[str, float]:
        """返回当前设备显存快照，单位为 GB；CPU 环境返回 0。"""
        if self.amp_device_type != "cuda" or not torch.cuda.is_available():
            return {"allocated_gb": 0.0, "reserved_gb": 0.0}
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        }

    @staticmethod
    def _clone_state_to_cpu(value: Any) -> Any:
        """递归复制训练状态到 CPU，避免事务快照额外占用显存。"""
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: PPOAgent._clone_state_to_cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [PPOAgent._clone_state_to_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(PPOAgent._clone_state_to_cpu(item) for item in value)
        return copy.deepcopy(value)

    @staticmethod
    def _is_cuda_oom_error(exc: BaseException) -> bool:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
        message = str(exc).lower()
        return (
            ("out of memory" in message and any(token in message for token in ("cuda", "gpu", "device")))
            or "defaultcpuallocator" in message
            or "not enough memory" in message
        )

    def _capture_update_transaction(self) -> Dict[str, Any]:
        """保存可完整回滚一次 PPO 更新所需的训练与随机状态。"""
        transaction = {
            "policy": self._clone_state_to_cpu(self.policy.state_dict()),
            "optimizer": self._clone_state_to_cpu(self.optimizer.state_dict()),
            "scaler": copy.deepcopy(self.scaler.state_dict()),
            "current_step": int(self.current_step),
            "batch_size": int(self.batch_size),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
        }
        if self.use_ema and hasattr(self, "ema_policy"):
            transaction["ema_policy"] = self._clone_state_to_cpu(
                self.ema_policy.state_dict()
            )
        if torch.cuda.is_available():
            transaction["cuda_rng"] = torch.cuda.get_rng_state_all()
        return transaction

    def _restore_update_transaction(self, transaction: Dict[str, Any]) -> None:
        """恢复 OOM 前的模型、优化器、缩放器及随机状态。"""
        self.policy.load_state_dict(transaction["policy"])
        self.optimizer.load_state_dict(transaction["optimizer"])
        self.scaler.load_state_dict(transaction["scaler"])
        self.current_step = int(transaction["current_step"])
        self.batch_size = int(transaction["batch_size"])
        random.setstate(transaction["python_rng"])
        np.random.set_state(transaction["numpy_rng"])
        torch.set_rng_state(transaction["torch_rng"])
        if "ema_policy" in transaction:
            self.ema_policy.load_state_dict(transaction["ema_policy"])
        if "cuda_rng" in transaction and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(transaction["cuda_rng"])

    def _cleanup_failed_update(self) -> None:
        """移除失败反向传播留下的梯度、图模板和 CUDA 缓存引用。"""
        try:
            self.optimizer.zero_grad(set_to_none=True)
        except TypeError:
            self.optimizer.zero_grad()
        self.gpu_graph_manager.clear()
    def ensure_hetero_batch_vectors(self, batch: HeteroData, batch_size: Optional[int] = None) -> int:
        """确保 task/station/worker 都带有 PyG batch 向量。"""
        if batch_size is None:
            if hasattr(batch, 'y_task'):
                batch_size = int(batch.y_task.view(-1).numel())
            elif hasattr(batch['task'], 'ptr'):
                batch_size = int(batch['task'].ptr.numel() - 1)
            else:
                batch_size = 1
        if batch_size <= 0:
            raise ValueError("HeteroData batch_size 必须大于 0")

        repaired = 0
        node_types = ['task', 'station', 'worker']
        if 'skill' in batch.node_types:
            node_types.append('skill')
        for node_type in node_types:
            storage = batch[node_type]
            if not hasattr(storage, 'x') or storage.x is None:
                continue
            total_nodes = int(storage.x.size(0))
            if hasattr(storage, 'batch') and storage.batch is not None and storage.batch.numel() == total_nodes:
                storage.batch = storage.batch.to(self.device, dtype=torch.long)
                continue
            if hasattr(storage, 'ptr') and storage.ptr is not None and storage.ptr.numel() == batch_size + 1:
                counts = (storage.ptr[1:] - storage.ptr[:-1]).to(self.device, dtype=torch.long)
                storage.batch = torch.repeat_interleave(
                    torch.arange(batch_size, device=self.device, dtype=torch.long),
                    counts,
                )
            else:
                if total_nodes % batch_size != 0:
                    raise ValueError(
                        f"无法为 {node_type} 推断 batch 向量: total_nodes={total_nodes}, batch_size={batch_size}"
                    )
                nodes_per_graph = total_nodes // batch_size
                storage.batch = torch.arange(batch_size, device=self.device, dtype=torch.long).repeat_interleave(nodes_per_graph)
            repaired += 1
        return repaired

    def compute_static_worker_constraint_mask(
        self,
        batch: HeteroData,
        *,
        selected_task: torch.Tensor,
        selected_station: torch.Tensor,
        max_workers: int,
    ) -> torch.Tensor:
        """按 sparse batch 索引计算 worker 技能与工位锁定约束。

        输出形状: [B, Max_W]，True 表示该 worker 在当前动作下不可选。
        该逻辑只依赖原始图特征和动作目标，不依赖网络编码结果，因此可避开
        PPO 内层中对 `batch['task'].x` 与 `batch['worker'].x` 的重复 dense 化。
        """
        selected_task = selected_task.view(-1).to(self.device, dtype=torch.long)
        selected_station = selected_station.view(-1).to(self.device, dtype=torch.long)
        batch_size = int(selected_task.numel())
        if batch_size <= 0:
            raise ValueError("selected_task 不能为空")
        if int(max_workers) <= 0:
            raise ValueError("max_workers 必须大于 0")

        task_x = batch["task"].x.to(self.device)
        worker_x = batch["worker"].x.to(self.device)
        worker_layout = resolve_worker_feature_layout(self.config)
        num_skill_types = worker_layout.num_skill_types
        task_skill_end = 5 + num_skill_types
        if task_x.size(1) < task_skill_end:
            raise ValueError(f"任务特征维度不足: {task_x.size(1)} < {task_skill_end}")
        if worker_x.size(1) != worker_layout.total_dim:
            raise ValueError(
                f"工人特征维度错误: {worker_x.size(1)} != {worker_layout.total_dim}"
            )
        task_batch = batch["task"].batch.to(self.device, dtype=torch.long)
        worker_batch = batch["worker"].batch.to(self.device, dtype=torch.long)

        if hasattr(batch["task"], "ptr") and batch["task"].ptr is not None:
            task_ptr = batch["task"].ptr.to(self.device, dtype=torch.long)
            selected_task_global = task_ptr[:-1] + selected_task
        else:
            selected_task_global = torch.empty(batch_size, device=self.device, dtype=torch.long)
            for batch_id in range(batch_size):
                task_nodes = torch.nonzero(task_batch == batch_id, as_tuple=False).view(-1)
                selected_task_global[batch_id] = task_nodes[selected_task[batch_id]]

        selected_task_raw = task_x[selected_task_global]
        task_type_idx = torch.argmax(selected_task_raw[:, 5:task_skill_end], dim=1)

        if hasattr(batch["worker"], "ptr") and batch["worker"].ptr is not None:
            worker_ptr = batch["worker"].ptr.to(self.device, dtype=torch.long)
            worker_local_idx = torch.arange(worker_x.size(0), device=self.device) - worker_ptr[worker_batch]
        else:
            worker_local_idx = torch.empty(worker_x.size(0), device=self.device, dtype=torch.long)
            for batch_id in range(batch_size):
                worker_nodes = torch.nonzero(worker_batch == batch_id, as_tuple=False).view(-1)
                worker_local_idx[worker_nodes] = torch.arange(worker_nodes.numel(), device=self.device)

        if worker_local_idx.numel() > 0 and int(worker_local_idx.max().item()) >= int(max_workers):
            raise ValueError("max_workers 小于 batch 中实际 worker 数")

        worker_skill_idx = task_type_idx[worker_batch]
        row_idx = torch.arange(worker_x.size(0), device=self.device, dtype=torch.long)
        has_skill = worker_x[:, worker_layout.skill_slice][row_idx, worker_skill_idx] > 0.5
        skill_mask_flat = ~has_skill

        worker_locks = torch.argmax(worker_x[:, worker_layout.lock_slice], dim=1)
        station_action = selected_station + 1
        lock_mask_flat = (worker_locks != 0) & (worker_locks != station_action[worker_batch])

        static_mask = torch.ones(batch_size, int(max_workers), device=self.device, dtype=torch.bool)
        static_mask[worker_batch, worker_local_idx] = skill_mask_flat | lock_mask_flat
        return static_mask

    @staticmethod
    def compute_gae_returns(
        rewards: List[float],
        terminals: List[bool],
        values: List[Any],
        gamma: float,
        gae_lambda: float,
        truncated: Optional[List[bool]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        璁＄畻 GAE 浼樺娍鍜?Critic 鐩爣鍥炴姤銆?

        truncated 琛ㄧず rollout 杈圭晫锛涢亣鍒?terminal 鎴?truncated 閮介噸缃€掓帹锛?
        闃叉澶氫釜 APAL 鐜杞ㄨ抗鎷兼帴鍚庝紭鍔夸及璁¤法杈圭晫娉勬紡銆?
        """
        if truncated is None or len(truncated) != len(rewards):
            truncated = [False] * len(rewards)

        value_tensor = torch.as_tensor(values, dtype=torch.float32)
        advantages: List[float] = []
        gae = 0.0
        next_value = 0.0

        for step in reversed(range(len(rewards))):
            if terminals[step] or truncated[step]:
                next_value = 0.0
                gae = 0.0

            delta = float(rewards[step]) + gamma * float(next_value) - float(value_tensor[step])
            gae = delta + gamma * gae_lambda * gae
            advantages.insert(0, gae)
            next_value = float(value_tensor[step])

        advantage_tensor = torch.tensor(advantages, dtype=torch.float32)
        return_tensor = advantage_tensor + value_tensor
        return advantage_tensor, return_tensor

    @staticmethod
    def compute_stable_log_ratio_and_ratio(
        current_logprob: torch.Tensor,
        old_logprob: torch.Tensor,
        clamp_abs: float = 20.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        璁＄畻 PPO log_ratio 涓庢暟鍊煎畨鍏?ratio銆?

        涓嶄慨鏀圭湡瀹?current_logprob锛屽彧鍦?exp 鍓嶈鍓?log_ratio锛岄伩鍏嶆瀬绔鐜囨瘮婧㈠嚭銆?
        """
        log_ratio = current_logprob - old_logprob
        safe_log_ratio = torch.clamp(log_ratio, min=-clamp_abs, max=clamp_abs)
        ratio = torch.exp(safe_log_ratio)
        return log_ratio, safe_log_ratio, ratio

    @staticmethod
    def compute_value_loss(
        state_values: torch.Tensor,
        returns: torch.Tensor,
        old_values: Optional[torch.Tensor] = None,
        clip_range: float = 0.2,
    ) -> torch.Tensor:
        """
        璁＄畻 PPO value clipping loss銆?

        褰?old_values 缂哄け鏃跺洖閫€涓烘櫘閫?MSE锛屼繚鎸佸巻鍙?Memory 鍏煎銆?
        """
        value_losses = (state_values - returns).pow(2)
        if old_values is None:
            return value_losses.mean()

        old_values = old_values.to(device=state_values.device, dtype=state_values.dtype)
        clipped_values = old_values + torch.clamp(state_values - old_values, -clip_range, clip_range)
        clipped_losses = (clipped_values - returns).pow(2)
        return torch.max(value_losses, clipped_losses).mean()

    def select_action(
        self,
        obs: HeteroData,
        mask_task: Optional[torch.Tensor] = None,
        mask_station_matrix: Optional[torch.Tensor] = None,
        mask_worker: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        temperature: float = 1.0,
        is_eval: bool = False,
        *,
        compute_value: bool = True,
        manage_optimizer_mode: bool = True,
    ) -> Tuple[Optional[Tuple[int, int, List[int]]], float, float, Optional[torch.Tensor], bool]:
        """
        閫夋嫨鍔ㄤ綔 (Select Action)銆?
        
        Args:
            obs: 寮傛瀯鍥捐娴嬫暟鎹?(HeteroData)
            mask_task: [N] Bool Tensor, True=Invalid
            mask_station_matrix: [N, S] Bool Tensor, True=Invalid
            mask_worker: [W] Bool Tensor, True=Invalid (Global)
            deterministic: 鏄惁纭畾鎬ч€夋嫨 (ArgMax vs Sampling)
            temperature: 閲囨牱娓╁害锛孴瓒婂皬瓒婅椽濠紝T瓒婂ぇ瓒婇殢鏈猴紝蹇界暐褰?deterministic=True 鏃?
            
        Returns:
            action_tuple: (task_id, station_id, team_indices_list)
            action_logprob: float
            state_value: float
            specific_station_mask: 鐢ㄤ簬 Memory 璁板綍
        """
        self.last_gated_team_trace = None
        no_mask = self.config.ablation_no_mask
        
        if self.use_schedule_free and manage_optimizer_mode:
            if is_eval:
                self.optimizer.eval()
            else:
                self.optimizer.train()
        
        # 鍐冲畾褰撳墠婵€娲荤殑澶ц剳 (Eval+EMA 鏃朵娇鐢ㄥ奖瀛愮綉缁?
        if is_eval and getattr(self, 'use_ema', False) and hasattr(self, 'ema_policy'):
            active_policy = self.ema_policy
        else:
            active_policy = self.policy

        inference_context = torch.inference_mode if is_eval else torch.no_grad
        with inference_context():
            with self.autocast_context():
                x_dict, global_context = active_policy(obs)
                
                # 浣跨敤瀹夊叏鐨勫父閲?-1e4锛堥槻姝?FP16 涓?-1e9/finfo 鏋佸皬鍊煎湪 autocast 鏈熼棿婧㈠嚭锛?
                mask_value = -1e4
                
                # ------------------
                # 1. 閫夋嫨宸ュ簭 (Select Task)
                # ------------------
                task_logits = active_policy.task_head(x_dict['task'], global_context, mask=mask_task if not no_mask else None)
            
            # [Robustness] 妫€鏌ュ苟澶勭悊 NaN
            if torch.isnan(task_logits).any():
                task_logits = torch.nan_to_num(task_logits, nan=mask_value)
            
            if deterministic:
                if mask_task is not None and not no_mask:
                    task_logits = task_logits.masked_fill(mask_task, mask_value)
                task_action = torch.argmax(task_logits)
                task_logprob = torch.tensor(0.0).to(self.device)
            else:
                if mask_task is not None and not no_mask:
                     task_logits = task_logits.masked_fill(mask_task, mask_value)
                
                # Check for all -inf
                if (task_logits <= mask_value * 0.99).all():
                     print("WARNING: All Task Logits -inf in select_action. Force picking 0.")
                     task_action = torch.tensor(0).to(self.device)
                     task_logprob = torch.tensor(0.0).to(self.device)
                else:
                    if temperature != 1.0:
                        task_logits = task_logits / max(temperature, 1e-5)
                    task_dist = Categorical(logits=task_logits.float())
                    task_action = task_dist.sample()
                    task_logprob = task_dist.log_prob(task_action)
            
            t_idx = task_action.item()
            selected_task_emb = x_dict['task'][t_idx].unsqueeze(0) # [1, H]
            
            # 鑾峰彇浠诲姟鐨勪汉鏁伴渶姹?
            demand = self.get_task_demand(obs['task'].x, t_idx)
            action_scope = str(
                getattr(self.config, "policy_action_scope", "operation_station_worker")
            )
            is_virtual_task = not bool(
                (obs['task'].x[t_idx, 5 : 5 + int(self.config.num_skill_types)] > 0.5).any()
            )
            if is_virtual_task:
                if compute_value:
                    with self.autocast_context():
                        state_value = active_policy.get_value(obs, actor_x_dict_encoded=x_dict)
                else:
                    state_value = torch.zeros((), device=self.device)
                return (
                    (t_idx, -1, []),
                    float(task_logprob.item()),
                    float(state_value.item()),
                    None,
                    False,
                )

            if action_scope == "operation":
                task_station_mask = (
                    mask_station_matrix[t_idx]
                    if mask_station_matrix is not None
                    else None
                )
                completed = self.action_completer.complete(
                    obs,
                    task_id=t_idx,
                    station_mask=task_station_mask,
                    worker_mask=mask_worker,
                )
                if completed is None:
                    return None, 0.0, 0.0, None, True
                if compute_value:
                    with self.autocast_context():
                        state_value = active_policy.get_value(
                            obs, actor_x_dict_encoded=x_dict
                        )
                else:
                    state_value = torch.zeros((), device=self.device)
                return (
                    (t_idx, completed.station_id, list(completed.team)),
                    float(task_logprob.item()),
                    float(state_value.item()),
                    None,
                    False,
                )
            
            # ------------------
            # 2. 閫夋嫨绔欎綅 (Select Station)
            # ------------------
            specific_station_mask = None
            if mask_station_matrix is not None:
                # [N, S] -> [1, S]
                specific_station_mask = mask_station_matrix[t_idx].unsqueeze(0)
            
            with self.autocast_context():
                station_embs = x_dict['station'].unsqueeze(0)
                station_logits = active_policy.station_head(selected_task_emb, station_embs, mask=specific_station_mask if not no_mask else None)
            
            if torch.isnan(station_logits).any():
                station_logits = torch.nan_to_num(station_logits, nan=mask_value)
            
            if deterministic:
                if specific_station_mask is not None and not no_mask:
                     station_logits = station_logits.masked_fill(specific_station_mask, mask_value)
                station_action = torch.argmax(station_logits)
                station_logprob = torch.tensor(0.0).to(self.device)
            else:
                if specific_station_mask is not None and not no_mask:
                     station_logits = station_logits.masked_fill(specific_station_mask, mask_value)
                
                if (station_logits <= mask_value * 0.99).all():
                     print("WARNING: All Station Logits -inf. Force picking 0.")
                     station_action = torch.tensor(0).to(self.device)
                     station_logprob = torch.tensor(0.0).to(self.device)
                else:
                    if temperature != 1.0:
                        station_logits = station_logits / max(temperature, 1e-5)
                    station_dist = Categorical(logits=station_logits.float())
                    station_action = station_dist.sample()
                    station_logprob = station_dist.log_prob(station_action)

            if action_scope in {"operation_station", "operation_station_gated_team"}:
                selected_station_id = int(station_action.item())
                team_logprob = torch.zeros((), device=self.device)
                if action_scope == "operation_station_gated_team":
                    gated_team = self._select_gated_team(
                        active_policy,
                        obs=obs,
                        task_id=t_idx,
                        station_id=selected_station_id,
                        worker_mask=mask_worker,
                        task_emb=selected_task_emb,
                        station_emb=x_dict["station"][selected_station_id].unsqueeze(0),
                        worker_embs=x_dict["worker"],
                        deterministic=deterministic,
                        temperature=temperature,
                    )
                    if gated_team is None:
                        return None, 0.0, 0.0, None, True
                    selected_team, team_logprob, gated_trace = gated_team
                    self.last_gated_team_trace = gated_trace
                    action_station_id = selected_station_id
                else:
                    completed = self.action_completer.complete(
                        obs,
                        task_id=t_idx,
                        station_mask=(
                            specific_station_mask.reshape(-1)
                            if specific_station_mask is not None
                            else None
                        ),
                        worker_mask=mask_worker,
                        selected_station=selected_station_id,
                    )
                    if completed is None:
                        return None, 0.0, 0.0, None, True
                    selected_team = list(completed.team)
                    action_station_id = completed.station_id
                if compute_value:
                    with self.autocast_context():
                        state_value = active_policy.get_value(
                            obs, actor_x_dict_encoded=x_dict
                        )
                else:
                    state_value = torch.zeros((), device=self.device)
                return (
                    (t_idx, action_station_id, selected_team),
                    float((task_logprob + station_logprob + team_logprob).item()),
                    float(state_value.item()),
                    specific_station_mask,
                    False,
                )
                
            # ------------------
            # 3. 閫夋嫨宸ヤ汉 (Select Workers) - 鑷洖褰?
            # ------------------
            team_indices = []
            worker_logprobs = []
            
            # 鍔ㄦ€?Mask: 鍒濆 Mask + 鎶€鑳?Mask
            current_worker_mask = mask_worker.clone() if mask_worker is not None else torch.zeros(obs['worker'].num_nodes, dtype=torch.bool).to(self.device)
            
            worker_feats = obs['worker'].x
            worker_layout = resolve_worker_feature_layout(self.config)
            if worker_feats.size(1) != worker_layout.total_dim:
                raise ValueError(
                    f"工人特征维度错误: {worker_feats.size(1)} != {worker_layout.total_dim}"
                )
            task_skill_end = 5 + worker_layout.num_skill_types
            if obs['task'].x.size(1) < task_skill_end:
                raise ValueError(f"任务特征维度不足: {obs['task'].x.size(1)} < {task_skill_end}")
            worker_skills = worker_feats[:, worker_layout.skill_slice]
            
            task_type_idx = torch.argmax(obs['task'].x[t_idx, 5:task_skill_end]).item()
            
            has_skill = worker_skills[:, task_type_idx] > 0.5
            skill_mask = ~has_skill 
            
            s_act = station_action.item() + 1
            worker_locks = torch.argmax(worker_feats[:, worker_layout.lock_slice], dim=1)
            lock_mask = (worker_locks != 0) & (worker_locks != s_act)
            
            if no_mask:
                current_worker_mask = skill_mask.to(self.device)
            else:
                current_worker_mask = current_worker_mask | skill_mask.to(self.device) | lock_mask.to(self.device)

            worker_embs = x_dict['worker'].unsqueeze(0)
            
            # 鍔犲叆杩唬闃堝€煎拰 Fallback 闃叉鍥犳帺鐮佽繃搴﹂噸鍙犲彂鐢熸寰幆
            max_iter = demand * 2
            iter_cnt = 0
            
            # 鍒濆鍖栧洟闃熻蹇?
            current_team_emb = None 
            
            while len(team_indices) < demand and iter_cnt < max_iter:
                iter_cnt += 1
                
                # 杩樻湁鍙€夊伐浜哄悧?
                if current_worker_mask.all():
                    break
                
                with self.autocast_context():
                    worker_logits = active_policy.worker_head.forward_choice(selected_task_emb, worker_embs, mask=current_worker_mask, current_team_emb=current_team_emb)
                
                if torch.isnan(worker_logits).any():
                    worker_logits = torch.nan_to_num(worker_logits, nan=mask_value)
                
                if deterministic:
                     if not no_mask: worker_logits = worker_logits.masked_fill(current_worker_mask, mask_value)
                     if (worker_logits <= mask_value * 0.99).all(): break
                     
                     w_action = torch.argmax(worker_logits)
                     w_lp = torch.tensor(0.0).to(self.device)
                else:
                     if not no_mask: worker_logits = worker_logits.masked_fill(current_worker_mask, mask_value)
                     
                     if (worker_logits <= mask_value * 0.99).all():
                         break # 鏃犳硶缁х画閫変汉
                     
                     if temperature != 1.0:
                         worker_logits = worker_logits / max(temperature, 1e-5)
                         
                     w_dist = Categorical(logits=worker_logits)
                     w_action = w_dist.sample()
                     w_lp = w_dist.log_prob(w_action)
                
                w_idx = w_action.item()
                team_indices.append(w_idx)
                worker_logprobs.append(w_lp)
                
                # 鍒锋柊宸查€夊洟闃熻〃寰佽蹇?
                selected_worker_feats = worker_embs[0, team_indices, :]
                current_team_emb = selected_worker_feats.mean(dim=0, keepdim=True) # [1, H]
                
                # 鏇存柊 Mask (閫夎繃鐨勪汉涓嶈兘鍐嶉€?
                current_worker_mask = current_worker_mask.clone() # 纭繚涓?鍘熷湴淇敼 褰卞搷涓嬩竴杞?
                current_worker_mask[w_idx] = True
            
            # [鍏滃簳閫昏緫] 鑻ュ洜杩囧害绔炰簤鎴栨閿侀€変笉澶熶汉閫?
            if len(team_indices) < demand:
                if is_eval:
                    # [Evaluation Strict Mode] 楠岃瘉鏈熼棿缁濆涓嶅厑璁稿厹搴曚綔寮婏紒
                    # 濡傛灉閫変笉澶熶汉锛岃鏄庣瓥鐣ュ嚭鐜版柇灞傛閿侊紝鐩存帴灏嗗け璐ヤ笂浼犱互鏂藉姞鐪熷疄鐨勯獙璇侀泦鎯╃綒銆?
                    return None, 0.0, 0.0, None, True
                    
                # [Zero-Fallback Enforcement] 鍘熸湁鐨勫厹搴曟満鍒跺凡琚交搴曠Щ闄ゃ€?
                # 鐢变簬鐜鐨?get_masks() 宸茬粡鍦ㄧ墿鐞嗗拰鎷撴墤灞傞潰涓婁繚璇佷簡鍙湁褰撴弧瓒?demand 浜烘暟锛堜笖鎶€鑳姐€佸伐浣嶉攣瀹氱姸鎬侀兘绗﹀悎瑕佹眰锛夋椂锛?
                # 绔欎綅鍜屼换鍔℃墠鏄悎娉曠殑銆傚鏋滃湪杩欓噷閫変笉鍑鸿冻澶熺殑浜猴紝璇存槑鍓嶇疆鎺╃爜涓庡唴灞傞€変汉鎺╃爜瀛樺湪閫昏緫鑴辫妭锛屾垨鍑虹幇浜嗘湭鐭ョ殑璁＄畻婕忔礊銆?
                # 姝ゆ椂缁濅笉鍙啀鍑戞暟濉炲叆鍋囦汉鎴栧瓨鍏ュ亣姒傜巼锛岃繖浼氬鑷村悗鏈?update 浜х敓鐖嗙偢鐨勮櫄鍋?KL 骞惰鍙戜竴杩炰覆鐨勫穿婧冿紒
                raise RuntimeError(
                    f"FATAL DEADLOCK: Failed to select enough valid workers (needed {demand}, got {len(team_indices)}).\n"
                    f"The masking logic in environment get_masks() strictly guarantees worker sufficiency.\n"
                    f"No manual fallback is ever allowed to preserve the KL purity. Please inspect the mask consistency!"
                )
            
            
            total_worker_logprob = sum(worker_logprobs) if worker_logprobs else torch.tensor(0.0).to(self.device)
            
            action_logprob = task_logprob + station_logprob + total_worker_logprob
            # 鐗╃悊闅旂 Critic 闃叉鍏跺法澶х殑 Value Error 姊害鎹ｆ瘉搴曞眰鍏变韩 GAT 鎷撴墤鐗瑰緛
            # 浼犲叆瀹屾暣鐨?state (batch_data)锛岀敱浜庡浜?with torch.no_grad() 涓嬶紝姝ゅ鏃犻渶 detach锛岀洿鎺ュ墠鍚戞彁鍙栦环鍊笺€?
            if compute_value:
                with self.autocast_context():
                    state_value = active_policy.get_value(obs, actor_x_dict_encoded=x_dict)
            else:
                state_value = torch.zeros((), device=self.device)
            
            action_tuple = (t_idx, station_action.item(), team_indices)
            
            # 妫€鏌?action 瀵逛簬 soft penalty 鐨勬湁鏁堟€?
            is_invalid_action = False
            if mask_task is not None and mask_task[t_idx].item():
                is_invalid_action = True
            if specific_station_mask is not None and specific_station_mask[0, station_action.item()].item():
                is_invalid_action = True
            if mask_worker is not None:
                for w_idx in team_indices:
                    if mask_worker[w_idx].item():
                        is_invalid_action = True
            
        return action_tuple, action_logprob.item(), state_value.item(), specific_station_mask, is_invalid_action

    def select_actions_batch(
        self,
        obs_list: List[HeteroData],
        mask_task_list: List[Optional[torch.Tensor]],
        mask_station_matrix_list: List[Optional[torch.Tensor]],
        mask_worker_list: List[Optional[torch.Tensor]],
        deterministic: bool = False,
        temperature: float = 1.0,
        is_eval: bool = False,
        *,
        profile_breakdown: bool = False,
    ) -> List[Tuple[Tuple[int, int, List[int]], float, float, Optional[torch.Tensor], bool]]:
        """
        鎵归噺閫夋嫨鍔ㄤ綔 (Batch Select Action)銆?
        
        Args:
            obs_list: 寮傛瀯鍥捐娴嬫暟鎹垪琛?(List[HeteroData])
            mask_task_list: 姣忎釜鐜瀵瑰簲鐨?[N] Bool Tensor, True=Invalid
            mask_station_matrix_list: 姣忎釜鐜瀵瑰簲鐨?[N, S] Bool Tensor, True=Invalid
            mask_worker_list: 姣忎釜鐜瀵瑰簲鐨?[W] Bool Tensor, True=Invalid (Global)
            deterministic: 鏄惁纭畾鎬ч€夋嫨
            temperature: 閲囨牱娓╁害
            is_eval: 鏄惁鏄瘎浼版ā寮?
            
        Returns:
            results: List of Tuples containing (action_tuple, action_logprob, state_value, specific_station_mask, is_invalid_action)
        """
        no_mask = self.config.ablation_no_mask
        mask_value = -1e4
        
        if self.use_schedule_free:
            if is_eval:
                self.optimizer.eval()
            else:
                self.optimizer.train()
        
        # 鍐冲畾褰撳墠婵€娲荤殑澶ц剳 (Eval+EMA 鏃朵娇鐢ㄥ奖瀛愮綉缁?
        if is_eval and getattr(self, 'use_ema', False) and hasattr(self, 'ema_policy'):
            active_policy = self.ema_policy
        else:
            active_policy = self.policy

        batch_size = len(obs_list)
        if len(mask_task_list) != batch_size:
            raise ValueError("obs_list 与动作掩码批次数量不一致")
        results = []
        state_value_tensors = []
        decoded_actions = []
        task_mask_refs = []
        worker_mask_refs = []
        eval_fail_flags = []
        gated_team_traces: list[FrozenGatedTeamTrace | None] = [None] * batch_size

        profile: Dict[str, float] = {}

        def _profile_sync() -> None:
            if profile_breakdown and self.amp_device_type == "cuda":
                torch.cuda.synchronize(self.device)

        with torch.no_grad():
            # 1. 鎵归噺鎵撳寘寮傛瀯鍥捐娴嬫暟鎹苟閫佸叆 GPU锛屼互 O(1) 澶嶆潅搴﹁繍琛?GNN 缂栫爜鍜?Critic 浠峰€肩綉缁?
            if profile_breakdown:
                _profile_sync()
                stage_started = time.perf_counter()
            batch_obs = Batch.from_data_list(obs_list)
            task_ptr = batch_obs['task'].ptr.tolist()
            station_ptr = batch_obs['station'].ptr.tolist()
            worker_ptr = batch_obs['worker'].ptr.tolist()
            if profile_breakdown:
                _profile_sync()
                profile["batch_build_ms"] = (time.perf_counter() - stage_started) * 1000.0

            if profile_breakdown:
                stage_started = time.perf_counter()
            batch_obs = batch_obs.to(self.device)
            if profile_breakdown:
                _profile_sync()
                profile["h2d_ms"] = (time.perf_counter() - stage_started) * 1000.0

            with self.autocast_context():
                if profile_breakdown:
                    stage_started = time.perf_counter()
                x_dict_batch, global_context_batch = active_policy(batch_obs)
                if profile_breakdown:
                    _profile_sync()
                    profile["actor_encoder_ms"] = (time.perf_counter() - stage_started) * 1000.0

                if profile_breakdown:
                    stage_started = time.perf_counter()
                state_values_batch = active_policy.get_value(batch_obs, actor_x_dict_encoded=x_dict_batch)
                if profile_breakdown:
                    _profile_sync()
                    profile["critic_encoder_ms"] = (time.perf_counter() - stage_started) * 1000.0

            # 2. 閫愪釜鎻愬彇鍚勭幆澧冪殑灞€閮ㄥ瓙鍥剧壒寰侊紝鍦ㄤ富杩涚▼鎵ц杞婚噺鐨?Pointer Head 鑷洖褰掑姩浣滈€夋嫨
            if profile_breakdown:
                stage_started = time.perf_counter()
            for i in range(batch_size):
                # 渚濋潬 PyG 鐨?.batch 绱㈠紩瀵瑰瓙鍥剧壒寰佽繘琛屽眬閮ㄥ垏鍒?
                task_start, task_end = task_ptr[i], task_ptr[i + 1]
                station_start, station_end = station_ptr[i], station_ptr[i + 1]
                worker_start, worker_end = worker_ptr[i], worker_ptr[i + 1]

                task_embs = x_dict_batch['task'][task_start:task_end]
                station_embs = x_dict_batch['station'][station_start:station_end]
                worker_embs = x_dict_batch['worker'][worker_start:worker_end]
                
                global_context_i = global_context_batch[i].unsqueeze(0)
                
                # ------------------
                # 2.1 閫夋嫨宸ュ簭 (Select Task)
                # ------------------
                m_task = mask_task_list[i].to(self.device) if mask_task_list[i] is not None else None
                with self.autocast_context():
                    task_logits = active_policy.task_head(task_embs, global_context_i, mask=m_task if not no_mask else None)
                
                if torch.isnan(task_logits).any():
                    task_logits = torch.nan_to_num(task_logits, nan=mask_value)
                
                if deterministic:
                    if m_task is not None and not no_mask:
                        task_logits = task_logits.masked_fill(m_task, mask_value)
                    task_action = torch.argmax(task_logits)
                    task_logprob = torch.tensor(0.0).to(self.device)
                else:
                    if m_task is not None and not no_mask:
                         task_logits = task_logits.masked_fill(m_task, mask_value)
                    
                    if (task_logits <= mask_value * 0.99).all():
                         print(f"WARNING: All Task Logits -inf in select_actions_batch for env {i}. Force picking 0.")
                         task_action = torch.tensor(0).to(self.device)
                         task_logprob = torch.tensor(0.0).to(self.device)
                    else:
                        if temperature != 1.0:
                            task_logits = task_logits / max(temperature, 1e-5)
                        task_dist = Categorical(logits=task_logits.float())
                        task_action = task_dist.sample()
                        task_logprob = task_dist.log_prob(task_action)
                
                t_idx = task_action.item()
                selected_task_emb = task_embs[t_idx].unsqueeze(0) # [1, H]
                
                # 鑾峰彇璇ヤ换鍔＄殑宸ヤ汉浜烘暟闇€姹?
                source_task_x = obs_list[i]['task'].x
                demand = self.get_task_demand(source_task_x, t_idx)
                action_scope = str(
                    getattr(self.config, "policy_action_scope", "operation_station_worker")
                )
                is_virtual_task = not bool(
                    (
                        source_task_x[
                            t_idx,
                            5 : 5 + int(self.config.num_skill_types),
                        ]
                        > 0.5
                    ).any()
                )
                if is_virtual_task:
                    state_value_tensors.append(state_values_batch[i])
                    decoded_actions.append((t_idx, 0, [], task_logprob, None))
                    task_mask_refs.append(m_task)
                    worker_mask_refs.append(None)
                    eval_fail_flags.append(False)
                    continue

                if action_scope == "operation":
                    raw_station_mask = mask_station_matrix_list[i]
                    completed = self.action_completer.complete(
                        obs_list[i],
                        task_id=t_idx,
                        station_mask=(
                            raw_station_mask[t_idx]
                            if raw_station_mask is not None
                            else None
                        ),
                        worker_mask=mask_worker_list[i],
                    )
                    if completed is None:
                        state_value_tensors.append(torch.tensor(0.0, device=self.device))
                        decoded_actions.append((0, 1, [], torch.tensor(0.0, device=self.device), None))
                        task_mask_refs.append(None)
                        worker_mask_refs.append(None)
                        eval_fail_flags.append(True)
                        continue
                    state_value_tensors.append(state_values_batch[i])
                    decoded_actions.append(
                        (
                            t_idx,
                            completed.station_id + 1,
                            list(completed.team),
                            task_logprob,
                            None,
                        )
                    )
                    task_mask_refs.append(m_task)
                    worker_mask_refs.append(mask_worker_list[i])
                    eval_fail_flags.append(False)
                    continue
                
                # ------------------
                # 2.2 閫夋嫨绔欎綅 (Select Station)
                # ------------------
                m_station = mask_station_matrix_list[i]
                specific_station_mask = None
                if m_station is not None:
                    specific_station_mask = m_station[t_idx].unsqueeze(0).to(self.device)
                
                with self.autocast_context():
                    station_embs_i = station_embs.unsqueeze(0) # [1, S, H]
                    station_logits = active_policy.station_head(selected_task_emb, station_embs_i, mask=specific_station_mask if not no_mask else None)
                
                if torch.isnan(station_logits).any():
                    station_logits = torch.nan_to_num(station_logits, nan=mask_value)
                
                if deterministic:
                    if specific_station_mask is not None and not no_mask:
                         station_logits = station_logits.masked_fill(specific_station_mask, mask_value)
                    station_action = torch.argmax(station_logits)
                    station_logprob = torch.tensor(0.0).to(self.device)
                else:
                    if specific_station_mask is not None and not no_mask:
                         station_logits = station_logits.masked_fill(specific_station_mask, mask_value)
                    
                    if (station_logits <= mask_value * 0.99).all():
                         print(f"WARNING: All Station Logits -inf in select_actions_batch for env {i}. Force picking 0.")
                         station_action = torch.tensor(0).to(self.device)
                         station_logprob = torch.tensor(0.0).to(self.device)
                    else:
                        if temperature != 1.0:
                            station_logits = station_logits / max(temperature, 1e-5)
                        station_dist = Categorical(logits=station_logits.float())
                        station_action = station_dist.sample()
                        station_logprob = station_dist.log_prob(station_action)

                if action_scope in {"operation_station", "operation_station_gated_team"}:
                    selected_station_id = int(station_action.item())
                    team_logprob = torch.zeros((), device=self.device)
                    if action_scope == "operation_station_gated_team":
                        gated_team = self._select_gated_team(
                            active_policy,
                            obs=obs_list[i],
                            task_id=t_idx,
                            station_id=selected_station_id,
                            worker_mask=mask_worker_list[i],
                            task_emb=selected_task_emb,
                            station_emb=station_embs[selected_station_id].unsqueeze(0),
                            worker_embs=worker_embs,
                            deterministic=deterministic,
                            temperature=temperature,
                        )
                        if gated_team is None:
                            state_value_tensors.append(torch.tensor(0.0, device=self.device))
                            decoded_actions.append((0, 1, [], torch.tensor(0.0, device=self.device), None))
                            task_mask_refs.append(None)
                            worker_mask_refs.append(None)
                            eval_fail_flags.append(True)
                            continue
                        selected_team, team_logprob, gated_trace = gated_team
                        gated_team_traces[i] = gated_trace
                        action_station_id = selected_station_id
                    else:
                        completed = self.action_completer.complete(
                            obs_list[i],
                            task_id=t_idx,
                            station_mask=(
                                specific_station_mask.reshape(-1)
                                if specific_station_mask is not None
                                else None
                            ),
                            worker_mask=mask_worker_list[i],
                            selected_station=selected_station_id,
                        )
                        if completed is None:
                            state_value_tensors.append(torch.tensor(0.0, device=self.device))
                            decoded_actions.append((0, 1, [], torch.tensor(0.0, device=self.device), None))
                            task_mask_refs.append(None)
                            worker_mask_refs.append(None)
                            eval_fail_flags.append(True)
                            continue
                        selected_team = list(completed.team)
                        action_station_id = completed.station_id
                    state_value_tensors.append(state_values_batch[i])
                    decoded_actions.append(
                        (
                            t_idx,
                            action_station_id + 1,
                            selected_team,
                            task_logprob + station_logprob + team_logprob,
                            specific_station_mask,
                        )
                    )
                    task_mask_refs.append(m_task)
                    worker_mask_refs.append(mask_worker_list[i])
                    eval_fail_flags.append(False)
                    continue
                
                # ------------------
                # 2.3 閫夋嫨宸ヤ汉 (Select Workers) - 鑷洖褰?
                # ------------------
                team_indices = []
                worker_logprobs = []
                
                m_worker = mask_worker_list[i]
                obs_worker_num_nodes = worker_end - worker_start
                current_worker_mask = m_worker.clone().to(self.device) if m_worker is not None else torch.zeros(obs_worker_num_nodes, dtype=torch.bool).to(self.device)
                
                worker_feats = obs_list[i]['worker'].x
                worker_layout = resolve_worker_feature_layout(self.config)
                if worker_feats.size(1) != worker_layout.total_dim:
                    raise ValueError(
                        f"工人特征维度错误: {worker_feats.size(1)} != {worker_layout.total_dim}"
                    )
                task_skill_end = 5 + worker_layout.num_skill_types
                if source_task_x.size(1) < task_skill_end:
                    raise ValueError(
                        f"任务特征维度不足: {source_task_x.size(1)} < {task_skill_end}"
                    )
                worker_skills = worker_feats[:, worker_layout.skill_slice]
                
                task_type_idx = torch.argmax(source_task_x[t_idx, 5:task_skill_end]).item()
                has_skill = worker_skills[:, task_type_idx] > 0.5
                skill_mask = ~has_skill 
                
                s_act = station_action.item() + 1
                worker_locks = torch.argmax(worker_feats[:, worker_layout.lock_slice], dim=1)
                lock_mask = (worker_locks != 0) & (worker_locks != s_act)
                
                if no_mask:
                    current_worker_mask = skill_mask.to(self.device)
                else:
                    current_worker_mask = current_worker_mask | skill_mask.to(self.device) | lock_mask.to(self.device)
    
                worker_embs_i = worker_embs.unsqueeze(0) # [1, W, H]
                
                max_iter = demand * 2
                iter_cnt = 0
                current_team_emb = None 
                
                while len(team_indices) < demand and iter_cnt < max_iter:
                    iter_cnt += 1
                    
                    if current_worker_mask.all():
                        break
                    
                    with self.autocast_context():
                        worker_logits = active_policy.worker_head.forward_choice(selected_task_emb, worker_embs_i, mask=current_worker_mask, current_team_emb=current_team_emb)
                    
                    if torch.isnan(worker_logits).any():
                        worker_logits = torch.nan_to_num(worker_logits, nan=mask_value)
                    
                    if deterministic:
                         if not no_mask: worker_logits = worker_logits.masked_fill(current_worker_mask, mask_value)
                         if (worker_logits <= mask_value * 0.99).all(): break
                         
                         w_action = torch.argmax(worker_logits)
                         w_lp = torch.tensor(0.0).to(self.device)
                    else:
                         if not no_mask: worker_logits = worker_logits.masked_fill(current_worker_mask, mask_value)
                         if (worker_logits <= mask_value * 0.99).all():
                             break
                         
                         if temperature != 1.0:
                             worker_logits = worker_logits / max(temperature, 1e-5)
                             
                         w_dist = Categorical(logits=worker_logits)
                         w_action = w_dist.sample()
                         w_lp = w_dist.log_prob(w_action)
                    
                    w_idx = w_action.item()
                    team_indices.append(w_idx)
                    worker_logprobs.append(w_lp)
                    
                    selected_worker_feats = worker_embs_i[0, team_indices, :]
                    current_team_emb = selected_worker_feats.mean(dim=0, keepdim=True)
                    
                    current_worker_mask = current_worker_mask.clone()
                    current_worker_mask[w_idx] = True
                
                # 鑻ュ洜杩囧害绔炰簤鎴栨閿侀€変笉澶熷伐浜?
                if len(team_indices) < demand:
                    if is_eval:
                        state_value_tensors.append(torch.tensor(0.0, device=self.device))
                        decoded_actions.append(
                            (0, 0, [], torch.tensor(0.0, device=self.device), None)
                        )
                        task_mask_refs.append(None)
                        worker_mask_refs.append(None)
                        eval_fail_flags.append(True)
                        continue
                    raise RuntimeError(
                        f"FATAL DEADLOCK: Failed to select enough valid workers (needed {demand}, got {len(team_indices)}).\n"
                        f"Please inspect the mask consistency!"
                    )
                
                total_worker_logprob = (
                    sum(worker_logprobs)
                    if worker_logprobs
                    else torch.tensor(0.0, device=self.device)
                )
                state_value_tensors.append(state_values_batch[i])
                decoded_actions.append(
                    (
                        t_idx,
                        s_act,
                        team_indices,
                        task_logprob + station_logprob + total_worker_logprob,
                        specific_station_mask,
                    )
                )
                task_mask_refs.append(m_task)
                worker_mask_refs.append(m_worker)
                eval_fail_flags.append(False)

        state_values = (
            torch.stack(state_value_tensors)
            .detach()
            .float()
            .reshape(-1)
            .cpu()
            .tolist()
        )
        action_logprobs = (
            torch.stack([decoded[3] for decoded in decoded_actions])
            .detach()
            .float()
            .cpu()
            .tolist()
        )
        for index, decoded in enumerate(decoded_actions):
            if eval_fail_flags[index]:
                results.append((None, 0.0, 0.0, None, True))
                continue

            task_index, station_index, team_indices, _action_logprob, station_mask = decoded
            is_invalid_action = False
            task_mask = task_mask_refs[index]
            worker_mask = worker_mask_refs[index]
            if task_mask is not None and task_mask[task_index].item():
                is_invalid_action = True
            if station_mask is not None and station_mask[0, station_index - 1].item():
                is_invalid_action = True
            if worker_mask is not None:
                for worker_index in team_indices:
                    if worker_mask[worker_index].item():
                        is_invalid_action = True

            results.append(
                (
                    (task_index, station_index - 1, team_indices),
                    action_logprobs[index],
                    state_values[index],
                    station_mask,
                    is_invalid_action,
                )
            )

        if profile_breakdown:
            _profile_sync()
            profile["action_decode_ms"] = (time.perf_counter() - stage_started) * 1000.0
        self.last_action_profile = profile if profile_breakdown else {}
        self.last_gated_team_traces = gated_team_traces

        return results

    def update(self, memory: Any, env: Any = None, current_ep: int = 1) -> Dict[str, float]:
        """执行可回滚的 PPO 更新；CUDA OOM 时回滚并跳过本轮。"""
        transactional = bool(getattr(self.config, "oom_transactional_updates", True))
        skip_on_oom = bool(getattr(self.config, "skip_update_on_oom", True))
        try:
            transaction = self._capture_update_transaction() if transactional else None
        except RuntimeError as exc:
            if not self._is_cuda_oom_error(exc):
                raise
            self._cleanup_failed_update()
            if skip_on_oom:
                print(
                    "WARNING: PPO 更新在事务快照阶段发生 OOM；"
                    f"episode={current_ep}, batch_size={self.batch_size}，本轮更新已回滚并跳过。"
                )
                return {"OOM/SkippedUpdate": 1.0, "_skip_training_log": 1.0}
            raise RuntimeError(
                "PPO 更新在事务快照阶段发生 OOM，且 skip_update_on_oom=false。"
            ) from exc

        while True:
            try:
                metrics = self._update_once(memory, env, current_ep=current_ep)
                metrics["OOM/SkippedUpdate"] = 0.0
                metrics["OOM/EffectiveBatchSize"] = float(self.batch_size)
                return metrics
            except RuntimeError as exc:
                if not self._is_cuda_oom_error(exc):
                    raise

                failed_batch_size = int(self.batch_size)
                self._cleanup_failed_update()
                if transaction is not None:
                    self._restore_update_transaction(transaction)
                    self._cleanup_failed_update()

                if skip_on_oom and transaction is not None:
                    self.batch_size = failed_batch_size
                    print(
                        "WARNING: PPO 更新发生 CUDA OOM，已完整回滚并跳过；"
                        f"episode={current_ep}, batch_size 保持为 {failed_batch_size}。"
                    )
                    return {"OOM/SkippedUpdate": 1.0, "_skip_training_log": 1.0}

                raise RuntimeError(
                    "PPO 更新发生 CUDA OOM，且无法安全回滚跳过。"
                    "请启用 oom_transactional_updates，或进一步降低 PPO batch_size。"
                ) from exc

    def _update_once(self, memory: Any, env: Any = None, current_ep: int = 1) -> Dict[str, float]:
        """执行一次 PPO 更新，并返回 TensorBoard 指标。"""
        # 无引用缓存清理只能缓解碎片化；活跃张量由显式生命周期管理。
        self.clear_device_cache()
        distill_lifecycle = {
            "Distill/Enabled": 0.0,
            "Distill/TeacherReloaded": 0.0,
            "Distill/TeacherVersion": 0.0,
            "Distill/TeacherScore": 0.0,
        }
        if self.best_anchor_teacher is not None:
            distill_lifecycle = self.best_anchor_teacher.on_update_started()

        # 1. 计算广义优势估计（GAE）。
        self.validate_snapshot_homogeneity(memory.states)
        mem_rewards = memory.rewards
        mem_is_terminals = memory.is_terminals
        mem_is_truncated = getattr(memory, 'is_truncated', [False] * len(mem_rewards))
        
        # 鎻愬彇瀛樺偍鍦?states 涓殑 state_values
        # (杩欓渶瑕佸湪 select_action 涔嬪悗琚褰曚笅鏉ワ紝濡傛灉娌℃湁璁板綍锛屽洖閫€涓烘櫘閫氱殑 MC 鍥炴姤鍔犲熀绾?
        if hasattr(memory, 'values') and len(memory.values) == len(mem_rewards):
            advantages, rewards = self.compute_gae_returns(
                rewards=mem_rewards,
                terminals=mem_is_terminals,
                values=memory.values,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                truncated=mem_is_truncated,
            )
        else:
            # Fallback 鍒?Monte-Carlo + Advantage (濡傛灉缂哄皯 Value 璁板綍)
            rewards = []
            discounted_reward = 0
            for reward, is_terminal, is_trunc in zip(reversed(mem_rewards), reversed(mem_is_terminals), reversed(mem_is_truncated)):
                if is_terminal or is_trunc:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
                
            rewards = torch.tensor(rewards, dtype=torch.float32)
            # 鍏煎澶勭悊
            advantages = rewards.clone()
            
        # 褰掍竴鍖?Advantages 涓?Returns (鏈夊姪浜庨暱鏈熻礋鍙嶉鐜鐨勮缁冪ǔ瀹氭€?
        if advantages.std() > 1e-7:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        else:
            advantages = advantages - advantages.mean()
            
        # [CRITICAL FIX: Removed Return Normalization]
        # 缁濅笉搴斿 Critic 鐨?Target Returns 杩涜鍔ㄦ€佹壒娆℃爣鍑嗗寲锛?
        # 鍚﹀垯姣忎竴杞?Update 鐨勫潎鍊煎拰鏂瑰樊閮藉湪鍙橈紙绉诲姩闈讹級锛屽鑷?Critic 姘歌繙鏃犳硶鏀舵暃锛屼骇鐢熷法澶х殑姊害闇囪崱銆?
        # 鎴戜滑鏀圭敤閰嶇疆涓殑闈欐€佺郴鏁扮缉灏忓叏灞€ reward銆?
        
        # 2. 鍑嗗 Batch 鏁版嵁
        old_actions = memory.actions 
        old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32)
        action_scope = str(
            getattr(self.config, "policy_action_scope", "operation_station_worker")
        )
        if action_scope == "operation_station_gated_team":
            traces = getattr(memory, "gated_team_traces", None)
            if traces is None or len(traces) != len(memory.states):
                raise RuntimeError(
                    "门控团队 PPO 轨迹未与冻结候选序列一一对齐："
                    f"states={len(memory.states)} traces={0 if traces is None else len(traces)}"
                )
        
        # Pad Team List (鍙橀暱 -> 瀹氶暱 Tensor)
        max_team_size = max(len(a[2]) for a in old_actions) if old_actions else 1
        
        b_task = torch.tensor([a[0] for a in old_actions], dtype=torch.long)
        b_station = torch.tensor([a[1] for a in old_actions], dtype=torch.long)
        
        team_list = []
        for a in old_actions:
            t = a[2]
            pad = [-1] * (max_team_size - len(t))
            team_list.append(t + pad)
        b_team = torch.tensor(team_list, dtype=torch.long)
        
        # Attach targets to Data objects for Batching
        enable_gpu_batch = getattr(self.config, 'enable_gpu_batch_rebuild', False) and env is not None
        N_samples = len(memory.states)
        gpu_rebuild_fallback_count = 0
        gpu_rebuild_fallback_messages = []
        if enable_gpu_batch and memory.states and isinstance(memory.states[0], dict):
            self.gpu_graph_manager.retain_dataset(
                int(memory.states[0].get("dataset_idx", 0))
            )

        def _attach_update_targets(state, idx: int):
            """为单个重建图绑定 PPO 更新所需字段。"""
            state.y_task = b_task[idx].unsqueeze(0)
            state.y_station = b_station[idx].unsqueeze(0)
            state.y_team = b_team[idx].unsqueeze(0)
            state.y_logprob = old_logprobs[idx].unsqueeze(0)
            state.y_memory_index = torch.tensor([idx], dtype=torch.long)
            state.y_reward = rewards[idx].unsqueeze(0)
            state.y_advantage = advantages[idx].unsqueeze(0)
            if len(memory.values) > idx:
                state.y_value = torch.tensor([memory.values[idx]], dtype=torch.float32)
            if idx < len(memory.masks):
                t_mask, s_mask, w_mask = memory.masks[idx]
                state.y_task_mask = t_mask
                state.y_station_mask = s_mask
                state.y_worker_mask = w_mask
            return state

        def _build_cpu_rebuild_batch(batch_indices):
            """GPU 快速重建不可用时，回退到逐 snapshot 的标准 CPU rebuild。"""
            assert env is not None, "CPU rebuild fallback 需要可用的 APAL 环境实例"
            batch_data_list = []
            for raw_idx in batch_indices:
                idx = int(raw_idx)
                state = env.rebuild_state_from_snapshot(memory.states[idx])
                batch_data_list.append(_attach_update_targets(state, idx))
            return Batch.from_data_list(batch_data_list).to(self.device)

        def _bind_gpu_batch_targets(batch, batch_indices):
            """为 GPU 原地重建后的 Batch 绑定同一批 PPO 标签与 mask。"""
            batch.y_task = b_task[batch_indices].to(self.device)
            batch.y_station = b_station[batch_indices].to(self.device)
            batch.y_team = b_team[batch_indices].to(self.device)
            batch.y_logprob = old_logprobs[batch_indices].to(self.device)
            batch.y_memory_index = torch.as_tensor(
                batch_indices, dtype=torch.long, device=self.device
            )
            batch.y_reward = rewards[batch_indices].to(self.device)
            batch.y_advantage = advantages[batch_indices].to(self.device)
            if len(memory.values) > 0:
                b_vals = [memory.values[int(idx)] for idx in batch_indices]
                batch.y_value = torch.tensor(b_vals, dtype=torch.float32, device=self.device)
            if len(memory.masks) >= len(memory.states):
                b_masks = [memory.masks[int(idx)] for idx in batch_indices]
                batch.y_task_mask = torch.cat([m[0] for m in b_masks], dim=0).to(self.device)
                batch.y_station_mask = torch.cat([m[1] for m in b_masks], dim=0).to(self.device)
                batch.y_worker_mask = torch.cat([m[2] for m in b_masks], dim=0).to(self.device)
            return batch

        def _gpu_rebuild_or_cpu_fallback(snapshots_batch, batch_indices):
            """GPU fast rebuild 失败时自动降级，避免训练在 batch.y_task 处崩溃。"""
            nonlocal gpu_rebuild_fallback_count
            try:
                batch = self.gpu_graph_manager.batched_rebuild_on_gpu(snapshots_batch, env)
                if batch is None:
                    raise RuntimeError("GPUBatchGraphManager.batched_rebuild_on_gpu 杩斿洖 None")
                return _bind_gpu_batch_targets(batch, batch_indices)
            except Exception as exc:
                if self._is_cuda_oom_error(exc):
                    raise
                gpu_rebuild_fallback_count += 1
                if len(gpu_rebuild_fallback_messages) < 3:
                    gpu_rebuild_fallback_messages.append(f"{type(exc).__name__}: {exc}")
                if gpu_rebuild_fallback_count <= 3:
                    print(f"WARNING: GPU batch rebuild failed; falling back to CPU rebuild for this PPO batch. Reason: {type(exc).__name__}: {exc}")
                return _build_cpu_rebuild_batch(batch_indices)
        
        if not enable_gpu_batch:
            rebuilt_states = []
            if env is not None:
                for snap in memory.states:
                    rebuilt_states.append(env.rebuild_state_from_snapshot(snap))
            else:
                rebuilt_states = memory.states
                
            for i, state in enumerate(rebuilt_states):
                _attach_update_targets(state, i)
            
            import os
            num_workers = 0 if os.name == 'nt' else 4
            loader = DataLoader(rebuilt_states, batch_size=self.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
            num_batches = len(loader)
            print(f"PPO Update: BatchSize={self.batch_size}, Total Batches={num_batches} (CPU DataLoader)")
        else:
            num_batches = (N_samples + self.batch_size - 1) // self.batch_size
            print(f"PPO Update: BatchSize={self.batch_size}, Total Batches={num_batches} (GPU In-place Rebuild)")
        
        # 3. PPO Optimization Loop
        update_counts = 0
        approx_kls = []
        kl_exceeded_flags = []
        explained_vars = []
        ratio_means = []
        ratio_stds = []
        clip_fractions = []
        batch_vector_repair_count = 0
        total_loss_values = []
        policy_loss_values = []
        value_loss_values = []
        entropy_loss_values = []
        task_entropy_values = []
        station_entropy_values = []
        team_entropy_values = []
        ctg_ppo_gate_values = []
        ctg_ppo_alternative_probability_values = []
        ctg_ppo_alternative_gap_values = []
        ctg_ppo_alternative_beats_values = []
        distill_task_values = []
        distill_station_values = []
        distill_weighted_values = []
        distill_lambda_values = []
        
        # 鏀堕泦 batch 绾?Critic 棰勬祴鍋忓樊涓庝紭鍔垮垎甯?
        batch_pred_vals = []
        batch_target_rets = []
        batch_abs_errors = []
        batch_adv_means = []
        batch_adv_stds = []
        
        if self.use_schedule_free:
             self.optimizer.train()
        
        self.optimizer.zero_grad()
            
        final_epoch = self.k_epochs - 1
        kl_meltdown_occurred = False 
        total_batches_diagnosed = 0 
        kl_exceeded_count = 0
        for i_epoch in range(self.k_epochs):
            epoch_kls = []
            
            if enable_gpu_batch:
                shuffled_indices = np.random.permutation(N_samples)
                step_items = range(num_batches)
            else:
                step_items = loader
                
            for step_idx, item in enumerate(step_items):
                if enable_gpu_batch:
                    start_idx = item * self.batch_size
                    end_idx = min(start_idx + self.batch_size, N_samples)
                    batch_idx = shuffled_indices[start_idx:end_idx]
                    snapshots_batch = [memory.states[idx] for idx in batch_idx]
                    
                    dataset_ids = [snap.get('dataset_idx', 0) for snap in snapshots_batch]
                    worker_lens = [len(snap['worker_free_time']) for snap in snapshots_batch]
                    if len(set(dataset_ids)) > 1 or len(set(worker_lens)) > 1:
                        batch = _build_cpu_rebuild_batch(batch_idx)
                    else:
                        batch = _gpu_rebuild_or_cpu_fallback(snapshots_batch, batch_idx)

                else:
                    batch = item.to(self.device)
                    # 棰嗗煙闅忔満鍖栧悗 HeteroData 鐨?worker 鏁板彲鑳戒笉鍚岋紝
                    # PyG DataLoader 浜у嚭瑁?HeteroData 鏃剁己灏?.batch 灞炴€э紝
                    # 姝ゆ椂寮哄埗鍖呰涓?Batch 浠ョ‘淇?to_dense_batch 鍙敤
                    if not hasattr(batch['task'], 'batch'):
                        batch = Batch.from_data_list([batch]).to(self.device)
                batch_vector_repair_count += self.ensure_hetero_batch_vectors(
                    batch,
                    batch_size=int(batch.y_task.view(-1).numel()) if hasattr(batch, 'y_task') else None,
                )
                teacher_outputs = self._teacher_task_station_logits(batch)
                
                with self.autocast_context():
                    # 褰撳墠绛栫暐鐨勫墠鍚戜紶鎾?
                    x_dict, global_context = self.policy(batch)
                    
                    # 鐙珛楠ㄥ共璇勪及 state_values
                    state_values = self.policy.get_value(batch, actor_x_dict_encoded=x_dict).view(-1)
                    
                    # --- Re-evaluate LogProbs ---
                    # A. Task LogProb
                    task_x, p_mask = to_dense_batch(x_dict['task'], batch['task'].batch)
                    
                    # 鎭㈠ Mask
                    if hasattr(batch, 'y_task_mask'):
                        logical_task_mask, _ = to_dense_batch(batch.y_task_mask, batch['task'].batch)
                        combined_task_mask = logical_task_mask | (~p_mask)
                    else:
                        combined_task_mask = ~p_mask
                        
                    task_logits = self.policy.task_head(task_x, global_context, mask=combined_task_mask)
                    if torch.isnan(task_logits).any(): task_logits = torch.nan_to_num(task_logits, nan=(torch.finfo(task_logits.dtype).min / 2.0))
                    
                    task_dist = Categorical(logits=task_logits.float())
                    task_lp = task_dist.log_prob(batch.y_task)
                    task_entropy = task_dist.entropy()
                    action_scope = str(
                        getattr(
                            self.config,
                            "policy_action_scope",
                            "operation_station_worker",
                        )
                    )
                    
                    # B. Station LogProb
                    batch_indices = torch.arange(batch.y_task.size(0)).to(self.device)
                    sel_task_emb = task_x[batch_indices, batch.y_task] 
                    
                    station_x, s_p_mask = to_dense_batch(x_dict['station'], batch['station'].batch)
                    
                    if hasattr(batch, 'y_station_mask'):
                        dense_s_mask, _ = to_dense_batch(batch.y_station_mask, batch['task'].batch)
                        specific_station_mask = dense_s_mask[batch_indices, batch.y_task]
                        curr_s_mask = specific_station_mask | (~s_p_mask)
                    else:
                        curr_s_mask = ~s_p_mask
                    
                    if action_scope == "operation":
                        station_logits = torch.zeros_like(curr_s_mask, dtype=task_logits.dtype)
                        station_lp = torch.zeros_like(task_lp)
                        station_entropy = torch.zeros_like(task_entropy)
                    else:
                        station_logits = self.policy.station_head(sel_task_emb, station_x, mask=curr_s_mask)
                        if torch.isnan(station_logits).any(): station_logits = torch.nan_to_num(station_logits, nan=(torch.finfo(station_logits.dtype).min / 2.0))

                        station_dist = Categorical(logits=station_logits.float())
                        physical_action = batch.y_station >= 0
                        station_lp = station_dist.log_prob(torch.clamp(batch.y_station, min=0))
                        station_entropy = station_dist.entropy()
                        station_lp = station_lp * physical_action.to(station_lp.dtype)
                        station_entropy = station_entropy * physical_action.to(station_entropy.dtype)
                    
                    # C. Worker Team LogProb
                    worker_x, w_p_mask = to_dense_batch(x_dict['worker'], batch['worker'].batch)
                    team_lp = torch.zeros_like(task_lp)
                    team_entropy = torch.zeros_like(task_entropy)
                    
                    if hasattr(batch, 'y_worker_mask') and not getattr(self.config, 'ablation_no_mask', False):
                         d_w_mask, _ = to_dense_batch(batch.y_worker_mask.float(), batch['worker'].batch)
                         curr_mask = (d_w_mask > 0.5) | (~w_p_mask)
                    else:
                         curr_mask = (~w_p_mask)

                    if hasattr(batch, 'y_worker_mask'):
                        replay_worker_mask, _ = to_dense_batch(
                            batch.y_worker_mask.float(), batch['worker'].batch
                        )
                        replay_worker_mask = replay_worker_mask > 0.5
                    else:
                        replay_worker_mask = torch.zeros_like(w_p_mask, dtype=torch.bool)
                    
                    B_size, Max_W_size = int(worker_x.size(0)), int(worker_x.size(1))
                    static_worker_mask = self.compute_static_worker_constraint_mask(
                        batch,
                        selected_task=batch.y_task,
                        selected_station=batch.y_station,
                        max_workers=Max_W_size,
                    )
                    curr_mask = curr_mask | static_worker_mask
                    
                    current_team_emb = None # [B, H]
                    team_emb_sum = torch.zeros(B_size, worker_x.size(-1)).to(self.device)
                    team_cnt = torch.zeros(B_size, 1).to(self.device)
                    
                    worker_steps = (
                        range(batch.y_team.size(1))
                        if action_scope == "operation_station_worker"
                        else range(0)
                    )
                    for k in worker_steps:
                        target = batch.y_team[:, k] 
                        valid_step = (target != -1)
                        if not valid_step.any(): continue
                        
                        logits = self.policy.worker_head.forward_choice(sel_task_emb, worker_x, mask=curr_mask, current_team_emb=current_team_emb)
                        if torch.isnan(logits).any(): logits = torch.nan_to_num(logits, nan=(torch.finfo(logits.dtype).min / 2.0))
                        
                        dist = Categorical(logits=logits.float())
                        step_lp = dist.log_prob(torch.clamp(target, min=0)) 
                        team_lp[valid_step] += step_lp[valid_step]
                        team_entropy[valid_step] += dist.entropy()[valid_step]
                        
                        # Update current_team_emb
                        valid_b_indices = torch.nonzero(valid_step).squeeze(-1)
                        valid_targets = target[valid_step]
                        
                        selected_feats = worker_x[valid_b_indices, valid_targets]
                        
                        # 浣跨敤 clone() 淇濋殰 PyTorch 鑷姩姹傚鏈哄埗鐨勮繛缁€?(Gradient Preservation)
                        next_team_emb_sum = team_emb_sum.clone()
                        next_team_cnt = team_cnt.clone()
                        
                        next_team_emb_sum[valid_b_indices] += selected_feats
                        next_team_cnt[valid_b_indices] += 1
                        
                        team_emb_sum = next_team_emb_sum
                        team_cnt = next_team_cnt
                        
                        current_team_emb = team_emb_sum / torch.clamp(team_cnt, min=1.0)
                        
                        # Update mask for next worker in team
                        curr_mask = curr_mask.clone()
                        curr_mask[valid_b_indices, target[valid_step]] = True

                    if action_scope == "operation_station_gated_team":
                        raw_task_x, _ = to_dense_batch(
                            batch['task'].x, batch['task'].batch
                        )
                        raw_station_x, _ = to_dense_batch(
                            batch['station'].x, batch['station'].batch
                        )
                        raw_worker_x, _ = to_dense_batch(
                            batch['worker'].x, batch['worker'].batch
                        )
                        selected_station_embeddings = station_x[
                            batch_indices, torch.clamp(batch.y_station, min=0)
                        ]
                        memory_indices = batch.y_memory_index.detach().cpu().tolist()
                        frozen_traces = [
                            memory.gated_team_traces[int(memory_index)]
                            for memory_index in memory_indices
                        ]
                        team_lp, team_entropy, team_diagnostics = self._recompute_gated_team_logprobs(
                            batch=batch,
                            task_embeddings=sel_task_emb,
                            station_embeddings=selected_station_embeddings,
                            worker_embeddings=worker_x,
                            raw_task_x=raw_task_x,
                            raw_station_x=raw_station_x,
                            raw_worker_x=raw_worker_x,
                            worker_masks=replay_worker_mask,
                            selected_task=batch.y_task,
                            selected_station=batch.y_station,
                            selected_teams=batch.y_team,
                            frozen_traces=frozen_traces,
                        )
                        ctg_ppo_gate_values.append(team_diagnostics["gate"].mean().detach())
                        multi_candidate = team_diagnostics["multi"].to(
                            dtype=team_lp.dtype
                        )
                        multi_count = multi_candidate.sum().clamp_min(1.0)
                        alternative_probabilities = team_diagnostics[
                            "alternative_probability"
                        ]
                        alternative_gaps = team_diagnostics["best_alternative_gap"]
                        ctg_ppo_alternative_probability_values.append(
                            (
                                (alternative_probabilities * multi_candidate).sum()
                                / multi_count
                            ).detach()
                        )
                        ctg_ppo_alternative_gap_values.append(
                            ((alternative_gaps * multi_candidate).sum() / multi_count).detach()
                        )
                        ctg_ppo_alternative_beats_values.append(
                            (
                                ((alternative_gaps > 0.0).to(team_lp.dtype) * multi_candidate).sum()
                                / multi_count
                            ).detach()
                        )
                                
                    if action_scope == "operation":
                        total_lp = task_lp
                        entropy = task_entropy
                    elif action_scope == "operation_station":
                        total_lp = task_lp + station_lp
                        entropy = task_entropy + station_entropy
                    else:
                        total_lp = task_lp + station_lp + team_lp
                        entropy = task_entropy + station_entropy + team_entropy
                    old_lp = batch.y_logprob.view(-1)
                    log_ratio, safe_log_ratio, ratios = self.compute_stable_log_ratio_and_ratio(total_lp, old_lp)
                    
                    # --- PPO Loss Calculation ---
                    with torch.no_grad():
                        approx_kl = ((ratios - 1) - safe_log_ratio).mean()
                        epoch_kls.append(approx_kl.detach())
    
                    # 鏋佺畝 KL 鐔旀柇鏈哄埗 (Meltdown Protection)
                    hard_limit = self.kl_early_stop
                    kl_exceeded = approx_kl.detach() > hard_limit
                    loss_scale = torch.where(
                        kl_exceeded,
                        torch.tensor(0.01, device=self.device, dtype=approx_kl.dtype),
                        torch.tensor(1.0, device=self.device, dtype=approx_kl.dtype),
                    )
                    kl_exceeded_flags.append(kl_exceeded.float())
    
                    # Use GAE advantages if available, else batch.y_reward - state_values (MC fallback)
                    b_reward = batch.y_reward.view(-1)
                    b_adv = batch.y_advantage.view(-1) if hasattr(batch, 'y_advantage') else (b_reward - state_values.detach())
                    
                    # Calculate Explained Variance
                    var_y = torch.var(b_reward, correction=0)
                    if b_reward.numel() <= 1 or var_y <= 1e-8:
                        exp_var = torch.tensor(0.0, device=b_reward.device)
                    else:
                        exp_var = 1.0 - torch.var(b_reward - state_values.detach(), correction=0) / (var_y + 1e-8)
                    explained_vars.append(exp_var.detach())
                    
                    # 鍔ㄦ€佽“鍑忔帰绱笂闄?
                    progress = min(1.0, self.current_step / max(1, self.total_timesteps))
                    eps_clip_end = float(getattr(self.config, 'eps_clip_end', 0.05))
                    curr_eps_clip = self.eps_clip - progress * (self.eps_clip - eps_clip_end)
                    
                    surr1 = ratios * b_adv
                    surr2 = torch.clamp(ratios, 1-curr_eps_clip, 1+curr_eps_clip) * b_adv
                    with torch.no_grad():
                        ratio_means.append(ratios.mean().detach())
                        ratio_stds.append(ratios.std(unbiased=False).detach())
                        clip_mask = torch.abs(ratios - 1.0) > curr_eps_clip
                        clip_fractions.append(clip_mask.float().mean().detach())
                    
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    c_val = self.config.c_value
                    decay_eps = max(1, self.config.entropy_decay_episodes)
                    
                    ent_progress = min(1.0, current_ep / decay_eps)
                    
                    c_ent_base = self.config.c_entropy
                    c_ent_end = self.config.c_entropy_end
                    import math
                    c_ent = c_ent_end + (c_ent_base - c_ent_end) * math.exp(-3.0 * ent_progress)
                    
                    c_pol = self.config.c_policy
                    
                    old_values = batch.y_value.view(-1) if hasattr(batch, 'y_value') else None
                    value_loss_raw = self.compute_value_loss(
                        state_values=state_values,
                        returns=b_reward,
                        old_values=old_values,
                        clip_range=curr_eps_clip,
                    )
                    value_loss = c_val * value_loss_raw
                    
                    # 鍔ㄦ€佽幏鍙栧綋鍓?batch 涓悇鍐崇瓥鍒嗘敮鐨勬渶澶у姩浣滅淮搴︿互杩涜鐔靛綊涓€鍖?
                    max_task_dim = float(task_logits.size(-1))
                    max_station_dim = float(station_logits.size(-1))
                    max_worker_dim = float(worker_x.size(1))
                    
                    # 瀵瑰悇鍒嗘敮鐨勭喌鍒嗗埆鍦ㄥ悇鑷渶澶у姩浣滅淮搴︿笅閲囩敤瀵规暟灏哄害杩涜 [0, 1] 褰掍竴鍖?
                    norm_task_entropy = task_entropy / math.log(max(2.0, max_task_dim))
                    norm_station_entropy = station_entropy / math.log(max(2.0, max_station_dim))
                    # 鍥㈤槦宸ヤ汉鐨勬渶澶х喌浠ラ€夋嫨 2 涓伐浜轰负鍩哄簳绠椾綔 2.0 鍊嶇殑鏈€澶у伐浜虹淮鏁板鏁?
                    norm_team_entropy = team_entropy / (math.log(max(2.0, max_worker_dim)) * 2.0)
                    
                    # 璁剧珛鍒嗘敮瑙ｈ€︾郴鏁颁互鐙珛鎺у埗鍜屽钩琛″悇鍒嗘敮鎺㈢储鍔涘害
                    c_ent_task = 1.0
                    c_ent_station = 1.5
                    c_ent_team = 0.5
                    
                    avg_normalized_entropy = (
                        c_ent_task * norm_task_entropy.mean() +
                        c_ent_station * norm_station_entropy.mean() +
                        c_ent_team * norm_team_entropy.mean()
                    )
                    entropy_loss = -c_ent * avg_normalized_entropy

                    distill_task_loss = torch.zeros((), device=self.device)
                    distill_station_loss = torch.zeros((), device=self.device)
                    distill_loss = torch.zeros((), device=self.device)
                    distill_lambda = 0.0
                    if teacher_outputs is not None and self.best_anchor_teacher is not None:
                        (
                            teacher_task_logits,
                            teacher_station_logits,
                            teacher_task_mask,
                            teacher_station_mask,
                        ) = teacher_outputs
                        temperature = float(self.config.best_anchor_distill_temperature)
                        distill_task_loss = self._masked_kl(
                            teacher_task_logits,
                            task_logits,
                            teacher_task_mask,
                            temperature,
                        ).mean()
                        station_kl = self._masked_kl(
                            teacher_station_logits,
                            station_logits,
                            teacher_station_mask,
                            temperature,
                        )
                        physical_action = batch.y_station >= 0
                        if physical_action.any():
                            distill_station_loss = station_kl[physical_action].mean()
                        distill_loss = 0.5 * (distill_task_loss + distill_station_loss)
                        distill_lambda = self.best_anchor_teacher.current_lambda()

                    loss = (
                        c_pol * policy_loss
                        + value_loss
                        + entropy_loss
                        + float(distill_lambda) * distill_loss
                    )
                    raw_total_loss = loss
                    
                    # 搴旂敤杞啍鏂缉鏀?
                    loss = loss * loss_scale
                    
                    # Backprop
                    loss = loss / self.accumulation_steps # 褰掍竴鍖?Gradient
                
                # Scaled Backprop
                self.scaler.scale(loss).backward()
                
                if ((step_idx + 1) % self.accumulation_steps == 0) or (step_idx + 1 == num_batches):
                    self.scaler.unscale_(self.optimizer)
                    
                    # 鐙珛鍙傛暟姊害瑁佸壀
                    torch.nn.utils.clip_grad_norm_(self.actor_parameters, max_norm=0.5)
                    # 缁?Critic 鎸傝杩滄瘮 Actor 鏇磋杽寮辩殑瑁呯敳锛岄槻姝㈠眬閮ㄨ剦鍐插甫宕╁叏鐩?
                    torch.nn.utils.clip_grad_norm_(self.critic_parameters, max_norm=self.config.clip_v_grad_norm)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    update_counts += 1
                
                # Log Stats (璁板綍鍘熷鏈缉鏀剧殑 loss 鐢ㄤ簬璇婃柇)
                total_loss_values.append(raw_total_loss.detach())
                policy_loss_values.append(policy_loss.detach())
                value_loss_values.append(value_loss_raw.detach())
                entropy_loss_values.append(entropy.mean().detach())
                task_entropy_values.append(task_entropy.mean().detach())
                station_entropy_values.append(station_entropy.mean().detach())
                team_entropy_values.append(team_entropy.mean().detach())
                distill_task_values.append(distill_task_loss.detach())
                distill_station_values.append(distill_station_loss.detach())
                distill_weighted_values.append((float(distill_lambda) * distill_loss).detach())
                distill_lambda_values.append(torch.tensor(float(distill_lambda), device=self.device))
                total_batches_diagnosed += 1
                
                # 璁板綍 batch 绾х殑棰勬祴鍋忓樊涓庝紭鍔垮垎甯冿紝杈呭姪缁嗗寲 TensorBoard 璇婃柇
                with torch.no_grad():
                    batch_pred_vals.append(state_values.mean().detach())
                    batch_target_rets.append(b_reward.mean().detach())
                    batch_abs_errors.append(torch.abs(state_values - b_reward).mean().detach())
                    batch_adv_means.append(b_adv.mean().detach())
                    batch_adv_stds.append(b_adv.std(unbiased=False).detach())
            
            # 灏芥棭瑙﹀彂 early stopping锛岄槻姝㈤€€鍖?
            # 璁＄畻褰撳墠 epoch 鐨勫钩鍧?KL
            curr_epoch_kl = (
                float(torch.stack(epoch_kls).mean().detach().float().cpu().item())
                if epoch_kls else 0.0
            )
            
            # 鎴戜滑濮嬬粓璁板綍鏈€鍚庝竴杞湭鎺愭柇鐨?KL 浣滀负鑷€傚簲寮曟搸鐨勫弬鑰?
            approx_kls = epoch_kls
            
            # 濡傛灉鍋忕瓒呰繃纭槇鍊硷紝鎻愬墠缁堟鏈 Update 寰幆浠ヤ繚鎶ゆā鍨?
            if curr_epoch_kl > self.kl_early_stop:
                print(f"      -> Early stopping at epoch {i_epoch+1} due to reaching max KL: {curr_epoch_kl:.4f}")
                break
                
        kl_exceeded_count = (
            int(torch.stack(kl_exceeded_flags).sum().detach().float().cpu().item())
            if kl_exceeded_flags else 0
        )
        if kl_exceeded_count > 0:
            print(f"      [KL Warning] {kl_exceeded_count}/{total_batches_diagnosed} batches exceeded KL threshold {self.kl_early_stop}. (Extreme Braking Applied)")
            
        # (宸茬Щ闄ゅ啑鏉傜殑瀛︿範鐜囦笅闄嶉€昏緫锛屽畬鍏ㄤ氦缁?Schedule-Free 鎴栨亽瀹?LR)
        def _mean_scalar(values, default: float) -> float:
            if not values:
                return float(default)
            return float(torch.stack([value.detach().float() for value in values]).mean().cpu().item())

        mean_kl = _mean_scalar(approx_kls, 0.0)
        mean_ratio = _mean_scalar(ratio_means, 1.0)
        std_ratio = _mean_scalar(ratio_stds, 0.0)
        mean_clip_fraction = _mean_scalar(clip_fractions, 0.0)
        
        self.current_step += 1
        
        # [EMA 鏇存柊] 姣忎竴杞閮?Update 缁撴潫锛堝寘鎷唴閮?k_epochs锛夊悗锛岀敱涓绘ā鍨嬪悜褰卞瓙妯″瀷杩涜涓€娆?Exponential Moving Averaging 鍚屾
        if getattr(self, 'use_ema', False) and hasattr(self, 'ema_policy'):
            alpha = self.ema_decay
            with torch.no_grad():
                for ema_param, param in zip(self.ema_policy.parameters(), self.policy.parameters()):
                    ema_param.data.copy_(alpha * ema_param.data + (1.0 - alpha) * param.data)
                
        mean_exp_var = _mean_scalar(explained_vars, 0.0)
        
        mean_pred_val = _mean_scalar(batch_pred_vals, 0.0)
        mean_target_ret = _mean_scalar(batch_target_rets, 0.0)
        mean_abs_err = _mean_scalar(batch_abs_errors, 0.0)
        mean_adv = _mean_scalar(batch_adv_means, 0.0)
        std_adv = _mean_scalar(batch_adv_stds, 0.0)
        memory_snapshot = self.get_memory_snapshot()
        metrics = {
            'Loss/Total': _mean_scalar(total_loss_values, 0.0),
            'Loss/Policy': _mean_scalar(policy_loss_values, 0.0),
            'Loss/Value': _mean_scalar(value_loss_values, 0.0),
            'Loss/Entropy': _mean_scalar(entropy_loss_values, 0.0),
            'Entropy/Task': _mean_scalar(task_entropy_values, 0.0),
            'Entropy/Station': _mean_scalar(station_entropy_values, 0.0),
            'Entropy/WorkerTeam': _mean_scalar(team_entropy_values, 0.0),
            'Distill/KLTask': _mean_scalar(distill_task_values, 0.0),
            'Distill/KLStation': _mean_scalar(distill_station_values, 0.0),
            'Distill/WeightedLoss': _mean_scalar(distill_weighted_values, 0.0),
            'Distill/Lambda': _mean_scalar(distill_lambda_values, 0.0),
            'Critic/Explained_Variance': mean_exp_var,
            'Critic/Value_Predictions_Mean': mean_pred_val,
            'Critic/Target_Returns_Mean': mean_target_ret,
            'Critic/Absolute_Error_Mean': mean_abs_err,
            'Critic/Advantage_Mean': mean_adv,
            'Critic/Advantage_Std': std_adv,
            'Policy/ApproxKL': mean_kl,
            'Policy/ClipFraction': mean_clip_fraction,
            'Policy/RatioMean': mean_ratio,
            'Policy/RatioStd': std_ratio,
            'Policy/Meltdown_Count': kl_exceeded_count,
            'PPO/GPURebuildFallbackCount': gpu_rebuild_fallback_count,
            'PPO/BatchVectorRepairCount': batch_vector_repair_count,
            'Train/LearningRate': self.optimizer.param_groups[0]['lr'],
            'Train/ActorLearningRate': self.optimizer.param_groups[0]['lr'],
            'Train/CriticLearningRate': self.optimizer.param_groups[1]['lr'] if len(self.optimizer.param_groups) > 1 else self.optimizer.param_groups[0]['lr'],
            'Train/ScheduleFreeEnabled': 1.0 if self.use_schedule_free else 0.0,
            'Memory/Allocated_GB': memory_snapshot['allocated_gb'],
            'Memory/Reserved_GB': memory_snapshot['reserved_gb'],
        }
        if action_scope == "operation_station_gated_team":
            metrics.update(self._gated_team_rollout_metrics(memory))
            metrics.update(
                {
                    "CTG/PPOGateMean": _mean_scalar(ctg_ppo_gate_values, 0.0),
                    "CTG/PPONonBaselineProbMean": _mean_scalar(
                        ctg_ppo_alternative_probability_values, 0.0
                    ),
                    "CTG/PPOAltVsBaseLogitGapMean": _mean_scalar(
                        ctg_ppo_alternative_gap_values, 0.0
                    ),
                    "CTG/PPOAltBeatsBaseRate": _mean_scalar(
                        ctg_ppo_alternative_beats_values, 0.0
                    ),
                }
            )
        metrics.update(distill_lifecycle)
        return metrics
