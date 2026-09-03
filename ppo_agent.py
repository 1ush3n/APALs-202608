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
from dataclasses import dataclass, field
from typing import Callable, Tuple, List, Dict, Optional, Any
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_dense_batch
from configs import configs
from core.action_completion import (
    EarliestFinishActionCompleter,
    TeamCandidates,
    build_action_completer,
)
from models.worker_pointer_context import (
    PHYSICAL_PREDECESSOR_EDGE,
    WorkerPointerV2State,
    WorkerPressureContext,
    build_worker_eft_features,
    build_worker_pressure_context,
    gather_selected_task_skills,
)
from worker_feature_layout import resolve_worker_feature_layout
from utils.baseline_identity import (
    build_station_baseline_match,
    build_worker_baseline_membership,
)
from training.best_anchor_teacher import BestAnchorTeacherManager
from training.worker_pointer_v2_diagnostics import WorkerPointerV2Diagnostics
from runtime.batch_semantics import (
    resolve_effective_ppo_batch_size,
    resolve_v2_logical_batch_size,
)
from runtime.modes import (
    is_batched_vectorized_v2_replay,
    is_fast_exact_mode,
    is_worker_pointer_v2_mode,
    uses_behavior_group_exact_replay,
)
try:
    from schedulefree import AdamWScheduleFree
except ImportError:
    AdamWScheduleFree = None


def _finalize_action_logits(
    logits: torch.Tensor,
    invalid_mask: torch.Tensor | None,
    *,
    decision: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """在构造动作分布前统一执行形状、数值和合法性检查。"""
    logits_float = logits.float()
    mask = (
        torch.zeros_like(logits_float, dtype=torch.bool)
        if invalid_mask is None
        else invalid_mask
    )
    if mask.dtype != torch.bool:
        raise ValueError(f"{decision} mask dtype 必须为 bool，收到 {mask.dtype}")
    if mask.device != logits_float.device:
        raise ValueError(
            f"{decision} mask device 与 logits 不一致: "
            f"{mask.device} != {logits_float.device}"
        )
    if mask.shape != logits_float.shape:
        raise ValueError(
            f"{decision} mask/logits shape 不一致: "
            f"{tuple(mask.shape)} != {tuple(logits_float.shape)}"
        )

    legal = ~mask
    bad_legal = legal & ~torch.isfinite(logits_float)
    if bool(bad_legal.any()):
        rows = torch.nonzero(
            bad_legal.any(dim=-1), as_tuple=False
        ).reshape(-1).tolist()
        raise FloatingPointError(
            f"{decision} 合法候选 logits 出现 NaN/Inf，rows={rows}"
        )

    usable = legal.any(dim=-1)
    # [B, A] | [B, 1] -> [B, A]；-inf 在 AMP loss scaling 下会产生 NaN 梯度。
    final_mask = mask | (~usable).unsqueeze(-1)
    return logits_float.masked_fill(final_mask, -1.0e4), usable


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


@dataclass(frozen=True)
class FrozenAnchorProposalTrace:
    """采样时冻结的锚点条件完整团队提议轨迹（APCF full_team_v1）。

    PPO 重算必须在与采样时相同的离散动作空间上进行。本 trace 保存锚点、
    提议的有序 worker 序列、每步合法 worker ID 集合与门控特征，不保存图张量。
    即使 z=0 最终执行锚点，提议的完整对数概率也必须参与重算。
    """

    task_id: int
    station_id: int
    anchor_team: tuple[int, ...]
    proposal_team: tuple[int, ...]
    proposal_worker_sequence: tuple[int, ...]
    per_step_worker_ids: tuple[tuple[int, ...], ...]
    proposal_available: bool
    selected_branch: int
    raw_argmax_branch: int
    branch_floor: float
    gate_features: tuple[float, ...]
    hamming_distance: int
    proposal_pointer_logprob: float
    proposal_pointer_entropy_mean: float
    predicted_delta_a: float
    gate_value: float
    raw_branch_logit_gap: float
    sampled_proposal_branch_logprob: float = 0.0
    sampled_task_embedding: torch.Tensor | None = field(default=None, repr=False, compare=False)
    sampled_station_embedding: torch.Tensor | None = field(default=None, repr=False, compare=False)
    sampled_worker_embeddings: torch.Tensor | None = field(default=None, repr=False, compare=False)


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
        if bool(getattr(self.config, "ablation_no_mask", False)):
            raise ValueError(
                "ablation_no_mask 已禁用：主 PPO 方法必须在分布边界保留硬约束 mask"
            )
        self.action_completer = build_action_completer(self.config)
        self.eft_action_completer = EarliestFinishActionCompleter(self.config)
        self.policy = model.to(device)
        # 仅供 rollout_service 在同一次批量解码后立即写入 Memory；不参与 checkpoint。
        self.last_gated_team_trace: FrozenGatedTeamTrace | None = None
        self.last_gated_team_traces: list[FrozenGatedTeamTrace | None] = []
        self.worker_pointer_v2_diagnostics = WorkerPointerV2Diagnostics(num_skills=5)
        self.worker_pointer_v2_coverage_checked = False
        self._v2_fast_exact_profile: dict[str, float] | None = None
        self.last_anchor_proposal_traces: list[FrozenAnchorProposalTrace | None] = []
        self._apcf_update_count: int = 0
        
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
        self.batch_size = resolve_effective_ppo_batch_size(
            requested_batch_size,
            self.config,
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
        (
            self.amp_enabled,
            self.amp_dtype,
            scaler_enabled,
        ) = self.resolve_amp_settings(
            str(getattr(self.config, "lightning_precision", "16-mixed")),
            self.amp_device_type,
        )
        self.scaler = torch.amp.GradScaler(self.amp_device_type, enabled=scaler_enabled)
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

    @staticmethod
    def resolve_amp_settings(
        precision: str,
        device_type: str,
    ) -> tuple[bool, torch.dtype | None, bool]:
        """把 Lightning 精度语义映射为 PPO 内部 autocast 与缩放器配置。"""
        if device_type != "cuda":
            return False, None, False
        normalized = str(precision).strip().lower()
        if normalized == "16-mixed":
            return True, torch.float16, True
        if normalized == "bf16-mixed":
            return True, torch.bfloat16, False
        if normalized == "32-true":
            return False, None, False
        raise ValueError(f"不支持的 lightning_precision: {precision}")

    def autocast_context(self) -> Any:
        """返回与当前设备匹配的 AMP 上下文，CPU 路径默认禁用混合精度。"""
        if self.amp_enabled:
            assert self.amp_dtype is not None
            return torch.amp.autocast(
                device_type=self.amp_device_type,
                dtype=self.amp_dtype,
            )
        return nullcontext()

    def _v2_fast_exact_profile_sync(self) -> None:
        """仅在显式 profiling 时同步 CUDA，避免普通训练增加同步成本。"""
        if self._v2_fast_exact_profile is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _v2_fast_exact_profile_add_time(
        self,
        key: str,
        started: float,
    ) -> None:
        """累加 profile 阶段耗时，调用方负责在结束前同步 CUDA。"""
        if self._v2_fast_exact_profile is not None:
            self._v2_fast_exact_profile[key] = (
                self._v2_fast_exact_profile.get(key, 0.0)
                + (time.perf_counter() - started) * 1000.0
            )

    @staticmethod
    def _rescale_fast_exact_gradient_delta(
        gradient_before_batch: tuple[
            tuple[torch.nn.Parameter, torch.Tensor | None], ...
        ],
        loss_scale: float,
    ) -> None:
        """只缩放当前 logical batch 新增的梯度，保留窗口内既有累计梯度。"""
        scale = float(loss_scale)
        if scale == 1.0:
            return
        with torch.no_grad():
            for parameter, previous in gradient_before_batch:
                current = parameter.grad
                if current is None:
                    continue
                if previous is None:
                    current.mul_(scale)
                else:
                    current.copy_(previous + (current - previous) * scale)

    def _build_v2_pressure_context(
        self,
        *,
        task_features: torch.Tensor,
        worker_features: torch.Tensor,
        task_present: torch.Tensor | None,
        task_action_invalid: torch.Tensor | None,
        worker_present: torch.Tensor | None,
        worker_queue_invalid: torch.Tensor | None,
        physical_predecessor_edges: list[torch.Tensor] | torch.Tensor | None = None,
    ) -> WorkerPressureContext:
        """统一构造 rollout、eval 与 PPO 重算共用的 v2 压力上下文。"""

        if task_features.ndim == 2:
            task_features = task_features.unsqueeze(0)
        if worker_features.ndim == 2:
            worker_features = worker_features.unsqueeze(0)
        task_features = task_features.to(self.device)
        worker_features = worker_features.to(self.device)
        batch_size, num_tasks, _ = task_features.shape
        worker_batch, num_workers, _ = worker_features.shape
        assert batch_size == worker_batch
        if task_present is None:
            task_present = torch.ones((batch_size, num_tasks), dtype=torch.bool, device=self.device)
        if task_action_invalid is None:
            task_action_invalid = torch.zeros((batch_size, num_tasks), dtype=torch.bool, device=self.device)
        if worker_present is None:
            worker_present = torch.ones((batch_size, num_workers), dtype=torch.bool, device=self.device)
        if worker_queue_invalid is None:
            worker_queue_invalid = torch.zeros((batch_size, num_workers), dtype=torch.bool, device=self.device)
        return build_worker_pressure_context(
            task_features=task_features,
            worker_features=worker_features,
            task_present=task_present.to(self.device).reshape(batch_size, num_tasks),
            task_action_invalid=task_action_invalid.to(self.device).reshape(batch_size, num_tasks),
            worker_present=worker_present.to(self.device).reshape(batch_size, num_workers),
            worker_queue_invalid=worker_queue_invalid.to(self.device).reshape(batch_size, num_workers),
            temperature=float(self.config.worker_pointer_pressure_temperature),
            supply_epsilon=float(self.config.worker_pointer_supply_epsilon),
            physical_predecessor_edges=physical_predecessor_edges,
        )

    @staticmethod
    def _physical_predecessor_edge_list(
        batch_data: Any,
        batch_size: int,
    ) -> list[torch.Tensor]:
        """从 PyG Batch 提取每个图的局部 physical predecessor 稀疏边。"""
        if PHYSICAL_PREDECESSOR_EDGE not in batch_data.edge_types:
            return [
                torch.empty((2, 0), dtype=torch.long, device=batch_data["task"].x.device)
                for _ in range(batch_size)
            ]
        edge_index = batch_data[PHYSICAL_PREDECESSOR_EDGE].edge_index
        if batch_size == 1 and not hasattr(batch_data["task"], "batch"):
            return [edge_index]
        task_store = batch_data["task"]
        task_batch = task_store.batch
        task_ptr = task_store.ptr
        result: list[torch.Tensor] = []
        for graph_index in range(batch_size):
            if edge_index.numel() == 0:
                result.append(edge_index.new_empty((2, 0)))
                continue
            same_graph = (
                (task_batch[edge_index[0]] == graph_index)
                & (task_batch[edge_index[1]] == graph_index)
            )
            local_edges = edge_index[:, same_graph] - task_ptr[graph_index]
            result.append(local_edges)
        assert len(result) == batch_size
        return result

    def _build_v2_dynamic_eft_features(
        self,
        *,
        team_state: WorkerPointerV2State,
        worker_features: torch.Tensor,
        station_features: torch.Tensor,
        selected_station: torch.Tensor,
        task_duration: torch.Tensor,
        demand: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | None:
        """为 v2 pointer 构造末端融合的、随部分团队更新的 EFT 特征。"""

        if not bool(getattr(self.config, "worker_pointer_v2_dynamic_eft_features", False)):
            return None
        if worker_features.ndim == 2:
            worker_features = worker_features.unsqueeze(0)
        if station_features.ndim == 2:
            station_features = station_features.unsqueeze(0)
        batch_size, num_workers, _ = worker_features.shape
        assert station_features.ndim == 3 and station_features.size(0) == batch_size
        assert mask.shape == (batch_size, num_workers)
        station_ids = selected_station.to(self.device, dtype=torch.long).reshape(batch_size)
        virtual_rows = station_ids == -1
        assert torch.all((station_ids >= -1) & (station_ids < station_features.size(1)))
        safe_station_ids = station_ids.clamp_min(0)
        layout = resolve_worker_feature_layout(self.config)
        raw_worker = worker_features.to(self.device)
        raw_station = station_features.to(self.device)
        worker_wait = torch.expm1(
            raw_worker[:, :, layout.wait_idx].float().clamp_min(0.0)
        )
        worker_capacity = (
            raw_worker[:, :, layout.efficiency_idx].float()
            * raw_worker[:, :, layout.fatigue_idx].float()
        ).clamp_min(1.0e-6)
        batch_indices = torch.arange(batch_size, device=self.device)
        station_wait = torch.expm1(
            raw_station[batch_indices, safe_station_ids, 4].float().clamp_min(0.0)
        )
        features = build_worker_eft_features(
            team_state=team_state,
            worker_wait=worker_wait,
            worker_capacity=worker_capacity,
            station_wait=station_wait,
            task_duration=task_duration.to(self.device),
            demand=demand.to(self.device),
            mask=mask.to(self.device),
            clip=float(self.config.worker_pointer_v2_dynamic_eft_feature_clip),
        )
        return torch.where(virtual_rows[:, None, None], torch.zeros_like(features), features)

    def _record_v2_marginal_scarcity(
        self,
        worker_head: Any,
        *,
        worker_invalid_mask: torch.Tensor,
        selected_worker_index: int | torch.Tensor,
    ) -> None:
        marginal_total = getattr(worker_head, "_last_v2_marginal_total", None)
        marginal_extra = getattr(worker_head, "_last_v2_marginal_extra", None)
        if marginal_total is None or marginal_extra is None:
            return
        self.worker_pointer_v2_diagnostics.record_scarcity(
            marginal_total=marginal_total,
            marginal_extra=marginal_extra,
            worker_invalid_mask=worker_invalid_mask,
            selected_worker_index=selected_worker_index,
        )

    def reset_worker_pointer_v2_diagnostics(self) -> None:
        self.worker_pointer_v2_diagnostics.reset()

    def finalize_worker_pointer_v2_diagnostics(self) -> dict[str, float]:
        require_coverage = not self.worker_pointer_v2_coverage_checked
        metrics = self.worker_pointer_v2_diagnostics.finalize(
            require_coverage=require_coverage
        )
        if require_coverage:
            self.worker_pointer_v2_coverage_checked = True
        return metrics

    def _apcf_float32_pointer_logits(
        self,
        policy: Any,
        *,
        task_emb: torch.Tensor,
        station_emb: torch.Tensor,
        anchor_emb: torch.Tensor,
        worker_embs: torch.Tensor,
        mask: torch.Tensor,
        current_team_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        """? float32 APCF pointer head ?? rollout/recompute ? AMP ?????"""
        if policy.anchor_team_head is None:
            raise RuntimeError("APCF ?? anchor_team_head")
        context = current_team_emb.float() if current_team_emb is not None else None
        with torch.autocast(device_type=self.amp_device_type, enabled=False):
            scores = policy.anchor_team_head.forward_choice(
                task_emb.float(),
                station_emb.float(),
                anchor_emb.float(),
                worker_embs.float(),
                mask=mask,
                current_team_emb=context,
            )
        return scores.float()

    def _apcf_float32_gate_logits(
        self,
        policy: Any,
        *,
        task_emb: torch.Tensor,
        station_emb: torch.Tensor,
        anchor_emb: torch.Tensor,
        proposal_emb: torch.Tensor,
        gate_features: torch.Tensor,
        hamming: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """? float32 APCF gate head ???? logits ?????"""
        if policy.anchor_proposal_gate is None:
            raise RuntimeError("APCF ?? anchor_proposal_gate")
        with torch.autocast(device_type=self.amp_device_type, enabled=False):
            branch_logits, delta_a, gate_value = policy.anchor_proposal_gate(
                task_emb.float(),
                station_emb.float(),
                anchor_emb.float(),
                proposal_emb.float(),
                gate_features.float(),
                hamming.float(),
            )
        return branch_logits.float(), delta_a.float(), gate_value.float()

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
            baseline_snapshots = getattr(
                batch, "_baseline_identity_snapshots", [None] * int(batch_indices.numel())
            )
            teacher_station_baseline_match = torch.cat(
                [
                    build_station_baseline_match(
                        None
                        if snapshot is None
                        else snapshot.get("baseline_station"),
                        selected_tasks=batch.y_task[index : index + 1],
                        candidate_station_ids=torch.arange(
                            teacher_station_x.size(1), device=self.device
                        ),
                        enabled=bool(
                            getattr(
                                self.config,
                                "reschedule_baseline_identity_conditioning",
                                False,
                            )
                        ),
                        device=self.device,
                    )
                    for index, snapshot in enumerate(baseline_snapshots)
                ],
                dim=0,
            )
            teacher_station_logits = manager.teacher.station_head(
                teacher_selected_task,
                teacher_station_x,
                mask=station_mask,
                station_baseline_match=teacher_station_baseline_match,
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
        candidates = self.eft_action_completer.enumerate_team_candidates(
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
        logits, usable = _finalize_action_logits(
            logits,
            None,
            decision="gated_team",
        )
        if not bool(usable.item()):
            return None
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
                candidates.gate_features.to(device=logits.device),
                candidates.relative_finish_costs.to(device=logits.device),
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

    def _select_anchor_proposal_team(
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
        branch_floor: float,
    ) -> tuple[list[int], torch.Tensor, FrozenAnchorProposalTrace] | None:
        """锚点条件完整团队提议 + 反事实门控（APCF full_team_v1）。

        单环境与批量路径共用。流程：(o,s) → H → P → z → T：
          1) 锚点 H 由 EarliestFinishActionCompleter.complete 确定性生成；
          2) 提议 P 由 AnchorConditionedTeamPointer 自回归生成 d 人合法团队，
             首步强制非锚点成员（存在合法替代时），保证 P ≠ H；
          3) 门控 ℓ_H=0、ℓ_P=−ρ+g·6·tanh(ΔÂ/0.01)，训练按 ε 探索下限采样 z，
             评估按原始 logits 温度 0 确定性选择；
          4) 即使 z=0 执行锚点，提议的完整对数概率也计入 team_logprob。
        """
        completer = self.eft_action_completer
        layout = resolve_worker_feature_layout(self.config)
        anchor = completer.complete(
            obs,
            task_id=int(task_id),
            station_mask=None,
            worker_mask=worker_mask,
            selected_station=int(station_id),
        )
        if anchor is None:
            return None
        anchor_team = tuple(int(worker_id) for worker_id in anchor.team)
        demand = len(anchor_team)
        if demand < 1:
            return None

        worker_embs3 = (
            worker_embs.unsqueeze(0) if worker_embs.ndim == 2 else worker_embs
        )  # [1, N, H]
        assert worker_embs3.ndim == 3
        device = worker_embs3.device
        worker_feats = obs["worker"].x.to(device=device)
        anchor_emb = worker_embs3[:, list(anchor_team), :].mean(dim=1)  # [1, H]

        # ---- 合法工人掩码（技能 / 锁定 / 环境 / 去重 / 首步非锚点）----
        worker_skills = worker_feats[:, layout.skill_slice]  # [N, K]
        task_skill_vec = obs["task"].x[
            int(task_id), 5 : 5 + worker_skills.size(1)
        ].to(device=device)
        skill_idx = int(torch.argmax(task_skill_vec).item())
        has_skill = worker_skills[:, skill_idx] > 0.5  # [N]
        locks = torch.argmax(worker_feats[:, layout.lock_slice], dim=1)  # [N]
        lock_ok = (locks == 0) | (locks == (int(station_id) + 1))
        illegal = (~has_skill) | (~lock_ok)  # True=非法
        if worker_mask is not None:
            wm = worker_mask.to(device=device, dtype=torch.bool).reshape(-1)
            illegal = illegal | wm

        # ---- 自回归生成提议 P ----
        proposal_seq: list[int] = []
        per_step_ids: list[tuple[int, ...]] = []
        step_logprobs: list[torch.Tensor] = []
        step_entropies: list[torch.Tensor] = []
        current_illegal = illegal.clone()
        require_diff = bool(
            getattr(self.config, "anchor_proposal_require_difference", True)
        )
        available = True
        for step in range(demand):
            step_illegal = current_illegal.clone()
            if step == 0 and require_diff:
                for worker_id in anchor_team:
                    step_illegal[worker_id] = True
            valid_ids = torch.nonzero(~step_illegal).reshape(-1).tolist()
            if not valid_ids:
                available = False
                break
            per_step_ids.append(tuple(int(worker_id) for worker_id in valid_ids))
            context = (
                worker_embs3[0, proposal_seq, :].mean(dim=0, keepdim=True)
                if proposal_seq
                else None
            )
            scores = self._apcf_float32_pointer_logits(
                policy=policy,
                task_emb=task_emb,
                station_emb=station_emb,
                anchor_emb=anchor_emb,
                worker_embs=worker_embs3,
                mask=None,
                current_team_emb=context,
            )
            scores_float, score_usable = _finalize_action_logits(
                scores,
                step_illegal.reshape(1, -1),
                decision=f"anchor_proposal.worker[step={step}]",
            )
            if not bool(score_usable.item()):
                available = False
                break
            if deterministic or temperature <= 0.0:
                chosen = int(torch.argmax(scores_float, dim=1).item())
                step_lp = scores.new_zeros(())
                step_entropy = scores.new_zeros(())
            else:
                dist = Categorical(logits=scores_float)
                sampled = dist.sample()
                chosen = int(sampled.item())
                step_lp = dist.log_prob(sampled)
                step_entropy = dist.entropy()
            step_logprobs.append(step_lp.reshape(()))
            step_entropies.append(step_entropy.reshape(()))
            proposal_seq.append(chosen)
            current_illegal[chosen] = True  # 去重

        # ---- 门控分支 ----
        selected_branch = 0
        branch_lp = torch.zeros((), device=device)
        gate_features_flat = (0.0, 0.0, 1.0, 0.0, 0.0)
        hamming = 0
        if not available:
            team = list(anchor_team)
            trace = FrozenAnchorProposalTrace(
                task_id=int(task_id),
                station_id=int(station_id),
                anchor_team=anchor_team,
                proposal_team=(),
                proposal_worker_sequence=(),
                per_step_worker_ids=(),
                proposal_available=False,
                selected_branch=0,
                raw_argmax_branch=0,
                branch_floor=float(branch_floor),
                gate_features=gate_features_flat,
                hamming_distance=0,
                proposal_pointer_logprob=0.0,
                proposal_pointer_entropy_mean=0.0,
                predicted_delta_a=0.0,
                gate_value=0.0,
                raw_branch_logit_gap=0.0,
            )
            return team, branch_lp, trace

        proposal_emb = worker_embs3[0, proposal_seq, :].mean(dim=0, keepdim=True)  # [1, H]
        hamming = len(set(proposal_seq) - set(anchor_team))
        req = completer._extract_task_requirements(obs["task"].x, int(task_id))
        if req is None:
            return None
        _skill_req, _demand_req, task_duration = req
        num_workers = int(worker_feats.size(0))
        legal_count = int((~illegal).sum().item())
        station_wait = torch.expm1(obs["station"].x[:, 4]).clamp_min(0.0)
        worker_wait = torch.expm1(worker_feats[:, layout.wait_idx]).clamp_min(0.0)
        scale = max(float(task_duration), 1.0e-6)
        gate_features = torch.tensor(
            [
                float(demand) / max(num_workers, 1),
                float(legal_count) / max(num_workers, 1),
                1.0,
                float(station_wait[int(station_id)]) / scale,
                float(worker_wait.std(unbiased=False).item()) / scale,
            ],
            dtype=torch.float32,
            device=device,
        ).reshape(1, -1)
        gate_features_flat = tuple(float(value) for value in gate_features.reshape(-1).tolist())
        branch_logits, _delta_a, _gate_val = self._apcf_float32_gate_logits(
            policy=policy,
            task_emb=task_emb,
            station_emb=station_emb,
            anchor_emb=anchor_emb,
            proposal_emb=proposal_emb,
            gate_features=gate_features,
            hamming=torch.tensor([[float(hamming)]], dtype=torch.float32, device=device),
        )
        branch_logits, branch_usable = _finalize_action_logits(
            branch_logits,
            None,
            decision="anchor_proposal.branch",
        )
        if not bool(branch_usable.item()):
            return None
        raw_argmax_branch = int(torch.argmax(branch_logits, dim=1).item())
        if deterministic or temperature <= 0.0:
            selected_branch = raw_argmax_branch
        else:
            eps = max(float(branch_floor), 0.0)
            soft = torch.softmax(branch_logits, dim=1)  # [1, 2]
            mixed = eps + (1.0 - 2.0 * eps) * soft
            dist = Categorical(probs=mixed)
            sampled = dist.sample()
            selected_branch = int(sampled.item())
            branch_lp = dist.log_prob(sampled)
        step_sum = torch.zeros((), device=device)
        for step_lp in step_logprobs:
            step_sum = step_sum + step_lp.to(device=device)
        entropy_sum = torch.zeros((), device=device)
        for step_entropy in step_entropies:
            entropy_sum = entropy_sum + step_entropy.to(device=device)
        team_logprob = step_sum + branch_lp
        team = proposal_seq if selected_branch == 1 else list(anchor_team)
        pointer_entropy_mean = entropy_sum / max(len(step_entropies), 1)
        trace = FrozenAnchorProposalTrace(
            task_id=int(task_id),
            station_id=int(station_id),
            anchor_team=anchor_team,
            proposal_team=tuple(proposal_seq),
            proposal_worker_sequence=tuple(proposal_seq),
            per_step_worker_ids=tuple(per_step_ids),
            proposal_available=True,
            selected_branch=selected_branch,
            raw_argmax_branch=raw_argmax_branch,
            branch_floor=float(branch_floor),
            gate_features=gate_features_flat,
            hamming_distance=int(hamming),
            sampled_proposal_branch_logprob=float(team_logprob.detach().float().item()),
            sampled_task_embedding=task_emb.detach().to("cpu"),
            sampled_station_embedding=station_emb.detach().to("cpu"),
            sampled_worker_embeddings=worker_embs3.detach().to("cpu"),
            proposal_pointer_logprob=float(step_sum.detach().item()),
            proposal_pointer_entropy_mean=float(pointer_entropy_mean.detach().item()),
            predicted_delta_a=float(_delta_a.detach().reshape(-1)[0].item()),
            gate_value=float(_gate_val.detach().reshape(-1)[0].item()),
            raw_branch_logit_gap=float((branch_logits[0, 1] - branch_logits[0, 0]).detach().item()),
        )
        return team, team_logprob, trace

    @staticmethod
    def _is_valid_anchor_proposal_trace(trace: FrozenAnchorProposalTrace) -> bool:
        proposal = tuple(int(worker_id) for worker_id in trace.proposal_team)
        anchor = tuple(int(worker_id) for worker_id in trace.anchor_team)
        sequence = tuple(int(worker_id) for worker_id in trace.proposal_worker_sequence)
        if not trace.proposal_available or not proposal or proposal == anchor:
            return False
        if len(proposal) != len(anchor) or len(set(proposal)) != len(proposal):
            return False
        if proposal != sequence or len(trace.per_step_worker_ids) != len(sequence):
            return False
        return all(
            int(chosen) in tuple(int(worker_id) for worker_id in valid_ids)
            for chosen, valid_ids in zip(sequence, trace.per_step_worker_ids, strict=True)
        )
    @staticmethod
    def _anchor_proposal_rollout_metrics(memory: Any) -> dict[str, float]:
        """从冻结轨迹汇总 APCF 采样期诊断指标。"""
        traces = [
            trace for trace in getattr(memory, "anchor_proposal_traces", [])
            if isinstance(trace, FrozenAnchorProposalTrace)
        ]
        if not traces:
            return {
                "APCF/RolloutDecisionCount": 0.0,
                "APCF/RolloutProposalAvailableCount": 0.0,
                "APCF/RolloutProposalAvailableRate": 0.0,
                "APCF/RolloutHammingDistanceMean": 0.0,
                "APCF/RolloutTwoWorkerEditRate": 0.0,
                "APCF/RolloutTrainProposalSelectRate": 0.0,
                "APCF/RolloutRawProposalSelectRate": 0.0,
                "APCF/RolloutValidProposalRate": 1.0,
                "APCF/RolloutProposalPointerLogprobMean": 0.0,
                "APCF/RolloutProposalPointerEntropyMean": 0.0,
                "APCF/RolloutPredictedDeltaAMean": 0.0,
                "APCF/RolloutGateValueMean": 0.0,
                "APCF/RolloutRawBranchLogitGapMean": 0.0,
            }
        available = [t for t in traces if t.proposal_available]
        hamming_values = [float(t.hamming_distance) for t in available]
        raw_select = [
            float(t.raw_argmax_branch)
            for t in available
        ]
        def _available_mean(field_name: str) -> float:
            values = [float(getattr(trace, field_name)) for trace in available]
            return float(sum(values) / len(values)) if values else 0.0
        valid_count = sum(1 for trace in available if PPOAgent._is_valid_anchor_proposal_trace(trace))
        return {
            "APCF/RolloutDecisionCount": float(len(traces)),
            "APCF/RolloutProposalAvailableCount": float(len(available)),
            "APCF/RolloutProposalAvailableRate": float(len(available) / len(traces)),
            "APCF/RolloutHammingDistanceMean": float(
                sum(hamming_values) / len(hamming_values) if hamming_values else 0.0
            ),
            "APCF/RolloutTwoWorkerEditRate": float(
                sum(1 for h in hamming_values if h >= 2) / len(hamming_values)
                if hamming_values
                else 0.0
            ),
            "APCF/RolloutTrainProposalSelectRate": float(
                sum(t.selected_branch for t in available) / len(available)
                if available
                else 0.0
            ),
            "APCF/RolloutRawProposalSelectRate": float(
                sum(raw_select) / len(raw_select) if raw_select else 0.0
            ),
            "APCF/RolloutValidProposalRate": float(valid_count / len(available)) if available else 1.0,
            "APCF/RolloutProposalPointerLogprobMean": _available_mean(
                "proposal_pointer_logprob"
            ),
            "APCF/RolloutProposalPointerEntropyMean": _available_mean(
                "proposal_pointer_entropy_mean"
            ),
            "APCF/RolloutPredictedDeltaAMean": _available_mean(
                "predicted_delta_a"
            ),
            "APCF/RolloutGateValueMean": _available_mean("gate_value"),
            "APCF/RolloutRawBranchLogitGapMean": _available_mean(
                "raw_branch_logit_gap"
            ),
        }

    def _current_anchor_branch_floor(self) -> float:
        """当前 PPO 更新进度下的训练期探索下限 ε_t（前 decay_fraction 内线性退火）。

        行为策略使用 p_train(z=1)=ε_t+(1−2ε_t)σ(ℓ_P−ℓ_H)；评估/异步选择/最终验证
        一律用原始 logits、温度 0、无探索。
        """
        start = float(self.config.anchor_proposal_train_branch_floor_start)
        end = float(self.config.anchor_proposal_train_branch_floor_end)
        frac = float(self.config.anchor_proposal_branch_floor_decay_fraction)
        total = max(int(getattr(self.config, "max_episodes", 300)) or 1, 1)
        denominator = max(total * frac, 1.0)
        progress = min(float(self._apcf_update_count) / denominator, 1.0)
        return float(start + (end - start) * progress)

    def advance_apcf_update(self) -> None:
        """训练循环在每次 PPO 参数更新后推进 APCF 探索退火进度。"""
        self._apcf_update_count += 1

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
            None,
            candidate_prior_costs,
        )
        logits, usable = _finalize_action_logits(
            logits,
            candidate_mask,
            decision="ppo_replay.gated_team",
        )
        if not bool(usable.all()):
            raise RuntimeError("gated-team replay 的团队动作空间全屏蔽")
        dist = Categorical(logits=logits)
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

    def _validate_apcf_trace_alignment(
        self,
        traces: list[FrozenAnchorProposalTrace | None],
        *,
        actual_tasks: list[int],
        actual_stations: list[int],
        actual_teams: list[list[int]],
    ) -> None:
        """fail-fast：冻结 trace 必须与 PPO batch 中实际执行的动作逐项一致。

        任何错位（task/station 不匹配、执行团队与 trace 记录的锚点/提议团队不一致、
        提议成员不在对应步骤合法集合内）立即抛错，防止"静默错位 + 错误对数概率"。
        """
        for b, trace in enumerate(traces):
            if trace is None:
                continue
            task_id = int(actual_tasks[b])
            station_id = int(actual_stations[b])
            if int(trace.task_id) != task_id or int(trace.station_id) != station_id:
                raise RuntimeError(
                    f"APCF trace 与 batch 动作错位（index={b}）："
                    f"trace=({trace.task_id},{trace.station_id}) "
                    f"实际=({task_id},{station_id})"
                )
            executed = (
                list(trace.proposal_worker_sequence)
                if trace.selected_branch == 1
                else list(trace.anchor_team)
            )
            actual = actual_teams[b]
            if executed != actual:
                raise RuntimeError(
                    f"APCF 执行团队与 trace 不一致（index={b}, branch={trace.selected_branch}）："
                    f"trace={executed} 实际={actual}"
                )
            if trace.proposal_available:
                for j, chosen in enumerate(trace.proposal_worker_sequence):
                    valid_ids = trace.per_step_worker_ids[j]
                    if int(chosen) not in valid_ids:
                        raise RuntimeError(
                            f"APCF 提议成员不在步骤合法集合内（index={b}, step={j}）："
                            f"chosen={chosen} valid={valid_ids}"
                        )

    def _recompute_anchor_proposal_logprobs(
        self,
        *,
        task_embeddings: torch.Tensor,
        station_embeddings: torch.Tensor,
        worker_embeddings: torch.Tensor,
        frozen_traces: list[FrozenAnchorProposalTrace | None],
        use_frozen_behavior_embeddings: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """按冻结轨迹用当前策略重算 APCF 团队对数概率。

        完整对数概率 = Σ_j log q(w_j | w_{<j}, H, o, s) + log π̃(z | H, P, o, s)。
        即使 z=0 执行锚点，提议的每一步对数概率也必须计入（保证提议器获得 PPO 梯度）。
        无提议（proposal_available=False）时为确定性锚点，对数概率为 0。
        """
        batch_size = task_embeddings.size(0)
        if self.policy.anchor_team_head is None or self.policy.anchor_proposal_gate is None:
            raise RuntimeError(
                "operation_station_anchor_proposal_team 缺少 anchor_team_head / anchor_proposal_gate"
            )
        num_workers = worker_embeddings.size(1)
        team_lp_rows: list[torch.Tensor] = []
        entropy_rows: list[torch.Tensor] = []
        sampled_logprob_rows: list[torch.Tensor] = []
        for b in range(batch_size):
            trace = frozen_traces[b]
            zero_lp = torch.zeros((), device=worker_embeddings.device)
            if trace is None:
                team_lp_rows.append(zero_lp)
                entropy_rows.append(zero_lp)
                sampled_logprob_rows.append(zero_lp)
                continue
            task_emb = task_embeddings[b].unsqueeze(0)  # [1, H]
            station_emb = station_embeddings[b].unsqueeze(0)
            worker_embs = worker_embeddings[b].unsqueeze(0)
            if (
                use_frozen_behavior_embeddings
                and trace.sampled_task_embedding is not None
                and trace.sampled_station_embedding is not None
                and trace.sampled_worker_embeddings is not None
            ):
                task_emb = trace.sampled_task_embedding.to(worker_embeddings.device)
                station_emb = trace.sampled_station_embedding.to(worker_embeddings.device)
                worker_embs = trace.sampled_worker_embeddings.to(worker_embeddings.device)
            anchor_emb = worker_embs[:, list(trace.anchor_team), :].mean(dim=1)
            if not trace.proposal_available:
                team_lp_rows.append(zero_lp)
                entropy_rows.append(zero_lp)
                sampled_logprob_rows.append(zero_lp)
                continue
            proposal_emb = worker_embs[
                :, list(trace.proposal_worker_sequence), :
            ].mean(dim=1)
            step_lp_sum = zero_lp
            entropy_sum = zero_lp
            for j, chosen in enumerate(trace.proposal_worker_sequence):
                valid_ids = trace.per_step_worker_ids[j]
                step_mask = torch.full(
                    (1, num_workers), True, device=worker_embeddings.device, dtype=torch.bool
                )
                step_mask[0, list(valid_ids)] = False
                context = (
                    worker_embs[
                        :, list(trace.proposal_worker_sequence[:j]), :
                    ].mean(dim=1)
                    if j > 0
                    else None
                )
                scores = self._apcf_float32_pointer_logits(
                    policy=self.policy,
                    task_emb=task_emb,
                    station_emb=station_emb,
                    anchor_emb=anchor_emb,
                    worker_embs=worker_embs,
                    mask=None,
                    current_team_emb=context,
                )
                scores_float, score_usable = _finalize_action_logits(
                    scores,
                    step_mask,
                    decision=f"ppo_replay.anchor_proposal.worker[step={j}]",
                )
                if not bool(score_usable.item()):
                    raise RuntimeError("APCF replay 的工人动作空间全屏蔽")
                dist = Categorical(logits=scores_float)
                chosen_t = torch.tensor(
                    [[chosen]], device=worker_embeddings.device
                )
                step_lp_sum = step_lp_sum + dist.log_prob(chosen_t)[0].reshape(())
                entropy_sum = entropy_sum + dist.entropy()[0].reshape(())
            gate_features = torch.tensor(
                list(trace.gate_features), dtype=torch.float32,
                device=worker_embeddings.device,
            ).reshape(1, -1)
            hamming = torch.tensor(
                [[float(trace.hamming_distance)]], dtype=torch.float32,
                device=worker_embeddings.device,
            )
            branch_logits, _delta_a, _g = self._apcf_float32_gate_logits(
                policy=self.policy,
                task_emb=task_emb,
                station_emb=station_emb,
                anchor_emb=anchor_emb,
                proposal_emb=proposal_emb,
                gate_features=gate_features,
                hamming=hamming,
            )
            branch_logits, branch_usable = _finalize_action_logits(
                branch_logits,
                None,
                decision="ppo_replay.anchor_proposal.branch",
            )
            if not bool(branch_usable.item()):
                raise RuntimeError("APCF replay 的分支动作空间全屏蔽")
            eps = max(float(trace.branch_floor), 0.0)
            soft = torch.softmax(branch_logits, dim=1)
            mixed = eps + (1.0 - 2.0 * eps) * soft
            bdist = Categorical(probs=mixed)
            branch_t = torch.tensor(
                [[trace.selected_branch]], device=worker_embeddings.device
            )
            team_lp_rows.append(
                (step_lp_sum + bdist.log_prob(branch_t)[0]).reshape(())
            )
            entropy_rows.append(
                (entropy_sum + bdist.entropy()[0]).reshape(())
            )
            sampled_logprob_rows.append(torch.tensor(float(trace.sampled_proposal_branch_logprob), device=worker_embeddings.device))
        team_lp = torch.stack(team_lp_rows)
        team_entropy = torch.stack(entropy_rows)
        sampled_logprob = torch.stack(sampled_logprob_rows).float()
        absolute_error = (team_lp.detach().float() - sampled_logprob).abs()
        diagnostics = {
            "sampled_proposal_branch_logprob_mae": absolute_error.mean(),
            "sampled_proposal_branch_logprob_max_abs_error": absolute_error.max(),
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

    @staticmethod
    def _collect_gradient_diagnostics(named_parameters: Any) -> dict[str, float]:
        """? optimizer.step ??????????????"""
        finite = True
        actor_sq = 0.0
        critic_sq = 0.0
        apcf_sq = 0.0
        trunk_sq = 0.0
        apcf_nonzero = 0
        trunk_nonzero = 0
        v2_sq = 0.0
        v2_total = 0
        v2_nonzero = 0
        module_gradient_sq = {
            "gat_encoder": 0.0,
            "task_head": 0.0,
            "station_head": 0.0,
            "worker_v2": 0.0,
            "critic": 0.0,
            "dynamic_eft_projection": 0.0,
        }
        module_parameter_sq = dict.fromkeys(module_gradient_sq, 0.0)
        for name, parameter in named_parameters:
            is_v2_parameter = name.startswith("worker_head.v2_")
            module_name = None
            if name.startswith("encoder."):
                module_name = "gat_encoder"
            elif name.startswith("task_head."):
                module_name = "task_head"
            elif name.startswith("station_head."):
                module_name = "station_head"
            elif is_v2_parameter:
                module_name = "worker_v2"
            elif name.startswith("critic"):
                module_name = "critic"
            if module_name is not None:
                parameter_norm = float(torch.linalg.vector_norm(parameter.detach().float()).item())
                module_parameter_sq[module_name] += parameter_norm * parameter_norm
            if name.startswith("worker_head.v2_eft_proj."):
                parameter_norm = float(torch.linalg.vector_norm(parameter.detach().float()).item())
                module_parameter_sq["dynamic_eft_projection"] += parameter_norm * parameter_norm
            if is_v2_parameter:
                v2_total += 1
            gradient = parameter.grad
            if gradient is None:
                continue
            gradient_float = gradient.detach().float()
            if not bool(torch.isfinite(gradient_float).all()):
                finite = False
                continue
            norm = float(torch.linalg.vector_norm(gradient_float).item())
            if module_name is not None:
                module_gradient_sq[module_name] += norm * norm
            if name.startswith("worker_head.v2_eft_proj."):
                module_gradient_sq["dynamic_eft_projection"] += norm * norm
            if is_v2_parameter:
                v2_sq += norm * norm
                if norm > 0.0:
                    v2_nonzero += 1
            if name.startswith("critic") or ".critic" in name:
                critic_sq += norm * norm
            elif name.startswith("anchor_team_head.") or name.startswith("anchor_proposal_gate."):
                apcf_sq += norm * norm
                if norm > 0.0:
                    apcf_nonzero += 1
            else:
                actor_sq += norm * norm
                trunk_sq += norm * norm
                if norm > 0.0:
                    trunk_nonzero += 1
        metrics = {
            "finite": 1.0 if finite else 0.0,
            "actor_grad_norm": float(actor_sq ** 0.5),
            "critic_grad_norm": float(critic_sq ** 0.5),
            "apcf_grad_norm": float(apcf_sq ** 0.5),
            "trunk_grad_norm": float(trunk_sq ** 0.5),
            "apcf_nonzero": float(apcf_nonzero > 0),
            "trunk_nonzero": float(trunk_nonzero > 0),
            "v2_grad_norm": float(v2_sq ** 0.5),
            "v2_gradient_coverage": float(v2_nonzero / max(1, v2_total)),
        }
        for module_name, gradient_sq in module_gradient_sq.items():
            gradient_norm = float(gradient_sq ** 0.5)
            parameter_norm = float(module_parameter_sq[module_name] ** 0.5)
            metrics[f"{module_name}_grad_norm"] = gradient_norm
            metrics[f"{module_name}_param_norm"] = parameter_norm
            metrics[f"{module_name}_grad_to_param"] = gradient_norm / max(
                parameter_norm, 1.0e-12
            )
        return metrics
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


    @staticmethod
    def combine_factorized_value_loss(
        *,
        base_value_loss: torch.Tensor,
        conditional_value_loss: torch.Tensor,
        coefficient: float,
    ) -> torch.Tensor:
        """保持 factorized critic 扩展前后的 value loss 整体尺度。"""
        assert base_value_loss.ndim == 0
        assert conditional_value_loss.ndim == 0
        coefficient_value = float(coefficient)
        assert math.isfinite(coefficient_value) and coefficient_value >= 0.0
        combined = 0.5 * (
            base_value_loss + coefficient_value * conditional_value_loss
        )
        assert torch.isfinite(combined)
        return combined


    @staticmethod
    def compute_factorized_component_advantages(
        *,
        returns: torch.Tensor,
        old_values: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """按 component 的 active 样本生成一次冻结、归一化 advantage。"""

        assert returns.ndim == 1
        batch_size = returns.size(0)
        assert old_values.shape == active_mask.shape
        assert old_values.ndim == 2 and old_values.shape[0] == batch_size
        assert old_values.shape[1] == 3 and active_mask.dtype == torch.bool
        raw = returns.float().unsqueeze(-1) - old_values.float()
        mask = active_mask.to(dtype=raw.dtype)
        count = mask.sum(dim=0)
        mean = (raw * mask).sum(dim=0) / count.clamp_min(1.0)
        variance = ((raw - mean).square() * mask).sum(dim=0) / count.clamp_min(1.0)
        std = variance.sqrt()
        normalized = (raw - mean) / std.clamp_min(1.0e-7)
        normalized = torch.where(active_mask & (std > 1.0e-7), normalized, torch.zeros_like(normalized))
        assert normalized.shape == (batch_size, 3)
        assert torch.isfinite(normalized).all()
        return normalized

    @staticmethod
    def compute_factorized_clipped_surrogate(
        *,
        current_logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        active_mask: torch.Tensor,
        clip_range: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算按样本 active component 平均的 factorized PPO surrogate。"""

        assert current_logprobs.shape == old_logprobs.shape == advantages.shape
        assert current_logprobs.ndim == 2 and current_logprobs.shape[1] == 3
        assert active_mask.shape == current_logprobs.shape
        assert active_mask.dtype == torch.bool
        assert float(clip_range) >= 0.0
        safe_log_ratio = torch.clamp(
            current_logprobs.float() - old_logprobs.float(), -20.0, 20.0
        )
        ratios = torch.exp(safe_log_ratio)
        clipped_ratios = torch.clamp(
            ratios, 1.0 - float(clip_range), 1.0 + float(clip_range)
        )
        surrogate = torch.minimum(
            ratios * advantages,
            clipped_ratios * advantages,
        )
        component_losses = -surrogate
        active = active_mask.to(dtype=component_losses.dtype)
        per_sample_loss = (component_losses * active).sum(dim=-1) / active.sum(
            dim=-1
        ).clamp_min(1.0)
        loss = per_sample_loss.mean()
        assert torch.isfinite(loss)
        assert torch.isfinite(ratios).all()
        return loss, ratios, component_losses

    @staticmethod
    def compute_factorized_value_loss(
        *,
        current_values: torch.Tensor,
        old_values: torch.Tensor,
        targets: torch.Tensor,
        active_mask: torch.Tensor,
        clip_range: float,
    ) -> torch.Tensor:
        """计算按样本 active component 平均的 clipped conditional value loss。"""

        assert current_values.shape == old_values.shape == active_mask.shape
        assert current_values.ndim == 2 and current_values.shape[1] == 3
        assert targets.ndim == 1 and targets.shape[0] == current_values.shape[0]
        assert active_mask.dtype == torch.bool
        clipped_values = old_values + torch.clamp(
            current_values - old_values,
            -float(clip_range),
            float(clip_range),
        )
        losses = torch.maximum(
            (current_values - targets.unsqueeze(-1)).square(),
            (clipped_values - targets.unsqueeze(-1)).square(),
        )
        active = active_mask.to(dtype=losses.dtype)
        per_sample_loss = (losses * active).sum(dim=-1) / active.sum(
            dim=-1
        ).clamp_min(1.0)
        valid_sample = active.any(dim=-1)
        loss = per_sample_loss[valid_sample].mean()
        assert torch.isfinite(loss)
        return loss

    @staticmethod
    def _prepare_factorized_update_data(
        memory: Any,
        returns: torch.Tensor,
        b_station: torch.Tensor,
        b_team: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """准备一次 PPO update 内冻结的 component 行为数据和 advantage。"""
        required_fields = (
            "old_task_logprob",
            "old_station_logprob",
            "old_team_logprob",
            "old_V_task",
            "old_V_station",
            "old_V_worker",
        )
        missing_fields = [
            field_name
            for field_name in required_fields
            if len(getattr(memory, field_name, [])) != len(memory.states)
            or any(value is None for value in getattr(memory, field_name, []))
        ]
        if missing_fields:
            raise RuntimeError(
                "factorized conditional PPO 缺少完整 behavior-time Memory 字段: "
                + ", ".join(missing_fields)
            )
        old_logprobs = torch.tensor(
            list(
                zip(
                    memory.old_task_logprob,
                    memory.old_station_logprob,
                    memory.old_team_logprob,
                    strict=True,
                )
            ),
            dtype=torch.float32,
        )
        old_values = torch.tensor(
            list(
                zip(
                    memory.old_V_task,
                    memory.old_V_station,
                    memory.old_V_worker,
                    strict=True,
                )
            ),
            dtype=torch.float32,
        )
        active_mask = torch.stack(
            [
                torch.ones_like(b_station, dtype=torch.bool),
                b_station >= 0,
                (b_team >= 0).any(dim=1),
            ],
            dim=1,
        )
        advantages = PPOAgent.compute_factorized_component_advantages(
            returns=returns.float(),
            old_values=old_values,
            active_mask=active_mask,
        )
        return old_logprobs, old_values, active_mask, advantages

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
        baseline_snapshot: dict | None = None,
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
        self.last_anchor_proposal_trace = None
        
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
                
                # ------------------
                # 1. 閫夋嫨宸ュ簭 (Select Task)
                # ------------------
                task_logits = active_policy.task_head(
                    x_dict["task"], global_context, mask=None
                )
            
            if not deterministic and temperature != 1.0:
                task_logits = task_logits / max(float(temperature), 1.0e-5)
            task_invalid = (
                mask_task.to(device=task_logits.device).reshape(task_logits.shape)
                if mask_task is not None
                else None
            )
            task_logits, task_usable = _finalize_action_logits(
                task_logits,
                task_invalid,
                decision="task",
            )
            if not bool(task_usable.item()):
                return None, 0.0, 0.0, None, True
            
            if deterministic:
                task_action = torch.argmax(task_logits, dim=-1)
                task_logprob = torch.zeros((), device=self.device)
            else:
                task_dist = Categorical(logits=task_logits)
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
                station_baseline_match = build_station_baseline_match(
                    None if baseline_snapshot is None else baseline_snapshot.get("baseline_station"),
                    selected_tasks=torch.tensor([t_idx], device=self.device),
                    candidate_station_ids=torch.arange(
                        station_embs.size(1), device=self.device
                    ),
                    enabled=bool(
                        getattr(
                            self.config,
                            "reschedule_baseline_identity_conditioning",
                            False,
                        )
                    ),
                    device=self.device,
                )
                station_logits = active_policy.station_head(
                    selected_task_emb,
                    station_embs,
                    mask=None,
                    station_baseline_match=station_baseline_match,
                )
            if not deterministic and temperature != 1.0:
                station_logits = station_logits / max(float(temperature), 1.0e-5)
            station_invalid = (
                specific_station_mask.to(device=station_logits.device).reshape(
                    station_logits.shape
                )
                if specific_station_mask is not None
                else None
            )
            station_logits, station_usable = _finalize_action_logits(
                station_logits,
                station_invalid,
                decision="station",
            )
            if not bool(station_usable.item()):
                return None, 0.0, 0.0, specific_station_mask, True
            
            if deterministic:
                station_action = torch.argmax(station_logits, dim=-1)
                station_logprob = torch.zeros((), device=self.device)
            else:
                station_dist = Categorical(logits=station_logits)
                station_action = station_dist.sample()
                station_logprob = station_dist.log_prob(station_action)

            if action_scope in {
                "operation_station",
                "operation_station_gated_team",
                "operation_station_anchor_proposal_team",
            }:
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
                elif action_scope == "operation_station_anchor_proposal_team":
                    apcf_team = self._select_anchor_proposal_team(
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
                        branch_floor=self._current_anchor_branch_floor(),
                    )
                    if apcf_team is None:
                        return None, 0.0, 0.0, None, True
                    selected_team, team_logprob, apcf_trace = apcf_team
                    self.last_anchor_proposal_trace = apcf_trace
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
            
            current_worker_mask = (
                current_worker_mask
                | skill_mask.to(self.device)
                | lock_mask.to(self.device)
            )

            worker_embs = x_dict['worker'].unsqueeze(0)
            v2_mode = is_worker_pointer_v2_mode(self.config)
            v2_pressure = None
            v2_team_state = None
            v2_decode_cache = None
            v2_demand = None
            worker_baseline_member = None
            if v2_mode:
                worker_baseline_member = build_worker_baseline_membership(
                    None
                    if baseline_snapshot is None
                    else baseline_snapshot.get("baseline_team_edge_index"),
                    selected_tasks=torch.tensor([t_idx], device=self.device),
                    num_workers=worker_embs.size(1),
                    enabled=bool(
                        getattr(
                            self.config,
                            "reschedule_baseline_identity_conditioning",
                            False,
                        )
                    ),
                    device=self.device,
                )
                v2_pressure = self._build_v2_pressure_context(
                    task_features=obs['task'].x,
                    worker_features=worker_feats,
                    task_present=None,
                    task_action_invalid=mask_task,
                    worker_present=None,
                    worker_queue_invalid=mask_worker,
                    physical_predecessor_edges=self._physical_predecessor_edge_list(
                        obs, 1
                    ),
                )
                v2_team_state = active_policy.worker_head.initialize_v2_state(
                    batch_size=1, device=self.device
                )
                v2_demand = torch.tensor([float(demand)], device=self.device)
                v2_decode_cache = active_policy.worker_head.build_v2_decode_cache(
                    task_emb=selected_task_emb,
                    station_emb=x_dict['station'][int(station_action.item())].unsqueeze(0),
                    global_context=global_context,
                    worker_embs=worker_embs,
                    pressure_context=v2_pressure,
                    demand=v2_demand,
                    candidate_skills=worker_skills.unsqueeze(0),
                    task_required_skills=obs['task'].x[
                        t_idx, 5:task_skill_end
                    ].reshape(1, -1),
                )
            
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
                    if v2_mode:
                        assert v2_pressure is not None and v2_team_state is not None
                        assert v2_decode_cache is not None and v2_demand is not None
                        self.worker_pointer_v2_diagnostics.record_team_state(
                            selected_max_wait=v2_team_state.selected_max_wait,
                            selected_capacity_sum=v2_team_state.selected_capacity_sum,
                        )
                        worker_logits = active_policy.worker_head.forward_choice_v2(
                            task_emb=selected_task_emb,
                            station_emb=x_dict['station'][int(station_action.item())].unsqueeze(0),
                            global_context=global_context,
                            worker_embs=worker_embs,
                            pressure_context=v2_pressure,
                            team_state=v2_team_state,
                            demand=v2_demand,
                            mask=None,
                            decode_cache=v2_decode_cache,
                            dynamic_eft_features=self._build_v2_dynamic_eft_features(
                                team_state=v2_team_state,
                                worker_features=worker_feats,
                                station_features=obs['station'].x,
                                selected_station=station_action.reshape(1),
                                task_duration=obs['task'].x[t_idx, 0].reshape(1),
                                demand=v2_demand,
                                mask=current_worker_mask.unsqueeze(0),
                            ),
                            worker_baseline_member=worker_baseline_member,
                        )
                    else:
                        worker_logits = active_policy.worker_head.forward_choice(selected_task_emb, worker_embs, mask=None, current_team_emb=current_team_emb)
                
                if not deterministic and temperature != 1.0:
                    worker_logits = worker_logits / max(float(temperature), 1.0e-5)
                worker_invalid = current_worker_mask.reshape(worker_logits.shape)
                worker_logits, worker_usable = _finalize_action_logits(
                    worker_logits,
                    worker_invalid,
                    decision=f"worker[step={iter_cnt}]",
                )
                if not bool(worker_usable.item()):
                    break
                
                if deterministic:
                    w_action = torch.argmax(worker_logits, dim=-1)
                    w_lp = torch.zeros((), device=self.device)
                else:
                    w_dist = Categorical(logits=worker_logits)
                    w_action = w_dist.sample()
                    w_lp = w_dist.log_prob(w_action)
                
                w_idx = w_action.item()
                team_indices.append(w_idx)
                worker_logprobs.append(w_lp)
                
                # 鍒锋柊宸查€夊洟闃熻〃寰佽蹇?
                if v2_mode:
                    assert v2_team_state is not None
                    self._record_v2_marginal_scarcity(
                        active_policy.worker_head,
                        worker_invalid_mask=current_worker_mask.unsqueeze(0),
                        selected_worker_index=w_idx,
                    )
                    v2_team_state = active_policy.worker_head.advance_v2_state(
                        v2_team_state,
                        worker_embs[:, w_idx, :],
                        worker_skills[w_idx].unsqueeze(0).to(self.device),
                        selected_wait=torch.expm1(
                            worker_feats[w_idx, worker_layout.wait_idx].float().clamp_min(0.0)
                        ).reshape(1),
                        selected_capacity=(
                            worker_feats[w_idx, worker_layout.efficiency_idx].float()
                            * worker_feats[w_idx, worker_layout.fatigue_idx].float()
                        ).reshape(1),
                    )
                else:
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
        snapshots: Optional[List[dict]] = None,
        baseline_snapshots: Optional[List[dict | None]] = None,
        fast_exact_builder: Any = None,
    ) -> List[Tuple[Optional[Tuple[int, int, List[int]]], float, float, Optional[torch.Tensor], bool]]:
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

        if snapshots is not None:
            batch_size = len(snapshots)
        else:
            batch_size = len(obs_list)
        if len(mask_task_list) != batch_size:
            raise ValueError("obs_list 与动作掩码批次数量不一致")
        if baseline_snapshots is None:
            baseline_snapshots = [None] * batch_size
        if len(baseline_snapshots) != batch_size:
            raise ValueError("baseline_snapshots count must match batch_size")
        results = []
        state_value_tensors = []
        decoded_actions = []
        task_mask_refs = []
        worker_mask_refs = []
        eval_fail_flags = []
        gated_team_traces: list[FrozenGatedTeamTrace | None] = [None] * batch_size
        anchor_proposal_traces: list[FrozenAnchorProposalTrace | None] = [None] * batch_size
        # WorkerPointer v2 行为三部分 log-prob（task/station/team），与 results 顺序对齐。
        self.last_v2_behavior_logprobs = [None] * batch_size
        self.last_v2_behavior_values = [None] * batch_size
        v2_behavior_mode = (
            str(getattr(self.config, "policy_action_scope", "operation_station_worker"))
            == "operation_station_worker"
            and is_worker_pointer_v2_mode(self.config)
        )

        profile: Dict[str, float] = {}

        def _profile_sync() -> None:
            if profile_breakdown and self.amp_device_type == "cuda":
                torch.cuda.synchronize(self.device)

        with torch.no_grad():
            # 1. 鎵归噺鎵撳寘寮傛瀯鍥捐娴嬫暟鎹苟閫佸叆 GPU锛屼互 O(1) 澶嶆潅搴﹁繍琛?GNN 缂栫爜鍜?Critic 浠峰€肩綉缁?
            if profile_breakdown:
                _profile_sync()
                stage_started = time.perf_counter()
            batch_obs = None
            fast_batch = None
            if snapshots is not None:
                # Fast-Exact 快路径：由 GPU 模板原位构建，不创建 CPU PyG 图。
                if fast_exact_builder is None:
                    raise RuntimeError(
                        "select_actions_batch 的 snapshots 模式要求提供 fast_exact_builder"
                    )
                if not is_fast_exact_mode(self.config):
                    raise RuntimeError(
                        "snapshots 预构建路径仅支持 "
                        "autoregressive_pressure_v2_fast_exact 模式"
                    )
                if str(getattr(self.config, "policy_action_scope", "")) != "operation_station_worker":
                    raise RuntimeError(
                        "Fast-Exact snapshots 预构建路径仅支持 "
                        "policy_action_scope=operation_station_worker"
                    )
                if len(snapshots) != batch_size:
                    raise ValueError(
                        f"snapshots 数量与 batch_size 不一致: {len(snapshots)} != {batch_size}"
                    )
                fast_batch = fast_exact_builder.build(
                    list(snapshots), masks=None, memory_indices=None
                )
                batch_obs = fast_batch.batch
                task_ptr = fast_batch.task_ptr
                station_ptr = fast_batch.station_ptr
                worker_ptr = fast_batch.worker_ptr
            else:
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
                conditional_mode = str(
                    getattr(active_policy, "conditional_head_baseline_mode", "off")
                )
                conditional_critic_active = (
                    v2_behavior_mode and conditional_mode != "off"
                )
                critic_x_dict_batch = None
                critic_context_batch = None
                if conditional_critic_active:
                    critic_x_dict_batch, critic_context_batch = (
                        active_policy.get_critic_context(
                            batch_obs,
                            actor_x_dict_encoded=x_dict_batch,
                        )
                    )
                    state_values_batch = active_policy.critic(critic_context_batch)
                else:
                    state_values_batch = active_policy.get_value(
                        batch_obs,
                        actor_x_dict_encoded=x_dict_batch,
                    )
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
                station_embs = (
                    x_dict_batch['station'][station_start:station_end]
                    if 'station' in x_dict_batch
                    else None
                )
                worker_embs = (
                    x_dict_batch['worker'][worker_start:worker_end]
                    if 'worker' in x_dict_batch
                    else None
                )
                raw_task_features = batch_obs['task'].x[task_start:task_end]
                raw_station_features = batch_obs['station'].x[station_start:station_end]
                raw_worker_features = batch_obs['worker'].x[worker_start:worker_end]
                if fast_batch is not None:
                    per_graph_task_x = fast_batch.raw_task_slices[i]
                    per_graph_worker_x = fast_batch.raw_worker_slices[i]
                else:
                    per_graph_task_x = obs_list[i]['task'].x
                    per_graph_worker_x = obs_list[i]['worker'].x
                
                global_context_i = global_context_batch[i].unsqueeze(0)
                
                # ------------------
                # 2.1 閫夋嫨宸ュ簭 (Select Task)
                # ------------------
                m_task = mask_task_list[i].to(self.device) if mask_task_list[i] is not None else None
                with self.autocast_context():
                    task_logits = active_policy.task_head(
                        task_embs, global_context_i, mask=None
                    )
                if not deterministic and temperature != 1.0:
                    task_logits = task_logits / max(float(temperature), 1.0e-5)
                task_invalid = (
                    m_task.reshape(task_logits.shape)
                    if m_task is not None
                    else None
                )
                task_logits, task_usable = _finalize_action_logits(
                    task_logits,
                    task_invalid,
                    decision=f"task[env={i}]",
                )
                if not bool(task_usable.item()):
                    state_value_tensors.append(state_values_batch[i])
                    decoded_actions.append(
                        (0, 0, [], torch.tensor(0.0, device=self.device), None)
                    )
                    task_mask_refs.append(None)
                    worker_mask_refs.append(None)
                    eval_fail_flags.append(True)
                    continue
                
                if deterministic:
                    task_action = torch.argmax(task_logits, dim=-1)
                    task_logprob = torch.zeros((), device=self.device)
                else:
                    task_dist = Categorical(logits=task_logits)
                    task_action = task_dist.sample()
                    task_logprob = task_dist.log_prob(task_action)
                
                t_idx = task_action.item()
                selected_task_emb = task_embs[t_idx].unsqueeze(0) # [1, H]
                
                # 鑾峰彇璇ヤ换鍔＄殑宸ヤ汉浜烘暟闇€姹?
                source_task_x = per_graph_task_x
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
                    if v2_behavior_mode:
                        # 虚拟任务不产生 station/team 决策，三部分 log-prob 中后两列为零。
                        self.last_v2_behavior_logprobs[i] = (
                            float(task_logprob.detach().float().item()),
                            0.0,
                            0.0,
                        )
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
                assert station_embs is not None, (
                    "非 operation scope 必须编码 station 节点"
                )
                m_station = mask_station_matrix_list[i]
                specific_station_mask = None
                if m_station is not None:
                    specific_station_mask = m_station[t_idx].unsqueeze(0).to(self.device)

                if is_worker_pointer_v2_mode(self.config):
                    ready_task_count = (
                        (~m_task).sum()
                        if m_task is not None
                        else torch.tensor(task_embs.size(0), device=self.device)
                    )
                    legal_station_count = (
                        (~specific_station_mask).sum()
                        if specific_station_mask is not None
                        else torch.tensor(station_embs.size(0), device=self.device)
                    )
                    self.worker_pointer_v2_diagnostics.record_action_space(
                        ready_task_count=ready_task_count,
                        legal_station_count=legal_station_count,
                    )
                
                with self.autocast_context():
                    station_embs_i = station_embs.unsqueeze(0) # [1, S, H]
                    bic_enabled = bool(
                        getattr(
                            self.config,
                            "reschedule_baseline_identity_conditioning",
                            False,
                        )
                    )
                    if fast_batch is not None:
                        baseline_stations_i = fast_batch.baseline_station_slices[i]
                        baseline_edges_i = fast_batch.baseline_team_edge_index
                        identity_task_i = t_idx + int(task_ptr[i])
                        worker_offset_i = int(worker_ptr[i])
                    else:
                        baseline_snapshot_i = baseline_snapshots[i]
                        baseline_stations_i = (
                            None
                            if baseline_snapshot_i is None
                            else baseline_snapshot_i.get("baseline_station")
                        )
                        baseline_edges_i = (
                            None
                            if baseline_snapshot_i is None
                            else baseline_snapshot_i.get("baseline_team_edge_index")
                        )
                        identity_task_i = t_idx
                        worker_offset_i = 0
                    station_baseline_match_i = build_station_baseline_match(
                        baseline_stations_i,
                        selected_tasks=torch.tensor(
                            [t_idx], device=self.device
                        ),
                        candidate_station_ids=torch.arange(
                            station_embs.size(0), device=self.device
                        ),
                        enabled=bic_enabled,
                        device=self.device,
                    )
                    worker_baseline_member_i = (
                        build_worker_baseline_membership(
                            baseline_edges_i,
                            selected_tasks=torch.tensor(
                                [identity_task_i], device=self.device
                            ),
                            num_workers=worker_embs.size(0),
                            candidate_worker_offsets=torch.tensor(
                                [worker_offset_i], device=self.device
                            ),
                            enabled=bic_enabled,
                            device=self.device,
                        )
                        if worker_embs is not None
                        else None
                    )
                    station_logits = active_policy.station_head(
                        selected_task_emb,
                        station_embs_i,
                        mask=None,
                        station_baseline_match=station_baseline_match_i,
                    )
                if not deterministic and temperature != 1.0:
                    station_logits = station_logits / max(float(temperature), 1.0e-5)
                station_invalid = (
                    specific_station_mask.reshape(station_logits.shape)
                    if specific_station_mask is not None
                    else None
                )
                station_logits, station_usable = _finalize_action_logits(
                    station_logits,
                    station_invalid,
                    decision=f"station[env={i}]",
                )
                if not bool(station_usable.item()):
                    state_value_tensors.append(state_values_batch[i])
                    decoded_actions.append(
                        (0, 0, [], torch.tensor(0.0, device=self.device), None)
                    )
                    task_mask_refs.append(None)
                    worker_mask_refs.append(None)
                    eval_fail_flags.append(True)
                    continue
                
                if deterministic:
                    station_action = torch.argmax(station_logits, dim=-1)
                    station_logprob = torch.zeros((), device=self.device)
                else:
                    station_dist = Categorical(logits=station_logits)
                    station_action = station_dist.sample()
                    station_logprob = station_dist.log_prob(station_action)

                if action_scope in {
                    "operation_station",
                    "operation_station_gated_team",
                    "operation_station_anchor_proposal_team",
                }:
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
                    elif action_scope == "operation_station_anchor_proposal_team":
                        apcf_team = self._select_anchor_proposal_team(
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
                            branch_floor=self._current_anchor_branch_floor(),
                        )
                        if apcf_team is None:
                            state_value_tensors.append(torch.tensor(0.0, device=self.device))
                            decoded_actions.append((0, 1, [], torch.tensor(0.0, device=self.device), None))
                            task_mask_refs.append(None)
                            worker_mask_refs.append(None)
                            eval_fail_flags.append(True)
                            continue
                        selected_team, team_logprob, apcf_trace = apcf_team
                        anchor_proposal_traces[i] = apcf_trace
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
                
                worker_feats = per_graph_worker_x
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
                
                current_worker_mask = (
                    current_worker_mask
                    | skill_mask.to(self.device)
                    | lock_mask.to(self.device)
                )
    
                worker_embs_i = worker_embs.unsqueeze(0) # [1, W, H]
                v2_mode = is_worker_pointer_v2_mode(self.config)
                v2_pressure = None
                v2_team_state = None
                v2_decode_cache = None
                v2_demand = None
                v2_worker_skills = None
                if v2_mode:
                    v2_context_started = time.perf_counter()
                    v2_pressure = self._build_v2_pressure_context(
                        task_features=raw_task_features,
                        worker_features=raw_worker_features,
                        task_present=None,
                        task_action_invalid=m_task,
                        worker_present=None,
                        worker_queue_invalid=m_worker,
                        physical_predecessor_edges=self._physical_predecessor_edge_list(
                            batch_obs, batch_size
                        )[i : i + 1],
                    )
                    self.worker_pointer_v2_diagnostics.record_context(
                        v2_pressure,
                        host_elapsed_ms=(time.perf_counter() - v2_context_started) * 1000.0,
                    )
                    v2_team_state = active_policy.worker_head.initialize_v2_state(
                        batch_size=1, device=self.device
                    )
                    v2_demand = torch.tensor([float(demand)], device=self.device)
                    v2_worker_skills = raw_worker_features[
                        :, worker_layout.skill_slice
                    ]
                    v2_decode_cache = active_policy.worker_head.build_v2_decode_cache(
                        task_emb=selected_task_emb,
                        station_emb=station_embs[int(station_action.item())].unsqueeze(0),
                        global_context=global_context_i,
                        worker_embs=worker_embs_i,
                        pressure_context=v2_pressure,
                        demand=v2_demand,
                        candidate_skills=v2_worker_skills.unsqueeze(0),
                        task_required_skills=source_task_x[
                            t_idx, 5:task_skill_end
                        ].reshape(1, -1),
                    )
                
                max_iter = demand * 2
                iter_cnt = 0
                current_team_emb = None 
                
                while len(team_indices) < demand and iter_cnt < max_iter:
                    iter_cnt += 1
                    
                    if current_worker_mask.all():
                        break

                    if v2_mode:
                        self.worker_pointer_v2_diagnostics.record_action_space(
                            legal_worker_count=(~current_worker_mask).sum()
                        )

                    dynamic_eft_features = None
                    if v2_mode:
                        assert v2_team_state is not None and v2_demand is not None
                        dynamic_eft_features = self._build_v2_dynamic_eft_features(
                            team_state=v2_team_state,
                            worker_features=raw_worker_features,
                            station_features=raw_station_features,
                            selected_station=station_action.reshape(1),
                            task_duration=source_task_x[t_idx, 0].reshape(1),
                            demand=v2_demand,
                            mask=current_worker_mask.unsqueeze(0),
                        )
                    
                    with self.autocast_context():
                        if v2_mode:
                            assert v2_pressure is not None and v2_team_state is not None
                            assert v2_decode_cache is not None and v2_demand is not None
                            self.worker_pointer_v2_diagnostics.record_team_state(
                                selected_max_wait=v2_team_state.selected_max_wait,
                                selected_capacity_sum=v2_team_state.selected_capacity_sum,
                            )
                            worker_logits = active_policy.worker_head.forward_choice_v2(
                                task_emb=selected_task_emb,
                                station_emb=station_embs[int(station_action.item())].unsqueeze(0),
                                global_context=global_context_i,
                                worker_embs=worker_embs_i,
                                pressure_context=v2_pressure,
                                team_state=v2_team_state,
                                demand=v2_demand,
                                mask=None,
                                decode_cache=v2_decode_cache,
                                dynamic_eft_features=dynamic_eft_features,
                                worker_baseline_member=worker_baseline_member_i,
                            )
                        else:
                            worker_logits = active_policy.worker_head.forward_choice(selected_task_emb, worker_embs_i, mask=None, current_team_emb=current_team_emb)
                    
                    if not deterministic and temperature != 1.0:
                        worker_logits = worker_logits / max(float(temperature), 1.0e-5)
                    worker_invalid = current_worker_mask.reshape(worker_logits.shape)
                    worker_logits, worker_usable = _finalize_action_logits(
                        worker_logits,
                        worker_invalid,
                        decision=f"worker[env={i},step={iter_cnt}]",
                    )
                    if not bool(worker_usable.item()):
                        break
                    
                    if deterministic:
                        w_action = torch.argmax(worker_logits, dim=-1)
                        w_lp = torch.zeros((), device=self.device)
                    else:
                        w_dist = Categorical(logits=worker_logits)
                        w_action = w_dist.sample()
                        w_lp = w_dist.log_prob(w_action)
                    
                    w_idx = w_action.item()
                    team_indices.append(w_idx)
                    worker_logprobs.append(w_lp)
                    
                    if v2_mode:
                        assert v2_team_state is not None and v2_pressure is not None
                        assert v2_worker_skills is not None
                        self._record_v2_marginal_scarcity(
                            active_policy.worker_head,
                            worker_invalid_mask=current_worker_mask.unsqueeze(0),
                            selected_worker_index=w_idx,
                        )
                        selected_exposure = torch.cat(
                            (
                                v2_pressure.candidate_exposure[:, w_idx, :],
                                v2_pressure.candidate_max_exposure[:, w_idx, :],
                            ),
                            dim=-1,
                        )
                        entropy = (
                            Categorical(logits=worker_logits.float()).entropy()
                            if deterministic
                            else w_dist.entropy()
                        )
                        self.worker_pointer_v2_diagnostics.record_selection(
                            selected_exposure=selected_exposure,
                            entropy=entropy,
                            dynamic_eft_features=dynamic_eft_features,
                            selected_worker_index=w_idx,
                            worker_invalid_mask=current_worker_mask.unsqueeze(0),
                        )
                        v2_team_state = active_policy.worker_head.advance_v2_state(
                            v2_team_state,
                            worker_embs_i[:, w_idx, :],
                            v2_worker_skills[w_idx].unsqueeze(0),
                            selected_wait=torch.expm1(
                                raw_worker_features[w_idx, worker_layout.wait_idx].float().clamp_min(0.0)
                            ).reshape(1),
                            selected_capacity=(
                                raw_worker_features[w_idx, worker_layout.efficiency_idx].float()
                                * raw_worker_features[w_idx, worker_layout.fatigue_idx].float()
                            ).reshape(1),
                        )
                    else:
                        selected_worker_feats = worker_embs_i[0, team_indices, :]
                        current_team_emb = selected_worker_feats.mean(dim=0, keepdim=True)
                    
                    current_worker_mask = current_worker_mask.clone()
                    current_worker_mask[w_idx] = True

                if v2_mode:
                    assert v2_team_state is not None and v2_pressure is not None
                    team_consumption = v2_team_state.selected_skill_sum / (
                        v2_pressure.supply_all.clamp_min(
                            float(self.config.worker_pointer_supply_epsilon)
                        )
                    )
                    self.worker_pointer_v2_diagnostics.record_team(team_consumption)
                
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
                if v2_behavior_mode:
                    # 记录行为三部分 log-prob，供 group-aware 同形重放使用。
                    self.last_v2_behavior_logprobs[i] = (
                        float(task_logprob.detach().float().item()),
                        float(station_logprob.detach().float().item()),
                        float(total_worker_logprob.detach().float().item()),
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

        if conditional_critic_active:
            assert critic_x_dict_batch is not None and critic_context_batch is not None
            for index, decoded in enumerate(decoded_actions):
                if eval_fail_flags[index]:
                    continue
                task_index, station_index, _team_indices, _lp, _station_mask = decoded
                task_global = task_ptr[index] + int(task_index)
                task_emb = critic_x_dict_batch["task"][task_global].unsqueeze(0)
                virtual_station = int(station_index) == 0
                if virtual_station:
                    station_emb = torch.zeros_like(task_emb)
                else:
                    station_global = station_ptr[index] + int(station_index) - 1
                    station_emb = critic_x_dict_batch["station"][
                        station_global
                    ].unsqueeze(0)
                conditional_values = active_policy.compute_conditional_values(
                    critic_context=critic_context_batch[index].unsqueeze(0),
                    critic_task_emb=task_emb,
                    critic_station_emb=station_emb,
                    virtual_station=torch.tensor(
                        [virtual_station], device=self.device, dtype=torch.bool
                    ),
                    detach_inputs=True,
                )
                self.last_v2_behavior_values[index] = (
                    float(conditional_values["task"].item()),
                    0.0
                    if virtual_station
                    else float(conditional_values["station"].item()),
                    0.0
                    if virtual_station
                    else float(conditional_values["worker"].item()),
                )

        state_values = (
            torch.stack([v.reshape(-1) for v in state_value_tensors])
            .detach()
            .float()
            .reshape(-1)
            .cpu()
            .tolist()
        )
        action_logprobs = (
            torch.stack([decoded[3].reshape(-1) for decoded in decoded_actions])
            .detach()
            .float()
            .reshape(-1)
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
        self.last_anchor_proposal_traces = anchor_proposal_traces

        return results

    def update(self, memory: Any, env: Any = None, current_ep: int = 1) -> Dict[str, float]:
        """执行可回滚的 PPO 更新；CUDA OOM 时回滚并跳过本轮。"""
        transactional = bool(getattr(self.config, "oom_transactional_updates", True))
        skip_on_oom = bool(getattr(self.config, "skip_update_on_oom", True))
        strict_fail = is_fast_exact_mode(self.config) and bool(
            getattr(self.config, "worker_pointer_v2_strict_gpu_replay", False)
        )
        if strict_fail:
            # Fast-Exact：正式训练严禁自动降级，回滚完成后必须重新抛出并报告。
            skip_on_oom = False
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
                    self._cleanup_failed_update()
                    if transaction is not None:
                        self._restore_update_transaction(transaction)
                        self._cleanup_failed_update()
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

            except Exception:
                self._cleanup_failed_update()
                if transaction is not None:
                    self._restore_update_transaction(transaction)
                    self._cleanup_failed_update()
                raise

    def _update_once(self, memory: Any, env: Any = None, current_ep: int = 1) -> Dict[str, float]:
        """执行一次 PPO 更新，并返回 TensorBoard 指标。"""
        update_started = time.perf_counter()
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
        factorized_enabled = (
            action_scope == "operation_station_worker"
            and str(
                getattr(self.config, "conditional_head_baseline_mode", "off")
            )
            == "factorized"
        )
        if action_scope == "operation_station_gated_team":
            traces = getattr(memory, "gated_team_traces", None)
            if traces is None or len(traces) != len(memory.states):
                raise RuntimeError(
                    "门控团队 PPO 轨迹未与冻结候选序列一一对齐："
                    f"states={len(memory.states)} traces={0 if traces is None else len(traces)}"
                )
        elif action_scope == "operation_station_anchor_proposal_team":
            traces = getattr(memory, "anchor_proposal_traces", None)
            if traces is None or len(traces) != len(memory.states):
                raise RuntimeError(
                    "锚点提议 PPO 轨迹未与冻结提议序列一一对齐："
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

        old_component_logprobs = None
        old_component_values = None
        component_active_mask = None
        component_advantages = None
        if factorized_enabled:
            (
                old_component_logprobs,
                old_component_values,
                component_active_mask,
                component_advantages,
            ) = self._prepare_factorized_update_data(
                memory=memory,
                returns=rewards,
                b_station=b_station,
                b_team=b_team,
            )
        
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
            for field_name in (
                "old_task_logprob",
                "old_station_logprob",
                "old_team_logprob",
                "old_V_task",
                "old_V_station",
                "old_V_worker",
            ):
                values = getattr(memory, field_name, [])
                value = values[idx] if idx < len(values) and values[idx] is not None else 0.0
                setattr(
                    state,
                    f"y_{field_name}",
                    torch.tensor([float(value)], dtype=torch.float32),
                )
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
            batch = Batch.from_data_list(batch_data_list).to(self.device)
            batch._baseline_identity_snapshots = [
                memory.states[int(index)] for index in batch_indices
            ]
            return batch

        def _bind_gpu_batch_targets(batch, batch_indices):
            """为 GPU 原地重建后的 Batch 绑定同一批 PPO 标签与 mask。"""
            batch.y_task = b_task[batch_indices].to(self.device)
            batch.y_station = b_station[batch_indices].to(self.device)
            batch.y_team = b_team[batch_indices].to(self.device)
            batch.y_logprob = old_logprobs[batch_indices].to(self.device)
            for field_name in (
                "old_task_logprob",
                "old_station_logprob",
                "old_team_logprob",
                "old_V_task",
                "old_V_station",
                "old_V_worker",
            ):
                values = getattr(memory, field_name, [])
                setattr(
                    batch,
                    f"y_{field_name}",
                    torch.tensor(
                        [
                            float(values[int(idx)])
                            if int(idx) < len(values) and values[int(idx)] is not None
                            else 0.0
                            for idx in batch_indices
                        ],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                )
            batch.y_memory_index = torch.as_tensor(
                batch_indices, dtype=torch.long, device=self.device
            )
            batch.y_reward = rewards[batch_indices].to(self.device)
            batch.y_advantage = advantages[batch_indices].to(self.device)
            batch._baseline_identity_snapshots = [
                memory.states[int(index)] for index in batch_indices
            ]
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
        gradient_finite_values = []
        actor_gradient_norms = []
        critic_gradient_norms = []
        apcf_gradient_norms = []
        trunk_gradient_norms = []
        v2_gradient_norms = []
        v2_gradient_coverages = []
        module_grad_to_param_ratios = {
            "gat_encoder": [],
            "task_head": [],
            "station_head": [],
            "worker_v2": [],
            "critic": [],
        }
        dynamic_eft_projection_grad_norms = []
        dynamic_eft_projection_param_norms = []
        dynamic_eft_projection_grad_to_param_ratios = []
        apcf_first_recompute_mae = 0.0
        apcf_first_recompute_max_abs_error = 0.0
        apcf_recompute_checked = False
        v2_first_recompute_mae = 0.0
        v2_first_recompute_max_abs_error = 0.0
        v2_recompute_checked = False
        
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

        if is_fast_exact_mode(self.config):
            if env is None:
                raise RuntimeError(
                    "v2 fast-exact 同形重放要求提供 env 实例（GPU 模板依赖数据集上下文）"
                )
            return self._run_v2_fast_exact_replay_update(
                memory=memory,
                env=env,
                current_ep=current_ep,
                advantages=advantages,
                rewards=rewards,
                old_logprobs=old_logprobs,
                b_task=b_task,
                b_station=b_station,
                b_team=b_team,
                action_scope=action_scope,
            )
        if (
            action_scope == "operation_station_worker"
            and uses_behavior_group_exact_replay(self.config)
        ):
            if env is None:
                raise RuntimeError("v2 行为同形重放要求提供 env 实例（snapshot 重建依赖 env）")
            return self._run_v2_behavior_replay_update(
                memory=memory,
                env=env,
                current_ep=current_ep,
                advantages=advantages,
                rewards=rewards,
                old_logprobs=old_logprobs,
                b_task=b_task,
                b_station=b_station,
                b_team=b_team,
                action_scope=action_scope,
            )

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
                    batch._baseline_identity_snapshots = [
                        memory.states[int(index)]
                        for index in batch.y_memory_index.detach().cpu().tolist()
                    ]
                batch_vector_repair_count += self.ensure_hetero_batch_vectors(
                    batch,
                    batch_size=int(batch.y_task.view(-1).numel()) if hasattr(batch, 'y_task') else None,
                )
                batch_baseline_snapshots = getattr(
                    batch,
                    "_baseline_identity_snapshots",
                    [None] * int(batch.y_task.size(0)),
                )
                teacher_outputs = self._teacher_task_station_logits(batch)
                
                with self.autocast_context():
                    # 褰撳墠绛栫暐鐨勫墠鍚戜紶鎾?
                    x_dict, global_context = self.policy(batch)
                    
                    # 鐙珛楠ㄥ共璇勪及 state_values
                    conditional_value_predictions = None
                    if factorized_enabled:
                        critic_x_dict, critic_context = self.policy.get_critic_context(
                            batch,
                            actor_x_dict_encoded=x_dict,
                        )
                        state_values = self.policy.critic(critic_context).reshape(-1)
                        batch_indices = torch.arange(
                            batch.y_task.size(0), device=self.device
                        )
                        conditional_value_predictions = (
                            self.policy.get_conditional_values(
                                batch,
                                selected_task=batch.y_task,
                                selected_station=batch.y_station,
                                batch_indices=batch_indices,
                                actor_x_dict_encoded=x_dict,
                                critic_x_dict_encoded=critic_x_dict,
                                critic_context=critic_context,
                            )
                        )
                    else:
                        state_values = self.policy.get_value(
                            batch,
                            actor_x_dict_encoded=x_dict,
                        ).view(-1)
                    
                    # --- Re-evaluate LogProbs ---
                    # A. Task LogProb
                    task_x, p_mask = to_dense_batch(x_dict['task'], batch['task'].batch)
                    
                    # 鎭㈠ Mask
                    if hasattr(batch, 'y_task_mask'):
                        logical_task_mask, _ = to_dense_batch(batch.y_task_mask, batch['task'].batch)
                        combined_task_mask = logical_task_mask | (~p_mask)
                    else:
                        combined_task_mask = ~p_mask
                        
                    task_logits = self.policy.task_head(task_x, global_context, mask=None)
                    # 非有限合法 logits 由统一分布入口拒绝。
                    
                    task_logits, task_usable = _finalize_action_logits(
                        task_logits,
                        combined_task_mask,
                        decision="ppo_replay.task",
                    )
                    if not bool(task_usable.all()):
                        raise RuntimeError("PPO replay 的 task 动作空间全屏蔽")

                    task_dist = Categorical(logits=task_logits)
                    task_lp = task_dist.log_prob(batch.y_task)
                    task_entropy = task_dist.entropy()
                    norm_task_entropy = self._normalized_categorical_entropy(
                        task_entropy, combined_task_mask
                    )
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
                    
                    if "station" in x_dict:
                        station_x, s_p_mask = to_dense_batch(
                            x_dict["station"],
                            batch["station"].batch,
                        )
                    else:
                        _, s_p_mask = to_dense_batch(
                            batch["station"].x[:, :1],
                            batch["station"].batch,
                        )
                        station_x = None
                    
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
                        norm_station_entropy = torch.zeros_like(station_entropy)
                    else:
                        assert station_x is not None, (
                            "operation_station 及完整动作必须编码 station 节点"
                        )
                        station_baseline_match = torch.cat(
                            [
                                build_station_baseline_match(
                                    None
                                    if snapshot is None
                                    else snapshot.get("baseline_station"),
                                    selected_tasks=batch.y_task[index : index + 1],
                                    candidate_station_ids=torch.arange(
                                        station_x.size(1), device=self.device
                                    ),
                                    enabled=bool(
                                        getattr(
                                            self.config,
                                            "reschedule_baseline_identity_conditioning",
                                            False,
                                        )
                                    ),
                                    device=self.device,
                                )
                                for index, snapshot in enumerate(batch_baseline_snapshots)
                            ],
                            dim=0,
                        )
                        station_logits = self.policy.station_head(
                            sel_task_emb,
                            station_x,
                            mask=None,
                            station_baseline_match=station_baseline_match,
                        )
                        # 非有限合法 logits 由统一分布入口拒绝。

                        station_logits, station_usable = _finalize_action_logits(
                            station_logits,
                            curr_s_mask,
                            decision="ppo_replay.station",
                        )
                        if not bool(station_usable.all()):
                            raise RuntimeError("PPO replay 的 station 动作空间全屏蔽")

                        station_dist = Categorical(logits=station_logits)
                        physical_action = batch.y_station >= 0
                        station_lp = station_dist.log_prob(torch.clamp(batch.y_station, min=0))
                        station_entropy = station_dist.entropy()
                        station_lp = station_lp * physical_action.to(station_lp.dtype)
                        station_entropy = station_entropy * physical_action.to(station_entropy.dtype)
                        norm_station_entropy = self._normalized_categorical_entropy(
                            station_entropy, curr_s_mask
                        )
                    
                    # C. Worker Team LogProb
                    if "worker" in x_dict:
                        worker_x, w_p_mask = to_dense_batch(
                            x_dict["worker"],
                            batch["worker"].batch,
                        )
                    else:
                        _, w_p_mask = to_dense_batch(
                            batch["worker"].x[:, :1],
                            batch["worker"].batch,
                        )
                        worker_x = torch.zeros(
                            (
                                w_p_mask.size(0),
                                w_p_mask.size(1),
                                int(self.config.hidden_dim),
                            ),
                            dtype=task_logits.dtype,
                            device=task_logits.device,
                        )
                    team_lp = torch.zeros_like(task_lp)
                    team_entropy = torch.zeros_like(task_entropy)
                    normalized_team_entropy = torch.zeros_like(team_entropy)
                    
                    if hasattr(batch, 'y_worker_mask'):
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
                    v2_mode = (
                        action_scope == "operation_station_worker"
                        and is_worker_pointer_v2_mode(self.config)
                    )
                    v2_pressure = None
                    v2_team_state = None
                    v2_decode_cache = None
                    raw_worker_x_v2 = None
                    raw_station_x_v2 = None
                    worker_layout = resolve_worker_feature_layout(self.config)
                    selected_station_emb_v2 = None
                    selected_demand_v2 = None
                    if v2_mode:
                        raw_task_x_v2, raw_task_present = to_dense_batch(
                            batch['task'].x, batch['task'].batch
                        )
                        raw_worker_x_v2, raw_worker_present = to_dense_batch(
                            batch['worker'].x, batch['worker'].batch
                        )
                        raw_station_x_v2, _ = to_dense_batch(
                            batch['station'].x, batch['station'].batch
                        )
                        worker_baseline_member = torch.cat(
                            [
                                build_worker_baseline_membership(
                                    None
                                    if snapshot is None
                                    else snapshot.get("baseline_team_edge_index"),
                                    selected_tasks=batch.y_task[index : index + 1],
                                    num_workers=worker_x.size(1),
                                    enabled=bool(
                                        getattr(
                                            self.config,
                                            "reschedule_baseline_identity_conditioning",
                                            False,
                                        )
                                    ),
                                    device=self.device,
                                )
                                for index, snapshot in enumerate(batch_baseline_snapshots)
                            ],
                            dim=0,
                        )
                        v2_pressure = self._build_v2_pressure_context(
                            task_features=raw_task_x_v2,
                            worker_features=raw_worker_x_v2,
                            task_present=raw_task_present,
                            task_action_invalid=combined_task_mask,
                            worker_present=raw_worker_present,
                            worker_queue_invalid=replay_worker_mask,
                            physical_predecessor_edges=self._physical_predecessor_edge_list(
                                batch, B_size
                            ),
                        )
                        v2_team_state = self.policy.worker_head.initialize_v2_state(
                            batch_size=B_size, device=self.device
                        )
                        selected_station_emb_v2 = station_x[
                            batch_indices, torch.clamp(batch.y_station, min=0)
                        ]
                        selected_demand_v2 = raw_task_x_v2[
                            batch_indices, batch.y_task, 16
                        ]
                        selected_task_skills_v2 = gather_selected_task_skills(
                            raw_task_x_v2, batch.y_task
                        )
                        v2_decode_cache = self.policy.worker_head.build_v2_decode_cache(
                            task_emb=sel_task_emb,
                            station_emb=selected_station_emb_v2,
                            global_context=global_context,
                            worker_embs=worker_x,
                            pressure_context=v2_pressure,
                            demand=selected_demand_v2,
                            candidate_skills=raw_worker_x_v2[
                                ..., worker_layout.skill_slice
                            ],
                            task_required_skills=selected_task_skills_v2,
                        )
                    
                    worker_steps = (
                        range(batch.y_team.size(1))
                        if action_scope == "operation_station_worker"
                        else range(0)
                    )
                    for k in worker_steps:
                        target = batch.y_team[:, k] 
                        valid_step = (target != -1)
                        if not valid_step.any(): continue
                        
                        if v2_mode:
                            assert v2_pressure is not None
                            assert v2_team_state is not None
                            assert selected_station_emb_v2 is not None
                            assert selected_demand_v2 is not None
                            assert v2_decode_cache is not None
                            assert raw_task_x_v2 is not None and raw_worker_x_v2 is not None
                            assert raw_station_x_v2 is not None
                            self.worker_pointer_v2_diagnostics.record_team_state(
                                selected_max_wait=v2_team_state.selected_max_wait,
                                selected_capacity_sum=v2_team_state.selected_capacity_sum,
                            )
                            logits = self.policy.worker_head.forward_choice_v2(
                                task_emb=sel_task_emb,
                                station_emb=selected_station_emb_v2,
                                global_context=global_context,
                                worker_embs=worker_x,
                                pressure_context=v2_pressure,
                                team_state=v2_team_state,
                                demand=selected_demand_v2,
                                mask=None,
                                decode_cache=v2_decode_cache,
                                dynamic_eft_features=self._build_v2_dynamic_eft_features(
                                    team_state=v2_team_state,
                                    worker_features=raw_worker_x_v2,
                                    station_features=raw_station_x_v2,
                                    selected_station=batch.y_station,
                                    task_duration=raw_task_x_v2[
                                        batch_indices, batch.y_task, 0
                                    ],
                                    demand=selected_demand_v2,
                                    mask=curr_mask,
                                ),
                                worker_baseline_member=worker_baseline_member,
                            )
                        else:
                            logits = self.policy.worker_head.forward_choice(sel_task_emb, worker_x, mask=None, current_team_emb=current_team_emb)
                        # 非有限合法 logits 由统一分布入口拒绝。
                        
                        logits, worker_usable = _finalize_action_logits(
                            logits,
                            curr_mask,
                            decision=f"ppo_replay.worker[step={k}]",
                        )
                        if not bool(worker_usable[valid_step].all()):
                            raise RuntimeError(
                                f"PPO replay 的 worker 动作空间全屏蔽，step={k}"
                            )

                        dist = Categorical(logits=logits)
                        step_lp = dist.log_prob(torch.clamp(target, min=0)) 
                        team_lp[valid_step] += step_lp[valid_step]
                        step_entropy = dist.entropy()
                        team_entropy[valid_step] += step_entropy[valid_step]
                        step_normalized_entropy = self._normalized_categorical_entropy(
                            step_entropy, curr_mask
                        )
                        normalized_team_entropy[valid_step] += (
                            step_normalized_entropy[valid_step]
                        )
                        
                        # Update current_team_emb
                        valid_b_indices = torch.nonzero(valid_step).squeeze(-1)
                        valid_targets = target[valid_step]
                        
                        selected_feats = (
                            worker_x[valid_b_indices, valid_targets]
                            if not v2_mode
                            else None
                        )
                        
                        # 浣跨敤 clone() 淇濋殰 PyTorch 鑷姩姹傚鏈哄埗鐨勮繛缁€?(Gradient Preservation)
                        next_team_emb_sum = (
                            team_emb_sum.clone() if not v2_mode else team_emb_sum
                        )
                        next_team_cnt = team_cnt.clone() if not v2_mode else team_cnt
                        
                        if not v2_mode:
                            assert selected_feats is not None
                            next_team_emb_sum[valid_b_indices] += selected_feats
                            next_team_cnt[valid_b_indices] += 1
                        
                        team_emb_sum = next_team_emb_sum
                        team_cnt = next_team_cnt
                        
                        current_team_emb = (
                            team_emb_sum / torch.clamp(team_cnt, min=1.0)
                            if not v2_mode
                            else None
                        )
                        if v2_mode:
                            assert v2_team_state is not None and raw_worker_x_v2 is not None
                            safe_target = torch.clamp(target, min=0)
                            selected_emb_all = worker_x[batch_indices, safe_target]
                            selected_skills_all = raw_worker_x_v2[
                                batch_indices, safe_target, 1:6
                            ]
                            v2_team_state = self.policy.worker_head.advance_v2_state(
                                v2_team_state,
                                selected_emb_all,
                                selected_skills_all,
                                valid=valid_step,
                                selected_wait=torch.expm1(
                                    raw_worker_x_v2[
                                        batch_indices, safe_target, worker_layout.wait_idx
                                    ].float().clamp_min(0.0)
                                ),
                                selected_capacity=(
                                    raw_worker_x_v2[
                                        batch_indices, safe_target, worker_layout.efficiency_idx
                                    ].float()
                                    * raw_worker_x_v2[
                                        batch_indices, safe_target, worker_layout.fatigue_idx
                                    ].float()
                                ),
                            )
                        
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

                    if action_scope == "operation_station_anchor_proposal_team":
                        memory_indices = batch.y_memory_index.detach().cpu().tolist()
                        apcf_traces = [
                            memory.anchor_proposal_traces[int(memory_index)]
                            for memory_index in memory_indices
                        ]
                        self._validate_apcf_trace_alignment(
                            apcf_traces,
                            actual_tasks=batch.y_task.detach().cpu().tolist(),
                            actual_stations=batch.y_station.detach().cpu().tolist(),
                            actual_teams=[
                                [
                                    int(w)
                                    for w in row
                                    if int(w) >= 0
                                ]
                                for row in batch.y_team.detach().cpu().tolist()
                            ],
                        )
                        apcf_station_embeddings = station_x[
                            batch_indices, torch.clamp(batch.y_station, min=0)
                        ]
                        team_lp, team_entropy, _apcf_diag = self._recompute_anchor_proposal_logprobs(
                            task_embeddings=sel_task_emb,
                            station_embeddings=apcf_station_embeddings,
                            worker_embeddings=worker_x,
                            frozen_traces=apcf_traces,
                            use_frozen_behavior_embeddings=not apcf_recompute_checked,
                        )
                        if not apcf_recompute_checked:
                            apcf_first_recompute_mae = float(_apcf_diag["sampled_proposal_branch_logprob_mae"].detach().float().item())
                            apcf_first_recompute_max_abs_error = float(_apcf_diag["sampled_proposal_branch_logprob_max_abs_error"].detach().float().item())
                            apcf_recompute_checked = True

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
                    if v2_mode and not v2_recompute_checked:
                        recompute_error = torch.abs(total_lp.detach().float() - old_lp.float())
                        v2_first_recompute_mae = float(recompute_error.mean().cpu())
                        v2_first_recompute_max_abs_error = float(recompute_error.max().cpu())
                        if uses_behavior_group_exact_replay(self.config):
                            v2_threshold = 1.0e-3 if self.amp_dtype == torch.bfloat16 else 1.0e-4
                            if v2_first_recompute_max_abs_error > v2_threshold:
                                raise RuntimeError(
                                    "WorkerPointer v2 首次 PPO 重算与行为策略不一致："
                                    f"max_abs_error={v2_first_recompute_max_abs_error:.6g}，"
                                    f"threshold={v2_threshold:.6g}"
                                )
                        v2_recompute_checked = True
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
                    
                    component_ratios = None
                    if factorized_enabled:
                        assert old_component_logprobs is not None
                        assert component_advantages is not None
                        assert component_active_mask is not None
                        memory_indices = batch.y_memory_index.reshape(-1).to(
                            device=self.device, dtype=torch.long
                        )
                        current_component_logprobs = torch.stack(
                            [task_lp, station_lp, team_lp], dim=1
                        )
                        policy_loss, component_ratios, _component_losses = (
                            self.compute_factorized_clipped_surrogate(
                                current_logprobs=current_component_logprobs,
                                old_logprobs=old_component_logprobs[
                                    memory_indices
                                ].to(self.device),
                                advantages=component_advantages[
                                    memory_indices
                                ].to(self.device),
                                active_mask=component_active_mask[
                                    memory_indices
                                ].to(self.device),
                                clip_range=curr_eps_clip,
                            )
                        )
                    else:
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
                    if factorized_enabled:
                        assert conditional_value_predictions is not None
                        assert old_component_values is not None
                        assert component_active_mask is not None
                        memory_indices = batch.y_memory_index.reshape(-1).to(
                            device=self.device, dtype=torch.long
                        )
                        current_conditional_values = torch.cat(
                            [
                                conditional_value_predictions["task"],
                                conditional_value_predictions["station"],
                                conditional_value_predictions["worker"],
                            ],
                            dim=1,
                        )
                        conditional_value_loss = (
                            self.compute_factorized_value_loss(
                                current_values=current_conditional_values,
                                old_values=old_component_values[
                                    memory_indices
                                ].to(self.device),
                                targets=b_reward,
                                active_mask=component_active_mask[
                                    memory_indices
                                ].to(self.device),
                                clip_range=curr_eps_clip,
                            )
                        )
                        value_loss_raw = self.combine_factorized_value_loss(
                            base_value_loss=value_loss_raw,
                            conditional_value_loss=conditional_value_loss,
                            coefficient=float(
                                getattr(
                                    self.config,
                                    "conditional_head_value_coef",
                                    1.0,
                                )
                            ),
                        )
                    value_loss = c_val * value_loss_raw
                    
                    # 鍔ㄦ€佽幏鍙栧綋鍓?batch 涓悇鍐崇瓥鍒嗘敮鐨勬渶澶у姩浣滅淮搴︿互杩涜鐔靛綊涓€鍖?
                    
                    # 瀵瑰悇鍒嗘敮鐨勭喌鍒嗗埆鍦ㄥ悇鑷渶澶у姩浣滅淮搴︿笅閲囩敤瀵规暟灏哄害杩涜 [0, 1] 褰掍竴鍖?
                    # 鍥㈤槦宸ヤ汉鐨勬渶澶х喌浠ラ€夋嫨 2 涓伐浜轰负鍩哄簳绠椾綔 2.0 鍊嶇殑鏈€澶у伐浜虹淮鏁板鏁?
                    
                    # 璁剧珛鍒嗘敮瑙ｈ€︾郴鏁颁互鐙珛鎺у埗鍜屽钩琛″悇鍒嗘敮鎺㈢储鍔涘害
                    c_ent_task = 1.0
                    c_ent_station = 1.5
                    c_ent_team = 0.5
                    
                    avg_normalized_entropy = (
                        c_ent_task * norm_task_entropy.mean() +
                        c_ent_station * norm_station_entropy.mean() +
                        c_ent_team * normalized_team_entropy.mean()
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
                    
                    gradient_diagnostics = self._collect_gradient_diagnostics(self.policy.named_parameters())
                    gradient_finite_values.append(gradient_diagnostics["finite"])
                    actor_gradient_norms.append(gradient_diagnostics["actor_grad_norm"])
                    critic_gradient_norms.append(gradient_diagnostics["critic_grad_norm"])
                    apcf_gradient_norms.append(gradient_diagnostics["apcf_grad_norm"])
                    trunk_gradient_norms.append(gradient_diagnostics["trunk_grad_norm"])
                    v2_gradient_norms.append(gradient_diagnostics["v2_grad_norm"])
                    v2_gradient_coverages.append(
                        gradient_diagnostics["v2_gradient_coverage"]
                    )
                    for module_name, values in module_grad_to_param_ratios.items():
                        values.append(gradient_diagnostics[f"{module_name}_grad_to_param"])
                    dynamic_eft_projection_grad_norms.append(
                        gradient_diagnostics["dynamic_eft_projection_grad_norm"]
                    )
                    dynamic_eft_projection_param_norms.append(
                        gradient_diagnostics["dynamic_eft_projection_param_norm"]
                    )
                    dynamic_eft_projection_grad_to_param_ratios.append(
                        gradient_diagnostics["dynamic_eft_projection_grad_to_param"]
                    )
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
            if isinstance(values[0], torch.Tensor):
                return float(
                    torch.stack([value.detach().float() for value in values])
                    .mean()
                    .cpu()
                    .item()
                )
            return float(sum(float(value) for value in values) / len(values))

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
            'PPO/GradientsFinite': float(min(gradient_finite_values)) if gradient_finite_values else 1.0,
            'PPO/ActorGradNorm': float(sum(actor_gradient_norms) / len(actor_gradient_norms)) if actor_gradient_norms else 0.0,
            'PPO/CriticGradNorm': float(sum(critic_gradient_norms) / len(critic_gradient_norms)) if critic_gradient_norms else 0.0,
            'PointerV2/GradientNorm': float(sum(v2_gradient_norms) / len(v2_gradient_norms)) if v2_gradient_norms else 0.0,
            'PointerV2/GradientCoverage': float(sum(v2_gradient_coverages) / len(v2_gradient_coverages)) if v2_gradient_coverages else 0.0,
            'PointerV2/PPOFirstRecomputeMAE': v2_first_recompute_mae,
            'PointerV2/PPOFirstRecomputeMaxAE': v2_first_recompute_max_abs_error,
            'Train/ActorLearningRate': self.optimizer.param_groups[0]['lr'],
            'Train/CriticLearningRate': self.optimizer.param_groups[1]['lr'] if len(self.optimizer.param_groups) > 1 else self.optimizer.param_groups[0]['lr'],
            'Train/ScheduleFreeEnabled': 1.0 if self.use_schedule_free else 0.0,
            'Memory/Allocated_GB': memory_snapshot['allocated_gb'],
            'Memory/Reserved_GB': memory_snapshot['reserved_gb'],
        }
        metrics.update(
            {
                f"Gradient/{label}GradToParamRatio": _mean_scalar(
                    module_grad_to_param_ratios[module_name], 0.0
                )
                for module_name, label in (
                    ("gat_encoder", "GATEncoder"),
                    ("task_head", "TaskHead"),
                    ("station_head", "StationHead"),
                    ("worker_v2", "WorkerV2"),
                    ("critic", "Critic"),
                )
            }
        )
        if action_scope == "operation_station_worker" and is_worker_pointer_v2_mode(
            self.config
        ):
            metrics.update(
                {
                    "PointerV2/AutocastEnabled": 1.0 if self.amp_enabled else 0.0,
                    "PointerV2/AutocastBF16": 1.0
                    if self.amp_enabled and self.amp_dtype == torch.bfloat16
                    else 0.0,
                    "PointerV2/GradScalerEnabled": 1.0
                    if self.scaler.is_enabled()
                    else 0.0,
                    "PointerV2/NonFiniteCount": 0.0,
                }
            )
            if bool(getattr(self.config, "worker_pointer_v2_dynamic_eft_features", False)):
                metrics.update(
                    {
                        "PointerV2/DynamicEFT/ProjectionGradNorm": _mean_scalar(
                            dynamic_eft_projection_grad_norms, 0.0
                        ),
                        "PointerV2/DynamicEFT/ProjectionWeightNorm": _mean_scalar(
                            dynamic_eft_projection_param_norms, 0.0
                        ),
                        "PointerV2/DynamicEFT/ProjectionGradToParamRatio": _mean_scalar(
                            dynamic_eft_projection_grad_to_param_ratios, 0.0
                        ),
                    }
                )
        if action_scope == "operation_station_anchor_proposal_team":
            metrics.update(
                {
                    'APCF/GradientNorm': float(sum(apcf_gradient_norms) / len(apcf_gradient_norms)) if apcf_gradient_norms else 0.0,
                    'APCF/PPOProposalBranchRecomputeMAE': apcf_first_recompute_mae,
                    'APCF/PPOProposalBranchRecomputeMaxAE': apcf_first_recompute_max_abs_error,
                    'APCF/AutocastEnabled': 1.0 if self.amp_enabled else 0.0,
                    'APCF/AutocastBF16Target': 1.0 if self.amp_enabled and self.amp_dtype == torch.bfloat16 else 0.0,
                    'APCF/TrunkGradientNorm': float(sum(trunk_gradient_norms) / len(trunk_gradient_norms)) if trunk_gradient_norms else 0.0,
                    'APCF/PretrainLoadedModelKeyCount': float(getattr(self.config, 'apcf_pretrain_loaded_model_key_count', 0) or 0),
                }
            )
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
        if self.best_anchor_teacher is not None:
            metrics.update(
                {
                    'Distill/KLTask': _mean_scalar(distill_task_values, 0.0),
                    'Distill/KLStation': _mean_scalar(distill_station_values, 0.0),
                    'Distill/WeightedLoss': _mean_scalar(distill_weighted_values, 0.0),
                    'Distill/Lambda': _mean_scalar(distill_lambda_values, 0.0),
                }
            )
            metrics.update(distill_lifecycle)
        if is_batched_vectorized_v2_replay(self.config):
            metrics["V2/BatchedReplayUpdateSeconds"] = float(
                time.perf_counter() - update_started
            )
        return metrics

    def _build_v2_behavior_group_batch(
        self,
        *,
        memory: Any,
        env: Any,
        memory_indices: list[int],
        b_task: torch.Tensor,
        b_station: torch.Tensor,
        b_team: torch.Tensor,
        old_logprobs: torch.Tensor,
        rewards: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Any:
        """按行为组原顺序重建 PyG Batch，并绑定节点级 mask 与样本级目标。"""
        if not memory_indices:
            raise ValueError("v2 行为组不能为空")
        data_list: list[Any] = []
        max_team = int(b_team.shape[1]) if b_team.ndim == 2 else 1
        for raw_index in memory_indices:
            index = int(raw_index)
            state = env.rebuild_state_from_snapshot(memory.states[index])
            state.y_task = b_task[index].reshape(1)
            state.y_station = b_station[index].reshape(1)
            state.y_team = b_team[index, :max_team].reshape(1, max_team)
            state.y_logprob = old_logprobs[index].reshape(1)
            for field_name in (
                "old_task_logprob",
                "old_station_logprob",
                "old_team_logprob",
                "old_V_task",
                "old_V_station",
                "old_V_worker",
            ):
                values = getattr(memory, field_name, [])
                value = values[index] if index < len(values) and values[index] is not None else 0.0
                setattr(
                    state,
                    f"y_{field_name}",
                    torch.tensor([float(value)], dtype=torch.float32),
                )
            state.y_reward = rewards[index].reshape(1)
            state.y_advantage = advantages[index].reshape(1)
            state.y_memory_index = torch.tensor([index], dtype=torch.long)
            if len(memory.values) > index:
                state.y_value = torch.as_tensor(
                    [float(memory.values[index])], dtype=torch.float32
                )
            if len(memory.masks) <= index:
                raise RuntimeError(f"v2 行为组缺少动作 mask: memory_index={index}")
            task_mask, station_mask, worker_mask = memory.masks[index]
            # 节点级 mask 保持原秩；PyG Batch 会沿节点维拼接。
            state.y_task_mask = torch.as_tensor(task_mask, dtype=torch.bool)
            state.y_station_mask = torch.as_tensor(station_mask, dtype=torch.bool)
            state.y_worker_mask = torch.as_tensor(worker_mask, dtype=torch.bool)
            data_list.append(state)
        group_batch = Batch.from_data_list(data_list).to(self.device)
        assert int(group_batch.num_graphs) == len(memory_indices)
        group_batch._baseline_identity_snapshots = [
            memory.states[int(index)] for index in memory_indices
        ]
        return group_batch

    @staticmethod
    def _normalized_categorical_entropy(
        entropy: torch.Tensor,
        invalid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """按每个样本自身合法动作数归一化熵；单一合法动作的归一化熵为零。"""
        valid_count = (~invalid_mask).sum(dim=-1).to(dtype=torch.float32)
        denominator = torch.log(valid_count.clamp_min(2.0))
        normalized = entropy.float() / denominator
        return torch.where(valid_count > 1.0, normalized, torch.zeros_like(normalized))

    def _replay_v2_behavior_group(self, batch: Any) -> list[dict[str, torch.Tensor]]:
        """编码器/critic 按原行为组前向，动作 head 按 rollout 的 B=1 逐样本重放。"""
        with self.autocast_context():
            x_dict, global_context = self.policy(batch)
            conditional_value_predictions = None
            if (
                str(getattr(self.config, "policy_action_scope", ""))
                == "operation_station_worker"
                and str(
                    getattr(self.config, "conditional_head_baseline_mode", "off")
                )
                == "factorized"
            ):
                critic_x_dict, critic_context = self.policy.get_critic_context(
                    batch, actor_x_dict_encoded=x_dict
                )
                state_values = self.policy.critic(critic_context).reshape(-1)
                batch_indices = torch.arange(
                    batch.y_task.size(0), device=self.device
                )
                conditional_value_predictions = self.policy.get_conditional_values(
                    batch,
                    selected_task=batch.y_task,
                    selected_station=batch.y_station,
                    batch_indices=batch_indices,
                    actor_x_dict_encoded=x_dict,
                    critic_x_dict_encoded=critic_x_dict,
                    critic_context=critic_context,
                )
            else:
                state_values = self.policy.get_value(
                    batch, actor_x_dict_encoded=x_dict
                ).reshape(-1)

        group_size = int(batch.num_graphs)
        assert global_context.shape[0] == group_size
        assert state_values.shape == (group_size,)
        task_ptr = batch["task"].ptr.detach().cpu().tolist()
        station_ptr = batch["station"].ptr.detach().cpu().tolist()
        worker_ptr = batch["worker"].ptr.detach().cpu().tolist()
        task_targets_cpu = batch.y_task.detach().cpu().tolist()
        station_targets_cpu = batch.y_station.detach().cpu().tolist()
        team_targets_cpu = batch.y_team.detach().cpu().tolist()
        baseline_identity_snapshots = getattr(
            batch, "_baseline_identity_snapshots", [None] * group_size
        )
        outputs: list[dict[str, torch.Tensor]] = []
        worker_layout = resolve_worker_feature_layout(self.config)
        num_skills = int(worker_layout.num_skill_types)

        for sample_index in range(group_size):
            task_start, task_end = task_ptr[sample_index : sample_index + 2]
            station_start, station_end = station_ptr[sample_index : sample_index + 2]
            worker_start, worker_end = worker_ptr[sample_index : sample_index + 2]
            task_embs = x_dict["task"][task_start:task_end]
            station_embs = x_dict["station"][station_start:station_end]
            worker_embs = x_dict["worker"][worker_start:worker_end]
            raw_task = batch["task"].x[task_start:task_end]
            raw_worker = batch["worker"].x[worker_start:worker_end]
            raw_station = batch["station"].x[station_start:station_end]
            task_mask = batch.y_task_mask[task_start:task_end].bool()
            station_mask_matrix = batch.y_station_mask[task_start:task_end].bool()
            worker_queue_mask = batch.y_worker_mask[worker_start:worker_end].bool()
            global_i = global_context[sample_index].unsqueeze(0)
            task_target = batch.y_task[sample_index].reshape(1)
            task_id = int(task_targets_cpu[sample_index])
            assert 0 <= task_id < task_embs.shape[0]

            with self.autocast_context():
                task_logits = self.policy.task_head(
                    task_embs, global_i, mask=None
                )
            # 非有限合法 logits 由统一分布入口拒绝。
            task_logits, task_usable = _finalize_action_logits(
                task_logits,
                task_mask.unsqueeze(0),
                decision=f"v2_behavior.task[sample={sample_index}]",
            )
            if not bool(task_usable.item()):
                raise RuntimeError("v2 behavior replay 的 task 动作空间全屏蔽")

            task_dist = Categorical(logits=task_logits)
            task_lp = task_dist.log_prob(task_target)
            task_entropy = task_dist.entropy()
            normalized_task_entropy = self._normalized_categorical_entropy(
                task_entropy, task_mask.unsqueeze(0)
            )
            selected_task_emb = task_embs[task_id].unsqueeze(0)

            team_target_ids = [
                int(worker_id)
                for worker_id in team_targets_cpu[sample_index]
                if int(worker_id) >= 0
            ]
            is_virtual = not team_target_ids
            zero = torch.zeros_like(task_lp)
            if is_virtual:
                conditional_values = None
                if conditional_value_predictions is not None:
                    conditional_values = torch.cat(
                        [
                            conditional_value_predictions["task"],
                            conditional_value_predictions["station"],
                            conditional_value_predictions["worker"],
                        ],
                        dim=1,
                    )[sample_index].reshape(1, 3)
                outputs.append(
                    {
                        "task": task_lp,
                        "station": zero,
                        "team": zero,
                        "entropy": task_entropy,
                        "normalized_entropy": normalized_task_entropy,
                        "state_value": state_values[sample_index].reshape(1),
                        "conditional_values": conditional_values,
                    }
                )
                continue

            station_target = batch.y_station[sample_index].reshape(1)
            station_id = int(station_targets_cpu[sample_index])
            assert 0 <= station_id < station_embs.shape[0]
            baseline_snapshot = baseline_identity_snapshots[sample_index]
            station_baseline_match = build_station_baseline_match(
                None
                if baseline_snapshot is None
                else baseline_snapshot.get("baseline_station"),
                selected_tasks=torch.tensor([task_id], device=self.device),
                candidate_station_ids=torch.arange(
                    station_embs.size(0), device=self.device
                ),
                enabled=bool(
                    getattr(
                        self.config,
                        "reschedule_baseline_identity_conditioning",
                        False,
                    )
                ),
                device=self.device,
            )
            station_mask = station_mask_matrix[task_id].unsqueeze(0)
            if self._v2_fast_exact_profile is not None:
                self._v2_fast_exact_profile["ActionHeadCalls"] = (
                    self._v2_fast_exact_profile.get("ActionHeadCalls", 0.0) + 1.0
                )
            with self.autocast_context():
                station_logits = self.policy.station_head(
                    selected_task_emb,
                    station_embs.unsqueeze(0),
                    mask=None,
                    station_baseline_match=station_baseline_match,
                )
            # 非有限合法 logits 由统一分布入口拒绝。
            station_logits, station_usable = _finalize_action_logits(
                station_logits,
                station_mask,
                decision=f"v2_behavior.station[sample={sample_index}]",
            )
            if not bool(station_usable.item()):
                raise RuntimeError("v2 behavior replay 的 station 动作空间全屏蔽")

            station_dist = Categorical(logits=station_logits)
            station_lp = station_dist.log_prob(station_target)
            station_entropy = station_dist.entropy()
            normalized_station_entropy = self._normalized_categorical_entropy(
                station_entropy, station_mask
            )

            task_skill = torch.argmax(
                raw_task[task_id, 5 : 5 + num_skills]
            ).reshape(1)
            worker_skills = raw_worker[:, worker_layout.skill_slice]
            skill_invalid = ~(
                worker_skills.index_select(1, task_skill).squeeze(1) > 0.5
            )
            station_action_id = station_id + 1
            worker_locks = torch.argmax(
                raw_worker[:, worker_layout.lock_slice], dim=1
            )
            lock_invalid = (worker_locks != 0) & (worker_locks != station_action_id)
            current_mask = worker_queue_mask | skill_invalid | lock_invalid
            pressure = self._build_v2_pressure_context(
                task_features=raw_task,
                worker_features=raw_worker,
                task_present=None,
                task_action_invalid=task_mask,
                worker_present=None,
                worker_queue_invalid=worker_queue_mask,
                physical_predecessor_edges=self._physical_predecessor_edge_list(
                    batch, group_size
                )[sample_index : sample_index + 1],
            )
            team_state = self.policy.worker_head.initialize_v2_state(
                batch_size=1, device=self.device
            )
            worker_embs_i = worker_embs.unsqueeze(0)
            worker_baseline_member = build_worker_baseline_membership(
                None
                if baseline_snapshot is None
                else baseline_snapshot.get("baseline_team_edge_index"),
                selected_tasks=torch.tensor([task_id], device=self.device),
                num_workers=worker_embs.size(0),
                enabled=bool(
                    getattr(
                        self.config,
                        "reschedule_baseline_identity_conditioning",
                        False,
                    )
                ),
                device=self.device,
            )
            station_emb_i = station_embs[station_id].unsqueeze(0)
            demand = torch.tensor(
                [float(len(team_target_ids))],
                device=self.device,
            )
            decode_cache = self.policy.worker_head.build_v2_decode_cache(
                task_emb=selected_task_emb,
                station_emb=station_emb_i,
                global_context=global_i,
                worker_embs=worker_embs_i,
                pressure_context=pressure,
                demand=demand,
                candidate_skills=raw_worker[:, worker_layout.skill_slice].unsqueeze(0),
                task_required_skills=raw_task[
                    task_id, 5 : 5 + worker_layout.num_skill_types
                ].reshape(1, -1),
            )
            team_lp = torch.zeros_like(task_lp)
            team_entropy = torch.zeros_like(task_entropy)
            normalized_team_entropy = torch.zeros_like(normalized_task_entropy)
            for worker_id in team_target_ids:
                assert worker_id < worker_embs.shape[0]
                self.worker_pointer_v2_diagnostics.record_team_state(
                    selected_max_wait=team_state.selected_max_wait,
                    selected_capacity_sum=team_state.selected_capacity_sum,
                )
                with self.autocast_context():
                    worker_logits = self.policy.worker_head.forward_choice_v2(
                        task_emb=selected_task_emb,
                        station_emb=station_emb_i,
                        global_context=global_i,
                        worker_embs=worker_embs_i,
                        pressure_context=pressure,
                        team_state=team_state,
                        demand=demand,
                        mask=None,
                        decode_cache=decode_cache,
                        dynamic_eft_features=self._build_v2_dynamic_eft_features(
                            team_state=team_state,
                            worker_features=raw_worker,
                            station_features=raw_station,
                            selected_station=station_target,
                            task_duration=raw_task[task_id, 0].reshape(1),
                            demand=demand,
                            mask=current_mask.unsqueeze(0),
                        ),
                        worker_baseline_member=worker_baseline_member,
                    )
                # 非有限合法 logits 由统一分布入口拒绝。
                self._record_v2_marginal_scarcity(
                    self.policy.worker_head,
                    worker_invalid_mask=current_mask.unsqueeze(0),
                    selected_worker_index=worker_id,
                )
                worker_logits, worker_usable = _finalize_action_logits(
                    worker_logits,
                    current_mask.unsqueeze(0),
                    decision=(
                        f"v2_behavior.worker[sample={sample_index},worker={worker_id}]"
                    ),
                )
                if not bool(worker_usable.item()):
                    raise RuntimeError(
                        "v2 behavior replay 的 worker 动作空间全屏蔽"
                    )

                worker_dist = Categorical(logits=worker_logits)
                target = torch.tensor([worker_id], device=self.device)
                team_lp = team_lp + worker_dist.log_prob(target)
                step_entropy = worker_dist.entropy()
                team_entropy = team_entropy + step_entropy
                normalized_team_entropy = normalized_team_entropy + (
                    self._normalized_categorical_entropy(
                        step_entropy, current_mask.unsqueeze(0)
                    )
                )
                team_state = self.policy.worker_head.advance_v2_state(
                    team_state,
                    worker_embs_i[:, worker_id, :],
                    worker_skills[worker_id].unsqueeze(0),
                    selected_wait=torch.expm1(
                        raw_worker[worker_id, worker_layout.wait_idx].float().clamp_min(0.0)
                    ).reshape(1),
                    selected_capacity=(
                        raw_worker[worker_id, worker_layout.efficiency_idx].float()
                        * raw_worker[worker_id, worker_layout.fatigue_idx].float()
                    ).reshape(1),
                )
                current_mask = current_mask.clone()
                current_mask[worker_id] = True

            normalized_entropy = (
                normalized_task_entropy
                + 1.5 * normalized_station_entropy
                + 0.5 * normalized_team_entropy
            )
            outputs.append(
                {
                    "task": task_lp,
                    "station": station_lp,
                    "team": team_lp,
                    "entropy": task_entropy + station_entropy + team_entropy,
                    "normalized_entropy": normalized_entropy,
                    "state_value": state_values[sample_index].reshape(1),
                    "conditional_values": (
                        torch.cat(
                            [
                                conditional_value_predictions["task"],
                                conditional_value_predictions["station"],
                                conditional_value_predictions["worker"],
                            ],
                            dim=1,
                        )[sample_index].reshape(1, 3)
                        if conditional_value_predictions is not None
                        else None
                    ),
                }
            )
        assert len(outputs) == group_size
        return outputs

    def _run_v2_fast_exact_replay_update(
        self,
        memory: Any,
        env: Any,
        *,
        current_ep: int,
        advantages: torch.Tensor,
        rewards: torch.Tensor,
        old_logprobs: torch.Tensor,
        b_task: torch.Tensor,
        b_station: torch.Tensor,
        b_team: torch.Tensor,
        action_scope: str,
        fast_exact_builder: Any = None,
    ) -> dict[str, float]:
        """Fast-Exact 同形 GPU group 重放 PPO 更新。

        - 每个物理 group 经 GPUExactBatchBuilder 构建（布局元数据 CPU 缓存，
          热路径无 ``.cpu().tolist()`` / 逐样本 ``.item()``）；
        - 首次同形合同为 actor-only 无梯度预检（不计算 critic 与熵），仅用于
          first recompute contract；
        - 正式梯度前向同时产生当前 KL 所需 logprob 与 actor + critic PPO loss。
        """
        import math

        from training.worker_pointer_v2_replay import (
            check_first_recompute_contract,
            normalize_group_loss_sum,
            plan_behavior_groups,
            select_validation_groups,
        )
        from training.fast_exact_benchmark import summarize_group_sizes

        started = time.perf_counter()
        profile_enabled = bool(
            getattr(self.config, "worker_pointer_v2_fast_exact_profile", False)
        )
        self._v2_fast_exact_profile = {} if profile_enabled else None
        traces = list(getattr(memory, "worker_pointer_v2_behavior_traces", []) or [])
        if len(traces) != len(memory.states):
            raise RuntimeError(
                "v2 fast-exact 行为轨迹与 memory.states 数量不一致: "
                f"{len(traces)} vs {len(memory.states)}"
            )
        if any(trace is None for trace in traces):
            raise RuntimeError("v2 fast-exact 行为轨迹包含空项，禁止静默跳过样本")

        grouped: dict[tuple[int, int], list[tuple[int, Any]]] = {}
        for memory_index, trace in enumerate(traces):
            grouped.setdefault(trace.group_id, []).append((memory_index, trace))
        restored: list[list[tuple[int, Any]]] = []
        for group_id in sorted(grouped):
            members = sorted(grouped[group_id], key=lambda item: item[1].group_position)
            expected_size = int(members[0][1].group_size)
            positions = [int(item[1].group_position) for item in members]
            env_indices = [int(item[1].env_index) for item in members]
            if len(members) != expected_size or positions != list(range(expected_size)):
                raise RuntimeError(
                    f"v2 fast-exact 行为组不完整: group_id={group_id!r}"
                )
            if len(set(env_indices)) != len(env_indices):
                raise RuntimeError(
                    f"v2 fast-exact 行为组存在重复环境: group_id={group_id!r}"
                )
            if any(int(item[1].group_size) != expected_size for item in members):
                raise RuntimeError(
                    f"v2 fast-exact 行为组 group_size 不一致: group_id={group_id!r}"
                )
            restored.append(members)
        if not restored:
            raise RuntimeError("v2 fast-exact 行为重放：没有有效的行为组可重放")

        group_sizes = [len(group) for group in restored]
        group_upper_bound = int(
            getattr(self.config, "worker_pointer_v2_rollout_group_upper_bound", 4)
        )
        if max(group_sizes) > group_upper_bound:
            raise RuntimeError(
                "v2 fast-exact 行为组超过配置上限: "
                f"max={max(group_sizes)} upper_bound={group_upper_bound}"
            )
        logical_cap = resolve_v2_logical_batch_size(int(self.batch_size), self.config)
        accumulation_steps = max(1, int(self.accumulation_steps))
        seed = int(getattr(self.config, "seed", 42))
        threshold = 1.0e-3 if self.amp_dtype == torch.bfloat16 else 1.0e-4

        factorized_enabled = (
            action_scope == "operation_station_worker"
            and str(
                getattr(self.config, "conditional_head_baseline_mode", "off")
            )
            == "factorized"
        )
        old_component_logprobs = old_component_values = None
        component_active_mask = component_advantages = None
        if factorized_enabled:
            (
                old_component_logprobs,
                old_component_values,
                component_active_mask,
                component_advantages,
            ) = self._prepare_factorized_update_data(
                memory=memory,
                returns=rewards,
                b_station=b_station,
                b_team=b_team,
            )

        if fast_exact_builder is None:
            cached = getattr(self, "_v2_fast_exact_builder", None)
            if cached is None or getattr(cached, "env", None) is not env:
                from training.v2_fast_exact_batch import GPUExactBatchBuilder

                cached = GPUExactBatchBuilder(
                    config=self.config, env=env, device=self.device
                )
                self._v2_fast_exact_builder = cached
            fast_exact_builder = cached

        profile_builder_calls_before = int(
            getattr(fast_exact_builder, "build_calls", 0)
        )
        profile_template_hits_before = int(
            getattr(fast_exact_builder, "template_hits", 0)
        )
        profile_template_misses_before = int(
            getattr(fast_exact_builder, "template_misses", 0)
        )

        batching_mode = str(
            getattr(self.config, "worker_pointer_v2_fast_replay_batching", "physical_group")
        )
        encoder_batch_cap = max(
            1,
            int(
                getattr(
                    self.config,
                    "worker_pointer_v2_fast_replay_encoder_batch_cap",
                    16,
                )
            ),
        )
        if batching_mode not in {"physical_group", "logical_batch_v1"}:
            raise RuntimeError(
                "worker_pointer_v2_fast_replay_batching 配置无效: "
                f"{batching_mode!r}"
            )

        def _build_memory_group(memory_indices: list[int]) -> Any:
            if not memory_indices:
                raise ValueError("v2 fast-exact replay batch 不能为空")
            if self._v2_fast_exact_profile is not None:
                self._v2_fast_exact_profile_sync()
                build_started = time.perf_counter()
            fast_batch = self._build_v2_fast_exact_group(
                memory=memory,
                memory_indices=memory_indices,
                b_task=b_task,
                b_station=b_station,
                b_team=b_team,
                old_logprobs=old_logprobs,
                rewards=rewards,
                advantages=advantages,
                fast_exact_builder=fast_exact_builder,
            )
            if self._v2_fast_exact_profile is not None:
                self._v2_fast_exact_profile_sync()
                self._v2_fast_exact_profile_add_time("BuilderMs", build_started)
                self._v2_fast_exact_profile["BuilderCalls"] = (
                    self._v2_fast_exact_profile.get("BuilderCalls", 0.0) + 1.0
                )
            return fast_batch

        def _build_group(group_index: int) -> Any:
            return _build_memory_group(
                [item[0] for item in restored[group_index]]
            )

        def _plan_formal_replay_units(
            group_indices: list[int],
        ) -> list[list[tuple[int, Any]]]:
            if batching_mode == "physical_group":
                return [list(restored[group_index]) for group_index in group_indices]
            buckets: dict[int, list[list[tuple[int, Any]]]] = {}
            for group_index in group_indices:
                members = list(restored[group_index])
                if len(members) > encoder_batch_cap:
                    raise RuntimeError(
                        "Fast-Exact logical encoder batch cap 小于 physical group: "
                        f"group_size={len(members)} cap={encoder_batch_cap}"
                    )
                dataset_idx = int(memory.states[members[0][0]].get("dataset_idx", 0))
                dataset_units = buckets.setdefault(dataset_idx, [])
                if not dataset_units or len(dataset_units[-1]) + len(members) > encoder_batch_cap:
                    dataset_units.append([])
                dataset_units[-1].extend(members)
            return [unit for dataset_units in buckets.values() for unit in dataset_units]

        logical_batches_by_epoch = [
            plan_behavior_groups(
                group_sizes,
                logical_cap=logical_cap,
                seed=seed,
                current_step=int(self.current_step),
                epoch=epoch,
            )
            for epoch in range(self.k_epochs)
        ]
        if not logical_batches_by_epoch[0]:
            raise RuntimeError("v2 fast-exact 行为重放未生成逻辑 batch")

        # ---- 首次同形合同：actor-only 无梯度预检，仅用于 identity validation ----
        group_team_sizes = [
            [len(memory.actions[memory_index][2]) for memory_index, _trace in group]
            for group in restored
        ]
        validation_groups = select_validation_groups(
            group_team_sizes,
            logical_cap=logical_cap,
        )
        behavior_rows: list[tuple[float, float, float]] = []
        replayed_rows: list[tuple[float, float, float]] = []
        with torch.no_grad():
            for group_index in validation_groups:
                if self._v2_fast_exact_profile is not None:
                    self._v2_fast_exact_profile_sync()
                    precheck_started = time.perf_counter()
                outputs = self._replay_v2_fast_exact_group(
                    _build_group(group_index), actor_only=True
                )
                if self._v2_fast_exact_profile is not None:
                    self._v2_fast_exact_profile_sync()
                    self._v2_fast_exact_profile_add_time(
                        "PrecheckMs", precheck_started
                    )
                assert len(outputs) == len(restored[group_index])
                for output, (_memory_index, trace) in zip(
                    outputs, restored[group_index]
                ):
                    behavior_rows.append(
                        (float(trace.task_lp), float(trace.station_lp), float(trace.team_lp))
                    )
                    replayed_rows.append(
                        (
                            float(output["task"].detach().float().item()),
                            float(output["station"].detach().float().item()),
                            float(output["team"].detach().float().item()),
                        )
                    )
        first_contract_report = check_first_recompute_contract(
            behavior_rows,
            replayed_rows,
            max_abs_error=threshold,
        )

        decay_eps = max(1, int(self.config.entropy_decay_episodes))
        ent_progress = min(1.0, current_ep / decay_eps)
        c_ent = float(self.config.c_entropy_end) + (
            float(self.config.c_entropy) - float(self.config.c_entropy_end)
        ) * math.exp(-3.0 * ent_progress)
        c_policy = float(self.config.c_policy)
        c_value = float(self.config.c_value)
        progress = min(1.0, self.current_step / max(1, self.total_timesteps))
        eps_clip_end = float(getattr(self.config, "eps_clip_end", 0.05))
        current_eps_clip = self.eps_clip - progress * (self.eps_clip - eps_clip_end)

        metric_values: dict[str, list[torch.Tensor]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "normalized_entropy": [],
            "ratio": [],
            "approx_kl": [],
            "clip": [],
        }
        update_steps = 0
        logical_batch_count = 0
        physical_group_count = 0
        logical_batch_sizes: list[int] = []
        gradient_diagnostics_all: list[dict[str, float]] = []
        precheck_reused = 0
        stop_after_epoch = False
        optimizer_parameters = tuple(
            parameter
            for parameter_group in self.optimizer.param_groups
            for parameter in parameter_group["params"]
            if parameter.requires_grad
        )

        def _backward_formal_group(
            *,
            members: list[tuple[int, Any]],
            group_batch: Any,
            outputs: list[dict[str, torch.Tensor]],
            window_sample_count: int,
        ) -> None:
            """立即消费单个 group 的正式输出，避免跨 group 保留 autograd graph。"""
            sample_losses: list[torch.Tensor] = []
            assert len(outputs) == len(members)
            for local_index, (output, (memory_index, _trace)) in enumerate(
                zip(outputs, members)
            ):
                total_lp = output["task"] + output["station"] + output["team"]
                old_lp = group_batch.batch.y_logprob[local_index].reshape(1).float()
                _, _, ratio = self.compute_stable_log_ratio_and_ratio(
                    total_lp, old_lp
                )
                advantage = group_batch.batch.y_advantage[local_index].reshape(1)
                if factorized_enabled:
                    assert old_component_logprobs is not None
                    assert component_advantages is not None
                    assert component_active_mask is not None
                    current_component_logprobs = torch.stack(
                        [output["task"], output["station"], output["team"]], dim=1
                    )
                    policy_loss, _component_ratios, _component_losses = (
                        self.compute_factorized_clipped_surrogate(
                            current_logprobs=current_component_logprobs,
                            old_logprobs=old_component_logprobs[memory_index]
                            .reshape(1, 3)
                            .to(self.device),
                            advantages=component_advantages[memory_index]
                            .reshape(1, 3)
                            .to(self.device),
                            active_mask=component_active_mask[memory_index]
                            .reshape(1, 3)
                            .to(self.device),
                            clip_range=current_eps_clip,
                        )
                    )
                else:
                    surrogate = torch.minimum(
                        ratio * advantage,
                        torch.clamp(
                            ratio,
                            1.0 - current_eps_clip,
                            1.0 + current_eps_clip,
                        )
                        * advantage,
                    )
                    policy_loss = -surrogate.mean()
                old_value = (
                    group_batch.batch.y_value[local_index].reshape(1)
                    if hasattr(group_batch.batch, "y_value")
                    else None
                )
                value_loss = self.compute_value_loss(
                    state_values=output["state_value"],
                    returns=group_batch.batch.y_reward[local_index].reshape(1),
                    old_values=old_value,
                    clip_range=current_eps_clip,
                )
                if factorized_enabled:
                    assert old_component_values is not None
                    assert component_active_mask is not None
                    conditional_values = output.get("conditional_values")
                    assert conditional_values is not None
                    conditional_value_loss = self.compute_factorized_value_loss(
                        current_values=conditional_values.reshape(1, 3),
                        old_values=old_component_values[memory_index]
                        .reshape(1, 3)
                        .to(self.device),
                        targets=group_batch.batch.y_reward[local_index].reshape(1),
                        active_mask=component_active_mask[memory_index]
                        .reshape(1, 3)
                        .to(self.device),
                        clip_range=current_eps_clip,
                    )
                    value_loss = self.combine_factorized_value_loss(
                        base_value_loss=value_loss,
                        conditional_value_loss=conditional_value_loss,
                        coefficient=float(
                            getattr(
                                self.config,
                                "conditional_head_value_coef",
                                1.0,
                            )
                        ),
                    )
                normalized_entropy = output["normalized_entropy"].mean()
                sample_loss = (
                    c_policy * policy_loss
                    + c_value * value_loss
                    - c_ent * normalized_entropy
                )
                sample_losses.append(sample_loss)
                metric_values["policy_loss"].append(
                    policy_loss.detach().float().reshape(1)
                )
                metric_values["value_loss"].append(
                    value_loss.detach().float().reshape(1)
                )
                metric_values["entropy"].append(
                    output["entropy"].detach().float().reshape(1)
                )
                metric_values["normalized_entropy"].append(
                    normalized_entropy.detach().float().reshape(1)
                )
                metric_values["ratio"].append(ratio.detach().float().reshape(1))
            assert sample_losses
            group_loss_sum = torch.stack(sample_losses).sum()
            scaled_loss = normalize_group_loss_sum(
                group_loss_sum,
                window_sample_count=window_sample_count,
            )
            if self._v2_fast_exact_profile is not None:
                self._v2_fast_exact_profile_sync()
                backward_started = time.perf_counter()
            self.scaler.scale(scaled_loss).backward()
            if self._v2_fast_exact_profile is not None:
                self._v2_fast_exact_profile_sync()
                self._v2_fast_exact_profile_add_time(
                    "BackwardMs", backward_started
                )

        for epoch, logical_batches in enumerate(logical_batches_by_epoch):
            epoch_kl_sum = 0.0
            epoch_samples = 0
            batch_sample_counts = [
                sum(group_sizes[group_index] for group_index in batch_groups)
                for batch_groups in logical_batches
            ]
            logical_batch_sizes.extend(batch_sample_counts)
            for window_start in range(0, len(logical_batches), accumulation_steps):
                window_batches = logical_batches[
                    window_start : window_start + accumulation_steps
                ]
                window_sample_count = sum(
                    batch_sample_counts[window_start + offset]
                    for offset in range(len(window_batches))
                )
                assert window_sample_count > 0
                self.optimizer.zero_grad()
                for batch_offset, batch_groups in enumerate(window_batches):
                    logical_batch_count += 1
                    gradient_before_batch = tuple(
                        (
                            parameter,
                            parameter.grad.detach().clone()
                            if parameter.grad is not None
                            else None,
                        )
                        for parameter in optimizer_parameters
                    )
                    # formal replay 输出同时用于当前 KL 和 PPO loss，禁止跨 step 缓存 logits。
                    formal_total_lp: list[torch.Tensor] = []
                    formal_old_lp: list[torch.Tensor] = []
                    physical_group_count += len(batch_groups)
                    replay_units = _plan_formal_replay_units(batch_groups)
                    for replay_members in replay_units:
                        group_batch = _build_memory_group(
                            [item[0] for item in replay_members]
                        )
                        if self._v2_fast_exact_profile is not None:
                            self._v2_fast_exact_profile_sync()
                            formal_started = time.perf_counter()
                        outputs = self._replay_v2_fast_exact_group(group_batch)
                        if self._v2_fast_exact_profile is not None:
                            self._v2_fast_exact_profile_sync()
                            self._v2_fast_exact_profile_add_time(
                                "FormalReplayMs", formal_started
                            )
                            self._v2_fast_exact_profile["FormalReplayCalls"] = (
                                self._v2_fast_exact_profile.get(
                                    "FormalReplayCalls", 0.0
                                )
                                + 1.0
                            )
                            self._v2_fast_exact_profile["ReplaySampleCount"] = (
                                self._v2_fast_exact_profile.get(
                                    "ReplaySampleCount", 0.0
                                )
                                + float(len(outputs))
                            )
                        for output, (memory_index, _trace) in zip(
                            outputs, replay_members
                        ):
                            formal_total_lp.append(
                                (
                                    output["task"]
                                    + output["station"]
                                    + output["team"]
                                ).detach()
                            )
                            formal_old_lp.append(
                                old_logprobs[memory_index]
                                .to(self.device)
                                .reshape(1)
                            )
                        _backward_formal_group(
                            members=replay_members,
                            group_batch=group_batch,
                            outputs=outputs,
                            window_sample_count=window_sample_count,
                        )
                    with torch.no_grad():
                        batch_total_lp = torch.cat(formal_total_lp).float()
                        batch_old_lp = torch.cat(formal_old_lp).float()
                        _, safe_log_ratio, batch_ratios = (
                            self.compute_stable_log_ratio_and_ratio(
                                batch_total_lp, batch_old_lp
                            )
                        )
                        batch_kl_values = (batch_ratios - 1.0) - safe_log_ratio
                        batch_kl = batch_kl_values.mean()
                        loss_scale = 0.01 if batch_kl > self.kl_early_stop else 1.0
                        epoch_kl_sum += float(batch_kl_values.sum().item())
                        epoch_samples += int(batch_kl_values.numel())
                        metric_values["approx_kl"].append(
                            batch_kl_values.detach().float()
                        )
                        metric_values["clip"].append(
                            (torch.abs(batch_ratios - 1.0) > current_eps_clip)
                            .float()
                            .detach()
                        )
                    self._rescale_fast_exact_gradient_delta(
                        gradient_before_batch,
                        loss_scale,
                    )

                self.scaler.unscale_(self.optimizer)
                gradient_diagnostics = self._collect_gradient_diagnostics(
                    self.policy.named_parameters()
                )
                gradient_diagnostics_all.append(gradient_diagnostics)
                if gradient_diagnostics["finite"] < 1.0:
                    raise RuntimeError(
                        "v2 fast-exact 行为重放产生非有限梯度，拒绝 optimizer.step"
                    )
                torch.nn.utils.clip_grad_norm_(self.actor_parameters, max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(
                    self.critic_parameters,
                    max_norm=float(self.config.clip_v_grad_norm),
                )
                if self._v2_fast_exact_profile is not None:
                    self._v2_fast_exact_profile_sync()
                    optimizer_started = time.perf_counter()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                if self._v2_fast_exact_profile is not None:
                    self._v2_fast_exact_profile_sync()
                    self._v2_fast_exact_profile_add_time(
                        "OptimizerMs", optimizer_started
                    )
                update_steps += 1
                # actor-only 输出只对生成它们的策略参数版本有效。每次参数更新后
                # 立即失效，防止后续累计窗口或 PPO epoch 使用旧策略计算 KL。
                # 窗口边界同步：模板字段张量在本 window 释放后可能被下一 window
                # 的 build 直接复用；在无同步的异步执行下（autograd 引擎跨 stream）
                # 会与上一 window 的尾部 kernel 竞争同一块显存，表现为后续 GAT
                # 前向随机段错误。每个累积窗口收尾时强制同步一次，开销可忽略。
                torch.cuda.synchronize(self.device)

            mean_epoch_kl = epoch_kl_sum / max(1, epoch_samples)
            if mean_epoch_kl > self.kl_early_stop:
                stop_after_epoch = True
            if stop_after_epoch:
                break

        self.current_step += 1
        if getattr(self, "use_ema", False) and hasattr(self, "ema_policy"):
            with torch.no_grad():
                for ema_parameter, parameter in zip(
                    self.ema_policy.parameters(), self.policy.parameters()
                ):
                    ema_parameter.data.copy_(
                        self.ema_decay * ema_parameter.data
                        + (1.0 - self.ema_decay) * parameter.data
                    )

        def _sample_mean(name: str) -> float:
            values = metric_values[name]
            if not values:
                return 0.0
            flattened = torch.cat([value.reshape(-1) for value in values])
            return float(flattened.mean().cpu().item())

        elapsed = time.perf_counter() - started
        metrics: dict[str, float] = {
            "PPO/UpdateSteps": float(update_steps),
            "PPO/Loss": _sample_mean("policy_loss"),
            "PPO/ValueLoss": _sample_mean("value_loss"),
            "PPO/Entropy": _sample_mean("entropy"),
            "PPO/NormalizedEntropy": _sample_mean("normalized_entropy"),
            "PPO/ApproxKL": _sample_mean("approx_kl"),
            "PPO/ClipFraction": _sample_mean("clip"),
            "PPO/RatioMean": _sample_mean("ratio"),
            "V2/FastExact/BehaviorReplayGroups": float(len(restored)),
            "V2/FastExact/BehaviorReplaySamples": float(sum(group_sizes)),
            "V2/FastExact/PrecheckGroups": float(len(validation_groups)),
            "V2/FastExact/PrecheckReusedGroups": float(precheck_reused),
            "V2/FastExact/LogicalBatchCount": float(logical_batch_count),
            "V2/FastExact/PhysicalGroupCount": float(physical_group_count),
            "V2/FastExact/MaxPhysicalGroup": float(max(group_sizes)),
            "V2/FastExact/ReplayUpdateSeconds": float(elapsed),
            "V2/FastExact/LogicalBatchMeanSize": float(sum(logical_batch_sizes))
            / max(1, len(logical_batch_sizes)),
            "V2/FirstContractTaskMAE": float(first_contract_report["task"]["mae"]),
            "V2/FirstContractTaskMaxAE": float(
                first_contract_report["task"]["max_abs_error"]
            ),
            "V2/FirstContractStationMAE": float(
                first_contract_report["station"]["mae"]
            ),
            "V2/FirstContractStationMaxAE": float(
                first_contract_report["station"]["max_abs_error"]
            ),
            "V2/FirstContractTeamMAE": float(first_contract_report["team"]["mae"]),
            "V2/FirstContractTeamMaxAE": float(
                first_contract_report["team"]["max_abs_error"]
            ),
            "V2/FirstContractTotalMAE": float(first_contract_report["total"]["mae"]),
            "V2/FirstContractTotalMaxAE": float(
                first_contract_report["total"]["max_abs_error"]
            ),
        }
        if gradient_diagnostics_all:
            metrics["Gradient/Finite"] = min(
                item["finite"] for item in gradient_diagnostics_all
            )
            metrics["Gradient/V2Norm"] = sum(
                item["v2_grad_norm"] for item in gradient_diagnostics_all
            ) / len(gradient_diagnostics_all)
            metrics["Gradient/V2Coverage"] = sum(
                item["v2_gradient_coverage"] for item in gradient_diagnostics_all
            ) / len(gradient_diagnostics_all)
        metrics.update(
            {
                "Policy/ApproxKL": metrics["PPO/ApproxKL"],
                "PPO/GradientsFinite": metrics.get("Gradient/Finite", 0.0),
                "PointerV2/GradientNorm": metrics.get("Gradient/V2Norm", 0.0),
                "PointerV2/GradientCoverage": metrics.get(
                    "Gradient/V2Coverage", 0.0
                ),
                "PointerV2/PPOFirstRecomputeMAE": metrics[
                    "V2/FirstContractTotalMAE"
                ],
                "PointerV2/PPOFirstRecomputeMaxAE": metrics[
                    "V2/FirstContractTotalMaxAE"
                ],
                "PointerV2/AutocastEnabled": float(self.amp_enabled),
                "PointerV2/AutocastBF16": float(
                    self.amp_enabled and self.amp_dtype == torch.bfloat16
                ),
                "PointerV2/GradScalerEnabled": float(self.scaler.is_enabled()),
                "PointerV2/NonFiniteCount": 0.0,
            }
        )
        if self._v2_fast_exact_profile is not None:
            group_summary = summarize_group_sizes(group_sizes)
            profile = self._v2_fast_exact_profile
            metrics.update(
                {
                    "V2/FastExact/Profile/PhysicalGroupCount": float(
                        physical_group_count
                    ),
                    "V2/FastExact/Profile/LogicalBatchCount": float(
                        logical_batch_count
                    ),
                    "V2/FastExact/Profile/PhysicalGroupMeanSize": group_summary[
                        "mean"
                    ],
                    "V2/FastExact/Profile/PhysicalGroupP50Size": group_summary[
                        "p50"
                    ],
                    "V2/FastExact/Profile/PhysicalGroupP95Size": group_summary[
                        "p95"
                    ],
                    "V2/FastExact/Profile/BuilderCalls": profile.get(
                        "BuilderCalls",
                        float(
                            int(getattr(fast_exact_builder, "build_calls", 0))
                            - profile_builder_calls_before
                        ),
                    ),
                    "V2/FastExact/Profile/BuilderMs": profile.get("BuilderMs", 0.0),
                    "V2/FastExact/Profile/TemplateHits": float(
                        int(getattr(fast_exact_builder, "template_hits", 0))
                        - profile_template_hits_before
                    ),
                    "V2/FastExact/Profile/TemplateMisses": float(
                        int(getattr(fast_exact_builder, "template_misses", 0))
                        - profile_template_misses_before
                    ),
                    "V2/FastExact/Profile/EncoderCalls": profile.get(
                        "EncoderCalls", 0.0
                    ),
                    "V2/FastExact/Profile/EncoderMs": profile.get(
                        "EncoderMs", 0.0
                    ),
                    "V2/FastExact/Profile/ActionHeadCalls": profile.get(
                        "ActionHeadCalls", 0.0
                    ),
                    "V2/FastExact/Profile/ActionHeadMs": profile.get(
                        "ActionHeadMs", 0.0
                    ),
                    "V2/FastExact/Profile/WorkerPointerCalls": profile.get(
                        "WorkerPointerCalls", 0.0
                    ),
                    "V2/FastExact/Profile/WorkerPointerMs": profile.get(
                        "WorkerPointerMs", 0.0
                    ),
                    "V2/FastExact/Profile/PrecheckMs": profile.get(
                        "PrecheckMs", 0.0
                    ),
                    "V2/FastExact/Profile/FormalReplayCalls": profile.get(
                        "FormalReplayCalls", 0.0
                    ),
                    "V2/FastExact/Profile/FormalReplayMs": profile.get(
                        "FormalReplayMs", 0.0
                    ),
                    "V2/FastExact/Profile/BackwardMs": profile.get(
                        "BackwardMs", 0.0
                    ),
                    "V2/FastExact/Profile/OptimizerMs": profile.get(
                        "OptimizerMs", 0.0
                    ),
                    "V2/FastExact/Profile/ReplaySamplesPerSec": profile.get(
                        "ReplaySampleCount", 0.0
                    )
                    / max(elapsed, 1.0e-9),
                }
            )
            self._v2_fast_exact_profile = None
        return metrics

    def _run_v2_behavior_replay_update(
        self,
        memory: Any,
        env: Any,
        *,
        current_ep: int,
        advantages: torch.Tensor,
        rewards: torch.Tensor,
        old_logprobs: torch.Tensor,
        b_task: torch.Tensor,
        b_station: torch.Tensor,
        b_team: torch.Tensor,
        action_scope: str,
    ) -> dict[str, float]:
        """v2 行为组同形重放 PPO 更新。

        每个物理 group 按 rollout 原形状（group size 图）前向 GNN/critic，
        三个动作 head 以 B=1 逐样本重算；逻辑 batch 内按样本均值聚合，
        accumulation 窗口（≤accumulation_steps 个逻辑 batch）梯度按窗口
        实际总样本数归一化。首次 PPO update 在 backward 前对首个逻辑
        batch 执行无梯度同形重放合同（MaxAE ≤ 1e-3/1e-4），超阈值 fail-closed。
        """
        import math

        from training.worker_pointer_v2_replay import (
            check_first_recompute_contract,
            normalize_group_loss_sum,
            plan_behavior_groups,
            select_validation_groups,
        )

        started = time.perf_counter()
        traces = list(getattr(memory, "worker_pointer_v2_behavior_traces", []) or [])
        if len(traces) != len(memory.states):
            raise RuntimeError(
                "v2 行为轨迹与 memory.states 数量不一致: "
                f"{len(traces)} vs {len(memory.states)}"
            )
        if any(trace is None for trace in traces):
            raise RuntimeError("v2 行为轨迹包含空项，禁止静默跳过样本")

        grouped: dict[tuple[int, int], list[tuple[int, Any]]] = {}
        for memory_index, trace in enumerate(traces):
            grouped.setdefault(trace.group_id, []).append((memory_index, trace))
        restored: list[list[tuple[int, Any]]] = []
        for group_id in sorted(grouped):
            members = sorted(grouped[group_id], key=lambda item: item[1].group_position)
            expected_size = int(members[0][1].group_size)
            positions = [int(item[1].group_position) for item in members]
            env_indices = [int(item[1].env_index) for item in members]
            if len(members) != expected_size or positions != list(range(expected_size)):
                raise RuntimeError(f"v2 行为组不完整: group_id={group_id!r}")
            if len(set(env_indices)) != len(env_indices):
                raise RuntimeError(f"v2 行为组存在重复环境: group_id={group_id!r}")
            if any(int(item[1].group_size) != expected_size for item in members):
                raise RuntimeError(f"v2 行为组 group_size 不一致: group_id={group_id!r}")
            restored.append(members)
        if not restored:
            raise RuntimeError("v2 行为重放：没有有效的行为组可重放")

        group_sizes = [len(group) for group in restored]
        group_upper_bound = int(
            getattr(self.config, "worker_pointer_v2_rollout_group_upper_bound", 4)
        )
        if max(group_sizes) > group_upper_bound:
            raise RuntimeError(
                "v2 行为组超过配置上限: "
                f"max={max(group_sizes)} upper_bound={group_upper_bound}"
            )
        requested_logical_cap = int(self.batch_size)
        logical_cap = resolve_v2_logical_batch_size(
            int(self.batch_size),
            self.config,
        )
        accumulation_steps = max(1, int(self.accumulation_steps))
        seed = int(getattr(self.config, "seed", 42))
        threshold = 1.0e-3 if self.amp_dtype == torch.bfloat16 else 1.0e-4

        factorized_enabled = (
            action_scope == "operation_station_worker"
            and str(
                getattr(self.config, "conditional_head_baseline_mode", "off")
            )
            == "factorized"
        )
        old_component_logprobs = old_component_values = None
        component_active_mask = component_advantages = None
        if factorized_enabled:
            (
                old_component_logprobs,
                old_component_values,
                component_active_mask,
                component_advantages,
            ) = self._prepare_factorized_update_data(
                memory=memory,
                returns=rewards,
                b_station=b_station,
                b_team=b_team,
            )

        def _build_group(group_index: int) -> Any:
            memory_indices = [item[0] for item in restored[group_index]]
            return self._build_v2_behavior_group_batch(
                memory=memory,
                env=env,
                memory_indices=memory_indices,
                b_task=b_task,
                b_station=b_station,
                b_team=b_team,
                old_logprobs=old_logprobs,
                rewards=rewards,
                advantages=advantages,
            )

        logical_batches_by_epoch = [
            plan_behavior_groups(
                group_sizes,
                logical_cap=logical_cap,
                seed=seed,
                current_step=int(self.current_step),
                epoch=epoch,
            )
            for epoch in range(self.k_epochs)
        ]
        if not logical_batches_by_epoch[0]:
            raise RuntimeError("v2 行为重放未生成逻辑 batch")

        # 首次合同在 optimizer.zero_grad/backward/step 前完成，失败时不产生部分更新。
        group_team_sizes = [
            [len(memory.actions[memory_index][2]) for memory_index, _trace in group]
            for group in restored
        ]
        validation_groups = select_validation_groups(
            group_team_sizes,
            logical_cap=logical_cap,
        )
        behavior_rows: list[tuple[float, float, float]] = []
        replayed_rows: list[tuple[float, float, float]] = []
        with torch.no_grad():
            for group_index in validation_groups:
                outputs = self._replay_v2_behavior_group(_build_group(group_index))
                assert len(outputs) == len(restored[group_index])
                for output, (_memory_index, trace) in zip(
                    outputs, restored[group_index]
                ):
                    behavior_rows.append(
                        (float(trace.task_lp), float(trace.station_lp), float(trace.team_lp))
                    )
                    replayed_rows.append(
                        (
                            float(output["task"].detach().float().item()),
                            float(output["station"].detach().float().item()),
                            float(output["team"].detach().float().item()),
                        )
                    )
        first_contract_report = check_first_recompute_contract(
            behavior_rows,
            replayed_rows,
            max_abs_error=threshold,
        )

        decay_eps = max(1, int(self.config.entropy_decay_episodes))
        ent_progress = min(1.0, current_ep / decay_eps)
        c_ent = float(self.config.c_entropy_end) + (
            float(self.config.c_entropy) - float(self.config.c_entropy_end)
        ) * math.exp(-3.0 * ent_progress)
        c_policy = float(self.config.c_policy)
        c_value = float(self.config.c_value)
        progress = min(1.0, self.current_step / max(1, self.total_timesteps))
        eps_clip_end = float(getattr(self.config, "eps_clip_end", 0.05))
        current_eps_clip = self.eps_clip - progress * (self.eps_clip - eps_clip_end)

        metric_values: dict[str, list[torch.Tensor]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "normalized_entropy": [],
            "ratio": [],
            "approx_kl": [],
            "clip": [],
        }
        update_steps = 0
        logical_batch_count = 0
        physical_group_count = 0
        logical_batch_sizes: list[int] = []
        gradient_diagnostics_all: list[dict[str, float]] = []
        stop_after_epoch = False

        for epoch, logical_batches in enumerate(logical_batches_by_epoch):
            epoch_kl_sum = 0.0
            epoch_samples = 0
            batch_sample_counts = [
                sum(group_sizes[group_index] for group_index in batch_groups)
                for batch_groups in logical_batches
            ]
            logical_batch_sizes.extend(batch_sample_counts)
            for window_start in range(0, len(logical_batches), accumulation_steps):
                window_batches = logical_batches[
                    window_start : window_start + accumulation_steps
                ]
                window_sample_count = sum(
                    batch_sample_counts[window_start + offset]
                    for offset in range(len(window_batches))
                )
                assert window_sample_count > 0
                self.optimizer.zero_grad()
                for batch_offset, batch_groups in enumerate(window_batches):
                    logical_batch_count += 1
                    batch_sample_count = batch_sample_counts[
                        window_start + batch_offset
                    ]
                    prepass_total_lp: list[torch.Tensor] = []
                    prepass_old_lp: list[torch.Tensor] = []
                    # 每个逻辑批先无梯度重放，统一计算 KL 熔断尺度。
                    with torch.no_grad():
                        for group_index in batch_groups:
                            outputs = self._replay_v2_behavior_group(
                                _build_group(group_index)
                            )
                            for output, (memory_index, _trace) in zip(
                                outputs, restored[group_index]
                            ):
                                prepass_total_lp.append(
                                    output["task"] + output["station"] + output["team"]
                                )
                                prepass_old_lp.append(
                                    old_logprobs[memory_index]
                                    .to(self.device)
                                    .reshape(1)
                                )
                        batch_total_lp = torch.cat(prepass_total_lp).float()
                        batch_old_lp = torch.cat(prepass_old_lp).float()
                        _, safe_log_ratio, batch_ratios = (
                            self.compute_stable_log_ratio_and_ratio(
                                batch_total_lp, batch_old_lp
                            )
                        )
                        batch_kl_values = (batch_ratios - 1.0) - safe_log_ratio
                        batch_kl = batch_kl_values.mean()
                        loss_scale = 0.01 if batch_kl > self.kl_early_stop else 1.0
                        epoch_kl_sum += float(batch_kl_values.sum().item())
                        epoch_samples += int(batch_kl_values.numel())
                        metric_values["approx_kl"].append(
                            batch_kl_values.detach().float()
                        )
                        metric_values["clip"].append(
                            (torch.abs(batch_ratios - 1.0) > current_eps_clip)
                            .float()
                            .detach()
                        )

                    # 物理组逐个保留 rollout encoder 形状；损失以样本和/窗口实际样本数反传。
                    for group_index in batch_groups:
                        physical_group_count += 1
                        group_batch = _build_group(group_index)
                        outputs = self._replay_v2_behavior_group(group_batch)
                        sample_losses: list[torch.Tensor] = []
                        for local_index, (output, (memory_index, _trace)) in enumerate(
                            zip(outputs, restored[group_index])
                        ):
                            total_lp = output["task"] + output["station"] + output["team"]
                            old_lp = group_batch.y_logprob[local_index].reshape(1).float()
                            _, _, ratio = self.compute_stable_log_ratio_and_ratio(
                                total_lp, old_lp
                            )
                            advantage = group_batch.y_advantage[local_index].reshape(1)
                            if factorized_enabled:
                                assert old_component_logprobs is not None
                                assert component_advantages is not None
                                assert component_active_mask is not None
                                current_component_logprobs = torch.stack(
                                    [output["task"], output["station"], output["team"]],
                                    dim=1,
                                )
                                policy_loss, _component_ratios, _component_losses = (
                                    self.compute_factorized_clipped_surrogate(
                                        current_logprobs=current_component_logprobs,
                                        old_logprobs=old_component_logprobs[
                                            memory_index
                                        ].reshape(1, 3).to(self.device),
                                        advantages=component_advantages[
                                            memory_index
                                        ].reshape(1, 3).to(self.device),
                                        active_mask=component_active_mask[
                                            memory_index
                                        ].reshape(1, 3).to(self.device),
                                        clip_range=current_eps_clip,
                                    )
                                )
                            else:
                                surrogate = torch.minimum(
                                    ratio * advantage,
                                    torch.clamp(
                                        ratio,
                                        1.0 - current_eps_clip,
                                        1.0 + current_eps_clip,
                                    )
                                    * advantage,
                                )
                                policy_loss = -surrogate.mean()
                            old_value = (
                                group_batch.y_value[local_index].reshape(1)
                                if hasattr(group_batch, "y_value")
                                else None
                            )
                            value_loss = self.compute_value_loss(
                                state_values=output["state_value"],
                                returns=group_batch.y_reward[local_index].reshape(1),
                                old_values=old_value,
                                clip_range=current_eps_clip,
                            )
                            if factorized_enabled:
                                assert old_component_values is not None
                                assert component_active_mask is not None
                                conditional_values = output.get("conditional_values")
                                assert conditional_values is not None
                                conditional_value_loss = (
                                    self.compute_factorized_value_loss(
                                        current_values=conditional_values.reshape(1, 3),
                                        old_values=old_component_values[
                                            memory_index
                                        ].reshape(1, 3).to(self.device),
                                        targets=group_batch.y_reward[
                                            local_index
                                        ].reshape(1),
                                        active_mask=component_active_mask[
                                            memory_index
                                        ].reshape(1, 3).to(self.device),
                                        clip_range=current_eps_clip,
                                    )
                                )
                                value_loss = self.combine_factorized_value_loss(
                                    base_value_loss=value_loss,
                                    conditional_value_loss=conditional_value_loss,
                                    coefficient=float(
                                        getattr(
                                            self.config,
                                            "conditional_head_value_coef",
                                            1.0,
                                        )
                                    ),
                                )
                            normalized_entropy = output["normalized_entropy"].mean()
                            sample_loss = (
                                c_policy * policy_loss
                                + c_value * value_loss
                                - c_ent * normalized_entropy
                            )
                            sample_losses.append(sample_loss)
                            metric_values["policy_loss"].append(
                                policy_loss.detach().float().reshape(1)
                            )
                            metric_values["value_loss"].append(
                                value_loss.detach().float().reshape(1)
                            )
                            metric_values["entropy"].append(
                                output["entropy"].detach().float().reshape(1)
                            )
                            metric_values["normalized_entropy"].append(
                                normalized_entropy.detach().float().reshape(1)
                            )
                            metric_values["ratio"].append(
                                ratio.detach().float().reshape(1)
                            )
                        group_loss_sum = torch.stack(sample_losses).sum()
                        scaled_loss = normalize_group_loss_sum(
                            group_loss_sum * float(loss_scale),
                            window_sample_count=window_sample_count,
                        )
                        self.scaler.scale(scaled_loss).backward()

                self.scaler.unscale_(self.optimizer)
                gradient_diagnostics = self._collect_gradient_diagnostics(
                    self.policy.named_parameters()
                )
                gradient_diagnostics_all.append(gradient_diagnostics)
                if gradient_diagnostics["finite"] < 1.0:
                    raise RuntimeError("v2 行为重放产生非有限梯度，拒绝 optimizer.step")
                torch.nn.utils.clip_grad_norm_(self.actor_parameters, max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(
                    self.critic_parameters,
                    max_norm=float(self.config.clip_v_grad_norm),
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                update_steps += 1

            mean_epoch_kl = epoch_kl_sum / max(1, epoch_samples)
            if mean_epoch_kl > self.kl_early_stop:
                stop_after_epoch = True
            if stop_after_epoch:
                break

        self.current_step += 1
        if getattr(self, "use_ema", False) and hasattr(self, "ema_policy"):
            with torch.no_grad():
                for ema_parameter, parameter in zip(
                    self.ema_policy.parameters(), self.policy.parameters()
                ):
                    ema_parameter.data.copy_(
                        self.ema_decay * ema_parameter.data
                        + (1.0 - self.ema_decay) * parameter.data
                    )

        def _sample_mean(name: str) -> float:
            values = metric_values[name]
            if not values:
                return 0.0
            flattened = torch.cat([value.reshape(-1) for value in values])
            return float(flattened.mean().cpu().item())

        elapsed = time.perf_counter() - started
        metrics: dict[str, float] = {
            "PPO/UpdateSteps": float(update_steps),
            "PPO/Loss": _sample_mean("policy_loss"),
            "PPO/ValueLoss": _sample_mean("value_loss"),
            "PPO/Entropy": _sample_mean("entropy"),
            "PPO/NormalizedEntropy": _sample_mean("normalized_entropy"),
            "PPO/ApproxKL": _sample_mean("approx_kl"),
            "PPO/ClipFraction": _sample_mean("clip"),
            "PPO/RatioMean": _sample_mean("ratio"),
            "V2/BehaviorReplayGroups": float(len(restored)),
            "V2/BehaviorReplaySamples": float(sum(group_sizes)),
            "V2/ReplayRequestedLogicalCap": float(requested_logical_cap),
            "V2/ReplayEffectiveLogicalCap": float(logical_cap),
            "V2/ReplayLogicalBatchCount": float(logical_batch_count),
            "V2/ReplayLogicalBatchMeanSize": float(sum(logical_batch_sizes))
            / max(1, len(logical_batch_sizes)),
            "V2/ReplayLogicalBatchMinSize": float(min(logical_batch_sizes)),
            "V2/ReplayLogicalBatchMaxSize": float(max(logical_batch_sizes)),
            "V2/ReplayPhysicalGroupCount": float(physical_group_count),
            "V2/ReplayMaxPhysicalGroup": float(max(group_sizes)),
            "V2/ReplayGroupIntegrity": 1.0,
            "V2/ReplayUpdateSeconds": float(elapsed),
            "V2/ReplayGroupSPS": float(physical_group_count) / max(elapsed, 1.0e-9),
            "V2/ReplayAccumulationSteps": float(accumulation_steps),
            "V2/FirstContractTaskMAE": float(first_contract_report["task"]["mae"]),
            "V2/FirstContractTaskMaxAE": float(
                first_contract_report["task"]["max_abs_error"]
            ),
            "V2/FirstContractStationMAE": float(
                first_contract_report["station"]["mae"]
            ),
            "V2/FirstContractStationMaxAE": float(
                first_contract_report["station"]["max_abs_error"]
            ),
            "V2/FirstContractTeamMAE": float(first_contract_report["team"]["mae"]),
            "V2/FirstContractTeamMaxAE": float(
                first_contract_report["team"]["max_abs_error"]
            ),
            "V2/FirstContractTotalMAE": float(first_contract_report["total"]["mae"]),
            "V2/FirstContractTotalMaxAE": float(
                first_contract_report["total"]["max_abs_error"]
            ),
        }
        if gradient_diagnostics_all:
            metrics["Gradient/Finite"] = min(
                item["finite"] for item in gradient_diagnostics_all
            )
            metrics["Gradient/V2Norm"] = sum(
                item["v2_grad_norm"] for item in gradient_diagnostics_all
            ) / len(gradient_diagnostics_all)
            metrics["Gradient/V2Coverage"] = sum(
                item["v2_gradient_coverage"] for item in gradient_diagnostics_all
            ) / len(gradient_diagnostics_all)
        metrics.update(
            {
                # 保留已有训练审计器使用的稳定别名，同时输出上面的 replay 专属细粒度指标。
                "Policy/ApproxKL": metrics["PPO/ApproxKL"],
                "PPO/GradientsFinite": metrics.get("Gradient/Finite", 0.0),
                "PointerV2/GradientNorm": metrics.get("Gradient/V2Norm", 0.0),
                "PointerV2/GradientCoverage": metrics.get(
                    "Gradient/V2Coverage", 0.0
                ),
                "PointerV2/PPOFirstRecomputeMAE": metrics[
                    "V2/FirstContractTotalMAE"
                ],
                "PointerV2/PPOFirstRecomputeMaxAE": metrics[
                    "V2/FirstContractTotalMaxAE"
                ],
                "PointerV2/AutocastEnabled": float(self.amp_enabled),
                "PointerV2/AutocastBF16": float(
                    self.amp_enabled and self.amp_dtype == torch.bfloat16
                ),
                "PointerV2/GradScalerEnabled": float(
                    self.scaler.is_enabled()
                ),
                "PointerV2/NonFiniteCount": 0.0,
            }
        )
        return metrics

    def _build_v2_fast_exact_group(
        self,
        *,
        memory: Any,
        memory_indices: list[int],
        b_task: torch.Tensor,
        b_station: torch.Tensor,
        b_team: torch.Tensor,
        old_logprobs: torch.Tensor,
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        fast_exact_builder: Any,
    ) -> Any:
        """Fast-Exact：用 GPU 模板构建组 Batch 并绑定 PPO 动作目标。"""
        if not memory_indices:
            raise ValueError("v2 fast-exact 行为组不能为空")
        snapshots = [memory.states[int(index)] for index in memory_indices]
        masks = [memory.masks[int(index)] for index in memory_indices]
        fast_batch = fast_exact_builder.build(
            snapshots,
            masks=masks,
            memory_indices=list(memory_indices),
        )
        batch = fast_batch.batch
        max_team = int(b_team.shape[1]) if b_team.ndim == 2 else 1
        indices_t = torch.tensor(
            memory_indices, dtype=torch.long, device=self.device
        )
        batch.y_task = b_task.to(self.device)[indices_t]
        batch.y_station = b_station.to(self.device)[indices_t]
        batch.y_team = b_team.to(self.device)[indices_t, :max_team]
        batch.y_logprob = old_logprobs.to(self.device)[indices_t]
        for field_name in (
            "old_task_logprob",
            "old_station_logprob",
            "old_team_logprob",
            "old_V_task",
            "old_V_station",
            "old_V_worker",
        ):
            values = getattr(memory, field_name, [])
            setattr(
                batch,
                f"y_{field_name}",
                torch.tensor(
                    [
                        float(values[int(index)])
                        if int(index) < len(values) and values[int(index)] is not None
                        else 0.0
                        for index in memory_indices
                    ],
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
        batch.y_reward = rewards.to(self.device)[indices_t]
        batch.y_advantage = advantages.to(self.device)[indices_t]
        batch.y_memory_index = indices_t
        if len(memory.values) > max(int(index) for index in memory_indices):
            batch.y_value = torch.tensor(
                [float(memory.values[int(index)]) for index in memory_indices],
                dtype=torch.float32,
                device=self.device,
            )
        return fast_batch

    def _replay_v2_fast_exact_group(
        self,
        fast_batch: Any,
        *,
        actor_only: bool = False,
    ) -> list[dict[str, torch.Tensor]]:
        """Fast-Exact 同形组重放：编码器按原 group 前向，动作 head 按 B=1 逐样本。

        热路径全程使用布局元数据（CPU 整数 ptr）与 GPU 索引张量，
        禁止 ``.cpu().tolist()`` 与逐样本 ``.item()`` 同步。
        ``actor_only=True`` 时不计算 critic 与熵（首次同形合同预检）。
        """
        batch = fast_batch.batch
        group_size = fast_batch.group_size
        if self._v2_fast_exact_profile is not None:
            self._v2_fast_exact_profile_sync()
            encoder_started = time.perf_counter()
        with self.autocast_context():
            x_dict, global_context = self.policy(batch)
            state_values = None
            conditional_value_predictions = None
            if not actor_only:
                if (
                    str(getattr(self.config, "policy_action_scope", ""))
                    == "operation_station_worker"
                    and str(
                        getattr(self.config, "conditional_head_baseline_mode", "off")
                    )
                    == "factorized"
                ):
                    critic_x_dict, critic_context = self.policy.get_critic_context(
                        batch, actor_x_dict_encoded=x_dict
                    )
                    state_values = self.policy.critic(critic_context).reshape(-1)
                    batch_indices = torch.arange(group_size, device=self.device)
                    conditional_value_predictions = (
                        self.policy.get_conditional_values(
                            batch,
                            selected_task=batch.y_task,
                            selected_station=batch.y_station,
                            batch_indices=batch_indices,
                            actor_x_dict_encoded=x_dict,
                            critic_x_dict_encoded=critic_x_dict,
                            critic_context=critic_context,
                        )
                    )
                else:
                    state_values = self.policy.get_value(
                        batch, actor_x_dict_encoded=x_dict
                    ).reshape(-1)
        if self._v2_fast_exact_profile is not None:
            self._v2_fast_exact_profile_sync()
            self._v2_fast_exact_profile_add_time("EncoderMs", encoder_started)
            self._v2_fast_exact_profile["EncoderCalls"] = (
                self._v2_fast_exact_profile.get("EncoderCalls", 0.0) + 1.0
            )

        outputs: list[dict[str, torch.Tensor]] = []
        worker_layout = resolve_worker_feature_layout(self.config)
        num_skills = int(worker_layout.num_skill_types)
        task_ptr = fast_batch.task_ptr
        station_ptr = fast_batch.station_ptr
        worker_ptr = fast_batch.worker_ptr
        y_task = batch.y_task
        y_station = batch.y_station
        y_team = batch.y_team
        if self._v2_fast_exact_profile is not None:
            self._v2_fast_exact_profile_sync()
            action_head_started = time.perf_counter()

        for sample_index in range(group_size):
            task_start, task_end = task_ptr[sample_index], task_ptr[sample_index + 1]
            station_start, station_end = (
                station_ptr[sample_index],
                station_ptr[sample_index + 1],
            )
            worker_start, worker_end = (
                worker_ptr[sample_index],
                worker_ptr[sample_index + 1],
            )
            task_embs = x_dict["task"][task_start:task_end]
            station_embs = x_dict["station"][station_start:station_end]
            worker_embs = x_dict["worker"][worker_start:worker_end]
            raw_task = fast_batch.raw_task_slices[sample_index]
            raw_worker = fast_batch.raw_worker_slices[sample_index]
            raw_station = batch["station"].x[station_start:station_end]
            task_mask = batch.y_task_mask[task_start:task_end].bool()
            station_mask_matrix = batch.y_station_mask[task_start:task_end].bool()
            worker_queue_mask = batch.y_worker_mask[worker_start:worker_end].bool()
            global_i = global_context[sample_index].unsqueeze(0)

            task_target = y_task[sample_index].reshape(1)
            if self._v2_fast_exact_profile is not None:
                self._v2_fast_exact_profile["ActionHeadCalls"] = (
                    self._v2_fast_exact_profile.get("ActionHeadCalls", 0.0) + 1.0
                )
            with self.autocast_context():
                task_logits = self.policy.task_head(
                    task_embs, global_i, mask=None
                )
            # 非有限合法 logits 由统一分布入口拒绝。
            task_logits, task_usable = _finalize_action_logits(
                task_logits,
                task_mask.unsqueeze(0),
                decision=f"fast_exact.task[sample={sample_index}]",
            )
            if not bool(task_usable.item()):
                raise RuntimeError("fast-exact replay 的 task 动作空间全屏蔽")

            task_dist = Categorical(logits=task_logits)
            task_lp = task_dist.log_prob(task_target)
            task_entropy = (
                task_dist.entropy() if not actor_only else torch.zeros_like(task_lp)
            )
            normalized_task_entropy = self._normalized_categorical_entropy(
                task_entropy, task_mask.unsqueeze(0)
            )
            selected_task_emb = task_embs.index_select(
                0, y_task[sample_index].reshape(1)
            )  # [1, H]; 1-dim 索引规避 CUDA index_add_ 0-dim 快速路径 bug

            station_baseline_match = build_station_baseline_match(
                fast_batch.baseline_station_slices[sample_index],
                selected_tasks=y_task[sample_index].reshape(1),
                candidate_station_ids=torch.arange(
                    station_embs.size(0), device=self.device
                ),
                enabled=bool(
                    getattr(
                        self.config,
                        "reschedule_baseline_identity_conditioning",
                        False,
                    )
                ),
                device=self.device,
            )
            team_ids = y_team[sample_index]
            valid_team = team_ids[team_ids >= 0]
            is_virtual = int(valid_team.numel()) == 0
            zero = torch.zeros_like(task_lp)
            if is_virtual:
                conditional_values = None
                if conditional_value_predictions is not None:
                    conditional_values = torch.cat(
                        [
                            conditional_value_predictions["task"],
                            conditional_value_predictions["station"],
                            conditional_value_predictions["worker"],
                        ],
                        dim=1,
                    )[sample_index].reshape(1, 3)
                outputs.append(
                    {
                        "task": task_lp,
                        "station": zero,
                        "team": zero,
                        "entropy": task_entropy,
                        "normalized_entropy": normalized_task_entropy,
                        "state_value": (
                            state_values[sample_index].reshape(1)
                            if state_values is not None
                            else None
                        ),
                        "conditional_values": conditional_values,
                    }
                )
                continue

            station_target = y_station[sample_index].reshape(1)
            station_mask = station_mask_matrix.index_select(
                0, y_task[sample_index].reshape(1)
            )  # [1, S]; 与 legacy 路径一致保持 2-D，避免 masked_fill 把 logits 广播成 3-D
            with self.autocast_context():
                station_logits = self.policy.station_head(
                    selected_task_emb,
                    station_embs.unsqueeze(0),
                    mask=None,
                    station_baseline_match=station_baseline_match,
                )
            # 非有限合法 logits 由统一分布入口拒绝。
            station_logits, station_usable = _finalize_action_logits(
                station_logits,
                station_mask,
                decision=f"fast_exact.station[sample={sample_index}]",
            )
            if not bool(station_usable.item()):
                raise RuntimeError("fast-exact replay 的 station 动作空间全屏蔽")

            station_dist = Categorical(logits=station_logits)
            station_lp = station_dist.log_prob(station_target)
            station_entropy = (
                station_dist.entropy()
                if not actor_only
                else torch.zeros_like(station_lp)
            )
            normalized_station_entropy = self._normalized_categorical_entropy(
                station_entropy, station_mask
            )

            task_skill = torch.argmax(
                raw_task.index_select(0, y_task[sample_index].reshape(1))[:, 5 : 5 + num_skills]
            ).reshape(1)
            worker_skills = raw_worker[:, worker_layout.skill_slice]
            skill_invalid = ~(
                worker_skills.index_select(1, task_skill).squeeze(1) > 0.5
            )
            station_action_id = y_station[sample_index] + 1
            worker_locks = torch.argmax(
                raw_worker[:, worker_layout.lock_slice], dim=1
            )
            lock_invalid = (worker_locks != 0) & (worker_locks != station_action_id)
            current_mask = worker_queue_mask | skill_invalid | lock_invalid
            pressure = self._build_v2_pressure_context(
                task_features=raw_task,
                worker_features=raw_worker,
                task_present=None,
                task_action_invalid=task_mask,
                worker_present=None,
                worker_queue_invalid=worker_queue_mask,
                physical_predecessor_edges=self._physical_predecessor_edge_list(
                    batch, group_size
                )[sample_index : sample_index + 1],
            )
            team_state = self.policy.worker_head.initialize_v2_state(
                batch_size=1, device=self.device
            )
            worker_embs_i = worker_embs.unsqueeze(0)
            worker_baseline_member = build_worker_baseline_membership(
                fast_batch.baseline_team_edge_index,
                selected_tasks=(
                    y_task[sample_index].reshape(1)
                    + int(task_ptr[sample_index])
                ),
                num_workers=worker_embs.size(0),
                candidate_worker_offsets=torch.tensor(
                    [int(worker_ptr[sample_index])], device=self.device
                ),
                enabled=bool(
                    getattr(
                        self.config,
                        "reschedule_baseline_identity_conditioning",
                        False,
                    )
                ),
                device=self.device,
            )
            station_emb_i = station_embs.index_select(
                0, y_station[sample_index].reshape(1)
            )  # [1, H]; 1-dim 索引规避 CUDA index_add_ 0-dim 快速路径 bug
            demand = torch.tensor(
                [int(valid_team.numel())], dtype=torch.float32, device=self.device
            )
            decode_cache = self.policy.worker_head.build_v2_decode_cache(
                task_emb=selected_task_emb,
                station_emb=station_emb_i,
                global_context=global_i,
                worker_embs=worker_embs_i,
                pressure_context=pressure,
                demand=demand,
                candidate_skills=raw_worker[:, worker_layout.skill_slice].unsqueeze(0),
                task_required_skills=raw_task[
                    y_task[sample_index],
                    5 : 5 + worker_layout.num_skill_types,
                ].reshape(1, -1),
            )
            team_lp = torch.zeros_like(task_lp)
            team_entropy = torch.zeros_like(task_entropy)
            normalized_team_entropy = torch.zeros_like(normalized_task_entropy)
            if self._v2_fast_exact_profile is not None:
                self._v2_fast_exact_profile_sync()
                worker_pointer_started = time.perf_counter()
            for step in range(int(valid_team.numel())):
                worker_id_t = valid_team[step]
                self.worker_pointer_v2_diagnostics.record_team_state(
                    selected_max_wait=team_state.selected_max_wait,
                    selected_capacity_sum=team_state.selected_capacity_sum,
                )
                with self.autocast_context():
                    if self._v2_fast_exact_profile is not None:
                        self._v2_fast_exact_profile["ActionHeadCalls"] = (
                            self._v2_fast_exact_profile.get("ActionHeadCalls", 0.0)
                            + 1.0
                        )
                        self._v2_fast_exact_profile["WorkerPointerCalls"] = (
                            self._v2_fast_exact_profile.get(
                                "WorkerPointerCalls", 0.0
                            )
                            + 1.0
                        )
                    worker_logits = self.policy.worker_head.forward_choice_v2(
                        task_emb=selected_task_emb,
                        station_emb=station_emb_i,
                        global_context=global_i,
                        worker_embs=worker_embs_i,
                        pressure_context=pressure,
                        team_state=team_state,
                        demand=demand,
                        mask=None,
                        decode_cache=decode_cache,
                        dynamic_eft_features=self._build_v2_dynamic_eft_features(
                            team_state=team_state,
                            worker_features=raw_worker,
                            station_features=raw_station,
                            selected_station=station_target,
                            task_duration=raw_task.index_select(
                                0, y_task[sample_index].reshape(1)
                            )[:, 0],
                            demand=demand,
                            mask=current_mask.unsqueeze(0),
                        ),
                        worker_baseline_member=worker_baseline_member,
                    )
                # 非有限合法 logits 由统一分布入口拒绝。
                self._record_v2_marginal_scarcity(
                    self.policy.worker_head,
                    worker_invalid_mask=current_mask.unsqueeze(0),
                    selected_worker_index=worker_id_t,
                )
                worker_logits, worker_usable = _finalize_action_logits(
                    worker_logits,
                    current_mask.unsqueeze(0),
                    decision=(
                        f"fast_exact.worker[sample={sample_index},step={step}]"
                    ),
                )
                if not bool(worker_usable.item()):
                    raise RuntimeError(
                        "fast-exact replay 的 worker 动作空间全屏蔽"
                    )

                worker_dist = Categorical(logits=worker_logits)
                team_lp = team_lp + worker_dist.log_prob(worker_id_t.reshape(1))
                step_entropy = (
                    worker_dist.entropy()
                    if not actor_only
                    else torch.zeros_like(team_lp)
                )
                team_entropy = team_entropy + step_entropy
                normalized_team_entropy = normalized_team_entropy + (
                    self._normalized_categorical_entropy(
                        step_entropy, current_mask.unsqueeze(0)
                    )
                )
                # 0-dim 标量索引 worker_embs_i[:, worker_id_t, :] 的 backward 会
                # 触发 CUDA index_add_ 的 0-dim 快速路径（PyTorch 2.12 + CUDA>=12.8），
                # 真实训练中表现为 backward 永久挂起；改用 gather 显式提取单个
                # worker 嵌入（backward 走常规 scatter_add_），forward 经 reshape
                # 保持与历史路径一致的 [1,H]。
                worker_emb_i = worker_embs_i.gather(
                    1,
                    worker_id_t.reshape(1, 1, 1).expand(
                        1, 1, worker_embs_i.size(-1)
                    ),
                ).reshape(1, -1)
                team_state = self.policy.worker_head.advance_v2_state(
                    team_state,
                    worker_emb_i,
                    # 1-dim 索引规避 CUDA index_add_ 对 0-dim 索引的快速路径 bug。
                    worker_skills.index_select(0, worker_id_t.reshape(1)),
                    selected_wait=torch.expm1(
                        raw_worker.index_select(0, worker_id_t.reshape(1))[
                            :, worker_layout.wait_idx
                        ].float().clamp_min(0.0)
                    ).reshape(1),
                    selected_capacity=(
                        raw_worker.index_select(0, worker_id_t.reshape(1))[
                            :, worker_layout.efficiency_idx
                        ].float()
                        * raw_worker.index_select(0, worker_id_t.reshape(1))[
                            :, worker_layout.fatigue_idx
                        ].float()
                    ).reshape(1),
                )
                current_mask = current_mask.clone()
                current_mask[worker_id_t] = True

            if self._v2_fast_exact_profile is not None:
                self._v2_fast_exact_profile_sync()
                self._v2_fast_exact_profile_add_time(
                    "WorkerPointerMs", worker_pointer_started
                )

            normalized_entropy = (
                normalized_task_entropy
                + 1.5 * normalized_station_entropy
                + 0.5 * normalized_team_entropy
            )
            outputs.append(
                {
                    "task": task_lp,
                    "station": station_lp,
                    "team": team_lp,
                    "entropy": task_entropy + station_entropy + team_entropy,
                    "normalized_entropy": normalized_entropy,
                    "station_emb": station_emb_i,
                    "state_value": (
                        state_values[sample_index].reshape(1)
                        if state_values is not None
                        else None
                    ),
                    "conditional_values": (
                        torch.cat(
                            [
                                conditional_value_predictions["task"],
                                conditional_value_predictions["station"],
                                conditional_value_predictions["worker"],
                            ],
                            dim=1,
                        )[sample_index].reshape(1, 3)
                        if conditional_value_predictions is not None
                        else None
                    ),
                }
            )
        assert len(outputs) == group_size
        if self._v2_fast_exact_profile is not None:
            self._v2_fast_exact_profile_sync()
            self._v2_fast_exact_profile_add_time(
                "ActionHeadMs", action_head_started
            )
        return outputs
