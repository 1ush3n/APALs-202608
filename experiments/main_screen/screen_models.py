from __future__ import annotations

"""主方法快速筛查专用模型。

本模块不被正式训练入口导入。其目的仅是让候选结构能够在不改动
``models.hb_gat_pn.HBGATPN`` 的前提下，完成可追溯的短训练筛查。
"""

from typing import Any, Mapping

import torch
from torch import nn
from torch_geometric.nn import global_max_pool, global_mean_pool

from models.hb_gat_pn import HBGATPN, get_activation, get_head_layer_norm


class ScaleGatedContextHBGATPN(HBGATPN):
    """仅在 actor 全局读出端试验尺度条件的注意力/统计池化门控。

    输入和输出均保持 ``[B, 3H]``，因此工序、站位和工人团队的联合
    pointer 解码器、可行掩码、Skill Hub 与 critic 均保持原有实现。
    """

    def __init__(self, config: Any) -> None:
        # TaskPointer 的上下文维度必须维持 3H，故以原 attention 模式构造骨干。
        original_mode = str(getattr(config, "actor_context_mode", "attention"))
        config.actor_context_mode = "attention"
        try:
            super().__init__(config)
        finally:
            config.actor_context_mode = original_mode

        hidden_dim = int(config.hidden_dim)
        self.screen_pool_projections = nn.ModuleList(
            [nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(3)]
        )
        self.screen_scale_gate = nn.Sequential(
            nn.Linear(3, 16),
            nn.SiLU(),
            nn.Linear(16, 3),
        )
        # 初始时几乎完全复现原 attention 策略；训练再决定是否借助统计池化。
        final_layer = self.screen_scale_gate[-1]
        assert isinstance(final_layer, nn.Linear)
        nn.init.zeros_(final_layer.weight)
        nn.init.constant_(final_layer.bias, 4.0)
        self.last_screen_gate: torch.Tensor | None = None

    @staticmethod
    def _batch_counts(node_store: Any, *, batch_size: int, device: torch.device) -> torch.Tensor:
        batch = getattr(node_store, "batch", None)
        if batch is None:
            return torch.full((1,), int(node_store.x.size(0)), device=device, dtype=torch.float32)
        return torch.bincount(batch, minlength=batch_size).to(device=device, dtype=torch.float32)

    def _screen_mean_max_context(self, x_dict_encoded: dict[str, torch.Tensor], batch_data: Any) -> tuple[torch.Tensor, torch.Tensor]:
        station = x_dict_encoded["station"]
        task = x_dict_encoded["task"]
        worker = x_dict_encoded["worker"]
        station_batch = getattr(batch_data["station"], "batch", None)
        if station_batch is None:
            pooled = [
                torch.cat([station.mean(dim=0, keepdim=True), station.max(dim=0, keepdim=True).values], dim=1),
                torch.cat([task.mean(dim=0, keepdim=True), task.max(dim=0, keepdim=True).values], dim=1),
                torch.cat([worker.mean(dim=0, keepdim=True), worker.max(dim=0, keepdim=True).values], dim=1),
            ]
            counts = torch.tensor(
                [[station.size(0), task.size(0), worker.size(0)]],
                device=station.device,
                dtype=torch.float32,
            )
        else:
            batch_size = int(station_batch.max().item()) + 1
            pooled = [
                torch.cat([global_mean_pool(station, station_batch), global_max_pool(station, station_batch)], dim=1),
                torch.cat([global_mean_pool(task, batch_data["task"].batch), global_max_pool(task, batch_data["task"].batch)], dim=1),
                torch.cat([global_mean_pool(worker, batch_data["worker"].batch), global_max_pool(worker, batch_data["worker"].batch)], dim=1),
            ]
            counts = torch.stack(
                [
                    self._batch_counts(batch_data["station"], batch_size=batch_size, device=station.device),
                    self._batch_counts(batch_data["task"], batch_size=batch_size, device=station.device),
                    self._batch_counts(batch_data["worker"], batch_size=batch_size, device=station.device),
                ],
                dim=1,
            )
        projected = [layer(features) for layer, features in zip(self.screen_pool_projections, pooled, strict=True)]
        return torch.cat(projected, dim=1), torch.log1p(counts)

    def _compute_global_context(
        self,
        x_dict_encoded: dict[str, torch.Tensor],
        batch_data: Any,
        *,
        mode: str,
        station_attn: nn.Module,
        task_worker_attn: nn.Module,
    ) -> torch.Tensor:
        attention_context = super()._compute_global_context(
            x_dict_encoded,
            batch_data,
            mode=mode,
            station_attn=station_attn,
            task_worker_attn=task_worker_attn,
        )
        # 只改变 actor；critic 保持原 attention 读出，以便隔离策略端假设。
        if station_attn is not self.actor_station_attn:
            return attention_context
        mean_max_context, log_counts = self._screen_mean_max_context(x_dict_encoded, batch_data)
        gates = torch.sigmoid(self.screen_scale_gate(log_counts))  # [B, 3]
        hidden_dim = int(self.config.hidden_dim)
        expanded_gates = gates.repeat_interleave(hidden_dim, dim=1)  # [B, 3H]
        self.last_screen_gate = gates.detach()
        return expanded_gates * attention_context + (1.0 - expanded_gates) * mean_max_context


class DualAttentionContextHBGATPN(HBGATPN):
    """筛查用双头 attention 上下文模型。

    Actor 返回按 ``[station, task, worker]`` 拼接的两个 ``3H`` 上下文；
    Critic 继续复用基类的单头 ``3H`` attention 上下文。
    """

    _INIT_SEED_OFFSET = 2027

    def __init__(self, config: Any) -> None:
        original_mode = str(getattr(config, "actor_context_mode", "attention"))
        config.actor_context_mode = "attention"
        try:
            super().__init__(config)
        finally:
            config.actor_context_mode = original_mode

        hidden_dim = int(config.hidden_dim)
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.manual_seed(int(getattr(config, "seed", 42)) + self._INIT_SEED_OFFSET)
            self.dual_attention_station_attn = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                get_head_layer_norm(32),
                get_activation(),
                nn.Linear(32, 1),
            )
            self.dual_attention_task_worker_attn = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                get_head_layer_norm(32),
                get_activation(),
                nn.Linear(32, 1),
            )
        self.task_head.context_proj = nn.Linear(6 * hidden_dim, hidden_dim)

    def _compute_global_context(
        self,
        x_dict_encoded: dict[str, torch.Tensor],
        batch_data: Any,
        *,
        mode: str,
        station_attn: nn.Module,
        task_worker_attn: nn.Module,
    ) -> torch.Tensor:
        """计算 Actor 双头或 Critic 单头全局上下文。"""

        first_context = super()._compute_global_context(
            x_dict_encoded,
            batch_data,
            mode=mode,
            station_attn=station_attn,
            task_worker_attn=task_worker_attn,
        )
        if station_attn is not self.actor_station_attn:
            assert first_context.ndim == 2
            assert first_context.size(1) == 3 * int(self.config.hidden_dim)
            return first_context

        first_weights = self.last_s_weights
        first_variance = self.last_s_var
        second_context = super()._compute_global_context(
            x_dict_encoded,
            batch_data,
            mode=mode,
            station_attn=self.dual_attention_station_attn,
            task_worker_attn=self.dual_attention_task_worker_attn,
        )
        self.last_s_weights = first_weights
        self.last_s_var = first_variance

        hidden_dim = int(self.config.hidden_dim)
        assert first_context.shape == (second_context.size(0), 3 * hidden_dim)
        assert second_context.shape == first_context.shape
        # [B,3H] + [B,3H] -> [B,6H]，顺序固定为第一头再第二头。
        return torch.cat([first_context, second_context], dim=1)


def expand_dual_attention_task_projection(
    state_dict: Mapping[str, torch.Tensor],
    *,
    hidden_dim: int,
) -> dict[str, torch.Tensor]:
    """将旧单头 Task 投影扩展为双头投影，保持旧 logits 不变。"""

    weight_key = "task_head.context_proj.weight"
    bias_key = "task_head.context_proj.bias"
    old_weight = state_dict.get(weight_key)
    old_bias = state_dict.get(bias_key)
    expected_old_shape = (hidden_dim, 3 * hidden_dim)
    assert old_weight is not None and old_weight.shape == expected_old_shape
    assert old_bias is not None and old_bias.shape == (hidden_dim,)
    expanded_weight = old_weight.new_zeros((hidden_dim, 6 * hidden_dim))
    expanded_weight[:, : 3 * hidden_dim] = old_weight
    return {
        **state_dict,
        weight_key: expanded_weight,
        bias_key: old_bias,
    }
