from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import global_mean_pool

from configs import configs
from models.hb_gat_pn import FeatureEmbedder, HeteroGATEncoder, get_activation, get_head_layer_norm


GRAPH_BASELINE_FEATURE_MODE = "graph_hetero"


def task_demand_from_obs(obs: HeteroData, task_idx: int) -> int:
    """从主方法同一图特征中读取工序人数需求。"""
    task_x = obs["task"].x
    if task_x.size(1) > 16:
        demand = int(task_x[int(task_idx), 16].item())
    elif task_x.size(1) > 2:
        demand = int(task_x[int(task_idx), 2].item())
    else:
        demand = 1
    return max(1, demand)


def worker_static_mask_from_obs(
    obs: HeteroData,
    *,
    task_idx: int,
    station_idx: int,
    worker_mask: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    """计算与主方法一致的 worker 技能和工位锁定静态约束。True 表示不可选。"""
    worker_x = obs["worker"].x.to(device)
    task_x = obs["task"].x.to(device)
    if worker_mask is None:
        mask = torch.zeros(worker_x.size(0), dtype=torch.bool, device=device)
    else:
        mask = worker_mask.to(device=device, dtype=torch.bool).clone()

    task_type_idx = int(torch.argmax(task_x[int(task_idx), 5:15]).item())
    has_skill = worker_x[:, 1:11][:, task_type_idx] > 0.5
    worker_locks = torch.argmax(worker_x[:, 13:21], dim=1)
    station_action = int(station_idx) + 1
    valid_lock = (worker_locks == 0) | (worker_locks == station_action)
    return mask | (~has_skill) | (~valid_lock)


def _sample_or_argmax(scores: torch.Tensor, *, deterministic: bool, temperature: float) -> tuple[int, torch.Tensor]:
    if deterministic or temperature <= 0.0:
        action = torch.argmax(scores, dim=-1)
        return int(action.item()), torch.tensor(0.0, device=scores.device)
    dist = Categorical(logits=(scores / max(float(temperature), 1e-6)).float())
    action = dist.sample()
    return int(action.item()), dist.log_prob(action)


class SimpleTaskHead(nn.Module):
    """轻量 task 打分头：同图特征，不使用 HB-GAT-PN 指针头。"""

    def __init__(self, hidden_dim: int, context_dim: int) -> None:
        super().__init__()
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.task_proj = nn.Linear(hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            get_head_layer_norm(hidden_dim),
            get_activation(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, task_emb: torch.Tensor, global_context: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if task_emb.dim() == 2:
            task_emb = task_emb.unsqueeze(0)
        ctx = self.context_proj(global_context).unsqueeze(1).expand(-1, task_emb.size(1), -1)
        task = self.task_proj(task_emb)
        scores = self.mlp(torch.cat([ctx, task], dim=-1)).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask.bool(), -1e4)
        return scores


class SimpleStationHead(nn.Module):
    """轻量 station 条件打分头。"""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.task_proj = nn.Linear(hidden_dim, hidden_dim)
        self.station_proj = nn.Linear(hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            get_head_layer_norm(hidden_dim),
            get_activation(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, selected_task_emb: torch.Tensor, station_embs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        task = self.task_proj(selected_task_emb).unsqueeze(1).expand(-1, station_embs.size(1), -1)
        station = self.station_proj(station_embs)
        scores = self.mlp(torch.cat([task, station], dim=-1)).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask.bool(), -1e4)
        return scores


class SimpleWorkerHead(nn.Module):
    """轻量 worker 自回归打分头。"""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ar_query_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.worker_proj = nn.Linear(hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            get_head_layer_norm(hidden_dim),
            get_activation(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_choice(
        self,
        task_emb: torch.Tensor,
        worker_embs: torch.Tensor,
        mask: torch.Tensor | None = None,
        current_team_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if current_team_emb is None:
            query = self.query_proj(task_emb)
        else:
            query = self.ar_query_proj(torch.cat([task_emb, current_team_emb], dim=-1))
        query = query.unsqueeze(1).expand(-1, worker_embs.size(1), -1)
        worker = self.worker_proj(worker_embs)
        scores = self.mlp(torch.cat([query, worker], dim=-1)).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask.bool(), -1e4)
        return scores


class GraphBaselineActorCritic(nn.Module):
    """BasicPPO/DQN 共用的可泛化异构图基线网络。"""

    def __init__(self, config: Any = configs) -> None:
        super().__init__()
        self.config = config
        hidden_dim = int(config.hidden_dim)
        context_dim = hidden_dim * 3
        self.embedder = FeatureEmbedder(config)
        self.encoder = HeteroGATEncoder(config)
        self.task_head = SimpleTaskHead(hidden_dim, context_dim)
        self.station_head = SimpleStationHead(hidden_dim)
        self.worker_head = SimpleWorkerHead(hidden_dim)
        self.critic = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            get_head_layer_norm(hidden_dim),
            get_activation(),
            nn.Linear(hidden_dim, 1),
        )

    def _pool(self, x_dict: dict[str, torch.Tensor], batch_data: HeteroData) -> torch.Tensor:
        contexts = []
        for node_type in ("station", "task", "worker"):
            x = x_dict[node_type]
            batch_vec = getattr(batch_data[node_type], "batch", None)
            if batch_vec is None:
                contexts.append(x.mean(dim=0, keepdim=True))
            else:
                contexts.append(global_mean_pool(x, batch_vec.to(x.device)))
        return torch.cat(contexts, dim=-1)

    def forward(self, batch_data: HeteroData) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        x_dict = self.embedder(batch_data.x_dict)
        if getattr(self.config, "ablation_no_gat", False):
            encoded = x_dict
        else:
            encoded = self.encoder(x_dict, batch_data.edge_index_dict)
        return encoded, self._pool(encoded, batch_data)

    def get_value(self, batch_data: HeteroData, actor_x_dict_encoded: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        if actor_x_dict_encoded is None:
            actor_x_dict_encoded, _ = self.forward(batch_data)
        context = self._pool(actor_x_dict_encoded, batch_data)
        return self.critic(context)


@dataclass
class GraphActionResult:
    action: tuple[int, int, list[int]] | None
    logprob: torch.Tensor
    value: torch.Tensor | None


def select_graph_action(
    model: GraphBaselineActorCritic,
    obs: HeteroData,
    *,
    masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    deterministic: bool,
    temperature: float = 0.0,
    need_value: bool = False,
) -> GraphActionResult:
    """供图版 PPO/DQN 验证复用的单环境动作选择。"""
    batch_obs = Batch.from_data_list([obs]).to(device)
    task_mask, station_mask_matrix, worker_mask = masks
    task_mask = task_mask.to(device=device, dtype=torch.bool)
    station_mask_matrix = station_mask_matrix.to(device=device, dtype=torch.bool)
    worker_mask = worker_mask.to(device=device, dtype=torch.bool)
    if bool(task_mask.all()):
        return GraphActionResult(None, torch.tensor(0.0, device=device), None)

    with torch.no_grad():
        x_dict, context = model(batch_obs)
        task_scores = model.task_head(x_dict["task"], context, mask=task_mask.unsqueeze(0))
        task_idx, task_lp = _sample_or_argmax(task_scores.squeeze(0), deterministic=deterministic, temperature=temperature)

        station_mask = station_mask_matrix[task_idx].unsqueeze(0)
        selected_task_emb = x_dict["task"][task_idx].unsqueeze(0)
        station_scores = model.station_head(selected_task_emb, x_dict["station"].unsqueeze(0), mask=station_mask)
        station_idx, station_lp = _sample_or_argmax(station_scores.squeeze(0), deterministic=deterministic, temperature=temperature)

        demand = task_demand_from_obs(obs, task_idx)
        current_worker_mask = worker_static_mask_from_obs(
            obs,
            task_idx=task_idx,
            station_idx=station_idx,
            worker_mask=worker_mask,
            device=device,
        ).unsqueeze(0)
        worker_embs = x_dict["worker"].unsqueeze(0)
        team: list[int] = []
        worker_lps: list[torch.Tensor] = []
        current_team_emb = None
        for _ in range(demand):
            if bool(current_worker_mask.all()):
                return GraphActionResult(None, torch.tensor(0.0, device=device), None)
            scores = model.worker_head.forward_choice(
                selected_task_emb,
                worker_embs,
                mask=current_worker_mask,
                current_team_emb=current_team_emb,
            )
            worker_idx, worker_lp = _sample_or_argmax(scores.squeeze(0), deterministic=deterministic, temperature=temperature)
            team.append(worker_idx)
            worker_lps.append(worker_lp)
            selected = worker_embs[0, team, :]
            current_team_emb = selected.mean(dim=0, keepdim=True)
            current_worker_mask = current_worker_mask.clone()
            current_worker_mask[0, worker_idx] = True

        value = model.get_value(batch_obs, actor_x_dict_encoded=x_dict).view(-1)[0] if need_value else None
        logprob = task_lp + station_lp + (sum(worker_lps) if worker_lps else torch.tensor(0.0, device=device))
        return GraphActionResult((task_idx, station_idx, team), logprob, value)
