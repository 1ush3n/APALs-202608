"""WorkerPointer v2 的纯函数式五技能压力上下文。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


NUM_SKILL_TYPES = 5
TASK_SKILL_SLICE = slice(5, 10)
TASK_DEMAND_INDEX = 16
WORKER_SKILL_SLICE = slice(1, 6)
WORKER_LOG_WAIT_INDEX = 6
PHYSICAL_PREDECESSOR_EDGE = ("task", "physical_precedes_aux", "task")


@dataclass(frozen=True)
class WorkerPressureContext:
    """批量 WorkerPointer 压力特征；所有浮点张量固定为 float32。"""

    demand_all: torch.Tensor
    demand_near: torch.Tensor
    supply_all: torch.Tensor
    supply_near: torch.Tensor
    pressure_all: torch.Tensor
    pressure_near: torch.Tensor
    zero_supply_all: torch.Tensor
    zero_supply_near: torch.Tensor
    candidate_exposure: torch.Tensor
    candidate_max_exposure: torch.Tensor
    pressure_next_frontier: torch.Tensor | None = None
    next_frontier_demand: torch.Tensor | None = None
    next_frontier_mask: torch.Tensor | None = None
    unfinished_physical_predecessor_count: torch.Tensor | None = None
    remaining_physical_predecessor_count: torch.Tensor | None = None


@dataclass(frozen=True)
class WorkerPointerV2State:
    """自回归团队的增量 DeepSets 状态。"""

    mapped_sum: torch.Tensor
    mapped_max: torch.Tensor
    count: torch.Tensor
    selected_skill_sum: torch.Tensor
    selected_max_wait: torch.Tensor
    selected_capacity_sum: torch.Tensor


@dataclass(frozen=True)
class WorkerPointerV2DecodeCache:
    """单个工序—工位团队解码中不随成员选择变化的 float32 张量。"""

    candidate_keys: torch.Tensor
    query_prefix: torch.Tensor
    pressure_features: torch.Tensor
    supply_all: torch.Tensor
    demand: torch.Tensor
    candidate_skills: torch.Tensor | None = None
    task_required_skills: torch.Tensor | None = None


def gather_selected_task_skills(
    task_features: torch.Tensor,
    selected_task: torch.Tensor,
) -> torch.Tensor:
    """从原始任务特征中提取当前物理任务的 5 维需求技能 one-hot。"""

    assert task_features.ndim == 3
    batch_size, num_tasks, task_dim = task_features.shape
    assert task_dim >= TASK_SKILL_SLICE.stop
    assert selected_task.ndim == 1 and selected_task.shape == (batch_size,)
    assert torch.all((selected_task >= 0) & (selected_task < num_tasks))
    batch_index = torch.arange(batch_size, device=task_features.device)
    selected_skills = task_features.float()[
        batch_index, selected_task.to(device=task_features.device), TASK_SKILL_SLICE
    ]
    assert selected_skills.shape == (batch_size, NUM_SKILL_TYPES)
    assert torch.isfinite(selected_skills).all()
    assert ((selected_skills > 0.5).sum(dim=-1) == 1).all()
    return selected_skills


def build_v2_marginal_reserve_scarcity(
    *,
    demand_all: torch.Tensor,
    supply_all: torch.Tensor,
    selected_skill_sum: torch.Tensor,
    candidate_skills: torch.Tensor,
    task_required_skills: torch.Tensor,
    epsilon: float,
    clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算 ALL horizon 下候选工人的总/可避免技能储备稀缺边际。"""

    assert demand_all.ndim == supply_all.ndim == selected_skill_sum.ndim == 2
    batch_size = demand_all.size(0)
    assert demand_all.shape == supply_all.shape == selected_skill_sum.shape
    assert demand_all.shape == (batch_size, NUM_SKILL_TYPES)
    assert candidate_skills.ndim == 3
    assert candidate_skills.shape[0] == batch_size
    assert candidate_skills.size(-1) == NUM_SKILL_TYPES
    assert task_required_skills.shape == (batch_size, NUM_SKILL_TYPES)
    if not torch.isfinite(torch.tensor(float(epsilon))) or float(epsilon) <= 0.0:
        raise ValueError("epsilon 必须是大于 0 的有限数")
    if not torch.isfinite(torch.tensor(float(clip))) or float(clip) <= 0.0:
        raise ValueError("clip 必须是大于 0 的有限数")

    with torch.autocast(device_type=demand_all.device.type, enabled=False):
        demand = demand_all.float().clamp_min(0.0)
        supply = supply_all.float().clamp_min(0.0)
        selected = selected_skill_sum.float().clamp_min(0.0)
        candidates = candidate_skills.float().clamp_min(0.0)
        required = task_required_skills.float().clamp(0.0, 1.0)
        remaining_before = (supply - selected).clamp_min(float(epsilon))
        remaining_after = (
            remaining_before.unsqueeze(1) - candidates
        ).clamp_min(float(epsilon))
        pressure_before = torch.log1p(demand / remaining_before)
        pressure_after = torch.log1p(
            demand.unsqueeze(1) / remaining_after
        )
        delta_by_skill = (
            pressure_after - pressure_before.unsqueeze(1)
        ).clamp_min(0.0)
        marginal_total = (delta_by_skill * candidates).sum(
            dim=-1, keepdim=True
        )
        extra_skill_mask = candidates * (1.0 - required.unsqueeze(1))
        marginal_extra = (delta_by_skill * extra_skill_mask).sum(
            dim=-1, keepdim=True
        )
        marginal_total = marginal_total.clamp(0.0, float(clip))
        marginal_extra = marginal_extra.clamp(0.0, float(clip))

    assert marginal_total.shape == marginal_extra.shape == (
        batch_size,
        candidate_skills.size(1),
        1,
    )
    assert torch.isfinite(marginal_total).all()
    assert torch.isfinite(marginal_extra).all()
    assert (marginal_extra <= marginal_total + 1.0e-6).all()
    return marginal_total, marginal_extra


def build_worker_eft_features(
    *,
    team_state: WorkerPointerV2State,
    worker_wait: torch.Tensor,
    worker_capacity: torch.Tensor,
    station_wait: torch.Tensor,
    task_duration: torch.Tensor,
    demand: torch.Tensor,
    mask: torch.Tensor,
    clip: float,
) -> torch.Tensor:
    """按当前部分团队计算每名工人的动态 EFT 特征。"""

    if not torch.isfinite(torch.tensor(float(clip))) or float(clip) <= 0.0:
        raise ValueError("worker_pointer_v2_dynamic_eft_feature_clip 必须是大于 0 的有限数")
    assert worker_wait.ndim == worker_capacity.ndim == mask.ndim == 2
    batch_size, num_workers = worker_wait.shape
    assert worker_capacity.shape == mask.shape == (batch_size, num_workers)
    assert station_wait.reshape(-1).shape == task_duration.reshape(-1).shape == demand.reshape(-1).shape == (batch_size,)
    assert team_state.count.shape == team_state.selected_max_wait.shape == team_state.selected_capacity_sum.shape == (batch_size, 1)

    with torch.autocast(device_type=worker_wait.device.type, enabled=False):
        legal = ~mask.bool()
        duration = task_duration.float().reshape(batch_size, 1).clamp_min(1.0e-6)
        task_demand = demand.float().reshape(batch_size, 1).clamp_min(1.0)
        station = station_wait.float().reshape(batch_size, 1).clamp_min(0.0)
        candidate_ready = torch.maximum(
            torch.maximum(team_state.selected_max_wait.float(), worker_wait.float().clamp_min(0.0)),
            station,
        )
        candidate_capacity = (
            team_state.selected_capacity_sum.float()
            + worker_capacity.float().clamp_min(1.0e-6)
        )
        synergy = torch.pow(
            torch.tensor(0.95, device=worker_wait.device, dtype=torch.float32),
            team_state.count.float(),
        )
        finish = candidate_ready + duration * task_demand / (candidate_capacity * synergy)
        legal_finish = finish.masked_fill(~legal, float("inf"))
        minimum = legal_finish.amin(dim=1, keepdim=True)
        legal_count = legal.sum(dim=1, keepdim=True)
        safe_minimum = torch.where(legal_count > 0, minimum, torch.zeros_like(minimum))
        legal_sum = torch.where(legal, finish, torch.zeros_like(finish)).sum(dim=1, keepdim=True)
        mean = legal_sum / legal_count.clamp_min(1).to(dtype=torch.float32)
        variance = torch.where(legal, (finish - mean).square(), torch.zeros_like(finish)).sum(dim=1, keepdim=True)
        std = torch.sqrt(variance / legal_count.clamp_min(1).to(dtype=torch.float32))
        relative = ((finish - safe_minimum) / duration).clamp(0.0, float(clip))
        zscore = ((finish - mean) / std.clamp_min(1.0e-6)).clamp(-float(clip), float(clip))
        features = torch.stack([relative, zscore], dim=-1)
        features = torch.where(legal.unsqueeze(-1), features, torch.zeros_like(features))
    assert features.shape == (batch_size, num_workers, 2)
    assert torch.isfinite(features).all()
    return features


def build_worker_pressure_context(
    *,
    task_features: torch.Tensor,
    worker_features: torch.Tensor,
    task_present: torch.Tensor,
    task_action_invalid: torch.Tensor,
    worker_present: torch.Tensor,
    worker_queue_invalid: torch.Tensor,
    temperature: float,
    supply_epsilon: float,
    physical_predecessor_edges: Sequence[torch.Tensor] | torch.Tensor | None = None,
) -> WorkerPressureContext:
    """从原始节点特征构造归一化工作量/人数口径的双尺度压力。"""

    assert task_features.ndim == 3, task_features.shape
    assert worker_features.ndim == 3, worker_features.shape
    batch_size, num_tasks, task_dim = task_features.shape
    worker_batch, num_workers, worker_dim = worker_features.shape
    assert batch_size == worker_batch
    assert task_dim >= 18 and worker_dim >= 17
    assert task_present.shape == task_action_invalid.shape == (batch_size, num_tasks)
    assert worker_present.shape == worker_queue_invalid.shape == (batch_size, num_workers)
    if not torch.isfinite(torch.tensor(float(temperature))) or float(temperature) <= 0.0:
        raise ValueError("worker_pointer_pressure_temperature 必须是大于 0 的有限数")
    if not torch.isfinite(torch.tensor(float(supply_epsilon))) or float(supply_epsilon) <= 0.0:
        raise ValueError("worker_pointer_supply_epsilon 必须是大于 0 的有限数")

    # 输入形状：[B,T,F]/[B,N,F]；压力归约始终使用 float32。
    task_x = task_features.float()
    worker_x = worker_features.float()
    task_skills = task_x[..., TASK_SKILL_SLICE].clamp(0.0, 1.0)  # [B,T,5]
    worker_skills = worker_x[..., WORKER_SKILL_SLICE].clamp(0.0, 1.0)  # [B,N,5]
    duration = task_x[..., 0].clamp_min(0.0)
    demand = task_x[..., TASK_DEMAND_INDEX].clamp_min(0.0)
    workload = duration * demand  # [B,T]

    not_ready = task_x[..., 1] > 0.5
    ready = task_x[..., 2] > 0.5
    unscheduled = task_present.bool() & (not_ready | ready)
    actionable = task_present.bool() & ready & (~task_action_invalid.bool())
    physical = task_skills.sum(dim=-1) > 0.0
    demand_all = torch.sum(
        task_skills * (workload * (unscheduled & physical).float()).unsqueeze(-1), dim=1
    )
    demand_near = torch.sum(
        task_skills * (workload * (actionable & physical).float()).unsqueeze(-1), dim=1
    )

    present = worker_present.bool()
    supply_eligible = present & (~worker_queue_invalid.bool())
    supply_all = torch.sum(worker_skills * present.unsqueeze(-1).float(), dim=1)
    normalized_wait = torch.expm1(worker_x[..., WORKER_LOG_WAIT_INDEX].clamp_min(0.0))
    wait_discount = torch.exp(-normalized_wait / float(temperature))
    supply_near = torch.sum(
        worker_skills
        * (supply_eligible.float() * wait_discount).unsqueeze(-1),
        dim=1,
    )

    epsilon = float(supply_epsilon)
    zero_supply_all = supply_all <= epsilon
    zero_supply_near = supply_near <= epsilon
    pressure_all = torch.log1p(demand_all / supply_all.clamp_min(epsilon))
    pressure_near = torch.log1p(demand_near / supply_near.clamp_min(epsilon))

    valid_worker_skills = worker_skills * present.unsqueeze(-1).float()
    exposure_all = valid_worker_skills * pressure_all.unsqueeze(1)  # [B,N,5]
    exposure_near = valid_worker_skills * pressure_near.unsqueeze(1)  # [B,N,5]
    candidate_exposure = torch.cat([exposure_all, exposure_near], dim=-1)  # [B,N,10]
    candidate_max_exposure = torch.stack(
        [exposure_all.amax(dim=-1), exposure_near.amax(dim=-1)], dim=-1
    )  # [B,N,2]
    assert candidate_exposure.shape == (batch_size, num_workers, 10)
    assert candidate_max_exposure.shape == (batch_size, num_workers, 2)
    assert torch.isfinite(pressure_all).all() and torch.isfinite(pressure_near).all()
    assert torch.isfinite(candidate_exposure).all()
    pressure_next_frontier = None
    next_frontier_demand = None
    next_frontier_mask = None
    unfinished_pred_count = None
    remaining_pred_count = None
    if physical_predecessor_edges is not None:
        if isinstance(physical_predecessor_edges, torch.Tensor):
            edge_list = [physical_predecessor_edges]
        else:
            edge_list = list(physical_predecessor_edges)
        if len(edge_list) != batch_size:
            raise ValueError(
                "physical_predecessor_edges 必须按 batch 提供局部稀疏边"
            )
        pressure_rows: list[torch.Tensor] = []
        demand_rows: list[torch.Tensor] = []
        frontier_rows: list[torch.Tensor] = []
        before_rows: list[torch.Tensor] = []
        after_rows: list[torch.Tensor] = []
        for batch_index, edge_index in enumerate(edge_list):
            edge_index = edge_index.to(device=task_x.device, dtype=torch.long)
            assert edge_index.ndim == 2 and edge_index.size(0) == 2
            source = edge_index[0]
            target = edge_index[1]
            assert torch.all((source >= 0) & (source < num_tasks))
            assert torch.all((target >= 0) & (target < num_tasks))
            physical = task_skills[batch_index].sum(dim=-1) > 0.0
            present_tasks = task_present[batch_index].bool()
            not_ready_tasks = task_x[batch_index, :, 1] > 0.5
            ready_tasks = task_x[batch_index, :, 2] > 0.5
            done_tasks = task_x[batch_index, :, 3] > 0.5
            now_tasks = (
                ready_tasks
                & present_tasks
                & (~task_action_invalid[batch_index].bool())
            )
            before = torch.zeros(num_tasks, device=task_x.device, dtype=torch.float32)
            after = torch.zeros_like(before)
            if source.numel() > 0:
                valid_edge = (
                    physical[source]
                    & physical[target]
                    & present_tasks[source]
                    & present_tasks[target]
                )
                unfinished = valid_edge & (~done_tasks[source])
                before.scatter_add_(0, target, unfinished.float())
                after.scatter_add_(
                    0,
                    target,
                    (unfinished & (~now_tasks[source])).float(),
                )
            frontier = (
                not_ready_tasks
                & physical
                & present_tasks
                & (before > 0.0)
                & (after == 0.0)
            )
            frontier_demand = (
                task_skills[batch_index]
                * (workload[batch_index] * frontier.float()).unsqueeze(-1)
            ).sum(dim=0)
            frontier_pressure = torch.log1p(
                frontier_demand / supply_all[batch_index].clamp_min(epsilon)
            )
            pressure_rows.append(frontier_pressure)
            demand_rows.append(frontier_demand)
            frontier_rows.append(frontier)
            before_rows.append(before)
            after_rows.append(after)
        pressure_next_frontier = torch.stack(pressure_rows, dim=0)
        next_frontier_demand = torch.stack(demand_rows, dim=0)
        next_frontier_mask = torch.stack(frontier_rows, dim=0)
        unfinished_pred_count = torch.stack(before_rows, dim=0)
        remaining_pred_count = torch.stack(after_rows, dim=0)
        assert pressure_next_frontier.shape == (batch_size, NUM_SKILL_TYPES)
        assert next_frontier_demand.shape == (batch_size, NUM_SKILL_TYPES)
        assert next_frontier_mask.shape == (batch_size, num_tasks)
        assert unfinished_pred_count.shape == remaining_pred_count.shape == (
            batch_size,
            num_tasks,
        )
        assert torch.isfinite(pressure_next_frontier).all()
        assert torch.isfinite(next_frontier_demand).all()
    return WorkerPressureContext(
        demand_all=demand_all,
        demand_near=demand_near,
        supply_all=supply_all,
        supply_near=supply_near,
        pressure_all=pressure_all,
        pressure_near=pressure_near,
        zero_supply_all=zero_supply_all,
        zero_supply_near=zero_supply_near,
        candidate_exposure=candidate_exposure,
        candidate_max_exposure=candidate_max_exposure,
        pressure_next_frontier=pressure_next_frontier,
        next_frontier_demand=next_frontier_demand,
        next_frontier_mask=next_frontier_mask,
        unfinished_physical_predecessor_count=unfinished_pred_count,
        remaining_physical_predecessor_count=remaining_pred_count,
    )


__all__ = [
    "WorkerPointerV2State",
    "WorkerPressureContext",
    "WorkerPointerV2DecodeCache",
    "gather_selected_task_skills",
    "build_v2_marginal_reserve_scarcity",
    "build_worker_eft_features",
    "build_worker_pressure_context",
    "PHYSICAL_PREDECESSOR_EDGE",
]
