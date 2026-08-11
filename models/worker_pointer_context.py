"""WorkerPointer v2 的纯函数式五技能压力上下文。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


NUM_SKILL_TYPES = 5
TASK_SKILL_SLICE = slice(5, 10)
TASK_DEMAND_INDEX = 16
WORKER_SKILL_SLICE = slice(1, 6)
WORKER_LOG_WAIT_INDEX = 6


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


@dataclass(frozen=True)
class WorkerPointerV2State:
    """自回归团队的增量 DeepSets 状态。"""

    mapped_sum: torch.Tensor
    mapped_max: torch.Tensor
    count: torch.Tensor
    selected_skill_sum: torch.Tensor


@dataclass(frozen=True)
class WorkerPointerV2DecodeCache:
    """单个工序—工位团队解码中不随成员选择变化的 float32 张量。"""

    candidate_keys: torch.Tensor
    query_prefix: torch.Tensor
    pressure_features: torch.Tensor
    supply_all: torch.Tensor
    demand: torch.Tensor


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
    )


__all__ = [
    "WorkerPointerV2State",
    "WorkerPressureContext",
    "WorkerPointerV2DecodeCache",
    "build_worker_pressure_context",
]
