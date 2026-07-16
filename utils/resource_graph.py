from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch_geometric.data import HeteroData


LEGACY_RESOURCE_EDGE = ("worker", "can_do", "task")
SKILL_FORWARD_EDGES = (
    ("worker", "has_skill", "skill"),
    ("skill", "required_by", "task"),
)
SKILL_REVERSE_EDGES = (
    ("task", "requires", "skill"),
    ("skill", "provided_by", "worker"),
)
ALL_RESOURCE_EDGES = (
    LEGACY_RESOURCE_EDGE,
    *SKILL_FORWARD_EDGES,
    *SKILL_REVERSE_EDGES,
)


class ResourceGraphConfig(Protocol):
    use_skill_hub: bool
    skill_hub_bidirectional: bool
    num_skill_types: int


@dataclass(frozen=True)
class SkillHubTopology:
    """Skill Hub 的静态边；节点特征仍按当前状态动态计算。"""

    worker_to_skill: torch.Tensor
    skill_to_task: torch.Tensor


def _empty_edge_index(device: torch.device) -> torch.Tensor:
    return torch.empty((2, 0), dtype=torch.long, device=device)


def clear_resource_graph(data: HeteroData) -> None:
    """清除可能来自另一种配置模式的资源节点与关系。"""
    for edge_type in ALL_RESOURCE_EDGES:
        if edge_type in data.edge_types:
            del data[edge_type]
    if "skill" in data.node_types:
        del data["skill"]


def build_worker_skill_edges(worker_x: torch.Tensor, num_skill_types: int) -> torch.Tensor:
    assert worker_x.dim() == 2, f"worker_x 必须为二维张量，实际为 {tuple(worker_x.shape)}"
    assert worker_x.size(1) >= 1 + num_skill_types, (
        f"worker_x 特征不足: {worker_x.size(1)} < {1 + num_skill_types}"
    )
    worker_ids, skill_ids = torch.nonzero(
        worker_x[:, 1 : 1 + num_skill_types] > 0.5,
        as_tuple=True,
    )
    if worker_ids.numel() == 0:
        return _empty_edge_index(worker_x.device)
    return torch.stack((worker_ids, skill_ids), dim=0).long()


def build_task_skill_edges(task_x: torch.Tensor, num_skill_types: int) -> torch.Tensor:
    assert task_x.dim() == 2, f"task_x 必须为二维张量，实际为 {tuple(task_x.shape)}"
    assert task_x.size(1) >= 5 + num_skill_types, (
        f"task_x 特征不足: {task_x.size(1)} < {5 + num_skill_types}"
    )
    skill_slice = task_x[:, 5 : 5 + num_skill_types]
    has_skill = skill_slice.sum(dim=1) > 0.5
    task_ids = torch.nonzero(has_skill, as_tuple=False).squeeze(1)
    if task_ids.numel() == 0:
        return _empty_edge_index(task_x.device)
    skill_ids = torch.argmax(skill_slice[task_ids], dim=1)
    return torch.stack((skill_ids, task_ids), dim=0).long()


def build_skill_hub_topology(
    task_x: torch.Tensor,
    worker_x: torch.Tensor,
    num_skill_types: int,
) -> SkillHubTopology:
    """从静态技能列生成可跨 step 复用的 Skill Hub 拓扑。"""
    return SkillHubTopology(
        worker_to_skill=build_worker_skill_edges(worker_x, num_skill_types),
        skill_to_task=build_task_skill_edges(task_x, num_skill_types),
    )


def worker_topology_key(worker_x: torch.Tensor, num_skill_types: int) -> tuple[int, bytes]:
    """生成只依赖工人技能矩阵的稳定缓存键。"""
    assert worker_x.dim() == 2
    skill_mask = (
        worker_x[:, 1 : 1 + num_skill_types]
        .detach()
        .to(device="cpu")
        .gt(0.5)
        .contiguous()
    )
    return int(worker_x.size(0)), skill_mask.numpy().tobytes()


def build_skill_features(worker_x: torch.Tensor, num_skill_types: int) -> torch.Tensor:
    """构建 Skill 节点特征：[技能 one-hot, 六项资源统计]。"""
    assert worker_x.dim() == 2, f"worker_x 必须为二维张量，实际为 {tuple(worker_x.shape)}"
    skill_mask = worker_x[:, 1 : 1 + num_skill_types] > 0.5
    num_workers = max(1, int(worker_x.size(0)))
    counts = skill_mask.sum(dim=0)
    safe_counts = counts.clamp(min=1).to(worker_x.dtype)

    one_hot = torch.eye(
        num_skill_types,
        dtype=worker_x.dtype,
        device=worker_x.device,
    )
    eligible_ratio = counts.to(worker_x.dtype) / float(num_workers)

    is_free = (
        worker_x[:, 12] > 0.5
        if worker_x.size(1) > 12
        else torch.ones(worker_x.size(0), dtype=torch.bool, device=worker_x.device)
    )
    is_mobile = (
        worker_x[:, 13] > 0.5
        if worker_x.size(1) > 13
        else torch.ones(worker_x.size(0), dtype=torch.bool, device=worker_x.device)
    )
    efficiency = worker_x[:, 0]
    wait_time = worker_x[:, 11] if worker_x.size(1) > 11 else torch.zeros_like(efficiency)
    fatigue = worker_x[:, 21] if worker_x.size(1) > 21 else torch.ones_like(efficiency)

    mask_f = skill_mask.to(worker_x.dtype)
    free_ratio = (mask_f * is_free[:, None]).sum(dim=0) / safe_counts
    mobile_ratio = (mask_f * is_mobile[:, None]).sum(dim=0) / safe_counts
    mean_efficiency = (mask_f * efficiency[:, None]).sum(dim=0) / safe_counts
    mean_wait = (mask_f * wait_time[:, None]).sum(dim=0) / safe_counts
    mean_fatigue = (mask_f * fatigue[:, None]).sum(dim=0) / safe_counts

    has_workers = counts > 0
    dynamic = torch.stack(
        (
            eligible_ratio,
            free_ratio,
            mobile_ratio,
            mean_efficiency,
            mean_wait,
            mean_fatigue,
        ),
        dim=1,
    )
    dynamic = dynamic * has_workers[:, None].to(dynamic.dtype)
    return torch.cat((one_hot, dynamic), dim=1)


def apply_resource_graph(
    data: HeteroData,
    task_x: torch.Tensor,
    worker_x: torch.Tensor,
    config: ResourceGraphConfig,
    *,
    skill_hub_topology: SkillHubTopology | None = None,
) -> HeteroData:
    """按配置写入旧直接边或 Skill Hub，二者不会同时启用。"""
    clear_resource_graph(data)
    num_skill_types = int(config.num_skill_types)
    if num_skill_types <= 0:
        raise ValueError("num_skill_types 必须大于 0")

    if not bool(config.use_skill_hub):
        worker_skill = build_worker_skill_edges(worker_x, num_skill_types)
        task_skill = build_task_skill_edges(task_x, num_skill_types)
        skill_to_tasks: list[list[int]] = [[] for _ in range(num_skill_types)]
        for edge_idx in range(task_skill.size(1)):
            skill_to_tasks[int(task_skill[0, edge_idx])].append(int(task_skill[1, edge_idx]))

        src_parts = []
        dst_parts = []
        for edge_idx in range(worker_skill.size(1)):
            worker_id = int(worker_skill[0, edge_idx])
            skill_id = int(worker_skill[1, edge_idx])
            tasks = skill_to_tasks[skill_id]
            if tasks:
                src_parts.append(torch.full((len(tasks),), worker_id, dtype=torch.long, device=worker_x.device))
                dst_parts.append(torch.tensor(tasks, dtype=torch.long, device=worker_x.device))
        data[LEGACY_RESOURCE_EDGE].edge_index = (
            torch.stack((torch.cat(src_parts), torch.cat(dst_parts)), dim=0)
            if src_parts
            else _empty_edge_index(worker_x.device)
        )
        return data

    topology = skill_hub_topology or build_skill_hub_topology(
        task_x,
        worker_x,
        num_skill_types,
    )
    worker_to_skill = topology.worker_to_skill.to(worker_x.device)
    skill_to_task = topology.skill_to_task.to(task_x.device)
    if worker_to_skill.numel() and int(worker_to_skill[0].max()) >= worker_x.size(0):
        raise ValueError("缓存的 Worker-Skill 拓扑与当前工人节点数不匹配")
    if skill_to_task.numel() and int(skill_to_task[1].max()) >= task_x.size(0):
        raise ValueError("缓存的 Task-Skill 拓扑与当前任务节点数不匹配")
    skill_x = build_skill_features(worker_x, num_skill_types)
    expected_skill_dim = int(getattr(config, "skill_feat_dim", skill_x.size(1)))
    if skill_x.size(1) != expected_skill_dim:
        raise ValueError(
            f"skill_feat_dim 配置错误: {expected_skill_dim}，实际需要 {skill_x.size(1)}"
        )
    data["skill"].x = skill_x
    data[SKILL_FORWARD_EDGES[0]].edge_index = worker_to_skill
    data[SKILL_FORWARD_EDGES[1]].edge_index = skill_to_task

    if bool(config.skill_hub_bidirectional):
        data[SKILL_REVERSE_EDGES[0]].edge_index = torch.stack(
            (skill_to_task[1], skill_to_task[0]),
            dim=0,
        )
        data[SKILL_REVERSE_EDGES[1]].edge_index = torch.stack(
            (worker_to_skill[1], worker_to_skill[0]),
            dim=0,
        )
    return data


def apply_batched_resource_graph(
    data: HeteroData,
    task_x: torch.Tensor,
    worker_x: torch.Tensor,
    *,
    batch_size: int,
    num_tasks: int,
    num_workers: int,
    config: ResourceGraphConfig,
    task_skill_edges: torch.Tensor | None = None,
    worker_skill_edges: list[torch.Tensor] | None = None,
) -> HeteroData:
    """为固定节点数的 PyG Batch 构建互不跨图的资源关系。"""
    assert task_x.shape[0] == batch_size * num_tasks
    assert worker_x.shape[0] == batch_size * num_workers
    clear_resource_graph(data)

    num_skill_types = int(config.num_skill_types)
    if not bool(config.use_skill_hub):
        edge_parts: list[torch.Tensor] = []
        for batch_idx in range(batch_size):
            local = HeteroData()
            task_slice = task_x[batch_idx * num_tasks : (batch_idx + 1) * num_tasks]
            worker_slice = worker_x[batch_idx * num_workers : (batch_idx + 1) * num_workers]
            apply_resource_graph(local, task_slice, worker_slice, config)
            edge_index = local[LEGACY_RESOURCE_EDGE].edge_index.clone()
            edge_index[0] += batch_idx * num_workers
            edge_index[1] += batch_idx * num_tasks
            edge_parts.append(edge_index)
        data[LEGACY_RESOURCE_EDGE].edge_index = torch.cat(edge_parts, dim=1)
        return data

    if task_skill_edges is None:
        task_skill_edges = build_task_skill_edges(task_x[:num_tasks], num_skill_types)
    if worker_skill_edges is None:
        worker_skill_edges = [
            build_worker_skill_edges(
                worker_x[batch_idx * num_workers : (batch_idx + 1) * num_workers],
                num_skill_types,
            )
            for batch_idx in range(batch_size)
        ]
    if len(worker_skill_edges) != batch_size:
        raise ValueError(
            f"Worker-Skill 拓扑数量错误: {len(worker_skill_edges)} != {batch_size}"
        )

    skill_parts = [
        build_skill_features(
            worker_x[batch_idx * num_workers : (batch_idx + 1) * num_workers],
            num_skill_types,
        )
        for batch_idx in range(batch_size)
    ]
    expected_skill_dim = int(getattr(config, "skill_feat_dim", skill_parts[0].size(1)))
    if any(part.size(1) != expected_skill_dim for part in skill_parts):
        raise ValueError(f"skill_feat_dim 配置错误: {expected_skill_dim}")
    data["skill"].x = torch.cat(skill_parts, dim=0)

    worker_parts = []
    task_parts = []
    for batch_idx, worker_edges in enumerate(worker_skill_edges):
        worker_edge_index = worker_edges.to(worker_x.device).clone()
        task_edge_index = task_skill_edges.to(task_x.device).clone()
        worker_edge_index[0] += batch_idx * num_workers
        worker_edge_index[1] += batch_idx * num_skill_types
        task_edge_index[0] += batch_idx * num_skill_types
        task_edge_index[1] += batch_idx * num_tasks
        worker_parts.append(worker_edge_index)
        task_parts.append(task_edge_index)

    worker_to_skill = torch.cat(worker_parts, dim=1)
    skill_to_task = torch.cat(task_parts, dim=1)
    data[SKILL_FORWARD_EDGES[0]].edge_index = worker_to_skill
    data[SKILL_FORWARD_EDGES[1]].edge_index = skill_to_task
    if bool(config.skill_hub_bidirectional):
        data[SKILL_REVERSE_EDGES[0]].edge_index = torch.stack(
            (skill_to_task[1], skill_to_task[0]),
            dim=0,
        )
        data[SKILL_REVERSE_EDGES[1]].edge_index = torch.stack(
            (worker_to_skill[1], worker_to_skill[0]),
            dim=0,
        )
    return data
