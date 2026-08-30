# -*- coding: utf-8 -*-
"""WorkerPointer v2 Fast-Exact：rollout/replay 共用的 GPU 图构建器接口。

- ``V2FastExactBatchBuilder``：统一 Protocol；
- ``CPUExactBatchBuilder``：测试、CPU deterministic eval 与异步验证的参考实现，
  不作为 GPU 训练的自动 fallback；
- ``V2FastExactBatch``：GPU/CPU 图 Batch + 布局元数据 + 轨迹身份 + 原始特征切片。

关键设计：
- 布局元数据（各节点类型 ptr / 每图节点数）在组创建阶段从 CPU 态 Batch 提取，
  热路径禁止对 GPU 张量做 ``.cpu().tolist()`` 与逐样本 ``.item()``；
- 组内每图 worker 数允许不同（异质），以每图 ptr/offset 而非固定偏移表示；
- 原始 task/worker 特征切片在迁移到目标设备前 clone，供压力上下文与动作 mask
  计算直接读取。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np
import torch
from torch_geometric.data import Batch, HeteroData

from configs import Config
from models.worker_pointer_context import PHYSICAL_PREDECESSOR_EDGE
from utils.resource_graph import (
    LEGACY_RESOURCE_EDGE,
    SKILL_FORWARD_EDGES,
    SKILL_REVERSE_EDGES,
    apply_resource_graph,
    build_skill_features,
    build_task_skill_edges,
    build_worker_skill_edges,
    clear_resource_graph,
    worker_topology_key,
)
from worker_feature_layout import resolve_worker_feature_layout


EnvironmentSnapshot = dict
ActionMasks = Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None


@dataclass
class V2FastExactBatch:
    """一次组级图构建的完整产物。"""

    batch: Any  # HeteroData/Batch，已在目标 device，含 batch 向量
    # 布局元数据：每图各节点类型的起始偏移（长度 group_size + 1）
    task_ptr: Tuple[int, ...]
    worker_ptr: Tuple[int, ...]
    station_ptr: Tuple[int, ...]
    # 每图节点数（异质 worker 数以列表而非单一值表示）
    task_counts: Tuple[int, ...]
    worker_counts: Tuple[int, ...]
    station_counts: Tuple[int, ...]
    # 轨迹身份元数据
    memory_indices: Tuple[int, ...]
    group_positions: Tuple[int, ...]
    group_id: object | None = None
    # 每图原始特征切片（独立 CPU tensor），供压力上下文与 mask 计算
    raw_task_slices: list[torch.Tensor] = field(default_factory=list)
    raw_worker_slices: list[torch.Tensor] = field(default_factory=list)
    # 节点级动作 mask（可选绑定）
    task_mask: Any = None
    station_mask: Any = None
    worker_mask: Any = None
    # Skill Hub 布局（启用 skill_hub 时非 None）
    skill_ptr: Tuple[int, ...] | None = None
    skill_counts: Tuple[int, ...] | None = None

    @property
    def group_size(self) -> int:
        """物理组内轨迹数。"""
        return len(self.memory_indices)


@runtime_checkable
class V2FastExactBatchBuilder(Protocol):
    """统一构建接口：snapshot 组 -> V2FastExactBatch。"""

    def build(
        self,
        snapshots: Sequence[EnvironmentSnapshot],
        masks: ActionMasks = None,
        memory_indices: Sequence[int] | None = None,
        *,
        group_id: object | None = None,
    ) -> V2FastExactBatch:
        ...


def _extract_ptr(counts: Sequence[int]) -> Tuple[int, ...]:
    """由每图节点数派生单调 ptr（含末尾偏移）。"""
    ptr = [0]
    total = 0
    for count in counts:
        total += int(count)
        ptr.append(total)
    return tuple(ptr)


def _cumulative_offsets(counts: Sequence[int]) -> Tuple[int, ...]:
    """每图节点起始偏移（长度 group_size + 1）。"""
    offsets = [0]
    total = 0
    for count in counts:
        total += int(count)
        offsets.append(total)
    return tuple(offsets)


def apply_batched_resource_graph_offsets(
    data: HeteroData,
    task_x: torch.Tensor,
    worker_x: torch.Tensor,
    *,
    task_offsets: Tuple[int, ...],
    worker_offsets: Tuple[int, ...],
    config: Any,
    task_skill_edges: torch.Tensor | None = None,
    worker_skill_edges: list[torch.Tensor] | None = None,
) -> HeteroData:
    """异质 worker 数版本的批量资源图装配（每图 worker 节点数可不同）。

    与 ``apply_batched_resource_graph`` 数值一致，仅将固定 ``batch_idx * num_workers``
    偏移替换为逐图累积的 ``worker_offsets`` / ``task_offsets``。
    """
    group_size = len(worker_offsets) - 1
    if len(task_offsets) != group_size + 1:
        raise ValueError(
            f"task_offsets 长度与组大小不一致: {len(task_offsets)} != {group_size + 1}"
        )
    num_skill_types = int(config.num_skill_types)
    worker_layout = resolve_worker_feature_layout(config)
    if worker_x.size(1) != worker_layout.total_dim:
        raise ValueError(
            f"批量 worker_x 特征维度错误: {worker_x.size(1)} != {worker_layout.total_dim}"
        )
    clear_resource_graph(data)

    def _slice(x: torch.Tensor, offsets: Tuple[int, ...], index: int) -> torch.Tensor:
        return x[offsets[index] : offsets[index + 1]]

    if not bool(config.use_skill_hub):
        edge_parts: list[torch.Tensor] = []
        for index in range(group_size):
            local = HeteroData()
            task_slice = _slice(task_x, task_offsets, index)
            worker_slice = _slice(worker_x, worker_offsets, index)
            apply_resource_graph(local, task_slice, worker_slice, config)
            edge_index = local[LEGACY_RESOURCE_EDGE].edge_index.clone()
            edge_index[0] += worker_offsets[index]
            edge_index[1] += task_offsets[index]
            edge_parts.append(edge_index)
        data[LEGACY_RESOURCE_EDGE].edge_index = torch.cat(edge_parts, dim=1)
        return data

    if task_skill_edges is None:
        task_skill_edges = build_task_skill_edges(
            task_x[: task_offsets[1]],
            num_skill_types,
        )
    if worker_skill_edges is None:
        worker_skill_edges = [
            build_worker_skill_edges(
                _slice(worker_x, worker_offsets, index),
                num_skill_types,
            )
            for index in range(group_size)
        ]
    if len(worker_skill_edges) != group_size:
        raise ValueError(
            f"Worker-Skill 拓扑数量错误: {len(worker_skill_edges)} != {group_size}"
        )

    skill_parts = [
        build_skill_features(
            _slice(worker_x, worker_offsets, index),
            num_skill_types,
            worker_layout.skill_slots,
        )
        for index in range(group_size)
    ]
    expected_skill_dim = int(
        getattr(config, "skill_feat_dim", skill_parts[0].size(1))
    )
    if any(part.size(1) != expected_skill_dim for part in skill_parts):
        raise ValueError(f"skill_feat_dim 配置错误: {expected_skill_dim}")
    data["skill"].x = torch.cat(skill_parts, dim=0)

    skill_offsets = _cumulative_offsets(
        [num_skill_types] * group_size
    )
    worker_parts: list[torch.Tensor] = []
    task_parts: list[torch.Tensor] = []
    for index, worker_edges in enumerate(worker_skill_edges):
        worker_edge_index = worker_edges.to(worker_x.device).clone()
        task_edge_index = task_skill_edges.to(task_x.device).clone()
        worker_edge_index[0] += worker_offsets[index]
        worker_edge_index[1] += skill_offsets[index]
        task_edge_index[0] += skill_offsets[index]
        task_edge_index[1] += task_offsets[index]
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


class CPUExactBatchBuilder:
    """参考实现：逐 snapshot CPU rebuild 后拼接 PyG Batch。

    用于测试、CPU deterministic eval 与异步验证；不承担训练 GPU fallback 职责。
    """

    def __init__(
        self,
        *,
        config: Config,
        env: Any,
        device: torch.device,
    ) -> None:
        self.config = config
        self.env = env
        self.device = device

    def build(
        self,
        snapshots: Sequence[EnvironmentSnapshot],
        masks: ActionMasks = None,
        memory_indices: Sequence[int] | None = None,
        *,
        group_id: object | None = None,
    ) -> V2FastExactBatch:
        if not snapshots:
            raise ValueError("CPUExactBatchBuilder 收到空 snapshots")

        data_list = [
            self.env.rebuild_state_from_snapshot(snapshot)
            for snapshot in snapshots
        ]
        group_size = len(data_list)
        batch = Batch.from_data_list(data_list)

        # 布局元数据在 CPU 态提取，避免热路径 GPU 同步。
        task_ptr = tuple(int(p) for p in batch["task"].ptr.tolist())
        worker_ptr = tuple(int(p) for p in batch["worker"].ptr.tolist())
        station_ptr = tuple(int(p) for p in batch["station"].ptr.tolist())
        task_counts = tuple(
            task_ptr[index + 1] - task_ptr[index] for index in range(group_size)
        )
        worker_counts = tuple(
            worker_ptr[index + 1] - worker_ptr[index]
            for index in range(group_size)
        )
        station_counts = tuple(
            station_ptr[index + 1] - station_ptr[index]
            for index in range(group_size)
        )
        skill_ptr: tuple[int, ...] | None = None
        skill_counts: tuple[int, ...] | None = None
        if "skill" in batch.node_types:
            skill_ptr = tuple(int(p) for p in batch["skill"].ptr.tolist())
            skill_counts = tuple(
                skill_ptr[index + 1] - skill_ptr[index]
                for index in range(group_size)
            )

        # 原始特征切片在迁移到目标设备前 clone，保持独立引用。
        raw_task_slices = [
            data_list[index]["task"].x.clone() for index in range(group_size)
        ]
        raw_worker_slices = [
            data_list[index]["worker"].x.clone() for index in range(group_size)
        ]

        if memory_indices is None:
            memory_indices = tuple(range(group_size))
        resolved_indices = tuple(int(index) for index in memory_indices)
        if len(resolved_indices) != group_size:
            raise ValueError(
                "memory_indices 数量与 snapshot 组大小不一致: "
                f"{len(resolved_indices)} != {group_size}"
            )

        task_mask = station_mask = worker_mask = None
        if masks is not None:
            if len(masks) != group_size:
                raise ValueError(
                    f"masks 数量与 snapshot 组大小不一致: {len(masks)} != {group_size}"
                )
            task_mask = torch.cat([mask[0] for mask in masks], dim=0)
            station_mask = torch.cat([mask[1] for mask in masks], dim=0)
            worker_mask = torch.cat([mask[2] for mask in masks], dim=0)

        batch = batch.to(self.device)
        if task_mask is not None:
            # 节点级 mask 绑定到 Batch，供同形重放逐节点读取。
            batch.y_task_mask = task_mask.to(self.device)
            batch.y_station_mask = station_mask.to(self.device)
            batch.y_worker_mask = worker_mask.to(self.device)
        return V2FastExactBatch(
            batch=batch,
            task_ptr=task_ptr,
            worker_ptr=worker_ptr,
            station_ptr=station_ptr,
            task_counts=task_counts,
            worker_counts=worker_counts,
            station_counts=station_counts,
            memory_indices=resolved_indices,
            group_positions=tuple(range(group_size)),
            group_id=group_id,
            raw_task_slices=raw_task_slices,
            raw_worker_slices=raw_worker_slices,
            task_mask=task_mask,
            station_mask=station_mask,
            worker_mask=worker_mask,
            skill_ptr=skill_ptr,
            skill_counts=skill_counts,
        )


_SNAPSHOT_REQUIRED_KEYS = (
    "worker_free_time",
    "base_task_x",
    "base_worker_x",
    "current_time",
    "task_status",
    "worker_locks",
    "station_loads",
    "assigned_tasks",
)


class GPUExactBatchBuilder:
    """生产实现：GPU 常驻图模板原位更新，支持组内异质 worker 数。

    - 模板缓存键含数据集、组大小、每图节点数、worker 拓扑与特征布局版本；
    - 布局元数据（各节点类型 ptr / 每图节点数）在模板创建阶段提取为 CPU
      整数元数据并挂在模板上，热路径不再对 GPU 张量执行 ``.cpu().tolist()``；
    - 动态特征与动态边通过批量张量原位覆写，静态拓扑与 batch 向量常驻；
    - 严格失败：模板键不匹配即重建（禁止复用不兼容模板），组内数据集不一致
      或快照缺字段直接抛错，不降级不切 CPU。
    """

    def __init__(
        self,
        *,
        config: Config,
        env: Any,
        device: torch.device,
        max_templates: int = 8,
    ) -> None:
        self.config = config
        self.env = env
        self.device = device
        self.max_templates = max(1, int(max_templates))
        # 有界 LRU：键 -> 模板 Batch（含 _fast_exact_layout 布局元数据）
        self._templates: "OrderedDict[tuple, Any]" = OrderedDict()
        # (dataset_idx, worker_topology_key) -> worker->skill 边
        self._worker_skill_topologies: dict = {}
        self.template_hits = 0
        self.template_misses = 0

    def _validate_snapshots(self, snapshots: Sequence[EnvironmentSnapshot]) -> int:
        if not snapshots:
            raise ValueError("GPUExactBatchBuilder 收到空 snapshots")
        dataset_idx = int(snapshots[0].get("dataset_idx", 0))
        for index, snap in enumerate(snapshots):
            missing = [
                key for key in _SNAPSHOT_REQUIRED_KEYS if key not in snap
            ]
            if missing:
                raise KeyError(
                    f"snapshot[{index}] 缺少 GPUExactBatchBuilder 必需字段: {missing}"
                )
            if int(snap.get("dataset_idx", 0)) != dataset_idx:
                raise ValueError(
                    "Fast-Exact 组内禁止混合数据集: "
                    f"snapshot[0].dataset_idx={dataset_idx} != snapshot[{index}].dataset_idx"
                )
        return dataset_idx

    def _template_key(
        self,
        *,
        dataset_idx: int,
        task_counts: Tuple[int, ...],
        worker_counts: Tuple[int, ...],
        station_counts: Tuple[int, ...],
        topo_keys: Tuple[object, ...],
    ) -> tuple:
        layout = resolve_worker_feature_layout(self.config)
        return (
            dataset_idx,
            tuple(task_counts),
            tuple(worker_counts),
            tuple(station_counts),
            bool(getattr(self.config, "use_skill_hub", False)),
            bool(getattr(self.config, "skill_hub_bidirectional", False)),
            layout.total_dim,
            int(getattr(self.config, "task_feat_dim", 18)),
            int(getattr(self.config, "station_feat_dim", 15)),
            tuple(topo_keys),
        )

    def _create_template(
        self,
        *,
        dataset_idx: int,
        task_counts: Tuple[int, ...],
        worker_counts: Tuple[int, ...],
        station_counts: Tuple[int, ...],
        ctx: Any,
    ) -> Any:
        """为当前组签名构建常驻模板（静态骨架 + batch 向量 + 布局元数据）。"""
        group_size = len(worker_counts)
        num_skills = int(getattr(self.config, "num_skill_types", 5))
        use_hub = bool(getattr(self.config, "use_skill_hub", False))
        worker_dim = resolve_worker_feature_layout(self.config).total_dim
        precedes = (
            ctx["base_data"]["task", "precedes", "task"].edge_index.clone()
            if ("task", "precedes", "task") in ctx["base_data"].edge_types
            else torch.empty((2, 0), dtype=torch.long)
        )
        physical_precedes = (
            ctx["base_data"][PHYSICAL_PREDECESSOR_EDGE].edge_index.clone()
            if PHYSICAL_PREDECESSOR_EDGE in ctx["base_data"].edge_types
            else torch.empty((2, 0), dtype=torch.long)
        )

        data_list: list[HeteroData] = []
        for index in range(group_size):
            data = HeteroData()
            data["task"].x = torch.zeros(
                (task_counts[index], int(self.config.task_feat_dim))
            )
            data["worker"].x = torch.zeros((worker_counts[index], worker_dim))
            data["station"].x = torch.zeros(
                (station_counts[index], int(self.config.station_feat_dim))
            )
            if use_hub:
                data["skill"].x = torch.zeros(
                    (num_skills, int(self.config.skill_feat_dim))
                )
            data["task", "precedes", "task"].edge_index = precedes.clone()
            data[PHYSICAL_PREDECESSOR_EDGE].edge_index = physical_precedes.clone()
            # 动态边占位，随后原位覆写
            data["task", "assigned_to", "station"].edge_index = torch.empty(
                (2, 0), dtype=torch.long
            )
            data["station", "has_task", "task"].edge_index = torch.empty(
                (2, 0), dtype=torch.long
            )
            data["task", "done_by", "worker"].edge_index = torch.empty(
                (2, 0), dtype=torch.long
            )
            data_list.append(data)

        template = Batch.from_data_list(data_list).to(self.device)
        # 布局元数据在模板创建阶段提取为 CPU 整数元数据（热路径免同步）。
        template._fast_exact_layout = {
            "task_ptr": tuple(int(p) for p in template["task"].ptr.tolist()),
            "worker_ptr": tuple(int(p) for p in template["worker"].ptr.tolist()),
            "station_ptr": tuple(int(p) for p in template["station"].ptr.tolist()),
            "task_counts": tuple(
                int(p)
                for p in template["task"].ptr.diff().tolist()
            ) if hasattr(template["task"].ptr, "diff") else task_counts,
            "worker_counts": tuple(
                int(p)
                for p in template["worker"].ptr.diff().tolist()
            ) if hasattr(template["worker"].ptr, "diff") else worker_counts,
            "station_counts": tuple(
                int(p)
                for p in template["station"].ptr.diff().tolist()
            ) if hasattr(template["station"].ptr, "diff") else station_counts,
        }
        if "skill" in template.node_types:
            template._fast_exact_layout["skill_ptr"] = tuple(
                int(p) for p in template["skill"].ptr.tolist()
            )
        self._templates = OrderedDict(
            (key, value) for key, value in self._templates.items()
        )
        return template

    def _get_or_create_template(
        self,
        *,
        key: tuple,
        ctx: Any,
        task_counts: Tuple[int, ...],
        worker_counts: Tuple[int, ...],
        station_counts: Tuple[int, ...],
        dataset_idx: int,
    ) -> Any:
        if key in self._templates:
            self._templates.move_to_end(key)
            self.template_hits += 1
            return self._templates[key]
        template = self._create_template(
            dataset_idx=dataset_idx,
            task_counts=task_counts,
            worker_counts=worker_counts,
            station_counts=station_counts,
            ctx=ctx,
        )
        self._templates[key] = template
        self.template_misses += 1
        while len(self._templates) > self.max_templates:
            self._templates.popitem(last=False)
        return template

    def _rebuild_in_place(
        self,
        template: Any,
        *,
        ctx: Any,
        snapshots: Sequence[EnvironmentSnapshot],
        task_offsets: Tuple[int, ...],
        worker_offsets: Tuple[int, ...],
        station_offsets: Tuple[int, ...],
        dataset_idx: int,
        topo_keys: Tuple[object, ...],
    ) -> Any:
        """批量原位覆写动态特征与动态边；worker 节点偏移逐图累积。"""
        t = self.device
        group_size = len(snapshots)
        num_tasks = int(ctx["num_tasks"])
        num_stations = int(ctx["base_station_x"].shape[0])
        total_workers = worker_offsets[-1]
        total_tasks = task_offsets[-1]
        mean_task_time = float(ctx["mean_task_time"])
        ideal_station_load = float(ctx["ideal_station_load"])
        worker_layout = resolve_worker_feature_layout(self.config)
        num_skills = int(self.config.num_skill_types)
        use_hub = bool(getattr(self.config, "use_skill_hub", False))
        max_slots = int(getattr(self.config, "max_slots_per_station", 3))

        # 1. 静态特征（批量拼接后整体覆写）
        base_task_x = torch.cat(
            [torch.as_tensor(snap["base_task_x"]) for snap in snapshots],
            dim=0,
        ).to(t)
        if base_task_x.shape != (total_tasks, int(self.config.task_feat_dim)):
            raise ValueError(
                f"GPU 快照任务特征形状不一致: {tuple(base_task_x.shape)}"
            )
        template["task"].x = base_task_x
        base_worker_x = torch.cat(
            [torch.as_tensor(snap["base_worker_x"]) for snap in snapshots],
            dim=0,
        ).to(t)
        if base_worker_x.shape != (total_workers, worker_layout.total_dim):
            raise ValueError(
                f"GPU 快照工人特征形状不一致: {tuple(base_worker_x.shape)}"
            )
        template["worker"].x = base_worker_x
        template["station"].x = ctx["base_station_x"].to(t).repeat(group_size, 1)

        # 2. 动态状态收集（numpy 中间层只在 CPU 侧收集，不触发 GPU 同步）
        statuses = np.concatenate([snap["task_status"] for snap in snapshots])
        mat_readies = np.concatenate(
            [snap.get("task_material_ready", np.zeros(num_tasks)) for snap in snapshots]
        )
        current_times = [float(snap["current_time"]) for snap in snapshots]
        worker_free_times = np.concatenate(
            [snap["worker_free_time"] for snap in snapshots]
        )
        worker_locks = np.concatenate([snap["worker_locks"] for snap in snapshots])
        worker_cum = np.concatenate(
            [
                snap.get("worker_cumulative_work", np.zeros(worker_offsets[index + 1] - worker_offsets[index]))
                for index, snap in enumerate(snapshots)
            ]
        )
        worker_last = np.concatenate(
            [
                snap.get("worker_last_busy_end", np.zeros(worker_offsets[index + 1] - worker_offsets[index]))
                for index, snap in enumerate(snapshots)
            ]
        )
        station_loads = np.concatenate([snap["station_loads"] for snap in snapshots])
        station_slots = np.concatenate(
            [
                snap.get(
                    "station_available_slots",
                    np.full(num_stations, max_slots),
                )
                for snap in snapshots
            ]
        )
        station_wait_times = np.zeros((group_size, num_stations), dtype=np.float32)

        ts_src: list[int] = []
        ts_dst: list[int] = []
        tw_src: list[int] = []
        tw_dst: list[int] = []
        for index, snap in enumerate(snapshots):
            task_offset = task_offsets[index]
            station_offset = station_offsets[index]
            worker_offset = worker_offsets[index]
            cur_t = current_times[index]
            station_finish_lists: list[list[float]] = [[] for _ in range(num_stations)]
            for t_id, s_id, team, _start_t, finish_t in snap["assigned_tasks"]:
                if s_id != -1:
                    if not (0 <= int(t_id) < num_tasks):
                        raise ValueError(
                            "Fast-Exact assigned_tasks 任务索引越界: "
                            f"t_id={int(t_id)} num_tasks={num_tasks}"
                        )
                    if not (0 <= int(s_id) < num_stations):
                        raise ValueError(
                            "Fast-Exact assigned_tasks 工位索引越界: "
                            f"s_id={int(s_id)} num_stations={num_stations}"
                        )
                    num_workers_here = worker_offsets[index + 1] - worker_offsets[index]
                    for w_id in team:
                        if not (0 <= int(w_id) < num_workers_here):
                            raise ValueError(
                                "Fast-Exact assigned_tasks 工人索引越界: "
                                f"w_id={int(w_id)} num_workers={num_workers_here}"
                            )
                    ts_src.append(t_id + task_offset)
                    ts_dst.append(s_id + station_offset)
                    for w_id in team:
                        tw_src.append(t_id + task_offset)
                        tw_dst.append(w_id + worker_offset)
                    if finish_t > cur_t:
                        station_finish_lists[s_id].append(finish_t)
            allowed_slots = station_slots[
                index * num_stations : (index + 1) * num_stations
            ]
            for s in range(num_stations):
                lst = station_finish_lists[s]
                lst.sort()
                lim = int(allowed_slots[s])
                if len(lst) >= lim:
                    station_wait_times[index, s] = max(0.0, lst[0] - cur_t)

        # 3. 批量张量打包与单次 H2D
        flat_status = torch.as_tensor(statuses, dtype=torch.long, device=t)
        flat_mat_ready = torch.as_tensor(mat_readies, dtype=torch.float32, device=t)
        flat_current_times_t = (
            torch.tensor(current_times, dtype=torch.float32, device=t)
            .repeat_interleave(num_tasks)
        )
        flat_free_time = torch.as_tensor(
            worker_free_times, dtype=torch.float32, device=t
        )
        flat_current_times_w = (
            torch.tensor(current_times, dtype=torch.float32, device=t)
            .repeat_interleave(
                torch.tensor(
                    [worker_offsets[index + 1] - worker_offsets[index] for index in range(group_size)],
                    dtype=torch.long,
                    device=t,
                )
            )
        )
        flat_locks = torch.as_tensor(worker_locks, dtype=torch.long, device=t)
        flat_cum = torch.as_tensor(worker_cum, dtype=torch.float32, device=t)
        flat_last = torch.as_tensor(worker_last, dtype=torch.float32, device=t)
        flat_loads = torch.as_tensor(station_loads, dtype=torch.float32, device=t)
        flat_slots = torch.as_tensor(station_slots, dtype=torch.float32, device=t)
        flat_station_wait = torch.as_tensor(
            station_wait_times, dtype=torch.float32, device=t
        ).reshape(-1)

        # 4. Task 动态特征覆写
        template["task"].x[:, 1:5] = 0.0
        template["task"].x[
            torch.arange(total_tasks, device=t), flat_status + 1
        ] = 1.0
        wait_t = torch.clamp(flat_mat_ready - flat_current_times_t, min=0.0)
        template["task"].x[:, 17] = torch.log1p(wait_t / mean_task_time)

        # 5. Worker 动态特征覆写
        wait_w = torch.clamp(flat_free_time - flat_current_times_w, min=0.0)
        template["worker"].x[:, worker_layout.wait_idx] = torch.log1p(
            wait_w / mean_task_time
        )
        template["worker"].x[:, worker_layout.free_idx] = (
            flat_free_time <= flat_current_times_w
        ).float()
        template["worker"].x[:, worker_layout.lock_slice] = 0.0
        flat_locks_clamped = torch.clamp(flat_locks, max=7)
        template["worker"].x[
            torch.arange(total_workers, device=t),
            worker_layout.lock_start + flat_locks_clamped,
        ] = 1.0

        fatigue_recovery = float(getattr(self.config, "fatigue_recovery_ratio", 0.5))
        fatigue_threshold = float(getattr(self.config, "fatigue_threshold_hours", 4.0))
        fatigue_decay = float(getattr(self.config, "fatigue_decay_slope", 0.05))
        fatigue_floor = float(getattr(self.config, "fatigue_efficiency_floor", 0.60))
        idle_time = torch.clamp(flat_current_times_w - flat_last, min=0.0)
        has_last = (flat_last > 0) & (flat_current_times_w > flat_last)
        cum_work = flat_cum.clone()
        cum_work[has_last] = torch.clamp(
            cum_work[has_last] - idle_time[has_last] * fatigue_recovery,
            min=0.0,
        )
        overtime = torch.clamp(cum_work - fatigue_threshold, min=0.0)
        fatigue_f = torch.clamp(
            1.0 - fatigue_decay * overtime / (fatigue_threshold * 2),
            min=fatigue_floor,
        )
        template["worker"].x[:, worker_layout.fatigue_idx] = fatigue_f

        # 6. Station 动态特征覆写
        template["station"].x[:, 0] = flat_loads / max(1.0, ideal_station_load)
        loads_tensor = flat_loads.reshape(group_size, num_stations)
        sum_loads = loads_tensor.sum(dim=1, keepdim=True)
        max_load = loads_tensor.max(dim=1, keepdim=True)[0]
        template["station"].x[:, 5] = (loads_tensor / (sum_loads + 1.0e-6)).reshape(-1)
        template["station"].x[:, 6] = (loads_tensor / (max_load + 1.0e-6)).reshape(-1)
        template["station"].x[:, 7] = flat_slots / max_slots
        template["station"].x[:, 4] = torch.log1p(
            flat_station_wait / mean_task_time
        )

        # 6.1 站位宏观特征（逐图 worker 切片；与 _fill_station_macro_features 一致）
        is_free_flat = flat_free_time <= flat_current_times_w
        for index in range(group_size):
            w0 = worker_offsets[index]
            w1 = worker_offsets[index + 1]
            locks_i = flat_locks[w0:w1]
            free_i = is_free_flat[w0:w1]
            mobile_ratio = (locks_i == 0).sum().float() / (w1 - w0)
            template["station"].x[
                station_offsets[index] : station_offsets[index + 1], 2
            ] = mobile_ratio
            for s in range(num_stations):
                bound_ratio = (locks_i == s + 1).sum().float() / (w1 - w0)
                free_bound_ratio = (
                    (locks_i == s + 1) & free_i
                ).sum().float() / (w1 - w0)
                station_node = station_offsets[index] + s
                template["station"].x[station_node, 1] = bound_ratio
                template["station"].x[station_node, 3] = free_bound_ratio

        # 7. 动态边覆写
        if ts_src:
            assigned = torch.tensor([ts_src, ts_dst], dtype=torch.long, device=t)
            template["task", "assigned_to", "station"].edge_index = assigned
            template["station", "has_task", "task"].edge_index = torch.stack(
                (assigned[1], assigned[0]), dim=0
            )
        else:
            template["task", "assigned_to", "station"].edge_index = torch.empty(
                (2, 0), dtype=torch.long, device=t
            )
            template["station", "has_task", "task"].edge_index = torch.empty(
                (2, 0), dtype=torch.long, device=t
            )
        if tw_src:
            template["task", "done_by", "worker"].edge_index = torch.tensor(
                [tw_src, tw_dst], dtype=torch.long, device=t
            )
        else:
            template["task", "done_by", "worker"].edge_index = torch.empty(
                (2, 0), dtype=torch.long, device=t
            )

        # 8. 资源关系边（can_do / skill hub），逐图 worker 偏移
        worker_skill_edges: list[torch.Tensor] = []
        for snap, topo_key in zip(snapshots, topo_keys):
            cache_key = (int(dataset_idx), topo_key)
            edges = self._worker_skill_topologies.get(cache_key)
            if edges is None:
                edges = build_worker_skill_edges(
                    torch.as_tensor(snap["base_worker_x"]),
                    num_skills,
                )
                self._worker_skill_topologies[cache_key] = edges
            worker_skill_edges.append(edges)

        if use_hub:
            apply_batched_resource_graph_offsets(
                template,
                template["task"].x,
                template["worker"].x,
                task_offsets=task_offsets,
                worker_offsets=worker_offsets,
                config=self.config,
                task_skill_edges=ctx.get("task_skill_edge_index"),
                worker_skill_edges=worker_skill_edges,
            )
        else:
            apply_batched_resource_graph_offsets(
                template,
                template["task"].x,
                template["worker"].x,
                task_offsets=task_offsets,
                worker_offsets=worker_offsets,
                config=self.config,
            )

        self._assert_layout_integrity(template)
        return template

    def _assert_layout_integrity(self, template: Any) -> None:
        """严格断言各节点类型 batch 向量与布局一致（非数值正确性替代品）。"""
        layout = template._fast_exact_layout
        for node_type, expected_total in (
            ("task", layout["task_ptr"][-1]),
            ("worker", layout["worker_ptr"][-1]),
            ("station", layout["station_ptr"][-1]),
        ):
            storage = template[node_type]
            total_nodes = int(storage.x.size(0))
            if total_nodes != expected_total:
                raise ValueError(
                    f"{node_type}.x 节点数与模板布局不一致: "
                    f"{total_nodes} != {expected_total}"
                )
            if (
                not hasattr(storage, "batch")
                or storage.batch is None
                or int(storage.batch.numel()) != total_nodes
            ):
                raise ValueError(
                    f"{node_type} 缺少与节点数一致的 batch 向量: "
                    f"batch={None if not hasattr(storage, 'batch') else int(storage.batch.numel())}, "
                    f"nodes={total_nodes}"
                )
        if "skill" in template.node_types and "skill_ptr" in layout:
            skill_total = int(template["skill"].x.size(0))
            if skill_total != layout["skill_ptr"][-1]:
                raise ValueError(
                    f"skill.x 节点数与模板布局不一致: {skill_total} != {layout['skill_ptr'][-1]}"
                )

    def _load_dataset_context(self, dataset_idx: int) -> dict:
        """获取数据集上下文；EnvProxy 在主进程按路径懒加载。"""
        pool = getattr(self.env, "dataset_pool", None)
        if isinstance(pool, list):
            while len(pool) <= dataset_idx:
                pool.append(None)
            ctx = pool[dataset_idx]
            if ctx is None or "base_data" not in ctx:
                load_locally = getattr(
                    self.env, "_load_dataset_context_locally", None
                )
                if not callable(load_locally):
                    raise RuntimeError(
                        "无法初始化数据集上下文 "
                        f"dataset_idx={dataset_idx}: env 无本地加载通道"
                    )
                ctx = load_locally(dataset_idx)
            return ctx
        if isinstance(pool, dict):
            ctx = pool.get(dataset_idx)
            if ctx is None or "base_data" not in ctx:
                raise RuntimeError(
                    f"数据集上下文缺失 dataset_idx={dataset_idx}"
                )
            return ctx
        raise RuntimeError(f"不支持的 dataset_pool 类型: {type(pool)}")

    def build(
        self,
        snapshots: Sequence[EnvironmentSnapshot],
        masks: ActionMasks = None,
        memory_indices: Sequence[int] | None = None,
        *,
        group_id: object | None = None,
    ) -> V2FastExactBatch:
        dataset_idx = self._validate_snapshots(snapshots)
        group_size = len(snapshots)
        ctx = self._load_dataset_context(dataset_idx)
        num_tasks = int(ctx["num_tasks"])
        num_stations = int(ctx["base_station_x"].shape[0])
        task_counts = (num_tasks,) * group_size
        station_counts = (num_stations,) * group_size
        worker_counts = tuple(
            int(len(snap["worker_free_time"])) for snap in snapshots
        )
        topo_keys = tuple(
            snap.get("worker_topology_key")
            or worker_topology_key(
                torch.as_tensor(snap["base_worker_x"]),
                int(getattr(self.config, "num_skill_types", 5)),
            )
            for snap in snapshots
        )
        task_offsets = _cumulative_offsets(task_counts)
        worker_offsets = _cumulative_offsets(worker_counts)
        station_offsets = _cumulative_offsets(station_counts)

        key = self._template_key(
            dataset_idx=dataset_idx,
            task_counts=task_counts,
            worker_counts=worker_counts,
            station_counts=station_counts,
            topo_keys=topo_keys,
        )
        template = self._get_or_create_template(
            key=key,
            ctx=ctx,
            task_counts=task_counts,
            worker_counts=worker_counts,
            station_counts=station_counts,
            dataset_idx=dataset_idx,
        )
        self._rebuild_in_place(
            template,
            ctx=ctx,
            snapshots=snapshots,
            task_offsets=task_offsets,
            worker_offsets=worker_offsets,
            station_offsets=station_offsets,
            dataset_idx=dataset_idx,
            topo_keys=topo_keys,
        )

        layout = template._fast_exact_layout
        if memory_indices is None:
            memory_indices = tuple(range(group_size))
        resolved_indices = tuple(int(index) for index in memory_indices)
        if len(resolved_indices) != group_size:
            raise ValueError(
                "memory_indices 数量与 snapshot 组大小不一致: "
                f"{len(resolved_indices)} != {group_size}"
            )

        task_mask = station_mask = worker_mask = None
        if masks is not None:
            if len(masks) != group_size:
                raise ValueError(
                    f"masks 数量与 snapshot 组大小不一致: {len(masks)} != {group_size}"
                )
            task_mask = torch.cat([mask[0] for mask in masks], dim=0)
            station_mask = torch.cat([mask[1] for mask in masks], dim=0)
            worker_mask = torch.cat([mask[2] for mask in masks], dim=0)

        if task_mask is not None:
            # 节点级 mask 绑定到 Batch，供同形重放逐节点读取。
            template.y_task_mask = task_mask.to(self.device)
            template.y_station_mask = station_mask.to(self.device)
            template.y_worker_mask = worker_mask.to(self.device)

        return V2FastExactBatch(
            batch=template,
            task_ptr=layout["task_ptr"],
            worker_ptr=layout["worker_ptr"],
            station_ptr=layout["station_ptr"],
            task_counts=layout["task_counts"],
            worker_counts=layout["worker_counts"],
            station_counts=layout["station_counts"],
            memory_indices=resolved_indices,
            group_positions=tuple(range(group_size)),
            group_id=group_id,
            raw_task_slices=[
                template["task"].x[
                    task_offsets[index] : task_offsets[index + 1]
                ]
                for index in range(group_size)
            ],
            raw_worker_slices=[
                template["worker"].x[
                    worker_offsets[index] : worker_offsets[index + 1]
                ]
                for index in range(group_size)
            ],
            task_mask=task_mask,
            station_mask=station_mask,
            worker_mask=worker_mask,
            skill_ptr=layout.get("skill_ptr"),
            skill_counts=(
                tuple(
                    layout["skill_ptr"][index + 1] - layout["skill_ptr"][index]
                    for index in range(group_size)
                )
                if "skill_ptr" in layout
                else None
            ),
        )


__all__ = [
    "EnvironmentSnapshot",
    "ActionMasks",
    "V2FastExactBatch",
    "V2FastExactBatchBuilder",
    "CPUExactBatchBuilder",
    "GPUExactBatchBuilder",
    "apply_batched_resource_graph_offsets",
    "_extract_ptr",
]
