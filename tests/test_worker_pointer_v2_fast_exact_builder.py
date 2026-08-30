# -*- coding: utf-8 -*-
"""WorkerPointer v2 Fast-Exact 阶段一：V2FastExactBatch 与 builder 接口测试。

- 统一 builder 接口 V2FastExactBatchBuilder（Protocol）；
- CPUExactBatchBuilder 参考实现能构建 V2FastExactBatch；
- 支持组内异质 worker 数（每图 ptr/offset，而非固定第一张图偏移）；
- 布局与单 snapshot CPU rebuild 完全等价（节点特征与边类型）；
- 轨迹元数据（memory index / group position）保持原顺序。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from configs import Config
from environment import AirLineEnv_Graph
from models.worker_pointer_context import PHYSICAL_PREDECESSOR_EDGE
from tests.runtime_safety import temporary_config
from training.v2_fast_exact_batch import (
    CPUExactBatchBuilder,
    GPUExactBatchBuilder,
    V2FastExactBatch,
    V2FastExactBatchBuilder,
)
from configs import configs as global_configs


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "283.csv"


def _make_env(seed: int = 42) -> AirLineEnv_Graph:
    env = AirLineEnv_Graph(str(DATA_PATH), seed=seed)
    env.reset(seed=seed)
    return env


def _truncate_worker(snapshot: dict, num_workers: int) -> dict:
    """截取快照中的工人相关字段，构造不同工人数的异质快照。"""
    result = dict(snapshot)

    def _slice_worker_field(value: object) -> object:
        if torch.is_tensor(value):
            return value[:num_workers].clone()
        return value[:num_workers].copy()

    result["base_worker_x"] = _slice_worker_field(snapshot["base_worker_x"])
    result["worker_free_time"] = _slice_worker_field(snapshot["worker_free_time"])
    result["worker_locks"] = _slice_worker_field(snapshot["worker_locks"])
    result["worker_cumulative_work"] = _slice_worker_field(
        snapshot["worker_cumulative_work"]
    )
    result["worker_last_busy_end"] = _slice_worker_field(
        snapshot["worker_last_busy_end"]
    )
    # 原快照的 worker_topology_key 按全量工人数生成；截断后必须移除，
    # 让 rebuild 按新工人数重新计算拓扑（worker_topology_key 本身含工人数）。
    result.pop("worker_topology_key", None)
    return result


def _cpu_builder(env: AirLineEnv_Graph) -> CPUExactBatchBuilder:
    return CPUExactBatchBuilder(
        config=Config(),
        env=env,
        device=torch.device("cpu"),
    )


def test_cpu_builder_is_v2_fast_exact_builder_protocol() -> None:
    with temporary_config(global_configs, {}):
        env = _make_env()
        builder = _cpu_builder(env)
        assert isinstance(builder, V2FastExactBatchBuilder)


def test_cpu_builder_builds_layout_with_heterogeneous_worker_counts() -> None:
    with temporary_config(global_configs, {}):
        env = _make_env()
        snapshot = env.get_state_snapshot()
        num_tasks = int(env.num_tasks)
        num_stations = int(snapshot["station_loads"].shape[0])
        snap1 = _truncate_worker(snapshot, 60)
        snap2 = _truncate_worker(snapshot, 72)

        out = _cpu_builder(env).build(
            [snap1, snap2],
            masks=None,
            memory_indices=[10, 11],
        )

        assert isinstance(out, V2FastExactBatch)
        assert out.worker_counts == (60, 72)
        assert out.worker_ptr == (0, 60, 132)
        assert out.task_counts == (num_tasks, num_tasks)
        assert out.task_ptr == (0, num_tasks, 2 * num_tasks)
        assert out.station_counts == (num_stations, num_stations)
        assert out.batch["worker"].x.size(0) == 132
        assert out.batch["worker"].batch.tolist() == [0] * 60 + [1] * 72
        assert out.batch["task"].batch.tolist() == [0] * num_tasks + [1] * num_tasks
        assert out.memory_indices == (10, 11)
        assert out.group_positions == (0, 1)


def test_cpu_builder_matches_single_snapshot_rebuild() -> None:
    with temporary_config(global_configs, {}):
        env = _make_env()
        snapshot = env.get_state_snapshot()

        out = _cpu_builder(env).build(
            [snapshot],
            masks=None,
            memory_indices=[0],
        )
        rebuilt = env.rebuild_state_from_snapshot(snapshot)

        torch.testing.assert_close(out.batch["task"].x, rebuilt["task"].x)
        torch.testing.assert_close(out.batch["worker"].x, rebuilt["worker"].x)
        torch.testing.assert_close(out.batch["station"].x, rebuilt["station"].x)
        assert set(out.batch.edge_types) == set(rebuilt.edge_types)


def test_cpu_builder_exposes_per_graph_raw_feature_slices() -> None:
    with temporary_config(global_configs, {}):
        env = _make_env()
        snapshot = env.get_state_snapshot()
        num_tasks = int(env.num_tasks)

        out = _cpu_builder(env).build(
            [_truncate_worker(snapshot, 64), _truncate_worker(snapshot, 68)],
            masks=None,
            memory_indices=[0, 1],
        )

        assert len(out.raw_task_slices) == 2
        assert len(out.raw_worker_slices) == 2
        assert out.raw_task_slices[0].shape == (num_tasks, 18)
        assert out.raw_worker_slices[0].shape == (64, 17)
        assert out.raw_worker_slices[1].shape == (68, 17)


def _gpu_builder(env: AirLineEnv_Graph) -> GPUExactBatchBuilder:
    return GPUExactBatchBuilder(
        config=Config(),
        env=env,
        device=torch.device("cpu"),
    )


def test_gpu_builder_matches_cpu_rebuild_homogeneous() -> None:
    with temporary_config(global_configs, {}):
        env = _make_env()
        snapshot = env.get_state_snapshot()

        gpu_out = _gpu_builder(env).build([snapshot], masks=None, memory_indices=[0])
        cpu_out = _cpu_builder(env).build([snapshot], masks=None, memory_indices=[0])

        torch.testing.assert_close(
            gpu_out.batch["task"].x, cpu_out.batch["task"].x, atol=1.0e-6, rtol=0.0
        )
        torch.testing.assert_close(
            gpu_out.batch["worker"].x,
            cpu_out.batch["worker"].x,
            atol=1.0e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            gpu_out.batch["station"].x,
            cpu_out.batch["station"].x,
            atol=1.0e-6,
            rtol=0.0,
        )
        assert set(gpu_out.batch.edge_types) == set(cpu_out.batch.edge_types)


def test_gpu_builder_preserves_physical_predecessor_offsets() -> None:
    with temporary_config(global_configs, {}):
        env = _make_env()
        snapshot = env.get_state_snapshot()
        num_tasks = int(env.num_tasks)
        base_edges = env.base_data[PHYSICAL_PREDECESSOR_EDGE].edge_index

        out = _gpu_builder(env).build(
            [snapshot, snapshot], masks=None, memory_indices=[0, 1]
        )
        batched_edges = out.batch[PHYSICAL_PREDECESSOR_EDGE].edge_index
        edge_count = int(base_edges.size(1))

        assert edge_count > 0
        torch.testing.assert_close(
            batched_edges[:, :edge_count], base_edges, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            batched_edges[:, edge_count:], base_edges + num_tasks,
            atol=0.0,
            rtol=0.0,
        )


def test_gpu_builder_matches_cpu_rebuild_heterogeneous_workers() -> None:
    with temporary_config(global_configs, {}):
        env = _make_env()
        snapshot = env.get_state_snapshot()
        snap1 = _truncate_worker(snapshot, 60)
        snap2 = _truncate_worker(snapshot, 72)

        gpu_out = _gpu_builder(env).build(
            [snap1, snap2], masks=None, memory_indices=[10, 11]
        )
        cpu_out = _cpu_builder(env).build(
            [snap1, snap2], masks=None, memory_indices=[10, 11]
        )

        assert gpu_out.worker_counts == cpu_out.worker_counts == (60, 72)
        assert gpu_out.worker_ptr == cpu_out.worker_ptr
        assert gpu_out.task_ptr == cpu_out.task_ptr
        torch.testing.assert_close(
            gpu_out.batch["worker"].x, cpu_out.batch["worker"].x, atol=1.0e-6, rtol=0.0
        )
        torch.testing.assert_close(
            gpu_out.batch["task"].x, cpu_out.batch["task"].x, atol=1.0e-6, rtol=0.0
        )
        torch.testing.assert_close(
            gpu_out.batch["station"].x,
            cpu_out.batch["station"].x,
            atol=1.0e-6,
            rtol=0.0,
        )
        assert set(gpu_out.batch.edge_types) == set(cpu_out.batch.edge_types)


def test_gpu_builder_returns_layout_and_trajectory_metadata() -> None:
    with temporary_config(global_configs, {}):
        env = _make_env()
        snapshot = env.get_state_snapshot()
        num_tasks = int(env.num_tasks)

        out = _gpu_builder(env).build(
            [_truncate_worker(snapshot, 64), _truncate_worker(snapshot, 68)],
            masks=None,
            memory_indices=[5, 6],
            group_id=("ep", 7),
        )

        assert isinstance(out, V2FastExactBatch)
        assert out.group_id == ("ep", 7)
        assert out.memory_indices == (5, 6)
        assert out.group_positions == (0, 1)
        assert out.task_ptr == (0, num_tasks, 2 * num_tasks)
        assert len(out.raw_worker_slices) == 2
        assert out.raw_worker_slices[0].shape == (64, 17)
        assert out.raw_worker_slices[1].shape == (68, 17)
