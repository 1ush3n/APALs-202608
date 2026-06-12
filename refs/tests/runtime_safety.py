# -*- coding: utf-8 -*-
"""
低显存验证护栏工具。

本模块只服务测试：统一锁种子、读取 CUDA 状态、断言显存阈值，并在测试结束后清理缓存。
"""

from __future__ import annotations

import gc
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, Mapping

import numpy as np
import torch


DEFAULT_ALLOCATED_LIMIT_BYTES = int(2.5 * 1024**3)
DEFAULT_RESERVED_LIMIT_BYTES = int(3.5 * 1024**3)


@dataclass(frozen=True)
class CudaMemorySnapshot:
    """CUDA 显存快照，单位为字节。"""

    allocated: int
    reserved: int
    max_allocated: int
    max_reserved: int
    total: int


def seed_everything(seed: int = 42) -> None:
    """统一锁定测试随机种子，保证 APAL 小规模验证可复现。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cuda_info() -> Dict[str, object]:
    """返回当前 CUDA 运行时信息；无 CUDA 时返回 available=False。"""

    if not torch.cuda.is_available():
        return {
            "available": False,
            "device_name": None,
            "cuda_version": torch.version.cuda,
            "device_count": 0,
            "total_bytes": 0,
        }

    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "device_name": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "total_bytes": props.total_memory,
    }


def cuda_memory_snapshot() -> CudaMemorySnapshot:
    """读取当前设备显存状态；无 CUDA 时返回全零。"""

    if not torch.cuda.is_available():
        return CudaMemorySnapshot(0, 0, 0, 0, 0)

    torch.cuda.synchronize()
    props = torch.cuda.get_device_properties(0)
    return CudaMemorySnapshot(
        allocated=torch.cuda.memory_allocated(),
        reserved=torch.cuda.memory_reserved(),
        max_allocated=torch.cuda.max_memory_allocated(),
        max_reserved=torch.cuda.max_memory_reserved(),
        total=props.total_memory,
    )


def cleanup_cuda() -> None:
    """测试结束后释放 Python 与 CUDA 缓存，降低连续测试的 OOM 风险。"""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def assert_cuda_memory_below(
    allocated_limit: int = DEFAULT_ALLOCATED_LIMIT_BYTES,
    reserved_limit: int = DEFAULT_RESERVED_LIMIT_BYTES,
) -> CudaMemorySnapshot:
    """断言单测峰值显存低于低显存护栏阈值。"""

    snap = cuda_memory_snapshot()
    assert snap.max_allocated <= allocated_limit, (
        f"CUDA 峰值已分配显存过高: {snap.max_allocated / 1024**3:.2f}GB > "
        f"{allocated_limit / 1024**3:.2f}GB"
    )
    assert snap.max_reserved <= reserved_limit, (
        f"CUDA 峰值保留显存过高: {snap.max_reserved / 1024**3:.2f}GB > "
        f"{reserved_limit / 1024**3:.2f}GB"
    )
    return snap


@contextmanager
def guarded_cuda_test(seed: int = 42) -> Iterator[None]:
    """低显存 CUDA 测试上下文：锁种子、清峰值、结束后断言并清理。"""

    seed_everything(seed)
    cleanup_cuda()
    try:
        yield
        assert_cuda_memory_below()
    finally:
        cleanup_cuda()


@contextmanager
def temporary_config(config: object, overrides: Mapping[str, object]) -> Iterator[None]:
    """临时覆写全局配置，退出后恢复原值。"""

    old_values = {key: getattr(config, key) for key in overrides}
    try:
        for key, value in overrides.items():
            setattr(config, key, value)
        yield
    finally:
        for key, value in old_values.items():
            setattr(config, key, value)
