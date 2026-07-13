"""APAL 仿真时间比较的统一数值语义。"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def release_time_tolerance(config_obj: Any) -> float:
    """返回以小时为单位的物料到达时间容差。"""
    tolerance = float(getattr(config_obj, "release_time_tolerance_hours", 1.0e-5))
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("release_time_tolerance_hours 必须是非负有限数")
    return tolerance


def time_reached_scalar(target_time: float, current_time: float, tolerance: float) -> bool:
    """使用 float64 判断目标时刻是否已经到达。"""
    return bool(np.float64(target_time) <= np.float64(current_time) + np.float64(tolerance))


def time_reached_numpy(
    target_times: np.ndarray,
    current_time: float,
    tolerance: float,
) -> np.ndarray:
    """向量化 float64 时间到达判断。"""
    targets = np.asarray(target_times, dtype=np.float64)
    threshold = np.float64(current_time) + np.float64(tolerance)
    return targets <= threshold


def time_reached_tensor(
    target_times: np.ndarray | torch.Tensor,
    current_time: float,
    tolerance: float,
    *,
    device: torch.device,
) -> torch.Tensor:
    """在指定设备上执行 float64 时间到达判断。"""
    targets = torch.as_tensor(target_times, dtype=torch.float64, device=device)
    threshold = torch.tensor(
        np.float64(current_time) + np.float64(tolerance),
        dtype=torch.float64,
        device=device,
    )
    return targets <= threshold


__all__ = [
    "release_time_tolerance",
    "time_reached_numpy",
    "time_reached_scalar",
    "time_reached_tensor",
]
