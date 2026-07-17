"""APAL 工人节点特征布局的唯一来源。

当前项目只使用五类工种。工人特征固定为：
``[效率 | 5 个技能 | 等待时间 | 空闲标记 | 8 个工位锁状态 | 疲劳因子]``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


LOCK_STATE_DIM = 8
"""工位锁状态 one-hot 的固定宽度：未绑定 + 7 个可表示工位状态。"""


@dataclass(frozen=True)
class WorkerFeatureLayout:
    """工人节点特征的显式索引与切片。"""

    num_skill_types: int
    skill_slots: int
    efficiency_idx: int
    skill_start: int
    wait_idx: int
    free_idx: int
    lock_start: int
    lock_end: int
    fatigue_idx: int
    total_dim: int

    @property
    def skill_slice(self) -> slice:
        """返回有效工种 one-hot 的切片。"""

        return slice(self.skill_start, self.wait_idx)

    @property
    def lock_slice(self) -> slice:
        """返回工位锁状态 one-hot 的切片。"""

        return slice(self.lock_start, self.lock_end)


def build_worker_feature_layout(
    num_skill_types: int,
    worker_skill_feature_slots: int | None = None,
) -> WorkerFeatureLayout:
    """根据当前工种数生成唯一合法的工人特征布局。"""

    skill_types = int(num_skill_types)
    skill_slots = skill_types if worker_skill_feature_slots is None else int(worker_skill_feature_slots)
    if skill_types != 5:
        raise ValueError(f"当前 APAL 实验仅支持五类工种，收到 num_skill_types={skill_types}")
    if skill_slots != skill_types:
        raise ValueError(
            "worker_skill_feature_slots 必须与 num_skill_types 相等；"
            "当前五技能口径不再保留旧的十槽位填充。"
        )

    efficiency_idx = 0
    skill_start = efficiency_idx + 1
    wait_idx = skill_start + skill_slots
    free_idx = wait_idx + 1
    lock_start = free_idx + 1
    lock_end = lock_start + LOCK_STATE_DIM
    fatigue_idx = lock_end
    return WorkerFeatureLayout(
        num_skill_types=skill_types,
        skill_slots=skill_slots,
        efficiency_idx=efficiency_idx,
        skill_start=skill_start,
        wait_idx=wait_idx,
        free_idx=free_idx,
        lock_start=lock_start,
        lock_end=lock_end,
        fatigue_idx=fatigue_idx,
        total_dim=fatigue_idx + 1,
    )


def resolve_worker_feature_layout(config: Any) -> WorkerFeatureLayout:
    """由配置对象解析布局，并拒绝与特征维度不一致的配置。"""

    layout = build_worker_feature_layout(
        int(getattr(config, "num_skill_types")),
        getattr(config, "worker_skill_feature_slots", None),
    )
    configured_dim = int(getattr(config, "worker_feat_dim", layout.total_dim))
    if configured_dim != layout.total_dim:
        raise ValueError(
            f"worker_feat_dim={configured_dim} 与五技能工人特征布局不一致；"
            f"应为 {layout.total_dim}。"
        )
    return layout
