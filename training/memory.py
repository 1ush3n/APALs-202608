from __future__ import annotations

import gc

import torch


class Memory:
    """存储 PPO 更新所需的轨迹数据。"""

    def __init__(self) -> None:
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.is_truncated = []
        self.masks = []
        self.values = []
        # 与每条动作严格对齐；仅门控团队动作保存冻结候选，其他动作为 None。
        self.gated_team_traces = []
        self.anchor_proposal_traces = []
        # WorkerPointer v2 行为轨迹；与动作严格对齐，仅 v2 模式写入，其余为 None。
        self.worker_pointer_v2_behavior_traces = []

    def clear(self) -> None:
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.is_truncated[:]
        del self.masks[:]
        del self.values[:]
        del self.worker_pointer_v2_behavior_traces[:]
        del self.gated_team_traces[:]
        del self.anchor_proposal_traces[:]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

__all__ = ["Memory"]
