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

    def clear(self) -> None:
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.is_truncated[:]
        del self.masks[:]
        del self.values[:]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

__all__ = ["Memory"]
