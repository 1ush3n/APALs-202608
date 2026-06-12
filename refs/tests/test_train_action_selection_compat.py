# -*- coding: utf-8 -*-
"""训练 rollout 动作选择兼容层测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ppo_agent import PPOAgent
from train import select_actions_batch_compat


class _LegacyAgent:
    def __init__(self) -> None:
        self.calls = 0

    def select_action(
        self,
        obs,
        *,
        mask_task,
        mask_station_matrix,
        mask_worker,
        deterministic,
        temperature,
        is_eval,
    ):
        self.calls += 1
        assert deterministic is False
        assert temperature == 1.0
        assert is_eval is False
        return (0, 0, [0]), -0.1, 0.2, mask_station_matrix, False


class _BatchAgent:
    def __init__(self) -> None:
        self.calls = 0

    def select_actions_batch(self, **kwargs):
        self.calls += 1
        return [("batch", len(kwargs["obs_list"]))]


def test_ppo_agent_exposes_batch_action_api() -> None:
    assert hasattr(PPOAgent, "select_actions_batch")
    assert callable(getattr(PPOAgent, "select_actions_batch"))


def test_select_actions_batch_compat_uses_fast_api_when_available() -> None:
    agent = _BatchAgent()
    result = select_actions_batch_compat(
        agent,
        obs_list=[object(), object()],
        mask_task_list=[None, None],
        mask_station_matrix_list=[None, None],
        mask_worker_list=[None, None],
        deterministic=False,
        temperature=1.0,
        is_eval=False,
    )

    assert result == [("batch", 2)]
    assert agent.calls == 1


def test_select_actions_batch_compat_falls_back_to_single_env_api() -> None:
    agent = _LegacyAgent()
    station_mask = torch.zeros((1, 1), dtype=torch.bool)
    result = select_actions_batch_compat(
        agent,
        obs_list=[object(), object()],
        mask_task_list=[torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)],
        mask_station_matrix_list=[station_mask, station_mask],
        mask_worker_list=[torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)],
        deterministic=False,
        temperature=1.0,
        is_eval=False,
    )

    assert len(result) == 2
    assert agent.calls == 2
    assert result[0][0] == (0, 0, [0])
    assert result[0][-1] is False
