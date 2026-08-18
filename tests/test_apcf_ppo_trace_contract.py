# -*- coding: utf-8 -*-
"""APCF PPO proposal trace 与 rollout 指标契约测试。"""

from dataclasses import replace

import numpy as np
import torch

from configs import configs
from environment import AirLineEnv_Graph
from ppo_agent import FrozenAnchorProposalTrace, PPOAgent
from tests.runtime_safety import seed_everything, temporary_config
from tests.test_apcf_anchor_proposal import (
    DATA_PATH,
    _advance_to_ready_physical_task,
    _make_agent,
)
from training.memory import Memory


def test_sampled_proposal_branch_logprob_is_full_float32_chain() -> None:
    """采样端必须保存完整 proposal chain 的 float32 log-prob。"""
    seed_everything(42)
    agent, overrides = _make_agent()
    env = AirLineEnv_Graph(DATA_PATH, seed=42)
    obs, masks = _advance_to_ready_physical_task(env)
    with temporary_config(configs, overrides):
        action, _logprob, _value, _smask, invalid = agent.select_action(
            obs,
            mask_task=masks[0],
            mask_station_matrix=masks[1],
            mask_worker=masks[2],
            deterministic=True,
            temperature=0.0,
        )
    assert action is not None and not invalid
    encoded, _context = agent.policy(obs)
    task_id, station_id = int(action[0]), int(action[1])
    with temporary_config(configs, overrides):
        sampled = agent._select_anchor_proposal_team(
            agent.policy,
            obs=obs,
            task_id=task_id,
            station_id=station_id,
            worker_mask=masks[2],
            task_emb=encoded["task"][task_id].unsqueeze(0),
            station_emb=encoded["station"][station_id].unsqueeze(0),
            worker_embs=encoded["worker"],
            deterministic=False,
            temperature=1.0,
            branch_floor=agent._current_anchor_branch_floor(),
        )
    assert sampled is not None
    _team, sampled_lp, trace = sampled
    assert trace.proposal_available
    assert np.isclose(
        trace.sampled_proposal_branch_logprob,
        float(sampled_lp.detach().float().item()),
        atol=1.0e-7,
    )

    recomputed, _entropy, diagnostics = agent._recompute_anchor_proposal_logprobs(
        task_embeddings=encoded["task"][task_id].unsqueeze(0),
        station_embeddings=encoded["station"][station_id].unsqueeze(0),
        worker_embeddings=encoded["worker"].unsqueeze(0),
        frozen_traces=[trace],
    )
    assert torch.allclose(sampled_lp.reshape(()), recomputed[0], atol=1.0e-5)
    assert diagnostics["sampled_proposal_branch_logprob_mae"] <= 1.0e-5
    assert diagnostics["sampled_proposal_branch_logprob_max_abs_error"] <= 1.0e-4


def test_rollout_metrics_count_available_proposals_and_reject_duplicates() -> None:
    """proposal 合法率只统计可用 proposal，且拒绝与 anchor 重复的团队。"""
    seed_everything(42)
    agent, overrides = _make_agent()
    env = AirLineEnv_Graph(DATA_PATH, seed=42)
    obs, masks = _advance_to_ready_physical_task(env)
    with temporary_config(configs, overrides):
        _action, _logprob, _value, _smask, invalid = agent.select_action(
            obs,
            mask_task=masks[0],
            mask_station_matrix=masks[1],
            mask_worker=masks[2],
            deterministic=False,
            temperature=1.0,
        )
    assert not invalid
    trace = agent.last_anchor_proposal_trace
    assert isinstance(trace, FrozenAnchorProposalTrace)
    assert trace.proposal_available
    memory = Memory()
    memory.anchor_proposal_traces.append(
        replace(trace, proposal_team=trace.anchor_team)
    )
    metrics = PPOAgent._anchor_proposal_rollout_metrics(memory)
    assert metrics["APCF/RolloutProposalAvailableCount"] == 1.0
    assert metrics["APCF/RolloutValidProposalRate"] == 0.0
