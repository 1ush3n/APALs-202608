# -*- coding: utf-8 -*-
"""APCF PPO optimizer.step 前梯度审计契约。"""

import torch

from ppo_agent import PPOAgent


def test_gradient_diagnostics_distinguish_apcf_trunk_and_nonfinite_gradients() -> None:
    """梯度审计必须识别 APCF head、主干、critic 和非有限梯度。"""
    apcf = torch.nn.Parameter(torch.ones(2))
    trunk = torch.nn.Parameter(torch.ones(2))
    critic = torch.nn.Parameter(torch.ones(2))
    broken = torch.nn.Parameter(torch.ones(2))
    apcf.grad = torch.tensor([1.0, 2.0])
    trunk.grad = torch.tensor([3.0, 4.0])
    critic.grad = torch.tensor([5.0, 6.0])
    broken.grad = torch.tensor([float("nan"), 0.0])

    diagnostics = PPOAgent._collect_gradient_diagnostics(
        [
            ("anchor_team_head.weight", apcf),
            ("encoder.layers.0.weight", trunk),
            ("critic.value.weight", critic),
            ("anchor_proposal_gate.bias", broken),
        ]
    )

    assert diagnostics["finite"] == 0.0
    assert diagnostics["apcf_nonzero"] == 1.0
    assert diagnostics["trunk_nonzero"] == 1.0
    assert diagnostics["critic_grad_norm"] > 0.0
