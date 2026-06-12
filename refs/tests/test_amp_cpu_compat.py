# -*- coding: utf-8 -*-
"""
PPOAgent AMP/GradScaler CPU 兼容性测试。

该测试不做训练，只验证在 CPU 设备上不会硬编码 CUDA autocast 或启用 CUDA GradScaler。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import configs
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from tests.runtime_safety import temporary_config


def test_ppo_agent_amp_helpers_are_cpu_safe() -> None:
    overrides = {
        "hidden_dim": 16,
        "num_gat_layers": 1,
        "num_heads": 1,
        "use_schedule_free": False,
        "use_ema": False,
    }
    with temporary_config(configs, overrides):
        device = torch.device("cpu")
        model = HBGATPN(configs).to(device)
        agent = PPOAgent(
            model=model,
            lr=configs.lr,
            gamma=configs.gamma,
            k_epochs=1,
            eps_clip=configs.eps_clip,
            device=device,
            batch_size=1,
            total_timesteps=1,
        )

        assert agent.amp_device_type == "cpu"
        assert not agent.amp_enabled
        assert not agent.scaler.is_enabled()

        with agent.autocast_context():
            x = torch.randn(2, 2)
            y = x @ x
        assert y.device.type == "cpu"
        assert torch.isfinite(y).all()
        assert agent.get_memory_snapshot() == {"allocated_gb": 0.0, "reserved_gb": 0.0}
