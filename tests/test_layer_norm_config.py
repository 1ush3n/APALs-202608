# -*- coding: utf-8 -*-
"""验证输入嵌入层与 GAT 层的 LayerNorm 开关已经解耦。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import configs
from models.hb_gat_pn import HBGATPN
from tests.runtime_safety import temporary_config


def test_input_layer_norm_independent_from_gat_layer_norm() -> None:
    overrides = {
        "use_layer_norm": False,
        "use_input_layer_norm": True,
        "use_gat_layer_norm": False,
        "use_head_layer_norm": False,
    }

    with temporary_config(configs, overrides):
        model = HBGATPN(configs)

    assert isinstance(model.embedder.task_emb[1], nn.LayerNorm)
    assert isinstance(model.embedder.worker_emb[1], nn.LayerNorm)
    assert isinstance(model.embedder.station_emb[1], nn.LayerNorm)

    first_norms = model.encoder.norms[0]
    assert isinstance(first_norms["task"], nn.Identity)
    assert isinstance(first_norms["worker"], nn.Identity)
    assert isinstance(first_norms["station"], nn.Identity)

    assert isinstance(model.actor_station_attn[1], nn.Identity)
    assert isinstance(model.actor_task_worker_attn[1], nn.Identity)
    assert isinstance(model.critic_station_attn[1], nn.Identity)
    assert isinstance(model.critic_task_worker_attn[1], nn.Identity)
