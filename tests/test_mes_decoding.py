# -*- coding: utf-8 -*-
"""
单元测试：站位最小前瞻贪心解码 (MES Decoding: Minimum Earliest Station)
"""

from __future__ import annotations

import pytest
import torch

from core.mes_decoding import apply_mes_station_logits


def test_mes_soft_penalty_decays_higher_stations() -> None:
    # 假设 5 个站位，Station 0 非法，合法站位为 1, 2, 3, 4 (s_min = 1)
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    invalid = torch.tensor([[True, False, False, False, False]], dtype=torch.bool)

    # 原始 argmax 会选 Station 4 (logit=4.0)
    assert torch.argmax(logits.masked_fill(invalid, -1.0e4), dim=-1).item() == 4

    # 启用 MES soft_penalty，lambda = 2.0
    # penalty: s=1: 0; s=2: 2; s=3: 4; s=4: 6
    # 调整后 logits:
    # s=1: 1.0 - 0.0 = 1.0
    # s=2: 2.0 - 2.0 = 0.0
    # s=3: 3.0 - 4.0 = -1.0
    # s=4: 4.0 - 6.0 = -2.0
    adj_logits, adj_invalid = apply_mes_station_logits(
        logits,
        invalid,
        penalty_lambda=2.0,
        mode="soft_penalty",
    )

    assert adj_invalid is not None
    assert torch.equal(adj_invalid, invalid)  # soft_penalty 模式不改变 invalid 掩码
    assert torch.argmax(adj_logits, dim=-1).item() == 1  # 成功引导至 s_min (Station 1)
    assert pytest.approx(adj_logits[0, 1].item(), abs=1e-4) == 1.0
    assert pytest.approx(adj_logits[0, 2].item(), abs=1e-4) == 0.0
    assert pytest.approx(adj_logits[0, 3].item(), abs=1e-4) == -1.0
    assert pytest.approx(adj_logits[0, 4].item(), abs=1e-4) == -2.0
    assert adj_logits[0, 0].item() <= -1.0e4


def test_mes_hard_bound_masks_distant_stations() -> None:
    # s_min = 1, max_station_jump = 1 -> 仅允许 Station 1 和 Station 2
    logits = torch.tensor([[0.0, 1.0, 2.0, 5.0, 6.0]], dtype=torch.float32)
    invalid = torch.tensor([[True, False, False, False, False]], dtype=torch.bool)

    adj_logits, adj_invalid = apply_mes_station_logits(
        logits,
        invalid,
        max_station_jump=1,
        mode="hard_bound",
    )

    assert adj_invalid is not None
    # Station 0 (原本非法), Station 3, 4 (超出 jump 上限) 必须为 True (invalid)
    assert adj_invalid[0, 0].item() is True
    assert adj_invalid[0, 1].item() is False
    assert adj_invalid[0, 2].item() is False
    assert adj_invalid[0, 3].item() is True
    assert adj_invalid[0, 4].item() is True

    # 原始 argmax 会选 4，截断后只能在 1 和 2 之间选，由于 logits[2] > logits[1]，选中 2
    assert torch.argmax(adj_logits, dim=-1).item() == 2


def test_mes_combined_mode() -> None:
    # combined: 既截断又惩罚
    logits = torch.tensor([[0.0, 1.0, 2.0, 5.0, 6.0]], dtype=torch.float32)
    invalid = torch.tensor([[True, False, False, False, False]], dtype=torch.bool)

    adj_logits, adj_invalid = apply_mes_station_logits(
        logits,
        invalid,
        penalty_lambda=2.0,
        max_station_jump=1,
        mode="combined",
    )

    assert adj_invalid is not None
    assert adj_invalid[0, 3].item() is True
    assert adj_invalid[0, 4].item() is True
    # Station 1: 1.0 - 0 = 1.0; Station 2: 2.0 - 2 = 0.0
    assert torch.argmax(adj_logits, dim=-1).item() == 1


def test_mes_edge_cases() -> None:
    # 情况 1: 全 invalid
    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    invalid = torch.tensor([[True, True, True]], dtype=torch.bool)
    adj_logits, adj_invalid = apply_mes_station_logits(logits, invalid)
    assert torch.equal(adj_invalid, invalid)

    # 情况 2: 仅 1 个合法站位
    invalid2 = torch.tensor([[True, True, False]], dtype=torch.bool)
    adj_logits2, adj_invalid2 = apply_mes_station_logits(
        logits, invalid2, max_station_jump=1, mode="combined"
    )
    assert adj_invalid2[0, 2].item() is False
    assert torch.argmax(adj_logits2, dim=-1).item() == 2

    # 情况 3: 最后一站合法 (s_min = 2)
    assert torch.argmax(adj_logits2, dim=-1).item() == 2

    # 情况 4: 1D 张量支持
    logits_1d = torch.tensor([0.0, 1.0, 5.0], dtype=torch.float32)
    invalid_1d = torch.tensor([False, False, False], dtype=torch.bool)
    adj_1d, inv_1d = apply_mes_station_logits(
        logits_1d, invalid_1d, penalty_lambda=3.0, mode="soft_penalty"
    )
    assert adj_1d.shape == (3,)
    assert torch.argmax(adj_1d).item() == 0


def test_mes_batch_processing() -> None:
    # 两个 batch 元素，各自有不同的 s_min
    logits = torch.tensor(
        [
            [5.0, 5.0, 5.0, 5.0, 5.0],  # batch 0: s_min = 0
            [0.0, 0.0, 0.0, 5.0, 5.0],  # batch 1: s_min = 3
        ],
        dtype=torch.float32,
    )
    invalid = torch.tensor(
        [
            [False, False, False, False, False],
            [True, True, True, False, False],
        ],
        dtype=torch.bool,
    )

    adj_logits, adj_invalid = apply_mes_station_logits(
        logits,
        invalid,
        penalty_lambda=2.0,
        max_station_jump=1,
        mode="combined",
    )

    # Batch 0: s_min = 0, jump=1 -> Station 0, 1 合法; penalty 让 0 胜出 (5.0 vs 3.0)
    assert torch.argmax(adj_logits[0]).item() == 0
    assert adj_invalid[0, 2].item() is True
    assert adj_invalid[0, 3].item() is True
    assert adj_invalid[0, 4].item() is True

    # Batch 1: s_min = 3, jump=1 -> Station 3, 4 合法; penalty 让 3 胜出 (5.0 vs 3.0)
    assert torch.argmax(adj_logits[1]).item() == 3
    assert adj_invalid[1, 3].item() is False
    assert adj_invalid[1, 4].item() is False


def test_mes_zero_regression_when_disabled() -> None:
    from configs import Config
    from ppo_agent import PPOAgent

    cfg = Config()
    assert cfg.enable_mes_decoding is False
    assert cfg.mes_penalty_lambda == 2.0
    assert cfg.mes_max_station_jump == 1
    assert cfg.mes_mode == "soft_penalty"


def test_mes_shadow_mask_verification_passes_with_mes() -> None:
    from configs import configs
    from environment import AirLineEnv_Graph

    orig_enable = configs.enable_mes_decoding
    orig_jump = configs.mes_max_station_jump
    orig_shadow = configs.enable_shadow_mask_verification
    try:
        configs.enable_mes_decoding = True
        configs.mes_max_station_jump = 1
        configs.enable_shadow_mask_verification = True

        env = AirLineEnv_Graph(data_path_or_dir="data/283.csv", seed=42)
        env.reset()
        for _ in range(5):
            t_mask, s_mask, w_mask = env.get_masks()
            assert t_mask is not None
            assert s_mask is not None
            assert w_mask is not None
    finally:
        configs.enable_mes_decoding = orig_enable
        configs.mes_max_station_jump = orig_jump
        configs.enable_shadow_mask_verification = orig_shadow


