"""
站位最小前瞻贪心解码 (Minimum Earliest Station, MES)

针对航空脉动装配线（APAL）中由于 DAG 强汇聚拓扑与工艺单调性硬约束耦合导致的
后序工位（如 Station 5）早熟挤压问题，在推理与决策解码阶段引入站位非必要不越级引导。
"""

from __future__ import annotations

import torch


def apply_mes_station_logits(
    station_logits: torch.Tensor,
    station_invalid: torch.Tensor | None = None,
    *,
    penalty_lambda: float = 2.0,
    max_station_jump: int | None = 1,
    mode: str = "soft_penalty",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """对 Station Logits 应用站位最小前瞻引导。

    Args:
        station_logits: [B, S] 或 [S] 的 Float Tensor，表示各站位的未归一化对数概率。
        station_invalid: [B, S] 或 [S] 的 Bool Tensor，True 表示站位非法，False 表示合法。
        penalty_lambda: 衰减惩罚系数 lambda >= 0。越大越倾向于选择紧邻前驱的最早站位。
        max_station_jump: 允许的最大跃迁站位步长 (如 1 表示最多允许从 s_min 跳到 s_min + 1)。
                          若为 None 则不启用硬截断。
        mode: 解码模式：
              - "soft_penalty": 仅施加 Logits 线性衰减惩罚 Logits'(s) = Logits(s) - lambda * (s - s_min)
              - "hard_bound": 仅将 s > s_min + max_station_jump 的站位设为 invalid
              - "combined": 既硬截断又施加软惩罚

    Returns:
        adjusted_logits: 调整后的 Logits 张量 (与原输入同 device, dtype, shape)。
        updated_invalid: 更新后的 station_invalid 掩码 (若原输入不为 None)。
    """
    if station_logits is None:
        return station_logits, station_invalid

    original_dim = station_logits.dim()
    if original_dim == 1:
        station_logits_2d = station_logits.unsqueeze(0)
        station_invalid_2d = (
            station_invalid.unsqueeze(0) if station_invalid is not None else None
        )
    else:
        station_logits_2d = station_logits
        station_invalid_2d = station_invalid

    batch_size, num_stations = station_logits_2d.shape
    device = station_logits_2d.device

    # 复制以避免原位污染外部张量
    adjusted_logits = station_logits_2d.clone()
    updated_invalid = (
        station_invalid_2d.clone()
        if station_invalid_2d is not None
        else torch.zeros_like(adjusted_logits, dtype=torch.bool)
    )

    stations_arange = torch.arange(num_stations, device=device, dtype=torch.long)

    for b in range(batch_size):
        curr_invalid = updated_invalid[b]
        legal_mask = ~curr_invalid
        if not legal_mask.any():
            continue

        legal_indices = torch.nonzero(legal_mask, as_tuple=False).squeeze(-1)
        s_min = int(legal_indices[0].item())

        # 1. 硬截断: 若启用且 max_station_jump is not None
        if mode in ("hard_bound", "combined") and max_station_jump is not None:
            max_allowed_s = s_min + max(0, int(max_station_jump))
            jump_mask = stations_arange > max_allowed_s
            updated_invalid[b] = updated_invalid[b] | jump_mask
            # 确保至少 s_min 保留为合法
            updated_invalid[b, s_min] = False

        # 2. 软衰减惩罚: 若启用且 penalty_lambda > 0
        if mode in ("soft_penalty", "combined") and penalty_lambda > 0.0:
            diff = (stations_arange.float() - float(s_min)).clamp(min=0.0)
            penalty = float(penalty_lambda) * diff
            adjusted_logits[b] = adjusted_logits[b] - penalty

        # 3. 重新对所有 invalid 站位填充 -1.0e4，确保不可选
        adjusted_logits[b] = adjusted_logits[b].masked_fill(updated_invalid[b], -1.0e4)

    if original_dim == 1:
        final_logits = adjusted_logits.squeeze(0)
        final_invalid = (
            updated_invalid.squeeze(0) if station_invalid is not None else None
        )
    else:
        final_logits = adjusted_logits
        final_invalid = updated_invalid if station_invalid is not None else None

    return final_logits, final_invalid
