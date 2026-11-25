"""将完整 ScheduleFree checkpoint 导出为可独立部署的 eval_x 副本。"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.optim import Optimizer

from runtime.schedulefree_checkpoint import (
    save_checkpoint_with_schedulefree_eval_parameters,
    schedulefree_parameter_mode,
)


def _policy_state_for_lightning(policy: torch.nn.Module) -> dict[str, torch.Tensor]:
    """将策略权重转换为当前 Lightning checkpoint 的 policy.* 键空间。"""
    return {
        f"policy.{name}": value.detach().cpu().clone()
        for name, value in policy.state_dict().items()
    }


def export_schedulefree_eval_payload(
    *,
    payload: Mapping[str, Any],
    policy: torch.nn.Module,
    optimizer: Optimizer,
) -> dict[str, Any]:
    """基于完整 optimizer state 重建 x 参数，并返回不修改源 payload 的副本。"""
    if schedulefree_parameter_mode(optimizer, schedulefree_enabled=True) != "train_y":
        raise ValueError("仅允许从 train_y ScheduleFree checkpoint 导出 eval_x 副本")
    if not any("z" in state for state in optimizer.state.values()):
        raise ValueError("ScheduleFree optimizer 缺少 z 状态，无法可靠重建 eval_x 参数")
    source_state = payload.get("state_dict")
    if not isinstance(source_state, Mapping) or not source_state:
        raise ValueError("checkpoint 缺少 state_dict")
    if any(not str(name).startswith("policy.") for name in source_state):
        raise ValueError("checkpoint state_dict 不完全属于 policy.*，拒绝不安全导出")
    source_optimizer_states = payload.get("optimizer_states")
    if not isinstance(source_optimizer_states, list) or not source_optimizer_states:
        raise ValueError("checkpoint 缺少 optimizer_states")

    exported = copy.deepcopy(dict(payload))

    def _capture(_path: Path) -> None:
        exported["state_dict"] = _policy_state_for_lightning(policy)
        optimizer_states = copy.deepcopy(source_optimizer_states)
        optimizer_states[0] = copy.deepcopy(optimizer.state_dict())
        exported["optimizer_states"] = optimizer_states
        metadata = exported.get("apal_metadata")
        if not isinstance(metadata, dict):
            raise ValueError("checkpoint 缺少可写入的 apal_metadata")
        metadata["schedulefree_parameter_state"] = "eval_x"

    save_state = save_checkpoint_with_schedulefree_eval_parameters(
        save_checkpoint=_capture,
        path=Path("schedulefree_eval_export.ckpt"),
        optimizer=optimizer,
        schedulefree_enabled=True,
    )
    if save_state.saved_mode != "eval_x":
        raise RuntimeError("ScheduleFree 导出未捕获 eval_x 参数")
    return exported


__all__ = ["export_schedulefree_eval_payload"]
