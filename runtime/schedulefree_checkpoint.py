"""ScheduleFree checkpoint 的参数态保护。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from torch.optim import Optimizer


ScheduleFreeParameterMode = Literal["disabled", "train_y", "eval_x"]


@dataclass(frozen=True)
class ScheduleFreeCheckpointSaveState:
    """一次保存前、中、后的 ScheduleFree 参数态。"""

    source_mode: ScheduleFreeParameterMode
    saved_mode: ScheduleFreeParameterMode
    restored_mode: ScheduleFreeParameterMode


def schedulefree_parameter_mode(
    optimizer: Optimizer,
    *,
    schedulefree_enabled: bool,
) -> ScheduleFreeParameterMode:
    """读取并验证所有参数组共享的 ScheduleFree 参数态。"""
    if not schedulefree_enabled:
        return "disabled"
    modes = [group.get("train_mode") for group in optimizer.param_groups]
    if not modes or any(not isinstance(mode, bool) for mode in modes):
        raise RuntimeError("ScheduleFree optimizer 缺少布尔 train_mode 参数组状态")
    if len(set(modes)) != 1:
        raise RuntimeError("ScheduleFree optimizer 的参数组 train_mode 不一致，拒绝保存 checkpoint")
    return "train_y" if bool(modes[0]) else "eval_x"


def save_checkpoint_with_schedulefree_eval_parameters(
    *,
    save_checkpoint: Callable[[Path], None],
    path: Path,
    optimizer: Optimizer,
    schedulefree_enabled: bool,
) -> ScheduleFreeCheckpointSaveState:
    """以评估平均参数保存 checkpoint，并恢复调用方的原训练态。"""
    target = Path(path)
    source_mode = schedulefree_parameter_mode(
        optimizer,
        schedulefree_enabled=schedulefree_enabled,
    )
    switched_from_train = source_mode == "train_y"
    if switched_from_train:
        optimizer.eval()
    saved_mode = schedulefree_parameter_mode(
        optimizer,
        schedulefree_enabled=schedulefree_enabled,
    )
    if schedulefree_enabled and saved_mode != "eval_x":
        raise RuntimeError("ScheduleFree checkpoint 保存前未进入 eval_x 参数态")
    try:
        save_checkpoint(target)
    finally:
        if switched_from_train:
            optimizer.train()
    restored_mode = schedulefree_parameter_mode(
        optimizer,
        schedulefree_enabled=schedulefree_enabled,
    )
    if restored_mode != source_mode:
        raise RuntimeError(
            "ScheduleFree checkpoint 保存后未恢复原参数态："
            f"before={source_mode}, after={restored_mode}"
        )
    return ScheduleFreeCheckpointSaveState(
        source_mode=source_mode,
        saved_mode=saved_mode,
        restored_mode=restored_mode,
    )


__all__ = [
    "ScheduleFreeCheckpointSaveState",
    "ScheduleFreeParameterMode",
    "save_checkpoint_with_schedulefree_eval_parameters",
    "schedulefree_parameter_mode",
]
