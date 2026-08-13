# -*- coding: utf-8 -*-
"""WorkerPointer v2 Fast-Exact 阶段一：平台默认环境数与 CLI 最高优先级测试。

解析规则：
- 新模式未显式指定 num_envs 时使用平台 fast_exact 默认（Windows 4 / Linux 16）；
- CLI --num_envs/--num-envs 必须进入最终覆盖表且不受 OLD_CLI_FLAGS 拒绝；
- 历史 v2 模式保持自身已解析的 num_envs 不变。
"""

from __future__ import annotations

from configs import Config
from runtime.batch_semantics import (
    resolve_effective_ppo_batch_size,
    resolve_v2_logical_batch_size,
)
from runtime.configuration import resolve_fast_exact_num_envs
from runtime.hydra_config import FINAL_PRIORITY_CLI_FLAGS, OLD_CLI_FLAGS


def _fast_exact_config(num_envs: int, fast_default: int = 4) -> Config:
    cfg = Config()
    cfg.team_selection_mode = "autoregressive_pressure_v2_fast_exact"
    cfg.num_envs = num_envs
    cfg.worker_pointer_v2_fast_default_num_envs = fast_default
    return cfg


def test_fast_exact_default_field_exists_on_config() -> None:
    cfg = Config()
    assert hasattr(cfg, "worker_pointer_v2_fast_default_num_envs")


def test_windows_default_resolves_to_4_when_cli_absent() -> None:
    # 模拟 Windows hardware 覆盖 num_envs=2，fast_exact 平台默认应为 4。
    cfg = _fast_exact_config(num_envs=2, fast_default=4)
    assert resolve_fast_exact_num_envs(cfg, cli_explicit_num_envs=False) == 4


def test_linux_default_resolves_to_16_when_cli_absent() -> None:
    cfg = _fast_exact_config(num_envs=16, fast_default=16)
    assert resolve_fast_exact_num_envs(cfg, cli_explicit_num_envs=False) == 16


def test_cli_num_envs_has_final_priority() -> None:
    # CLI 显式 8 必须压过平台默认 16。
    cfg = _fast_exact_config(num_envs=16, fast_default=16)
    cfg.num_envs = 8
    assert resolve_fast_exact_num_envs(cfg, cli_explicit_num_envs=True) == 8


def test_legacy_v2_keeps_resolved_num_envs() -> None:
    cfg = Config()
    cfg.team_selection_mode = "autoregressive_pressure_v2"
    cfg.num_envs = 4
    cfg.worker_pointer_v2_fast_default_num_envs = 16
    assert resolve_fast_exact_num_envs(cfg, cli_explicit_num_envs=False) == 4


def test_cli_num_envs_flag_has_final_priority_channel() -> None:
    assert FINAL_PRIORITY_CLI_FLAGS["--num_envs"] == "num_envs"
    assert FINAL_PRIORITY_CLI_FLAGS["--num-envs"] == "num_envs"
    assert "--num_envs" not in OLD_CLI_FLAGS
    assert "--num-envs" not in OLD_CLI_FLAGS


def test_fast_exact_logical_batch_ignores_platform_cap() -> None:
    """Fast-Exact 逻辑 batch 唯一来源是最终 batch_size，不受平台 cap 限幅。"""
    cfg = Config()
    cfg.team_selection_mode = "autoregressive_pressure_v2_fast_exact"
    cfg.ppo_batch_size_cap = 4
    cfg.worker_pointer_v2_logical_batch_cap = 64
    assert resolve_effective_ppo_batch_size(256, cfg) == 256
    assert resolve_v2_logical_batch_size(256, cfg) == 256


def test_legacy_mode_batch_still_respects_platform_cap() -> None:
    cfg = Config()
    cfg.team_selection_mode = "autoregressive"
    cfg.ppo_batch_size_cap = 4
    assert resolve_effective_ppo_batch_size(256, cfg) == 4


def test_legacy_v2_batch_ignores_platform_cap() -> None:
    cfg = Config()
    cfg.team_selection_mode = "autoregressive_pressure_v2"
    cfg.ppo_batch_size_cap = 4
    assert resolve_effective_ppo_batch_size(256, cfg) == 256
