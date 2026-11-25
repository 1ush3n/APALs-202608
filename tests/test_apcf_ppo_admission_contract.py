# -*- coding: utf-8 -*-
"""APCF PPO smoke 的输出隔离与 checkpoint 元数据契约测试。"""

from pathlib import Path

import pytest

from configs import Config
from runtime.checkpoints import build_checkpoint_metadata


def test_config_exposes_apcf_smoke_guard_and_loaded_key_count() -> None:
    """配置必须显式保存 smoke 根目录和预训练加载键数量。"""
    config = Config()
    assert hasattr(config, "apcf_smoke_guard_root")
    assert hasattr(config, "apcf_pretrain_loaded_model_key_count")


def test_apcf_smoke_output_guard_accepts_only_paths_under_root(tmp_path: Path) -> None:
    """smoke 输出必须全部位于新的 smoke 根目录内。"""
    from runtime.artifacts import assert_apcf_smoke_output_isolated

    smoke_root = tmp_path / "smoke"
    assert_apcf_smoke_output_isolated(
        smoke_root,
        [smoke_root / "run", smoke_root / "checkpoints", smoke_root / "tensorboard"],
    )
    with pytest.raises(ValueError, match="smoke"):
        assert_apcf_smoke_output_isolated(
            smoke_root,
            [smoke_root / "run", tmp_path / "formal_checkpoints"],
        )


def test_checkpoint_metadata_records_pretrain_source_and_loaded_key_count() -> None:
    """PPO checkpoint 必须记录 APCF 预训练源 checkpoint 和加载键数量。"""
    config = Config()
    config.policy_action_scope = "operation_station_anchor_proposal_team"
    config.anchor_proposal_pretrain_source_sha256 = "a" * 64
    config.apcf_pretrain_loaded_model_key_count = 582
    metadata = build_checkpoint_metadata(config)
    assert metadata["apcf_pretrain_source_sha256"] == "a" * 64
    assert metadata["apcf_pretrain_loaded_model_key_count"] == 582


def test_checkpoint_metadata_repair_writes_lightning_payload(tmp_path: Path) -> None:
    """?? callback ????? metadata ????? APCF ?????"""
    import torch
    from train_lightning import _ensure_checkpoint_metadata

    path = tmp_path / "last.ckpt"
    torch.save({"state_dict": {}}, path)

    class Module:
        def on_save_checkpoint(self, checkpoint: dict) -> None:
            checkpoint["apal_metadata"] = {
                "model_spec": {"policy_action_scope": "operation_station_anchor_proposal_team"},
                "apcf_pretrain_source_sha256": "a" * 64,
            }

