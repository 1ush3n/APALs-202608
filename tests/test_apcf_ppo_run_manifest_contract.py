# -*- coding: utf-8 -*-
"""APCF PPO run manifest 的预训练加载记录契约。"""

import json
from pathlib import Path
import pytest


def test_run_manifest_records_apcf_pretrain_load(tmp_path: Path) -> None:
    """run_manifest 必须记录预训练源 SHA 和加载模型键数量。"""
    from train_lightning import record_apcf_pretrain_load

    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps({"command": "train"}), encoding="utf-8")
    record_apcf_pretrain_load(
        manifest_path,
        source_sha256="a" * 64,
        loaded_model_key_count=582,
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["apcf_pretrain_loaded"] is True
    assert payload["apcf_pretrain_source_sha256"] == "a" * 64
    assert payload["apcf_pretrain_loaded_model_key_count"] == 582


def test_pretrain_load_record_requires_loaded_values(tmp_path: Path) -> None:
    """?????????????????????????"""
    from train_lightning import record_apcf_pretrain_load

    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="source_sha256"):
        record_apcf_pretrain_load(
            manifest_path,
            source_sha256="",
            loaded_model_key_count=582,
        )
