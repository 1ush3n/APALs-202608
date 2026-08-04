from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from runtime.training_data_manifest import resolve_explicit_five_skill_initial_training_paths


def _write_graph(path: Path) -> None:
    rows = [
        {"序号": 1, "AO号": "A", "类型": 1, "专业编码": "", "工种": -1, "紧前工序AO号": "", "需求人数": 0, "加工时间/h": 0, "限定站位": "", "部位容量": ""},
        {"序号": 2, "AO号": "RA0001", "类型": 2, "专业编码": "A", "工种": 0, "紧前工序AO号": "A", "需求人数": 1, "加工时间/h": 1, "限定站位": "", "部位容量": ""},
        {"序号": 3, "AO号": "RD0002", "类型": 2, "专业编码": "D", "工种": 1, "紧前工序AO号": "A", "需求人数": 1, "加工时间/h": 1, "限定站位": "", "部位容量": ""},
        {"序号": 4, "AO号": "RB0003", "类型": 2, "专业编码": "B", "工种": 2, "紧前工序AO号": "A", "需求人数": 1, "加工时间/h": 1, "限定站位": "", "部位容量": ""},
        {"序号": 5, "AO号": "RN0004", "类型": 2, "专业编码": "N", "工种": 3, "紧前工序AO号": "A", "需求人数": 1, "加工时间/h": 1, "限定站位": "", "部位容量": ""},
        {"序号": 6, "AO号": "RC0005", "类型": 2, "专业编码": "C", "工种": 4, "紧前工序AO号": "A", "需求人数": 1, "加工时间/h": 1, "限定站位": "", "部位容量": ""},
    ]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def test_initial_training_manifest_binds_exact_files(tmp_path: Path) -> None:
    train = tmp_path / "train"
    train.mkdir()
    graph = train / "g.csv"
    _write_graph(graph)
    digest = hashlib.sha256(graph.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"protocol": "explicit_fiveskill_v1", "files": [{"file": "g.csv", "sha256": digest}]}), encoding="utf-8")
    assert resolve_explicit_five_skill_initial_training_paths(manifest, train) == (graph.resolve(),)
    _write_graph(train / "extra.csv")
    with pytest.raises(ValueError, match="精确文件列表不一致"):
        resolve_explicit_five_skill_initial_training_paths(manifest, train)
