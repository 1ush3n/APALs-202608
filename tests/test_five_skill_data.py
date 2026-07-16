from __future__ import annotations

from uuid import uuid4

import pandas as pd
import torch

from data_loader import load_data
from scripts.build_3182_dataset import (
    PROJECT_ROOT,
    _assert_legacy_projection_equal,
    build_dataset,
    load_mapping_config,
)
from utils.resource_graph import build_task_skill_edges


CONFIG_PATH = PROJECT_ROOT / "conf" / "data" / "skill_groups_5.yaml"
CSV_PATH = PROJECT_ROOT / "data" / "3182.csv"


def test_3182_is_reproducible_from_authoritative_excel() -> None:
    config = load_mapping_config(CONFIG_PATH)
    expected, legacy = build_dataset(config)
    actual = pd.read_csv(CSV_PATH)

    _assert_legacy_projection_equal(actual, legacy)
    assert actual["专业编码"].fillna("").tolist() == expected["专业编码"].tolist()
    assert actual["工种"].astype(int).tolist() == expected["工种"].astype(int).tolist()


def test_node_type_and_skill_type_are_independent() -> None:
    loaded = load_data(CSV_PATH)["task_df"]
    physical = loaded["node_type"].eq(2)

    assert len(loaded) == 3299
    assert int(physical.sum()) == 3182
    assert set(loaded.loc[physical, "skill_type"].unique()) == set(range(5))
    assert set(loaded.loc[~physical, "skill_type"].unique()) == {-1}
    assert loaded.loc[physical, "profession_code"].nunique() == 17


def test_legacy_type_column_is_not_reused_as_skill() -> None:
    legacy_path = PROJECT_ROOT / "tests" / f".legacy_{uuid4().hex}.csv"
    pd.DataFrame(
        {
            "AO号": ["A", "AA0001-0010"],
            "类型": [1, 2],
            "紧前工序AO号": ["", "A"],
            "需求人数": [0, 2],
            "加工时间/h": [0.0, 1.0],
        }
    ).to_csv(legacy_path, index=False)

    try:
        loaded = load_data(legacy_path)["task_df"]
        assert loaded["node_type"].tolist() == [1, 2]
        assert loaded["skill_type"].tolist() == [-1, 0]
    finally:
        legacy_path.unlink(missing_ok=True)


def test_virtual_nodes_have_no_skill_hub_edge() -> None:
    # task_x 形状：[3, 18]；前两个物理工序分别需要工种 0 和 4，第三个为虚拟节点。
    task_x = torch.zeros((3, 18))
    task_x[0, 5] = 1.0
    task_x[1, 9] = 1.0
    edges = build_task_skill_edges(task_x, num_skill_types=5)

    assert edges.tolist() == [[0, 4], [0, 1]]
