"""从权威原始 Excel 可复现地构建带有 5 类工种的 3182 数据集。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_COLUMNS = [
    "序号",
    "AO号",
    "类型",
    "紧前工序AO号",
    "需求人数",
    "加工时间/h",
    "限定站位",
    "部位容量",
]


def _resolve_project_path(value: str | Path) -> Path:
    """将配置中的相对路径稳定地解析到项目根目录。"""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_mapping_config(config_path: Path) -> dict[str, Any]:
    """读取并验证 5 类工种映射配置。"""
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    assert isinstance(config, dict), "工种映射配置必须为字典"
    assert isinstance(config.get("source"), dict), "配置缺少 source"
    assert isinstance(config.get("schema"), dict), "配置缺少 schema"
    assert isinstance(config.get("groups"), dict), "配置缺少 groups"

    groups = {int(group_id): list(codes) for group_id, codes in config["groups"].items()}
    assert sorted(groups) == list(range(len(groups))), "工种编号必须从 0 连续编号"
    flattened = [str(code).strip().upper() for codes in groups.values() for code in codes]
    assert len(flattened) == len(set(flattened)), "专业编码不得被分配到多个工种"
    assert len(groups) == 5, "当前转换必须恰好生成 5 类工种"
    config["groups"] = groups
    return config


def _canonical_text(series: pd.Series) -> pd.Series:
    """统一缺失值与文本表达，但保留非空文本内容。"""
    return series.fillna("").astype(str)


def _assert_legacy_projection_equal(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    """确认旧 8 字段与权威 Excel 完全一致。"""
    assert list(actual.columns.intersection(LEGACY_COLUMNS)) == LEGACY_COLUMNS, (
        "输出缺少旧版字段或字段顺序发生变化"
    )
    assert len(actual) == len(expected), "输出行数与权威 Excel 不一致"

    numeric_columns = ["序号", "类型", "需求人数", "加工时间/h", "限定站位", "部位容量"]
    text_columns = ["AO号", "紧前工序AO号"]
    for column in numeric_columns:
        left = pd.to_numeric(actual[column], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(expected[column], errors="coerce").to_numpy(dtype=float)
        assert np.allclose(left, right, equal_nan=True), f"旧字段 {column} 与权威 Excel 不一致"
    for column in text_columns:
        left = _canonical_text(actual[column])
        right = _canonical_text(expected[column])
        assert left.equals(right), f"旧字段 {column} 与权威 Excel 不一致"


def build_dataset(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """从 Excel 构建派生数据，并返回输出表与旧字段权威投影。"""
    source = config["source"]
    schema = config["schema"]
    workbook = _resolve_project_path(source["workbook"])
    assert workbook.exists(), f"权威数据不存在：{workbook}"

    raw = pd.read_excel(workbook, sheet_name=source["sheet_name"], usecols="A:M")
    raw = raw.dropna(how="all").reset_index(drop=True)
    required = set(LEGACY_COLUMNS) | {schema["source_profession_column"]}
    missing = sorted(required - set(raw.columns))
    assert not missing, f"权威 Excel 缺少字段：{missing}"
    assert raw["AO号"].notna().all(), "AO号不得为空"
    assert raw["AO号"].astype(str).is_unique, "AO号必须唯一"

    node_type = pd.to_numeric(raw["类型"], errors="raise").astype(int)
    physical_type = int(schema["physical_node_type"])
    physical_mask = node_type.eq(physical_type)
    assert set(node_type.unique()) == {1, physical_type}, "节点类型必须仅包含 1 和 2"

    ao = raw["AO号"].astype(str)
    assert ao[physical_mask].str.len().ge(2).all(), "物理工序 AO号 长度不足，无法提取专业编码"
    profession_code = pd.Series("", index=raw.index, dtype="object")
    profession_code.loc[physical_mask] = ao.loc[physical_mask].str[1].str.upper()

    code_to_group = {
        str(code).strip().upper(): int(group_id)
        for group_id, codes in config["groups"].items()
        for code in codes
    }
    observed_codes = set(profession_code.loc[physical_mask].unique())
    assert observed_codes == set(code_to_group), (
        f"专业编码集合不一致；数据={sorted(observed_codes)}，配置={sorted(code_to_group)}"
    )

    skill_type = pd.Series(int(schema["virtual_skill_type"]), index=raw.index, dtype="int64")
    skill_type.loc[physical_mask] = profession_code.loc[physical_mask].map(code_to_group).astype(int)

    output = raw.copy()
    output[schema["profession_code_column"]] = profession_code
    output[schema["skill_type_column"]] = skill_type
    output = output[config["output_columns"]].copy()
    legacy_projection = raw[LEGACY_COLUMNS].copy()
    _assert_legacy_projection_equal(output, legacy_projection)
    return output, legacy_projection


def write_and_verify(output: pd.DataFrame, expected_legacy: pd.DataFrame, output_path: Path) -> None:
    """写入 CSV 后重新读取并执行端到端一致性校验。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    reloaded = pd.read_csv(output_path)
    _assert_legacy_projection_equal(reloaded, expected_legacy)

    physical = pd.to_numeric(reloaded["类型"], errors="raise").eq(2)
    assert set(pd.to_numeric(reloaded.loc[physical, "工种"], errors="raise").astype(int)) == set(range(5))
    assert pd.to_numeric(reloaded.loc[~physical, "工种"], errors="raise").eq(-1).all()
    assert reloaded.loc[physical, "专业编码"].nunique() == 17


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "conf" / "data" / "skill_groups_5.yaml",
        help="工种映射 YAML 路径",
    )
    parser.add_argument("--check-only", action="store_true", help="只构建并校验，不写入 CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_mapping_config(args.config.resolve())
    output, legacy = build_dataset(config)
    output_path = _resolve_project_path(config["source"]["output_csv"])
    if not args.check_only:
        write_and_verify(output, legacy, output_path)

    physical = output["类型"].eq(2)
    workload = output.loc[physical, "加工时间/h"] * output.loc[physical, "需求人数"]
    summary = output.loc[physical, ["工种"]].assign(劳动量=workload).groupby("工种").agg(
        工序数=("工种", "size"),
        总劳动量=("劳动量", "sum"),
    )
    mode = "校验完成" if args.check_only else f"已写入 {output_path}"
    print(f"[3182转换] {mode}；总行数={len(output)}，物理工序={int(physical.sum())}")
    print(summary.to_string())


if __name__ == "__main__":
    main()
