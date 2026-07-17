"""从权威原始 Excel 可复现地构建带有 5 类工种的 3182 数据集。"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from utils.skill_mapping import assign_skill_columns


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

CORRECTIONS_PATH = PROJECT_ROOT / "data" / "data_corrections.yaml"
PREDECESSOR_COLUMN = LEGACY_COLUMNS[3]


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


def _assert_legacy_projection_equal(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    compare_predecessors: bool = True,
) -> None:
    """确认稳定旧字段一致；前驱列可由已登记数据修正覆盖。"""
    assert list(actual.columns.intersection(LEGACY_COLUMNS)) == LEGACY_COLUMNS, (
        "输出缺少旧版字段或字段顺序发生变化"
    )
    assert len(actual) == len(expected), "输出行数与权威 Excel 不一致"

    numeric_columns = ["序号", "类型", "需求人数", "加工时间/h", "限定站位", "部位容量"]
    text_columns = ["AO号"]
    if compare_predecessors:
        text_columns.append(PREDECESSOR_COLUMN)
    for column in numeric_columns:
        left = pd.to_numeric(actual[column], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(expected[column], errors="coerce").to_numpy(dtype=float)
        assert np.allclose(left, right, equal_nan=True), f"旧字段 {column} 与权威 Excel 不一致"
    for column in text_columns:
        left = _canonical_text(actual[column])
        right = _canonical_text(expected[column])
        assert left.equals(right), f"旧字段 {column} 与权威 Excel 不一致"


def _normalize_predecessor_tokens(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if text.lower() in {"", "0", "nan", "none"}:
        return []
    return [
        token.strip()
        for token in re.split(r"[,;，、]", text)
        if token.strip() and token.strip().lower() not in {"0", "nan", "none"}
    ]


def _project_relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def apply_registered_data_corrections(
    frame: pd.DataFrame,
    *,
    output_path: Path,
    corrections_path: Path = CORRECTIONS_PATH,
) -> pd.DataFrame:
    """把经数据所有者确认的前驱修正显式应用到派生 CSV。"""
    if not corrections_path.exists():
        return frame.copy()

    raw = yaml.safe_load(corrections_path.read_text(encoding="utf-8")) or {}
    corrections = raw.get("corrections", [])
    if not isinstance(corrections, list):
        raise ValueError(f"数据修正文件 corrections 必须为列表: {corrections_path}")

    target = _project_relative_path(output_path)
    adjusted = frame.copy()
    for correction in corrections:
        if not isinstance(correction, dict):
            raise ValueError(f"无效的数据修正规则: {correction!r}")
        datasets = {
            Path(str(item)).as_posix()
            for item in correction.get("datasets", [])
        }
        if target not in datasets:
            continue

        task_ao = str(correction["task_ao"])
        removed = str(correction["removed_predecessor_ao"])
        retained = [str(item) for item in correction.get("retained_predecessors", [])]
        matches = adjusted["AO号"].astype(str).eq(task_ao)
        if int(matches.sum()) != 1:
            raise ValueError(f"数据修正目标 AO 必须在 {output_path} 中唯一: {task_ao}")

        original = _normalize_predecessor_tokens(adjusted.loc[matches, PREDECESSOR_COLUMN].iloc[0])
        if removed not in original:
            raise ValueError(
                f"数据修正目标未包含待删除前驱: {task_ao} -> {removed}; 当前={original}"
            )
        actual_retained = [item for item in original if item != removed]
        if actual_retained != retained:
            raise ValueError(
                f"数据修正后的前驱与登记值不一致: {task_ao}; "
                f"实际={actual_retained}, 登记={retained}"
            )
        adjusted.loc[matches, PREDECESSOR_COLUMN] = ",".join(retained) if retained else np.nan
    return adjusted


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

    output = assign_skill_columns(
        raw,
        groups=config["groups"],
        physical_node_type=physical_type,
        virtual_skill_type=int(schema["virtual_skill_type"]),
        require_all_codes=True,
    )
    output = output[config["output_columns"]].copy()
    legacy_projection = raw[LEGACY_COLUMNS].copy()
    _assert_legacy_projection_equal(output, legacy_projection)
    output = apply_registered_data_corrections(
        output,
        output_path=_resolve_project_path(source["output_csv"]),
    )
    _assert_legacy_projection_equal(output, legacy_projection, compare_predecessors=False)
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
