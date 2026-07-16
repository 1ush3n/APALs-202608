"""节点类型与五类工种的统一派生规则。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


def build_code_to_skill(groups: Mapping[Any, Sequence[Any]]) -> dict[str, int]:
    """把 YAML 中的分组配置转换为“专业编码→工种编号”映射。"""
    mapping = {
        str(code).strip().upper(): int(group_id)
        for group_id, codes in groups.items()
        for code in codes
    }
    assert mapping, "工种映射不得为空"
    assert len(mapping) == sum(len(codes) for codes in groups.values()), "专业编码不得重复分组"
    assert sorted({int(group_id) for group_id in groups}) == list(range(len(groups))), (
        "工种编号必须从 0 连续编号"
    )
    return mapping


def assign_skill_columns(
    frame: pd.DataFrame,
    *,
    groups: Mapping[Any, Sequence[Any]],
    physical_node_type: int = 2,
    virtual_skill_type: int = -1,
    require_all_codes: bool = False,
) -> pd.DataFrame:
    """由 AO号第二个字符派生专业编码和工种，并保留所有原始字段。"""
    required_columns = {"AO号", "类型"}
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"数据缺少工种派生必要字段：{missing}")

    result = frame.copy()
    node_type = pd.to_numeric(result["类型"], errors="raise").astype(int)
    physical_mask = node_type.eq(int(physical_node_type))
    ao = result["AO号"].fillna("").astype(str)
    if not ao[physical_mask].str.len().ge(2).all():
        raise ValueError("存在长度不足 2 的物理工序 AO号，无法提取专业编码")

    code_to_skill = build_code_to_skill(groups)
    profession = pd.Series("", index=result.index, dtype="object")
    profession.loc[physical_mask] = ao.loc[physical_mask].str[1].str.upper()
    observed_codes = set(profession.loc[physical_mask].unique())
    unmapped = sorted(observed_codes - set(code_to_skill))
    if unmapped:
        raise ValueError(f"存在未映射的专业编码：{unmapped}")
    if require_all_codes and observed_codes != set(code_to_skill):
        missing_codes = sorted(set(code_to_skill) - observed_codes)
        raise ValueError(f"权威全集缺少配置中的专业编码：{missing_codes}")

    skill_type = pd.Series(int(virtual_skill_type), index=result.index, dtype="int64")
    skill_type.loc[physical_mask] = profession.loc[physical_mask].map(code_to_skill).astype(int)
    result["专业编码"] = profession
    result["工种"] = skill_type
    return result
