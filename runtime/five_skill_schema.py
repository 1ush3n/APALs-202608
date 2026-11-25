"""五技能 APAL 生产数据的统一、不可绕过 schema 校验。"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
import yaml

from runtime.paths import PROJECT_ROOT


EXPLICIT_FIVE_SKILL_PROTOCOL: Final[str] = "explicit_fiveskill_v1"
ALLOWED_PRODUCTION_PROTOCOLS: Final[frozenset[str]] = frozenset({EXPLICIT_FIVE_SKILL_PROTOCOL})
REQUIRED_CSV_COLUMNS: Final[tuple[str, ...]] = (
    "序号", "AO号", "类型", "专业编码", "工种", "紧前工序AO号", "需求人数",
    "加工时间/h", "限定站位", "部位容量",
)
REQUIRED_SKILL_IDS: Final[frozenset[int]] = frozenset(range(5))


def load_profession_skill_mapping(
    mapping_path: str | Path = PROJECT_ROOT / "conf" / "data" / "skill_groups_5.yaml",
) -> dict[str, int]:
    """读取唯一正式的专业编码→五技能映射。"""
    path = Path(mapping_path)
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    groups = payload.get("groups", {})
    mapping = {
        str(code).strip().upper(): int(skill_id)
        for skill_id, codes in groups.items()
        for code in codes
    }
    if set(mapping.values()) != REQUIRED_SKILL_IDS:
        raise ValueError(f"技能映射必须且只能覆盖 0–4，实际={sorted(set(mapping.values()))}")
    return mapping


def validate_explicit_five_skill_frame(
    frame: pd.DataFrame,
    *,
    source: str | Path,
    require_all_skills: bool,
    mapping_path: str | Path = PROJECT_ROOT / "conf" / "data" / "skill_groups_5.yaml",
) -> dict[str, int]:
    """校验正式 APAL CSV 的字段与五技能语义，失败即拒绝运行。"""
    label = str(source)
    missing = [column for column in REQUIRED_CSV_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} 缺少正式五技能字段: {missing}；历史无工种 CSV 不可运行")
    if frame.empty:
        raise ValueError(f"{label} 为空，不能作为 APAL 生产数据")

    try:
        node_type = pd.to_numeric(frame["类型"], errors="raise").astype(int)
        skill_type = pd.to_numeric(frame["工种"], errors="raise").astype(int)
        demand = pd.to_numeric(frame["需求人数"], errors="raise").astype(int)
        duration = pd.to_numeric(frame["加工时间/h"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 的类型、工种、需求人数或工时含非数值字段") from exc

    physical = node_type.eq(2)
    virtual = node_type.eq(1)
    if not bool((physical | virtual).all()):
        raise ValueError(f"{label} 存在非 1/2 的节点类型")
    if not bool(skill_type[physical].isin(REQUIRED_SKILL_IDS).all()):
        values = sorted(set(skill_type[physical].tolist()))
        raise ValueError(f"{label} 物理工序工种必须为 0–4，实际={values}")
    if not bool(skill_type[virtual].eq(-1).all()):
        raise ValueError(f"{label} 虚拟节点工种必须为 -1")
    if not bool(duration[physical].gt(0).all()) or not bool(duration[virtual].eq(0).all()):
        raise ValueError(f"{label} 的物理/虚拟节点工时不符合正式 schema")
    if not bool(demand[physical].ge(1).all()) or not bool(demand[virtual].eq(0).all()):
        raise ValueError(f"{label} 的物理/虚拟节点需求人数不符合正式 schema")

    ao = frame["AO号"].fillna("").astype(str).str.strip()
    if not bool(ao.ne("").all()) or not bool(ao.is_unique):
        raise ValueError(f"{label} 的 AO号 不能为空且必须唯一")
    profession = frame["专业编码"].fillna("").astype(str).str.strip().str.upper()
    if not bool(profession[physical].ne("").all()):
        raise ValueError(f"{label} 的物理工序缺少专业编码")
    expected_profession = ao[physical].str[1].str.upper()
    if not bool((profession[physical] == expected_profession).all()):
        raise ValueError(f"{label} 的专业编码必须等于物理 AO号第二字符")
    mapping = load_profession_skill_mapping(mapping_path)
    expected_skill = profession[physical].map(mapping)
    if expected_skill.isna().any():
        unknown = sorted(set(profession[physical][expected_skill.isna()].tolist()))
        raise ValueError(f"{label} 存在未映射专业编码: {unknown}")
    if not bool((expected_skill.astype(int) == skill_type[physical]).all()):
        raise ValueError(f"{label} 的工种与专业编码—技能映射不一致")
    observed = frozenset(int(value) for value in skill_type[physical].tolist())
    if require_all_skills and observed != REQUIRED_SKILL_IDS:
        raise ValueError(f"{label} 未完整覆盖五技能 0–4，实际={sorted(observed)}")
    return {str(skill): int((skill_type[physical] == skill).sum()) for skill in sorted(REQUIRED_SKILL_IDS)}


def validate_explicit_five_skill_csv(
    path: str | Path,
    *,
    require_all_skills: bool = False,
    mapping_path: str | Path = PROJECT_ROOT / "conf" / "data" / "skill_groups_5.yaml",
) -> dict[str, int]:
    """读取并校验一份正式 CSV；供所有生产入口复用。"""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"正式数据 CSV 不存在: {csv_path}")
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"正式五技能数据必须为 CSV，拒绝: {csv_path}")
    return validate_explicit_five_skill_frame(
        pd.read_csv(csv_path), source=csv_path, require_all_skills=require_all_skills, mapping_path=mapping_path,
    )


__all__ = [
    "ALLOWED_PRODUCTION_PROTOCOLS", "EXPLICIT_FIVE_SKILL_PROTOCOL", "REQUIRED_CSV_COLUMNS",
    "REQUIRED_SKILL_IDS", "load_profession_skill_mapping", "validate_explicit_five_skill_csv",
    "validate_explicit_five_skill_frame",
]
