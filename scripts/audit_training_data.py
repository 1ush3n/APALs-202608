"""审计动态训练数据集的来源重合、五工种语义和结构合法性。"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = [
    "序号",
    "AO号",
    "类型",
    "专业编码",
    "工种",
    "紧前工序AO号",
    "需求人数",
    "加工时间/h",
    "限定站位",
    "部位容量",
]


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _split_predecessors(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if text.lower() in {"", "0", "nan", "none"}:
        return []
    parts = [part.strip() for part in re.split(r"[,，;；]", text)]
    return [part for part in parts if part.lower() not in {"", "0", "nan", "none"}]


def _load_skill_mapping(config_path: Path) -> dict[str, int]:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    groups = config["groups"]
    return {
        str(code).strip().upper(): int(group_id)
        for group_id, codes in groups.items()
        for code in codes
    }


def audit_training_data(
    train_dir: Path,
    reference_path: Path,
    worker_pool_path: Path,
    mapping_path: Path,
    *,
    file_pattern: str = "*.csv",
    min_ops: int = 400,
    max_ops: int = 800,
) -> dict[str, Any]:
    files = sorted(train_dir.glob(file_pattern))
    if not files:
        raise ValueError(f"训练目录中没有 CSV：{train_dir}")

    reference = pd.read_csv(reference_path)
    assert reference["AO号"].astype(str).is_unique, "参考数据 AO号 必须唯一"
    reference_ids = set(reference["AO号"].astype(str))
    reference_external_predecessors = {
        predecessor
        for value in reference["紧前工序AO号"]
        for predecessor in _split_predecessors(value)
        if predecessor not in reference_ids
    }
    assert not reference_external_predecessors, (
        "参考数据存在悬空前驱 AO，必须先显式纠错，禁止把它们作为允许例外："
        f"{sorted(reference_external_predecessors)}"
    )
    reference = reference.set_index(reference["AO号"].astype(str), drop=False)
    code_to_skill = _load_skill_mapping(mapping_path)

    total_rows = 0
    overlap_rows = 0
    duration_matches = 0
    demand_matches = 0
    node_type_matches = 0
    three_field_matches = 0
    profession_matches = 0
    skill_matches = 0
    positive_duration_rows = 0
    physical_rows = 0
    physical_counts: list[int] = []
    skill_counts: Counter[int] = Counter()
    nonoverlap_skill_counts: Counter[int] = Counter()
    known_external_predecessor_references = 0
    all_frames: list[pd.DataFrame] = []

    for path in files:
        frame = pd.read_csv(path)
        assert frame.columns.tolist() == REQUIRED_COLUMNS, f"{path.name} 字段或顺序不一致"
        assert frame["AO号"].notna().all(), f"{path.name} 存在空 AO号"
        assert frame["AO号"].astype(str).is_unique, f"{path.name} 存在重复 AO号"
        assert pd.to_numeric(frame["序号"], errors="raise").astype(int).tolist() == list(
            range(1, len(frame) + 1)
        ), f"{path.name} 序号不连续"

        node_type = pd.to_numeric(frame["类型"], errors="raise").astype(int)
        skill_type = pd.to_numeric(frame["工种"], errors="raise").astype(int)
        duration = pd.to_numeric(frame["加工时间/h"], errors="raise")
        demand = pd.to_numeric(frame["需求人数"], errors="raise").astype(int)
        physical = node_type.eq(2)
        virtual = node_type.eq(1)
        assert (physical | virtual).all(), f"{path.name} 存在非法节点类型"
        assert skill_type[physical].between(0, 4).all(), f"{path.name} 物理工序工种越界"
        assert skill_type[virtual].eq(-1).all(), f"{path.name} 虚拟节点工种不是 -1"
        assert duration[physical].gt(0).all(), f"{path.name} 物理工序存在非正工时"
        assert duration[virtual].eq(0).all(), f"{path.name} 虚拟节点工时不是 0"
        assert demand[physical].ge(1).all(), f"{path.name} 物理工序需求人数小于 1"
        assert demand[virtual].eq(0).all(), f"{path.name} 虚拟节点需求人数不是 0"

        ao = frame["AO号"].astype(str)
        profession = frame["专业编码"].fillna("").astype(str).str.upper()
        assert (profession[physical] == ao[physical].str[1].str.upper()).all(), (
            f"{path.name} 专业编码与 AO号 第二字符不一致"
        )
        expected_skill = profession[physical].map(code_to_skill)
        assert expected_skill.notna().all(), f"{path.name} 存在未映射专业编码"
        assert (expected_skill.astype(int) == skill_type[physical]).all(), (
            f"{path.name} 工种与专业编码映射不一致"
        )

        ids = set(ao)
        graph = nx.DiGraph()
        graph.add_nodes_from(ids)
        for task_id, predecessors in zip(ao, frame["紧前工序AO号"], strict=True):
            for predecessor in _split_predecessors(predecessors):
                if predecessor not in ids:
                    raise AssertionError(f"{path.name} 存在悬空前驱 {predecessor}")
                graph.add_edge(predecessor, task_id)
        assert nx.is_directed_acyclic_graph(graph), f"{path.name} 工艺网络存在环"

        physical_count = int(physical.sum())
        assert min_ops <= physical_count <= max_ops, (
            f"{path.name} 物理工序数不在 [{min_ops}, {max_ops}]"
        )
        physical_counts.append(physical_count)
        physical_rows += physical_count
        positive = duration.gt(0)
        positive_duration_rows += int(positive.sum())
        assert node_type[positive].eq(2).all(), f"{path.name} 正工时节点类型不全为 2"
        skill_counts.update(skill_type[physical].tolist())

        overlap = ao.isin(reference.index)
        overlap_rows += int(overlap.sum())
        total_rows += len(frame)
        if overlap.any():
            matched_reference = reference.loc[ao[overlap]].reset_index(drop=True)
            matched_frame = frame.loc[overlap].reset_index(drop=True)
            same_duration = np.isclose(
                pd.to_numeric(matched_frame["加工时间/h"], errors="raise"),
                pd.to_numeric(matched_reference["加工时间/h"], errors="raise"),
            )
            same_demand = (
                pd.to_numeric(matched_frame["需求人数"], errors="raise").to_numpy()
                == pd.to_numeric(matched_reference["需求人数"], errors="raise").to_numpy()
            )
            same_type = (
                pd.to_numeric(matched_frame["类型"], errors="raise").to_numpy()
                == pd.to_numeric(matched_reference["类型"], errors="raise").to_numpy()
            )
            same_profession = (
                matched_frame["专业编码"].fillna("").astype(str).to_numpy()
                == matched_reference["专业编码"].fillna("").astype(str).to_numpy()
            )
            same_skill = (
                pd.to_numeric(matched_frame["工种"], errors="raise").to_numpy()
                == pd.to_numeric(matched_reference["工种"], errors="raise").to_numpy()
            )
            duration_matches += int(same_duration.sum())
            demand_matches += int(same_demand.sum())
            node_type_matches += int(same_type.sum())
            three_field_matches += int((same_duration & same_demand & same_type).sum())
            profession_matches += int(same_profession.sum())
            skill_matches += int(same_skill.sum())
        nonoverlap_skill_counts.update(skill_type[physical & ~overlap].tolist())
        all_frames.append(frame)

    workers = pd.read_csv(worker_pool_path)
    skill_columns = [f"skill_{skill_id}" for skill_id in range(5)]
    assert workers.columns.tolist() == ["worker_id", "efficiency", *skill_columns]
    assert workers["worker_id"].is_unique, "工人编号不唯一"
    worker_skills = workers[skill_columns].apply(pd.to_numeric, errors="raise").astype(int)
    assert worker_skills.isin([0, 1]).all().all(), "工人技能矩阵存在非 0/1 值"
    skills_per_worker = worker_skills.sum(axis=1)
    assert skills_per_worker.between(2, 4).all(), "每名工人必须具有 2~4 类技能"
    worker_coverage = worker_skills.sum(axis=0).astype(int)
    assert int(worker_coverage.max() - worker_coverage.min()) <= 1, "工人技能覆盖不均衡"
    efficiency = pd.to_numeric(workers["efficiency"], errors="raise")
    assert efficiency.between(0.8, 1.2).all(), "工人效率超出 [0.8, 1.2]"

    combined = pd.concat(all_frames, ignore_index=True)
    combined_skill = pd.to_numeric(combined["工种"], errors="raise").astype(int)
    combined_demand = pd.to_numeric(combined["需求人数"], errors="raise").astype(int)
    combined_duration = pd.to_numeric(combined["加工时间/h"], errors="raise")
    physical_combined = pd.to_numeric(combined["类型"], errors="raise").eq(2)
    workload = (combined_duration * combined_demand)[physical_combined]
    workload_by_skill = workload.groupby(combined_skill[physical_combined]).sum()
    total_workload = float(workload_by_skill.sum())
    maximum_demand = combined_demand.groupby(combined_skill).max().drop(index=-1, errors="ignore")
    for skill_id, demand_value in maximum_demand.items():
        assert int(demand_value) <= int(worker_coverage[f"skill_{skill_id}"]), (
            f"工种 {skill_id} 最大需求超过候选工人池覆盖"
        )

    nonoverlap_rows = total_rows - overlap_rows
    return {
        "训练文件数量": len(files),
        "总行数": total_rows,
        "AO号可在2338.csv找到的行": overlap_rows,
        "重合率": overlap_rows / total_rows,
        "非重合行": nonoverlap_rows,
        "重合行工时一致数": duration_matches,
        "重合行需求人数一致数": demand_matches,
        "重合行类型一致数": node_type_matches,
        "三个字段同时一致数": three_field_matches,
        "重合行专业编码一致数": profession_matches,
        "重合行工种一致数": skill_matches,
        "训练集中正工时行": positive_duration_rows,
        "物理工序总数": physical_rows,
        "单文件物理工序数范围": [min(physical_counts), max(physical_counts)],
        "五工种物理工序分布": {str(key): int(skill_counts[key]) for key in range(5)},
        "五工种总劳动量": {
            str(key): round(float(workload_by_skill.get(key, 0.0)), 2) for key in range(5)
        },
        "五工种劳动量占比": {
            str(key): float(workload_by_skill.get(key, 0.0)) / total_workload for key in range(5)
        },
        "非重合物理工序五工种分布": {
            str(key): int(nonoverlap_skill_counts[key]) for key in range(5)
        },
        "工人数量": len(workers),
        "工人技能覆盖": {
            str(skill_id): int(worker_coverage[f"skill_{skill_id}"]) for skill_id in range(5)
        },
        "每名工人技能数范围": [int(skills_per_worker.min()), int(skills_per_worker.max())],
        "工人效率范围": [float(efficiency.min()), float(efficiency.max())],
        "源数据已知外部前驱AO": sorted(reference_external_predecessors),
        "训练集已知外部前驱引用数": known_external_predecessor_references,
        "结构与语义校验": "全部通过",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", default="data/scale_400_800_datasets")
    parser.add_argument("--reference", default="data/2338.csv")
    parser.add_argument("--worker-pool", default="data/worker_pool_fixed.csv")
    parser.add_argument("--mapping", default="conf/data/skill_groups_5.yaml")
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--min-ops", type=int, default=400)
    parser.add_argument("--max-ops", type=int, default=800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_training_data(
        _resolve_path(args.train_dir),
        _resolve_path(args.reference),
        _resolve_path(args.worker_pool),
        _resolve_path(args.mapping),
        file_pattern=args.pattern,
        min_ops=args.min_ops,
        max_ops=args.max_ops,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
