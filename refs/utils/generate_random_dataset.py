from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_loader import load_data


@dataclass(frozen=True)
class GeneratedDatasetRecord:
    file: str
    target_task_count: int
    actual_task_count: int
    graph_node_count: int
    sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于指定 APAL 模板生成窄规模合法变体")
    parser.add_argument("--template", type=Path, required=True, help="唯一基准模板 CSV")
    parser.add_argument("--output_dir", type=Path, required=True, help="分桶输出目录")
    parser.add_argument("--min_length", type=int, required=True, help="最少真实工序数（类型=2）")
    parser.add_argument("--max_length", type=int, required=True, help="最多真实工序数（类型=2）")
    parser.add_argument("--num_samples", type=int, default=10, help="生成变体数量")
    parser.add_argument("--time_var", type=float, default=0.2, help="工时高斯扰动系数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_predecessors(value: Any) -> list[str]:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "0"}:
        return []
    return [item.strip() for item in re.split(r"[,，]", text) if item.strip()]


def get_active_ancestors(
    node: str,
    drop_set: set[str],
    pred_map: dict[str, list[str]],
    memo: dict[str, set[str]],
    visited: set[str],
) -> set[str]:
    if node in visited:
        return set()
    visited.add(node)
    if node in memo:
        return memo[node]
    if node not in pred_map or not pred_map[node]:
        return {node} if node not in drop_set else set()

    active_preds: set[str] = set()
    for predecessor in pred_map[node]:
        if predecessor in drop_set:
            active_preds.update(
                get_active_ancestors(predecessor, drop_set, pred_map, memo, visited.copy())
            )
        else:
            active_preds.add(predecessor)
    memo[node] = active_preds
    return active_preds


def generate_random_dataset(
    template_path: str | Path,
    output_path: str | Path,
    target_length: int,
    time_var: float,
) -> tuple[int, int]:
    template = Path(template_path)
    output = Path(output_path)
    df = pd.read_csv(template, dtype=str)
    required_columns = {"AO号", "类型", "紧前工序AO号", "需求人数", "加工时间/h"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"模板缺少列: {sorted(missing)}")

    df["类型"] = df["类型"].astype(int)
    df["加工时间/h"] = df["加工时间/h"].astype(float)
    pred_map = {
        str(row["AO号"]).strip(): _split_predecessors(row.get("紧前工序AO号", ""))
        for _, row in df.iterrows()
    }

    type2_indices = df[df["类型"] == 2].index.tolist()
    current_type2_len = len(type2_indices)
    if current_type2_len >= target_length:
        base_drop = current_type2_len - target_length
        extra = random.randint(0, int(target_length * 0.1))
        num_to_drop = base_drop + extra
        num_to_add = extra
    else:
        base_add = target_length - current_type2_len
        extra = random.randint(0, int(current_type2_len * 0.05))
        num_to_drop = extra
        num_to_add = base_add + extra

    num_to_drop = min(num_to_drop, len(type2_indices))
    drop_set: set[str] = set()
    if num_to_drop > 0:
        drop_indices = np.random.choice(type2_indices, num_to_drop, replace=False)
        drop_set = set(df.loc[drop_indices, "AO号"].str.strip())
        df = df.drop(drop_indices)

    memo: dict[str, set[str]] = {}
    for idx, row in df.iterrows():
        new_preds: set[str] = set()
        for predecessor in pred_map.get(str(row["AO号"]).strip(), []):
            if predecessor in drop_set:
                new_preds.update(
                    get_active_ancestors(predecessor, drop_set, pred_map, memo, set())
                )
            else:
                new_preds.add(predecessor)
        df.at[idx, "紧前工序AO号"] = ",".join(sorted(new_preds))

    if num_to_add > 0:
        current_edges: list[tuple[str, str, Any]] = []
        for idx, row in df.iterrows():
            node_id = str(row["AO号"]).strip()
            current_edges.extend(
                (predecessor, node_id, idx)
                for predecessor in _split_predecessors(row.get("紧前工序AO号", ""))
            )

        existing_demands = df.loc[df["类型"] == 2, "需求人数"].dropna().tolist()
        if not existing_demands:
            existing_demands = ["1", "2"]

        added_count = 0
        while added_count < num_to_add and current_edges:
            edge_idx = random.randrange(len(current_edges))
            node_a, node_b, df_idx_b = current_edges.pop(edge_idx)
            node_n = f"RAND-N{random.randint(1000, 99999)}-{added_count}"
            a_rows = df[df["AO号"].str.strip() == node_a]
            mean_duration = (
                float(a_rows["加工时间/h"].values[0])
                if not a_rows.empty and float(a_rows["加工时间/h"].values[0]) > 0
                else 1.0
            )
            new_row = {
                "AO号": node_n,
                "类型": 2,
                "紧前工序AO号": node_a,
                "需求人数": random.choice(existing_demands),
                "加工时间/h": round(
                    max(0.1, np.random.normal(mean_duration, mean_duration * time_var)), 2
                ),
                "限定站位": "",
                "部位容量": "",
            }
            b_preds = _split_predecessors(df.at[df_idx_b, "紧前工序AO号"])
            if node_a in b_preds:
                b_preds.remove(node_a)
                b_preds.append(node_n)
                df.at[df_idx_b, "紧前工序AO号"] = ",".join(b_preds)
            inserted_idx = float(df_idx_b) - 0.001 - added_count * 0.0001
            while inserted_idx in df.index:
                inserted_idx -= 0.000001
            df.loc[inserted_idx] = new_row
            # 新节点形成 A -> N -> B，两条边都可继续细分，支持向上扩展较多工序。
            current_edges.append((node_a, node_n, inserted_idx))
            current_edges.append((node_n, node_b, df_idx_b))
            added_count += 1

        if added_count != num_to_add:
            raise ValueError(f"可细分边不足，仅新增 {added_count}/{num_to_add} 个工序")
        df = df.sort_index().reset_index(drop=True)

    # 剪枝后做一次全图引用闭包修复，避免下游节点残留已删除前驱。
    active_ids = {str(value).strip() for value in df["AO号"]}

    def resolve_surviving_predecessors(node: str, visited: set[str]) -> set[str]:
        if node in active_ids:
            return {node}
        if node in visited:
            return set()
        visited.add(node)
        resolved: set[str] = set()
        for predecessor in pred_map.get(node, []):
            resolved.update(resolve_surviving_predecessors(predecessor, visited.copy()))
        return resolved

    for idx, row in df.iterrows():
        repaired: set[str] = set()
        for predecessor in _split_predecessors(row.get("紧前工序AO号", "")):
            repaired.update(resolve_surviving_predecessors(predecessor, set()))
        repaired.discard(str(row["AO号"]).strip())
        df.at[idx, "紧前工序AO号"] = ",".join(sorted(repaired))

    type2_mask = df["类型"] == 2
    durations = df.loc[type2_mask, "加工时间/h"].astype(float).to_numpy()
    disturbed = np.random.normal(durations, durations * time_var)
    df.loc[type2_mask, "加工时间/h"] = np.round(np.maximum(0.1, disturbed), 2)
    df = df.fillna("").reset_index(drop=True)
    if "序号" in df.columns:
        df["序号"] = df.index + 1
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8-sig")
    return num_to_drop, num_to_add


def validate_generated_dataset(
    dataset_path: str | Path,
    worker_pool_path: str | Path,
    *,
    min_length: int,
    max_length: int,
) -> dict[str, int]:
    path = Path(dataset_path)
    df = pd.read_csv(path, dtype=str)
    task_rows = df[df["类型"].astype(int) == 2]
    task_count = len(task_rows)
    if not min_length <= task_count <= max_length:
        raise ValueError(f"真实工序数 {task_count} 不在 [{min_length}, {max_length}]")

    node_ids = {str(value).strip() for value in df["AO号"]}
    missing_refs = {
        predecessor
        for value in df["紧前工序AO号"]
        for predecessor in _split_predecessors(value)
        if predecessor not in node_ids
    }
    if missing_refs:
        raise ValueError(f"存在无效紧前引用: {sorted(missing_refs)[:5]}")

    raw_data = load_data(path)
    graph_node_count = int(raw_data["num_tasks"])
    precedence = raw_data["precedence_edges"]
    assert precedence.ndim == 2 and precedence.shape[0] == 2
    if precedence.numel() > 0:
        assert int(precedence.min()) >= 0
        assert int(precedence.max()) < graph_node_count
        successors: list[list[int]] = [[] for _ in range(graph_node_count)]
        indegree = [0] * graph_node_count
        for source, target in precedence.t().tolist():
            successors[int(source)].append(int(target))
            indegree[int(target)] += 1
        ready = [node for node, degree in enumerate(indegree) if degree == 0]
        visited_count = 0
        while ready:
            node = ready.pop()
            visited_count += 1
            for successor in successors[node]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
        if visited_count != graph_node_count:
            raise ValueError("生成数据包含有向环，不满足 APAL 紧前约束")

    worker_df = pd.read_csv(worker_pool_path)
    skill_capacity = worker_df[[f"skill_{idx}" for idx in range(10)]].sum(axis=0).to_numpy(dtype=int)
    raw_task_df = raw_data["task_df"]
    for _, row in raw_task_df.iterrows():
        skill = int(row["skill_type"])
        demand = max(1, int(row["demand_workers"]))
        if not 0 <= skill < len(skill_capacity):
            raise ValueError(f"技能编号越界: {skill}")
        if demand > skill_capacity[skill]:
            raise ValueError(f"技能 {skill} 的需求人数 {demand} 超过工人池容量")

    return {"task_count": task_count, "graph_node_count": graph_node_count}


def generate_bucket(
    template_path: Path,
    output_dir: Path,
    *,
    min_length: int,
    max_length: int,
    num_samples: int,
    time_var: float,
    seed: int,
    worker_pool_path: Path,
) -> dict[str, Any]:
    if min_length <= 0 or max_length < min_length:
        raise ValueError("工序区间非法")
    if num_samples < 1:
        raise ValueError("num_samples 必须大于 0")
    template = template_path.resolve()
    if not template.exists():
        raise FileNotFoundError(template)

    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    baseline_path = output_dir / f"baseline_{template.name}"
    shutil.copy2(template, baseline_path)

    records: list[GeneratedDatasetRecord] = []
    for sample_idx in range(1, num_samples + 1):
        target_length = random.randint(min_length, max_length)
        output_path = output_dir / (
            f"variant_{sample_idx:02d}_tasks_{target_length}_template_{template.stem}.csv"
        )
        generate_random_dataset(template, output_path, target_length, time_var)
        stats = validate_generated_dataset(
            output_path,
            worker_pool_path,
            min_length=min_length,
            max_length=max_length,
        )
        records.append(
            GeneratedDatasetRecord(
                file=output_path.name,
                target_task_count=target_length,
                actual_task_count=stats["task_count"],
                graph_node_count=stats["graph_node_count"],
                sha256=_sha256(output_path),
            )
        )

    manifest = {
        "version": 1,
        "template": str(template),
        "template_sha256": _sha256(template),
        "baseline_file": baseline_path.name,
        "min_length": min_length,
        "max_length": max_length,
        "num_samples": num_samples,
        "time_var": time_var,
        "seed": seed,
        "files": [asdict(record) for record in records],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = parse_args()
    template = args.template if args.template.is_absolute() else PROJECT_ROOT / args.template
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    manifest = generate_bucket(
        template,
        output_dir,
        min_length=args.min_length,
        max_length=args.max_length,
        num_samples=args.num_samples,
        time_var=args.time_var,
        seed=args.seed,
        worker_pool_path=PROJECT_ROOT / "data" / "worker_pool_fixed.csv",
    )
    print(
        f"生成完成: {output_dir}，模板={Path(manifest['template']).name}，"
        f"变体={len(manifest['files'])}"
    )


if __name__ == "__main__":
    main()
