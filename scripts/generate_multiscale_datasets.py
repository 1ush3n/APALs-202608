from __future__ import annotations

import argparse
import random
import sys
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from scripts.generate_synthetic_dataset import generate


@lru_cache(maxsize=8)
def template_task_capacity(template_path: str) -> int:
    """返回模板中类型为 2 的真实工序数量，这是旧合成器的 target_length 上限。"""
    df = pd.read_csv(template_path, dtype=str)
    type_col = None
    for col in df.columns:
        values = pd.to_numeric(df[col], errors="coerce")
        valid = values.dropna()
        if len(valid) == 0:
            continue
        ratio = float(valid.isin([1, 2]).mean())
        if ratio > 0.9 and int((values == 2).sum()) > 0:
            type_col = col
            break
    if type_col is None:
        raise ValueError(f"无法识别模板类型列: {template_path}")
    return int((pd.to_numeric(df[type_col], errors="coerce") == 2).sum())


def choose_template(target_ops: int) -> Path:
    """数据生成采用向上取模板：从容量足够的更大真实数据集删减。"""
    templates = [
        PROJECT_ROOT / "data" / "283.csv",
        PROJECT_ROOT / "data" / "680.csv",
        PROJECT_ROOT / "data" / "2338.csv",
        PROJECT_ROOT / "data" / "3182.csv",
    ]
    target = int(target_ops)
    for template in templates:
        if target <= template_task_capacity(str(template)):
            return template
    return templates[-1]


def clamp_target_for_template(target_ops: int, template: Path) -> int:
    """旧合成器不支持目标超过模板真实工序容量，这里提前裁剪。"""
    cap = template_task_capacity(str(template))
    return max(1, min(int(target_ops), cap))


def build_targets(count: int, min_ops: int, max_ops: int, seed: int) -> list[int]:
    rng = random.Random(int(seed))
    return [rng.randint(int(min_ops), int(max_ops)) for _ in range(int(count))]


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 200-3100 工序范围的多规模 APAL 合成数据集")
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--min-ops", type=int, default=200)
    parser.add_argument("--max-ops", type=int, default=3100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "multiscale_datasets"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = 0

    for offset, target_ops in enumerate(build_targets(args.count, args.min_ops, args.max_ops, args.seed)):
        seed = int(args.seed) + offset
        adjusted_target = int(target_ops)
        df = None
        template = choose_template(adjusted_target)
        for retry in range(6):
            template = choose_template(adjusted_target)
            safe_target = clamp_target_for_template(adjusted_target, template)
            df = generate(template, safe_target, seed + retry * 10_000)
            if df is None:
                adjusted_target = max(int(args.min_ops), adjusted_target - 50)
                continue
            actual_len = len(df)
            if int(args.min_ops) <= actual_len <= int(args.max_ops):
                break
            if actual_len > int(args.max_ops):
                adjusted_target = max(int(args.min_ops), adjusted_target - (actual_len - int(args.max_ops)) - 10)
            else:
                adjusted_target = min(int(args.max_ops), adjusted_target + (int(args.min_ops) - actual_len) + 10)
            df = None
        if df is None:
            print(f"[SKIP] target={target_ops} seed={seed} template={template.name}")
            continue
        out_path = output_dir / f"syn_{len(df)}_{seed}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        generated += 1
        print(f"[{generated}/{args.count}] {out_path.name} target={target_ops} adjusted={adjusted_target} template={template.name}")

    print(f"生成完成: {generated}/{args.count} -> {output_dir}")


if __name__ == "__main__":
    main()
