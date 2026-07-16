from __future__ import annotations

import random
import sys
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from runtime.hydra_config import (
    ExtraArgument,
    HydraCliError,
    hydra_help,
    initialize_keyvalue_args,
    should_show_help,
)
from scripts.generate_synthetic_dataset import generate


MULTISCALE_ARGS = {
    "count": ExtraArgument(default=80, help="生成数据集数量"),
    "min_ops": ExtraArgument(default=200, help="最小工序数量"),
    "max_ops": ExtraArgument(default=3100, help="最大工序数量"),
    "seed": ExtraArgument(default=42, help="随机种子"),
    "output_dir": ExtraArgument(default=str(PROJECT_ROOT / "data" / "multiscale_datasets"), help="输出目录"),
}


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
    """数据生成采用向上取模板：从容量足够的更大真实数据集裁剪。"""
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


def main(argv: list[str] | None = None) -> None:
    if should_show_help(argv):
        print(hydra_help(MULTISCALE_ARGS))
        return
    try:
        args = initialize_keyvalue_args(argv, extra_arguments=MULTISCALE_ARGS)
    except HydraCliError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)

    count = int(args.count)
    min_ops = int(args.min_ops)
    max_ops = int(args.max_ops)
    seed_base = int(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = 0

    for offset, target_ops in enumerate(build_targets(count, min_ops, max_ops, seed_base)):
        seed = seed_base + offset
        adjusted_target = int(target_ops)
        df = None
        template = choose_template(adjusted_target)
        for retry in range(6):
            template = choose_template(adjusted_target)
            safe_target = clamp_target_for_template(adjusted_target, template)
            df = generate(template, safe_target, seed + retry * 10_000)
            if df is None:
                adjusted_target = max(min_ops, adjusted_target - 50)
                continue
            actual_ops = int((pd.to_numeric(df['类型'], errors='raise') == 2).sum())
            if min_ops <= actual_ops <= max_ops:
                break
            if actual_ops > max_ops:
                adjusted_target = max(min_ops, adjusted_target - (actual_ops - max_ops) - 10)
            else:
                adjusted_target = min(max_ops, adjusted_target + (min_ops - actual_ops) + 10)
            df = None
        if df is None:
            print(f"[SKIP] target={target_ops} seed={seed} template={template.name}")
            continue
        actual_ops = int((pd.to_numeric(df['类型'], errors='raise') == 2).sum())
        out_path = output_dir / f"syn_{actual_ops}_{seed}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        generated += 1
        print(f"[{generated}/{count}] {out_path.name} target={target_ops} adjusted={adjusted_target} template={template.name}")

    if generated != count:
        raise RuntimeError(f"仅生成 {generated}/{count} 个训练文件，拒绝接受不完整训练集")
    print(f"生成完成：{generated}/{count} -> {output_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
