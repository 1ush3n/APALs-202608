from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.generate_random_dataset import generate_bucket


BUCKETS = {
    "283": ("data/283.csv", "data/generated/initial_283", 200, 350),
    "680": ("data/680.csv", "data/generated/initial_680", 550, 850),
    "2338": ("data/2338.csv", "data/generated/initial_2338", 2000, 2750),
    "3182": ("data/3182.csv", "data/generated/initial_3182", 2800, 3500),
}


def _clear_directory_contents(directory: Path) -> None:
    """清空目录内容但保留目录节点，兼容 OneDrive 只读重解析目录。"""
    for child in directory.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            try:
                child.unlink()
            except PermissionError:
                os.chmod(child, 0o666)
                child.unlink()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成四个 APAL 窄规模训练池")
    parser.add_argument("--bucket", choices=["all", *BUCKETS], default="all")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--time_var", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="删除对应旧分桶后重新生成")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    selected = BUCKETS if args.bucket == "all" else {args.bucket: BUCKETS[args.bucket]}
    worker_pool = PROJECT_ROOT / "data" / "worker_pool_fixed.csv"

    seed_offsets = {name: index for index, name in enumerate(BUCKETS)}
    for name, spec in selected.items():
        template_rel, output_rel, min_length, max_length = spec
        output_dir = PROJECT_ROOT / output_rel
        if output_dir.exists() and any(output_dir.iterdir()):
            if not args.overwrite:
                raise FileExistsError(
                    f"{output_dir} 已存在数据；如需重建请增加 --overwrite"
                )
            _clear_directory_contents(output_dir)

        manifest = generate_bucket(
            PROJECT_ROOT / template_rel,
            output_dir,
            min_length=min_length,
            max_length=max_length,
            num_samples=args.num_samples,
            time_var=args.time_var,
            seed=args.seed + seed_offsets[name],
            worker_pool_path=worker_pool,
        )
        print(
            f"[{name}] {output_dir} | 范围={min_length}-{max_length} | "
            f"变体={len(manifest['files'])}"
        )


if __name__ == "__main__":
    main()
