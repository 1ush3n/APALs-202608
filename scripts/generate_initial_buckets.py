from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.hydra_config import (
    ExtraArgument,
    HydraCliError,
    hydra_help,
    initialize_keyvalue_args,
    should_show_help,
)
from utils.generate_random_dataset import generate_bucket


BUCKETS = {
    "283": ("data/283.csv", "data/generated/initial_283", 200, 350),
    "680": ("data/680.csv", "data/generated/initial_680", 550, 850),
    "2338": ("data/2338.csv", "data/generated/initial_2338", 2000, 2750),
    "3182": ("data/3182.csv", "data/generated/initial_3182", 2800, 3500),
}

BUCKET_ARGS = {
    "bucket": ExtraArgument(default="all", help="数据桶名称，可选 all/283/680/2338/3182"),
    "num_samples": ExtraArgument(default=32, help="每个数据桶生成的样本数量"),
    "time_var": ExtraArgument(default=0.2, help="工时扰动系数"),
    "seed": ExtraArgument(default=42, help="随机种子"),
    "overwrite": ExtraArgument(default=False, help="是否覆盖已有输出目录"),
}


def _clear_output_directory(output_dir: Path) -> None:
    """清空受管生成目录内容，但保留 OneDrive 同步目录本身。"""
    resolved = output_dir.resolve()
    generated_root = (PROJECT_ROOT / "data" / "generated").resolve()
    if not resolved.is_relative_to(generated_root):
        raise ValueError(f"拒绝清理受管目录之外的路径：{resolved}")
    for child in resolved.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def main(argv: list[str] | None = None) -> None:
    if should_show_help(argv):
        print(hydra_help(BUCKET_ARGS))
        return
    try:
        args = initialize_keyvalue_args(argv, extra_arguments=BUCKET_ARGS)
    except HydraCliError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)

    bucket_name = str(args.bucket)
    if bucket_name not in {"all", *BUCKETS}:
        raise ValueError(f"未知数据桶: {bucket_name}")

    selected = BUCKETS if bucket_name == "all" else {bucket_name: BUCKETS[bucket_name]}
    worker_pool = PROJECT_ROOT / "data" / "worker_pool_fixed.csv"
    seed_offsets = {name: index for index, name in enumerate(BUCKETS)}

    for name, (template_rel, output_rel, min_length, max_length) in selected.items():
        output_dir = PROJECT_ROOT / output_rel
        if output_dir.exists() and any(output_dir.iterdir()):
            if not bool(args.overwrite):
                raise FileExistsError(f"{output_dir} 已存在数据；重建时请使用 overwrite=true")
            _clear_output_directory(output_dir)

        manifest = generate_bucket(
            PROJECT_ROOT / template_rel,
            output_dir,
            min_length=min_length,
            max_length=max_length,
            num_samples=int(args.num_samples),
            time_var=float(args.time_var),
            seed=int(args.seed) + seed_offsets[name],
            worker_pool_path=worker_pool,
        )
        print(
            f"[{name}] 范围={min_length}-{max_length}，"
            f"变体={len(manifest['files'])}，目录={output_dir}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
