from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

from configs import configs
from runtime.hydra_config import (
    HydraCliError,
    hydra_help,
    initialize_hydra_runtime,
    should_show_help,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def train(_args) -> None:
    """历史 legacy 训练入口已归档；主入口固定使用 Lightning。"""
    raise RuntimeError("legacy 训练入口已归档；请使用 `python train.py experiment=...` 启动 Lightning。")


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if should_show_help(raw_args):
        print(hydra_help())
        return 0

    try:
        args = initialize_hydra_runtime(
            raw_args,
            target=configs,
            project_root=PROJECT_ROOT,
            default_experiment="default",
        )
    except (HydraCliError, KeyError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if sys.platform == "win32":
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
    mp.freeze_support()

    from train_lightning import run as run_lightning

    run_lightning(args, config_initialized=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
