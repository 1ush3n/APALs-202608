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

# 过渡期兼容导出：现有评估与测试代码仍会导入这些公共能力。
# 后续应继续把这些符号迁移到 runtime/evaluation/training 模块。
from archive.legacy_train import (  # noqa: E402,F401
    Memory,
    _compute_assignment_utilization,
    _compute_reschedule_constraint_metrics,
    compute_apal_rollout_diagnostics,
    ensure_reschedule_baseline_available,
    ensure_reschedule_eval_scenarios_available,
    evaluate_initial_multi_benchmark,
    evaluate_model,
    evaluate_reschedule_model,
    load_warm_start_weights_with_input_expansion,
    refresh_env_observation,
    resolve_checkpoint_paths,
    resolve_tensorboard_log_root,
    resolve_workspace_path,
    sanitize_experiment_name,
    select_actions_batch_compat,
    set_seed,
    write_best_model_meta,
)


def initialize_training_config(args, argv=None, system_name: str | None = None):
    """兼容旧内部调用；新命令行入口应使用 Hydra key=value。"""
    from archive.legacy_train import initialize_training_config as _legacy_initialize

    return _legacy_initialize(args, argv=argv, system_name=system_name)


def train(_args) -> None:
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
