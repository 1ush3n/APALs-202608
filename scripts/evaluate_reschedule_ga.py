from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.heuristic.reschedule_ga import evaluate_reschedule_ga
from configs import configs
from runtime.artifacts import (
    resolve_run_output_dir,
    write_run_context_files,
    write_run_manifest,
)
from runtime.hydra_config import (
    ExtraArgument,
    HydraCliError,
    hydra_help,
    initialize_hydra_runtime,
    should_show_help,
)


GA_EXTRA_ARGS = {
    "pop_size": ExtraArgument(default=30, help="GA 种群规模"),
    "max_gen": ExtraArgument(default=20, help="GA 最大迭代代数"),
    "num_runs": ExtraArgument(default=None, help="可选评估轮数；缺省使用配置"),
    "seed": ExtraArgument(default=42, help="随机种子"),
    "quiet": ExtraArgument(default=False, help="是否关闭逐场景输出"),
    "output_dir": ExtraArgument(default=None, help="可选输出目录；缺省写入本次 run 的 eval 目录"),
}


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if should_show_help(raw_args):
        print(hydra_help(GA_EXTRA_ARGS))
        return 0
    try:
        args = initialize_hydra_runtime(
            raw_args,
            target=configs,
            project_root=PROJECT_ROOT,
            default_experiment="reschedule_task_delay",
            extra_arguments=GA_EXTRA_ARGS,
        )
    except (HydraCliError, KeyError, ValueError, RuntimeError) as exc:
        print(f"[CLI] {exc}", file=sys.stderr)
        return 2

    output_dir, context = resolve_run_output_dir(
        configs,
        PROJECT_ROOT,
        default_legacy_dir="results/reschedule_ga",
        run_subdir="reschedule_ga",
        explicit_dir=args.output_dir,
        section="eval",
    )
    manifest_extra = {
        "run_type": "evaluation",
        "artifact_kind": "reschedule_ga",
        "method": "GA",
        "output_dir": str(output_dir.resolve()),
    }
    if context is not None:
        write_run_context_files(context, configs, command="evaluate_reschedule_ga", extra=manifest_extra)
    else:
        write_run_manifest(output_dir, configs, command="evaluate_reschedule_ga", extra=manifest_extra)
    summary = evaluate_reschedule_ga(
        pop_size=args.pop_size,
        max_gen=args.max_gen,
        num_runs=args.num_runs,
        seed=args.seed,
        output_dir=output_dir,
        verbose=not args.quiet,
    )
    (output_dir / "reschedule_ga_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, indent=2))
    print(f"GA 重调度明细已保存到: {output_dir / 'reschedule_ga_eval.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
