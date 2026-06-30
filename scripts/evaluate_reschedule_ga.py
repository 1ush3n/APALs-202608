from __future__ import annotations

import argparse
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
from runtime.configuration import (
    add_common_config_arguments,
    parse_runtime_args,
    resolve_runtime_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 APAL 预测-反应式重调度 GA 基线")
    parser.add_argument("--config", type=str, default="conf/experiment/reschedule_task_delay.yaml")
    parser.add_argument("--pop_size", type=int, default=30)
    parser.add_argument("--max_gen", type=int, default=20)
    parser.add_argument("--num_runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    add_common_config_arguments(parser)
    args = parse_runtime_args(parser)

    resolve_runtime_config(args, target=configs)
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


if __name__ == "__main__":
    main()
