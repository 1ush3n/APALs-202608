from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.heuristic.reschedule_ga import evaluate_reschedule_ga
from configs import configs, load_config_files
from train import resolve_workspace_path


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 APAL 预测-反应式重调度 GA 基线")
    parser.add_argument("--config", type=str, default="conf/experiment/reschedule_task_delay.yaml")
    parser.add_argument("--pop_size", type=int, default=30)
    parser.add_argument("--max_gen", type=int, default=20)
    parser.add_argument("--num_runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/reschedule_ga")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config_path = resolve_workspace_path(args.config)
    load_config_files([str(config_path)])
    summary = evaluate_reschedule_ga(
        pop_size=args.pop_size,
        max_gen=args.max_gen,
        num_runs=args.num_runs,
        seed=args.seed,
        output_dir=resolve_workspace_path(args.output_dir),
        verbose=not args.quiet,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, indent=2))
    print(f"GA 重调度明细已保存到: {resolve_workspace_path(args.output_dir) / 'reschedule_ga_eval.csv'}")


if __name__ == "__main__":
    main()
