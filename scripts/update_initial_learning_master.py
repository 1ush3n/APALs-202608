"""将新版本初始调度学习型 baseline 训练记录追加到统一实验主表。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, default=Path("results/experiment_master_results.csv"))
    parser.add_argument("--ddqn-summary", type=Path, required=True)
    parser.add_argument("--l2d-summary", type=Path, required=True)
    args = parser.parse_args()

    master = args.master.resolve()
    table = pd.read_csv(master, encoding="utf-8-sig")
    summaries = {
        "Graph-DDQN-APAL": pd.read_json(args.ddqn_summary.resolve(), typ="series"),
        "L2D-PPO-APAL": pd.read_json(args.l2d_summary.resolve(), typ="series"),
    }
    rows: list[dict[str, object]] = []
    updated_existing = False
    for method, summary in summaries.items():
        experiment_id = (
            "initial_graph_ddqn_apal_scale400_800_680_seed42_20260721"
            if method.startswith("Graph")
            else "initial_l2d_ppo_apal_scale400_800_680_seed42_20260721"
        )
        if experiment_id in set(table["experiment_id"].astype(str)):
            archive_path = Path(str(summary.get("archive_directory"))).resolve()
            try:
                source_file = archive_path.relative_to(Path.cwd().resolve()).as_posix()
            except ValueError:
                source_file = str(summary.get("archive_directory"))
            table.loc[table["experiment_id"].astype(str) == experiment_id, "source_file"] = source_file
            updated_existing = True
            continue
        eval_summary = summary.get("eval", {})
        rows.append(
            {
                "experiment_id": experiment_id,
                "phase": "initial_schedule",
                "experiment_group": "initial_literature_baseline_training",
                "method": method,
                "variant": summary.get("variant"),
                "dataset": 680,
                "instance_id": "real_680",
                "scenario_level": "training",
                "eval_protocol": "training_auto_eval_only",
                "status": "converged_training",
                "priority": "high",
                "paper_table_role": "training_diagnostic",
                "fairness_status": "needs_formal_validation",
                "strict_main_table_eligible": "no",
                "seed": summary.get("seed"),
                "num_runs": 1,
                "scenario_count": None,
                "task_count": 680,
                "makespan": summary.get("best_makespan"),
                "makespan_mean": summary.get("best_makespan"),
                "makespan_std": None,
                "normalized_makespan": None,
                "selection_score": None,
                "score": None,
                "eligible_rate": None,
                "complete_rate": eval_summary.get("complete_rate"),
                "valid_rate": eval_summary.get("valid_rate"),
                "reward": None,
                "balance_std": None,
                "worker_utilization": None,
                "station_utilization": None,
                "duration_sec": None,
                "train_hours": None,
                "eval_wall_hours": None,
                "eval_infer_hours": None,
                "takt_violation_h": None,
                "start_deviation_mean_h": None,
                "station_change_rate": None,
                "team_change_rate": None,
                "violation_summary": "best schedule independent audit: all initial-schedule hard constraints = 0",
                "source_file": (
                    Path(str(summary.get("archive_directory"))).resolve().relative_to(Path.cwd().resolve()).as_posix()
                    if Path(str(summary.get("archive_directory"))).is_absolute()
                    else summary.get("archive_directory")
                ),
                "command_or_next_action": "按统一协议补跑 real_283/680/2338/3182 的正式验证",
                "notes": (
                    f"提前停止训练归档；target={summary.get('target_episodes')}，"
                    f"metrics_max_episode={summary.get('observed_metrics_max_episode')}，"
                    f"latest_checkpoint_episode={summary.get('latest_checkpoint_episode')}，"
                    f"best_episode={summary.get('best_checkpoint_episode')}；"
                    "当前仅为 training_auto_eval_only，不进入正式跨规模主表；保留 checkpoint 以支持续训。"
                ),
            }
        )
    if rows:
        table = pd.concat([table, pd.DataFrame(rows)], ignore_index=True)
    if rows or updated_existing:
        table.to_csv(master, index=False, encoding="utf-8-sig")
    print(f"appended={len(rows)} total={len(table)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
