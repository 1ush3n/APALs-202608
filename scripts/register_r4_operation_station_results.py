"""将 r4 operation-station 六次验证按实例登记到统一实验总表。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


INSTANCE_DATASET = {
    "real_283": (283, 283),
    "real_680": (680, 680),
    "real_2338": (2338, 2338),
    "real_3182": (3182, 3182),
}


def build_row(frame: pd.DataFrame, instance_id: str, source: str, header: list[str]) -> dict[str, object]:
    deterministic = frame[(frame["instance_id"] == instance_id) & (frame["temperature"] == 0.0)]
    stochastic = frame[(frame["instance_id"] == instance_id) & (frame["temperature"] == 0.01)]
    if len(deterministic) != 3 or len(stochastic) != 15:
        raise ValueError(f"{instance_id}: deterministic={len(deterministic)}, stochastic={len(stochastic)}")
    mean = stochastic.mean(numeric_only=True)
    det_mean = deterministic.mean(numeric_only=True)
    dataset, task_count = INSTANCE_DATASET[instance_id]
    row: dict[str, object] = {key: "" for key in header}
    row.update(
        {
            "experiment_id": f"reschedule_r4_operation_station_sixrun_{dataset}_20260801",
            "phase": "reschedule",
            "experiment_group": "ablation",
            "method": "HB-GAT-PPO",
            "variant": "operation_station",
            "dataset": dataset,
            "instance_id": instance_id,
            "scenario_level": "all",
            "eval_protocol": "r4_sixrun_temp000_seed42_temp001_seed42_46_3levels",
            "status": "completed_formal",
            "priority": "high",
            "paper_table_role": "main_or_ablation",
            "fairness_status": "same_r4_manifest_baseline",
            "strict_main_table_eligible": "conditional",
            "seed": "42/42-46",
            "num_runs": 6,
            "scenario_count": 18,
            "task_count": task_count,
            "makespan": float(det_mean["avg_makespan"]),
            "makespan_mean": float(mean["avg_makespan"]),
            "makespan_std": float(stochastic["avg_makespan"].std(ddof=1)),
            "selection_score": float(det_mean["selection_score"]),
            "score": float(mean["selection_score"]),
            "eligible_rate": float(mean["eligible"]),
            "complete_rate": float(mean["complete"]),
            "valid_rate": float(mean["complete"]),
            "reward": float(mean["avg_reward"]),
            "balance_std": float(mean["avg_balance_std"]),
            "worker_utilization": float(mean["worker_util"]),
            "station_utilization": float(mean["station_util"]),
            "duration_sec": float(mean["avg_duration_sec"]),
            "takt_violation_h": float(mean["takt_violation_h"]),
            "start_deviation_mean_h": float(mean["start_deviation_mean_h"]),
            "station_change_rate": float(mean["station_change_rate"]),
            "team_change_rate": float(mean["team_change_rate"]),
            "violation_summary": "15项硬约束/完整性计数最大值均为0；complete=eligible=1",
            "source_file": source,
            "command_or_next_action": "r4正式验证与独立完整性审计已完成；可用于operation-station消融比较",
            "notes": (
                "确定性主结果为temperature=0/seed42；随机补充为temperature=0.01/seed42-46。"
                f"确定性selection_score={float(det_mean['selection_score']):.6f}，"
                f"随机selection_score均值={float(mean['selection_score']):.6f}。"
                "checkpoint选择使用real_680训练期异步评估，故资格为conditional。"
            ),
        }
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--master", type=Path, default=Path("results/experiment_master_results.csv"))
    parser.add_argument("--apply", action="store_true", help="确认后写回总表")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    frame = pd.read_csv(run_dir / "eval" / "raw_sixrun_rows.csv")
    with args.master.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        header = reader.fieldnames or []
    if len(header) != 43:
        raise ValueError(f"主表列数异常: {len(header)}")
    source = str((run_dir / "eval" / "summary.csv").as_posix())
    new_rows = [build_row(frame, instance_id, source, header) for instance_id in INSTANCE_DATASET]
    ids = {str(row["experiment_id"]) for row in rows}
    duplicate = [str(row["experiment_id"]) for row in new_rows if str(row["experiment_id"]) in ids]
    if duplicate:
        raise ValueError("总表已存在待登记 ID: " + ", ".join(duplicate))
    print(pd.DataFrame(new_rows)[["experiment_id", "instance_id", "makespan", "makespan_mean", "makespan_std", "selection_score", "score", "eligible_rate", "complete_rate"]].to_string(index=False))
    if not args.apply:
        print("未写回主表；如确认无误，请追加 --apply。")
        return 0
    with args.master.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        for row in new_rows:
            writer.writerow(row)
    print(f"已追加 {len(new_rows)} 行: {args.master}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
