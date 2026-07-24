"""将初始调度随机补充验证的四规模汇总登记到实验主表。"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "results/experiment_master_results.csv"
SUMMARY = ROOT / "results/01_initial_main_stochastic_20260723/initial_stochastic_supplement_summary.csv"

def main() -> int:
    with MASTER.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    existing = {row["experiment_id"] for row in rows}
    with SUMMARY.open(encoding="utf-8-sig", newline="") as handle:
        summaries = list(csv.DictReader(handle))
    method_map = {
        "full_joint": ("HB-GAT-PPO", "full_joint"),
        "mean_max_pooling": ("HB-GAT-PPO", "mean_max_pooling"),
        "operation_only": ("HB-GAT-PPO", "operation_only"),
        "operation_station": ("HB-GAT-PPO", "operation_station"),
        "fixed_preallocation": ("HB-GAT-PPO", "fixed_preallocation"),
    }
    task_counts = {"283": 283, "680": 680, "2338": 2338, "3182": 3182}
    archive_base = "results/01_initial_main_stochastic_20260723"
    added = 0
    for item in summaries:
        variant = item["variant"]
        dataset = item["dataset"]
        method, variant_value = method_map[variant]
        experiment_id = f"initial_{variant}_stochastic_supplement_{dataset}_20260723"
        if experiment_id in existing:
            continue
        row = {field: "" for field in fields}
        row.update({
            "experiment_id": experiment_id,
            "phase": "initial_schedule",
            "experiment_group": "initial_ablation_stochastic_supplement",
            "method": method,
            "variant": variant_value,
            "dataset": dataset,
            "instance_id": f"real_{dataset}",
            "scenario_level": "standard",
            "eval_protocol": "temperature=0.01_seed42_46_supplement",
            "status": "completed_stochastic_supplement",
            "priority": "high",
            "paper_table_role": "supplementary_seed_sweep",
            "fairness_status": "same_checkpoint_same_dataset_same_temperature_seed_protocol",
            "strict_main_table_eligible": "conditional",
            "seed": "42-46",
            "num_runs": "5",
            "scenario_count": "1",
            "task_count": str(task_counts[dataset]),
            "makespan": item["makespan_mean"],
            "makespan_mean": item["makespan_mean"],
            "makespan_std": item["makespan_sample_std"],
            "eligible_rate": item["eligible_rate"],
            "complete_rate": item["complete_rate"],
            "valid_rate": item["eligible_rate"],
            "duration_sec": item["mean_duration_sec"],
            "violation_summary": "all hard violations=0; recomputed from 20 raw schedules",
            "source_file": f"{archive_base}/{('joint100_full_joint_seed42_20260719' if variant == 'full_joint' else 'ablation_joint100_' + variant + '_seed42_20260720')}/eval/initial_sixrun_20260722_stochastic",
            "command_or_next_action": "retain deterministic temp0 seed42 in unified_eval_parallel_20260720_214219; use this row only for stochastic supplement",
            "notes": "4 datasets x 5 stochastic seeds; summary makespan is recomputed mean; original per-run summary lacks an independent CLI temperature field, so strict eligibility remains conditional.",
        })
        rows.append(row)
        existing.add(experiment_id)
        added += 1
    with MASTER.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print({"added": added, "total_rows": len(rows)})
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
