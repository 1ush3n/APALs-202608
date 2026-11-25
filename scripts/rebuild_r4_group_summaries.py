"""从已完成的四个实例摘要重建 r4 每组的组合摘要，避免断点续跑留下单实例摘要。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


GROUPS = ["r4_temp000_seed42", *[f"r4_temp001_seed{seed}" for seed in range(42, 47)]]
INSTANCES = ["real_283", "real_680", "real_2338", "real_3182"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("data/r4/m.json"))
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = {item["instance_id"]: item for item in manifest["instances"] if item["instance_id"] in INSTANCES}
    for group in GROUPS:
        group_dir = run_dir / "eval" / group
        summaries: dict[str, dict] = {}
        rows: list[dict] = []
        model_path = None
        manifest_path = None
        for instance_id in INSTANCES:
            summary_path = group_dir / instance_id / "reschedule_ppo_eval_summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(summary_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summaries[instance_id] = summary
            model_path = summary.get("model_path", model_path)
            manifest_path = summary.get("manifest_path", manifest_path)
            entry = entries[instance_id]
            rows.append(
                {
                    "instance_id": instance_id,
                    "data_path": str(entry["data_path"]),
                    "baseline_schedule_path": str(entry["baseline_schedule_path"]),
                    "scenario_path": str(entry["scenario_path"]),
                    "num_tasks": entry["num_tasks"],
                    "baseline_makespan": entry["baseline_makespan"],
                    "scenario_count": summary.get("scenario_count", 0),
                    "avg_makespan": summary.get("avg_makespan", 0.0),
                    "avg_score": summary.get("avg_score", 0.0),
                    "avg_selection_score": summary.get("avg_selection_score", 0.0),
                    "eligible_rate": summary.get("eligible_rate", 0.0),
                    "avg_duration_sec": summary.get("avg_duration_sec", 0.0),
                    "worker_util": summary.get("worker_util", 0.0),
                    "station_util": summary.get("station_util", 0.0),
                }
            )
        with (group_dir / "reschedule_eval_by_instance.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        payload = {
            "model_path": model_path,
            "manifest_path": manifest_path,
            "model_format": "lightning",
            "instance_ids": INSTANCES,
            "rows": rows,
            "summaries": summaries,
            "scenario_ids": ["low_000", "medium_000", "high_000"],
        }
        (group_dir / "reschedule_eval_summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"rebuilt={group}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
