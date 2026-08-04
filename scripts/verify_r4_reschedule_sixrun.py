"""严格核验 r4 operation-station 六组、每实例18行重调度结果。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


GROUPS = ["r4_temp000_seed42", *[f"r4_temp001_seed{seed}" for seed in range(42, 47)]]
INSTANCES = ["real_283", "real_680", "real_2338", "real_3182"]
SCENARIOS = {"low_000", "medium_000", "high_000"}
HARD_FIELDS = [
    "frozen_violation_count",
    "release_violation_count",
    "precedence_violation_count",
    "worker_overlap_violation_count",
    "station_slot_violation_count",
    "skill_violation_count",
    "demand_violation_count",
    "fixed_station_violation_count",
    "station_range_violation_count",
    "physical_station_violation_count",
    "worker_station_binding_violation_count",
    "duplicate_task_count",
    "missing_task_count",
    "invalid_step_count",
    "mask_mismatch_recovery_count",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    eval_root = run_dir / "eval"
    errors: list[str] = []

    integrity = json.loads((eval_root / "integrity_check.json").read_text(encoding="utf-8"))
    summary = json.loads((eval_root / "summary.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(eval_root / "raw_sixrun_rows.csv")

    if integrity.get("protocol") != "r4":
        errors.append(f"integrity.protocol={integrity.get('protocol')}")
    if summary.get("protocol") != "r4":
        errors.append(f"summary.protocol={summary.get('protocol')}")
    if not integrity.get("passed"):
        errors.append("integrity.passed=false")
    if len(raw) != 72:
        errors.append(f"raw_rows={len(raw)}")
    if set(raw["eval_group"]) != set(GROUPS):
        errors.append("group_set_mismatch")
    if set(raw["instance_id"]) != set(INSTANCES):
        errors.append("instance_set_mismatch")
    if set(raw["scenario_id"]) != SCENARIOS:
        errors.append("scenario_set_mismatch")
    key_counts = raw.groupby(["eval_group", "instance_id", "scenario_id"]).size()
    if len(key_counts) != 72 or int(key_counts.max()) != 1:
        errors.append("duplicate_or_missing_group_instance_scenario_key")
    for instance_id in INSTANCES:
        count = int((raw["instance_id"] == instance_id).sum())
        if count != 18:
            errors.append(f"{instance_id}.rows={count}")
    for group in GROUPS:
        count = int((raw["eval_group"] == group).sum())
        if count != 12:
            errors.append(f"{group}.rows={count}")
    for field in HARD_FIELDS:
        if field not in raw:
            errors.append(f"missing_hard_field={field}")
        elif float(raw[field].fillna(0).max()) != 0.0:
            errors.append(f"{field}.max={float(raw[field].fillna(0).max())}")
    for field in ("complete", "eligible"):
        if field not in raw:
            errors.append(f"missing_field={field}")
        elif float(raw[field].fillna(0).min()) < 1.0:
            errors.append(f"{field}.min={float(raw[field].fillna(0).min())}")

    manifest = json.loads((eval_root / "file_manifest.json").read_text(encoding="utf-8"))
    manifest_errors: list[str] = []
    for item in manifest:
        path = run_dir / item["path"]
        if not path.is_file():
            manifest_errors.append(f"missing:{item['path']}")
            continue
        if sha256(path) != item["sha256"]:
            manifest_errors.append(f"hash:{item['path']}")
    if manifest_errors:
        errors.extend(manifest_errors)

    payload = {
        "passed": not errors,
        "errors": errors,
        "protocol": integrity.get("protocol"),
        "row_count": len(raw),
        "rows_per_instance": raw.groupby("instance_id").size().to_dict(),
        "rows_per_group": raw.groupby("eval_group").size().to_dict(),
        "complete_rate": float(raw["complete"].mean()),
        "eligible_rate": float(raw["eligible"].mean()),
        "hard_constraint_max": {
            field: float(raw[field].fillna(0).max()) for field in HARD_FIELDS if field in raw
        },
        "file_manifest_count": len(manifest),
        "file_manifest_errors": manifest_errors,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
