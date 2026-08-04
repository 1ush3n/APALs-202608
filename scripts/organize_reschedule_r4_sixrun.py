"""整理并审计 r4 五技能重调度的六组、每实例 18 行验证结果。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_GROUPS = {
    "r4_temp000_seed42": (0.0, 42),
    **{f"r4_temp001_seed{seed}": (0.01, seed) for seed in range(42, 47)},
}
INSTANCE_IDS = ("real_283", "real_680", "real_2338", "real_3182")
WORKER_MAP = {"real_283": 80, "real_680": 100, "real_2338": 140, "real_3182": 160}
EXPECTED_SCENARIO_IDS = ("low_000", "medium_000", "high_000")
HARD_FIELDS = (
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
)
SUMMARY_FIELDS = (
    "selection_score",
    "composite_score",
    "avg_makespan",
    "takt_h",
    "takt_violation_h",
    "avg_balance_std",
    "avg_reward",
    "complete",
    "eligible",
    "worker_util",
    "station_util",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def discover_rows(eval_root: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    raw_files: list[dict[str, Any]] = []
    missing: list[str] = []
    for group, (temperature, seed) in EXPECTED_GROUPS.items():
        for instance_id in INSTANCE_IDS:
            csv_path = eval_root / group / instance_id / "reschedule_ppo_eval.csv"
            summary_path = eval_root / group / instance_id / "reschedule_ppo_eval_summary.json"
            if not csv_path.is_file() or not summary_path.is_file():
                missing.append(str(csv_path))
                continue
            frame = pd.read_csv(csv_path)
            frame["eval_group"] = group
            frame["temperature"] = float(temperature)
            frame["eval_seed"] = int(seed)
            frame["instance_id"] = instance_id
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            for key in ("avg_makespan", "avg_balance_std", "avg_reward", "avg_duration_sec", "worker_util", "station_util"):
                frame[key] = float(summary_payload.get(key, 0.0))
            frames.append(frame)
            for path in (csv_path, summary_path):
                raw_files.append({
                    "path": str(path.relative_to(eval_root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                })
    if missing:
        raise RuntimeError("缺少验证原始文件:\n" + "\n".join(missing))
    if not frames:
        raise RuntimeError(f"没有发现正式验证文件: {eval_root}")
    return pd.concat(frames, ignore_index=True), raw_files


def validate(frame: pd.DataFrame) -> dict[str, Any]:
    expected_rows = len(EXPECTED_GROUPS) * len(INSTANCE_IDS) * len(EXPECTED_SCENARIO_IDS)
    problems: list[str] = []
    if len(frame) != expected_rows:
        problems.append(f"总行数={len(frame)}，期望={expected_rows}")
    if set(frame["eval_group"].unique()) != set(EXPECTED_GROUPS):
        problems.append("验证组集合不完整或包含非正式组")
    for group in EXPECTED_GROUPS:
        group_rows = frame[frame["eval_group"] == group]
        if len(group_rows) != len(INSTANCE_IDS) * len(EXPECTED_SCENARIO_IDS):
            problems.append(f"{group} 行数={len(group_rows)}")
        for instance_id in INSTANCE_IDS:
            rows = group_rows[group_rows["instance_id"] == instance_id]
            if len(rows) != len(EXPECTED_SCENARIO_IDS):
                problems.append(f"{group}/{instance_id} 行数={len(rows)}")
            scenario_ids = rows["scenario_id"].astype(str).tolist()
            if len(set(scenario_ids)) != len(EXPECTED_SCENARIO_IDS) or set(scenario_ids) != set(EXPECTED_SCENARIO_IDS):
                problems.append(f"{group}/{instance_id} scenario_id 不符合三个固定场景")
            levels = rows["scenario_id"].astype(str).str.split("_").str[0]
            counts = levels.value_counts().to_dict()
            if any(counts.get(level, 0) != 1 for level in ("low", "medium", "high")):
                problems.append(f"{group}/{instance_id} level_counts={counts}")
    for field in HARD_FIELDS:
        if field not in frame:
            problems.append(f"缺少硬约束字段: {field}")
        elif float(frame[field].fillna(0).max()) > 0:
            problems.append(f"{field} 最大值={float(frame[field].max())}")
    for field in ("complete", "eligible"):
        if field not in frame:
            problems.append(f"缺少字段: {field}")
        elif float(frame[field].fillna(0).min()) < 1:
            problems.append(f"{field} 存在非 1 行")
    return {
        "protocol": "r4",
        "validation_scope": "每实例三个固定场景 × 六组种子/温度",
        "expected_groups": list(EXPECTED_GROUPS),
        "observed_groups": sorted(frame["eval_group"].unique().tolist()),
        "expected_instances": list(INSTANCE_IDS),
        "worker_map": WORKER_MAP,
        "scenario_ids": list(EXPECTED_SCENARIO_IDS),
        "rows_per_instance": len(EXPECTED_GROUPS) * len(EXPECTED_SCENARIO_IDS),
        "row_count": int(len(frame)),
        "expected_row_count": expected_rows,
        "hard_constraint_max": {field: float(frame[field].fillna(0).max()) for field in HARD_FIELDS if field in frame},
        "complete_rate": float(frame["complete"].mean()) if "complete" in frame else 0.0,
        "eligible_rate": float(frame["eligible"].mean()) if "eligible" in frame else 0.0,
        "problems": problems,
        "passed": not problems,
    }


def aggregate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    available = [name for name in SUMMARY_FIELDS if name in frame.columns]
    base = frame.copy()
    base["level"] = base["scenario_id"].astype(str).str.split("_").str[0]
    group_keys = ["eval_group", "temperature", "eval_seed", "instance_id"]
    grouped = base.groupby(group_keys, dropna=False)
    summary = grouped[available].mean().reset_index()
    summary = grouped.size().rename("scenario_count").reset_index().merge(summary, on=group_keys)
    instance_level_keys = group_keys + ["level"]
    instance_level = base.groupby(instance_level_keys, dropna=False)[available].mean().reset_index()
    instance_level = base.groupby(instance_level_keys, dropna=False).size().rename("scenario_count").reset_index().merge(
        instance_level, on=instance_level_keys
    )
    level_keys = ["eval_group", "temperature", "eval_seed", "level"]
    level = base.groupby(level_keys, dropna=False)[available].mean().reset_index()
    level = base.groupby(level_keys, dropna=False).size().rename("scenario_count").reset_index().merge(level, on=level_keys)
    return summary, instance_level, level


def main() -> int:
    parser = argparse.ArgumentParser(description="整理并审计 r4 六组重调度验证")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    eval_root = run_dir / "eval"
    frame, raw_files = discover_rows(eval_root)
    integrity = validate(frame)
    if not integrity["passed"]:
        write_json(eval_root / "integrity_check.json", integrity)
        raise SystemExit("完整性检查失败，未生成正式汇总: " + "; ".join(integrity["problems"]))

    frame.to_csv(eval_root / "raw_sixrun_rows.csv", index=False, encoding="utf-8-sig")
    summary, instance_level, level = aggregate(frame)
    summary.to_csv(eval_root / "summary.csv", index=False, encoding="utf-8-sig")
    instance_level.to_csv(eval_root / "reschedule_eval_by_instance_level.csv", index=False, encoding="utf-8-sig")
    level.to_csv(eval_root / "reschedule_eval_by_level.csv", index=False, encoding="utf-8-sig")
    overall = {
        name: float(frame[name].mean())
        for name in ("selection_score", "composite_score", "avg_makespan", "takt_h", "complete", "eligible")
        if name in frame
    }
    write_json(eval_root / "summary.json", {
        "protocol": "r4",
        "method": run_dir.parent.name,
        "run_dir": str(run_dir),
        "row_count": len(frame),
        "rows_per_instance": {instance: int((frame["instance_id"] == instance).sum()) for instance in INSTANCE_IDS},
        "overall_mean": overall,
        "by_group_instance": summary.to_dict(orient="records"),
    })
    integrity["raw_files"] = raw_files
    integrity["checkpoint_files"] = [
        {
            "path": str(checkpoint.relative_to(run_dir)),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256(checkpoint),
        }
        for checkpoint in sorted((run_dir / "checkpoints").glob("*.ckpt"))
    ]
    write_json(eval_root / "integrity_check.json", integrity)
    file_manifest = [
        {"path": str(path.relative_to(run_dir)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in eval_root.rglob("*")
        if path.is_file() and path.name != "file_manifest.json"
    ] + integrity["checkpoint_files"]
    write_json(eval_root / "file_manifest.json", file_manifest)
    readme = (
        "# r4 重调度六组验证结果\n\n"
        f"- 协议：r4 五技能；四个真实实例各评估 low_000、medium_000、high_000，工人数为 80/100/140/160。\n"
        f"- 验证组：temperature=0.0/seed42，以及 temperature=0.01/seed42–46，共 6 组。\n"
        f"- 原始行数：{len(frame)}；每个实例 18 行；四实例合计 72 行。\n"
        f"- complete_rate={integrity['complete_rate']:.4f}；eligible_rate={integrity['eligible_rate']:.4f}。\n"
        f"- 硬约束最大值：{max(integrity['hard_constraint_max'].values() or [0.0]):.4f}。\n"
        f"- 全部72行平均 selection_score={float(frame['selection_score'].mean()):.6f}；avg_makespan={float(frame['avg_makespan'].mean()):.6f} h；软 takt_violation_h 平均={float(frame['takt_violation_h'].mean()):.6f} h、最大={float(frame['takt_violation_h'].max()):.6f} h。\n"
        "- 只有本目录的六组结果进入正式汇总；其他 manifest 场景不参与统计。\n"
        "- eval_inconsistent/ 中的旧 watcher 扁平化错误输出仅作失败证据，不参与正式汇总。\n"
        "- checkpoint SHA-256 记录于 integrity_check.json 和 file_manifest.json。\n"
    )
    (eval_root / "README.md").write_text(readme, encoding="utf-8")
    # README 在上面的汇总说明写入后才最终定稿，必须重新生成清单，
    # 否则 file_manifest.json 会记录旧 README 的哈希。
    file_manifest = [
        {"path": str(path.relative_to(run_dir)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in eval_root.rglob("*")
        if path.is_file() and path.name != "file_manifest.json"
    ] + integrity["checkpoint_files"]
    write_json(eval_root / "file_manifest.json", file_manifest)
    print(json.dumps({"run_dir": str(run_dir), "row_count": len(frame), "passed": integrity["passed"], "overall": overall}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
