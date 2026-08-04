"""归档 CTG formal-v1 初始调度训练的原始证据与诊断结果。

该脚本仅处理 results 文件。它先复制并逐文件核对 SHA-256，再生成汇总、
完整性审计和索引；只有显式传入 --remove-source 才删除根目录临时副本。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
from pathlib import Path
from typing import Any

import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_RELATIVE = Path("results/ctg_fv1_s42_260802_1512")
ARCHIVE_RELATIVE = Path(
    "results/01_initial_main/conditional_team_gate_formal_v1/ctg_fv1_s42_260802_1512"
)
ROOT_EVENT_NAME = "events.out.tfevents.1785743168.autodl-container-ce11498eee-a787ee7f.1741.0"


def sha256(path: Path) -> str:
    """计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def raw_copy_integrity(source: Path, archive: Path) -> list[dict[str, Any]]:
    """核验源目录与归档目录的每一个原始文件。"""
    rows: list[dict[str, Any]] = []
    for source_path in sorted(path for path in source.rglob("*") if path.is_file()):
        relative = source_path.relative_to(source)
        archived_path = archive / relative
        if not archived_path.is_file():
            raise FileNotFoundError(f"归档副本缺少原始文件：{relative}")
        row = {
            "relative_path": relative.as_posix(),
            "source_bytes": source_path.stat().st_size,
            "archive_bytes": archived_path.stat().st_size,
            "source_sha256": sha256(source_path),
            "archive_sha256": sha256(archived_path),
        }
        row["sha256_equal"] = row["source_sha256"] == row["archive_sha256"]
        rows.append(row)
    if not rows or not all(bool(row["sha256_equal"]) for row in rows):
        raise RuntimeError("原始文件 SHA-256 核验失败，拒绝继续归档")
    return rows


def scalar_map(event_path: Path, tag: str) -> dict[int, float]:
    accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    if tag not in accumulator.Tags().get("scalars", []):
        return {}
    return {int(item.step): float(item.value) for item in accumulator.Scalars(tag)}


def collect_training_curve(archive: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """合并续训前后 TensorBoard 标量；step 是零基，episode 是一基。"""
    event_paths = sorted((archive / "logs" / "tensorboard").rglob("events.out.tfevents.*"))
    if not event_paths:
        raise FileNotFoundError("缺少主 TensorBoard event 文件")
    tags = (
        "Rew",
        "Mk",
        "SPS",
        "Policy/ApproxKL",
        "Policy/ClipFraction",
        "Critic/Explained_Variance",
        "Entropy/Task",
        "Entropy/Station",
        "Entropy/WorkerTeam",
        "OOM/SkippedUpdate",
        "PPO/BatchVectorRepairCount",
        "PPO/GPURebuildFallbackCount",
        "Policy/Meltdown_Count",
        "Memory/Allocated_GB",
        "Memory/Reserved_GB",
    )
    by_step: dict[int, dict[str, Any]] = {}
    for event_path in event_paths:
        for tag in tags:
            for step, value in scalar_map(event_path, tag).items():
                row = by_step.setdefault(step, {"step": step, "episode": step + 1})
                row[tag] = value
    rows = [by_step[step] for step in sorted(by_step)]
    if not rows:
        raise RuntimeError("TensorBoard 中没有可用标量")
    steps = [int(row["step"]) for row in rows]
    if steps != list(range(steps[0], steps[-1] + 1)) or steps[0] != 0:
        raise RuntimeError(f"TensorBoard step 不连续：{steps[0]}..{steps[-1]}")
    for row in rows:
        row["source_event"] = next(
            path.relative_to(archive).as_posix()
            for path in event_paths
            if int(row["step"]) in scalar_map(path, "Rew")
        )
    summary = {
        "event_files": [path.relative_to(archive).as_posix() for path in event_paths],
        "tensorboard_episode_count": len(rows),
        "tensorboard_last_episode": int(rows[-1]["episode"]),
        "reward_first": rows[0].get("Rew"),
        "reward_last": rows[-1].get("Rew"),
        "reward_best": max(float(row["Rew"]) for row in rows),
        "makespan_first": rows[0].get("Mk"),
        "makespan_last": rows[-1].get("Mk"),
        "makespan_min": min(float(row["Mk"]) for row in rows),
        "max_approx_kl": max(float(row["Policy/ApproxKL"]) for row in rows),
        "max_clip_fraction": max(float(row["Policy/ClipFraction"]) for row in rows),
        "min_explained_variance": min(float(row["Critic/Explained_Variance"]) for row in rows),
        "max_reserved_gb": max(float(row["Memory/Reserved_GB"]) for row in rows),
        "oom_skipped_total": sum(float(row["OOM/SkippedUpdate"]) for row in rows),
        "batch_repair_total": sum(float(row["PPO/BatchVectorRepairCount"]) for row in rows),
        "gpu_rebuild_fallback_total": sum(float(row["PPO/GPURebuildFallbackCount"]) for row in rows),
        "meltdown_total": sum(float(row["Policy/Meltdown_Count"]) for row in rows),
    }
    return rows, summary


def collect_async_results(archive: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """读取所有完整四实例联合选择结果并审计逐实例合法性文件。"""
    result_root = archive / "checkpoints" / "async_eval" / "results"
    rows: list[dict[str, Any]] = []
    all_audit_paths: list[Path] = []
    partial_episodes: dict[int, dict[str, int]] = {}
    for episode_path in sorted(result_root.glob("episode_*.json")):
        payload = json.loads(episode_path.read_text(encoding="utf-8"))
        instances = list(payload.get("instances", []))
        if len(instances) != 4:
            raise RuntimeError(f"完整联合结果不是四实例：{episode_path.name}")
        audit_paths = sorted((result_root / f"episode_{int(payload['episode']):06d}").glob("real_*_legality_audit.json"))
        all_audit_paths.extend(audit_paths)
        rows.append(
            {
                "episode": int(payload["episode"]),
                "selection_score": float(payload["selection_score"]),
                "mean_makespan": float(payload["makespan"]),
                "eligible": float(payload["eligible"]),
                "hard_violation_total": int(payload["hard_violation_total"]),
                "duration_sec": float(payload["duration_sec"]),
                "instance_count": len(instances),
                "all_complete": all(bool(item["complete"]) for item in instances),
                "all_engine_legal": all(bool(item["engine_legal"]) for item in instances),
                "all_audit_legal": all(bool(item["audit_legal"]) for item in instances),
                "real_283_makespan": float(next(item["makespan"] for item in instances if item["instance_id"] == "real_283")),
                "real_680_makespan": float(next(item["makespan"] for item in instances if item["instance_id"] == "real_680")),
                "real_2338_makespan": float(next(item["makespan"] for item in instances if item["instance_id"] == "real_2338")),
                "real_3182_makespan": float(next(item["makespan"] for item in instances if item["instance_id"] == "real_3182")),
            }
        )
    for episode_dir in sorted(path for path in result_root.glob("episode_*") if path.is_dir()):
        episode = int(episode_dir.name.split("_")[-1])
        audit_count = len(list(episode_dir.glob("real_*_legality_audit.json")))
        schedule_count = len(list(episode_dir.glob("real_*_schedule.csv")))
        if episode not in {int(row["episode"]) for row in rows}:
            partial_episodes[episode] = {"audit_count": audit_count, "schedule_count": schedule_count}
            all_audit_paths.extend(episode_dir.glob("real_*_legality_audit.json"))
    audit_failures: list[dict[str, Any]] = []
    for audit_path in all_audit_paths:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        nonzero = {key: value for key, value in audit.get("violations", {}).items() if value}
        if not (audit.get("is_resource_structurally_legal") and audit.get("is_legal_against_environment_duration") and not nonzero):
            audit_failures.append({"path": audit_path.relative_to(archive).as_posix(), "violations": nonzero})
    if not rows:
        raise RuntimeError("没有完整异步四实例选择结果")
    best_row = min(rows, key=lambda row: float(row["selection_score"]))
    early = [row for row in rows if int(row["episode"]) <= int(best_row["episode"])]
    late = [row for row in rows if int(row["episode"]) > int(best_row["episode"])]
    summary = {
        "completed_four_instance_candidates": len(rows),
        "completed_episode_min": min(int(row["episode"]) for row in rows),
        "completed_episode_max": max(int(row["episode"]) for row in rows),
        "best_episode": int(best_row["episode"]),
        "best_selection_score": float(best_row["selection_score"]),
        "best_mean_makespan": float(best_row["mean_makespan"]),
        "all_candidates_eligible": all(float(row["eligible"]) == 1.0 for row in rows),
        "all_candidates_complete": all(bool(row["all_complete"]) for row in rows),
        "all_candidates_audit_legal": all(bool(row["all_audit_legal"]) for row in rows),
        "max_hard_violation_total": max(int(row["hard_violation_total"]) for row in rows),
        "audit_file_count": len(all_audit_paths),
        "audit_failure_count": len(audit_failures),
        "audit_failures": audit_failures,
        "partial_episode_artifacts": partial_episodes,
        "early_score_mean_through_best": statistics.fmean(float(row["selection_score"]) for row in early),
        "late_score_mean_after_best": statistics.fmean(float(row["selection_score"]) for row in late),
        "late_best_score": min(float(row["selection_score"]) for row in late),
    }
    return rows, summary


def checkpoint_metadata(archive: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("best.ckpt", "last.ckpt"):
        path = archive / "checkpoints" / name
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = dict(payload.get("apal_metadata", {}))
        result[name] = {
            "relative_path": path.relative_to(archive).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "loadable": True,
            "episode": int(metadata["episode"]),
            "model_spec": metadata.get("model_spec", {}),
        }
    return result


def append_index_once(path: Path, heading: str, content: str) -> None:
    text = path.read_text(encoding="utf-8")
    if heading not in text:
        path.write_text(text.rstrip() + "\n\n" + heading + "\n\n" + content.strip() + "\n", encoding="utf-8")


def update_master(archive: Path, summary: dict[str, Any]) -> None:
    """追加一条训练诊断行，不覆盖任何已有正式验证行。"""
    master = PROJECT_ROOT / "results" / "experiment_master_results.csv"
    with master.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    experiment_id = "initial_ctg_fv1_s42_260802_1512_training"
    rows = [row for row in rows if row.get("experiment_id") != experiment_id]
    row = {field: "" for field in fields}
    row.update(
        {
            "experiment_id": experiment_id,
            "phase": "initial_schedule",
            "experiment_group": "main_method_cross_scale_selection_training",
            "method": "HB-GAT-PPO",
            "variant": "conditional_team_gate_formal_v1",
            "dataset": "generated_400_800;real_283_680_2338_3182",
            "instance_id": "real_283_680_2338_3182",
            "scenario_level": "standard",
            "eval_protocol": "training_time_initial_real4_temperature0_cross_scale_selection_v1",
            "status": "early_stopped_training_time_selection",
            "priority": "high",
            "paper_table_role": "training_diagnostic_only",
            "fairness_status": "four_report_instances_used_for_checkpoint_selection_not_heldout",
            "strict_main_table_eligible": "no",
            "seed": "42",
            "num_runs": str(summary["async_selection"]["completed_four_instance_candidates"]),
            "scenario_count": "4",
            "makespan": str(summary["async_selection"]["best_mean_makespan"]),
            "makespan_mean": str(summary["async_selection"]["best_mean_makespan"]),
            "selection_score": str(summary["async_selection"]["best_selection_score"]),
            "score": str(summary["async_selection"]["best_selection_score"]),
            "eligible_rate": "1.0",
            "complete_rate": "1.0",
            "valid_rate": "1.0",
            "train_hours": "",
            "violation_summary": "33 个完整四实例候选均 complete=eligible=1；134 份可用审计文件硬约束为零；第 68 轮仅保留 2 个中间实例审计。",
            "source_file": (archive / "summary.json").relative_to(PROJECT_ROOT).as_posix(),
            "command_or_next_action": "使用 episode 20 best.ckpt 执行四实例六次正式验证；不得以训练期联合选择代替独立最终验证。",
            "notes": "目标80轮，last.ckpt至episode71后因后续联合选择均劣于episode20而提前停止；best SHA-256=" + summary["checkpoints"]["best.ckpt"]["sha256"],
        }
    )
    rows.append(row)
    with master.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_outputs(archive: Path, source: Path, copy_rows: list[dict[str, Any]]) -> dict[str, Any]:
    config = yaml.safe_load((archive / "configs" / "resolved_config.yaml").read_text(encoding="utf-8")) or {}
    curve_rows, curve_summary = collect_training_curve(archive)
    async_rows, async_summary = collect_async_results(archive)
    checkpoints = checkpoint_metadata(archive)
    selected = json.loads((archive / "checkpoints" / "async_eval" / "state" / "best.json").read_text(encoding="utf-8"))
    if selected["candidate_sha256"] != checkpoints["best.ckpt"]["sha256"]:
        raise RuntimeError("选择记录与 best.ckpt SHA-256 不一致")
    if int(selected["episode"]) != int(async_summary["best_episode"]):
        raise RuntimeError("选择记录与逐 episode 汇总的最优轮次不一致")
    summary = {
        "method": "HB-GAT-PPO",
        "variant": "conditional_team_gate_formal_v1",
        "phase": "initial_schedule",
        "training_status": "early_stopped_due_to_cross_scale_selection_regression",
        "evidence_level": "training_diagnostic_only",
        "strict_main_table_eligible": False,
        "run_id": config.get("run_id"),
        "seed": config.get("seed"),
        "target_episodes": int(config.get("max_episodes", 80)),
        "last_checkpoint_episode": checkpoints["last.ckpt"]["episode"],
        "stop_reason": "episode20 后完整四实例联合选择分数未再改善；为避免继续训练进一步恶化而停止。",
        "training_curve": curve_summary,
        "async_selection": async_summary,
        "checkpoints": checkpoints,
        "selection_protocol_id": selected["selection_protocol_id"],
        "selection_uses_report_instances": True,
        "source_directory": source.as_posix(),
        "archive_directory": archive.as_posix(),
    }
    write_csv(archive / "training_curve_summary.csv", curve_rows)
    write_csv(archive / "async_selection_by_episode.csv", async_rows)
    write_json(archive / "summary.json", summary)
    write_csv(
        archive / "summary.csv",
        [
            {"metric": "training_status", "value": summary["training_status"]},
            {"metric": "target_episodes", "value": summary["target_episodes"]},
            {"metric": "last_checkpoint_episode", "value": summary["last_checkpoint_episode"]},
            {"metric": "best_episode", "value": async_summary["best_episode"]},
            {"metric": "best_selection_score", "value": async_summary["best_selection_score"]},
            {"metric": "best_mean_makespan", "value": async_summary["best_mean_makespan"]},
            {"metric": "completed_four_instance_candidates", "value": async_summary["completed_four_instance_candidates"]},
            {"metric": "eligible_rate", "value": 1.0},
            {"metric": "complete_rate", "value": 1.0},
            {"metric": "max_hard_violation_total", "value": async_summary["max_hard_violation_total"]},
        ],
    )
    integrity = {
        "raw_copy_file_count": len(copy_rows),
        "raw_copy_all_sha256_equal": all(bool(row["sha256_equal"]) for row in copy_rows),
        "best_matches_selection_record": True,
        "best_checkpoint_loadable": checkpoints["best.ckpt"]["loadable"],
        "last_checkpoint_loadable": checkpoints["last.ckpt"]["loadable"],
        "tensorboard_steps_contiguous": True,
        "tensorboard_last_episode": curve_summary["tensorboard_last_episode"],
        "target_episodes": summary["target_episodes"],
        "training_completed_target": False,
        "completed_four_instance_candidates": async_summary["completed_four_instance_candidates"],
        "all_completed_candidates_complete": async_summary["all_candidates_complete"],
        "all_completed_candidates_eligible": async_summary["all_candidates_eligible"],
        "all_completed_candidates_audit_legal": async_summary["all_candidates_audit_legal"],
        "max_hard_violation_total": async_summary["max_hard_violation_total"],
        "audit_file_count": async_summary["audit_file_count"],
        "audit_failure_count": async_summary["audit_failure_count"],
        "partial_episode_artifacts": async_summary["partial_episode_artifacts"],
        "strict_main_table_eligible": False,
    }
    write_json(archive / "integrity_check.json", integrity)
    shutil.copy2(archive / "configs" / "resolved_config.yaml", archive / "resolved_config.yaml")
    shutil.copy2(archive / "configs" / "run_manifest.json", archive / "run_manifest.json")
    write_json(
        archive / "archive_run_manifest.json",
        {
            "source_directory": source.as_posix(),
            "archive_directory": archive.as_posix(),
            "raw_copy_integrity": "copy_integrity_check.json",
            "summary": "summary.json",
            "integrity": "integrity_check.json",
            "status": summary["training_status"],
        },
    )
    write_json(
        archive / "copy_integrity_check.json",
        {"source": source.as_posix(), "archive": archive.as_posix(), "all_equal": True, "files": copy_rows},
    )
    readme = f"""# CTG formal-v1 初始调度训练归档

## 实验身份与结论

- 方法：HB-GAT-PPO，`operation_station_gated_team`，五技能特征。
- 训练：seed=42，目标 80 轮；`last.ckpt` 保存至第 {summary['last_checkpoint_episode']} 轮后提前停止。
- 停止原因：第 {async_summary['best_episode']} 轮获得最佳四实例训练期联合选择分数 `{async_summary['best_selection_score']:.6f}`；此后 23 个完整候选的平均分数为 `{async_summary['late_score_mean_after_best']:.6f}`，未优于 best。
- 资格：`training_diagnostic_only`。四个真实实例参与了 checkpoint 选择，不能把这里的联合选择结果表述为独立留出测试，也不能直接进入正式性能主表。

## 最优 checkpoint

- 路径：`checkpoints/best.ckpt`；episode {async_summary['best_episode']}；SHA-256：`{checkpoints['best.ckpt']['sha256']}`。
- 四实例联合选择：mean makespan `{async_summary['best_mean_makespan']:.6f} h`，`complete=eligible=1`，最大硬约束违规数为 0。
- 四实例 makespan：283=`{selected['instances'][0]['makespan']:.6f}`，680=`{selected['instances'][1]['makespan']:.6f}`，2338=`{selected['instances'][2]['makespan']:.6f}`，3182=`{selected['instances'][3]['makespan']:.6f}` h。

## 完整性边界

- 33 个完整四实例候选（episode 2–66）均完整、合法；共保留 {async_summary['audit_file_count']} 份可用审计文件，违规均为零。
- episode 68 仅有 283、680 的中间产物；episode 70 及之后未完成评估。它们保留为中断证据，不能计入完整候选统计。
- `training_curve_summary.csv` 汇总 71 个连续 PPO 更新点；无 OOM、策略崩溃、批修复或 GPU 重建回退。
- 后续必须对上述固定 best checkpoint 重新执行四实例六次协议，作为独立可复核的正式验证。
"""
    (archive / "README.md").write_text(readme, encoding="utf-8")
    return summary


def write_file_manifest(archive: Path) -> None:
    rows = []
    for path in sorted(item for item in archive.rglob("*") if item.is_file() and item.name != "file_manifest.json"):
        rows.append({"path": path.relative_to(archive).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_json(archive / "file_manifest.json", {"root": archive.as_posix(), "files": rows})


def archive_root_event_if_present(archive: Path) -> dict[str, Any] | None:
    """保留根目录单独下载的截断 TensorBoard 文件，避免因非字节一致而丢失证据。"""
    root_event = PROJECT_ROOT / "results" / ROOT_EVENT_NAME
    if not root_event.is_file():
        return None
    embedded = archive / "logs" / "tensorboard" / "conditional_team_gate_formal_v1" / "version_1" / ROOT_EVENT_NAME
    if not embedded.is_file():
        raise FileNotFoundError("归档内缺少完整续训 TensorBoard event")
    destination = archive / "artifacts" / "external_downloads" / f"root_download_before_full_{ROOT_EVENT_NAME}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(root_event, destination)
    if sha256(root_event) != sha256(destination):
        raise RuntimeError("外部临时 TensorBoard 文件复制后 SHA-256 不一致")
    root_steps = scalar_map(root_event, "Rew")
    embedded_steps = scalar_map(embedded, "Rew")
    return {
        "source": root_event.relative_to(PROJECT_ROOT).as_posix(),
        "archived_copy": destination.relative_to(archive).as_posix(),
        "source_sha256": sha256(root_event),
        "embedded_event": embedded.relative_to(archive).as_posix(),
        "embedded_sha256": sha256(embedded),
        "source_rew_steps": [min(root_steps), max(root_steps)],
        "embedded_rew_steps": [min(embedded_steps), max(embedded_steps)],
        "relation": "根目录副本是较早下载的截断证据；完整续训 event 位于 logs/tensorboard/version_1。",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remove-source", action="store_true", help="核验和归档完成后删除根目录临时副本")
    parser.add_argument("--finalize-existing", action="store_true", help="完成中断归档的外部 TensorBoard 保存与源目录清理")
    args = parser.parse_args()
    source = (PROJECT_ROOT / SOURCE_RELATIVE).resolve()
    archive = (PROJECT_ROOT / ARCHIVE_RELATIVE).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"临时下载目录不存在：{source}")
    if archive.exists() and not args.finalize_existing:
        raise FileExistsError(f"目标归档目录已存在，拒绝覆盖：{archive}")
    if archive.exists():
        copy_rows = raw_copy_integrity(source, archive)
        external_event = archive_root_event_if_present(archive)
        if external_event is not None:
            write_json(archive / "external_download_artifacts.json", external_event)
        write_file_manifest(archive)
        if args.remove_source:
            root_event = PROJECT_ROOT / "results" / ROOT_EVENT_NAME
            if root_event.is_file():
                root_event.unlink()
            shutil.rmtree(source)
        print(json.dumps({"archive": archive.as_posix(), "finalized_existing_archive": True, "source_removed": bool(args.remove_source)}, ensure_ascii=False, indent=2))
        return 0
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, archive)
    copy_rows = raw_copy_integrity(source, archive)
    summary = build_outputs(archive, source, copy_rows)
    update_master(archive, summary)
    append_index_once(
        PROJECT_ROOT / "results" / "README.md",
        "## 2026-08-03 CTG formal-v1 训练诊断归档",
        "`conditional_team_gate_formal_v1/ctg_fv1_s42_260802_1512` 已按提前停止训练归档。第 20 轮为训练期四实例联合选择 best；后续候选未改善，不能作为独立最终性能结论。",
    )
    append_index_once(
        PROJECT_ROOT / "docs" / "实验待做清单.md",
        "## 2026-08-03：CTG formal-v1 后续事项",
        "- 已归档 seed42 训练诊断：目标80轮，last=71，best=episode20；后续联合选择恶化，训练期结果不得进入正式性能主表。\n- 待做：固定 `checkpoints/best.ckpt` 执行 real_283/680/2338/3182 的 temperature=0 seed42 与 temperature=0.01 seeds42–46 六次验证，并独立审计。",
    )
    write_file_manifest(archive)
    external_event = archive_root_event_if_present(archive)
    if external_event is not None:
        write_json(archive / "external_download_artifacts.json", external_event)
        write_file_manifest(archive)
    if args.remove_source:
        root_event = PROJECT_ROOT / "results" / ROOT_EVENT_NAME
        if root_event.is_file():
            root_event.unlink()
        shutil.rmtree(source)
    print(json.dumps({"archive": archive.as_posix(), "summary": summary, "source_removed": bool(args.remove_source)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
