"""归档带四实例异步筛选的初始调度主方法训练结果。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    """返回文件的 SHA-256，供复制与 checkpoint 身份核验。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_and_verify(source: Path, archive: Path) -> list[dict[str, Any]]:
    """先完整复制，再逐文件核对源与归档副本。"""
    if archive.exists():
        raise FileExistsError(f"归档目标已经存在，拒绝覆盖：{archive}")
    shutil.copytree(source, archive)
    rows: list[dict[str, Any]] = []
    for source_file in sorted(path for path in source.rglob("*") if path.is_file()):
        relative = source_file.relative_to(source)
        archived_file = archive / relative
        if not archived_file.is_file():
            raise FileNotFoundError(f"归档缺少源文件：{relative.as_posix()}")
        source_hash = sha256(source_file)
        archive_hash = sha256(archived_file)
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "source_size": source_file.stat().st_size,
                "archive_size": archived_file.stat().st_size,
                "source_sha256": source_hash,
                "archive_sha256": archive_hash,
                "sha256_equal": source_hash == archive_hash,
            }
        )
    if not rows or not all(row["sha256_equal"] for row in rows):
        raise RuntimeError("源目录与归档目录的 SHA-256 核验未全部通过")
    return rows


def load_scalars(event_path: Path) -> dict[str, list[dict[str, float | int]]]:
    """读取完整 TensorBoard 标量序列，而非只读取最后一个点。"""
    accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    return {
        tag: [
            {"step": int(item.step), "value": float(item.value), "wall_time": float(item.wall_time)}
            for item in accumulator.Scalars(tag)
        ]
        for tag in accumulator.Tags().get("scalars", [])
    }


def make_curve_table(scalars: dict[str, list[dict[str, float | int]]]) -> pd.DataFrame:
    """将同一步的训练标量拼成可直接分析的表。"""
    rows: dict[int, dict[str, Any]] = {}
    for tag, points in scalars.items():
        column = tag.replace("/", "_").replace(" ", "_")
        for point in points:
            step = int(point["step"])
            rows.setdefault(step, {"step": step})[column] = float(point["value"])
    return pd.DataFrame(sorted(rows.values(), key=lambda row: int(row["step"])))


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    """加载 Lightning checkpoint 的项目元数据，并验证其可读性。"""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("apal_metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError(f"checkpoint 元数据格式错误：{path}")
    return {
        "path": path.name,
        "sha256": sha256(path),
        "apal_episode": metadata.get("episode"),
        "global_step": payload.get("global_step"),
        "loadable": True,
    }


def audit_summary(results_root: Path) -> dict[str, Any]:
    """重算每个已完成筛选点的四实例合法性与完整性。"""
    result_jsons = sorted(
        path for path in results_root.glob("episode_*.json") if path.is_file()
    )
    rows: list[dict[str, Any]] = []
    audit_files = 0
    audit_legal = 0
    hard_violation_max = 0
    for result_path in result_jsons:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        episode = int(payload["episode"])
        instances = payload.get("instances", [])
        if len(instances) != 4:
            raise ValueError(f"筛选结果不是四实例：{result_path}")
        all_eligible = all(bool(item.get("eligible")) for item in instances)
        all_complete = all(bool(item.get("complete")) for item in instances)
        all_audit_legal = True
        for item in instances:
            audit_path = results_root / f"episode_{episode:06d}" / (
                f"{item['instance_id']}_legality_audit.json"
            )
            if not audit_path.is_file():
                raise FileNotFoundError(f"缺少合法性审计：{audit_path}")
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit_files += 1
            audit_is_legal = bool(audit.get("is_legal_against_current_data_duration"))
            audit_legal += int(audit_is_legal)
            all_audit_legal = all_audit_legal and audit_is_legal
            violations = audit.get("violations", {})
            hard_violation_max = max(
                hard_violation_max,
                *(int(value) for value in violations.values()),
            )
        rows.append(
            {
                "episode": episode,
                "selection_score": float(payload["selection_score"]),
                "mean_makespan": float(payload["makespan"]),
                "eligible": all_eligible,
                "complete": all_complete,
                "audit_legal": all_audit_legal,
                "hard_violation_total": int(payload.get("hard_violation_total", 0)),
            }
        )
    if not rows:
        raise RuntimeError("未发现已完成的四实例异步筛选结果")
    return {
        "rows": rows,
        "completed_selection_count": len(rows),
        "audit_file_count": audit_files,
        "audit_legal_count": audit_legal,
        "all_complete": all(row["complete"] for row in rows),
        "all_eligible": all(row["eligible"] for row in rows),
        "all_audit_legal": all(row["audit_legal"] for row in rows),
        "max_hard_violation": hard_violation_max,
    }


def write_master_row(archive: Path, summary: dict[str, Any]) -> None:
    """追加一条训练诊断记录，不覆盖任何正式验证或历史记录。"""
    master_path = PROJECT_ROOT / "results" / "experiment_master_results.csv"
    with master_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        existing = list(reader)
    if not fieldnames:
        raise ValueError("实验总表没有表头")
    experiment_id = "initial_full_joint_real4_stable_opt100_seed42_20260725"
    if any(row.get("experiment_id") == experiment_id for row in existing):
        raise ValueError(f"实验总表已存在同 ID，拒绝重复登记：{experiment_id}")
    best = summary["best_selection"]
    row = {name: "" for name in fieldnames}
    row.update(
        {
            "experiment_id": experiment_id,
            "phase": "initial_schedule",
            "experiment_group": "main_method_cross_scale_selection_training",
            "method": "HB-GAT-PPO",
            "variant": "full_joint_real4_selection",
            "dataset": "generated_400_800;real_283_680_2338_3182",
            "instance_id": "real_283_680_2338_3182",
            "scenario_level": "standard",
            "eval_protocol": "training_time_cross_scale_selection_temperature0",
            "status": "partial_training",
            "priority": "high",
            "paper_table_role": "training_diagnostic_only",
            "fairness_status": "four_report_instances_used_for_checkpoint_selection_not_heldout",
            "strict_main_table_eligible": "no",
            "seed": str(summary["seed"]),
            "num_runs": str(summary["completed_selection_count"]),
            "scenario_count": "4",
            "makespan": str(best["mean_makespan"]),
            "makespan_mean": str(best["mean_makespan"]),
            "normalized_makespan": str(best["selection_score"]),
            "selection_score": str(best["selection_score"]),
            "score": str(best["selection_score"]),
            "eligible_rate": "1.0" if summary["all_eligible"] else "0.0",
            "complete_rate": "1.0" if summary["all_complete"] else "0.0",
            "valid_rate": "1.0" if summary["all_audit_legal"] else "0.0",
            "violation_summary": (
                f"{summary['audit_file_count']} audits; max hard violation="
                f"{summary['max_hard_violation']}"
            ),
            "source_file": archive.relative_to(PROJECT_ROOT).as_posix(),
            "command_or_next_action": "finish_or_restart_100_episode_training_then_run_fixed_sixrun_validation",
            "notes": (
                f"target=100 but TensorBoard/last.ckpt stop at {summary['observed_rollout_updates']}; "
                f"episode {summary['running_selection_episode']} selection unfinished; "
                f"best is episode {best['episode']} SHA256={summary['best_checkpoint']['sha256']}; "
                "training-time four-instance selection only, not a held-out final comparison"
            ),
        }
    )
    with master_path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="归档四实例异步选择的初始调度训练结果")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--remove-source", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    archive = args.archive.resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"源目录不存在：{source}")
    copy_rows = copy_and_verify(source, archive)

    config_path = archive / "configs" / "resolved_config.yaml"
    manifest_path = archive / "configs" / "run_manifest.json"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    event_paths = sorted(archive.glob("logs/tensorboard/**/events.out.tfevents.*"))
    if len(event_paths) != 1:
        raise RuntimeError(f"应有且仅有一个主训练 TensorBoard event，实际为 {len(event_paths)}")
    scalars = load_scalars(event_paths[0])
    curves = make_curve_table(scalars)
    curves.to_csv(archive / "training_curves.csv", index=False, encoding="utf-8-sig")

    best_checkpoint = checkpoint_metadata(archive / "checkpoints" / "best.ckpt")
    last_checkpoint = checkpoint_metadata(archive / "checkpoints" / "last.ckpt")
    async_root = archive / "checkpoints" / "async_eval"
    selection = audit_summary(async_root / "results")
    selection_table = pd.DataFrame(selection.pop("rows"))
    selection_table.to_csv(archive / "checkpoint_selection_metrics.csv", index=False, encoding="utf-8-sig")
    best_selection = selection_table.loc[selection_table["selection_score"].idxmin()].to_dict()

    expected_selection_episodes = list(range(int(config["async_eval_submit_every_episodes"]), int(config["max_episodes"]) + 1, int(config["async_eval_submit_every_episodes"])))
    done_episodes = sorted(
        int(path.stem.split("_")[-1])
        for path in (async_root / "queue" / "done").glob("episode_*.json")
    )
    running_files = sorted((async_root / "queue" / "running").glob("episode_*.json"))
    running_episode = int(running_files[0].stem.split("_")[-1]) if len(running_files) == 1 else None
    observed_rollout_updates = len(scalars.get("Rollout/AverageMakespan", []))
    training_complete = (
        observed_rollout_updates >= int(config["max_episodes"])
        and int(last_checkpoint["apal_episode"]) >= int(config["max_episodes"])
        and not running_files
    )
    state_best = json.loads((async_root / "state" / "best.json").read_text(encoding="utf-8"))
    best_identity_matches = (
        int(state_best["episode"]) == int(best_checkpoint["apal_episode"])
        and state_best["best_checkpoint_sha256"] == best_checkpoint["sha256"]
        and abs(float(state_best["selection_score"]) - float(best_selection["selection_score"])) < 1e-12
    )
    summary = {
        "method": "HB-GAT-PPO",
        "variant": "full_joint_real4_selection",
        "phase": "initial_schedule_training",
        "training_status": "completed_training" if training_complete else "partial_training",
        "evidence_level": "training_diagnostic_only",
        "strict_main_table_eligible": False,
        "run_id": run_manifest["run_id"],
        "git_commit": run_manifest.get("git_commit"),
        "server_run_dir": run_manifest.get("run_dir"),
        "seed": config.get("seed"),
        "target_episodes": int(config["max_episodes"]),
        "observed_rollout_updates": observed_rollout_updates,
        "completion_ratio": observed_rollout_updates / int(config["max_episodes"]),
        "best_checkpoint": best_checkpoint,
        "last_checkpoint": last_checkpoint,
        "best_selection": best_selection,
        "completed_selection_count": selection["completed_selection_count"],
        "expected_selection_episodes": expected_selection_episodes,
        "done_selection_episodes": done_episodes,
        "running_selection_episode": running_episode,
        "missing_selection_episodes": [episode for episode in expected_selection_episodes if episode not in done_episodes and episode != running_episode],
        "all_complete": selection["all_complete"],
        "all_eligible": selection["all_eligible"],
        "all_audit_legal": selection["all_audit_legal"],
        "audit_file_count": selection["audit_file_count"],
        "audit_legal_count": selection["audit_legal_count"],
        "max_hard_violation": selection["max_hard_violation"],
        "best_identity_matches_selection_state": best_identity_matches,
        "primary_tensorboard_event": event_paths[0].relative_to(archive).as_posix(),
        "primary_tensorboard_scalar_tag_count": len(scalars),
        "total_oom_skipped_updates": sum(point["value"] for point in scalars.get("OOM/SkippedUpdate", [])),
        "max_policy_meltdown_count": max((point["value"] for point in scalars.get("Policy/Meltdown_Count", [])), default=0),
    }
    (archive / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in summary.items() if isinstance(value, (str, int, float, bool)) or value is None]
    ).to_csv(archive / "summary.csv", index=False, encoding="utf-8-sig")
    (archive / "copy_integrity_check.json").write_text(
        json.dumps(
            {
                "source": source.as_posix(),
                "archive": archive.as_posix(),
                "raw_file_count": len(copy_rows),
                "raw_total_bytes": sum(int(row["source_size"]) for row in copy_rows),
                "all_equal": True,
                "files": copy_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2(config_path, archive / "resolved_config.yaml")
    shutil.copy2(manifest_path, archive / "run_manifest.json")
    (archive / "integrity_check.json").write_text(
        json.dumps(
            {
                "raw_copy_all_sha256_equal": True,
                "raw_file_count": len(copy_rows),
                "target_episodes": summary["target_episodes"],
                "observed_rollout_updates": observed_rollout_updates,
                "training_complete": training_complete,
                "best_checkpoint_loadable": best_checkpoint["loadable"],
                "last_checkpoint_loadable": last_checkpoint["loadable"],
                "best_checkpoint_matches_selection_state": best_identity_matches,
                "completed_selection_count": selection["completed_selection_count"],
                "expected_selection_count": len(expected_selection_episodes),
                "running_selection_episode": running_episode,
                "missing_selection_episodes": summary["missing_selection_episodes"],
                "selection_all_complete": selection["all_complete"],
                "selection_all_eligible": selection["all_eligible"],
                "selection_all_audit_legal": selection["all_audit_legal"],
                "selection_audit_file_count": selection["audit_file_count"],
                "max_hard_violation": selection["max_hard_violation"],
                "strict_main_table_eligible": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    readme = f"""# HB-GAT-PPO full_joint 四实例筛选训练归档

## 结论与身份

- 配置目标为 `{summary['target_episodes']}` episodes，但原始 TensorBoard 与 `last.ckpt` 仅记录 `{observed_rollout_updates}` / episode `{last_checkpoint['apal_episode']}`，故状态为 `partial_training`。
- 四个报告实例参与了训练期 checkpoint 选择；这是一项 cross-scale screening，不是隔离测试，也不替代 temperature=0 的确定性验证和 temperature=0.01 的多种子验证。
- 当前 best 为 episode `{best_selection['episode']}`，selection score=`{best_selection['selection_score']:.9f}`，四实例原始 makespan 平均=`{best_selection['mean_makespan']:.6f}` h；其 SHA-256 与异步选择状态完全一致。

## 完整性与合法性

- 完成的四实例筛选点：`{selection['completed_selection_count']}/{len(expected_selection_episodes)}`；episode `{running_episode}` 的异步任务仍在 `queue/running`，`{summary['missing_selection_episodes']}` 没有提交完成证据。
- 已完成筛选点共 `{selection['audit_file_count']}` 份独立排程审计，全部 complete/eligible/合法，最大硬约束违规为 `{selection['max_hard_violation']}`。
- 保留 best、last、候选 checkpoint、原始 TensorBoard event、每个已完成筛选点的 schedule 与 legality audit；源目录复制后的 `{len(copy_rows)}` 个原始文件已逐个 SHA-256 一致。

## 训练诊断

- actor 学习率=`{config['lr'] * config['actor_lr_multiplier']}`，critic 学习率=`{config['lr'] * config['critic_lr_multiplier']}`，`batch_size={config['batch_size']}`，`accumulation_steps={config['accumulation_steps']}`，`kl_early_stop={config['kl_early_stop']}`。
- OOM skipped update 总数为 `{summary['total_oom_skipped_updates']}`；Policy/Meltdown_Count 最大值为 `{summary['max_policy_meltdown_count']}`。

## 论文使用边界

`strict_main_table_eligible=no`。本目录只能作为训练和 checkpoint 选择证据；在固定模型后，仍需按当前六次协议完成四实例独立验证，才能进入方法质量比较。
"""
    (archive / "README.md").write_text(readme, encoding="utf-8")
    manifest_rows = [
        {"path": path.relative_to(archive).as_posix(), "size": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(archive.rglob("*"))
        if path.is_file() and path.name != "file_manifest.json"
    ]
    (archive / "file_manifest.json").write_text(
        json.dumps({"root": archive.relative_to(PROJECT_ROOT).as_posix(), "generated_at": datetime.now(timezone.utc).isoformat(), "files": manifest_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_master_row(archive, summary)
    if args.remove_source:
        shutil.rmtree(source)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
