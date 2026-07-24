"""解析并整理最新版本初始调度结构消融训练结果。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scalars(event_path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalars: dict[str, list[dict[str, Any]]] = {}
    for tag in accumulator.Tags().get("scalars", []):
        scalars[tag] = [
            {
                "step": int(item.step),
                "value": float(item.value),
                "wall_time": float(item.wall_time),
            }
            for item in accumulator.Scalars(tag)
        ]
    return scalars, accumulator.Tags()


def scalar_table(scalars: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    rows: dict[int, dict[str, Any]] = {}
    for tag, values in scalars.items():
        column = tag.replace("/", "_").replace(" ", "_")
        for item in values:
            rows.setdefault(item["step"], {"step": item["step"]})[column] = item[
                "value"
            ]
    return pd.DataFrame(sorted(rows.values(), key=lambda row: row["step"]))


def value_at_last(scalars: dict[str, list[dict[str, Any]]], tag: str) -> float | None:
    values = scalars.get(tag, [])
    return values[-1]["value"] if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--converged-before-target",
        action="store_true",
        help="将用户确认的提前稳定训练记录为收敛训练终点",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    archive = args.archive.resolve()
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(source)
    if not archive.exists() or not archive.is_dir():
        raise FileNotFoundError(archive)

    config_path = archive / "configs" / "resolved_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = archive / "configs" / "run_manifest.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    event_paths = sorted(archive.glob("logs/tensorboard/**/events.out.tfevents.*"))
    nonempty = [path for path in event_paths if path.stat().st_size > 100]
    if not nonempty:
        raise RuntimeError(f"没有找到有效 TensorBoard event: {archive}")
    event_path = max(nonempty, key=lambda path: path.stat().st_size)
    scalars, tags = load_scalars(event_path)
    curves = scalar_table(scalars)
    curves.to_csv(archive / "training_curves.csv", index=False, encoding="utf-8-sig")

    best_path = archive / "checkpoints" / "best.ckpt"
    last_path = archive / "checkpoints" / "last.ckpt"
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    last_checkpoint_error: str | None = None
    try:
        last_checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # 保留损坏 latest checkpoint 的原始证据，不阻断 best 解析
        last_checkpoint = {}
        last_checkpoint_error = f"{type(exc).__name__}: {exc}"
    best_metadata = best_checkpoint.get("apal_metadata", {})
    last_metadata = last_checkpoint.get("apal_metadata", {})
    callbacks = best_checkpoint.get("callbacks", {})
    rollout_callback = callbacks.get("RolloutCheckpoint", {})
    eval_values = scalars.get("Eval/makespan", [])
    rollout_values = scalars.get("Rollout/AverageMakespan", [])
    best_eval = min(eval_values, key=lambda item: item["value"]) if eval_values else None
    last_eval = eval_values[-1] if eval_values else None
    target_episodes = int(config.get("max_episodes", 100))
    observed_updates = len(rollout_values)
    checkpoint_episode = int(best_metadata.get("episode", 0))
    last_episode = int(last_metadata.get("episode", 0))
    variant = str(source.name).removeprefix("joint100_").removesuffix("_seed42")
    training_complete = observed_updates >= target_episodes and last_episode >= target_episodes
    training_status = (
        "completed_training"
        if training_complete
        else "converged_training"
        if args.converged_before_target
        else "partial_training"
    )
    summary = {
        "method": "HB-GAT-PPO",
        "phase": "initial_schedule_training",
        "experiment_group": "initial_ablation_training",
        "variant": variant,
        "run_id": source_manifest.get("run_id", source.name),
        "training_status": training_status,
        "converged_before_target": bool(args.converged_before_target and not training_complete),
        "source_directory": source.as_posix(),
        "archive_directory": archive.as_posix(),
        "dataset_training_pool": config.get("train_data_path_or_dir"),
        "training_instance": config.get("data_file_path"),
        "seed": config.get("seed"),
        "target_episodes": target_episodes,
        "observed_rollout_updates": observed_updates,
        "best_checkpoint_episode": checkpoint_episode,
        "last_checkpoint_episode": last_episode,
        "completion_ratio": observed_updates / target_episodes if target_episodes else 0.0,
        "best_checkpoint_sha256": sha256(best_path),
        "last_checkpoint_sha256": sha256(last_path),
        "best_checkpoint_loadable": True,
        "last_checkpoint_loadable": last_checkpoint_error is None,
        "last_checkpoint_load_error": last_checkpoint_error,
        "best_equals_last": best_path.read_bytes() == last_path.read_bytes(),
        "best_checkpoint_score": rollout_callback.get("best_score"),
        "best_eval": best_eval,
        "last_eval": last_eval,
        "last_rollout": rollout_values[-1] if rollout_values else None,
        "tensorboard_event": event_path.relative_to(archive).as_posix(),
        "tensorboard_event_size": event_path.stat().st_size,
        "tensorboard_scalar_tag_count": len(tags.get("scalars", [])),
        "git_commit": source_manifest.get("git_commit"),
        "server_run_dir": source_manifest.get("run_dir"),
        "config_key_checks": {
            key: config.get(key)
            for key in [
                "actor_context_mode",
                "policy_action_scope",
                "use_attention_critic",
                "use_autoregressive_worker",
                "team_selection_mode",
                "workforce_binding_mode",
                "hidden_dim",
                "num_gat_layers",
                "num_heads",
                "use_skill_hub",
                "skill_hub_bidirectional",
                "n_w",
                "n_m",
                "use_schedule_free",
                "lightning_precision",
                "num_envs",
                "max_episodes",
                "eval_temperature",
            ]
        },
        "formal_cross_scale_validation_present": False,
        "independent_legality_replay_present": False,
    }
    (archive / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary_rows = [
        {"metric": "target_episodes", "value": target_episodes},
        {"metric": "observed_rollout_updates", "value": observed_updates},
        {"metric": "completion_ratio", "value": summary["completion_ratio"]},
        {"metric": "best_checkpoint_episode", "value": checkpoint_episode},
        {"metric": "last_checkpoint_episode", "value": last_episode},
        {"metric": "best_eval_makespan", "value": best_eval["value"] if best_eval else None},
        {"metric": "last_eval_makespan", "value": last_eval["value"] if last_eval else None},
        {"metric": "last_rollout_average_makespan", "value": rollout_values[-1]["value"] if rollout_values else None},
        {"metric": "last_eval_completion_rate", "value": value_at_last(scalars, "Eval/completion_rate")},
        {"metric": "last_steps_per_second", "value": value_at_last(scalars, "SPS")},
        {"metric": "last_explained_variance", "value": value_at_last(scalars, "Critic/Explained_Variance")},
        {"metric": "last_approx_kl", "value": value_at_last(scalars, "Policy/ApproxKL")},
        {"metric": "last_clip_fraction", "value": value_at_last(scalars, "Policy/ClipFraction")},
        {"metric": "total_oom_skipped_update", "value": sum(item["value"] for item in scalars.get("OOM/SkippedUpdate", []))},
        {"metric": "total_policy_meltdown", "value": sum(item["value"] for item in scalars.get("Policy/Meltdown_Count", []))},
    ]
    pd.DataFrame(summary_rows).to_csv(archive / "training_summary.csv", index=False, encoding="utf-8-sig")

    raw_source_files = sorted(path for path in source.rglob("*") if path.is_file())
    copy_rows = []
    for source_file in raw_source_files:
        target = archive / source_file.relative_to(source)
        copy_rows.append(
            {
                "relative_path": source_file.relative_to(source).as_posix(),
                "source_size": source_file.stat().st_size,
                "archive_size": target.stat().st_size,
                "source_sha256": sha256(source_file),
                "archive_sha256": sha256(target),
                "sha256_equal": sha256(source_file) == sha256(target),
            }
        )
    (archive / "copy_integrity_check.json").write_text(
        json.dumps(
            {
                "source": source.as_posix(),
                "archive": archive.as_posix(),
                "raw_file_count": len(copy_rows),
                "all_equal": all(row["sha256_equal"] for row in copy_rows),
                "files": copy_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (archive / "resolved_config.yaml").write_bytes(config_path.read_bytes())
    (archive / "run_manifest.json").write_text(
        json.dumps(
            {
                **source_manifest,
                "archive_directory": archive.as_posix(),
                "training_summary": "training_summary.json",
                "training_status": summary["training_status"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    version_events = []
    for path in event_paths:
        accumulator = EventAccumulator(str(path))
        accumulator.Reload()
        version_events.append(
            {
                "path": path.relative_to(archive).as_posix(),
                "size": path.stat().st_size,
                "has_scalars": bool(accumulator.Tags().get("scalars")),
            }
        )
    (archive / "integrity_check.json").write_text(
        json.dumps(
            {
                "raw_copy_all_sha256_equal": all(row["sha256_equal"] for row in copy_rows),
                "raw_file_count": len(copy_rows),
                "tensorboard_events": version_events,
                "target_episodes": target_episodes,
                "observed_rollout_updates": observed_updates,
                "best_eval_matches_checkpoint_score": best_eval is not None
                and rollout_callback.get("best_score") is not None
                and abs(float(best_eval["value"]) - float(rollout_callback["best_score"])) < 1e-3,
                "training_complete": training_complete,
                "converged_before_target": bool(args.converged_before_target and not training_complete),
                "best_checkpoint_last_checkpoint_equal": summary["best_equals_last"],
                "best_checkpoint_loadable": summary["best_checkpoint_loadable"],
                "last_checkpoint_loadable": summary["last_checkpoint_loadable"],
                "last_checkpoint_load_error": summary["last_checkpoint_load_error"],
                "git_commit_verified": bool(source_manifest.get("git_commit")),
                "formal_cross_scale_validation_present": False,
                "independent_legality_replay_present": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    readme = f"""# HB-GAT-PPO {variant} 初始调度消融训练记录

## 结果身份

- 方法：HB-GAT-PPO 当前版本初始调度结构消融，变体为 `{variant}`。
- 运行 ID：`{summary['run_id']}`；seed=`{summary['seed']}`；训练目标 `{target_episodes}` episodes。
- 训练池：`{summary['dataset_training_pool']}`；训练期配置中的评估数据：`{summary['training_instance']}`。
- 该结果属于训练期诊断证据，不是四规模正式验证，也不能与旧版本消融结果混合。

## 训练状态

- TensorBoard 有效 rollout/update 点：`{observed_updates}`；best checkpoint episode：`{checkpoint_episode}`；last checkpoint episode：`{last_episode}`；完成比例：`{summary['completion_ratio']:.1%}`。
- 按独立 `Eval/makespan` 最小值选择 best：`{best_eval['value'] if best_eval else '未记录'} h`；最后一次 Eval：`{last_eval['value'] if last_eval else '未记录'} h`。
- best/last checkpoint 字节级相同：`{summary['best_equals_last']}`；best SHA-256：`{summary['best_checkpoint_sha256']}`；last SHA-256：`{summary['last_checkpoint_sha256']}`。
- best checkpoint 可读取：`{summary['best_checkpoint_loadable']}`；last checkpoint 可读取：`{summary['last_checkpoint_loadable']}`。若 latest 损坏，错误信息保存在 `training_summary.json` 和 `integrity_check.json`。
- 训练期最后 completion rate：`{value_at_last(scalars, 'Eval/completion_rate')}`；OOM skipped update 总数：`{sum(item['value'] for item in scalars.get('OOM/SkippedUpdate', []))}`；policy meltdown 总数：`{sum(item['value'] for item in scalars.get('Policy/Meltdown_Count', []))}`。

## 证据限制

- 当前状态为 `{summary['training_status']}` / `training_diagnostic_only`；不等同于机械完成 `{target_episodes}` episodes，也不进入正式质量主表。
- 没有独立排程合法性回放，训练期 completion rate 不等于 `eligible_rate` 或正式 valid rate。
- `eval/` 不包含四个真实规模的正式验证；后续应在确定训练停止点后，按统一多种子/确定性协议验证 `real_283/680/2338/3182`。
- 原始运行清单的 `git_commit` 为 `{summary['git_commit']}`，服务器代码 provenance 未完全锁定。

详细文件：`training_curves.csv`、`training_summary.csv/json`、`integrity_check.json`、`copy_integrity_check.json`、`file_manifest.json`。
"""
    (archive / "README.md").write_text(readme, encoding="utf-8")

    manifest_rows = []
    for path in sorted(archive.rglob("*")):
        if path.is_file() and path.name != "file_manifest.json":
            manifest_rows.append(
                {"path": path.relative_to(archive).as_posix(), "size": path.stat().st_size, "sha256": sha256(path)}
            )
    (archive / "file_manifest.json").write_text(
        json.dumps({"root": archive.as_posix(), "files": manifest_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
