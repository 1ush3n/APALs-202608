"""解析并归档 joint100_full_joint_seed42 的未完成主方法训练结果。"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results" / "01_initial_main" / "joint100_full_joint_seed42_20260719"
SOURCE = ROOT / "results" / "joint100_full_joint_seed42"


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
            {"step": int(item.step), "value": float(item.value), "wall_time": float(item.wall_time)}
            for item in accumulator.Scalars(tag)
        ]
    return scalars, accumulator.Tags()


def scalar_table(scalars: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    rows: dict[int, dict[str, Any]] = {}
    for tag, values in scalars.items():
        column = tag.replace("/", "_").replace(" ", "_")
        for item in values:
            rows.setdefault(item["step"], {"step": item["step"]})[column] = item["value"]
    return pd.DataFrame(sorted(rows.values(), key=lambda row: row["step"]))


def main() -> int:
    archive = ARCHIVE
    event_paths = sorted(archive.glob("logs/tensorboard/**/events.out.tfevents.*"))
    nonempty = [path for path in event_paths if path.stat().st_size > 100]
    if not nonempty:
        raise RuntimeError("没有找到可解析的非空 TensorBoard event")
    event_path = max(nonempty, key=lambda path: path.stat().st_size)
    scalars, tags = load_scalars(event_path)
    curves = scalar_table(scalars)
    curves.to_csv(archive / "training_curves.csv", index=False, encoding="utf-8-sig")

    checkpoint_path = archive / "checkpoints" / "best.ckpt"
    last_path = archive / "checkpoints" / "last.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    last_checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("apal_metadata", {})
    config = metadata.get("config", {})
    episode = int(metadata.get("episode", 0))
    callback = checkpoint.get("callbacks", {})
    rollout_callback = next(
        (value for key, value in callback.items() if key == "RolloutCheckpoint"), {}
    )
    eval_values = scalars.get("Eval/makespan", [])
    rollout_values = scalars.get("Rollout/AverageMakespan", [])
    best_eval = min(eval_values, key=lambda item: item["value"]) if eval_values else None
    last_eval = eval_values[-1] if eval_values else None
    version0_events = [
        {
            "path": str(path.relative_to(archive)).replace("\\", "/"),
            "size": path.stat().st_size,
            "has_scalars": bool(EventAccumulator(str(path)).Reload().Tags().get("scalars")),
        }
        for path in event_paths
        if "version_0" in path.as_posix()
    ]
    observed_updates = len(rollout_values)
    target_episodes = int(config.get("max_episodes", 100))
    summary = {
        "method": "HB-GAT-PPO",
        "variant": "full_joint",
        "run_id": "joint100_full_joint_seed42",
        "phase": "initial_schedule_training",
        "training_status": "partial_training",
        "source_directory": "results/joint100_full_joint_seed42",
        "archive_directory": "results/01_initial_main/joint100_full_joint_seed42_20260719",
        "dataset_training_pool": "data/scale_400_800_datasets",
        "training_instance": "data/680.csv",
        "seed": 42,
        "target_episodes": target_episodes,
        "observed_rollout_updates": observed_updates,
        "checkpoint_episode": episode,
        "completion_ratio": observed_updates / target_episodes if target_episodes else 0.0,
        "checkpoint_sha256": sha256(checkpoint_path),
        "last_checkpoint_sha256": sha256(last_path),
        "best_equals_last": checkpoint_path.read_bytes() == last_path.read_bytes(),
        "best_checkpoint_score": rollout_callback.get("best_score"),
        "best_eval": best_eval,
        "last_eval": last_eval,
        "last_rollout": rollout_values[-1] if rollout_values else None,
        "tensorboard_event": str(event_path.relative_to(archive)).replace("\\", "/"),
        "tensorboard_event_size": event_path.stat().st_size,
        "tensorboard_scalar_tag_count": len(tags.get("scalars", [])),
        "tensorboard_version0_events": version0_events,
        "git_commit": None,
        "server_run_dir": "/root/autodl-tmp/v2/runs/scale_400_800_schedule/joint100_full_joint_seed42",
        "config_key_checks": {
            key: config.get(key)
            for key in [
                "hidden_dim",
                "num_gat_layers",
                "num_heads",
                "use_skill_hub",
                "skill_hub_bidirectional",
                "use_autoregressive_worker",
                "team_selection_mode",
                "workforce_binding_mode",
                "n_w",
                "n_m",
                "use_schedule_free",
                "lightning_precision",
                "num_envs",
                "max_episodes",
                "eval_temperature",
            ]
        },
    }
    (archive / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        [
            {"metric": "target_episodes", "value": target_episodes},
            {"metric": "observed_rollout_updates", "value": observed_updates},
            {"metric": "completion_ratio", "value": summary["completion_ratio"]},
            {"metric": "checkpoint_episode", "value": episode},
            {"metric": "best_eval_makespan", "value": best_eval["value"] if best_eval else None},
            {"metric": "last_eval_makespan", "value": last_eval["value"] if last_eval else None},
            {"metric": "last_rollout_average_makespan", "value": rollout_values[-1]["value"] if rollout_values else None},
            {"metric": "last_eval_completion_rate", "value": scalars.get("Eval/completion_rate", [{}])[-1].get("value")},
            {"metric": "last_rollout_completion_rate", "value": scalars.get("Rollout/CompletionRate", [{}])[-1].get("value")},
            {"metric": "last_steps_per_second", "value": scalars.get("SPS", [{}])[-1].get("value")},
            {"metric": "last_explained_variance", "value": scalars.get("Critic/Explained_Variance", [{}])[-1].get("value")},
            {"metric": "last_approx_kl", "value": scalars.get("Policy/ApproxKL", [{}])[-1].get("value")},
            {"metric": "last_clip_fraction", "value": scalars.get("Policy/ClipFraction", [{}])[-1].get("value")},
            {"metric": "total_oom_skipped_update", "value": sum(item["value"] for item in scalars.get("OOM/SkippedUpdate", []))},
            {"metric": "total_policy_meltdown", "value": sum(item["value"] for item in scalars.get("Policy/Meltdown_Count", []))},
        ]
    ).to_csv(archive / "training_summary.csv", index=False, encoding="utf-8-sig")

    raw_source_files = sorted(SOURCE.rglob("*"))
    raw_source_files = [path for path in raw_source_files if path.is_file()]
    raw_archive_files = [archive / path.relative_to(SOURCE) for path in raw_source_files]
    copy_rows = []
    for source, target in zip(raw_source_files, raw_archive_files):
        copy_rows.append(
            {
                "relative_path": str(source.relative_to(SOURCE)).replace("\\", "/"),
                "source_size": source.stat().st_size,
                "archive_size": target.stat().st_size,
                "source_sha256": sha256(source),
                "archive_sha256": sha256(target),
                "sha256_equal": sha256(source) == sha256(target),
            }
        )
    (archive / "copy_integrity_check.json").write_text(
        json.dumps(
            {
                "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
                "archive": str(archive.relative_to(ROOT)).replace("\\", "/"),
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
    (archive / "resolved_config.yaml").write_bytes((archive / "configs" / "resolved_config.yaml").read_bytes())
    (archive / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_type": "initial_schedule_training",
                "method": "HB-GAT-PPO",
                "variant": "full_joint",
                "run_id": "joint100_full_joint_seed42",
                "training_status": "partial_training",
                "target_episodes": target_episodes,
                "observed_rollout_updates": observed_updates,
                "source_run_manifest": "configs/run_manifest.json",
                "training_summary": "training_summary.json",
                "git_commit": None,
                "server_run_dir": summary["server_run_dir"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (archive / "integrity_check.json").write_text(
        json.dumps(
            {
                "raw_copy_all_sha256_equal": all(row["sha256_equal"] for row in copy_rows),
                "raw_file_count": len(copy_rows),
                "tensorboard_nonempty_event_count": len(nonempty),
                "tensorboard_version0_empty": all(not item["has_scalars"] for item in version0_events),
                "target_episodes": target_episodes,
                "observed_rollout_updates": observed_updates,
                "training_complete": observed_updates >= target_episodes and episode >= target_episodes,
                "checkpoint_best_last_equal": summary["best_equals_last"],
                "git_commit_verified": False,
                "formal_cross_scale_validation_present": (archive / "eval").exists()
                and any((archive / "eval").iterdir()),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    readme = f"""# HB-GAT-PPO full_joint 训练记录（阶段性）

## 结果身份

- 方法：HB-GAT-PPO 主方法，完整联合动作版本（`operation_station_worker`、自回归 worker/team、Skill Hub 双向图）。
- 运行 ID：`joint100_full_joint_seed42`；seed=42；训练目标为 `{target_episodes}` episodes。
- 训练数据池：`data/scale_400_800_datasets`；训练期配置中的固定评估数据：`data/680.csv`。
- 当前只记录训练证据，不是四规模正式验证，也不是消融或对比算法结果。

## 实际完成情况

- TensorBoard 非空 event 记录了 `{observed_updates}` 个 rollout/update 点；checkpoint 元数据 episode=`{episode}`，完成度 `{summary['completion_ratio']:.1%}`。
- best checkpoint SHA-256：`{summary['checkpoint_sha256']}`；last checkpoint SHA-256：`{summary['last_checkpoint_sha256']}`；二者字节级相同：`{summary['best_equals_last']}`。
- best checkpoint 的训练期 Eval makespan：`{best_eval['value'] if best_eval else '未记录'}` h；最后 Eval makespan：`{last_eval['value'] if last_eval else '未记录'}` h。
- 最后一次 Eval completion rate：`{scalars.get('Eval/completion_rate', [{}])[-1].get('value', '未记录')}`；训练期指标中 OOM skipped update 和 policy meltdown 均为 0。

## 证据限制

- 当前记录到 `{observed_updates}/{target_episodes}` 个 rollout/update 点，不能标记为 `completed`，也不能据此宣称最终主方法性能。
- 原始 `run_manifest.json` 的 `git_commit` 为 `null`，只能追溯到服务器路径和 resolved config；代码版本锁定证据不足。
- `eval/` 为空，没有四个真实规模的正式验证结果；`version_0` TensorBoard event 只有空壳文件，`version_1` 才包含有效训练曲线。
- 该目录应作为 `partial_training` 训练证据保留；后续可继续观察增量训练，直到指标稳定或明确停止，再进行正式跨规模验证，不要求机械完成 100 episodes。

详细文件：`training_curves.csv`、`training_summary.csv/json`、`integrity_check.json`、`copy_integrity_check.json`、`file_manifest.json`。
"""
    (archive / "README.md").write_text(readme, encoding="utf-8")

    manifest_rows = []
    for path in sorted(archive.rglob("*")):
        if path.is_file() and path.name != "file_manifest.json":
            manifest_rows.append(
                {
                    "path": str(path.relative_to(archive)).replace("\\", "/"),
                    "size": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    (archive / "file_manifest.json").write_text(
        json.dumps(
            {"root": str(archive.relative_to(ROOT)).replace("\\", "/"), "files": manifest_rows},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
