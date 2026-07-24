"""归档新版本初始调度学习型 baseline 的训练结果。

该脚本只处理结果文件，不修改训练、环境或模型代码。它先核对源目录与
归档目录的逐文件 SHA-256，再生成训练摘要、最佳排程合法性审计和文件清单。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_one(root: Path, name: str) -> Path:
    candidates = sorted(root.rglob(name), key=lambda item: (len(item.parts), item.as_posix()))
    if not candidates:
        raise FileNotFoundError(f"未找到 {name}: {root}")
    return candidates[0]


def relative_file_rows(source: Path, archive: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_file in sorted(path for path in source.rglob("*") if path.is_file()):
        relative = source_file.relative_to(source)
        target_file = archive / relative
        if not target_file.exists():
            raise FileNotFoundError(f"归档缺少源文件: {relative}")
        source_hash = sha256(source_file)
        archive_hash = sha256(target_file)
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "source_size": source_file.stat().st_size,
                "archive_size": target_file.stat().st_size,
                "source_sha256": source_hash,
                "archive_sha256": archive_hash,
                "sha256_equal": source_hash == archive_hash,
            }
        )
    return rows


def load_config(archive: Path) -> tuple[Path, dict[str, Any]]:
    path = find_one(archive, "resolved_config.yaml")
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return path, value


def load_manifests(archive: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    paths = sorted(archive.rglob("run_manifest.json"), key=lambda item: (len(item.parts), item.as_posix()))
    if not paths:
        raise FileNotFoundError(f"未找到 run_manifest.json: {archive}")
    manifests = [read_json(path) for path in paths]
    return paths[0], manifests[0], manifests


def meta_for(archive: Path, stem: str) -> Path:
    paths = sorted(archive.rglob(f"*{stem}.meta.json"))
    if not paths:
        raise FileNotFoundError(f"未找到 {stem} 元数据: {archive}")
    return paths[0]


def checkpoint_for(archive: Path, stem: str) -> Path | None:
    paths = sorted(archive.rglob(f"*{stem}.pth"))
    return paths[0] if paths else None


def schedule_for(archive: Path, stem: str) -> Path:
    paths = sorted(archive.rglob(f"*{stem}_schedule.csv"))
    if not paths:
        raise FileNotFoundError(f"未找到 {stem} 排程: {archive}")
    return paths[0]


def validate_best_schedule(schedule: Path) -> dict[str, Any]:
    scripts_dir = PROJECT_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from validate_initial_schedule import validate_schedule

    data_path = PROJECT_ROOT / "data" / "680.csv"
    return validate_schedule(data_path=data_path, schedule_path=schedule)


def main() -> int:
    parser = argparse.ArgumentParser(description="归档初始调度学习型 baseline 训练结果")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--method", choices=("Graph-DDQN-APAL", "L2D-PPO-APAL"), required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    archive = args.archive.resolve()
    if not source.is_dir() or not archive.is_dir():
        raise NotADirectoryError(f"源/归档目录不存在: {source} / {archive}")

    raw_rows = relative_file_rows(source, archive)
    if not raw_rows or not all(row["sha256_equal"] for row in raw_rows):
        raise RuntimeError("源目录与归档目录的 SHA-256 校验未全部通过，停止生成记录。")
    (archive / "copy_integrity_check.json").write_text(
        json.dumps(
            {
                "source": source.as_posix(),
                "archive": archive.as_posix(),
                "raw_file_count": len(raw_rows),
                "raw_total_bytes": sum(int(row["source_size"]) for row in raw_rows),
                "all_equal": True,
                "files": raw_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    config_path, config = load_config(archive)
    manifest_path, manifest, manifests = load_manifests(archive)
    (archive / "run_manifest_originals.json").write_text(
        json.dumps({"paths": [path.relative_to(archive).as_posix() for path in sorted(archive.rglob("run_manifest.json"))], "manifests": manifests}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if config_path.relative_to(archive).as_posix() != "resolved_config.yaml":
        shutil.copy2(config_path, archive / "resolved_config.yaml")

    metrics_path = find_one(archive, "train_metrics.csv")
    metrics = pd.read_csv(metrics_path)
    eval_metrics = metrics[metrics["eval_makespan"].notna()].copy() if "eval_makespan" in metrics else pd.DataFrame()
    best_meta_path = meta_for(archive, "best")
    latest_meta_path = meta_for(archive, "latest")
    best_meta = read_json(best_meta_path)
    latest_meta = read_json(latest_meta_path)
    best_schedule = schedule_for(archive, "best")
    legality = validate_best_schedule(best_schedule)

    checkpoint_paths: dict[str, str] = {}
    checkpoint_hashes: dict[str, str] = {}
    checkpoint_loadability: dict[str, bool] = {}
    for label, stem in (("best", "best"), ("latest", "latest"), ("exact_resume", "exact_resume"), ("final", "final")):
        path = checkpoint_for(archive, stem)
        if path is None:
            continue
        rel = path.relative_to(archive).as_posix()
        checkpoint_paths[label] = rel
        checkpoint_hashes[label] = sha256(path)
        try:
            torch.load(path, map_location="cpu", weights_only=False)
            checkpoint_loadability[label] = True
        except Exception:
            checkpoint_loadability[label] = False

    target_episodes = int(config.get("max_episodes", 0) or 0)
    observed_max_episode = int(metrics["episode"].max()) if "episode" in metrics and not metrics.empty else None
    incomplete_rows = int((metrics["complete"] < 1).sum()) if "complete" in metrics else None
    invalid_rows = int((metrics["invalid_count"] > 0).sum()) if "invalid_count" in metrics else None
    oom_column = "oom_skipped_updates" if "oom_skipped_updates" in metrics else "oom_skipped"
    oom_sum = float(metrics[oom_column].fillna(0).sum()) if oom_column in metrics else 0.0

    eval_summary: dict[str, Any] = {
        "count": int(len(eval_metrics)),
        "valid_rate": float(eval_metrics["eval_valid"].mean()) if "eval_valid" in eval_metrics and len(eval_metrics) else None,
        "complete_rate": float(eval_metrics["eval_complete"].mean()) if "eval_complete" in eval_metrics and len(eval_metrics) else None,
    }
    if len(eval_metrics):
        best_eval_row = eval_metrics.loc[eval_metrics["eval_makespan"].idxmin()]
        last_eval_row = eval_metrics.iloc[-1]
        recent = eval_metrics.tail(min(10, len(eval_metrics)))["eval_makespan"]
        eval_summary.update(
            {
                "best_makespan": float(best_eval_row["eval_makespan"]),
                "best_episode": int(best_eval_row["episode"]),
                "last_makespan": float(last_eval_row["eval_makespan"]),
                "last_episode": int(last_eval_row["episode"]),
                "recent10_min": float(recent.min()),
                "recent10_max": float(recent.max()),
                "recent10_mean": float(recent.mean()),
                "recent10_std": float(recent.std(ddof=1)) if len(recent) > 1 else 0.0,
            }
        )

    summary: dict[str, Any] = {
        "method": args.method,
        "phase": "initial_schedule_training",
        "experiment_group": "initial_literature_baseline_training",
        "variant": "literature_graph_double_dqn" if args.method.startswith("Graph") else "literature_graph_ppo",
        "training_status": "converged_training",
        "evidence_level": "training_diagnostic_only",
        "strict_main_table_eligible": False,
        "source_directory": source.as_posix(),
        "archive_directory": archive.as_posix(),
        "run_id": manifest.get("run_id") or manifest.get("experiment_name") or source.name,
        "seed": best_meta.get("seed", manifest.get("seed", config.get("seed"))),
        "target_episodes": target_episodes,
        "observed_metrics_max_episode": observed_max_episode,
        "latest_checkpoint_episode": latest_meta.get("episode"),
        "best_checkpoint_episode": best_meta.get("episode"),
        "best_makespan": best_meta.get("best_makespan"),
        "dataset_training_pool": best_meta.get("train_data_path_or_dir") or manifest.get("train_data_path_or_dir"),
        "training_instance": best_meta.get("data_file_path") or manifest.get("data_file_path"),
        "metrics_file": metrics_path.relative_to(archive).as_posix(),
        "metrics_rows": int(len(metrics)),
        "training_incomplete_rows": incomplete_rows,
        "training_invalid_rows": invalid_rows,
        "oom_skipped_total": oom_sum,
        "eval": eval_summary,
        "best_schedule": best_schedule.relative_to(archive).as_posix(),
        "best_schedule_makespan": legality.get("makespan_real_tasks"),
        "best_schedule_legal": bool(legality.get("is_legal_against_current_data_duration")),
        "checkpoint_paths": checkpoint_paths,
        "checkpoint_sha256": checkpoint_hashes,
        "checkpoint_loadability": checkpoint_loadability,
        "resume_recommendation": checkpoint_paths.get("exact_resume") or checkpoint_paths.get("latest") or checkpoint_paths.get("best"),
        "git_commit": manifest.get("git_commit"),
        "server_run_dir": manifest.get("run_dir"),
        "formal_cross_scale_validation_present": False,
    }
    (archive / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_rows = []
    for key, value in summary.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            summary_rows.append({"metric": key, "value": value})
    summary_rows.extend(
        [
            {"metric": "eval_best_makespan", "value": eval_summary.get("best_makespan")},
            {"metric": "eval_last_makespan", "value": eval_summary.get("last_makespan")},
            {"metric": "eval_valid_rate", "value": eval_summary.get("valid_rate")},
            {"metric": "eval_complete_rate", "value": eval_summary.get("complete_rate")},
        ]
    )
    pd.DataFrame(summary_rows).to_csv(archive / "summary.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(archive / "train_metrics_normalized.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"episode": int(row.episode), "eval_makespan": float(row.eval_makespan), "eval_valid": int(row.eval_valid), "eval_complete": int(row.eval_complete)} for row in eval_metrics.itertuples()]).to_csv(archive / "eval_metrics.csv", index=False, encoding="utf-8-sig")

    integrity = {
        "raw_copy_all_sha256_equal": True,
        "raw_file_count": len(raw_rows),
        "target_episodes": target_episodes,
        "metrics_rows": int(len(metrics)),
        "metrics_episode_duplicates": int(metrics.episode.duplicated().sum()) if "episode" in metrics else None,
        "best_schedule_legal": bool(legality.get("is_legal_against_current_data_duration")),
        "best_schedule_violations": legality.get("violations", {}),
        "best_checkpoint_loadable": checkpoint_loadability.get("best"),
        "latest_checkpoint_loadable": checkpoint_loadability.get("latest"),
        "resume_checkpoint_present": bool(summary["resume_recommendation"]),
        "training_status": summary["training_status"],
        "strict_main_table_eligible": False,
    }
    (archive / "integrity_check.json").write_text(json.dumps(integrity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest_out = dict(manifest)
    manifest_out.update(
        {
            "archive_directory": archive.as_posix(),
            "source_directory": source.as_posix(),
            "training_status": summary["training_status"],
            "evidence_level": summary["evidence_level"],
            "strict_main_table_eligible": False,
            "summary_file": "summary.json",
            "integrity_file": "integrity_check.json",
        }
    )
    # 源目录若本身有根级 run_manifest.json，必须保持原始字节不变；增强版另存。
    if (source / "run_manifest.json").exists():
        shutil.copy2(source / "run_manifest.json", archive / "run_manifest.json")
    else:
        (archive / "run_manifest.json").write_text(json.dumps(manifest_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (archive / "archive_run_manifest.json").write_text(json.dumps(manifest_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    readme = f"""# {args.method} 初始调度训练归档

## 实验身份

- 阶段：初始调度训练；训练实例：`real_680`。
- 训练池：`{summary['dataset_training_pool']}`。
- seed：`{summary['seed']}`；目标预算：`{target_episodes}` episodes。
- 归档状态：`converged_training`（按用户确认的提前停止口径）；仍保留可继续训练的 checkpoint。

## 训练结果

- best checkpoint episode：`{summary['best_checkpoint_episode']}`；best makespan：`{summary['best_makespan']}` h。
- latest checkpoint episode：`{summary['latest_checkpoint_episode']}`；训练指标最大 episode：`{summary['observed_metrics_max_episode']}`。
- 训练指标行数：`{summary['metrics_rows']}`；训练期 incomplete 行：`{summary['training_incomplete_rows']}`；invalid 行：`{summary['training_invalid_rows']}`；OOM 跳过累计：`{summary['oom_skipped_total']}`。
- 自动验证次数：`{eval_summary.get('count')}`；验证合法率：`{eval_summary.get('valid_rate')}`；完成率：`{eval_summary.get('complete_rate')}`。
- 最近验证 makespan：`{eval_summary.get('last_makespan')}` h；最近 10 次范围：`{eval_summary.get('recent10_min')}`–`{eval_summary.get('recent10_max')}` h。

## 合法性与证据边界

- best schedule：`{summary['best_schedule']}`，独立 680 数据集硬约束审计：`{summary['best_schedule_legal']}`。
- 详细违规计数见 `integrity_check.json`；源/归档逐文件 SHA-256 见 `copy_integrity_check.json`。
- 这是训练期自动验证证据，不是 `real_283/680/2338/3182` 的统一正式验证；`strict_main_table_eligible=no`，不能直接作为最终跨规模主表性能结论。
- 训练过程可继续：优先使用 `{summary['resume_recommendation']}`；所有 best/latest/final/exact-resume 权重和 replay（若存在）均保留。

## 主要文件

`summary.csv/json`、`train_metrics.csv`、`train_metrics_normalized.csv`、`eval_metrics.csv`、`integrity_check.json`、`copy_integrity_check.json`、`run_manifest.json`、`resolved_config.yaml`、`file_manifest.json`。
"""
    (archive / "README.md").write_text(readme, encoding="utf-8")

    manifest_rows = []
    for path in sorted(archive.rglob("*")):
        if path.is_file() and path.name != "file_manifest.json":
            manifest_rows.append({"path": path.relative_to(archive).as_posix(), "size": path.stat().st_size, "sha256": sha256(path)})
    (archive / "file_manifest.json").write_text(json.dumps({"root": archive.as_posix(), "files": manifest_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
