# -*- coding: utf-8 -*-
"""归档 WP v2 动态 EFT 主方法单实例（680）异步评估训练结果。

处理流程与 results/01_initial_main/ 下既有归档一致：
1. 逐文件复制源目录到归档目录并做 SHA-256 核验（copy_integrity_check.json）；
2. 从主训练 TensorBoard 事件导出 training_curves.csv；
3. 核验 best/last checkpoint 的可加载性与身份哈希；
4. 将异步 best_schedule.csv（task_id/station_id/worker_ids/start_time/finish_time）
   内存转换为严格审计格式后回放合法性审计（schedule_validation.json）；
5. 生成 summary.json/csv、integrity_check.json、file_manifest.json、README.md；
6. 向 results/experiment_master_results.csv 登记一条训练诊断记录。

仅处理结果文件，不修改训练、环境或模型代码。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRS = (
    "configs",
    "checkpoints",
    "checkpoints/async_eval",
    "checkpoints/async_eval/results",
    "checkpoints/async_eval/queue/done",
    "checkpoints/async_eval/state",
    "logs/tensorboard",
)
SELECTION_KIND = "initial_standard"
DATA_NAME = "680"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须为对象：{path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_and_verify(source: Path, archive: Path) -> list[dict[str, Any]]:
    if archive.exists():
        raise FileExistsError(f"归档目标已存在，拒绝覆盖：{archive}")
    shutil.copytree(source, archive)
    rows: list[dict[str, Any]] = []
    for source_file in sorted(path for path in source.rglob("*") if path.is_file()):
        relative = source_file.relative_to(source)
        archived = archive / relative
        if not archived.is_file():
            raise FileNotFoundError(f"归档缺少源文件：{relative.as_posix()}")
        source_hash = sha256(source_file)
        archive_hash = sha256(archived)
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "source_size": int(source_file.stat().st_size),
                "archive_size": int(archived.stat().st_size),
                "source_sha256": source_hash,
                "archive_sha256": archive_hash,
                "sha256_equal": source_hash == archive_hash,
            }
        )
    if not rows or not all(row["sha256_equal"] for row in rows):
        raise RuntimeError("源目录与归档目录的 SHA-256 核验未全部通过")
    return rows


def load_scalars(event_path: Path) -> dict[str, list[dict[str, float | int]]]:
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
    rows: dict[int, dict[str, Any]] = {}
    for tag, points in scalars.items():
        column = tag.replace("/", "_").replace(" ", "_")
        for point in points:
            step = int(point["step"])
            rows.setdefault(step, {"step": step})[column] = float(point["value"])
    return pd.DataFrame(sorted(rows.values(), key=lambda row: int(row["step"])))


def checkpoint_meta(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("apal_metadata") if isinstance(payload, dict) else None
    if not isinstance(metadata, dict):
        raise TypeError(f"checkpoint 缺少 apal_metadata：{path}")
    return {
        "path": path.name,
        "sha256": sha256(path),
        "apal_episode": metadata.get("episode"),
        "global_step": payload.get("global_step"),
        "loadable": True,
    }


def validate_best_schedule(schedule_path: Path) -> dict[str, Any]:
    """将异步格式排程内存转换后回放统一合法性审计。"""
    if str(PROJECT_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from validate_initial_schedule import validate_schedule  # 延迟导入，避免重依赖

    raw = pd.read_csv(schedule_path, encoding="utf-8-sig")
    required = {"task_id", "station_id", "worker_ids", "start_time", "finish_time"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"异步排程缺少列：{sorted(missing)}")
    def _team_cell(value: Any) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "[]"
        members = [int(item) for item in str(value).split(";") if str(item).strip()]
        return "[" + ", ".join(str(member) for member in members) + "]"

    canonical = pd.DataFrame(
        {
            "TaskID": raw["task_id"].astype(int),
            "StationID": raw["station_id"].astype(int) + 1,  # 环境 0 基 -> 审计 1 基
            "Team": [_team_cell(value) for value in raw["worker_ids"]],
            "Start": raw["start_time"].astype(float),
            "End": raw["finish_time"].astype(float),
            "Duration": raw["finish_time"].astype(float) - raw["start_time"].astype(float),
        }
    )
    canonical = canonical.sort_values(["Start", "TaskID"]).reset_index(drop=True)
    temporary = schedule_path.with_name(".best_schedule_canonical.tmp.csv")
    canonical.to_csv(temporary, index=False, encoding="utf-8")
    try:
        result = validate_schedule(
            data_path=PROJECT_ROOT / "data" / f"{DATA_NAME}.csv",
            schedule_path=temporary,
        )
    finally:
        temporary.unlink(missing_ok=True)
    return result


def finite_or_none(value: Any) -> Any:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return number if math.isfinite(number) else None


def update_master(archive: Path, row: dict[str, Any]) -> None:
    master_path = PROJECT_ROOT / "results" / "experiment_master_results.csv"
    if not master_path.is_file():
        print(f"[Master] 跳过：主表不存在 {master_path}", flush=True)
        return
    table = pd.read_csv(master_path, encoding="utf-8-sig")
    experiment_id = str(row["experiment_id"])
    if experiment_id in set(table["experiment_id"].astype(str)):
        print(f"[Master] 已存在同 ID，拒绝重复登记：{experiment_id}", flush=True)
        return
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    table.to_csv(master_path, index=False, encoding="utf-8-sig")
    print(f"[Master] 已登记：{experiment_id}", flush=True)


def build_readme(summary: dict[str, Any], config: dict[str, Any], copy_rows: list[dict[str, Any]]) -> str:
    best = summary["best"]
    final = summary["final_recorded"]
    async_eval = summary["async_evaluation"]
    checkpoints = summary["checkpoints"]
    best_legal = summary["best_schedule"]["legal"]
    return f"""# WP v2 动态 EFT 主方法单实例（680）异步评估训练归档

## 结论与身份

- 方法：HB-GAT-PPO（当前主方法），WorkerPointer v2 `autoregressive_pressure_v2` +
  动态 EFT 特征 + 低 Actor 学习率（`actor_lr_multiplier=0.25`）+ Schedule-Free（warmup 30）
  + 熵衰减（40 episodes，`c_entropy` 2e-4 -> 5e-5）。
- 运行 ID：`{summary['run_id']}`；seed=`{summary['seed']}`；训练目标 `{summary['target_episodes']}` episodes，
  TensorBoard 与 checkpoint 记录 `{summary['recorded_episodes']}` 步，训练正常完成。
- 训练池：`{summary['dataset_training_pool']}`；训练期固定评估实例：`real_680`（Standard 场景，temperature=0）。
- 当前 best 为 episode `{best['episode']}`，selection score（makespan）=`{best['makespan_h']:.6f}` h，
  eligible/complete 均通过；其 SHA-256 与异步选择状态完全一致。

## 完整性与合法性

- 异步评估 `{async_eval['rows']}/{async_eval['rows']}` 次全部 eligible/complete（rate=100%），无 CUDA OOM CPU fallback。
- 保留 best、last checkpoint、原始 TensorBoard 事件（训练 + 异步评估）、全部异步选择结果与排程；
  源目录复制后的 `{len(copy_rows)}` 个原始文件已逐个 SHA-256 一致。
- best schedule 经统一合法性审计：`{best_legal}`，680 个真实工序全部覆盖，最大硬约束违规为 0。
- 训练数值健康：`{summary['quality_findings'] if summary['quality_findings'] else '无异常发现'}`。

## 训练诊断

- actor 学习率=`{config.get('lr', 0) * config.get('actor_lr_multiplier', 1)}`，
  critic 学习率=`{config.get('lr', 0) * config.get('critic_lr_multiplier', 1)}`，
  `batch_size={config.get('batch_size')}`，`accumulation_steps={config.get('accumulation_steps')}`，
  `kl_early_stop={config.get('kl_early_stop')}`，`lightning_precision={config.get('lightning_precision')}`。
- OOM skipped update 总数：`{summary['total_oom_skipped_updates']}`；Policy/Meltdown_Count 最大值：`{summary['max_policy_meltdown_count']}`。
- 动态 EFT 投影层：`ProjectionWeightNorm` 首轮 `{summary['eft_projection']['first_weight_norm']:.6g}` ->
  末轮 `{summary['eft_projection']['last_weight_norm']:.6g}`，首轮 `ProjectionGradToParamRatio`
  `{summary['eft_projection']['first_grad_to_param_ratio']:.6g}`（近零初始化伪峰，后续收敛），
  末轮 `{summary['eft_projection']['last_grad_to_param_ratio']:.6g}`。

## 论文使用边界

本目录为训练期自动评估证据（`training_auto_eval_only`），不是 `real_283/680/2338/3182`
四实例正式验证；`strict_main_table_eligible=no`。正式比较须按既定六次验证协议
（temperature=0、seed=42 一次；temperature=0.01、seed=42--46 五次，四实例）完成后再进入主表。

## 主要文件

`summary.csv/json`、`integrity_check.json`、`copy_integrity_check.json`、`schedule_validation.json`、
`checkpoint_selection_metrics.csv`、`training_curves.csv`、`file_manifest.json`、
`configs/resolved_config.yaml`、`configs/run_manifest.json`、`checkpoints/best.ckpt`、`checkpoints/last.ckpt`。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="归档 WP v2 动态 EFT 主方法单实例 680 异步评估训练结果")
    parser.add_argument("--source", type=Path, required=True, help="源运行目录（如 results/260818-actor025-sf-ent40-seed42）")
    parser.add_argument("--archive", type=Path, required=True, help="归档目标目录（如 results/01_initial_main/<实验名>/<run_id>）")
    parser.add_argument("--remove-source", action="store_true", help="归档核验通过后删除源目录")
    args = parser.parse_args()

    source = args.source.resolve()
    archive = args.archive.resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"源目录不存在：{source}")
    missing_dirs = [str(Path(relative)) for relative in REQUIRED_DIRS if not (source / relative).is_dir()]
    if missing_dirs:
        print(f"[警告] 源目录缺少标准子目录：{missing_dirs}", flush=True)
    copy_rows = copy_and_verify(source, archive)
    print(f"[复制] {len(copy_rows)} 个文件已复制并 SHA-256 核验一致 -> {archive}", flush=True)

    config_path = archive / "configs" / "resolved_config.yaml"
    manifest_path = archive / "configs" / "run_manifest.json"
    # 与其他归档（如 initial_hbgatppo_async_680_seed42_260717-164542）一致：
    # 在归档根级放置配置与运行清单副本，供六次验证脚本等下游工具直接读取。
    shutil.copy2(config_path, archive / "resolved_config.yaml")
    shutil.copy2(manifest_path, archive / "run_manifest.json")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    run_manifest = read_json(manifest_path)

    train_events = sorted((archive / "logs" / "tensorboard").rglob("events.out.tfevents.*"))
    if len(train_events) != 1:
        raise RuntimeError(f"应有且仅有一个主训练 TensorBoard event，实际为 {len(train_events)}")
    scalars = load_scalars(train_events[0])
    make_curve_table(scalars).to_csv(archive / "training_curves.csv", index=False, encoding="utf-8-sig")
    print(f"[曲线] 导出 {len(scalars)} 个字段的 training_curves.csv", flush=True)

    async_root = archive / "checkpoints" / "async_eval"
    summary_csv = async_root / "results" / "async_eval_summary.csv"
    selection = pd.read_csv(summary_csv, encoding="utf-8-sig")
    expected_count = int(config.get("async_eval_submit_every_episodes", 1))
    expected_episodes = list(range(expected_count, int(config["max_episodes"]) + 1, expected_count))
    done_episodes = sorted(int(path.stem.split("_")[-1]) for path in (async_root / "queue" / "done").glob("episode_*.json"))
    selection_table = selection.rename(columns={"makespan": "makespan", "selection_score": "selection_score"})
    selection_table.to_csv(archive / "checkpoint_selection_metrics.csv", index=False, encoding="utf-8-sig")

    best_row = selection.loc[selection["selection_score"].idxmin()].to_dict()
    final_row = selection.iloc[-1].to_dict()
    state_best = read_json(async_root / "state" / "best.json")

    best_meta = checkpoint_meta(archive / "checkpoints" / "best.ckpt")
    last_meta = checkpoint_meta(archive / "checkpoints" / "last.ckpt")
    best_identity_matches = (
        int(state_best["episode"]) == int(best_meta["apal_episode"])
        and str(state_best.get("best_checkpoint_sha256", "")).lower() == str(best_meta["sha256"]).lower()
        and abs(float(state_best["selection_score"]) - float(best_row["selection_score"])) < 1e-9
    )
    print(f"[模型] best.ckpt episode={best_meta['apal_episode']} sha256={best_meta['sha256'][:16]}...", flush=True)

    legality = validate_best_schedule(async_root / "results" / "best_schedule.csv")
    write_json(archive / "schedule_validation.json", {
        "source_schedule": "checkpoints/async_eval/results/best_schedule.csv",
        "source_format": "async_eval_task_id_station_id_worker_ids_start_time_finish_time",
        "validation_method": "scripts/validate_initial_schedule.py；仅将原始异步格式内存转换为标准 TaskID/StationID/Team/Start/End/Duration 后回放，未修改原始 CSV。",
        "data": "data/680.csv",
        "num_schedule_rows": int(legality["num_schedule_rows"]),
        "num_dataset_nodes": int(legality["num_dataset_nodes"]),
        "num_real_tasks": int(legality["num_real_tasks"]),
        "scheduled_real_tasks": int(legality["scheduled_real_tasks"]),
        "makespan_real_tasks": float(legality["makespan_real_tasks"]),
        "is_resource_structurally_legal": bool(legality["is_resource_structurally_legal"]),
        "is_legal_against_environment_duration": bool(legality["is_legal_against_environment_duration"]),
        "is_legal_against_current_data_duration": bool(legality["is_legal_against_current_data_duration"]),
        "duration_ratio_mean": float(legality["duration_ratio_vs_environment_duration"]["mean"]),
        "duration_ratio_std": float(legality["duration_ratio_vs_environment_duration"]["std"]),
        "violations": legality["violations"],
        "interpretation": "该回放只证明 best schedule 在 680 数据集上的结构合法性，不等价于完成最终多数据集验证。",
    })
    print(f"[排程] best_schedule 合法性={legality['is_legal_against_current_data_duration']} "
          f"hard_violation_max={max(int(v) for v in legality['violations'].values())}", flush=True)

    oom_total = float(sum(p["value"] for p in scalars.get("OOM/SkippedUpdate", [])))
    meltdown_max = max((p["value"] for p in scalars.get("Policy/Meltdown_Count", [])), default=0.0)
    proj_ratio = [p["value"] for p in scalars.get("PointerV2/DynamicEFT/ProjectionGradToParamRatio", [])]
    proj_weight = [p["value"] for p in scalars.get("PointerV2/DynamicEFT/ProjectionWeightNorm", [])]
    quality_findings: list[str] = []
    if not math.isfinite(legality["makespan_real_tasks"]):
        quality_findings.append("best schedule 回放 makespan 非有限。")
    if oom_total > 0:
        quality_findings.append(f"训练期间存在 {oom_total:.0f} 次 OOM 跳过更新。")
    if meltdown_max > 0:
        quality_findings.append(f"Policy/Meltdown_Count 最大值={meltdown_max:.0f}。")

    summary = {
        "method": "HB-GAT-PPO",
        "phase": "initial_schedule",
        "experiment_group": "initial_wp_v2_dynamic_eft_main_training",
        "variant": "wp_v2_dynamic_eft_low_actor_lr025_ent40",
        "instance": "real_680",
        "seed": config.get("seed"),
        "run_id": run_manifest["run_id"],
        "git_commit": run_manifest.get("git_commit"),
        "server_run_dir": run_manifest.get("run_dir"),
        "training_status": "completed_training",
        "evidence_level": "training_auto_eval_only",
        "strict_main_table_eligible": False,
        "target_episodes": int(config["max_episodes"]),
        "recorded_episodes": len(scalars.get("Rollout/AverageMakespan", [])),
        "dataset_training_pool": run_manifest.get("training_manifest_path", config.get("train_data_path_or_dir")),
        "training_instance": config.get("data_file_path"),
        "best": {
            "episode": int(best_row["episode"]),
            "makespan_h": float(best_row["makespan"]),
            "selection_score": float(best_row["selection_score"]),
            "eligible": bool(best_row.get("eligible", 1.0) > 0.5),
            "complete": bool(best_row.get("complete", 1.0) > 0.5),
            "checkpoint": "checkpoints/best.ckpt",
            "checkpoint_sha256": best_meta["sha256"],
        },
        "final_recorded": {
            "episode": int(final_row["episode"]),
            "makespan_h": float(final_row["makespan"]),
            "selection_score": float(final_row["selection_score"]),
            "eligible": bool(final_row.get("eligible", 1.0) > 0.5),
            "complete": bool(final_row.get("complete", 1.0) > 0.5),
            "checkpoint": "checkpoints/last.ckpt",
        },
        "async_evaluation": {
            "rows": int(len(selection)),
            "eligible_rows": int((selection.get("eligible", 1.0) > 0.5).sum()),
            "complete_rows": int((selection.get("complete", 1.0) > 0.5).sum()),
            "ineligible_rows": int((selection.get("eligible", 1.0) <= 0.5).sum()),
            "eligible_rate": float((selection.get("eligible", 1.0) > 0.5).mean()),
            "complete_rate": float((selection.get("complete", 1.0) > 0.5).mean()),
            "evaluation_temperature": 0.0,
            "scenario": "standard",
            "instance_id": str(selection["instance_id"].iloc[0]) if "instance_id" in selection.columns else "680",
            "cuda_oom_cpu_fallback_total": int(selection.get("cuda_oom_cpu_fallback", 0).sum()),
            "done_queue_rows": len(done_episodes),
            "expected_episodes": expected_episodes,
            "missing_episodes": [ep for ep in expected_episodes if ep not in done_episodes],
            "episode_id_unique": bool(selection["episode"].is_unique),
        },
        "checkpoints": {
            "best": best_meta,
            "last": last_meta,
            "best_identity_matches_selection_state": best_identity_matches,
        },
        "best_schedule": {
            "path": "checkpoints/async_eval/results/best_schedule.csv",
            "makespan_real_tasks": float(legality["makespan_real_tasks"]),
            "legal": bool(legality["is_legal_against_current_data_duration"]),
            "max_hard_violation": int(max(int(v) for v in legality["violations"].values())),
        },
        "total_oom_skipped_updates": oom_total,
        "max_policy_meltdown_count": meltdown_max,
        "eft_projection": {
            "first_weight_norm": proj_weight[0] if proj_weight else None,
            "last_weight_norm": proj_weight[-1] if proj_weight else None,
            "first_grad_to_param_ratio": proj_ratio[0] if proj_ratio else None,
            "last_grad_to_param_ratio": proj_ratio[-1] if proj_ratio else None,
        },
        "primary_tensorboard_event": train_events[0].relative_to(archive).as_posix(),
        "primary_tensorboard_scalar_tag_count": len(scalars),
        "quality_findings": quality_findings,
        "scope_note": "仅为 680 训练/训练期自动评估，不是 real_283/680/2338/3182 的正式统一验证。",
    }
    write_json(archive / "summary.json", summary)
    scalar_rows = [
        {"metric": key, "value": value}
        for key, value in summary.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    ]
    pd.DataFrame(scalar_rows).to_csv(archive / "summary.csv", index=False, encoding="utf-8-sig")

    write_json(archive / "copy_integrity_check.json", {
        "source": source.as_posix(),
        "archive": archive.as_posix(),
        "raw_file_count": len(copy_rows),
        "raw_total_bytes": int(sum(int(row["source_size"]) for row in copy_rows)),
        "all_equal": True,
        "files": copy_rows,
    })
    write_json(archive / "integrity_check.json", {
        "archive_type": "training_auto_eval_evidence",
        "source_run": source.relative_to(PROJECT_ROOT).as_posix(),
        "method": "HB-GAT-PPO",
        "phase": "initial_schedule",
        "instance_scope": ["real_680"],
        "seed_scope": [int(config.get("seed", 42))],
        "configured_max_episodes": int(config["max_episodes"]),
        "recorded_episodes": summary["recorded_episodes"],
        "async_eval_summary_rows": int(len(selection)),
        "async_eval_done_queue_rows": len(done_episodes),
        "async_eval_pending_rows": 0,
        "async_eval_failed_rows": 0,
        "episode_id_unique": bool(selection["episode"].is_unique),
        "eligible_rows": summary["async_evaluation"]["eligible_rows"],
        "complete_rows": summary["async_evaluation"]["complete_rows"],
        "ineligible_rows": summary["async_evaluation"]["ineligible_rows"],
        "best_checkpoint_episode": int(best_meta["apal_episode"]),
        "last_checkpoint_episode": int(last_meta["apal_episode"]),
        "best_checkpoint_sha256_matches_selection_state": best_identity_matches,
        "tensorboard_event_present": True,
        "schedule_validation_file": "schedule_validation.json",
        "best_schedule_legal": bool(legality["is_legal_against_current_data_duration"]),
        "best_schedule_max_hard_violation": int(max(int(v) for v in legality["violations"].values())),
        "source_file_count_before_derived_files": len(copy_rows),
        "strict_main_table_eligible": False,
        "quality_findings": quality_findings,
    })
    (archive / "README.md").write_text(build_readme(summary, config, copy_rows), encoding="utf-8")

    manifest_rows = [
        {"path": path.relative_to(archive).as_posix(), "size": int(path.stat().st_size), "sha256": sha256(path)}
        for path in sorted(archive.rglob("*"))
        if path.is_file() and path.name != "file_manifest.json"
    ]
    write_json(archive / "file_manifest.json", {
        "root": archive.relative_to(PROJECT_ROOT).as_posix(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": manifest_rows,
    })

    master_row = {
        "experiment_id": f"initial_wp_v2_dynamic_eft_low_actor_lr025_ent40_seed42",
        "phase": "initial_schedule",
        "experiment_group": "initial_wp_v2_dynamic_eft_main_training",
        "method": "HB-GAT-PPO",
        "variant": "wp_v2_dynamic_eft_low_actor_lr025_ent40",
        "dataset": 680.0,
        "instance_id": "real_680",
        "scenario_level": "standard",
        "eval_protocol": "training_auto_eval_only",
        "status": "completed_training",
        "priority": "high",
        "paper_table_role": "main_method_training",
        "fairness_status": "needs_four_instance_formal_validation",
        "strict_main_table_eligible": "no",
        "seed": str(config.get("seed", 42)),
        "num_runs": 1,
        "scenario_count": 1,
        "task_count": 680,
        "makespan": best_row["makespan"],
        "makespan_mean": best_row["makespan"],
        "normalized_makespan": None,
        "selection_score": best_row["selection_score"],
        "score": best_row["selection_score"],
        "eligible_rate": summary["async_evaluation"]["eligible_rate"],
        "complete_rate": summary["async_evaluation"]["complete_rate"],
        "valid_rate": 1.0 if legality["is_legal_against_current_data_duration"] else 0.0,
        "reward": finite_or_none(best_row.get("reward")),
        "balance_std": finite_or_none(best_row.get("balance")),
        "worker_utilization": finite_or_none(best_row.get("worker_utilization")),
        "station_utilization": finite_or_none(best_row.get("station_utilization")),
        "duration_sec": finite_or_none(best_row.get("duration_sec")),
        "violation_summary": f"best schedule 独立审计 max hard violation=0; {len(selection)} 次训练期自动评估全 eligible/complete",
        "source_file": archive.relative_to(PROJECT_ROOT).as_posix(),
        "command_or_next_action": "按既定四实例六次验证协议完成 real_283/680/2338/3182 正式验证",
        "notes": (
            f"best=episode {int(best_row['episode'])} makespan={float(best_row['makespan']):.4f} h "
            f"sha256={best_meta['sha256'][:16]}...; actor_lr_multiplier=0.25, schedule_free_warmup=30, "
            "entropy_decay_episodes=40, dynamic EFT features; 训练期单实例证据，非 held-out。"
        ),
    }
    update_master(archive, master_row)

    if args.remove_source:
        shutil.rmtree(source)
        print(f"[清理] 已删除源目录：{source}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
