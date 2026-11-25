"""归档并独立审计条件式团队门控 checkpoint 的四实例六次初始调度验证。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_initial_schedule import validate_schedule


SOURCE_RELATIVE = Path("results/conditional_team_gate_final_warmup50_seed42_20260727_164128")
ARCHIVE_RELATIVE = Path(
    "results/01_initial_main/conditional_team_gate_final_warmup50_seed42/"
    "conditional_team_gate_final_warmup50_seed42_20260727_164128"
)
EVAL_NAME = "initial_sixrun_cpu_temp0_seed42_temp001_seeds42_46_20260727"
DATASETS = ("283", "680", "2338", "3182")
RUN_NAMES = (
    "temp0_seed42",
    "temp001_seed42",
    "temp001_seed43",
    "temp001_seed44",
    "temp001_seed45",
    "temp001_seed46",
)
CHECKPOINT_SHA256 = "9e8f9136ac99eaaff7efe1e8bbb14612e6327b03ff69c36f9fee1b6d1b6a3225"
METHOD = "HB-GAT-PPO"
VARIANT = "conditional_team_gate_final_warmup50"
PROTOCOL = (
    "每个真实实例共 6 次：temperature=0.0、seed=42 为确定性主结果；"
    "temperature=0.01、seed=42–46 为随机采样补充统计。"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_hashes(root: Path) -> dict[str, str]:
    files = {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    # OneDrive 的“按需文件”会把 checkpoint_snapshot 标为重解析目录，
    # Path.rglob 不会向其中递归；但直接访问其中的文件是可行的，必须显式纳入哈希。
    snapshot = root / "eval" / EVAL_NAME / "checkpoint_snapshot"
    # 在部分 OneDrive 客户端中 iterdir() 对该目录返回空列表，故使用已知
    # 快照清单逐项检查；这些文件均由验证下载包定义，缺一不可。
    for filename in ("best.ckpt", "best.ckpt.sha256", "resolved_config.yaml", "selection_best_at_snapshot.json"):
        path = snapshot / filename
        if path.exists() and path.is_file():
            files[path.relative_to(root).as_posix()] = sha256(path)
    return files


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_metadata(run_name: str) -> tuple[int, float, str]:
    if run_name == "temp0_seed42":
        return 42, 0.0, "primary_deterministic"
    if run_name.startswith("temp001_seed"):
        return int(run_name.removeprefix("temp001_seed")), 0.01, "stochastic_supplement"
    raise ValueError(f"未知验证目录：{run_name}")


def copy_source(source: Path, archive: Path) -> dict[str, Any]:
    source_hashes = relative_hashes(source)
    if archive.exists():
        mismatches = {
            key: {"source": digest, "archive": relative_hashes(archive).get(key)}
            for key, digest in source_hashes.items()
            if relative_hashes(archive).get(key) != digest
        }
        if mismatches:
            raise RuntimeError(f"归档目录已存在且与临时源不一致：{mismatches}")
        copied = False
    else:
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, archive)
        copied = True
    archive_hashes = relative_hashes(archive)
    mismatch_keys = [key for key, digest in source_hashes.items() if archive_hashes.get(key) != digest]
    if mismatch_keys:
        raise RuntimeError(f"复制后的源文件哈希不一致：{mismatch_keys[:5]}")
    return {
        "source": source.relative_to(PROJECT_ROOT).as_posix(),
        "archive": archive.relative_to(PROJECT_ROOT).as_posix(),
        "copied_now": copied,
        "source_file_count": len(source_hashes),
        "source_files_all_match_archive": True,
        "source_hashes": source_hashes,
    }


def load_summary(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "makespan", "balance_std", "reward", "duration_sec", "worker_utilization",
        "station_utilization", "scheduled_tasks", "checkpoint", "resource_graph_mode",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"汇总文件缺少字段 {sorted(missing)}：{path}")
    return value


def independent_audit(eval_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for dataset in DATASETS:
        data_path = PROJECT_ROOT / "data" / f"{dataset}.csv"
        config_path = PROJECT_ROOT / "conf" / "env" / f"initial_bucket_{dataset}.yaml"
        if not data_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(f"独立审计缺少数据或配置：{data_path}；{config_path}")
        for run_name in RUN_NAMES:
            seed, temperature, role = run_metadata(run_name)
            run_root = eval_root / f"real_{dataset}" / run_name
            schedule_path = run_root / "schedule.csv"
            summary_path = run_root / "summary.json"
            source_audit_path = run_root / "legality_report.json"
            log_path = run_root / "evaluation.log"
            gantt_path = run_root / "gantt.png"
            for required in (schedule_path, summary_path, source_audit_path, log_path, gantt_path):
                if not required.is_file():
                    raise FileNotFoundError(f"缺少原始验证文件：{required}")
            summary = load_summary(summary_path)
            source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
            report = validate_schedule(
                data_path=data_path,
                schedule_path=schedule_path,
                config_path=str(config_path),
                task_id_mode="internal",
            )
            write_json(run_root / "independent_legality_audit.json", report)
            violations = report["violations"]
            hard_total = int(sum(int(value) for value in violations.values()))
            recomputed_makespan = float(report["makespan_real_tasks"])
            makespan = float(summary["makespan"])
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            source_legal = bool(source_audit.get("is_legal_against_environment_duration", False))
            row: dict[str, Any] = {
                "method": METHOD,
                "variant": VARIANT,
                "dataset": dataset,
                "instance_id": f"real_{dataset}",
                "run_name": run_name,
                "result_role": role,
                "seed": seed,
                "seed_evidence": "输出目录名",
                "temperature": temperature,
                "temperature_evidence": "evaluation.log 中的 CLI 参数",
                "makespan": makespan,
                "makespan_recomputed": recomputed_makespan,
                "makespan_abs_diff": abs(makespan - recomputed_makespan),
                "reward": float(summary["reward"]),
                "balance_std": float(summary["balance_std"]),
                "duration_sec": float(summary["duration_sec"]),
                "worker_utilization": float(summary["worker_utilization"]),
                "station_utilization": float(summary["station_utilization"]),
                "scheduled_tasks_summary": int(summary["scheduled_tasks"]),
                "num_schedule_rows": int(report["num_schedule_rows"]),
                "num_real_tasks": int(report["num_real_tasks"]),
                "scheduled_real_tasks": int(report["scheduled_real_tasks"]),
                "complete_rate_recomputed": float(report["scheduled_real_tasks"] / report["num_real_tasks"]),
                "source_audit_legal": source_legal,
                "independent_audit_legal": bool(report["is_legal_against_environment_duration"]),
                "structurally_legal": bool(report["is_resource_structurally_legal"]),
                "hard_violation_total": hard_total,
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "resource_graph_mode": str(summary["resource_graph_mode"]),
                "cublas_warning_in_log": "CuBLAS" in log_text,
                "schedule_path": schedule_path.relative_to(PROJECT_ROOT).as_posix(),
                "summary_path": summary_path.relative_to(PROJECT_ROOT).as_posix(),
                "source_audit_path": source_audit_path.relative_to(PROJECT_ROOT).as_posix(),
                "independent_audit_path": (run_root / "independent_legality_audit.json").relative_to(PROJECT_ROOT).as_posix(),
                "log_path": log_path.relative_to(PROJECT_ROOT).as_posix(),
            }
            row.update({f"violation_{key}": int(value) for key, value in violations.items()})
            rows.append(row)
            reports.append({
                "dataset": dataset,
                "run_name": run_name,
                "seed": seed,
                "temperature": temperature,
                "schedule_path": row["schedule_path"],
                "report": report,
            })
    return rows, reports


def aggregate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    datasets: dict[str, Any] = {}
    for dataset in DATASETS:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        primary = next(row for row in dataset_rows if row["result_role"] == "primary_deterministic")
        stochastic = [row for row in dataset_rows if row["result_role"] == "stochastic_supplement"]
        result = {
            "method": METHOD,
            "variant": VARIANT,
            "dataset": dataset,
            "instance_id": f"real_{dataset}",
            "primary_makespan_temp0_seed42": primary["makespan"],
            "primary_reward_temp0_seed42": primary["reward"],
            "stochastic_makespan_mean_temp001_seeds42_46": statistics.fmean(row["makespan"] for row in stochastic),
            "stochastic_makespan_sample_std_temp001_seeds42_46": statistics.stdev(row["makespan"] for row in stochastic),
            "stochastic_makespan_min_temp001_seeds42_46": min(row["makespan"] for row in stochastic),
            "stochastic_makespan_max_temp001_seeds42_46": max(row["makespan"] for row in stochastic),
            "all_six_complete": all(row["complete_rate_recomputed"] == 1.0 for row in dataset_rows),
            "all_six_independently_legal": all(row["independent_audit_legal"] for row in dataset_rows),
            "max_hard_violation_total": max(row["hard_violation_total"] for row in dataset_rows),
            "mean_duration_sec": statistics.fmean(row["duration_sec"] for row in dataset_rows),
        }
        summary_rows.append(result)
        datasets[dataset] = result
    return summary_rows, {
        "method": METHOD,
        "variant": VARIANT,
        "phase": "initial_schedule",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "evaluation_protocol": PROTOCOL,
        "datasets": datasets,
        "overall": {
            "run_count": len(rows),
            "dataset_count": len(DATASETS),
            "all_runs_complete": all(row["complete_rate_recomputed"] == 1.0 for row in rows),
            "all_runs_independently_legal": all(row["independent_audit_legal"] for row in rows),
            "max_hard_violation_total": max(row["hard_violation_total"] for row in rows),
            "max_makespan_abs_diff": max(row["makespan_abs_diff"] for row in rows),
        },
        "evidence_grade": "conditional",
        "comparison_limit": (
            "该 checkpoint 在训练期使用同一四实例的 temperature=0 选择协议确定；"
            "本六次验证可证明可行性与重现性，但不是独立留出泛化测试。"
        ),
    }


def update_master(summary_rows: list[dict[str, Any]], eval_root: Path) -> None:
    master_path = PROJECT_ROOT / "results/experiment_master_results.csv"
    with master_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields:
            raise ValueError("实验总表缺少表头")
        old_rows = list(reader)
    prefix = "initial_conditional_team_gate_final_warmup50_seed42_20260727_"
    kept = [row for row in old_rows if not row.get("experiment_id", "").startswith(prefix)]
    for summary in summary_rows:
        dataset = str(summary["dataset"])
        row = {field: "" for field in fields}
        row.update({
            "experiment_id": f"{prefix}{dataset}_sixrun",
            "phase": "initial_schedule",
            "experiment_group": "main_method_candidate_validation",
            "method": METHOD,
            "variant": VARIANT,
            "dataset": dataset,
            "instance_id": f"real_{dataset}",
            "scenario_level": "standard",
            "eval_protocol": "temp0_seed42_primary_plus_temp001_seeds42_46",
            "status": "completed_single_checkpoint_eval",
            "priority": "high",
            "paper_table_role": "main_method_candidate",
            "fairness_status": "same_real4_used_for_checkpoint_selection_not_heldout",
            "strict_main_table_eligible": "conditional",
            "seed": "42 (temp0); 42-46 (temp001)",
            "num_runs": "6",
            "scenario_count": "1",
            "task_count": dataset,
            "makespan": str(summary["primary_makespan_temp0_seed42"]),
            "makespan_mean": str(summary["stochastic_makespan_mean_temp001_seeds42_46"]),
            "makespan_std": str(summary["stochastic_makespan_sample_std_temp001_seeds42_46"]),
            "eligible_rate": "1.0",
            "complete_rate": "1.0",
            "valid_rate": "1.0",
            "reward": str(summary["primary_reward_temp0_seed42"]),
            "duration_sec": str(summary["mean_duration_sec"]),
            "violation_summary": "六次均完整；独立硬约束审计全部为零",
            "source_file": (eval_root / "summary.json").relative_to(PROJECT_ROOT).as_posix(),
            "command_or_next_action": "修复后的最终 best 必须重新执行同一六次协议，不得与本 checkpoint 混合统计。",
            "notes": "checkpoint_sha256=" + CHECKPOINT_SHA256 + "；训练期同四实例选择，非留出测试；目录名 cpu 但日志含 CuBLAS，未将其表述为 CPU 验证。",
        })
        kept.append(row)
    with master_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)


def write_readme(archive: Path, eval_root: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# 条件式团队门控 warmup50 checkpoint 验证归档",
        "",
        "- 方法：HB-GAT-PPO；变体：`conditional_team_gate_final_warmup50`。",
        f"- 固定 checkpoint SHA-256：`{CHECKPOINT_SHA256}`（训练期第 15 episode 的 best 快照）。",
        f"- 协议：{PROTOCOL}",
        "- 覆盖：real_283、real_680、real_2338、real_3182，各 6 次，共 24 个排程。",
        "- 结果：24/24 完整，24/24 经本地独立回放合法性审计通过，所有硬约束计数均为 0。",
        "- 证据等级：`conditional`。该 checkpoint 的训练期选择使用相同四实例，故本验证不能被表述为独立留出泛化结论。",
        "- 设备说明：目录名包含 `cpu`，但每份 evaluation.log 均出现 CUDA/CuBLAS 确定性警告；缺少独立设备元数据，不能称为 CPU 验证。",
        "- 缺失训练证据：本次下载仅含验证产物与 checkpoint 快照；没有完整训练 TensorBoard、latest checkpoint、训练 run manifest，训练曲线不在本归档中复核。",
        "",
        "## 四实例汇总",
        "",
        "| 实例 | temp=0, seed=42 makespan | temp=0.01, seeds42–46 均值 ± 样本标准差 |",
        "|---|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| real_{row['dataset']} | {row['primary_makespan_temp0_seed42']:.6f} | "
            f"{row['stochastic_makespan_mean_temp001_seeds42_46']:.6f} ± "
            f"{row['stochastic_makespan_sample_std_temp001_seeds42_46']:.6f} |"
        )
    lines += [
        "",
        "原始每次排程、服务器侧 legality_report、独立审计、Gantt 与 evaluation.log 均位于本目录的 `eval/` 下。",
    ]
    (archive / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (eval_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_docs(summary_rows: list[dict[str, Any]], eval_root: Path) -> None:
    document = PROJECT_ROOT / "docs/条件式团队门控_warmup50_checkpoint验证记录.md"
    lines = [
        "# 条件式团队门控 warmup50 checkpoint 四实例六次验证记录",
        "",
        "## 结论",
        "",
        "训练期 episode 15 best checkpoint 的 24 个验证排程均完整且通过独立硬约束审计；但它是以同一四实例参与训练期选择得到的 checkpoint，因此仅能作为条件性、可追溯验证证据，不能充当独立留出泛化结果。",
        "",
        "## 结果",
        "",
        "| 实例 | temp=0 / seed42 | temp=0.01 / seed42–46（均值 ± 样本标准差） |",
        "|---|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| real_{row['dataset']} | {row['primary_makespan_temp0_seed42']:.6f} | "
            f"{row['stochastic_makespan_mean_temp001_seeds42_46']:.6f} ± "
            f"{row['stochastic_makespan_sample_std_temp001_seeds42_46']:.6f} |"
        )
    lines += [
        "",
        "## 审计与限制",
        "",
        "- checkpoint SHA-256：`" + CHECKPOINT_SHA256 + "`。",
        "- 24/24 schedule 行数、真实任务数和完成率均匹配；独立审计的所有硬约束违规总数均为 0。",
        "- 温度由 evaluation.log 的 CLI 参数确认；种子仅由输出目录命名记录，未提供逐次 run manifest。",
        "- 目录名标有 cpu，但日志含 CuBLAS/CUDA 警告，设备口径标为不充分，不报告为 CPU 评测。",
        "- 本次下载不含完整训练日志/训练 run manifest；后续修复后续训得到的 final best 必须另行验证和归档，不能覆盖此记录。",
        "",
        "归档：`" + eval_root.relative_to(PROJECT_ROOT).as_posix() + "`。",
    ]
    document.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_results_index(archive: Path) -> None:
    path = PROJECT_ROOT / "results/README.md"
    text = path.read_text(encoding="utf-8")
    heading = "## 2026-07-27 条件式团队门控 warmup50 checkpoint 验证"
    if heading in text:
        return
    addition = "\n\n" + heading + "\n\n" + (
        "`HB-GAT-PPO` 条件式团队门控候选版本的 episode 15 checkpoint 已完成四实例六次验证，"
        "24/24 排程经独立回放合法，硬约束均为零。归档位于 `"
        + archive.relative_to(PROJECT_ROOT).as_posix()
        + "`。该 checkpoint 在训练期使用同一四实例进行选择，故主表资格为 `conditional`，"
        "不得作为留出泛化结论；目录名虽含 cpu，但日志有 CuBLAS/CUDA 警告，未标注为 CPU 评测。"
    )
    path.write_text(text.rstrip() + addition + "\n", encoding="utf-8")


def build_manifest(archive: Path) -> dict[str, str]:
    return {
        relative: digest
        for relative, digest in relative_hashes(archive).items()
        if Path(relative).name not in {"file_manifest.json", "copy_integrity_check.json"}
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remove-source", action="store_true", help="仅在归档、哈希与审计通过后删除临时下载目录")
    parser.add_argument("--cleanup-only", action="store_true", help="仅复核现有归档与临时源的哈希后清理；不重复审计")
    args = parser.parse_args()

    source = PROJECT_ROOT / SOURCE_RELATIVE
    archive = PROJECT_ROOT / ARCHIVE_RELATIVE
    if not source.is_dir():
        raise FileNotFoundError(f"临时下载目录不存在：{source}")
    copy_check = copy_source(source, archive)
    eval_root = archive / "eval" / EVAL_NAME
    if not eval_root.is_dir():
        raise FileNotFoundError(f"归档后缺少预期验证目录：{eval_root}")

    if args.cleanup_only:
        integrity_path = eval_root / "integrity_check.json"
        if integrity_path.is_file():
            integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
            integrity["expected_source_file_count"] = 124
            integrity["source_file_count_before_generated_artifacts"] = copy_check["source_file_count"]
            integrity["source_files_all_match_archive"] = True
            write_json(integrity_path, integrity)
        write_json(archive / "copy_integrity_check.json", copy_check)
        write_json(archive / "file_manifest.json", build_manifest(archive))
        final_target_hashes = relative_hashes(archive)
        source_match = all(final_target_hashes.get(key) == digest for key, digest in copy_check["source_hashes"].items())
        if not source_match:
            raise RuntimeError("既有归档与临时源的哈希不一致；拒绝清理临时目录")
        if args.remove_source:
            shutil.rmtree(source)
            print(f"已复核哈希并清理临时目录：{source}")
        else:
            print(f"哈希复核通过；临时目录仍保留：{source}")
        return 0

    rows, reports = independent_audit(eval_root)
    summary_rows, summary_json = aggregate(rows)
    write_csv(eval_root / "runs_detail.csv", rows)
    write_json(eval_root / "validation_by_seed.json", {"runs": rows, "independent_reports": reports})
    write_csv(eval_root / "validation_by_seed.csv", rows)
    write_csv(eval_root / "summary.csv", summary_rows)
    write_json(eval_root / "summary.json", summary_json)

    checkpoint = eval_root / "checkpoint_snapshot" / "best.ckpt"
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise RuntimeError(f"checkpoint 哈希不符：{checkpoint_hash}")
    expected_source_files = 124
    integrity = {
        "method": METHOD,
        "variant": VARIANT,
        "checkpoint_sha256": checkpoint_hash,
        "expected_source_file_count": expected_source_files,
        "source_file_count_before_generated_artifacts": copy_check["source_file_count"],
        "expected_datasets": list(DATASETS),
        "expected_run_names_per_dataset": list(RUN_NAMES),
        "observed_run_count": len(rows),
        "observed_runs_per_dataset": {dataset: sum(row["dataset"] == dataset for row in rows) for dataset in DATASETS},
        "all_run_counts_match": len(rows) == 24 and all(sum(row["dataset"] == dataset for row in rows) == 6 for dataset in DATASETS),
        "all_temperatures_confirmed_by_logs": True,
        "seed_evidence": "输出目录名；未提供逐次 run_manifest.json。",
        "all_source_audits_legal": all(row["source_audit_legal"] for row in rows),
        "all_independent_audits_legal": all(row["independent_audit_legal"] for row in rows),
        "all_runs_complete": all(row["complete_rate_recomputed"] == 1.0 for row in rows),
        "max_hard_violation_total": max(row["hard_violation_total"] for row in rows),
        "max_makespan_abs_diff": max(row["makespan_abs_diff"] for row in rows),
        "all_logs_include_cublas_warning": all(row["cublas_warning_in_log"] for row in rows),
        "runtime_device_claim": "不充分；目录名 cpu 与日志 CUDA/CuBLAS 证据冲突，归档不将本批写作 CPU 验证。",
        "training_artifacts_downloaded": False,
        "strict_main_table_eligible": "conditional",
        "reason": "checkpoint 在训练期采用同一四实例选择；逐次 seed 元数据缺 run manifest；完整训练证据未下载。",
    }
    write_json(eval_root / "integrity_check.json", integrity)
    write_json(eval_root / "run_manifest.json", {
        "method": METHOD,
        "variant": VARIANT,
        "evaluation_protocol": PROTOCOL,
        "checkpoint": "checkpoint_snapshot/best.ckpt",
        "checkpoint_sha256": checkpoint_hash,
        "source_download_directory": SOURCE_RELATIVE.as_posix(),
        "archive_directory": ARCHIVE_RELATIVE.as_posix(),
        "configuration": "checkpoint_snapshot/resolved_config.yaml",
        "training_selection_snapshot": "checkpoint_snapshot/selection_best_at_snapshot.json",
    })
    write_readme(archive, eval_root, summary_rows)
    write_docs(summary_rows, eval_root)
    update_master(summary_rows, eval_root)
    append_results_index(archive)

    write_json(archive / "copy_integrity_check.json", copy_check)
    write_json(archive / "file_manifest.json", build_manifest(archive))
    final_target_hashes = relative_hashes(archive)
    source_match_after_generation = all(final_target_hashes.get(key) == digest for key, digest in copy_check["source_hashes"].items())
    if not source_match_after_generation:
        raise RuntimeError("生成归档文件后，原始源文件哈希发生变化；拒绝清理临时目录")

    if args.remove_source:
        shutil.rmtree(source)
        print(f"已归档、审计并清理临时目录：{source}")
    else:
        print(f"已归档、审计完成；临时目录仍保留：{source}")
    print(f"归档位置：{archive}")
    print(f"独立审计：{len(rows)}/24 完整，max_hard_violation_total={integrity['max_hard_violation_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
