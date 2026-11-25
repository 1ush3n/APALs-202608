"""归档主方法快速筛查（M0/M1）结果，并生成可复核的审计文件。

本脚本只处理 ``screening_only=true`` 的临时筛查实验；不得将其写入正式主表。
先复制、再逐文件哈希核验，只有显式给出 ``--remove-sources`` 时才会删除来源目录。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing import event_accumulator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
M0_SOURCE = RESULTS_ROOT / "initial_main_screen_m0_full_joint_warmstart_seed42_20260723_135559"
M0_DOWNLOAD_SNAPSHOT = RESULTS_ROOT / "screen_m0_full_joint_warmstart_seed42_20260723_135559"
M1_SOURCE = RESULTS_ROOT / "screen_m1_stableppo_warmstart_seed42_20260723_163837"
M2_SOURCE = RESULTS_ROOT / "screen_m2_scg_context_warmstart_seed42_20260724_111651"
M2_FAILED_LAUNCH = RESULTS_ROOT / "screen_m2_scg_context_warmstart_seed42_20260724_111016"
ARCHIVE_ROOT = RESULTS_ROOT / "90_legacy_and_smoke"


def sha256(path: Path) -> str:
    """返回单个文件的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest(root: Path) -> dict[str, dict[str, Any]]:
    """生成来源文件清单；键为 POSIX 相对路径。"""
    return {
        file.relative_to(root).as_posix(): {
            "size_bytes": file.stat().st_size,
            "sha256": sha256(file),
        }
        for file in sorted(root.rglob("*"))
        if file.is_file()
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_and_verify(source: Path, target: Path) -> dict[str, Any]:
    """复制 source 至空 target 并对来源文件逐项校验。"""
    assert source.is_dir(), f"来源目录不存在：{source}"
    assert not target.exists(), f"归档目标已存在，拒绝覆盖：{target}"
    before = source_manifest(source)
    shutil.copytree(source, target)
    after = source_manifest(target)
    missing = sorted(set(before) - set(after))
    extra = sorted(set(after) - set(before))
    changed = sorted(key for key in before.keys() & after.keys() if before[key] != after[key])
    result = {
        "source": str(source),
        "target": str(target),
        "source_file_count": len(before),
        "target_file_count_before_generated_records": len(after),
        "missing": missing,
        "extra": extra,
        "changed": changed,
        "passed": not (missing or extra or changed),
    }
    write_json(target / "copy_integrity_check.json", result)
    assert result["passed"], f"复制哈希校验失败：{result}"
    return result


def scalar_rows(root: Path) -> list[dict[str, Any]]:
    """导出 TensorBoard 标量；每行一个 tag/step，便于不依赖 UI 复核。"""
    event_files = sorted(root.glob("logs/tensorboard/**/*.0"))
    assert len(event_files) == 1, f"应有且仅有一个 event 文件，实际：{event_files}"
    accumulator = event_accumulator.EventAccumulator(str(event_files[0]))
    accumulator.Reload()
    rows: list[dict[str, Any]] = []
    for tag in sorted(accumulator.Tags().get("scalars", [])):
        for point in accumulator.Scalars(tag):
            rows.append(
                {
                    "tag": tag,
                    "step": point.step,
                    "wall_time": point.wall_time,
                    "value": point.value,
                }
            )
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tags = sorted({str(row["tag"]) for row in rows})
    result: dict[str, Any] = {}
    for tag in tags:
        values = [float(row["value"]) for row in rows if row["tag"] == tag]
        result[tag] = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "last": values[-1],
        }
    return result


def violation_total(audit: dict[str, Any]) -> int:
    values = audit.get("violations", {})
    assert isinstance(values, dict), f"审计 violations 格式错误：{values!r}"
    return sum(int(value) for value in values.values())


def m0_records(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for summary_path in sorted(root.glob("eval/*/real_*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        audit_path = summary_path.parent / "legality_audit.json"
        schedule_path = summary_path.parent / "schedule.csv"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        group = summary_path.parent.parent.name
        seed = int(group.removeprefix("temp0_seed")) if group.startswith("temp0_seed") else int(group.removeprefix("temp001_seed"))
        temperature = 0.0 if group.startswith("temp0_seed") else 0.01
        scheduled_real_tasks = int(audit["scheduled_real_tasks"])
        num_real_tasks = int(audit["num_real_tasks"])
        records.append(
            {
                "evaluation_group": group,
                "temperature": temperature,
                "seed": seed,
                "instance_id": summary_path.parent.name,
                "makespan": float(summary["makespan"]),
                "reward": float(summary["reward"]),
                "balance_std": float(summary["balance_std"]),
                "duration_sec": float(summary["duration_sec"]),
                "worker_utilization": float(summary["worker_utilization"]),
                "station_utilization": float(summary["station_utilization"]),
                "num_schedule_rows": int(audit["num_schedule_rows"]),
                "num_real_tasks": num_real_tasks,
                "scheduled_real_tasks": scheduled_real_tasks,
                "all_real_tasks_scheduled": scheduled_real_tasks == num_real_tasks,
                "total_hard_violations": violation_total(audit),
                "strictly_legal": bool(audit["is_resource_structurally_legal"])
                and bool(audit["is_legal_against_environment_duration"])
                and bool(audit["is_legal_against_current_data_duration"])
                and violation_total(audit) == 0,
                "schedule_file": schedule_path.relative_to(root).as_posix(),
                "audit_file": audit_path.relative_to(root).as_posix(),
            }
        )
    assert len(records) == 24, f"M0 应有 24 次评估，实际 {len(records)}"
    deterministic = [row for row in records if row["temperature"] == 0.0]
    stochastic = [row for row in records if row["temperature"] == 0.01]
    expected_instances = {"real_283", "real_680", "real_2338", "real_3182"}
    assert {row["instance_id"] for row in deterministic} == expected_instances
    assert len(stochastic) == 20
    assert {row["seed"] for row in stochastic} == {42, 43, 44, 45, 46}
    assert all(row["strictly_legal"] and row["all_real_tasks_scheduled"] for row in records)
    aggregate: list[dict[str, Any]] = []
    for instance_id in sorted(expected_instances):
        values = [row["makespan"] for row in stochastic if row["instance_id"] == instance_id]
        aggregate.append(
            {
                "instance_id": instance_id,
                "temperature": 0.01,
                "seeds": "42,43,44,45,46",
                "n": len(values),
                "makespan_mean": statistics.mean(values),
                "makespan_sample_std": statistics.stdev(values),
                "makespan_min": min(values),
                "makespan_max": max(values),
            }
        )
    integrity = {
        "expected_deterministic_runs": 4,
        "actual_deterministic_runs": len(deterministic),
        "expected_stochastic_runs": 20,
        "actual_stochastic_runs": len(stochastic),
        "stochastic_seeds": sorted({row["seed"] for row in stochastic}),
        "instances": sorted(expected_instances),
        "all_schedules_strictly_legal": all(row["strictly_legal"] for row in records),
        "all_real_tasks_scheduled": all(row["all_real_tasks_scheduled"] for row in records),
        "maximum_hard_violations": max(row["total_hard_violations"] for row in records),
        "strict_main_table_eligible": False,
        "reason": "screening_only 临时快速筛查，不得与正式训练/验证主表混合统计。",
    }
    write_csv(root / "validation_by_run.csv", records)
    write_csv(root / "summary_stochastic_temp001.csv", aggregate)
    write_json(root / "summary_stochastic_temp001.json", aggregate)
    write_json(root / "integrity_check.json", integrity)
    return records, integrity


def write_readme_m0(root: Path, records: list[dict[str, Any]], integrity: dict[str, Any]) -> None:
    deterministic = sorted((row for row in records if row["temperature"] == 0.0), key=lambda row: row["instance_id"])
    lines = [
        "# M0：full_joint warm-start 快速筛查",
        "",
        "- 实验身份：`screening_only`；用于快速观察主方法的改进方向，不是正式主方法结果，不进入 `experiment_master_results.csv`。",
        "- 初始化：严格加载 `joint100_full_joint_seed42_20260719` 的 best checkpoint；未恢复优化器状态。",
        "- 训练：实际记录 15 个 rollout/update 点（step 0–14）；best checkpoint 对应训练期 `real_680` 自动评估 makespan `449.596467 h`。",
        "- 独立评估：temperature=0、seed=42 覆盖四实例；temperature=0.01、seed=42–46 覆盖四实例，共 24 份排程。",
        f"- 合法性：24/24 完整排程，全部真实工序完成，最大硬约束违规 `{integrity['maximum_hard_violations']}`。",
        "- 可复核文件：`validation_by_run.csv`、`summary_stochastic_temp001.*`、`integrity_check.json`、`training_metrics.csv`、`file_manifest.json`。",
        "",
        "## temperature=0, seed=42",
        "",
        "| 实例 | makespan (h) |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {row['instance_id']} | {row['makespan']:.6f} |" for row in deterministic)
    lines.append("")
    lines.append("`strict_main_table_eligible=false`：原因是本实验明确为快速筛查，训练长度与模型选择协议均不构成正式比较证据。")
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme_m1(root: Path, metrics: dict[str, Any]) -> None:
    eval_metric = metrics["Eval/makespan"]
    lines = [
        "# M1：stable PPO warm-start 快速筛查",
        "",
        "- 实验身份：`screening_only`；仅用于快速筛查 PPO 稳定化方向，不进入正式主表。",
        "- 初始化：严格加载 `joint100_full_joint_seed42_20260719` 的 best checkpoint；未恢复优化器状态。",
        "- 训练：TensorBoard 有 14 个 rollout/update 点（step 0–13）、13 个训练期自动评估点；没有独立四实例排程验证。",
        f"- 训练期 Eval/makespan：最小 `{eval_metric['min']:.6f} h`，最后 `{eval_metric['last']:.6f} h`。",
        "- `strict_main_table_eligible=false`：缺少独立排程与硬约束回放审计，且属于临时快速筛查。",
        "- 可复核文件：`training_metrics.csv`、`training_metric_summary.json`、`integrity_check.json`、`file_manifest.json`。",
    ]
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def preserve_m0_download_differences(archive: Path) -> dict[str, Any]:
    """保留下载快照中与服务器归档不一致的文件，避免删除唯一版本。"""
    assert M0_DOWNLOAD_SNAPSHOT.is_dir(), f"M0 下载快照不存在：{M0_DOWNLOAD_SNAPSHOT}"
    canonical = source_manifest(M0_SOURCE)
    snapshot = source_manifest(M0_DOWNLOAD_SNAPSHOT)
    differing = sorted(key for key, value in snapshot.items() if canonical.get(key) != value)
    provenance = archive / "provenance" / "downloaded_snapshot_differences"
    for relative in differing:
        source_file = M0_DOWNLOAD_SNAPSHOT / relative
        target_file = provenance / relative
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target_file)
        assert sha256(source_file) == sha256(target_file)
    payload = {
        "snapshot_source": str(M0_DOWNLOAD_SNAPSHOT),
        "canonical_source": str(M0_SOURCE),
        "identical_file_count": len(snapshot) - len(differing),
        "differing_or_snapshot_only_files": differing,
        "preserved_path": str(provenance.relative_to(archive)),
    }
    write_json(archive / "provenance" / "downloaded_snapshot_comparison.json", payload)
    return payload


def write_file_manifest(root: Path) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": source_manifest(root),
    }
    write_json(root / "file_manifest.json", payload)


def archive_m0(remove_sources: bool) -> Path:
    target = ARCHIVE_ROOT / M0_SOURCE.name
    copy_and_verify(M0_SOURCE, target)
    preserve_m0_download_differences(target)
    records, integrity = m0_records(target)
    metrics = scalar_rows(target)
    write_csv(target / "training_metrics.csv", metrics)
    write_json(target / "training_metric_summary.json", metric_summary(metrics))
    write_readme_m0(target, records, integrity)
    write_file_manifest(target)
    if remove_sources:
        shutil.rmtree(M0_SOURCE)
        shutil.rmtree(M0_DOWNLOAD_SNAPSHOT)
    return target


def archive_m1(remove_sources: bool) -> Path:
    target = ARCHIVE_ROOT / "initial_main_screen_m1_stableppo_warmstart_seed42_20260723_163837"
    copy_and_verify(M1_SOURCE, target)
    metrics = scalar_rows(target)
    assert "Eval/makespan" in {row["tag"] for row in metrics}
    integrity = {
        "training_only": True,
        "tensorboard_scalar_count": len(metrics),
        "independent_schedule_validation_present": False,
        "strict_main_table_eligible": False,
        "reason": "screening_only 且未下载独立排程验证结果。",
    }
    write_csv(target / "training_metrics.csv", metrics)
    summary = metric_summary(metrics)
    write_json(target / "training_metric_summary.json", summary)
    write_json(target / "integrity_check.json", integrity)
    write_readme_m1(target, summary)
    write_file_manifest(target)
    if remove_sources:
        shutil.rmtree(M1_SOURCE)
    return target


def write_readme_m2(root: Path, metrics: dict[str, Any], best_sha256: str, last_sha256: str) -> None:
    """写入 M2 的训练筛查说明，避免训练期指标被误作正式验证。"""
    eval_metric = metrics["Eval/makespan"]
    lines = [
        "# M2：SCG 尺度门控上下文快速筛查",
        "",
        "- 实验身份：`screening_only`；仅测试 actor 全局上下文读出，不进入正式主表性能比较。",
        "- 初始化：从 `joint100_full_joint_seed42_20260719` 的 best checkpoint 加载共享权重；10 个 SCG 专属参数按筛查模块初始化，未恢复优化器状态。",
        "- 训练：14 个 rollout/update 点、13 个训练期自动评估点；最优 Eval/makespan=`483.337860 h`，最后=`492.940491 h`。",
        "- 结论：训练期最优值劣于 M0 full warm-start 的 `449.596467 h`，故为负向筛查证据；不建议合并到正式主方法。",
        "- 独立验证：本次下载包不含 `eval/`、`schedule.csv` 或 `legality_audit.json`，尚无四实例独立合法性验证。",
        f"- checkpoint：best SHA-256=`{best_sha256}`；last SHA-256=`{last_sha256}`。二者不同，均保留。",
        "- 可复核文件：`training_metrics.csv`、`training_metric_summary.json`、`integrity_check.json`、`copy_integrity_check.json`、`file_manifest.json`。",
    ]
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def archive_m2(remove_sources: bool) -> Path:
    """归档 M2 训练结果和此前导入失败的最小证据。"""
    target = ARCHIVE_ROOT / "initial_main_screen_m2_scg_context_warmstart_seed42_20260724_111651"
    copy_and_verify(M2_SOURCE, target)
    failed_target = target / "provenance" / "failed_launch_20260724_111016"
    assert M2_FAILED_LAUNCH.is_dir(), f"M2 失败启动记录不存在：{M2_FAILED_LAUNCH}"
    shutil.copytree(M2_FAILED_LAUNCH, failed_target)
    failed_before = source_manifest(M2_FAILED_LAUNCH)
    failed_after = source_manifest(failed_target)
    assert failed_before == failed_after, "M2 失败启动证据复制哈希不一致"

    metrics = scalar_rows(target)
    summary = metric_summary(metrics)
    assert "Eval/makespan" in summary, "M2 缺少训练期 Eval/makespan"
    best = target / "checkpoints" / "best.ckpt"
    last = target / "checkpoints" / "last.ckpt"
    assert best.is_file() and last.is_file(), "M2 缺少 best 或 last checkpoint"
    best_sha256 = sha256(best)
    last_sha256 = sha256(last)
    integrity = {
        "training_only": True,
        "screen_model": "scg",
        "training_update_points": len({int(row["step"]) for row in metrics if row["tag"] == "Rollout/AverageReward"}),
        "training_eval_points": int(summary["Eval/makespan"]["count"]),
        "training_best_eval_makespan": float(summary["Eval/makespan"]["min"]),
        "training_last_eval_makespan": float(summary["Eval/makespan"]["last"]),
        "independent_schedule_validation_present": False,
        "best_checkpoint_sha256": best_sha256,
        "last_checkpoint_sha256": last_sha256,
        "checkpoints_identical": best_sha256 == last_sha256,
        "strict_main_table_eligible": False,
        "reason": "SCG 快速筛查且未下载独立四实例排程与合法性审计；训练期自动评估不能替代正式验证。",
        "failed_launch_evidence": "provenance/failed_launch_20260724_111016",
    }
    write_csv(target / "training_metrics.csv", metrics)
    write_json(target / "training_metric_summary.json", summary)
    write_json(target / "integrity_check.json", integrity)
    write_readme_m2(target, summary, best_sha256, last_sha256)
    write_file_manifest(target)
    if remove_sources:
        shutil.rmtree(M2_SOURCE)
        shutil.rmtree(M2_FAILED_LAUNCH)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remove-sources", action="store_true", help="仅在复制与哈希校验成功后删除三个 results 根目录临时来源")
    parser.add_argument("--only", choices=("m0", "m1", "m2"), help="只归档指定筛查批次；用于中断后的安全续作")
    args = parser.parse_args()
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    targets: list[Path] = []
    if args.only in (None, "m0"):
        targets.append(archive_m0(args.remove_sources))
    if args.only in (None, "m1"):
        targets.append(archive_m1(args.remove_sources))
    if args.only in (None, "m2"):
        targets.append(archive_m2(args.remove_sources))
    print(json.dumps({"archived": [str(path) for path in targets], "removed_sources": args.remove_sources}, ensure_ascii=False))


if __name__ == "__main__":
    main()
