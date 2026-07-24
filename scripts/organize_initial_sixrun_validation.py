"""整理并审计初始调度六次验证结果。

输入为 baselines/ 下的 local_only 和 static_topq 临时验证目录，输出到对应
results/01_initial_main/ 消融归档的 eval/ 子目录，并对每个 schedule 执行独立
环境合法性回放。
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from validate_initial_schedule import validate_schedule


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "local_only": PROJECT_ROOT / "results/01_initial_main/ablation_joint100_local_only_seed42_20260721",
    "static_topq": PROJECT_ROOT / "results/01_initial_main/ablation_joint100_static_topq_seed42_20260721",
}
DATASETS = ("283", "680", "2338", "3182")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_and_audit(source: Path, target: Path) -> dict[str, Any]:
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=True)
    entries: list[dict[str, Any]] = []
    for source_file in sorted(path for path in source.rglob("*") if path.is_file()):
        relative = source_file.relative_to(source)
        target_file = target / relative
        entries.append(
            {
                "relative_path": relative.as_posix(),
                "source_size": source_file.stat().st_size,
                "archive_size": target_file.stat().st_size if target_file.exists() else None,
                "source_sha256": sha256(source_file),
                "archive_sha256": sha256(target_file) if target_file.exists() else None,
                "sha256_equal": target_file.exists() and sha256(source_file) == sha256(target_file),
            }
        )
    return {
        "source": str(source.resolve()),
        "archive": str(target.resolve()),
        "raw_file_count": len(entries),
        "all_equal": all(item["sha256_equal"] for item in entries),
        "files": entries,
    }


def parse_run_name(name: str) -> tuple[float, int]:
    if name == "temp0_seed42":
        return 0.0, 42
    prefix, raw_seed = name.split("_seed")
    return 0.01 if prefix == "temp001" else float(prefix.replace("temp", "")), int(raw_seed)


def main() -> int:
    generated_at = datetime.now().astimezone().isoformat()
    all_model_summaries: list[dict[str, Any]] = []
    for model, archive in MODELS.items():
        eval_root = archive / "eval"
        model_records: list[dict[str, Any]] = []
        copy_audits: list[dict[str, Any]] = []
        for dataset in DATASETS:
            source = PROJECT_ROOT / f"baselines/initial_{model}_sixrun__real_{dataset}_20260721"
            if not source.exists():
                raise FileNotFoundError(source)
            target = eval_root / source.name
            copy_audits.append(copy_and_audit(source, target))
            runs: list[dict[str, Any]] = []
            for run_dir in sorted(path for path in target.iterdir() if path.is_dir()):
                schedule = run_dir / "schedule.csv"
                summary_path = run_dir / "summary.json"
                if not schedule.exists() or not summary_path.exists():
                    raise FileNotFoundError(f"缺少验证文件: {run_dir}")
                temperature, seed = parse_run_name(run_dir.name)
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                validation_error: str | None = None
                try:
                    legality = validate_schedule(
                        data_path=PROJECT_ROOT / f"data/{dataset}.csv",
                        schedule_path=schedule,
                    )
                except Exception as exc:  # noqa: BLE001 - 单个损坏结果不能阻断其余场景归档
                    validation_error = f"{type(exc).__name__}: {exc}"
                    legality = {
                        "is_legal_against_environment_duration": False,
                        "is_resource_structurally_legal": False,
                        "violations": {"validation_error": 1},
                        "validation_error": validation_error,
                    }
                record = {
                    "model": model,
                    "dataset": dataset,
                    "run": run_dir.name,
                    "temperature": temperature,
                    "seed": seed,
                    "makespan": float(summary["makespan"]),
                    "scheduled_tasks": int(summary["scheduled_tasks"]),
                    "balance_std": float(summary["balance_std"]),
                    "reward": float(summary["reward"]),
                    "duration_sec": float(summary["duration_sec"]),
                    "worker_utilization": float(summary["worker_utilization"]),
                    "station_utilization": float(summary["station_utilization"]),
                    "schedule_path": str(schedule.relative_to(archive).as_posix()),
                    "legality_path": str((run_dir / "legality_audit.json").relative_to(archive).as_posix()),
                    "is_legal": bool(legality["is_legal_against_environment_duration"]),
                    "resource_structurally_legal": bool(legality["is_resource_structurally_legal"]),
                    "violations": legality["violations"],
                    "validation_error": validation_error,
                    "legality": legality,
                }
                (run_dir / "legality_audit.json").write_text(
                    json.dumps(legality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                runs.append(record)
                model_records.append(record)

            deterministic = next(item for item in runs if item["temperature"] == 0.0 and item["seed"] == 42)
            stochastic = [item for item in runs if item["temperature"] == 0.01 and 42 <= item["seed"] <= 46]
            valid_stochastic = [item for item in stochastic if item["is_legal"]]
            aggregate = {
                "model": model,
                "dataset": dataset,
                "protocol": "5x temperature=0.01 seeds=42..46 + 1x temperature=0.0 seed=42",
                "run_count": len(runs),
                "stochastic_run_count": len(stochastic),
                "stochastic_valid_run_count": len(valid_stochastic),
                "deterministic_makespan": deterministic["makespan"],
                "deterministic_is_legal": deterministic["is_legal"],
                "stochastic_makespan_mean": mean(item["makespan"] for item in valid_stochastic) if valid_stochastic else None,
                "stochastic_makespan_std": stdev(item["makespan"] for item in valid_stochastic) if len(valid_stochastic) >= 2 else None,
                "all_schedules_legal": all(item["is_legal"] for item in runs),
                "all_schedules_structurally_legal": all(item["resource_structurally_legal"] for item in runs),
                "scheduled_tasks": sorted({item["scheduled_tasks"] for item in runs}),
                "runs": runs,
            }
            (eval_root / f"summary_{model}_real_{dataset}_sixrun.json").write_text(
                json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            all_model_summaries.append(aggregate)

        csv_path = eval_root / f"summary_{model}_sixrun.csv"
        fields = [
            "model", "dataset", "protocol", "run_count", "stochastic_run_count",
            "stochastic_valid_run_count", "deterministic_is_legal",
            "deterministic_makespan", "stochastic_makespan_mean", "stochastic_makespan_std",
            "all_schedules_legal", "all_schedules_structurally_legal", "scheduled_tasks",
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for item in all_model_summaries:
                if item["model"] == model:
                    row = {key: item[key] for key in fields}
                    row["scheduled_tasks"] = ";".join(str(value) for value in item["scheduled_tasks"])
                    writer.writerow(row)

        all_files = []
        for path in sorted(eval_root.rglob("*")):
            if path.is_file():
                all_files.append({
                    "relative_path": path.relative_to(archive).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256(path),
                })
        (eval_root / f"file_manifest_{model}_sixrun.json").write_text(
            json.dumps({"generated_at": generated_at, "files": all_files}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (eval_root / f"copy_integrity_check_{model}_sixrun.json").write_text(
            json.dumps({"generated_at": generated_at, "sources": copy_audits, "all_equal": all(item["all_equal"] for item in copy_audits)}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        integrity = {
            "generated_at": generated_at,
            "model": model,
            "dataset_count": len(DATASETS),
            "run_count": len(model_records),
            "expected_run_count": 24,
            "all_schedules_legal": all(item["is_legal"] for item in model_records),
            "all_schedules_structurally_legal": all(item["resource_structurally_legal"] for item in model_records),
            "max_violation_by_type": {
                key: max(int(item["violations"].get(key, 0)) for item in model_records)
                for key in sorted({key for item in model_records for key in item["violations"]})
            },
            "protocol": "temperature=0.01 seeds=42..46 + temperature=0.0 seed=42",
            "source_directories": [item["source"] for item in copy_audits],
        }
        (eval_root / f"integrity_check_{model}_sixrun.json").write_text(
            json.dumps(integrity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        readme = (
            f"# {model} 初始调度六次验证\n\n"
            "协议：temperature=0.01、seed=42–46 五次随机验证；temperature=0.0、seed=42 一次确定性主结果。\n\n"
            f"本批次覆盖 real_283、real_680、real_2338、real_3182，共 {len(model_records)} 个 schedule。\n"
            f"独立合法性回放结果：{'全部通过' if integrity['all_schedules_legal'] else '存在失败，见 integrity_check'}。\n"
        )
        (eval_root / f"README_{model}_sixrun.md").write_text(readme, encoding="utf-8")

    (PROJECT_ROOT / "results/01_initial_main/initial_ablation_sixrun_eval_summary.json").write_text(
        json.dumps({"generated_at": generated_at, "results": all_model_summaries}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"models": list(MODELS), "datasets": list(DATASETS), "records": len(all_model_summaries) * 6}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
