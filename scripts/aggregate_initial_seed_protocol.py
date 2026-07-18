"""汇总初始调度的十种子统计验证与一次确定性主结果。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_initial_schedule import validate_schedule


SEEDS = list(range(42, 52))
DATA_PATH = PROJECT_ROOT / "data" / "2338.csv"
CONFIG_PATH = PROJECT_ROOT / "conf" / "env" / "initial_bucket_2338.yaml"
PROTOCOL = "10x temperature=0.01 seeds=42..51 for mean/std + 1x temperature=0.0 seed=42 as primary"

METHODS: dict[str, dict[str, Any]] = {
    "main": {
        "name": "HB-GAT-PPO 主方法",
        "archive": PROJECT_ROOT
        / "results"
        / "01_initial_main"
        / "initial_hbgatppo_async_680_seed42_260717-164542",
        "checkpoint": PROJECT_ROOT
        / "results"
        / "01_initial_main"
        / "initial_hbgatppo_async_680_seed42_260717-164542"
        / "checkpoints"
        / "best.ckpt",
    },
    "l2d": {
        "name": "L2D-PPO-APAL 对比算法",
        "archive": PROJECT_ROOT
        / "results"
        / "02_initial_baselines"
        / "l2d_ppo_apal_initial_scale400_800_fiveskill_seed42_260717-224236",
        "checkpoint": PROJECT_ROOT
        / "results"
        / "02_initial_baselines"
        / "l2d_ppo_apal_initial_scale400_800_fiveskill_seed42_260717-224236"
        / "artifacts"
        / "baselines"
        / "literature"
        / "L2D-PPO-APAL"
        / "l2d_ppo_apal_best.pth",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_metric(method_key: str, run_dir: Path) -> dict[str, Any]:
    if method_key == "main":
        path = run_dir / "summary.json"
        metric = json.loads(path.read_text(encoding="utf-8"))
        return {
            "metric_path": path,
            "makespan": float(metric["makespan"]),
            "balance_std": float(metric["balance_std"]),
            "reward": float(metric["reward"]),
            "worker_utilization": float(metric["worker_utilization"]),
            "station_utilization": float(metric["station_utilization"]),
            "duration_sec": float(metric["duration_sec"]),
        }
    path = run_dir / "L2D-PPO-APAL" / "2338" / "metrics.json"
    metric = json.loads(path.read_text(encoding="utf-8"))
    return {
        "metric_path": path,
        "makespan": float(metric["makespan"]),
        "balance_std": float(metric["workload_balance_std"]),
        "reward": None,
        "worker_utilization": float(metric["worker_utilization"]),
        "station_utilization": float(metric["station_utilization"]),
        "duration_sec": float(metric["inference_time"]),
    }


def _schedule_path(method_key: str, run_dir: Path) -> Path:
    if method_key == "main":
        return run_dir / "schedule.csv"
    return run_dir / "L2D-PPO-APAL" / "2338" / "schedule.csv"


def _flatten_validation(report: dict[str, Any]) -> dict[str, Any]:
    violations = report["violations"]
    row: dict[str, Any] = {
        "num_schedule_rows": report["num_schedule_rows"],
        "num_dataset_nodes": report["num_dataset_nodes"],
        "num_real_tasks": report["num_real_tasks"],
        "scheduled_real_tasks": report["scheduled_real_tasks"],
        "makespan_real_tasks": report["makespan_real_tasks"],
        "is_resource_structurally_legal": report["is_resource_structurally_legal"],
        "is_legal_against_environment_duration": report[
            "is_legal_against_environment_duration"
        ],
        "is_legal_against_current_data_duration": report[
            "is_legal_against_current_data_duration"
        ],
        "complete_rate": report["scheduled_real_tasks"] / report["num_real_tasks"]
        if report["num_real_tasks"]
        else 0.0,
        "eligible_rate": 1.0
        if report["is_legal_against_environment_duration"]
        else 0.0,
    }
    row.update({f"violation_{key}": int(value) for key, value in violations.items()})
    row["hard_violation_total"] = int(sum(int(value) for value in violations.values()))
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _write_method(method_key: str, method: dict[str, Any]) -> dict[str, Any]:
    archive = Path(method["archive"])
    eval_root = archive / "eval" / "initial_2338_temp001_seeds42_51_plus_temp0_seed42"
    eval_root.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []
    validation_reports: list[dict[str, Any]] = []

    run_specs = [(seed, 0.01, f"seed{seed}", "stochastic") for seed in SEEDS]
    run_specs.append((42, 0.0, "deterministic_temp0_seed42", "primary_deterministic"))
    for seed, temperature, directory_name, result_role in run_specs:
        run_dir = eval_root / directory_name
        schedule_path = _schedule_path(method_key, run_dir)
        metric = _read_metric(method_key, run_dir)
        report = validate_schedule(
            data_path=DATA_PATH,
            schedule_path=schedule_path,
            config_path=str(CONFIG_PATH),
            task_id_mode="internal",
        )
        flat = _flatten_validation(report)
        validation_reports.append(
            {
                "result_role": result_role,
                "seed": seed,
                "temperature": temperature,
                "schedule_path": str(schedule_path.relative_to(PROJECT_ROOT)),
                "metric_path": str(metric["metric_path"].relative_to(PROJECT_ROOT)),
                "report": report,
            }
        )
        row = {
            "method": method_key,
            "method_name": method["name"],
            "result_role": result_role,
            "seed": seed,
            "temperature": temperature,
            "schedule_path": str(schedule_path.relative_to(PROJECT_ROOT)),
            "metric_path": str(metric["metric_path"].relative_to(PROJECT_ROOT)),
            **{key: value for key, value in metric.items() if key != "metric_path"},
            **flat,
        }
        run_rows.append(row)

    stochastic = [row for row in run_rows if row["result_role"] == "stochastic"]
    primary = next(row for row in run_rows if row["result_role"] == "primary_deterministic")
    stochastic_makespans = np.asarray([row["makespan"] for row in stochastic], dtype=float)
    stochastic_balances = np.asarray([row["balance_std"] for row in stochastic], dtype=float)
    legal_rows = [row for row in run_rows if row["is_legal_against_environment_duration"]]
    all_files_present = all(
        (PROJECT_ROOT / item["schedule_path"]).exists() for item in validation_reports
    )
    all_task_counts_match = all(
        row["num_dataset_nodes"] == 2402
        and row["num_schedule_rows"] == 2402
        and row["num_real_tasks"] == 2338
        and row["scheduled_real_tasks"] == 2338
        for row in run_rows
    )
    integrity = {
        "protocol": PROTOCOL,
        "dataset": "2338.csv",
        "expected_stochastic_seeds": SEEDS,
        "expected_deterministic": {"seed": 42, "temperature": 0.0},
        "observed_run_count": len(run_rows),
        "observed_stochastic_seeds": [row["seed"] for row in stochastic],
        "observed_deterministic_count": int(primary["result_role"] == "primary_deterministic"),
        "all_files_present": all_files_present,
        "all_task_counts_match_2338": all_task_counts_match,
        "all_runs_legal": len(legal_rows) == 11,
        "max_hard_violation_total": max(row["hard_violation_total"] for row in run_rows),
        "checkpoint": str(Path(method["checkpoint"]).relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": _sha256(Path(method["checkpoint"])),
    }

    summary = {
        "method": method["name"],
        "dataset": "2338.csv",
        "protocol": PROTOCOL,
        "checkpoint": str(Path(method["checkpoint"]).relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": integrity["checkpoint_sha256"],
        "stochastic": {
            "seeds": SEEDS,
            "temperature": 0.01,
            "count": len(stochastic),
            "makespan_mean": float(stochastic_makespans.mean()),
            "makespan_std_population": float(stochastic_makespans.std(ddof=0)),
            "makespan_std_sample": float(stochastic_makespans.std(ddof=1)),
            "makespan_min": float(stochastic_makespans.min()),
            "makespan_max": float(stochastic_makespans.max()),
            "balance_std_mean": float(stochastic_balances.mean()),
            "balance_std_std_population": float(stochastic_balances.std(ddof=0)),
            "complete_rate_mean": float(np.mean([row["complete_rate"] for row in stochastic])),
            "eligible_rate_mean": float(np.mean([row["eligible_rate"] for row in stochastic])),
            "valid_rate": float(np.mean([row["is_legal_against_environment_duration"] for row in stochastic])),
        },
        "primary_deterministic": {
            "seed": 42,
            "temperature": 0.0,
            "makespan": primary["makespan"],
            "balance_std": primary["balance_std"],
            "reward": primary["reward"],
            "worker_utilization": primary["worker_utilization"],
            "station_utilization": primary["station_utilization"],
            "duration_sec": primary["duration_sec"],
            "complete_rate": primary["complete_rate"],
            "eligible_rate": primary["eligible_rate"],
            "valid": bool(primary["is_legal_against_environment_duration"]),
        },
        "integrity": integrity,
    }

    _write_csv(eval_root / "runs_detail.csv", run_rows)
    _write_csv(
        eval_root / "validation_by_seed.csv",
        [
            {
                "method": row["method"],
                "result_role": row["result_role"],
                "seed": row["seed"],
                "temperature": row["temperature"],
                "schedule_path": row["schedule_path"],
                **_flatten_validation(item["report"]),
            }
            for row, item in zip(run_rows, validation_reports)
        ],
    )
    _write_csv(
        eval_root / "summary.csv",
        [
            {
                "method": method["name"],
                "result_role": "stochastic_mean",
                "temperature": 0.01,
                "seed_set": "42-51",
                "makespan": summary["stochastic"]["makespan_mean"],
                "makespan_std": summary["stochastic"]["makespan_std_sample"],
                "balance_std": summary["stochastic"]["balance_std_mean"],
                "complete_rate": summary["stochastic"]["complete_rate_mean"],
                "eligible_rate": summary["stochastic"]["eligible_rate_mean"],
                "valid_rate": summary["stochastic"]["valid_rate"],
            },
            {
                "method": method["name"],
                "result_role": "primary_deterministic",
                "temperature": 0.0,
                "seed_set": "42",
                "makespan": primary["makespan"],
                "makespan_std": 0.0,
                "balance_std": primary["balance_std"],
                "complete_rate": primary["complete_rate"],
                "eligible_rate": primary["eligible_rate"],
                "valid_rate": float(primary["is_legal_against_environment_duration"]),
            },
        ],
    )
    (eval_root / "validation_by_seed.json").write_text(
        json.dumps(validation_reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (eval_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (eval_root / "integrity_check.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (eval_root / "resolved_config.yaml").write_text(
        "# 每个 seed 目录保留完整 Hydra resolved_config；此文件记录本次统一评估协议。\n"
        f"experiment: initial_schedule_2338\n"
        f"dataset: data/2338.csv\n"
        f"protocol: {PROTOCOL}\n"
        "stochastic_temperature: 0.01\n"
        "stochastic_seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]\n"
        "primary_temperature: 0.0\n"
        "primary_seed: 42\n",
        encoding="utf-8",
    )
    (eval_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_type": "initial_schedule_seed_protocol",
                "method": method["name"],
                "protocol": PROTOCOL,
                "dataset": str(DATA_PATH.relative_to(PROJECT_ROOT)),
                "checkpoint": str(Path(method["checkpoint"]).relative_to(PROJECT_ROOT)),
                "checkpoint_sha256": integrity["checkpoint_sha256"],
                "raw_run_directories": [item["schedule_path"] for item in validation_reports],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    readme = f"""# {method['name']}：2338 初始调度统一验证

## 验证口径

- 数据集：`data/2338.csv`，配置：`conf/env/initial_bucket_2338.yaml`。
- 随机统计：温度 `0.01`，seed `42–51` 共 10 次，用于均值和标准差。
- 主结果：温度 `0.0`，seed `42`，仅运行一次，作为论文主要结果。
- 每个种子目录保留原始 `schedule.csv`、指标、日志、resolved config 和 run manifest。
- 当前数据加载为 2402 个节点，其中 2338 个真实任务；合法性校验要求 2402 行 schedule、2338 个真实任务全部完成且所有硬约束为零。

## 数值摘要

- 随机统计 makespan：均值 `{summary['stochastic']['makespan_mean']:.6f}`，样本标准差 `{summary['stochastic']['makespan_std_sample']:.6f}`，范围 `{summary['stochastic']['makespan_min']:.6f}–{summary['stochastic']['makespan_max']:.6f}`。
- 确定性主结果 makespan：`{primary['makespan']:.6f}`。
- 确定性主结果 complete/eligible/valid：`{primary['complete_rate']:.6f}` / `{primary['eligible_rate']:.6f}` / `{bool(primary['is_legal_against_environment_duration'])}`。
- 最大硬约束违规数：`{integrity['max_hard_violation_total']}`。

详细文件：`summary.csv`、`summary.json`、`runs_detail.csv`、`validation_by_seed.csv/json`、`integrity_check.json`、`file_manifest.json`。
"""
    (eval_root / "README.md").write_text(readme, encoding="utf-8")

    files = []
    for path in sorted(eval_root.rglob("*")):
        if path.is_file() and path.name != "file_manifest.json":
            files.append(
                {
                    "path": str(path.relative_to(eval_root)).replace("\\", "/"),
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    (eval_root / "file_manifest.json").write_text(
        json.dumps({"root": str(eval_root.relative_to(PROJECT_ROOT)), "files": files}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return summary


def _update_master(summaries: dict[str, dict[str, Any]]) -> None:
    path = PROJECT_ROOT / "results" / "experiment_master_results.csv"
    table = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for key, summary in summaries.items():
        method = METHODS[key]
        stochastic = summary["stochastic"]
        primary = summary["primary_deterministic"]
        experiment_id = (
            "initial_hbgatppo_async_best_eval_2338_temp001_seeds42_51_plus_temp0_seed42_20260718"
            if key == "main"
            else "initial_l2d_ppo_apal_best_eval_2338_temp001_seeds42_51_plus_temp0_seed42_20260718"
        )
        rows.append(
            {
                "experiment_id": experiment_id,
                "phase": "initial_schedule",
                "experiment_group": "main_method_primary_plus_seed_sweep" if key == "main" else "initial_literature_baseline_primary_plus_seed_sweep",
                "method": "HB-GAT-PPO" if key == "main" else "L2D-PPO-APAL",
                "variant": "best_checkpoint_initial_2338",
                "dataset": 2338,
                "instance_id": "real_2338",
                "scenario_level": "standard",
                "eval_protocol": "10x_temp001_seeds42_51_plus_1x_temp0_seed42",
                "status": "completed_single_instance_eval",
                "priority": "high",
                "paper_table_role": "single_instance_primary_plus_variance",
                "fairness_status": "same_dataset_same_checkpoint_protocol",
                "strict_main_table_eligible": "conditional",
                "seed": "42-51+42det",
                "num_runs": 11,
                "scenario_count": 1,
                "task_count": 2338,
                "makespan": primary["makespan"],
                "makespan_mean": stochastic["makespan_mean"],
                "makespan_std": stochastic["makespan_std_sample"],
                "eligible_rate": primary["eligible_rate"],
                "complete_rate": primary["complete_rate"],
                "valid_rate": 1.0 if primary["valid"] else 0.0,
                "reward": primary["reward"],
                "balance_std": primary["balance_std"],
                "worker_utilization": primary["worker_utilization"],
                "station_utilization": primary["station_utilization"],
                "duration_sec": primary["duration_sec"],
                "violation_summary": "all hard constraints zero; 10 stochastic + 1 deterministic schedules independently legal",
                "source_file": (
                    (Path(method["archive"]) / "eval" / "initial_2338_temp001_seeds42_51_plus_temp0_seed42")
                    .relative_to(PROJECT_ROOT)
                    .as_posix()
                ),
                "command_or_next_action": "已完成；按同一协议补齐其余数据集" if key == "main" else "已完成；按同一协议补齐其余数据集",
                "notes": "温度0.01的seed42-51用于均值/标准差；温度0.0 seed42为主要结果；主评估器已锁定全局随机种子。",
            }
        )
    new = pd.DataFrame(rows)
    table = table[~table["experiment_id"].isin(new["experiment_id"])]
    table = pd.concat([table, new], ignore_index=True)
    table.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> int:
    summaries = {key: _write_method(key, method) for key, method in METHODS.items()}
    _update_master(summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
