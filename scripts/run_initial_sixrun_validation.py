"""执行、续跑并归档初始调度四实例六次验证协议。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.checkpoints import load_checkpoint


DATASETS = ("283", "680", "2338", "3182")
RUNS = (("temp0_seed42", 0.0, 42),) + tuple(
    (f"temp001_seed{seed}", 0.01, seed) for seed in range(42, 47)
)
PROTOCOL_ID = "initial_real4_sixrun_temp0_s42_temp001_s42_46_v1"
EXPECTED_RUN_COUNT = len(DATASETS) * len(RUNS)


@dataclass(frozen=True)
class ValidationProtocol:
    """不可变的初始调度验证协议，避免运行期混用不同 run 集合。"""

    protocol_id: str
    runs: tuple[tuple[str, float, int], ...]
    expected_run_count: int


def validation_protocol(*, deterministic_only: bool) -> ValidationProtocol:
    """返回标准六次协议或用于 checkpoint 迁移核验的四次确定性协议。"""
    selected_runs = RUNS[:1] if deterministic_only else RUNS
    protocol_id = (
        "initial_real4_temp0_seed42_v1"
        if deterministic_only
        else PROTOCOL_ID
    )
    return ValidationProtocol(
        protocol_id=protocol_id,
        runs=selected_runs,
        expected_run_count=len(DATASETS) * len(selected_runs),
    )


def utc_now() -> str:
    """返回可审计的 UTC 时间戳。"""
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    """计算文件哈希，用于锁定 checkpoint 与全部产物。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    """原子写入 JSON，避免监控进程读到半截 progress 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    """读取必须为对象的 JSON 产物。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须为对象：{path}")
    return payload


def run_key(dataset: str, run_name: str) -> str:
    return f"real_{dataset}/{run_name}"


def _finite_metric(payload: dict[str, Any], key: str) -> float:
    value = float(payload[key])
    if not math.isfinite(value):
        raise ValueError(f"指标不是有限数：{key}={value}")
    return value


def verify_completed_run(run_dir: Path) -> tuple[bool, str, dict[str, Any] | None]:
    """确认一次评估同时具备完整排程、指标和通过的合法性审计。"""
    schedule_path = run_dir / "schedule.csv"
    summary_path = run_dir / "summary.json"
    audit_path = run_dir / "legality_audit.json"
    required_paths = (schedule_path, summary_path, audit_path)
    missing = [str(path.name) for path in required_paths if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        return False, f"缺失或空文件：{', '.join(missing)}", None
    try:
        summary = read_json(summary_path)
        audit = read_json(audit_path)
        metrics = {
            key: _finite_metric(summary, key)
            for key in (
                "makespan",
                "reward",
                "balance_std",
                "duration_sec",
                "worker_utilization",
                "station_utilization",
            )
        }
        scheduled = int(audit["scheduled_real_tasks"])
        total = int(audit["num_real_tasks"])
        legal = bool(audit["is_legal_against_current_data_duration"])
        violations = audit.get("violations", {})
        if not isinstance(violations, dict):
            raise ValueError("violations 必须为对象")
        max_hard_violation = max((int(value) for value in violations.values()), default=0)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return False, f"指标或审计格式无效：{type(exc).__name__}: {exc}", None
    if scheduled != total:
        return False, f"排程不完整：scheduled_real_tasks={scheduled}, num_real_tasks={total}", None
    if not legal:
        return False, "合法性审计未通过", None
    if max_hard_violation != 0:
        return False, f"存在硬约束违规：max_hard_violation={max_hard_violation}", None
    # 一致性校验：summary 的 makespan 必须等于 schedule 的最大完工时刻，防止
    # 根汇总与逐次产物不一致（2026-08-19 ent40 real_680 事故：per-run summary 被改写为
    # 458.51 而 schedule.csv 仍为 525.51，两版结果混装且未被发现）。
    try:
        with schedule_path.open("r", encoding="utf-8-sig", newline="") as handle:
            ends = [
                float(row["End"])
                for row in csv.DictReader(handle)
                if str(row.get("Team", "")).strip() != "[]"
            ]
        schedule_makespan = max(ends) if ends else float("nan")
    except (OSError, KeyError, ValueError, TypeError) as exc:
        return False, f"schedule.csv 解析失败：{type(exc).__name__}: {exc}", None
    if not math.isclose(schedule_makespan, metrics["makespan"], rel_tol=1e-9, abs_tol=1e-3):
        return False, (
            f"summary makespan 与 schedule 最大完工时刻不一致："
            f"summary={metrics['makespan']}, schedule_max_end={schedule_makespan}"
        ), None
    return True, "ok", {
        **metrics,
        "complete": True,
        "legal": True,
        "max_hard_violation": max_hard_violation,
        "schedule_sha256": sha256(schedule_path),
    }


def archive_incomplete_run(output_root: Path, run_dir: Path) -> None:
    """保留不完整尝试，绝不直接删除其日志或排程。"""
    if not run_dir.exists():
        return
    relative = run_dir.relative_to(output_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = output_root / "incomplete_attempts" / relative / stamp
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), str(target))


def run_command(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    """执行单次评估；完整输出落盘，失败立即停止且不自动重试。"""
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"命令失败，exit={completed.returncode}，日志：{log_path}")


def _source_files(source_run_dir: Path) -> tuple[Path, Path]:
    config_path = source_run_dir / "resolved_config.yaml"
    manifest_path = source_run_dir / "run_manifest.json"
    for path in (config_path, manifest_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"训练归档缺少必要证据文件：{path}")
    return config_path, manifest_path


def _launch_contract(
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    source_run_dir: Path,
    source_config: Path,
    source_manifest: Path,
    checkpoint_format: str,
    model_spec: dict[str, Any],
    protocol: ValidationProtocol,
    extra_eval_args: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "run_type": "initial_schedule_sixrun_protocol",
        "protocol_id": protocol.protocol_id,
        "method": "HB-GAT-PPO",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_format": checkpoint_format,
        "model_spec": model_spec,
        "extra_eval_args": list(extra_eval_args),
        "source_run_dir": str(source_run_dir),
        "source_resolved_config_sha256": sha256(source_config),
        "source_run_manifest_sha256": sha256(source_manifest),
        "datasets": [f"real_{dataset}" for dataset in DATASETS],
        "runs": [
            {"run_name": name, "temperature": temperature, "seed": seed}
            for name, temperature, seed in protocol.runs
        ],
        "expected_run_count": protocol.expected_run_count,
        "formal_protocol": True,
        "strict_main_table_eligible": False,
        "evidence_level": "completed_conditional_after_success",
        "eligibility_note": "四个报告实例参与 checkpoint 选择；本地六次验证不是独立 held-out 测试。",
    }


def _same_launch_contract(existing: dict[str, Any], expected: dict[str, Any]) -> bool:
    keys = (
        "protocol_id",
        "checkpoint_sha256",
        "source_resolved_config_sha256",
        "source_run_manifest_sha256",
        "datasets",
        "runs",
        "expected_run_count",
        "extra_eval_args",
    )
    return all(existing.get(key) == expected.get(key) for key in keys)


def _copy_file_atomic(source: Path, destination: Path) -> None:
    """先写临时文件再原子替换，避免并行进程共享 output_root 时读到半截文件。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def _write_static_evidence(
    output_root: Path,
    source_config: Path,
    source_manifest: Path,
    launch_contract: dict[str, Any],
) -> None:
    _copy_file_atomic(source_config, output_root / "resolved_config.yaml")
    _copy_file_atomic(source_manifest, output_root / "source_run_manifest.json")
    write_json(output_root / "launch_manifest.json", {**launch_contract, "started_at": utc_now()})


def _build_detail_rows(
    output_root: Path,
    *,
    protocol: ValidationProtocol,
) -> list[dict[str, Any]]:
    detail_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for run_name, temperature, seed in protocol.runs:
            run_dir = output_root / f"real_{dataset}" / run_name
            valid, reason, payload = verify_completed_run(run_dir)
            if not valid or payload is None:
                raise RuntimeError(f"汇总前完整性检查失败：{run_key(dataset, run_name)}：{reason}")
            detail_rows.append(
                {
                    "instance_id": f"real_{dataset}",
                    "run_name": run_name,
                    "temperature": temperature,
                    "seed": seed,
                    **payload,
                }
            )
    return detail_rows


def _write_final_outputs(
    output_root: Path,
    launch_contract: dict[str, Any],
    progress: dict[str, Any],
    *,
    protocol: ValidationProtocol,
) -> None:
    detail_rows = _build_detail_rows(output_root, protocol=protocol)
    with (output_root / "runs_detail.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)

    summary_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        current = [row for row in detail_rows if row["instance_id"] == f"real_{dataset}"]
        deterministic = next(row for row in current if row["temperature"] == 0.0 and row["seed"] == 42)
        stochastic = [row["makespan"] for row in current if row["temperature"] == 0.01]
        summary_rows.append(
            {
                "instance_id": f"real_{dataset}",
                "deterministic_makespan_temp0_seed42": deterministic["makespan"],
                "stochastic_makespan_temp001_mean": mean(stochastic) if stochastic else None,
                "stochastic_makespan_temp001_std_sample": stdev(stochastic) if len(stochastic) >= 2 else None,
                "eligible_rate": mean(float(row["legal"]) for row in current),
                "complete_rate": mean(float(row["complete"]) for row in current),
                "max_hard_violation": max(int(row["max_hard_violation"]) for row in current),
            }
        )
    with (output_root / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    integrity = {
        "protocol_id": protocol.protocol_id,
        "expected_run_count": protocol.expected_run_count,
        "observed_run_count": len(detail_rows),
        "instance_count": len(DATASETS),
        "runs_per_instance": len(protocol.runs),
        "all_complete": all(row["complete"] for row in detail_rows),
        "all_legal": all(row["legal"] for row in detail_rows),
        "max_hard_violation": max(int(row["max_hard_violation"]) for row in detail_rows),
        "checkpoint_sha256": launch_contract["checkpoint_sha256"],
        "strict_main_table_eligible": False,
        "evidence_level": "completed_conditional",
    }
    write_json(
        output_root / "summary.json",
        {
            "method": "HB-GAT-PPO",
            "variant": "conditional_team_gate_formal_v1",
            "checkpoint_sha256": launch_contract["checkpoint_sha256"],
            "protocol_id": protocol.protocol_id,
            "protocol": "每实例 6 次：temperature=0、seed=42；temperature=0.01、seed=42–46。",
            "rows": summary_rows,
            "strict_main_table_eligible": False,
            "evidence_level": "completed_conditional",
            "note": "四个报告实例参与 checkpoint 选择；本结果不是独立 held-out 测试。",
        },
    )
    write_json(output_root / "integrity_check.json", integrity)
    write_json(
        output_root / "run_manifest.json",
        {
            **launch_contract,
            "completed_at": utc_now(),
            "integrity": integrity,
        },
    )
    (output_root / "README.md").write_text(
        "# HB-GAT-PPO 初始调度四实例六次验证\n\n"
        "- 固定实例：real_283、real_680、real_2338、real_3182。\n"
        "- 每实例 6 次：temperature=0、seed=42 为确定性结果；temperature=0.01、seed=42–46 用于均值和样本标准差。\n"
        "- 每次均导出任务级 schedule，并用 `validate_initial_schedule.py` 回放审核。\n"
        "- 验证使用训练期四实例选择出的 episode 20 `best.ckpt`；因此为完整、可审计但非 held-out 的 conditional 证据。\n"
        "- 不生成 Gantt 图，以降低本机验证时的文件数量和耗时。\n",
        encoding="utf-8",
    )
    progress.update({"status": "completed", "current": None, "failed": None, "completed_at": utc_now()})
    write_json(output_root / "progress.json", progress)
    manifest_rows = [
        {"path": path.relative_to(output_root).as_posix(), "size": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "file_manifest.json"
    ]
    write_json(output_root / "file_manifest.json", {"root": str(output_root), "files": manifest_rows})


def main() -> int:
    parser = argparse.ArgumentParser(description="运行初始调度四实例、每实例六次的正式验证协议")
    parser.add_argument("--checkpoint", type=Path, required=True, help="仅允许使用 best checkpoint")
    parser.add_argument("--expected-checkpoint-sha256", required=True, help="checkpoint 的预期 SHA-256")
    parser.add_argument("--source-run-dir", type=Path, required=True, help="训练归档目录，必须含 resolved_config.yaml")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--resume", action="store_true", help="只续跑缺失、损坏或审计失败的运行")
    parser.add_argument(
        "--deterministic-only",
        action="store_true",
        help="只运行四实例 temperature=0、seed=42；用于 checkpoint 参数态迁移复验",
    )
    parser.add_argument(
        "--extra-eval-args",
        nargs="*",
        default=[],
        help="追加到每次 evaluate_model.py 的额外 key=value（用于与 checkpoint 的 "
        "model_spec 语义对齐，例如 team_selection_mode=autoregressive_pressure_v2）；"
        "默认空，不影响既有模型的验证命令",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅校验输入与协议，不执行评估")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=None,
        help="只验证指定的实例子集（如 --datasets 283 680）；缺省验证全部四个实例。"
        "子集模式只执行评估、不写根级汇总文件，适合多终端并行；"
        "全部实例完成后用不带 --datasets 的完整命令 --resume 一次性生成汇总。",
    )
    args = parser.parse_args()

    # 并行模式：允许 --datasets 选择实例子集；协议身份（launch_contract 的
    # datasets/expected_run_count）始终为完整四实例，保证 --resume 汇总时契约一致。
    selected_datasets = tuple(args.datasets) if args.datasets else DATASETS
    if not selected_datasets:
        raise ValueError("--datasets 不能为空")
    partial_run = set(selected_datasets) != set(DATASETS)
    partial_total = len(selected_datasets) * len(RUNS[:1] if args.deterministic_only else RUNS)

    protocol = validation_protocol(deterministic_only=bool(args.deterministic_only))

    checkpoint = args.checkpoint.resolve()
    source_run_dir = args.source_run_dir.resolve()
    output_root = args.output_root.resolve()
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise FileNotFoundError(f"checkpoint 不存在或为空：{checkpoint}")
    source_config, source_manifest = _source_files(source_run_dir)
    for dataset in selected_datasets:
        data_path = PROJECT_ROOT / "data" / f"{dataset}.csv"
        if not data_path.is_file() or data_path.stat().st_size <= 0:
            raise FileNotFoundError(f"缺少验证数据：{data_path}")
    checkpoint_sha = sha256(checkpoint)
    if checkpoint_sha.casefold() != str(args.expected_checkpoint_sha256).strip().casefold():
        raise ValueError(
            "checkpoint SHA-256 不匹配："
            f"expected={args.expected_checkpoint_sha256} actual={checkpoint_sha}"
        )
    loaded_checkpoint = load_checkpoint(checkpoint, map_location="cpu")
    launch_contract = _launch_contract(
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        source_run_dir=source_run_dir,
        source_config=source_config,
        source_manifest=source_manifest,
        checkpoint_format=loaded_checkpoint.format_name,
        model_spec=asdict(loaded_checkpoint.model_spec),
        protocol=protocol,
        extra_eval_args=tuple(args.extra_eval_args),
    )

    if args.dry_run:
        if output_root.exists():
            raise FileExistsError(f"dry-run 要求目标目录尚不存在：{output_root}")
        print(
            f"[预检通过] protocol={protocol.protocol_id} runs={protocol.expected_run_count} "
            f"checkpoint_sha256={checkpoint_sha}",
            flush=True,
        )
        return 0

    if output_root.exists():
        # 并行子集模式下多个进程共享同一 output_root，允许直接续跑已存在目录；
        # 完整模式仍要求 --resume，避免误覆盖已有验证结果。
        if not args.resume and not partial_run:
            raise FileExistsError(f"输出目录已经存在；请使用 --resume：{output_root}")
        launch_path = output_root / "launch_manifest.json"
        if launch_path.is_file():
            if not _same_launch_contract(read_json(launch_path), launch_contract):
                raise ValueError("续跑目录的 checkpoint、训练配置或四实例协议与本次不一致")
        elif not args.resume and not partial_run:
            raise FileNotFoundError(f"不能续跑无 launch_manifest 的目录：{output_root}")
    else:
        if args.resume:
            raise FileNotFoundError(f"--resume 指定的输出目录不存在：{output_root}")
        output_root.mkdir(parents=True)
        _write_static_evidence(output_root, source_config, source_manifest, launch_contract)

    environment = os.environ.copy()
    environment.update({name: "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    progress_path = output_root / "progress.json"
    completed: list[dict[str, Any]] = []
    for dataset in selected_datasets:
        for run_name, _temperature, _seed in protocol.runs:
            valid, _reason, payload = verify_completed_run(output_root / f"real_{dataset}" / run_name)
            if valid and payload is not None:
                completed.append({"key": run_key(dataset, run_name), **payload})
    progress: dict[str, Any] = {
        "status": "running",
        "protocol_id": protocol.protocol_id,
        "expected_run_count": protocol.expected_run_count,
        "completed_run_count": len(completed),
        "completed": completed,
        "current": None,
        "failed": None,
        "started_or_resumed_at": utc_now(),
    }
    if not partial_run:
        write_json(progress_path, progress)

    total_runs = partial_total if partial_run else protocol.expected_run_count
    try:
        for dataset in selected_datasets:
            for run_name, temperature, seed in protocol.runs:
                key = run_key(dataset, run_name)
                run_dir = output_root / f"real_{dataset}" / run_name
                valid, reason, payload = verify_completed_run(run_dir)
                if valid and payload is not None:
                    print(f"[跳过] {key}：已通过完整性与合法性审计", flush=True)
                    continue
                if run_dir.exists():
                    print(f"[保留不完整尝试] {key}：{reason}", flush=True)
                    archive_incomplete_run(output_root, run_dir)
                run_dir.mkdir(parents=True, exist_ok=False)
                progress.update(
                    {
                        "current": {"key": key, "temperature": temperature, "seed": seed, "started_at": utc_now()},
                        "failed": None,
                    }
                )
                if not partial_run:
                    write_json(progress_path, progress)
                ordinal = len(progress["completed"]) + 1
                print(
                    f"[开始 {ordinal}/{total_runs}] {key} "
                    f"temperature={temperature} seed={seed}",
                    flush=True,
                )
                command = [
                    args.python_executable,
                    "-u",
                    "evaluate_model.py",
                    "experiment=initial_main_real4_async_selection",
                    f"model_path={checkpoint}",
                    f"test_data=data/{dataset}.csv",
                    "num_runs=1",
                    f"temperature={temperature}",
                    "scenario=standard",
                    f"seed={seed}",
                    "no_gantt=true",
                    "verbose_eval_progress=true",
                    f"output_dir={run_dir}",
                ]
                command.extend(args.extra_eval_args)
                run_command(
                    command,
                    run_dir / "evaluation.log",
                    environment,
                )
                run_command(
                    [
                        args.python_executable,
                        "-u",
                        "scripts/validate_initial_schedule.py",
                        "--data",
                        f"data/{dataset}.csv",
                        "--schedule",
                        str(run_dir / "schedule.csv"),
                        "--output",
                        str(run_dir / "legality_audit.json"),
                    ],
                    run_dir / "legality_audit.log",
                    environment,
                )
                valid, reason, payload = verify_completed_run(run_dir)
                if not valid or payload is None:
                    raise RuntimeError(f"评估后审计未通过：{key}：{reason}")
                progress["completed"].append({"key": key, **payload, "completed_at": utc_now()})
                progress["completed_run_count"] = len(progress["completed"])
                progress["current"] = None
                if not partial_run:
                    write_json(progress_path, progress)
                print(
                    f"[完成 {progress['completed_run_count']}/{total_runs}] {key} "
                    f"makespan={payload['makespan']:.6f} legal=yes hard=0",
                    flush=True,
                )
    except Exception as exc:
        progress.update(
            {
                "status": "failed",
                "failed": {"type": type(exc).__name__, "message": str(exc), "failed_at": utc_now()},
            }
        )
        write_json(progress_path, progress)
        raise

    if partial_run:
        print(
            f"[子集完成] {len(selected_datasets)} 个实例、{len(progress['completed'])} 次评估完成；"
            "请在所有实例并行完成后运行完整命令 --resume 一次性生成汇总。",
            flush=True,
        )
        return 0
    _write_final_outputs(output_root, launch_contract, progress, protocol=protocol)
    print(
        f"[全部完成] {protocol.expected_run_count}/{protocol.expected_run_count} 次评估与合法性审计完成",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
