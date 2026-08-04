"""构建 r4 五技能重调度资产，并在发布前执行严格完整性审计。

该脚本只创建 ``data/r4``，绝不覆盖或删除 ``data/r3``。它采用临时目录构建，
只有训练图、训练基准、真实基准/场景副本和所有审计均通过后才发布为 r4。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import configs
from environment import AirLineEnv_Graph
from runtime.hydra_config import (
    ExtraArgument,
    HydraCliError,
    hydra_help,
    initialize_hydra_runtime,
    should_show_help,
)
from runtime.initial_worker_mapping import apply_initial_worker_mapping
from runtime.reschedule_manifest import load_reschedule_manifest, to_manifest_path
from scripts.audit_training_data import audit_training_data
from scripts.prepare_reschedule_data import _generate_baseline, _validate_baseline_hard_constraints
from utils.generate_random_dataset import generate_bucket
from utils.reschedule import load_baseline_schedule


BUILD_ARGS = {
    "checkpoint": ExtraArgument(default="checkpoints/init/g15.ckpt", help="五技能初始排程 warm-start checkpoint"),
    "source_real_manifest": ExtraArgument(default="data/r3/m.json", help="仅复用经核验真实基准与场景的来源 manifest"),
    "output_root": ExtraArgument(default="data/r4", help="r4 正式资产根目录"),
    "train_count": ExtraArgument(default=30, help="训练图数量"),
    "min_ops": ExtraArgument(default=400, help="训练图最小物理工序数"),
    "max_ops": ExtraArgument(default=600, help="训练图最大物理工序数"),
    "seed": ExtraArgument(default=20260701, help="训练图随机种子；真实场景保持来源副本"),
    "time_var": ExtraArgument(default=0.2, help="训练图工时扰动系数"),
}

EXPECTED_CHECKPOINT_BYTES = 127_124_342
EXPECTED_CHECKPOINT_SHA256 = "9e8f9136ac99eaaff7efe1e8bbb14612e6327b03ff69c36f9fee1b6d1b6a3225"
REQUIRED_SKILLS = (0, 1, 2, 3, 4)
REAL_INSTANCE_IDS = ("real_283", "real_680", "real_2338", "real_3182")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _copy_verified(source: Path, target: Path) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(f"待复用资产不存在: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    source_hash = _sha256(source)
    target_hash = _sha256(target)
    if source_hash != target_hash:
        raise RuntimeError(f"复制后的 SHA-256 不一致: {source} -> {target}")
    return {"source": _relative(source), "target": _relative(target), "sha256": target_hash}


def _validate_scenario_library(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    required_columns = {"scenario_id", "level", "TaskID", "release_time"}
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"固定场景缺少字段 {missing}: {path}")
    ids = frame[["scenario_id", "level"]].drop_duplicates()
    if len(ids) != 60:
        raise ValueError(f"固定场景数必须为 60，实际为 {len(ids)}: {path}")
    per_level = ids.groupby("level")["scenario_id"].nunique().to_dict()
    if per_level != {"high": 20, "low": 20, "medium": 20}:
        raise ValueError(f"固定场景分层必须 low/medium/high 各 20，实际为 {per_level}: {path}")
    if not ids["scenario_id"].is_unique:
        raise ValueError(f"scenario_id 不能跨层重复: {path}")
    return {"scenario_count": 60, "per_level": per_level, "rows": int(len(frame))}


def _reset_real_entry(
    *,
    data_path: Path,
    baseline_path: Path,
    scenario_path: Path,
    explicit_fields: set[str],
) -> int:
    """按真实实例映射执行一次重调度 reset，验证基准、场景和工人规模可同时加载。"""
    backup = {
        key: getattr(configs, key)
        for key in (
            "enable_reschedule_mode",
            "reschedule_manifest_path",
            "reschedule_baseline_schedule_path",
            "reschedule_scenario_path",
            "reschedule_eval_scenario_path",
        )
    }
    try:
        configs.enable_reschedule_mode = True
        configs.reschedule_manifest_path = ""
        configs.reschedule_baseline_schedule_path = str(baseline_path)
        configs.reschedule_scenario_path = str(scenario_path)
        configs.reschedule_eval_scenario_path = str(scenario_path)
        apply_initial_worker_mapping(configs, data_path, explicit_fields=explicit_fields)
        env = AirLineEnv_Graph(data_path_or_dir=data_path, seed=42)
        env.reset(randomize_duration=False, randomize_workers=False, seed=42)
        if env.baseline_schedule is None or env.reschedule_scenario is None:
            raise RuntimeError(f"真实实例 reset 未加载基准或场景: {data_path.name}")
        return int(env.num_workers)
    finally:
        for key, value in backup.items():
            setattr(configs, key, value)


def _write_readme(root: Path) -> None:
    (root / "README.md").write_text(
        """# r4：五技能正式重调度资产

`r4` 是五技能 APAL 重调度的正式数据协议。它以 `data/680.csv` 的显式
`专业编码` 与 `工种` 字段生成 30 个 400–600 物理工序训练图；每张训练图都必须
覆盖工种 0–4，并经过 DAG、专业—工种映射、工人覆盖和训练基准硬约束审计。

目录约定：

- `t/`：30 个训练图和其生成 manifest；
- `b/t/`：训练图的初始合法基准排程；
- `b/r/`：四真实实例的已核验基准排程副本；
- `s/`：四真实实例固定 low/medium/high（各 20）场景及 metadata 副本；
- `m.json`：唯一可用于 r4 训练/验证的 manifest；
- `integrity_check.json`：哈希、语义、场景和 reset 审计结果。

`r3` 训练图缺少显式技能字段并在加载时塌缩为工种 0；其训练及派生结果仅作历史
调试证据，禁止与 r4 或五技能正式结果混合统计。r4 复用 r3 中已通过核验的四真实
实例基准与固定场景，以保持测试场景可比；训练图和训练基准全部重新生成。
""",
        encoding="utf-8",
    )


def build_r4(
    *,
    checkpoint: Path,
    source_real_manifest: Path,
    output_root: Path,
    train_count: int,
    min_ops: int,
    max_ops: int,
    seed: int,
    time_var: float,
    explicit_fields: set[str],
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    output_root = output_root.resolve()
    source_real_manifest = source_real_manifest.resolve()
    if output_root.exists():
        raise FileExistsError(f"拒绝覆盖既有 r4 资产目录: {output_root}")
    if output_root.parent != (PROJECT_ROOT / "data").resolve() or output_root.name != "r4":
        raise ValueError("output_root 必须精确为项目内 data/r4，防止误写其他目录")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint}")
    if checkpoint.stat().st_size != EXPECTED_CHECKPOINT_BYTES or _sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("g15 checkpoint 字节数或 SHA-256 不匹配；拒绝使用错误 warm-start")
    if not source_real_manifest.is_file():
        raise FileNotFoundError(f"来源真实 manifest 不存在: {source_real_manifest}")
    if train_count != 30 or min_ops != 400 or max_ops != 600:
        raise ValueError(
            "r4 是固定正式协议：必须精确使用 30 个训练图和 400–600 物理工序范围；"
            "单图 smoke 必须写入其他临时目录，禁止发布为 r4"
        )

    stage = Path(tempfile.mkdtemp(prefix="r4_staging_", dir=output_root.parent))
    records: dict[str, Any] = {
        "schema_version": "r4_asset_integrity_v1",
        "status": "building",
        "verified_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "checkpoint": {"path": _relative(checkpoint), "bytes": checkpoint.stat().st_size, "sha256": _sha256(checkpoint)},
        "source_real_manifest": _relative(source_real_manifest),
    }
    try:
        train_dir = stage / "t"
        baseline_train_dir = stage / "b" / "t"
        baseline_real_dir = stage / "b" / "r"
        scenario_dir = stage / "s"
        train_dir.mkdir(parents=True)

        generated = generate_bucket(
            PROJECT_ROOT / "data" / "680.csv",
            train_dir,
            min_length=min_ops,
            max_length=max_ops,
            num_samples=train_count,
            time_var=time_var,
            seed=seed,
            worker_pool_path=PROJECT_ROOT / "data" / "worker_pool_fixed.csv",
            require_explicit_skill_columns=True,
            required_skill_ids=REQUIRED_SKILLS,
            copy_template_to_output=False,
        )
        records["generated_training_bucket"] = generated
        training_audit = audit_training_data(
            train_dir,
            PROJECT_ROOT / "data" / "680.csv",
            PROJECT_ROOT / "data" / "worker_pool_fixed.csv",
            PROJECT_ROOT / "conf" / "data" / "skill_groups_5.yaml",
            file_pattern="variant_*.csv",
            min_ops=min_ops,
            max_ops=max_ops,
            required_skill_ids=REQUIRED_SKILLS,
        )
        records["training_data_audit"] = training_audit

        instances: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for index, item in enumerate(generated["files"], start=1):
            instance_id = f"train_{index:04d}"
            data_path = train_dir / str(item["file"])
            baseline_path = baseline_train_dir / f"{instance_id}_schedule.csv"
            result, error = _generate_baseline(
                model_path=checkpoint,
                data_path=data_path,
                output_path=baseline_path,
                overwrite=False,
                explicit_fields=explicit_fields,
            )
            if error is not None:
                error.update({"instance_id": instance_id, "split": "train"})
                skipped.append(error)
                raise RuntimeError(f"{instance_id} 训练基准生成失败: {error['reason']}")
            assert result is not None
            instances.append(
                {
                    "instance_id": instance_id,
                    "split": "train",
                    "source": "generated_fiveskill_r4",
                    "data_path": f"data/r4/t/{data_path.name}",
                    "scenario_path": "",
                    "baseline_schedule_path": f"data/r4/b/t/{baseline_path.name}",
                    "data_sha256": _sha256(data_path),
                    "baseline_sha256": _sha256(baseline_path),
                    "baseline_makespan": float(result["baseline_makespan"]),
                    "num_tasks": int(result["num_tasks"]),
                    "status": "ready",
                }
            )

        source_manifest = load_reschedule_manifest(source_real_manifest)
        copied_real: list[dict[str, Any]] = []
        real_reset_workers: dict[str, int] = {}
        for instance_id in REAL_INSTANCE_IDS:
            entry = source_manifest.get(instance_id)
            if entry.scenario_path is None:
                raise ValueError(f"来源 manifest 缺少真实场景: {instance_id}")
            target_baseline = baseline_real_dir / f"{instance_id}_schedule.csv"
            target_scenario = scenario_dir / entry.scenario_path.name
            copied_real.append(_copy_verified(entry.baseline_schedule_path, target_baseline))
            copied_real.append(_copy_verified(entry.scenario_path, target_scenario))
            source_metadata = entry.scenario_path.with_suffix(".metadata.json")
            if source_metadata.is_file():
                copied_real.append(_copy_verified(source_metadata, target_scenario.with_suffix(".metadata.json")))
            scenario_stats = _validate_scenario_library(target_scenario)
            apply_initial_worker_mapping(configs, entry.data_path, explicit_fields=explicit_fields)
            _validate_baseline_hard_constraints(data_path=entry.data_path, baseline_path=target_baseline)
            real_reset_workers[instance_id] = _reset_real_entry(
                data_path=entry.data_path,
                baseline_path=target_baseline,
                scenario_path=target_scenario,
                explicit_fields=explicit_fields,
            )
            baseline = load_baseline_schedule(target_baseline)
            instances.append(
                {
                    "instance_id": instance_id,
                    "split": "eval",
                    "source": "real_fixed_r4",
                    "data_path": to_manifest_path(entry.data_path),
                    "scenario_path": f"data/r4/s/{target_scenario.name}",
                    "baseline_schedule_path": f"data/r4/b/r/{target_baseline.name}",
                    "data_sha256": _sha256(entry.data_path),
                    "baseline_sha256": _sha256(target_baseline),
                    "scenario_sha256": _sha256(target_scenario),
                    "baseline_makespan": float(baseline.makespan),
                    "num_tasks": int(len(baseline.tasks)),
                    "status": "ready",
                    "scenario_validation": scenario_stats,
                }
            )

        if skipped or len(instances) != 34:
            raise RuntimeError("r4 manifest 条目数量异常；拒绝发布")
        manifest = {
            "version": 1,
            "kind": "reschedule_dataset_manifest",
            "protocol": "explicit_fiveskill_v1",
            "protocol_version": 1,
            "asset_id": "r4",
            "seed": int(seed),
            "train_count_requested": int(train_count),
            "min_ops": int(min_ops),
            "max_ops": int(max_ops),
            "time_var": float(time_var),
            "scenarios_per_level": 20,
            "initial_model_path": "checkpoints/init/g15.ckpt",
            "initial_model_sha256": _sha256(checkpoint),
            "initial_model_bytes": checkpoint.stat().st_size,
            "train_output_dir": "data/r4/t",
            "baseline_output_dir": "data/r4/b",
            "scenario_output_dir": "data/r4/s",
            "instances": instances,
            "skipped": skipped,
        }
        (stage / "m.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_readme(stage)
        stage_prefix = _relative(stage)
        records["real_asset_copies"] = [
            {
                **item,
                "target": item["target"].replace(stage_prefix, "data/r4", 1),
            }
            for item in copied_real
        ]
        records["real_reset_worker_counts"] = real_reset_workers
        records["manifest"] = {"ready": len(instances), "skipped": len(skipped), "path": "data/r4/m.json"}

        all_files = sorted(path for path in stage.rglob("*") if path.is_file())
        records["file_sha256"] = {
            path.relative_to(stage).as_posix(): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in all_files
        }
        records["status"] = "passed"
        (stage / "integrity_check.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        stage.rename(output_root)
        return records
    except Exception as exc:
        records["status"] = "failed"
        records["error"] = str(exc)
        (stage / "integrity_check.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(f"r4 构建失败；临时证据已保留在 {stage}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if should_show_help(raw_args):
        print(hydra_help(BUILD_ARGS))
        return 0
    try:
        args = initialize_hydra_runtime(
            raw_args,
            target=configs,
            project_root=PROJECT_ROOT,
            default_experiment="reschedule_task_delay",
            extra_arguments=BUILD_ARGS,
            create_run_context=False,
        )
        result = build_r4(
            checkpoint=(PROJECT_ROOT / str(args.checkpoint)) if not Path(str(args.checkpoint)).is_absolute() else Path(str(args.checkpoint)),
            source_real_manifest=(PROJECT_ROOT / str(args.source_real_manifest)) if not Path(str(args.source_real_manifest)).is_absolute() else Path(str(args.source_real_manifest)),
            output_root=(PROJECT_ROOT / str(args.output_root)) if not Path(str(args.output_root)).is_absolute() else Path(str(args.output_root)),
            train_count=int(args.train_count),
            min_ops=int(args.min_ops),
            max_ops=int(args.max_ops),
            seed=int(args.seed),
            time_var=float(args.time_var),
            explicit_fields=set(getattr(args, "explicit_config_fields", set())),
        )
    except (HydraCliError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"[r4] {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "manifest": "data/r4/m.json", "ready": result["manifest"]["ready"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
