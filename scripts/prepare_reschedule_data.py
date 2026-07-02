from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import configs
from runtime.hydra_config import (
    ExtraArgument,
    HydraCliError,
    hydra_help,
    initialize_hydra_runtime,
    should_show_help,
)
from runtime.paths import resolve_workspace_path
from runtime.reschedule_manifest import to_manifest_path
from scripts.generate_reschedule_load_scenarios import write_scenario_library
from scripts.generate_schedule import generate_schedule
from utils.generate_random_dataset import generate_bucket
from utils.reschedule import load_baseline_schedule


PREPARE_ARGS = {
    "initial_model_path": ExtraArgument(required=True, help="用于生成 baseline schedule 的初始调度 checkpoint"),
    "train_count": ExtraArgument(default=30, help="生成 400-600 训练实例数量"),
    "min_ops": ExtraArgument(default=400, help="训练实例最小工序数"),
    "max_ops": ExtraArgument(default=600, help="训练实例最大工序数"),
    "seed": ExtraArgument(default=20260701, help="数据和固定场景随机种子"),
    "time_var": ExtraArgument(default=0.2, help="训练随机实例工时扰动系数"),
    "scenarios_per_level": ExtraArgument(default=20, help="真实验证数据每个 low/medium/high 等级的固定场景数量"),
    "train_output_dir": ExtraArgument(default="data/generated/reschedule_train_400_600", help="训练随机实例输出目录"),
    "baseline_output_dir": ExtraArgument(default="data/generated/reschedule_baselines_400_600", help="baseline schedule 输出目录"),
    "scenario_output_dir": ExtraArgument(default="data/reschedule_scenarios", help="真实验证固定扰动场景输出目录"),
    "output_manifest": ExtraArgument(default="data/reschedule_manifests/reschedule_400_600_seed20260701.json", help="输出 manifest 路径"),
    "real_data_paths": ExtraArgument(default=["data/283.csv", "data/680.csv", "data/2338.csv", "data/3182.csv"], help="真实验证数据集列表"),
    "overwrite": ExtraArgument(default=False, help="是否覆盖已存在的训练实例、baseline 和场景"),
}


def _as_path_list(value: Any) -> list[Path]:
    if isinstance(value, (list, tuple)):
        raw_items = value
    else:
        raw_items = [part.strip() for part in str(value).split(",") if part.strip()]
    return [resolve_workspace_path(item) for item in raw_items]


def _real_instance_id(path: Path) -> str:
    return f"real_{path.stem}"


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_safe_output_dir(path: Path, label: str) -> Path:
    resolved = path.resolve()
    project_root = PROJECT_ROOT.resolve()
    if resolved == project_root or not resolved.is_relative_to(project_root):
        raise ValueError(f"{label} 必须位于项目目录内部，且不能是项目根目录: {path}")
    return resolved


def _disable_reschedule_for_initial_generation() -> dict[str, Any]:
    keys = (
        "enable_reschedule_mode",
        "enable_dynamic_events",
        "enable_station_breakdown",
        "enable_material_delay",
        "enable_online_duration_perturb",
        "enable_worker_fatigue",
        "randomize_durations",
        "reschedule_manifest_path",
        "reschedule_scenario_path",
        "reschedule_eval_scenario_path",
    )
    backup = {key: getattr(configs, key) for key in keys}
    configs.enable_reschedule_mode = False
    configs.enable_dynamic_events = False
    configs.enable_station_breakdown = False
    configs.enable_material_delay = False
    configs.enable_online_duration_perturb = False
    configs.enable_worker_fatigue = False
    configs.randomize_durations = False
    configs.reschedule_manifest_path = ""
    configs.reschedule_scenario_path = ""
    configs.reschedule_eval_scenario_path = ""
    return backup


def _restore_config(backup: dict[str, Any]) -> None:
    for key, value in backup.items():
        setattr(configs, key, value)


def _generate_baseline(
    *,
    model_path: Path,
    data_path: Path,
    output_path: Path,
    overwrite: bool,
    explicit_fields: set[str],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        if output_path.exists() and not overwrite:
            baseline = load_baseline_schedule(output_path)
            return {
                "baseline_schedule_path": to_manifest_path(output_path),
                "baseline_makespan": float(baseline.makespan),
                "num_tasks": int(len(baseline.tasks)),
                "status": "ready",
            }, None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        backup = _disable_reschedule_for_initial_generation()
        try:
            frame = generate_schedule(
                str(model_path),
                explicit_fields=explicit_fields,
                data_path=data_path,
                output_path=output_path,
                write_context=False,
            )
        finally:
            _restore_config(backup)
        baseline = load_baseline_schedule(output_path)
        return {
            "baseline_schedule_path": to_manifest_path(output_path),
            "baseline_makespan": float(baseline.makespan),
            "num_tasks": int(len(frame)),
            "status": "ready",
        }, None
    except Exception as exc:  # noqa: BLE001 - 数据准备脚本需要逐实例记录失败原因
        return None, {
            "data_path": to_manifest_path(data_path),
            "baseline_schedule_path": to_manifest_path(output_path),
            "status": "skipped",
            "reason": str(exc),
        }


def prepare_reschedule_data(
    *,
    initial_model_path: Path,
    train_count: int,
    min_ops: int,
    max_ops: int,
    seed: int,
    time_var: float,
    scenarios_per_level: int,
    train_output_dir: Path,
    baseline_output_dir: Path,
    scenario_output_dir: Path,
    output_manifest: Path,
    real_data_paths: list[Path],
    overwrite: bool,
    explicit_fields: set[str],
) -> dict[str, Any]:
    if not initial_model_path.exists():
        raise FileNotFoundError(f"初始调度模型不存在: {initial_model_path}")
    if train_count < 1:
        raise ValueError("train_count 必须大于 0")
    if max_ops < min_ops:
        raise ValueError("max_ops 必须大于等于 min_ops")

    safe_train_output_dir = _ensure_safe_output_dir(train_output_dir, "train_output_dir")
    if overwrite and safe_train_output_dir.exists():
        shutil.rmtree(safe_train_output_dir)

    train_output_dir.mkdir(parents=True, exist_ok=True)
    baseline_output_dir.mkdir(parents=True, exist_ok=True)
    scenario_output_dir.mkdir(parents=True, exist_ok=True)

    if overwrite or not (train_output_dir / "manifest.json").exists():
        generate_bucket(
            PROJECT_ROOT / "data" / "680.csv",
            train_output_dir,
            min_length=int(min_ops),
            max_length=int(max_ops),
            num_samples=int(train_count),
            time_var=float(time_var),
            seed=int(seed),
            worker_pool_path=PROJECT_ROOT / "data" / "worker_pool_fixed.csv",
        )

    generated_manifest = json.loads((train_output_dir / "manifest.json").read_text(encoding="utf-8"))
    copied_template = train_output_dir / str(generated_manifest.get("baseline_file", ""))
    if copied_template.exists():
        copied_template.unlink()
    instances: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    ready_train_files: set[Path] = set()

    for idx, item in enumerate(generated_manifest.get("files", []), start=1):
        data_path = train_output_dir / str(item["file"])
        instance_id = f"train_{idx:04d}"
        baseline_path = baseline_output_dir / "train" / f"{instance_id}_schedule.csv"
        result, error = _generate_baseline(
            model_path=initial_model_path,
            data_path=data_path,
            output_path=baseline_path,
            overwrite=bool(overwrite),
            explicit_fields=explicit_fields,
        )
        if error is not None:
            error["instance_id"] = instance_id
            error["split"] = "train"
            skipped.append(error)
            if data_path.exists():
                data_path.unlink()
            print(f"[SKIP] {instance_id}: {error['reason']}", flush=True)
            continue
        assert result is not None
        instances.append(
            {
                "instance_id": instance_id,
                "split": "train",
                "source": "generated",
                "data_path": to_manifest_path(data_path),
                "scenario_path": "",
                **result,
            }
        )
        ready_train_files.add(data_path.resolve())
        print(f"[READY] {instance_id} data={data_path.name} baseline={baseline_path.name}", flush=True)

    # 环境按目录抽样时不能抽到没有 baseline 映射的旧 CSV。
    for candidate in train_output_dir.glob("*.csv"):
        if candidate.resolve() not in ready_train_files:
            candidate.unlink()

    for real_path in real_data_paths:
        instance_id = _real_instance_id(real_path)
        baseline_path = baseline_output_dir / "real" / f"{instance_id}_schedule.csv"
        result, error = _generate_baseline(
            model_path=initial_model_path,
            data_path=real_path,
            output_path=baseline_path,
            overwrite=bool(overwrite),
            explicit_fields=explicit_fields,
        )
        if error is not None:
            error["instance_id"] = instance_id
            error["split"] = "eval"
            skipped.append(error)
            print(f"[SKIP] {instance_id}: {error['reason']}", flush=True)
            continue
        assert result is not None
        scenario_path = scenario_output_dir / f"{instance_id}_load_grid_seed{int(seed)}.csv"
        metadata_path = scenario_path.with_suffix(".metadata.json")
        write_scenario_library(
            baseline_path=baseline_path,
            output_path=scenario_path,
            metadata_path=metadata_path,
            seed=int(seed),
            scenarios_per_level=int(scenarios_per_level),
        )
        instances.append(
            {
                "instance_id": instance_id,
                "split": "eval",
                "source": "real",
                "data_path": to_manifest_path(real_path),
                "scenario_path": to_manifest_path(scenario_path),
                **result,
            }
        )
        print(f"[READY] {instance_id} scenarios={scenario_path.name}", flush=True)

    payload = {
        "version": 1,
        "kind": "reschedule_dataset_manifest",
        "seed": int(seed),
        "train_count_requested": int(train_count),
        "min_ops": int(min_ops),
        "max_ops": int(max_ops),
        "time_var": float(time_var),
        "scenarios_per_level": int(scenarios_per_level),
        "initial_model_path": to_manifest_path(initial_model_path),
        "train_output_dir": to_manifest_path(train_output_dir),
        "baseline_output_dir": to_manifest_path(baseline_output_dir),
        "scenario_output_dir": to_manifest_path(scenario_output_dir),
        "instances": instances,
        "skipped": skipped,
    }
    _write_manifest(output_manifest, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if should_show_help(raw_args):
        print(hydra_help(PREPARE_ARGS))
        return 0
    try:
        args = initialize_hydra_runtime(
            raw_args,
            target=configs,
            project_root=PROJECT_ROOT,
            default_experiment="reschedule_task_delay",
            extra_arguments=PREPARE_ARGS,
            create_run_context=False,
        )
        payload = prepare_reschedule_data(
            initial_model_path=resolve_workspace_path(args.initial_model_path),
            train_count=int(args.train_count),
            min_ops=int(args.min_ops),
            max_ops=int(args.max_ops),
            seed=int(args.seed),
            time_var=float(args.time_var),
            scenarios_per_level=int(args.scenarios_per_level),
            train_output_dir=resolve_workspace_path(args.train_output_dir),
            baseline_output_dir=resolve_workspace_path(args.baseline_output_dir),
            scenario_output_dir=resolve_workspace_path(args.scenario_output_dir),
            output_manifest=resolve_workspace_path(args.output_manifest),
            real_data_paths=_as_path_list(args.real_data_paths),
            overwrite=bool(args.overwrite),
            explicit_fields=set(getattr(args, "explicit_config_fields", set())),
        )
    except (HydraCliError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"[CLI] {exc}", file=sys.stderr)
        return 2

    ready = len(payload.get("instances", []))
    skipped = len(payload.get("skipped", []))
    print(
        json.dumps(
            {
                "manifest": str(resolve_workspace_path(args.output_manifest)),
                "ready": ready,
                "skipped": skipped,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
