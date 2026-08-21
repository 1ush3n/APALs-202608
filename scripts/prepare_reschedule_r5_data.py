"""生成 r5_task_delay_v1 的 24/6/4 数据、baseline 和固定场景资产。"""

from __future__ import annotations

import csv
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
from runtime.five_skill_schema import REQUIRED_SKILL_IDS, validate_explicit_five_skill_csv
from runtime.paths import resolve_workspace_path
from runtime.reschedule_manifest import (
    load_reschedule_manifest,
    to_manifest_path,
    validate_r5_manifest_assets,
)
from scripts.prepare_reschedule_data import (
    _as_path_list,
    _ensure_safe_output_dir,
    _generate_baseline,
    _sha256,
)
from utils.generate_random_dataset import generate_bucket
from utils.reschedule_r5 import write_r5_scenario_library


PREPARE_ARGS = {
    "initial_model_path": ExtraArgument(required=True, help="初始调度 checkpoint"),
    "train_count": ExtraArgument(default=30, help="生成实例总数，r5 固定为 30"),
    "validation_count": ExtraArgument(default=6, help="validation 实例数量，r5 固定为 6"),
    "min_ops": ExtraArgument(default=400, help="实例最小工序数"),
    "max_ops": ExtraArgument(default=600, help="实例最大工序数"),
    "seed": ExtraArgument(default=20260701, help="数据和场景种子"),
    "time_var": ExtraArgument(default=0.2, help="训练实例工时扰动系数"),
    "generated_output_dir": ExtraArgument(default="data/r5_task_delay_v1/_generated_30", help="临时生成目录"),
    "train_output_dir": ExtraArgument(default="data/r5_task_delay_v1/instances/train", help="24 个训练图目录"),
    "validation_output_dir": ExtraArgument(default="data/r5_task_delay_v1/instances/validation", help="6 个验证图目录"),
    "baseline_output_dir": ExtraArgument(default="data/r5_task_delay_v1/baselines", help="baseline 输出根目录"),
    "scenario_output_dir": ExtraArgument(default="data/r5_task_delay_v1/scenarios", help="场景输出根目录"),
    "output_manifest": ExtraArgument(default="data/r5_task_delay_v1/manifest.json", help="r5 manifest"),
    "real_data_paths": ExtraArgument(default=["data/283.csv", "data/680.csv", "data/2338.csv", "data/3182.csv"], help="四个真实实例"),
    "overwrite": ExtraArgument(default=False, help="是否覆盖 r5 新目录"),
}


def _task_count(path: Path) -> int:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        return 0
    first = {str(value).strip().lower() for value in rows[0]}
    return max(0, len(rows) - 1) if "taskid" in first else len(rows)


def split_r5_instance_files(
    files: list[Path],
    *,
    validation_count: int,
) -> tuple[list[Path], list[Path]]:
    """按工序数排序后固定抽取 validation，保证 24/6 切分可复现。"""
    if len(files) != 30:
        raise ValueError(f"r5 切分要求恰好 30 个生成文件，实际为 {len(files)}")
    if validation_count != 6:
        raise ValueError("r5 validation_count 必须为 6")
    ordered = sorted((Path(path) for path in files), key=lambda path: (_task_count(path), path.name))
    validation_indices = {
        int((index + 0.5) * len(ordered) / validation_count)
        for index in range(validation_count)
    }
    validation = [path for index, path in enumerate(ordered) if index in validation_indices]
    train = [path for index, path in enumerate(ordered) if index not in validation_indices]
    if len(train) != 24 or len(validation) != 6:
        raise RuntimeError("r5 确定性切分未得到 24 train 和 6 validation")
    return train, validation


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _real_instance_id(path: Path) -> str:
    return f"real_{path.stem}"


def prepare_r5_reschedule_data(
    *,
    initial_model_path: Path,
    train_count: int,
    validation_count: int,
    min_ops: int,
    max_ops: int,
    seed: int,
    time_var: float,
    generated_output_dir: Path,
    train_output_dir: Path,
    validation_output_dir: Path,
    baseline_output_dir: Path,
    scenario_output_dir: Path,
    output_manifest: Path,
    real_data_paths: list[Path],
    overwrite: bool,
    explicit_fields: set[str],
    stage_ratios: dict[str, float] | None = None,
    severity_specs: dict[str, tuple[float, float, float]] | None = None,
) -> dict[str, Any]:
    if not initial_model_path.is_file():
        raise FileNotFoundError(f"初始调度 checkpoint 不存在: {initial_model_path}")
    if (train_count, validation_count) != (30, 6):
        raise ValueError("r5 数据规模固定为 train_count=30、validation_count=6")
    if len(real_data_paths) != 4:
        raise ValueError("r5 必须提供四个真实实例")
    stage_ratios = stage_ratios or {
        "early": float(getattr(configs, "r5_early_start_ratio", 0.225)),
        "middle": float(getattr(configs, "r5_middle_start_ratio", 0.400)),
        "late": float(getattr(configs, "r5_late_start_ratio", 0.575)),
    }
    severity_specs = severity_specs or {
        "low": (
            float(getattr(configs, "r5_low_task_ratio", 0.05)),
            float(getattr(configs, "r5_low_delay_min", 5.0)),
            float(getattr(configs, "r5_low_delay_max", 15.0)),
        ),
        "medium": (
            float(getattr(configs, "r5_medium_task_ratio", 0.10)),
            float(getattr(configs, "r5_medium_delay_min", 10.0)),
            float(getattr(configs, "r5_medium_delay_max", 35.0)),
        ),
        "high": (
            float(getattr(configs, "r5_high_task_ratio", 0.18)),
            float(getattr(configs, "r5_high_delay_min", 20.0)),
            float(getattr(configs, "r5_high_delay_max", 60.0)),
        ),
    }

    roots = [
        _ensure_safe_output_dir(path, label)
        for path, label in (
            (generated_output_dir, "generated_output_dir"),
            (train_output_dir, "train_output_dir"),
            (validation_output_dir, "validation_output_dir"),
            (baseline_output_dir, "baseline_output_dir"),
            (scenario_output_dir, "scenario_output_dir"),
            (output_manifest.parent, "output_manifest.parent"),
        )
    ]
    if output_manifest.exists() and not overwrite:
        raise FileExistsError(f"r5 manifest 已存在，默认拒绝覆盖: {output_manifest}")
    if overwrite:
        for path in roots[:-1]:
            if path.exists():
                shutil.rmtree(path)
        output_manifest.unlink(missing_ok=True)

    generated_output_dir.mkdir(parents=True, exist_ok=True)
    train_output_dir.mkdir(parents=True, exist_ok=True)
    validation_output_dir.mkdir(parents=True, exist_ok=True)
    baseline_output_dir.mkdir(parents=True, exist_ok=True)
    scenario_output_dir.mkdir(parents=True, exist_ok=True)

    generate_bucket(
        PROJECT_ROOT / "data" / "680.csv",
        generated_output_dir,
        min_length=int(min_ops),
        max_length=int(max_ops),
        num_samples=int(train_count),
        time_var=float(time_var),
        seed=int(seed),
        worker_pool_path=PROJECT_ROOT / "data" / "worker_pool_fixed.csv",
        require_explicit_skill_columns=True,
        required_skill_ids=tuple(sorted(REQUIRED_SKILL_IDS)),
        copy_template_to_output=False,
    )
    generated_files = sorted(generated_output_dir.glob("*.csv"))
    train_files, validation_files = split_r5_instance_files(
        generated_files,
        validation_count=validation_count,
    )

    instances: list[dict[str, Any]] = []

    def prepare_instance(
        *,
        instance_id: str,
        split: str,
        source: str,
        data_path: Path,
        baseline_dir_name: str,
        scenario_dir_name: str | None,
    ) -> None:
        validate_explicit_five_skill_csv(data_path, require_all_skills=True)
        baseline_path = baseline_output_dir / baseline_dir_name / f"{instance_id}_schedule.csv"
        result, error = _generate_baseline(
            model_path=initial_model_path,
            data_path=data_path,
            output_path=baseline_path,
            overwrite=bool(overwrite),
            explicit_fields=explicit_fields,
        )
        if error is not None or result is None:
            raise RuntimeError(f"{instance_id} baseline 生成失败: {error}")
        row: dict[str, Any] = {
            "instance_id": instance_id,
            "split": split,
            "source": source,
            "data_path": to_manifest_path(data_path),
            "baseline_schedule_path": to_manifest_path(baseline_path),
            "data_sha256": _sha256(data_path),
            "baseline_sha256": _sha256(baseline_path),
            **result,
        }
        if scenario_dir_name is not None:
            scenario_path = scenario_output_dir / scenario_dir_name / f"{instance_id}_scenarios.csv"
            metadata_path = scenario_path.with_suffix(".metadata.json")
            write_r5_scenario_library(
                baseline_path=baseline_path,
                output_path=scenario_path,
                metadata_path=metadata_path,
                instance_id=instance_id,
                seed=int(seed),
                stage_ratios=stage_ratios,
                severity_specs=severity_specs,
            )
            row.update(
                {
                    "scenario_path": to_manifest_path(scenario_path),
                    "scenario_sha256": _sha256(scenario_path),
                    "scenario_metadata_path": to_manifest_path(metadata_path),
                    "scenario_metadata_sha256": _sha256(metadata_path),
                }
            )
        else:
            row["scenario_path"] = ""
        instances.append(row)

    for index, source_path in enumerate(train_files, start=1):
        destination = train_output_dir / f"train_{index:04d}.csv"
        shutil.move(str(source_path), str(destination))
        prepare_instance(
            instance_id=f"train_{index:04d}",
            split="train",
            source="generated",
            data_path=destination,
            baseline_dir_name="train",
            scenario_dir_name=None,
        )
    for index, source_path in enumerate(validation_files, start=1):
        destination = validation_output_dir / f"validation_{index:04d}.csv"
        shutil.move(str(source_path), str(destination))
        prepare_instance(
            instance_id=f"validation_{index:04d}",
            split="validation",
            source="generated",
            data_path=destination,
            baseline_dir_name="validation",
            scenario_dir_name="validation",
        )
    for source_path in real_data_paths:
        source_path = resolve_workspace_path(source_path)
        prepare_instance(
            instance_id=_real_instance_id(source_path),
            split="eval",
            source="real",
            data_path=source_path,
            baseline_dir_name="real",
            scenario_dir_name="real",
        )

    shutil.rmtree(generated_output_dir, ignore_errors=False)
    payload = {
        "version": 1,
        "kind": "reschedule_dataset_manifest",
        "protocol": "explicit_fiveskill_v1",
        "reschedule_protocol": "r5_task_delay_v1",
        "protocol_version": 1,
        "seed": int(seed),
        "train_count": 24,
        "validation_count": 6,
        "eval_count": 4,
        "train_count_requested": 30,
        "formal_eval_scenario_count": 36,
        "validation_scenario_count": 54,
        "split_method": "task_count_stratified_quantile_v1",
        "initial_model_path": to_manifest_path(initial_model_path),
        "initial_model_sha256": _sha256(initial_model_path),
        "train_output_dir": to_manifest_path(train_output_dir),
        "validation_output_dir": to_manifest_path(validation_output_dir),
        "baseline_output_dir": to_manifest_path(baseline_output_dir),
        "scenario_output_dir": to_manifest_path(scenario_output_dir),
        "scenario_protocol": "r5_task_delay_v1",
        "scenario_count_per_instance": 9,
        "instances": instances,
        "skipped": [],
    }
    _write_json(output_manifest, payload)
    validate_r5_manifest_assets(load_reschedule_manifest(output_manifest))
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
            default_experiment="reschedule_task_delay_r5",
            extra_arguments=PREPARE_ARGS,
            create_run_context=False,
        )
        payload = prepare_r5_reschedule_data(
            initial_model_path=resolve_workspace_path(args.initial_model_path),
            train_count=int(args.train_count),
            validation_count=int(args.validation_count),
            min_ops=int(args.min_ops),
            max_ops=int(args.max_ops),
            seed=int(args.seed),
            time_var=float(args.time_var),
            generated_output_dir=resolve_workspace_path(args.generated_output_dir),
            train_output_dir=resolve_workspace_path(args.train_output_dir),
            validation_output_dir=resolve_workspace_path(args.validation_output_dir),
            baseline_output_dir=resolve_workspace_path(args.baseline_output_dir),
            scenario_output_dir=resolve_workspace_path(args.scenario_output_dir),
            output_manifest=resolve_workspace_path(args.output_manifest),
            real_data_paths=_as_path_list(args.real_data_paths),
            overwrite=bool(args.overwrite),
            explicit_fields=set(getattr(args, "explicit_config_fields", set())),
        )
    except (HydraCliError, KeyError, ValueError, RuntimeError, FileNotFoundError, FileExistsError) as exc:
        print(f"[CLI] {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "manifest": str(resolve_workspace_path(args.output_manifest)),
                "train": sum(row["split"] == "train" for row in payload["instances"]),
                "validation": sum(row["split"] == "validation" for row in payload["instances"]),
                "eval": sum(row["split"] == "eval" for row in payload["instances"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
