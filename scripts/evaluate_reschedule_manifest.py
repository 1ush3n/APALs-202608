from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def _sanitize_thread_env() -> None:
    """修正非法线程数环境变量，避免 libgomp 在导入计算库时告警。"""

    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        value = os.environ.get(name)
        if value is None:
            continue
        if not str(value).strip().isdigit() or int(str(value).strip()) <= 0:
            os.environ[name] = "1"


_sanitize_thread_env()

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import configs
from evaluate_reschedule_model import evaluate_saved_reschedule_model
from runtime.checkpoints import apply_checkpoint_model_spec, load_checkpoint
from runtime.hydra_config import (
    ExtraArgument,
    HydraCliError,
    hydra_help,
    initialize_hydra_runtime,
    should_show_help,
)
from runtime.paths import resolve_workspace_path
from runtime.reschedule_manifest import (
    REAL_INSTANCE_IDS,
    load_reschedule_manifest,
    validate_r5_manifest_assets,
)
from runtime.initial_worker_mapping import apply_initial_worker_mapping


MANIFEST_EVAL_ARGS = {
    "model_path": ExtraArgument(default="checkpoints/reschedule_task_delay/bestmodel/best_model.pth", help="重调度模型 checkpoint"),
    "manifest_path": ExtraArgument(required=True, help="prepare_reschedule_data.py 生成的 manifest"),
    "instance_ids": ExtraArgument(default=["real_283", "real_680", "real_2338", "real_3182"], help="要评估的 manifest 实例 ID"),
    "num_runs": ExtraArgument(default=None, help="每个实例最多评估多少个固定场景；缺省评估全部"),
    "scenario_ids": ExtraArgument(default=None, help="显式指定每个实例评估的场景 ID 列表"),
    "temperature": ExtraArgument(default=0.0, help="评估动作温度，0 表示确定性"),
    "output_dir": ExtraArgument(default="results/reschedule_manifest_eval", help="输出目录"),
    "reschedule_eval_use_cached_observation": ExtraArgument(default=False, help="use cached observations as async evaluation"),
    "reschedule_eval_skip_value_estimation": ExtraArgument(default=False, help="skip value estimation as async evaluation"),
}


def _as_id_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _backup_config() -> dict[str, Any]:
    keys = (
        "data_file_path",
        "reschedule_baseline_schedule_path",
        "reschedule_eval_scenario_path",
        "reschedule_manifest_path",
        "reschedule_eval_instance_id",
        "enable_reschedule_mode",
        "verbose_reschedule_eval_progress",
        "n_w",
        "n_w_min",
        "n_w_max",
    )
    return {key: getattr(configs, key, False) for key in keys}


def _restore_config(backup: dict[str, Any]) -> None:
    for key, value in backup.items():
        setattr(configs, key, value)


def evaluate_manifest_instances(
    *,
    model_path: Path,
    manifest_path: Path,
    instance_ids: list[str],
    num_runs: int | None,
    scenario_ids: list[str] | None,
    temperature: float,
    output_dir: Path,
    explicit_fields: set[str] | None = None,
    use_cached_observation: bool = False,
    skip_value_estimation: bool = False,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(model_path)
    apply_checkpoint_model_spec(configs, checkpoint.model_spec, explicit_fields=explicit_fields)
    manifest = load_reschedule_manifest(manifest_path)
    is_r5 = str(manifest.payload.get("reschedule_protocol", "")).strip() == "r5_task_delay_v1"
    if is_r5:
        validate_r5_manifest_assets(manifest)
        if tuple(instance_ids) != REAL_INSTANCE_IDS:
            raise ValueError(f"r5 正式测试必须精确使用四个真实实例: {REAL_INSTANCE_IDS}")
        if num_runs is not None or scenario_ids is not None:
            raise ValueError("r5 正式测试必须读取每个真实实例的全部 9 个固定场景")
        if abs(float(temperature)) > 1e-12:
            raise ValueError("r5 正式测试必须使用 temperature=0")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    backup = _backup_config()
    try:
        configs.enable_reschedule_mode = True
        configs.reschedule_manifest_path = ""
        configs.reschedule_eval_instance_id = ""
        configs.verbose_reschedule_eval_progress = True
        for instance_id in instance_ids:
            entry = manifest.get(instance_id)
            if entry.scenario_path is None:
                raise ValueError(f"{instance_id} 没有固定重调度场景，不能用于 manifest 评估")
            configs.data_file_path = str(entry.data_path)
            configs.reschedule_baseline_schedule_path = str(entry.baseline_schedule_path)
            configs.reschedule_eval_scenario_path = str(entry.scenario_path)
            apply_initial_worker_mapping(configs, entry.data_path, explicit_fields=set())
            subdir = output_dir / instance_id
            print(
                f"[ManifestEval] start instance={instance_id} "
                f"data={entry.data_path} baseline={entry.baseline_schedule_path} "
                f"scenarios={entry.scenario_path} output={subdir}",
                flush=True,
            )
            summary = evaluate_saved_reschedule_model(
                model_path=model_path,
                num_runs=num_runs,
                scenario_ids=scenario_ids,
                temperature=float(temperature),
                output_dir=subdir,
                use_cached_observation=use_cached_observation,
                skip_value_estimation=skip_value_estimation,
            )
            summary_row = {
                "instance_id": instance_id,
                "data_path": str(entry.data_path),
                "baseline_schedule_path": str(entry.baseline_schedule_path),
                "scenario_path": str(entry.scenario_path),
                "data_sha256": entry.data_sha256,
                "baseline_sha256": entry.baseline_sha256,
                "scenario_sha256": entry.scenario_sha256,
                "scenario_metadata_path": str(entry.scenario_metadata_path or ""),
                "scenario_metadata_sha256": entry.scenario_metadata_sha256,
                "num_tasks": entry.num_tasks,
                "baseline_makespan": entry.baseline_makespan,
                "scenario_count": summary.get("scenario_count", 0),
                "avg_makespan": summary.get("avg_makespan", 0.0),
                "avg_score": summary.get("avg_score", 0.0),
                "avg_selection_score": summary.get("avg_selection_score", 0.0),
                "eligible_rate": summary.get("eligible_rate", 0.0),
                "avg_duration_sec": summary.get("avg_duration_sec", 0.0),
                "worker_util": summary.get("worker_util", 0.0),
                "station_util": summary.get("station_util", 0.0),
            }
            if is_r5 and int(summary_row["scenario_count"]) != 9:
                raise ValueError(
                    f"r5 实例 {instance_id} 必须恰好评估 9 个场景，"
                    f"实际为 {summary_row['scenario_count']}"
                )
            rows.append(summary_row)
            summaries[instance_id] = summary
            print(
                f"[ManifestEval] {instance_id} "
                f"score={float(summary_row['avg_score']):.4f} "
                f"elig={float(summary_row['eligible_rate']):.2f} "
                f"mk={float(summary_row['avg_makespan']):.2f}",
                flush=True,
            )
    finally:
        _restore_config(backup)

    if is_r5 and sum(int(row["scenario_count"]) for row in rows) != 36:
        raise ValueError("r5 正式测试必须恰好包含 36 个场景")

    csv_path = output_dir / "reschedule_eval_by_instance.csv"
    json_path = output_dir / "reschedule_eval_summary.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    payload = {
            "model_path": str(model_path.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "model_format": checkpoint.format_name,
            "instance_ids": instance_ids,
            "rows": rows,
            "summaries": summaries,
            "scenario_ids": scenario_ids,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if should_show_help(raw_args):
        print(hydra_help(MANIFEST_EVAL_ARGS))
        return 0
    try:
        args = initialize_hydra_runtime(
            raw_args,
            target=configs,
            project_root=PROJECT_ROOT,
            default_experiment="reschedule_task_delay",
            extra_arguments=MANIFEST_EVAL_ARGS,
        )
        summary = evaluate_manifest_instances(
            model_path=resolve_workspace_path(args.model_path),
            manifest_path=resolve_workspace_path(args.manifest_path),
            instance_ids=_as_id_list(args.instance_ids),
            num_runs=None if args.num_runs is None else int(args.num_runs),
            scenario_ids=_as_id_list(args.scenario_ids) if args.scenario_ids is not None else None,
            temperature=float(args.temperature),
            output_dir=resolve_workspace_path(args.output_dir),
            explicit_fields=set(getattr(args, "explicit_config_fields", set())),
            use_cached_observation=bool(args.reschedule_eval_use_cached_observation),
            skip_value_estimation=bool(args.reschedule_eval_skip_value_estimation),
        )
    except (HydraCliError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"[CLI] {exc}", file=sys.stderr)
        return 2

    print(json.dumps({key: value for key, value in summary.items() if key != "summaries"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
