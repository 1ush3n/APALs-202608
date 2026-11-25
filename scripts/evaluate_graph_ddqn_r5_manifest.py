from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.graph_baseline import select_graph_action
from baselines.literature.common import LiteraturePolicyAdapter
from baselines.literature.evaluate_literature_baseline import (
    _apply_model_config,
    _load_checkpoint,
)
from baselines.literature_dqn.train_graph_ddqn_apal import GraphDDQNAPAL
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
from runtime.paths import resolve_workspace_path
from runtime.reschedule_eval import evaluate_reschedule_model
from runtime.reschedule_manifest import (
    REAL_INSTANCE_IDS,
    R5_RESCHEDULE_PROTOCOL,
    load_reschedule_manifest,
    validate_r5_manifest_assets,
)
from runtime.seed import set_seed
from utils.reschedule import load_reschedule_scenarios
from utils.reschedule_r5 import R5_SEVERITIES, R5_STAGES


OUTPUT_DIR = Path("results") / "revalidation_r5_graph_ddqn_20260823"
SCHEDULE_COLUMNS = (
    "instance_id",
    "scenario_id",
    "task_id",
    "station_id",
    "worker_ids",
    "start_time",
    "finish_time",
)
SCENARIO_COLUMNS = ("instance_id", "scenario_id", "complete", "eligible", "makespan", "selection_score")
CHECKPOINT_DIMENSIONS = {
    "task": ("embedder.task_emb.0.weight", "task_feat_dim"),
    "station": ("embedder.station_emb.0.weight", "station_feat_dim"),
    "worker": ("embedder.worker_emb.0.weight", "worker_feat_dim"),
    "skill": ("embedder.skill_emb.0.weight", "skill_feat_dim"),
}
REQUIRED_CHECKPOINT_METADATA = (
    "algorithm",
    "literature_family",
    "model_type",
    "feature_mode",
    "hidden_dim",
    "num_gat_layers",
    "num_heads",
    "worker_feature_layout_version",
    "worker_skill_feature_slots",
    "reschedule_async_protocol",
    "experiment",
    "formal_r5_baseline",
    "selection_protocol",
    "selection_instance_ids",
    "selection_scenario_ids",
)


def expected_r5_scenario_ids() -> tuple[str, ...]:
    return tuple(
        f"{severity}_{stage}"
        for stage in R5_STAGES
        for severity in R5_SEVERITIES
    )


def validate_graph_ddqn_r5_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    observation_dims: Mapping[str, int],
) -> dict[str, Any]:
    """严格验证 Graph-DDQN 原生 checkpoint 与 r5 图观测契约。"""
    if str(checkpoint.get("algorithm", "")) != "Graph-DDQN-APAL":
        raise ValueError("checkpoint algorithm 必须是 Graph-DDQN-APAL")
    if str(checkpoint.get("literature_family", "")) != "graph_double_dqn":
        raise ValueError("checkpoint literature_family 必须是 graph_double_dqn")
    if str(checkpoint.get("model_type", "")) != "GraphDDQNAPAL":
        raise ValueError("checkpoint model_type 必须是 GraphDDQNAPAL")
    if str(checkpoint.get("feature_mode", "")) != "apal_hetero_graph":
        raise ValueError("checkpoint feature_mode 必须是 apal_hetero_graph")

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint 缺少 model_state_dict")

    actual_dims: dict[str, int] = {}
    for node_type, (weight_key, metadata_key) in CHECKPOINT_DIMENSIONS.items():
        weight = state_dict.get(weight_key)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise ValueError(f"checkpoint 缺少二维权重 {weight_key}")
        actual_dim = int(observation_dims[node_type])
        weight_dim = int(weight.shape[1])
        if weight_dim != actual_dim:
            raise ValueError(
                f"{node_type} feature dimension mismatch: checkpoint={weight_dim}, r5_observation={actual_dim}"
            )
        metadata_dim = checkpoint.get(metadata_key)
        if metadata_dim is None or int(metadata_dim) != actual_dim:
            raise ValueError(
                f"{metadata_key} 必须等于 r5 observation 维度 {actual_dim}"
            )
        actual_dims[node_type] = actual_dim

    missing = [
        key for key in REQUIRED_CHECKPOINT_METADATA
        if key not in checkpoint
    ]
    if missing:
        raise ValueError(f"checkpoint 缺少 Graph-DDQN r5 元数据: {missing}")
    if str(checkpoint["reschedule_async_protocol"]) != R5_RESCHEDULE_PROTOCOL:
        raise ValueError("checkpoint 必须来自 r5_task_delay_v1")
    if str(checkpoint["experiment"]) != "reschedule_task_delay_r5":
        raise ValueError("checkpoint experiment 必须是 reschedule_task_delay_r5")
    if checkpoint["formal_r5_baseline"] is not True:
        raise ValueError("checkpoint 不是 formal_r5_baseline")
    if str(checkpoint["selection_protocol"]) != "r5_validation_only":
        raise ValueError("checkpoint best 选择协议必须是 r5_validation_only")

    selection_instances = checkpoint["selection_instance_ids"]
    if not isinstance(selection_instances, (list, tuple)) or not selection_instances:
        raise ValueError("checkpoint 缺少 r5 validation selection_instance_ids")
    leaked = sorted(set(map(str, selection_instances)).intersection(REAL_INSTANCE_IDS))
    if leaked:
        raise ValueError(f"checkpoint best 选择包含真实测试实例，存在泄漏: {leaked}")
    selection_scenarios = checkpoint["selection_scenario_ids"]
    if not isinstance(selection_scenarios, (list, tuple)) or not selection_scenarios:
        raise ValueError("checkpoint 缺少 r5 validation selection_scenario_ids")

    return {
        "formal_r5_baseline": True,
        "observation_dims": actual_dims,
        "selection_instance_ids": [str(value) for value in selection_instances],
        "selection_scenario_ids": [str(value) for value in selection_scenarios],
    }


def serialize_scenario_schedule(
    *,
    instance_id: str,
    scenario_id: str,
    schedule: Sequence[Sequence[Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for assignment in schedule:
        if len(assignment) != 5:
            raise ValueError(f"排程 assignment 必须包含五个字段: {assignment!r}")
        task_id, station_id, worker_ids, start_time, finish_time = assignment
        rows.append(
            {
                "instance_id": str(instance_id),
                "scenario_id": str(scenario_id),
                "task_id": int(task_id),
                "station_id": int(station_id),
                "worker_ids": json.dumps(
                    [int(worker_id) for worker_id in worker_ids],
                    ensure_ascii=False,
                ),
                "start_time": float(start_time),
                "finish_time": float(finish_time),
            }
        )
    return rows


def summarize_r5_outcomes(
    scenario_rows: Sequence[Mapping[str, Any]],
    *,
    execution_complete: bool,
    audit_ok: bool,
    formal_r5_baseline: bool = False,
) -> dict[str, bool]:
    all_complete = bool(scenario_rows) and all(
        float(row["complete"]) == 1.0 for row in scenario_rows
    )
    all_eligible = bool(scenario_rows) and all(
        float(row["eligible"]) == 1.0 for row in scenario_rows
    )
    return {
        "execution_complete": bool(execution_complete),
        "all_scenarios_complete": all_complete,
        "all_scenarios_eligible": all_eligible,
        "formal_r5_baseline": bool(formal_r5_baseline),
        "main_table_eligible": bool(
            execution_complete
            and audit_ok
            and formal_r5_baseline
            and all_complete
            and all_eligible
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_scenario_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    required = {"scenario_id", "complete", "eligible", "makespan", "selection_score"}
    for row in rows:
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(f"场景结果缺少字段: {missing}")
        for key in ("complete", "eligible", "makespan", "selection_score"):
            value = float(row[key])
            if not math.isfinite(value):
                raise ValueError(f"场景结果字段 {key} 不是有限数值: {value!r}")


def _observation_dims(observation: Any) -> dict[str, int]:
    dimensions: dict[str, int] = {}
    for node_type in CHECKPOINT_DIMENSIONS:
        node = observation[node_type]
        features = getattr(node, "x", None)
        if features is None or features.ndim != 2:
            raise ValueError(f"r5 observation {node_type}.x 必须是二维张量")
        dimensions[node_type] = int(features.shape[1])
    return dimensions


def _probe_entry_observation(entry: Any) -> dict[str, int]:
    if entry.scenario_path is None:
        raise ValueError(f"{entry.instance_id} 缺少 scenario_path")
    scenarios = load_reschedule_scenarios(entry.scenario_path)
    expected = expected_r5_scenario_ids()
    actual = tuple(str(scenario_id) for scenario_id, _scenario in scenarios)
    if actual != expected:
        raise ValueError(
            f"{entry.instance_id} 场景必须严格为 {expected}，实际为 {actual}"
        )
    env = AirLineEnv_Graph(str(entry.data_path), seed=int(configs.seed))
    setattr(env, "_forced_reschedule_scenario", scenarios[0][1])
    try:
        observation = env.reset(
            randomize_duration=False,
            randomize_workers=False,
            seed=int(configs.seed),
        )
        return _observation_dims(observation)
    finally:
        if hasattr(env, "_forced_reschedule_scenario"):
            delattr(env, "_forced_reschedule_scenario")


def _assignment_from_export_row(row: Mapping[str, Any]) -> tuple[int, int, list[int], float, float]:
    worker_ids = json.loads(str(row["worker_ids"]))
    if not isinstance(worker_ids, list):
        raise ValueError("worker_ids 必须是 JSON 列表")
    return (
        int(row["task_id"]),
        int(row["station_id"]),
        [int(worker_id) for worker_id in worker_ids],
        float(row["start_time"]),
        float(row["finish_time"]),
    )


def _audit_instance_schedules(
    entry: Any,
    scenario_ids: Sequence[str],
    schedules: Sequence[Sequence[Sequence[Any]]],
    scenario_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(schedules) != len(scenario_ids) or len(scenario_rows) != len(scenario_ids):
        raise ValueError(f"{entry.instance_id} evaluator 场景与排程数量不一致")
    audit_env = AirLineEnv_Graph(str(entry.data_path), seed=int(configs.seed))
    audit_rows: list[dict[str, Any]] = []
    for index, scenario_id in enumerate(scenario_ids):
        if entry.scenario_path is None:
            raise ValueError(f"{entry.instance_id} 缺少场景文件")
        scenario_lookup = dict(load_reschedule_scenarios(entry.scenario_path))
        setattr(audit_env, "_forced_reschedule_scenario", scenario_lookup[scenario_id])
        audit_env.reset(
            randomize_duration=False,
            randomize_workers=False,
            seed=int(configs.seed) + index,
        )
        assignment_rows = [
            {
                "task_id": int(task_id),
                "station_id": int(station_id),
                "worker_ids": json.dumps([int(worker_id) for worker_id in worker_ids]),
                "start_time": float(start_time),
                "finish_time": float(finish_time),
            }
            for task_id, station_id, worker_ids, start_time, finish_time in schedules[index]
        ]
        assignments = [
            _assignment_from_export_row(row)
            for row in assignment_rows
        ]
        report = audit_env.validate_assignments(assignments)
        complete = float(scenario_rows[index]["complete"]) == 1.0
        eligible = float(scenario_rows[index]["eligible"]) == 1.0
        if complete and eligible != bool(report.is_legal):
            raise ValueError(
                f"{entry.instance_id}/{scenario_id} evaluator eligibility 与独立排程审计不一致"
            )
        audit_rows.append(
            {
                "instance_id": entry.instance_id,
                "scenario_id": scenario_id,
                "audit_is_legal": bool(report.is_legal),
                "audit_violations": report.violations,
            }
        )
    if hasattr(audit_env, "_forced_reschedule_scenario"):
        delattr(audit_env, "_forced_reschedule_scenario")
    return audit_rows


def evaluate_graph_ddqn_r5_manifest(
    *,
    model_path: Path,
    manifest_path: Path,
    output_dir: Path = OUTPUT_DIR,
    temperature: float = 0.0,
) -> dict[str, Any]:
    if abs(float(temperature)) > 1e-12:
        raise ValueError("Graph-DDQN r5 正式评估必须使用 temperature=0")
    if not torch.cuda.is_available():
        raise RuntimeError("Graph-DDQN r5 正式评估只允许 CUDA，禁止 CPU fallback")

    checkpoint = _load_checkpoint(model_path)
    manifest = load_reschedule_manifest(manifest_path)
    validate_r5_manifest_assets(manifest)
    eval_entries = manifest.filter(split="eval")
    if len(eval_entries) != 4:
        raise ValueError("Graph-DDQN r5 必须恰好包含四个 eval 实例")
    if tuple(entry.instance_id for entry in eval_entries) != REAL_INSTANCE_IDS:
        raise ValueError(f"Graph-DDQN r5 eval 实例必须是 {REAL_INSTANCE_IDS}")

    device = torch.device("cuda")
    _apply_model_config(checkpoint)
    backup = {
        key: getattr(configs, key, None)
        for key in (
            "enable_reschedule_mode",
            "reschedule_async_protocol",
            "reschedule_manifest_path",
            "data_file_path",
            "reschedule_baseline_schedule_path",
            "reschedule_eval_scenario_path",
            "reschedule_eval_instance_id",
            "verbose_reschedule_eval_progress",
        )
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_rows_all: list[dict[str, Any]] = []
    schedule_rows_all: list[dict[str, Any]] = []
    audit_rows_all: list[dict[str, Any]] = []
    instance_summaries: dict[str, Any] = {}
    model: torch.nn.Module | None = None
    adapter: LiteraturePolicyAdapter | None = None
    expected_ids = expected_r5_scenario_ids()
    try:
        configs.enable_reschedule_mode = True
        configs.reschedule_async_protocol = R5_RESCHEDULE_PROTOCOL
        configs.reschedule_manifest_path = str(manifest.path)
        configs.verbose_reschedule_eval_progress = False
        for entry in eval_entries:
            if entry.scenario_path is None:
                raise ValueError(f"{entry.instance_id} 缺少场景文件")
            configs.data_file_path = str(entry.data_path)
            configs.reschedule_baseline_schedule_path = str(entry.baseline_schedule_path)
            configs.reschedule_eval_scenario_path = str(entry.scenario_path)
            configs.reschedule_eval_instance_id = entry.instance_id
            apply_initial_worker_mapping(configs, entry.data_path, explicit_fields=set())
            observation_dims = _probe_entry_observation(entry)
            if model is None:
                validate_graph_ddqn_r5_checkpoint(
                    checkpoint,
                    observation_dims=observation_dims,
                )
                model = GraphDDQNAPAL(configs).to(device).float()
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                model.eval()
                if any(parameter.device.type != "cuda" for parameter in model.parameters()):
                    raise RuntimeError("Graph-DDQN 模型参数未全部位于 CUDA")
                adapter = LiteraturePolicyAdapter(model, device)
            else:
                expected_dims = _observation_dims(
                    _probe_observation_for_entry(entry)
                )
                if expected_dims != observation_dims:
                    raise ValueError(f"{entry.instance_id} r5 观测维度与首实例不一致")
            env = AirLineEnv_Graph(str(entry.data_path), seed=int(configs.seed))
            set_seed(int(configs.seed))
            with torch.inference_mode(), torch.amp.autocast(
                device_type="cuda",
                enabled=True,
            ):
                result = evaluate_reschedule_model(
                    env,
                    adapter,
                    num_runs=None,
                    temperature=0.0,
                    scenario_ids=expected_ids,
                    skip_value_estimation=True,
                )
            metrics = list(getattr(evaluate_reschedule_model, "last_scenario_metrics", []) or [])
            schedules = list(getattr(evaluate_reschedule_model, "last_scenario_schedules", []) or [])
            if len(metrics) != 9 or len(schedules) != 9:
                raise ValueError(f"{entry.instance_id} 必须输出九个场景及九个原始排程")
            _validate_scenario_rows(metrics)
            audit_rows = _audit_instance_schedules(entry, expected_ids, schedules, metrics)
            audit_rows_all.extend(audit_rows)
            for scenario_id, metric, schedule in zip(expected_ids, metrics, schedules):
                if str(metric["scenario_id"]) != scenario_id:
                    raise ValueError(f"{entry.instance_id} 场景顺序或 ID 不一致")
                scenario_rows_all.append(
                    {
                        "instance_id": entry.instance_id,
                        "scenario_id": scenario_id,
                        **dict(metric),
                    }
                )
                schedule_rows_all.extend(
                    serialize_scenario_schedule(
                        instance_id=entry.instance_id,
                        scenario_id=scenario_id,
                        schedule=schedule,
                    )
                )
            instance_summaries[entry.instance_id] = {
                "scenario_count": len(metrics),
                "avg_makespan": float(result[0]),
                "avg_selection_score": float(
                    getattr(evaluate_reschedule_model, "last_metrics", {}).get(
                        "selection_score", 0.0
                    )
                ),
                "eligible_rate": float(
                    getattr(evaluate_reschedule_model, "last_metrics", {}).get(
                        "eligible_rate", 0.0
                    )
                ),
            }
    finally:
        for key, value in backup.items():
            setattr(configs, key, value)

    if len(scenario_rows_all) != 36:
        raise ValueError(f"Graph-DDQN r5 必须归档 36 个场景，实际为 {len(scenario_rows_all)}")
    _validate_scenario_rows(scenario_rows_all)
    audit_ok = len(audit_rows_all) == 36
    flags = summarize_r5_outcomes(
        scenario_rows_all,
        execution_complete=True,
        audit_ok=audit_ok,
        formal_r5_baseline=True,
    )
    scenario_frame = pd.DataFrame(scenario_rows_all)
    scenario_frame.to_csv(output_dir / "graph_ddqn_r5_scenarios.csv", index=False)
    schedule_frame = pd.DataFrame(schedule_rows_all, columns=list(SCHEDULE_COLUMNS))
    schedule_frame.to_csv(output_dir / "graph_ddqn_r5_schedules.csv", index=False)
    summary = {
        **flags,
        "algorithm": "Graph-DDQN-APAL",
        "feature_mode": "apal_hetero_graph",
        "temperature": 0.0,
        "device": "cuda",
        "manifest_path": str(manifest.path.resolve()),
        "manifest_sha256": _sha256(manifest.path),
        "checkpoint_path": str(model_path.resolve()),
        "checkpoint_sha256": _sha256(model_path),
        "scenario_count": len(scenario_rows_all),
        "instance_ids": list(REAL_INSTANCE_IDS),
        "scenario_ids_per_instance": list(expected_ids),
        "instance_summaries": instance_summaries,
        "audit": audit_rows_all,
    }
    (output_dir / "graph_ddqn_r5_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    run_manifest = {
        "artifact_kind": "graph_ddqn_r5_revalidation",
        "protocol": R5_RESCHEDULE_PROTOCOL,
        "files": {
            name: _sha256(output_dir / name)
            for name in (
                "graph_ddqn_r5_scenarios.csv",
                "graph_ddqn_r5_schedules.csv",
                "graph_ddqn_r5_summary.json",
            )
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _probe_observation_for_entry(entry: Any) -> Any:
    if entry.scenario_path is None:
        raise ValueError(f"{entry.instance_id} 缺少场景文件")
    scenarios = load_reschedule_scenarios(entry.scenario_path)
    env = AirLineEnv_Graph(str(entry.data_path), seed=int(configs.seed))
    setattr(env, "_forced_reschedule_scenario", scenarios[0][1])
    try:
        return env.reset(
            randomize_duration=False,
            randomize_workers=False,
            seed=int(configs.seed),
        )
    finally:
        if hasattr(env, "_forced_reschedule_scenario"):
            delattr(env, "_forced_reschedule_scenario")


EXTRA_ARGS = {
    "model_path": ExtraArgument(required=True, help="正式 r5 Graph-DDQN 原生 checkpoint"),
    "manifest_path": ExtraArgument(required=True, help="r5_task_delay_v1 manifest"),
    "output_dir": ExtraArgument(default=str(OUTPUT_DIR), help="输出目录"),
    "temperature": ExtraArgument(default=0.0, help="正式评估必须为 0"),
}


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if should_show_help(raw_args):
        print(hydra_help(EXTRA_ARGS))
        return 0
    try:
        args = initialize_hydra_runtime(
            raw_args,
            target=configs,
            project_root=PROJECT_ROOT,
            default_experiment="reschedule_task_delay_r5",
            extra_arguments=EXTRA_ARGS,
        )
        payload = evaluate_graph_ddqn_r5_manifest(
            model_path=resolve_workspace_path(args.model_path),
            manifest_path=resolve_workspace_path(args.manifest_path),
            output_dir=resolve_workspace_path(args.output_dir),
            temperature=float(args.temperature),
        )
    except (HydraCliError, KeyError, ValueError, RuntimeError, FileNotFoundError, OSError) as exc:
        print(f"[CLI] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
