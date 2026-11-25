from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.literature.common import LiteraturePolicyAdapter
from baselines.literature_ppo.train_l2d_ppo_apal import SimpleHeteroGATPPO
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
from scripts.evaluate_graph_ddqn_r5_manifest import _audit_instance_schedules


OUTPUT_DIR = Path("results") / "revalidation_r5_l2d_ppo_20260823"
SCHEDULE_COLUMNS = (
    "instance_id",
    "scenario_id",
    "task_id",
    "station_id",
    "worker_ids",
    "start_time",
    "finish_time",
)
CHECKPOINT_DIMENSIONS = {
    "task": ("embedder.task_emb.0.weight", "task_feat_dim"),
    "station": ("embedder.station_emb.0.weight", "station_feat_dim"),
    "worker": ("embedder.worker_emb.0.weight", "worker_feat_dim"),
    "skill": ("embedder.skill_emb.0.weight", "skill_feat_dim"),
}
FORMAL_METADATA = (
    "algorithm",
    "literature_family",
    "model_type",
    "implementation_variant",
    "feature_mode",
    "training_protocol",
    "initialization",
    "team_selection_mode",
    "worker_pointer_v2_dynamic_eft_features",
    "task_feat_dim",
    "station_feat_dim",
    "worker_feat_dim",
    "skill_feat_dim",
    "hidden_dim",
    "num_gat_layers",
    "num_heads",
    "worker_feature_layout_version",
    "worker_skill_feature_slots",
    "reschedule_async_protocol",
    "experiment",
    "formal_r5_baseline",
    "manifest_path",
    "manifest_sha256",
    "formal_r5_checkpoint",
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _observation_dims(observation: Any) -> dict[str, int]:
    dimensions: dict[str, int] = {}
    for node_type in CHECKPOINT_DIMENSIONS:
        features = getattr(observation[node_type], "x", None)
        if features is None or features.ndim != 2:
            raise ValueError(f"r5 observation {node_type}.x 必须是二维张量")
        dimensions[node_type] = int(features.shape[1])
    return dimensions


def validate_l2d_ppo_r5_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    observation_dims: Mapping[str, int],
    require_formal: bool,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """验证原生 L2D checkpoint；旧名称只可作为辅助资产识别。"""
    if str(checkpoint.get("checkpoint_format", "")) != "literature_baseline_v2":
        raise ValueError("L2D checkpoint 必须是原生 literature_baseline_v2")
    if str(checkpoint.get("algorithm", "")) not in {"L2D-PPO-APAL", "Simple-HeteroGAT-PPO"}:
        raise ValueError("checkpoint algorithm 不是 L2D-PPO-APAL 或历史兼容别名")
    if str(checkpoint.get("literature_family", "")) != "learned_dispatching_rule_ppo":
        raise ValueError("checkpoint literature_family 不匹配")
    if str(checkpoint.get("model_type", "")) != "SimpleHeteroGATPPO":
        raise ValueError("checkpoint model_type 必须是 SimpleHeteroGATPPO")
    if str(checkpoint.get("feature_mode", "")) != "apal_hetero_graph":
        raise ValueError("checkpoint feature_mode 必须是 apal_hetero_graph")

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint 缺少 model_state_dict")
    for node_type, (weight_key, metadata_key) in CHECKPOINT_DIMENSIONS.items():
        weight = state_dict.get(weight_key)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise ValueError(f"checkpoint 缺少二维权重 {weight_key}")
        actual_dim = int(observation_dims[node_type])
        if int(weight.shape[1]) != actual_dim:
            raise ValueError(
                f"{node_type} feature dimension mismatch: "
                f"checkpoint={int(weight.shape[1])}, r5_observation={actual_dim}"
            )
        if checkpoint.get(metadata_key) is None or int(checkpoint[metadata_key]) != actual_dim:
            raise ValueError(f"{metadata_key} 必须等于 r5 observation 维度 {actual_dim}")

    formal_fields_present = all(key in checkpoint for key in FORMAL_METADATA)
    formal = bool(formal_fields_present)
    if formal:
        if str(checkpoint["algorithm"]) != "L2D-PPO-APAL":
            formal = False
        if str(checkpoint["training_protocol"]) != R5_RESCHEDULE_PROTOCOL:
            formal = False
        if str(checkpoint["reschedule_async_protocol"]) != R5_RESCHEDULE_PROTOCOL:
            formal = False
        if str(checkpoint["initialization"]) != "random":
            formal = False
        if str(checkpoint["team_selection_mode"]) != "autoregressive":
            formal = False
        if checkpoint["worker_pointer_v2_dynamic_eft_features"] is not False:
            formal = False
        if str(checkpoint["implementation_variant"]) != "apal_heterogat_joint_action_v1":
            formal = False
        if str(checkpoint["worker_feature_layout_version"]) != "five_skill_v2":
            formal = False
        if int(checkpoint["worker_skill_feature_slots"]) != 5:
            formal = False
        if str(checkpoint.get("experiment", "")) != "l2d_ppo_apal_r5":
            formal = False
        selection_instances = checkpoint["selection_instance_ids"]
        if not isinstance(selection_instances, (list, tuple)) or not selection_instances:
            formal = False
        elif set(map(str, selection_instances)).intersection(REAL_INSTANCE_IDS):
            raise ValueError("L2D checkpoint best 选择包含真实测试实例，存在测试泄漏")
        if str(checkpoint["selection_protocol"]) != "r5_validation_only":
            formal = False
        selection_scenarios = checkpoint["selection_scenario_ids"]
        if tuple(map(str, selection_scenarios)) != ("low_early", "medium_early", "high_early"):
            formal = False
        if checkpoint.get("formal_r5_checkpoint") is not True or checkpoint.get("formal_r5_baseline") is not True:
            formal = False
        if "optimizer_state_dict" not in checkpoint:
            formal = False

    if formal and manifest_path is not None:
        saved_path = Path(str(checkpoint["manifest_path"]))
        if not saved_path.is_absolute():
            saved_path = PROJECT_ROOT / saved_path
        if saved_path.resolve() != manifest_path.resolve():
            formal = False
        elif str(checkpoint["manifest_sha256"]).lower() != _sha256(manifest_path):
            raise ValueError("checkpoint manifest_sha256 与当前 manifest 不一致")

    if require_formal and not formal:
        missing = sorted(set(FORMAL_METADATA).difference(checkpoint))
        suffix = f"，缺少字段={missing}" if missing else ""
        raise ValueError(f"checkpoint 不是正式 L2D r5 checkpoint{suffix}")
    return {
        "formal_r5_checkpoint": formal,
        "comparison_role": "formal_r5_baseline" if formal else "auxiliary_initial_checkpoint",
    }


def _load_native_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint 顶层必须是对象")
    if "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint 缺少 model_state_dict")
    return checkpoint


def _apply_checkpoint_model_config(checkpoint: Mapping[str, Any]) -> None:
    for name in (
        "task_feat_dim",
        "station_feat_dim",
        "worker_feat_dim",
        "skill_feat_dim",
        "hidden_dim",
        "num_gat_layers",
        "num_heads",
        "num_skill_types",
        "worker_skill_feature_slots",
    ):
        if checkpoint.get(name) is not None:
            setattr(configs, name, int(checkpoint[name]))
    for name in ("use_skill_hub", "skill_hub_bidirectional"):
        if checkpoint.get(name) is not None:
            setattr(configs, name, bool(checkpoint[name]))
    configs.team_selection_mode = "autoregressive"
    configs.worker_pointer_v2_dynamic_eft_features = False


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
                "worker_ids": json.dumps([int(worker_id) for worker_id in worker_ids], ensure_ascii=False),
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
    formal_r5_checkpoint: bool,
) -> dict[str, bool]:
    all_complete = bool(scenario_rows) and all(float(row["complete"]) == 1.0 for row in scenario_rows)
    all_eligible = bool(scenario_rows) and all(float(row["eligible"]) == 1.0 for row in scenario_rows)
    return {
        "execution_complete": bool(execution_complete),
        "all_scenarios_complete": all_complete,
        "all_scenarios_eligible": all_eligible,
        "formal_r5_checkpoint": bool(formal_r5_checkpoint),
        "strict_main_table_eligible": bool(
            execution_complete and audit_ok and formal_r5_checkpoint and all_complete and all_eligible
        ),
    }


def _validate_scenario_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    required = {"scenario_id", "complete", "eligible", "makespan", "selection_score"}
    for row in rows:
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(f"场景结果缺少字段: {missing}")
        for key in ("complete", "eligible", "makespan", "selection_score"):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"场景结果字段 {key} 不是有限数值")


def _probe_entry_observation(entry: Any) -> tuple[Any, dict[str, int]]:
    if entry.scenario_path is None:
        raise ValueError(f"{entry.instance_id} 缺少 scenario_path")
    scenarios = load_reschedule_scenarios(entry.scenario_path)
    expected = expected_r5_scenario_ids()
    actual = tuple(str(scenario_id) for scenario_id, _ in scenarios)
    if actual != expected:
        raise ValueError(f"{entry.instance_id} 场景必须严格为 {expected}，实际为 {actual}")
    env = AirLineEnv_Graph(str(entry.data_path), seed=int(configs.seed))
    setattr(env, "_forced_reschedule_scenario", scenarios[0][1])
    try:
        observation = env.reset(randomize_duration=False, randomize_workers=False, seed=int(configs.seed))
        return observation, _observation_dims(observation)
    finally:
        if hasattr(env, "_forced_reschedule_scenario"):
            delattr(env, "_forced_reschedule_scenario")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError, UnicodeError):
        return None


def evaluate_l2d_ppo_r5_manifest(
    *,
    model_path: Path,
    manifest_path: Path,
    output_dir: Path = OUTPUT_DIR,
    temperature: float = 0.0,
) -> dict[str, Any]:
    if abs(float(temperature)) > 1e-12:
        raise ValueError("L2D-PPO-APAL r5 正式评估必须使用 temperature=0")
    if not torch.cuda.is_available():
        raise RuntimeError("L2D-PPO-APAL r5 正式评估只允许 CUDA，禁止 CPU fallback")

    checkpoint = _load_native_checkpoint(model_path)
    manifest = load_reschedule_manifest(manifest_path)
    if str(manifest.payload.get("reschedule_protocol", "")) != R5_RESCHEDULE_PROTOCOL:
        raise ValueError("manifest 必须是 r5_task_delay_v1")
    validate_r5_manifest_assets(manifest)
    eval_entries = manifest.filter(split="eval")
    if len(eval_entries) != 4 or tuple(entry.instance_id for entry in eval_entries) != REAL_INSTANCE_IDS:
        raise ValueError(f"L2D-PPO-APAL r5 必须精确使用四个真实实例: {REAL_INSTANCE_IDS}")

    device = torch.device("cuda")
    _apply_checkpoint_model_config(checkpoint)
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
            "n_w",
            "n_w_min",
            "n_w_max",
        )
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_ids = expected_r5_scenario_ids()
    scenario_rows_all: list[dict[str, Any]] = []
    schedule_rows_all: list[dict[str, Any]] = []
    audit_rows_all: list[dict[str, Any]] = []
    instance_summaries: dict[str, Any] = {}
    model: SimpleHeteroGATPPO | None = None
    adapter: LiteraturePolicyAdapter | None = None
    formal_info: dict[str, Any] = {"formal_r5_checkpoint": False}
    model_observation_dims: dict[str, int] | None = None
    resolved_config_snapshot: dict[str, Any] | None = None
    try:
        configs.enable_reschedule_mode = True
        configs.reschedule_async_protocol = R5_RESCHEDULE_PROTOCOL
        configs.reschedule_manifest_path = str(manifest.path)
        configs.verbose_reschedule_eval_progress = False
        for entry in eval_entries:
            if entry.scenario_path is None:
                raise ValueError(f"{entry.instance_id} 缺少 scenario_path")
            configs.data_file_path = str(entry.data_path)
            configs.reschedule_baseline_schedule_path = str(entry.baseline_schedule_path)
            configs.reschedule_eval_scenario_path = str(entry.scenario_path)
            configs.reschedule_eval_instance_id = entry.instance_id
            apply_initial_worker_mapping(configs, entry.data_path, explicit_fields=set())
            observation, observation_dims = _probe_entry_observation(entry)
            if model is None:
                formal_info = validate_l2d_ppo_r5_checkpoint(
                    checkpoint,
                    observation_dims=observation_dims,
                    require_formal=True,
                    manifest_path=manifest.path,
                )
                model_observation_dims = dict(observation_dims)
                model = SimpleHeteroGATPPO(configs).to(device).float()
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                model.eval()
                if any(parameter.device.type != "cuda" for parameter in model.parameters()):
                    raise RuntimeError("L2D 模型参数未全部位于 CUDA")
                if any(buffer.device.type != "cuda" for buffer in model.buffers()):
                    raise RuntimeError("L2D 模型 buffer 未全部位于 CUDA")
                adapter = LiteraturePolicyAdapter(model, device)
            elif observation_dims != model_observation_dims:
                raise ValueError(f"{entry.instance_id} r5 观测维度与首实例不一致")

            env = AirLineEnv_Graph(str(entry.data_path), seed=int(configs.seed))
            set_seed(int(configs.seed))
            assert adapter is not None
            with torch.inference_mode(), torch.amp.autocast(device_type="cuda", enabled=True):
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
            audit_rows_all.extend(
                _audit_instance_schedules(entry, expected_ids, schedules, metrics)
            )
            _validate_scenario_rows(metrics)
            for scenario_id, metric, schedule in zip(expected_ids, metrics, schedules):
                if str(metric["scenario_id"]) != scenario_id:
                    raise ValueError(f"{entry.instance_id} 场景顺序或 ID 不一致")
                scenario_rows_all.append({"instance_id": entry.instance_id, "scenario_id": scenario_id, **dict(metric)})
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
                "avg_selection_score": float(getattr(evaluate_reschedule_model, "last_metrics", {}).get("selection_score", 0.0)),
                "eligible_rate": float(getattr(evaluate_reschedule_model, "last_metrics", {}).get("eligible_rate", 0.0)),
            }
        resolved_config_snapshot = configs.to_flat_dict()
    finally:
        for key, value in backup.items():
            setattr(configs, key, value)

    if len(scenario_rows_all) != 36:
        raise ValueError(f"L2D-PPO-APAL r5 必须归档 36 个场景，实际为 {len(scenario_rows_all)}")
    _validate_scenario_rows(scenario_rows_all)
    # 归档优先：完整排程和独立审计结果均已按实例保存。
    audit_ok = len(audit_rows_all) == 36
    flags = summarize_r5_outcomes(
        scenario_rows_all,
        execution_complete=True,
        audit_ok=audit_ok,
        formal_r5_checkpoint=bool(formal_info.get("formal_r5_checkpoint", False)),
    )
    scenario_frame = pd.DataFrame(scenario_rows_all)
    scenario_frame.to_csv(output_dir / "l2d_ppo_r5_scenarios.csv", index=False)
    pd.DataFrame(schedule_rows_all, columns=list(SCHEDULE_COLUMNS)).to_csv(
        output_dir / "l2d_ppo_r5_schedules.csv", index=False
    )
    summary = {
        **flags,
        "algorithm": "L2D-PPO-APAL",
        "model_type": "SimpleHeteroGATPPO",
        "implementation_variant": "apal_heterogat_joint_action_v1",
        "feature_mode": "apal_hetero_graph",
        "initialization": "random",
        "temperature": 0.0,
        "device": "cuda",
        "protocol": R5_RESCHEDULE_PROTOCOL,
        "manifest_path": str(manifest.path.resolve()),
        "manifest_sha256": _sha256(manifest.path),
        "checkpoint_path": str(model_path.resolve()),
        "checkpoint_sha256": _sha256(model_path),
        "code_commit": _git_commit(),
        "scenario_count": len(scenario_rows_all),
        "instance_ids": list(REAL_INSTANCE_IDS),
        "scenario_ids_per_instance": list(expected_ids),
        "instance_summaries": instance_summaries,
        "incomplete_or_ineligible_scenarios": [
            {"instance_id": row["instance_id"], "scenario_id": row["scenario_id"]}
            for row in scenario_rows_all
            if float(row["complete"]) != 1.0 or float(row["eligible"]) != 1.0
        ],
        "audit": audit_rows_all,
    }
    (output_dir / "l2d_ppo_r5_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    import yaml

    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config_snapshot or configs.to_flat_dict(), allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    run_manifest = {
        "artifact_kind": "l2d_ppo_apal_r5_revalidation",
        "protocol": R5_RESCHEDULE_PROTOCOL,
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "manifest_sha256": summary["manifest_sha256"],
        "code_commit": summary["code_commit"],
        "files": {
            name: _sha256(output_dir / name)
            for name in (
                "l2d_ppo_r5_scenarios.csv",
                "l2d_ppo_r5_schedules.csv",
                "l2d_ppo_r5_summary.json",
                "resolved_config.yaml",
            )
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


EXTRA_ARGS = {
    "model_path": ExtraArgument(required=True, help="正式 L2D-PPO-APAL r5 原生 checkpoint"),
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
            default_experiment="l2d_ppo_apal_r5",
            extra_arguments=EXTRA_ARGS,
        )
        payload = evaluate_l2d_ppo_r5_manifest(
            model_path=resolve_workspace_path(args.model_path),
            manifest_path=resolve_workspace_path(args.manifest_path),
            output_dir=resolve_workspace_path(args.output_dir),
            temperature=float(args.temperature),
        )
    except (HydraCliError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"[CLI] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
