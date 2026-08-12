from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch

from configs import Config, configs, load_training_config
from runtime.artifacts import run_context as create_run_context
from runtime.artifacts import uses_runs_layout
from runtime.initial_checkpoint_selection import load_initial_checkpoint_selection_manifest
from runtime.paths import PROJECT_ROOT


EXPLICIT_OVERRIDES = {
    "data_path": "data_file_path",
    "train_data_path": "train_data_path_or_dir",
    "seed": "seed",
    "max_episodes": "max_episodes",
    "num_envs": "num_envs",
    "batch_size": "batch_size",
    "eval_freq": "eval_freq",
    "log_dir": "log_dir",
    "output_dir": "result_dir",
    "run_id": "run_id",
    "runs_root": "runs_root",
    "use_skill_hub": "use_skill_hub",
    "skill_hub_bidirectional": "skill_hub_bidirectional",
    "ablation_no_gat": "ablation_no_gat",
    "ablation_no_pointer": "ablation_no_pointer",
    "ablation_no_mask": "ablation_no_mask",
}

STRUCTURAL_FIELDS = {
    "hidden_dim", "num_gat_layers", "num_heads", "task_feat_dim",
    "worker_feat_dim", "station_feat_dim", "use_skill_hub",
    "skill_hub_bidirectional", "num_skill_types", "skill_feat_dim",
    "worker_skill_feature_slots",
    "use_input_layer_norm", "use_gat_layer_norm", "use_head_layer_norm",
    "use_shared_trunk", "policy_action_scope", "conditional_team_max_candidates",
    "conditional_team_gate_bias", "conditional_team_nonbaseline_logit",
    "conditional_team_scoring_mode", "conditional_team_prior_margin",
    "conditional_team_prior_weight", "workforce_binding_mode",
    "workforce_preallocation_ratio", "team_selection_mode",
    "worker_pointer_context_version", "worker_pointer_pressure_temperature",
    "worker_pointer_supply_epsilon", "worker_pointer_wait_discount_mode",
    "worker_pointer_v2_init_seed_offset",
    "graph_encoder_mode", "actor_context_mode",
    "anchor_proposal_mode", "anchor_proposal_prior_margin",
    "anchor_proposal_gate_bias",
    "anchor_proposal_train_branch_floor_start",
    "anchor_proposal_train_branch_floor_end",
    "anchor_proposal_branch_floor_decay_fraction",
    "anchor_proposal_require_difference",
}

_VALID_EXPERIMENT_MODES = {
    "policy_action_scope": {
        "operation", "operation_station", "operation_station_worker",
        "operation_station_gated_team", "operation_station_anchor_proposal_team",
    },
    "workforce_binding_mode": {"endogenous", "preallocated"},
    "team_selection_mode": {
        "autoregressive", "autoregressive_pressure_v2", "static_topq",
    },
    "graph_encoder_mode": {"hetero_gat", "homogeneous_graphsage", "none"},
    "actor_context_mode": {"attention", "mean_max", "local_only"},
    "conditional_team_scoring_mode": {
        "fixed_prior_v1", "relative_heuristic_prior_v1",
    },
}


def _validate_anchor_proposal_config(config: Config) -> None:
    """APCF scope 专用校验：禁止迁移到重调度，门控先验与探索下限必须合法。"""
    if bool(getattr(config, "enable_reschedule_mode", False)):
        raise ValueError(
            "operation_station_anchor_proposal_team 仅用于初始调度，"
            "不得与 enable_reschedule_mode=true 同时启用（未经验证不可迁移到重调度）"
        )
    mode = str(getattr(config, "anchor_proposal_mode", "full_team_v1"))
    if mode != "full_team_v1":
        raise ValueError(f"anchor_proposal_mode 当前仅支持 full_team_v1，收到 {mode!r}")
    margin = float(getattr(config, "anchor_proposal_prior_margin", 4.0))
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError("anchor_proposal_prior_margin 必须是大于 0 的有限数")
    gate_bias = float(getattr(config, "anchor_proposal_gate_bias", -4.0))
    if not math.isfinite(gate_bias) or gate_bias >= 0.0:
        raise ValueError("anchor_proposal_gate_bias 必须严格为负，保证初始确定性选择锚点")
    floor_start = float(getattr(config, "anchor_proposal_train_branch_floor_start", 0.20))
    floor_end = float(getattr(config, "anchor_proposal_train_branch_floor_end", 0.02))
    for name, value in (
        ("anchor_proposal_train_branch_floor_start", floor_start),
        ("anchor_proposal_train_branch_floor_end", floor_end),
    ):
        if not math.isfinite(value) or not 0.0 <= value < 0.5:
            raise ValueError(f"{name} 必须在 [0, 0.5) 内")
    if floor_start < floor_end:
        raise ValueError("anchor_proposal_train_branch_floor_start 不能小于 floor_end")
    decay_fraction = float(getattr(config, "anchor_proposal_branch_floor_decay_fraction", 0.40))
    if not math.isfinite(decay_fraction) or not 0.0 < decay_fraction <= 1.0:
        raise ValueError("anchor_proposal_branch_floor_decay_fraction 必须在 (0, 1] 内")
    config.anchor_proposal_prior_margin = margin
    config.anchor_proposal_gate_bias = gate_bias
    config.anchor_proposal_train_branch_floor_start = floor_start
    config.anchor_proposal_train_branch_floor_end = floor_end
    config.anchor_proposal_branch_floor_decay_fraction = decay_fraction


def parse_override_value(raw: str) -> Any:
    try:
        import yaml
        return yaml.safe_load(raw)
    except ImportError:
        from configs import _parse_yaml_scalar
        return _parse_yaml_scalar(raw)


def parse_set_overrides(items: list[str] | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items or []:
        key, separator, raw_value = item.partition("=")
        key = key.strip().replace("-", "_")
        if not separator or not key:
            raise ValueError(f"--set 必须使用 key=value 格式: {item!r}")
        overrides[key] = parse_override_value(raw_value)
    return overrides


def _normalize_override_key(raw_key: str) -> str:
    return raw_key.strip().replace("-", "_")


def _resolve_override_key(key: str, config: Config) -> str:
    normalized = _normalize_override_key(key)
    if hasattr(config, normalized):
        return normalized

    dotted_as_flat = normalized.replace(".", "_")
    if hasattr(config, dotted_as_flat):
        return dotted_as_flat

    tail = normalized.rsplit(".", 1)[-1]
    if hasattr(config, tail):
        return tail

    raise KeyError(f"未知配置字段: {key}")


def parse_hydra_overrides(items: list[str] | None, config: Config) -> dict[str, Any]:
    """解析 Hydra 风格的 key=value 覆盖，但仍落到当前扁平 Config 字段。"""
    overrides: dict[str, Any] = {}
    for item in items or []:
        if item.startswith("-"):
            raise ValueError(f"未知命令行参数: {item}")
        key, separator, raw_value = item.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"Hydra 风格覆盖必须使用 key=value 格式: {item!r}")
        resolved_key = _resolve_override_key(key, config)
        overrides[resolved_key] = parse_override_value(raw_value)
    return overrides


def _coerce_value(config: Config, key: str, value: Any) -> Any:
    if not hasattr(config, key):
        raise KeyError(f"未知配置字段: {key}")
    current = getattr(config, key)
    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise TypeError(f"配置 {key} 需要布尔值，收到 {value!r}")
        return value
    if isinstance(current, int):
        if isinstance(value, bool):
            raise TypeError(f"配置 {key} 需要整数，收到 {value!r}")
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, list):
        if not isinstance(value, list):
            raise TypeError(f"配置 {key} 需要列表，收到 {value!r}")
        return value
    if isinstance(current, tuple):
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"配置 {key} 需要列表，收到 {value!r}")
        return tuple(value)
    if isinstance(current, str):
        return str(value)
    return value


def collect_cli_overrides(args: argparse.Namespace) -> tuple[dict[str, Any], set[str]]:
    values = vars(args)
    overrides: dict[str, Any] = {}
    explicit_fields: set[str] = set()
    for argument, config_key in EXPLICIT_OVERRIDES.items():
        value = values.get(argument)
        if value is not None:
            overrides[config_key] = value
            explicit_fields.add(config_key)
    generic = parse_set_overrides(values.get("set_values"))
    overrides.update(generic)
    explicit_fields.update(generic)
    hydra_like = parse_hydra_overrides(values.get("hydra_overrides"), Config())
    overrides.update(hydra_like)
    explicit_fields.update(hydra_like)
    return overrides, explicit_fields


def parse_runtime_args(
    parser: argparse.ArgumentParser,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """兼容 argparse 旧参数和 Hydra 风格 key=value 覆盖。"""
    args, unknown = parser.parse_known_args(argv)
    setattr(args, "hydra_overrides", unknown)
    return args


def validate_runtime_config(config: Config) -> None:
    for field_name, choices in _VALID_EXPERIMENT_MODES.items():
        value = str(getattr(config, field_name)).lower()
        if value not in choices:
            raise ValueError(
                f"{field_name} 无效: {value!r}；允许值={sorted(choices)}"
            )
        setattr(config, field_name, value)
    if config.team_selection_mode == "autoregressive_pressure_v2":
        if config.policy_action_scope != "operation_station_worker":
            raise ValueError(
                "autoregressive_pressure_v2 仅允许 policy_action_scope=operation_station_worker"
            )
        if config.actor_context_mode != "attention":
            raise ValueError(
                "autoregressive_pressure_v2 要求 actor_context_mode=attention"
            )
        temperature = float(config.worker_pointer_pressure_temperature)
        epsilon = float(config.worker_pointer_supply_epsilon)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("worker_pointer_pressure_temperature 必须是大于 0 的有限数")
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("worker_pointer_supply_epsilon 必须是大于 0 的有限数")
        if str(config.worker_pointer_wait_discount_mode) != "physical_wait_exponential_v1":
            raise ValueError(
                "worker_pointer_wait_discount_mode 仅支持 physical_wait_exponential_v1"
            )
        if int(config.worker_pointer_v2_init_seed_offset) < 0:
            raise ValueError("worker_pointer_v2_init_seed_offset 不能为负数")
        if not bool(config.worker_pointer_v2_behavior_replay):
            raise ValueError(
                "autoregressive_pressure_v2 要求启用 worker_pointer_v2_behavior_replay"
            )
        if str(config.worker_pointer_v2_replay_mode) != "behavior_group_exact_v1":
            raise ValueError(
                "worker_pointer_v2_replay_mode 仅支持 behavior_group_exact_v1"
            )
        if int(config.worker_pointer_v2_logical_batch_cap) < 1:
            raise ValueError("worker_pointer_v2_logical_batch_cap 必须大于 0")
        if int(config.worker_pointer_v2_rollout_group_upper_bound) < 1:
            raise ValueError(
                "worker_pointer_v2_rollout_group_upper_bound 必须大于 0"
            )
    ratio = float(getattr(config, "workforce_preallocation_ratio", 1.0))
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError("workforce_preallocation_ratio 必须是 [0, 1] 内的有限数")
    if config.workforce_binding_mode == "preallocated" and ratio <= 0.0:
        raise ValueError("preallocated 模式要求 workforce_preallocation_ratio > 0")
    candidate_limit = int(getattr(config, "conditional_team_max_candidates", 4))
    if candidate_limit < 1:
        raise ValueError("conditional_team_max_candidates 必须大于等于 1")
    config.conditional_team_max_candidates = candidate_limit
    for field_name in (
        "conditional_team_gate_bias",
        "conditional_team_nonbaseline_logit",
        "conditional_team_prior_margin",
        "conditional_team_prior_weight",
    ):
        value = float(getattr(config, field_name))
        if not math.isfinite(value):
            raise ValueError(f"{field_name} 必须是有限数")
        setattr(config, field_name, value)
    if config.conditional_team_nonbaseline_logit >= 0.0:
        raise ValueError("conditional_team_nonbaseline_logit 必须严格为负数")
    if config.conditional_team_prior_margin <= 0.0:
        raise ValueError("conditional_team_prior_margin 必须严格大于 0")
    if config.conditional_team_prior_weight < 0.0:
        raise ValueError("conditional_team_prior_weight 必须大于等于 0")
    if config.policy_action_scope == "operation_station_anchor_proposal_team":
        _validate_anchor_proposal_config(config)
    if bool(getattr(config, "best_anchor_distill_enabled", False)):
        if config.policy_action_scope != "operation_station_gated_team":
            raise ValueError(
                "best_anchor_distill_enabled 仅允许用于 "
                "policy_action_scope=operation_station_gated_team"
            )
        if not bool(getattr(config, "async_eval_enabled", False)):
            raise ValueError("best-anchor 蒸馏要求启用异步四实例 checkpoint 选择")
        if str(getattr(config, "checkpoint_selection_protocol", "")).strip().lower() != "multiscale_manifest":
            raise ValueError("best-anchor 蒸馏要求 checkpoint_selection_protocol=multiscale_manifest")
        if not str(getattr(config, "checkpoint_selection_manifest_path", "")).strip():
            raise ValueError("best-anchor 蒸馏要求提供 checkpoint_selection_manifest_path")
        for field_name in (
            "best_anchor_distill_temperature",
            "best_anchor_distill_lambda_start",
            "best_anchor_distill_lambda_end",
            "best_anchor_distill_min_improvement",
        ):
            value = float(getattr(config, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} 必须是非负有限数")
            setattr(config, field_name, value)
        if float(config.best_anchor_distill_temperature) <= 0.0:
            raise ValueError("best_anchor_distill_temperature 必须大于 0")
        if int(getattr(config, "best_anchor_distill_ramp_updates", 0)) < 1:
            raise ValueError("best_anchor_distill_ramp_updates 必须大于 0")
        external_path = str(
            getattr(config, "best_anchor_distill_external_checkpoint_path", "")
        ).strip()
        if external_path:
            external_score = float(
                getattr(config, "best_anchor_distill_external_selection_score", float("nan"))
            )
            if not math.isfinite(external_score):
                raise ValueError(
                    "外部 best-anchor 教师必须提供 "
                    "best_anchor_distill_external_selection_score"
                )
            if not str(getattr(config, "best_anchor_distill_external_protocol_id", "")).strip():
                raise ValueError("外部 best-anchor 教师必须提供选择协议 ID")
            manifest_sha = str(
                getattr(config, "best_anchor_distill_external_manifest_sha256", "")
            ).strip().lower()
            if len(manifest_sha) != 64 or any(char not in "0123456789abcdef" for char in manifest_sha):
                raise ValueError("外部 best-anchor 教师必须提供合法的 manifest SHA-256")
    if bool(getattr(config, "ablation_no_gat", False)):
        raise ValueError("ablation_no_gat 已移除；请使用 graph_encoder_mode=none")
    if bool(getattr(config, "ablation_no_pointer", False)):
        raise ValueError("ablation_no_pointer 已移除；请使用 team_selection_mode=static_topq")
    if config.float32_matmul_precision not in {"highest", "high", "medium"}:
        raise ValueError(f"float32_matmul_precision 无效: {config.float32_matmul_precision}")
    if str(getattr(config, "artifact_layout", "runs")).lower() not in {"runs", "legacy"}:
        raise ValueError(f"artifact_layout 无效: {config.artifact_layout}")
    if int(config.num_envs) < 1 or int(config.batch_size) < 1 or int(config.eval_freq) < 1:
        raise ValueError("num_envs、batch_size 和 eval_freq 必须大于 0")
    release_tolerance = float(getattr(config, "release_time_tolerance_hours", 1.0e-5))
    if not math.isfinite(release_tolerance) or release_tolerance < 0.0:
        raise ValueError("release_time_tolerance_hours 必须是非负有限数")
    mismatch_policy = str(getattr(config, "eval_mask_mismatch_policy", "fail")).lower()
    if mismatch_policy not in {"fail", "recover"}:
        raise ValueError("eval_mask_mismatch_policy 必须是 fail 或 recover")
    if int(getattr(config, "eval_mask_mismatch_max_retries_per_time", 16)) < 1:
        raise ValueError("eval_mask_mismatch_max_retries_per_time 必须大于 0")
    if bool(getattr(config, "async_eval_enabled", False)):
        selection_protocol = str(
            getattr(config, "checkpoint_selection_protocol", "single_standard")
        ).strip().lower()
        if selection_protocol not in {"single_standard", "multiscale_manifest"}:
            raise ValueError(
                "checkpoint_selection_protocol 仅支持 single_standard 或 multiscale_manifest"
            )
        async_device = str(getattr(config, "async_eval_device", "cpu")).strip().lower()
        if async_device not in {"cpu", "cuda", "cuda:0"}:
            raise ValueError("async_eval_device 仅支持 cpu、cuda 或 cuda:0")
        if int(getattr(config, "async_eval_cpu_threads", 4)) < 1:
            raise ValueError("async_eval_cpu_threads 必须大于 0")
        if int(getattr(config, "async_eval_queue_capacity", 4)) < 1:
            raise ValueError("async_eval_queue_capacity 必须大于 0")
        if int(getattr(config, "async_eval_worker_count", 1)) < 1:
            raise ValueError("async_eval_worker_count 必须大于 0")
        if async_device.startswith("cuda") and int(getattr(config, "async_eval_worker_count", 1)) != 1:
            raise ValueError("CUDA 异步验证必须使用 async_eval_worker_count=1")
        if int(getattr(config, "async_eval_submit_every_episodes", 1)) < 1:
            raise ValueError("async_eval_submit_every_episodes 必须大于 0")
        if bool(getattr(config, "enable_reschedule_mode", False)):
            if selection_protocol != "single_standard":
                raise ValueError("重调度异步验证不支持 multiscale_manifest checkpoint 选择")
            if not str(getattr(config, "async_eval_instance_id", "")).strip():
                raise ValueError("重调度异步验证的 async_eval_instance_id 不能为空")
            if not str(getattr(config, "async_eval_scenario_id", "")).strip():
                raise ValueError("重调度异步验证的 async_eval_scenario_id 不能为空")
        else:
            scenarios = [str(item).lower() for item in getattr(config, "eval_scenarios", [])]
            if scenarios != ["standard"]:
                raise ValueError("初始调度单实例异步验证仅支持 eval_scenarios=[standard]")
            if selection_protocol == "multiscale_manifest":
                manifest_path = str(
                    getattr(config, "checkpoint_selection_manifest_path", "")
                ).strip()
                if not manifest_path:
                    raise ValueError(
                        "multiscale_manifest checkpoint 选择必须配置 checkpoint_selection_manifest_path"
                    )
                load_initial_checkpoint_selection_manifest(manifest_path)
            elif not str(getattr(config, "async_eval_initial_data_path", "")).strip():
                raise ValueError("初始调度异步验证的 async_eval_initial_data_path 不能为空")
        if not math.isclose(float(getattr(config, "eval_temperature", 0.0)), 0.0, abs_tol=1e-12):
            raise ValueError("异步 best 选择要求 eval_temperature=0.0")
        if int(getattr(config, "eval_freq", 1)) != 1:
            raise ValueError("异步验证要求 eval_freq=1，确保每个成功 PPO episode 都参与 best 选择")
        if str(getattr(config, "async_eval_failure_policy", "fail")).lower() != "fail":
            raise ValueError("async_eval_failure_policy 当前仅支持 fail")
        if int(getattr(config, "async_eval_max_retries", 1)) < 0:
            raise ValueError("async_eval_max_retries 不能小于 0")
        for field_name in (
            "async_eval_poll_interval_sec",
            "async_eval_heartbeat_interval_sec",
            "async_eval_stale_timeout_sec",
        ):
            value = float(getattr(config, field_name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} 必须是大于 0 的有限数")
    uses_synchronous_multi_benchmark = bool(
        getattr(config, "enable_multi_benchmark_eval", False)
        and not (
            bool(getattr(config, "async_eval_enabled", False))
            and not bool(getattr(config, "enable_reschedule_mode", False))
        )
    )
    if uses_synchronous_multi_benchmark:
        refs = getattr(config, "multi_benchmark_reference_makespans", {})
        if isinstance(refs, str):
            import json

            refs = json.loads(refs)
        paths = getattr(config, "multi_benchmark_data_paths", [])
        if not refs:
            raise ValueError("启用多基准评估时必须配置 multi_benchmark_reference_makespans")
        missing = [
            str(path).replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
            for path in paths
            if str(path).replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0] not in refs
        ]
        if missing:
            raise ValueError(f"多基准参考 makespan 缺失: {missing}")
    if config.skill_hub_bidirectional and not config.use_skill_hub:
        config.skill_hub_bidirectional = False


def resolve_runtime_config(
    args: argparse.Namespace,
    *,
    target: Config,
    system_name: str | None = None,
) -> tuple[Config, tuple[str, ...], set[str]]:
    raw_config = getattr(args, "config", ()) or ()
    if isinstance(raw_config, str):
        config_paths = (raw_config,)
    else:
        config_paths = tuple(raw_config)
    _, loaded_paths = load_training_config(
        config_paths,
        target=target,
        system_name=system_name,
    )
    overrides, explicit_fields = collect_cli_overrides(args)
    target.update_from_dict({
        key: _coerce_value(target, key, value)
        for key, value in overrides.items()
    })
    validate_runtime_config(target)
    target.config_paths = loaded_paths
    return target, loaded_paths, explicit_fields


def initialize_training_config(args, argv=None, system_name: str | None = None):
    """兼容旧内部调用；新命令行入口优先使用 runtime.hydra_config。"""
    _, loaded_paths, explicit_fields = resolve_runtime_config(
        args,
        target=configs,
        system_name=system_name,
    )
    args.explicit_config_fields = explicit_fields

    precision = str(configs.float32_matmul_precision)
    if precision not in {"highest", "high", "medium"}:
        raise ValueError(f"float32_matmul_precision 无效: {precision}")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision(precision)

    if (
        str(getattr(configs, "team_selection_mode", "autoregressive"))
        == "autoregressive_pressure_v2"
        and int(configs.num_envs) != 4
    ):
        print(
            "[Runtime][WorkerPointerV2] 警告：最终 num_envs="
            f"{int(configs.num_envs)}，正式探索训练应通过 CLI 再次覆盖为 4。",
            flush=True,
        )

    if uses_runs_layout(configs):
        if getattr(args, "resume", False) and not str(getattr(configs, "run_id", "") or "").strip():
            print("[Runtime] resume 未指定 run_id，使用旧 checkpoint/results 路径兼容模式。", flush=True)
        else:
            context = create_run_context(configs, PROJECT_ROOT, create_dirs=True)
            print(f"[Runtime] run_id={context.run_id} run_dir={context.run_dir}", flush=True)

    print(
        "[Runtime] "
        f"platform={system_name or __import__('platform').system()} "
        f"configs={[str(Path(path)) for path in loaded_paths]} "
        f"num_envs={configs.num_envs} "
        f"worker_threads={configs.vector_env_worker_threads} "
        f"start_method={configs.vector_env_start_method} "
        f"amp={configs.lightning_precision} "
        f"matmul_precision={precision}",
        flush=True,
    )
    return configs


def _add_argument_if_missing(
    parser: argparse.ArgumentParser,
    *flags: str,
    **kwargs: Any,
) -> None:
    if any(flag in parser._option_string_actions for flag in flags):
        return
    parser.add_argument(*flags, **kwargs)


def add_common_config_arguments(parser: argparse.ArgumentParser) -> None:
    _add_argument_if_missing(parser, "--config", action="append", default=[])
    _add_argument_if_missing(parser, "--set", dest="set_values", action="append", default=[])
    _add_argument_if_missing(parser, "--data-path", "--data_path", dest="data_path")
    _add_argument_if_missing(parser, "--train-data-path", dest="train_data_path")
    _add_argument_if_missing(parser, "--seed", type=int)
    _add_argument_if_missing(parser, "--max-episodes", "--max_episodes", dest="max_episodes", type=int)
    _add_argument_if_missing(parser, "--num-envs", dest="num_envs", type=int)
    _add_argument_if_missing(parser, "--batch-size", "--batch_size", dest="batch_size", type=int)
    _add_argument_if_missing(parser, "--eval-freq", dest="eval_freq", type=int)
    _add_argument_if_missing(parser, "--log-dir", "--log_dir", dest="log_dir")
    _add_argument_if_missing(parser, "--output-dir", "--output_dir", "--result_dir", dest="output_dir")
    _add_argument_if_missing(parser, "--run-id", "--run_id", dest="run_id")
    _add_argument_if_missing(parser, "--runs-root", "--runs_root", dest="runs_root")
    _add_argument_if_missing(parser, "--use-skill-hub", action=argparse.BooleanOptionalAction, default=None)
    _add_argument_if_missing(
        parser,
        "--skill-hub-bidirectional",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
