from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch

from configs import Config, configs, load_training_config
from runtime.artifacts import run_context as create_run_context
from runtime.artifacts import uses_runs_layout
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
    "use_input_layer_norm", "use_gat_layer_norm", "use_head_layer_norm",
    "use_shared_trunk", "use_attention_critic", "use_autoregressive_worker",
}


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
    if bool(getattr(config, "enable_multi_benchmark_eval", False)):
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
