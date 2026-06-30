from __future__ import annotations

import argparse
from typing import Any

from configs import Config, load_training_config


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
    return overrides, explicit_fields


def validate_runtime_config(config: Config) -> None:
    if config.float32_matmul_precision not in {"highest", "high", "medium"}:
        raise ValueError(f"float32_matmul_precision 无效: {config.float32_matmul_precision}")
    if str(getattr(config, "artifact_layout", "runs")).lower() not in {"runs", "legacy"}:
        raise ValueError(f"artifact_layout 无效: {config.artifact_layout}")
    if int(config.num_envs) < 1 or int(config.batch_size) < 1 or int(config.eval_freq) < 1:
        raise ValueError("num_envs、batch_size 和 eval_freq 必须大于 0")
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
    _, loaded_paths = load_training_config(
        tuple(getattr(args, "config", ()) or ()),
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


def add_common_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--set", dest="set_values", action="append", default=[])
    parser.add_argument("--data-path", "--data_path", dest="data_path")
    parser.add_argument("--train-data-path", dest="train_data_path")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-episodes", "--max_episodes", dest="max_episodes", type=int)
    parser.add_argument("--num-envs", dest="num_envs", type=int)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int)
    parser.add_argument("--eval-freq", dest="eval_freq", type=int)
    parser.add_argument("--log-dir", "--log_dir", dest="log_dir")
    parser.add_argument("--output-dir", "--result_dir", dest="output_dir")
    parser.add_argument("--run-id", "--run_id", dest="run_id")
    parser.add_argument("--runs-root", "--runs_root", dest="runs_root")
    parser.add_argument("--use-skill-hub", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--skill-hub-bidirectional",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
