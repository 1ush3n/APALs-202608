from __future__ import annotations

import platform
import sys
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from configs import Config
from runtime.artifacts import run_context, uses_runs_layout
from runtime.configuration import validate_runtime_config


class HydraCliError(ValueError):
    """命令行不符合当前 Hydra 原生入口约定。"""


@dataclass(frozen=True)
class ExtraArgument:
    default: Any = None
    required: bool = False
    help: str = ""


@dataclass(frozen=True)
class ParsedHydraArgs:
    experiment: str
    hardware: str
    resume: bool
    config_overrides: tuple[str, ...]
    extra_values: dict[str, Any] = field(default_factory=dict)
    explicit_fields: set[str] = field(default_factory=set)


OLD_CLI_FLAGS = {
    "--config",
    "--set",
    "--trainer",
    "--batch-size",
    "--batch_size",
    "--num-envs",
    "--max-episodes",
    "--eval-freq",
    "--log-dir",
    "--output-dir",
    "--run-id",
    "--runs-root",
}


def hardware_name_for_platform(system_name: str | None = None) -> str:
    system = system_name or platform.system()
    if system == "Windows":
        return "windows_4060_low_memory"
    if system == "Linux":
        return "linux_server"
    raise RuntimeError(f"不支持的训练平台: {system!r}；仅支持 Windows 和 Linux")


def _parse_bool(raw: str) -> bool:
    lowered = str(raw).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise HydraCliError(f"布尔参数必须为 true/false，收到: {raw!r}")


def _parse_value(raw: str) -> Any:
    cfg = OmegaConf.from_dotlist([f"value={raw}"])
    value = OmegaConf.to_container(cfg, resolve=True)["value"]
    return value


def _normalize_key(key: str) -> str:
    return key.strip().replace("-", "_")


def _field_name_from_override(key: str) -> str:
    return _normalize_key(key).rsplit(".", 1)[-1]


def _flatten_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, Mapping):
            flat.update(_flatten_mapping(value))
        else:
            flat[str(key)] = value
    return flat


def _check_old_cli(argv: Iterable[str]) -> None:
    for token in argv:
        if token in {"-h", "--help"}:
            continue
        flag = token.split("=", 1)[0]
        if flag in OLD_CLI_FLAGS or flag.startswith("--config") or flag.startswith("--set"):
            raise HydraCliError(
                "当前版本已切换为 Hydra 原生命令，不再支持旧 argparse 参数。\n"
                "示例: python train.py experiment=initial_schedule_283 train.batch_size=32\n"
                f"收到旧参数: {token}"
            )
        if token.startswith("--"):
            raise HydraCliError(
                "当前入口仅接受 Hydra key=value 覆盖；脚本专属参数也使用 key=value。\n"
                f"无法解析: {token}"
            )


def parse_hydra_args(
    argv: list[str] | None = None,
    *,
    default_experiment: str = "default",
    extra_arguments: Mapping[str, ExtraArgument] | None = None,
    system_name: str | None = None,
) -> ParsedHydraArgs:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    _check_old_cli(raw_args)

    experiment = default_experiment
    hardware = hardware_name_for_platform(system_name)
    resume = False
    config_overrides: list[str] = []
    extra_values: dict[str, Any] = {
        key: spec.default for key, spec in (extra_arguments or {}).items()
    }
    explicit_fields: set[str] = set()

    for token in raw_args:
        if token in {"-h", "--help"}:
            continue
        key, separator, raw_value = token.partition("=")
        if not separator:
            raise HydraCliError(f"Hydra 覆盖必须使用 key=value 格式: {token!r}")
        normalized_key = _normalize_key(key)
        leaf = _field_name_from_override(normalized_key)

        if normalized_key == "experiment":
            experiment = str(raw_value)
            continue
        if normalized_key == "hardware":
            raise HydraCliError("硬件配置由平台自动选择，请不要手动传入 hardware=...")
        if normalized_key in {"trainer", "runtime.trainer"}:
            if str(raw_value).lower() == "legacy":
                raise HydraCliError("legacy 训练入口已归档；请使用默认 Lightning 入口。")
            if str(raw_value).lower() == "lightning":
                continue
            raise HydraCliError(f"未知 trainer: {raw_value!r}")
        if normalized_key in {"resume", "runtime.resume"}:
            resume = _parse_bool(raw_value)
            continue

        if extra_arguments and leaf in extra_arguments:
            extra_values[leaf] = _parse_value(raw_value)
            continue

        config_overrides.append(f"{key}={raw_value}")
        explicit_fields.add(leaf)

    missing = [
        key for key, spec in (extra_arguments or {}).items()
        if spec.required and extra_values.get(key) in (None, "")
    ]
    if missing:
        raise HydraCliError(f"缺少必需参数: {', '.join(missing)}")

    return ParsedHydraArgs(
        experiment=experiment,
        hardware=hardware,
        resume=resume,
        config_overrides=tuple(config_overrides),
        extra_values=extra_values,
        explicit_fields=explicit_fields,
    )


def _compose_config(config_dir: Path, config_name: str) -> DictConfig:
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(config_dir.resolve()),
        job_name="apal",
    ):
        return compose(config_name=config_name)


def compose_hydra_config(
    parsed: ParsedHydraArgs,
    *,
    config_dir: Path,
) -> DictConfig:
    experiment_cfg = _compose_config(config_dir, f"experiment/{parsed.experiment}")
    hardware_cfg = _compose_config(config_dir, f"hardware/{parsed.hardware}")
    if "hardware" in hardware_cfg and isinstance(hardware_cfg["hardware"], DictConfig):
        hardware_cfg = OmegaConf.create(hardware_cfg["hardware"])
    override_cfg = OmegaConf.from_dotlist(list(parsed.config_overrides))
    OmegaConf.set_struct(experiment_cfg, False)
    OmegaConf.set_struct(hardware_cfg, False)
    return OmegaConf.merge(experiment_cfg, hardware_cfg, override_cfg)


def apply_hydra_config(
    hydra_cfg: DictConfig,
    *,
    target: Config,
    config_paths: tuple[str, ...],
) -> set[str]:
    resolved = OmegaConf.to_container(hydra_cfg, resolve=True)
    if not isinstance(resolved, Mapping):
        raise HydraCliError("Hydra 配置根节点必须是 mapping")

    flat = _flatten_mapping(resolved)
    unknown = sorted(key for key in flat if not hasattr(target, key))
    if unknown:
        raise KeyError(f"Hydra 配置包含未知字段: {unknown}")
    target.update_from_dict(flat)
    target.config_paths = config_paths
    validate_runtime_config(target)
    return set(flat)


def initialize_hydra_runtime(
    argv: list[str] | None,
    *,
    target: Config,
    project_root: Path,
    default_experiment: str = "default",
    extra_arguments: Mapping[str, ExtraArgument] | None = None,
    system_name: str | None = None,
    create_run_context: bool = True,
) -> Namespace:
    parsed = parse_hydra_args(
        argv,
        default_experiment=default_experiment,
        extra_arguments=extra_arguments,
        system_name=system_name,
    )
    config_dir = project_root / "conf"
    hydra_cfg = compose_hydra_config(parsed, config_dir=config_dir)
    config_paths = (
        str((config_dir / "experiment" / f"{parsed.experiment}.yaml").resolve()),
        str((config_dir / "hardware" / f"{parsed.hardware}.yaml").resolve()),
    )
    apply_hydra_config(hydra_cfg, target=target, config_paths=config_paths)

    precision = str(target.float32_matmul_precision)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision(precision)

    if create_run_context and uses_runs_layout(target):
        context = run_context(target, project_root, create_dirs=True)
        print(f"[Runtime] run_id={context.run_id} run_dir={context.run_dir}", flush=True)

    runtime_args = Namespace(
        resume=parsed.resume,
        explicit_config_fields=parsed.explicit_fields,
        hydra_experiment=parsed.experiment,
        hydra_hardware=parsed.hardware,
        hydra_overrides=list(parsed.config_overrides),
        **parsed.extra_values,
    )
    for key, value in target.to_flat_dict().items():
        if not hasattr(runtime_args, key):
            setattr(runtime_args, key, value)

    print(
        "[Runtime] "
        f"platform={system_name or platform.system()} "
        f"experiment={parsed.experiment} "
        f"hardware={parsed.hardware} "
        f"num_envs={target.num_envs} "
        f"worker_threads={target.vector_env_worker_threads} "
        f"start_method={target.vector_env_start_method} "
        f"amp={target.lightning_precision} "
        f"matmul_precision={precision}",
        flush=True,
    )
    return runtime_args


def initialize_keyvalue_args(
    argv: list[str] | None,
    *,
    extra_arguments: Mapping[str, ExtraArgument],
) -> Namespace:
    """解析脚本级 key=value 参数；不加载 YAML，也不修改全局训练配置。"""
    raw_args = list(sys.argv[1:] if argv is None else argv)
    _check_old_cli(raw_args)

    values: dict[str, Any] = {key: spec.default for key, spec in extra_arguments.items()}
    explicit: set[str] = set()
    for token in raw_args:
        if token in {"-h", "--help"}:
            continue
        key, separator, raw_value = token.partition("=")
        if not separator:
            raise HydraCliError(f"脚本参数必须使用 key=value 格式: {token!r}")
        normalized_key = _normalize_key(key)
        leaf = _field_name_from_override(normalized_key)
        if leaf not in extra_arguments:
            allowed = ", ".join(sorted(extra_arguments))
            raise HydraCliError(f"未知脚本参数 {key!r}；可用参数: {allowed}")
        values[leaf] = _parse_value(raw_value)
        explicit.add(leaf)

    missing = [
        key for key, spec in extra_arguments.items()
        if spec.required and values.get(key) in (None, "")
    ]
    if missing:
        raise HydraCliError(f"缺少必需参数: {', '.join(missing)}")

    return Namespace(explicit_fields=explicit, **values)


def hydra_help(extra_arguments: Mapping[str, ExtraArgument] | None = None) -> str:
    extra_lines = []
    for key, spec in (extra_arguments or {}).items():
        marker = "必需" if spec.required else f"默认={spec.default!r}"
        extra_lines.append(f"  {key}=...    {marker} {spec.help}".rstrip())
    extra_block = "\n".join(extra_lines) if extra_lines else "  无"
    return (
        "APAL Hydra 原生入口\n\n"
        "基本用法:\n"
        "  python train.py experiment=initial_schedule_283\n"
        "  python train.py experiment=initial_schedule_283 train.batch_size=32 train.num_envs=4\n"
        "  python train.py experiment=reschedule_task_delay resume=true\n\n"
        "说明:\n"
        "  - Windows/Linux 硬件配置会自动选择，不需要传 hardware=...\n"
        "  - 旧参数 --config、--set、--trainer legacy 已废弃\n"
        "  - 所有配置覆盖使用 Hydra key=value\n\n"
        "脚本专属参数:\n"
        f"{extra_block}\n"
    )


def should_show_help(argv: list[str] | None) -> bool:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    return any(token in {"-h", "--help"} for token in raw_args)
