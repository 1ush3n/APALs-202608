# -*- coding: utf-8 -*-
"""验证分层 YAML 配置加载保持旧 Config 单例兼容。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import (
    Config,
    load_config_files,
    load_training_config,
    resolve_platform_hardware_config,
)
from runtime.configuration import parse_set_overrides, resolve_runtime_config
from runtime.hydra_config import (
    ExtraArgument,
    HydraCliError,
    apply_hydra_config,
    compose_hydra_config,
    initialize_hydra_runtime,
    parse_hydra_args,
)
from runtime.artifacts import (
    build_run_id,
    checkpoint_paths as artifact_checkpoint_paths,
    resolve_run_output_dir,
    run_context,
)
from runtime.paths import PROJECT_ROOT, resolve_checkpoint_paths, resolve_tensorboard_log_root, write_best_model_meta


def test_layered_yaml_config_loads_into_flat_config() -> None:
    cfg = Config()
    load_config_files([str(PROJECT_ROOT / "conf" / "experiment" / "default.yaml")], target=cfg)

    assert cfg.use_input_layer_norm is True
    assert cfg.use_gat_layer_norm is False
    assert cfg.use_head_layer_norm is False
    assert cfg.use_rollout_snapshot_fastpath is True
    assert cfg.n_m == 5
    assert cfg.batch_size == 32


def test_hydra_root_config_loads_through_compat_loader() -> None:
    cfg = Config()
    load_config_files([str(PROJECT_ROOT / "conf" / "config.yaml")], target=cfg)

    assert cfg.use_skill_hub is True
    assert cfg.use_rollout_snapshot_fastpath is True
    assert cfg.batch_size == 32


def test_later_yaml_overrides_earlier_yaml(tmp_path: Path) -> None:
    override = tmp_path / "override.yaml"
    override.write_text(
        "train:\n"
        "  batch_size: 8\n"
        "model:\n"
        "  use_head_layer_norm: true\n",
        encoding="utf-8",
    )

    cfg = Config()
    load_config_files(
        [
            str(PROJECT_ROOT / "conf" / "experiment" / "default.yaml"),
            str(override),
        ],
        target=cfg,
    )

    assert cfg.batch_size == 8
    assert cfg.use_head_layer_norm is True


def test_initial_schedule_config_disables_dynamic_events_but_keeps_training_randomization() -> None:
    cfg = Config()
    load_config_files([str(PROJECT_ROOT / "conf" / "experiment" / "initial_schedule.yaml")], target=cfg)

    assert cfg.experiment_name == "initial_schedule"
    assert cfg.checkpoint_root == "checkpoints"
    assert cfg.enable_dynamic_events is False
    assert cfg.enable_station_breakdown is False
    assert cfg.enable_material_delay is False
    assert cfg.enable_online_duration_perturb is False
    assert cfg.enable_worker_fatigue is False
    assert cfg.prob_worker_absent_base == 0.0
    assert cfg.prob_worker_absent_max == 0.0
    assert cfg.prob_station_breakdown_base == 0.0
    assert cfg.prob_station_breakdown_max == 0.0
    assert cfg.prob_material_delay_base == 0.0
    assert cfg.prob_material_delay_max == 0.0
    assert cfg.online_perturb_prob_per_step == 0.0
    assert cfg.randomize_durations is True
    assert cfg.dur_random_range == 0.2


def test_experiment_checkpoint_paths_and_best_model_meta_are_isolated(tmp_path: Path) -> None:
    cfg = Config()
    cfg.experiment_name = "initial_schedule_test"
    cfg.checkpoint_root = str(tmp_path / "checkpoints")
    cfg.config_paths = ("conf/experiment/initial_schedule.yaml",)
    cfg.data_file_path = "data/283.csv"
    cfg.train_data_path_or_dir = "data/random_datasets"

    paths = resolve_checkpoint_paths(cfg)

    assert paths["checkpoint_path"] == tmp_path / "checkpoints" / "initial_schedule_test" / "latest_checkpoint.pth"
    assert paths["best_model_path"] == tmp_path / "checkpoints" / "initial_schedule_test" / "bestmodel" / "best_model.pth"
    assert paths["checkpoint_path"] != tmp_path / "checkpoints" / "latest_checkpoint.pth"

    write_best_model_meta(paths["best_model_meta_path"], episode=7, eval_makespan=123.4, config_obj=cfg)
    meta_text = paths["best_model_meta_path"].read_text(encoding="utf-8")

    assert '"episode": 7' in meta_text
    assert '"eval_makespan": 123.4' in meta_text
    assert "conf/experiment/initial_schedule.yaml" in meta_text


def test_tensorboard_log_root_strictly_uses_config() -> None:
    cfg = Config()
    cfg.log_dir = "/root/tf-logs"
    cfg.artifact_layout = "legacy"
    assert resolve_tensorboard_log_root(cfg) == Path("/root/tf-logs")


def test_runs_layout_creates_stable_run_context(tmp_path: Path) -> None:
    cfg = Config()
    cfg.experiment_name = "initial schedule 680"
    cfg.runs_root = str(tmp_path / "runs")
    cfg.run_id = "initial_schedule_680_260630-153000"

    context = run_context(cfg, PROJECT_ROOT, create_dirs=True)

    assert context.experiment_name == "initial_schedule_680"
    assert context.run_id == "initial_schedule_680_260630-153000"
    assert context.run_dir == tmp_path / "runs" / "initial_schedule_680" / "initial_schedule_680_260630-153000"
    assert context.checkpoint_dir.exists()
    assert context.configs_dir.exists()
    assert context.eval_dir.exists()
    assert str(context.run_dir) == cfg.run_dir


def test_runs_layout_checkpoint_paths_do_not_use_legacy_root(tmp_path: Path) -> None:
    cfg = Config()
    cfg.experiment_name = "initial_schedule_680"
    cfg.runs_root = str(tmp_path / "runs")
    cfg.run_id = "initial_schedule_680_260630-153000"

    paths = artifact_checkpoint_paths(cfg, PROJECT_ROOT)

    assert paths["lightning_latest"] == tmp_path / "runs" / "initial_schedule_680" / "initial_schedule_680_260630-153000" / "checkpoints" / "last.ckpt"
    assert paths["lightning_best"] == tmp_path / "runs" / "initial_schedule_680" / "initial_schedule_680_260630-153000" / "checkpoints" / "best.ckpt"
    assert paths["legacy_latest"].parent.name == "legacy"


def test_auto_run_id_uses_experiment_and_compact_timestamp() -> None:
    from datetime import datetime

    cfg = Config()
    cfg.experiment_name = "initial_schedule_680"

    assert build_run_id(cfg, now=datetime(2026, 6, 30, 15, 30, 0)) == "initial_schedule_680_260630-153000"


def test_training_config_selects_windows_low_memory_profile() -> None:
    cfg = Config()
    _, paths = load_training_config(
        [str(PROJECT_ROOT / "conf" / "experiment" / "initial_schedule_283.yaml")],
        target=cfg,
        system_name="Windows",
    )

    assert cfg.num_envs == 2
    assert cfg.vector_env_start_method == "spawn"
    assert cfg.batch_size == 4
    assert cfg.ppo_batch_size_cap == 4
    assert Path(paths[-1]).name == "windows_4060_low_memory.yaml"


def test_training_config_selects_linux_profile() -> None:
    cfg = Config()
    _, paths = load_training_config(
        [str(PROJECT_ROOT / "conf" / "experiment" / "initial_schedule_283.yaml")],
        target=cfg,
        system_name="Linux",
    )

    assert cfg.num_envs == 16
    assert cfg.vector_env_start_method == "forkserver"
    assert cfg.ppo_batch_size_cap == 0
    assert cfg.batch_size == 512
    assert Path(paths[-1]).name == "linux_server.yaml"


def test_experiments_do_not_embed_hardware_profiles() -> None:
    experiment_dir = PROJECT_ROOT / "conf" / "experiment"
    for path in experiment_dir.glob("*.yaml"):
        assert "../hardware/" not in path.read_text(encoding="utf-8")


def test_unsupported_platform_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="不支持的训练平台"):
        resolve_platform_hardware_config(system_name="Darwin")


def test_cli_overrides_yaml_and_platform_profile() -> None:
    cfg = Config()
    parsed = parse_hydra_args(
        [
            "experiment=initial_schedule_283",
            "train.batch_size=12",
            "parallel.num_envs=3",
            "use_skill_hub=false",
            "run_id=manual_260630-153000",
            "eval_scenarios=[standard,duration_noise]",
        ],
        system_name="Windows",
    )
    hydra_cfg = compose_hydra_config(parsed, config_dir=PROJECT_ROOT / "conf")
    explicit = apply_hydra_config(
        hydra_cfg,
        target=cfg,
        config_paths=(
            str(PROJECT_ROOT / "conf" / "experiment" / "initial_schedule_283.yaml"),
            str(PROJECT_ROOT / "conf" / "hardware" / "windows_4060_low_memory.yaml"),
        ),
    )

    assert cfg.batch_size == 12
    assert cfg.num_envs == 3
    assert cfg.use_skill_hub is False
    assert cfg.skill_hub_bidirectional is False
    assert cfg.run_id == "manual_260630-153000"
    assert cfg.eval_scenarios == ["standard", "duration_noise"]
    assert {"batch_size", "num_envs", "use_skill_hub", "run_id", "eval_scenarios"} <= explicit


def test_hydra_style_overrides_are_compatible_with_flat_config() -> None:
    cfg = Config()
    parsed = parse_hydra_args(
        [
            "experiment=initial_schedule_283",
            "train.batch_size=24",
            "parallel.num_envs=5",
            "artifacts.runs_root=tmp_runs",
            "experiment.experiment_name=hydra_compat",
        ],
        system_name="Windows",
    )
    hydra_cfg = compose_hydra_config(parsed, config_dir=PROJECT_ROOT / "conf")
    explicit = apply_hydra_config(
        hydra_cfg,
        target=cfg,
        config_paths=(
            str(PROJECT_ROOT / "conf" / "experiment" / "initial_schedule_283.yaml"),
            str(PROJECT_ROOT / "conf" / "hardware" / "windows_4060_low_memory.yaml"),
        ),
    )

    assert cfg.batch_size == 24
    assert cfg.num_envs == 5
    assert cfg.runs_root == "tmp_runs"
    assert cfg.experiment_name == "hydra_compat"
    assert {"batch_size", "num_envs", "runs_root", "experiment_name"} <= explicit


def test_initialize_runtime_cli_leaf_override_wins_nested_experiment_defaults() -> None:
    cfg = Config()
    args = initialize_hydra_runtime(
        [
            "experiment=scale_400_800_schedule",
            "train.batch_size=64",
            "train_data_path_or_dir=data/scale_400_800_datasets",
        ],
        target=cfg,
        project_root=PROJECT_ROOT,
        default_experiment="initial_schedule_283",
        system_name="Linux",
        create_run_context=False,
    )

    assert cfg.batch_size == 64
    assert args.batch_size == 64
    assert cfg.train_data_path_or_dir == "data/scale_400_800_datasets"


def test_script_extra_arguments_do_not_enter_hydra_config() -> None:
    parsed = parse_hydra_args(
        [
            "experiment=reschedule_task_delay",
            "manifest_path=data/reschedule_manifests/reschedule_400_600_seed20260701.json",
            "instance_ids=[real_283,real_680]",
        ],
        extra_arguments={
            "manifest_path": ExtraArgument(default=None),
            "instance_ids": ExtraArgument(default=None),
        },
        system_name="Linux",
    )

    assert "manifest_path" not in " ".join(parsed.config_overrides)
    assert "instance_ids" not in " ".join(parsed.config_overrides)
    assert parsed.extra_values["manifest_path"] == "data/reschedule_manifests/reschedule_400_600_seed20260701.json"
    assert parsed.extra_values["instance_ids"] == ["real_283", "real_680"]


def test_manifest_script_arguments_are_safe_if_misrouted_to_hydra() -> None:
    parsed = parse_hydra_args(
        [
            "experiment=reschedule_task_delay",
            "manifest_path=data/reschedule_manifests/reschedule_400_600_seed20260701.json",
            "instance_ids=[real_283,real_680]",
        ],
        system_name="Linux",
    )
    cfg = Config()
    hydra_cfg = compose_hydra_config(parsed, config_dir=PROJECT_ROOT / "conf")
    explicit = apply_hydra_config(
        hydra_cfg,
        target=cfg,
        config_paths=(
            str(PROJECT_ROOT / "conf" / "experiment" / "reschedule_task_delay.yaml"),
            str(PROJECT_ROOT / "conf" / "hardware" / "linux_server.yaml"),
        ),
    )

    assert cfg.manifest_path == "data/reschedule_manifests/reschedule_400_600_seed20260701.json"
    assert cfg.instance_ids == ["real_283", "real_680"]
    assert {"manifest_path", "instance_ids"} <= explicit


def test_hydra_style_override_rejects_unknown_fields() -> None:
    parsed = parse_hydra_args(["experiment=initial_schedule_283", "train.no_such_field=1"], system_name="Windows")
    hydra_cfg = compose_hydra_config(parsed, config_dir=PROJECT_ROOT / "conf")

    with pytest.raises(KeyError, match="未知字段"):
        apply_hydra_config(
            hydra_cfg,
            target=Config(),
            config_paths=(
                str(PROJECT_ROOT / "conf" / "experiment" / "initial_schedule_283.yaml"),
                str(PROJECT_ROOT / "conf" / "hardware" / "windows_4060_low_memory.yaml"),
            ),
        )


def test_old_public_cli_flags_are_rejected() -> None:
    with pytest.raises(HydraCliError, match="不再支持旧 argparse 参数"):
        parse_hydra_args(["--config", "conf/experiment/initial_schedule_283.yaml"], system_name="Windows")

    with pytest.raises(HydraCliError, match="legacy 训练入口已归档"):
        parse_hydra_args(["trainer=legacy"], system_name="Windows")


def test_single_string_config_and_run_output_dir_are_supported(tmp_path: Path) -> None:
    args = argparse.Namespace(
        config=str(PROJECT_ROOT / "conf" / "experiment" / "initial_schedule_283.yaml"),
        set_values=[],
        hydra_overrides=[],
    )
    cfg = Config()
    resolve_runtime_config(args, target=cfg, system_name="Windows")
    cfg.runs_root = str(tmp_path / "runs")
    cfg.run_id = "tool_run_260630-153000"

    output_dir, context = resolve_run_output_dir(
        cfg,
        PROJECT_ROOT,
        default_legacy_dir="results/eval_logs",
        run_subdir="baselines/heuristic",
        explicit_dir=None,
        section="artifacts",
    )

    assert context is not None
    assert output_dir == tmp_path / "runs" / cfg.experiment_name / "tool_run_260630-153000" / "artifacts" / "baselines" / "heuristic"
    assert output_dir.exists()


def test_set_rejects_invalid_syntax_and_unknown_fields() -> None:
    with pytest.raises(ValueError, match="key=value"):
        parse_set_overrides(["batch_size"])

    args = argparse.Namespace(
        config=[],
        set_values=["not_a_config_field=1"],
        hydra_overrides=[],
    )
    with pytest.raises(KeyError, match="未知配置字段"):
        resolve_runtime_config(args, target=Config(), system_name="Windows")
