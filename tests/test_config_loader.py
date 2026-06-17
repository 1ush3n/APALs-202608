# -*- coding: utf-8 -*-
"""验证分层 YAML 配置加载保持旧 Config 单例兼容。"""

from __future__ import annotations

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
from args_parser import get_base_parser
from runtime.configuration import parse_set_overrides, resolve_runtime_config
from train import PROJECT_ROOT, resolve_checkpoint_paths, resolve_tensorboard_log_root, write_best_model_meta


def test_layered_yaml_config_loads_into_flat_config() -> None:
    cfg = Config()
    load_config_files([str(PROJECT_ROOT / "conf" / "experiment" / "default.yaml")], target=cfg)

    assert cfg.use_input_layer_norm is True
    assert cfg.use_gat_layer_norm is False
    assert cfg.use_head_layer_norm is False
    assert cfg.use_rollout_snapshot_fastpath is True
    assert cfg.n_m == 5
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
    assert resolve_tensorboard_log_root(cfg) == Path("/root/tf-logs")


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
    parser = get_base_parser()
    args = parser.parse_args([
        "--config", str(PROJECT_ROOT / "conf" / "experiment" / "initial_schedule_283.yaml"),
        "--batch-size", "12",
        "--num-envs", "3",
        "--no-use-skill-hub",
        "--set", "eval_scenarios=[standard,duration_noise]",
    ])
    cfg = Config()
    _, _, explicit = resolve_runtime_config(args, target=cfg, system_name="Windows")

    assert cfg.batch_size == 12
    assert cfg.num_envs == 3
    assert cfg.use_skill_hub is False
    assert cfg.skill_hub_bidirectional is False
    assert cfg.eval_scenarios == ["standard", "duration_noise"]
    assert {"batch_size", "num_envs", "use_skill_hub", "eval_scenarios"} <= explicit


def test_set_rejects_invalid_syntax_and_unknown_fields() -> None:
    with pytest.raises(ValueError, match="key=value"):
        parse_set_overrides(["batch_size"])

    parser = get_base_parser()
    args = parser.parse_args(["--set", "not_a_config_field=1"])
    with pytest.raises(KeyError, match="未知配置字段"):
        resolve_runtime_config(args, target=Config(), system_name="Windows")
