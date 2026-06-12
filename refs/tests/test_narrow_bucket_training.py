from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from configs import Config, load_config_files
from ppo_agent import PPOAgent
from utils.generate_random_dataset import generate_bucket


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("name", "data_file", "pool_dir", "workers", "batch_size"),
    [
        ("283", "data/283.csv", "data/generated/initial_283", 80, 32),
        ("680", "data/680.csv", "data/generated/initial_680", 100, 16),
        ("2338", "data/2338.csv", "data/generated/initial_2338", 140, 8),
        ("3182", "data/3182.csv", "data/generated/initial_3182", 160, 4),
    ],
)
def test_bucket_experiment_config(
    name: str,
    data_file: str,
    pool_dir: str,
    workers: int,
    batch_size: int,
) -> None:
    cfg = Config()
    load_config_files(
        [str(PROJECT_ROOT / "conf" / "experiment" / f"initial_schedule_{name}.yaml")],
        target=cfg,
    )
    assert cfg.data_file_path == data_file
    assert cfg.train_data_path_or_dir == pool_dir
    assert cfg.n_w == workers
    assert cfg.n_w_min == workers
    assert cfg.batch_size == batch_size
    assert cfg.experiment_name == f"initial_schedule_{name}"


def test_windows_low_memory_override_forces_batch_four() -> None:
    cfg = Config()
    load_config_files(
        [
            str(PROJECT_ROOT / "conf" / "experiment" / "initial_schedule_283.yaml"),
            str(PROJECT_ROOT / "conf" / "hardware" / "windows_4060_low_memory.yaml"),
        ],
        target=cfg,
    )
    assert cfg.batch_size == 4
    assert cfg.num_envs_windows == 2


def test_generator_is_deterministic_and_writes_manifest(tmp_path: Path) -> None:
    worker_pool = PROJECT_ROOT / "data" / "worker_pool_fixed.csv"
    template = PROJECT_ROOT / "data" / "283.csv"
    output_a = tmp_path / "a"
    output_b = tmp_path / "b"
    kwargs = {
        "min_length": 280,
        "max_length": 280,
        "num_samples": 1,
        "time_var": 0.05,
        "seed": 123,
        "worker_pool_path": worker_pool,
    }
    manifest_a = generate_bucket(template, output_a, **kwargs)
    manifest_b = generate_bucket(template, output_b, **kwargs)

    assert manifest_a["template_sha256"] == manifest_b["template_sha256"]
    assert manifest_a["files"][0]["actual_task_count"] == 280
    assert manifest_b["files"][0]["actual_task_count"] == 280
    file_a = output_a / manifest_a["files"][0]["file"]
    file_b = output_b / manifest_b["files"][0]["file"]
    assert hashlib.sha256(file_a.read_bytes()).hexdigest() == hashlib.sha256(file_b.read_bytes()).hexdigest()
    assert (output_a / "baseline_283.csv").exists()
    assert (output_a / "manifest.json").exists()


def test_snapshot_homogeneity_accepts_one_graph_and_worker_count() -> None:
    states = [
        {"dataset_idx": 2, "worker_free_time": [0.0] * 80},
        {"dataset_idx": 2, "worker_free_time": [1.0] * 80},
    ]
    PPOAgent.validate_snapshot_homogeneity(states)


@pytest.mark.parametrize(
    "states",
    [
        [
            {"dataset_idx": 0, "worker_free_time": [0.0] * 80},
            {"dataset_idx": 1, "worker_free_time": [0.0] * 80},
        ],
        [
            {"dataset_idx": 0, "worker_free_time": [0.0] * 80},
            {"dataset_idx": 0, "worker_free_time": [0.0] * 81},
        ],
    ],
)
def test_snapshot_homogeneity_rejects_mixed_update(states: list[dict]) -> None:
    with pytest.raises(RuntimeError, match="同质窄池轨迹"):
        PPOAgent.validate_snapshot_homogeneity(states)


def test_legacy_large_scale_training_references_are_removed() -> None:
    forbidden = (
        "compute_batch_size_from_staircase",
        "compute_worker_count",
        "worker_scaling_mode",
        "data/random_datasets",
        "batchsize_staircase",
    )
    paths = [
        PROJECT_ROOT / "configs.py",
        PROJECT_ROOT / "environment.py",
        PROJECT_ROOT / "train.py",
        PROJECT_ROOT / "ppo_agent.py",
        PROJECT_ROOT / "conf" / "env" / "apal_default.yaml",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    for token in forbidden:
        assert token not in combined
