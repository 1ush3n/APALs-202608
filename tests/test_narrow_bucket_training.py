from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from configs import Config, load_config_files
from utils.generate_random_dataset import generate_bucket


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("name", "data_file", "pool_dir", "workers", "batch_size"),
    [
        ("283", "data/283.csv", "data/generated/initial_283", 80, 512),
        ("680", "data/680.csv", "data/generated/initial_680", 100, 256),
        ("2338", "data/2338.csv", "data/generated/initial_2338", 140, 128),
        ("3182", "data/3182.csv", "data/generated/initial_3182", 160, 64),
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


def test_generator_is_deterministic_and_manifest_is_portable(tmp_path: Path) -> None:
    kwargs = {
        "min_length": 280,
        "max_length": 280,
        "num_samples": 1,
        "time_var": 0.05,
        "seed": 123,
        "worker_pool_path": PROJECT_ROOT / "data" / "worker_pool_fixed.csv",
    }
    manifest_a = generate_bucket(PROJECT_ROOT / "data" / "283.csv", tmp_path / "a", **kwargs)
    manifest_b = generate_bucket(PROJECT_ROOT / "data" / "283.csv", tmp_path / "b", **kwargs)
    file_a = tmp_path / "a" / manifest_a["files"][0]["file"]
    file_b = tmp_path / "b" / manifest_b["files"][0]["file"]

    assert manifest_a["template"] == "data/283.csv"
    assert manifest_a["files"][0]["actual_task_count"] == 280
    assert hashlib.sha256(file_a.read_bytes()).hexdigest() == hashlib.sha256(file_b.read_bytes()).hexdigest()
