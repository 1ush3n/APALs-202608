from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from runtime.initial_worker_mapping import (
    apply_initial_worker_mapping,
    resolve_initial_worker_count,
)


def test_legacy_real_dataset_worker_mapping() -> None:
    assert resolve_initial_worker_count(Path("data/283.csv")) == 80
    assert resolve_initial_worker_count(Path("data/680.csv")) == 100
    assert resolve_initial_worker_count(Path("data/2338.csv")) == 140
    assert resolve_initial_worker_count(Path("data/3182.csv")) == 160
    assert resolve_initial_worker_count(Path("data/unknown.csv")) is None


def test_apply_mapping_updates_fixed_eval_worker_bounds() -> None:
    config = SimpleNamespace(n_w=100, n_w_min=60, n_w_max=100)
    result = apply_initial_worker_mapping(config, "data/2338.csv")
    assert result == 140
    assert config.n_w == 140
    assert config.n_w_min == 140
    assert config.n_w_max == 140


def test_explicit_worker_override_is_preserved() -> None:
    config = SimpleNamespace(n_w=100, n_w_min=60, n_w_max=100)
    result = apply_initial_worker_mapping(
        config,
        "data/283.csv",
        explicit_fields={"n_w"},
    )
    assert result == 80
    assert (config.n_w, config.n_w_min, config.n_w_max) == (100, 60, 100)
