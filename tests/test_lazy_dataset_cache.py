from __future__ import annotations

import shutil
from pathlib import Path

from configs import configs
from environment import AirLineEnv_Graph
from tests.runtime_safety import temporary_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dataset_pool_is_lazy_and_lru_bounded(tmp_path: Path) -> None:
    for idx in range(3):
        shutil.copy2(PROJECT_ROOT / "data" / "283.csv", tmp_path / f"variant_{idx}.csv")

    with temporary_config(configs, {"dataset_context_cache_size": 2, "n_w": 80}):
        env = AirLineEnv_Graph(tmp_path, seed=42)
        assert len(env.dataset_pool) == 3
        assert "base_data" in env.dataset_pool[0]
        assert "raw_data" not in env.dataset_pool[1]
        assert "raw_data" not in env.dataset_pool[2]

        env.switch_dataset(1)
        env.switch_dataset(2)

        loaded = [idx for idx, ctx in enumerate(env.dataset_pool) if "base_data" in ctx]
        assert loaded == [1, 2]
