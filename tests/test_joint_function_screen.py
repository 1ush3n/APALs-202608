from __future__ import annotations

import json
from pathlib import Path

from configs import Config
from runtime.hydra_config import initialize_hydra_runtime
from scripts.run_joint_variant_function_screen import (
    PROJECT_ROOT,
    VARIANTS,
    _parameter_counts,
    main,
)


def test_function_screen_config_uses_three_updates_and_real_680() -> None:
    config = Config()
    initialize_hydra_runtime(
        [
            "experiment=joint_variant_function_screen",
        ],
        target=config,
        project_root=PROJECT_ROOT,
        create_run_context=False,
    )

    assert config.max_episodes == 3
    assert config.update_every_episodes == 1
    assert config.eval_freq == 3
    assert config.eval_temperature == 0.0
    assert Path(config.train_data_path_or_dir).name == "syn_403_77.csv"
    assert Path(config.data_file_path).name == "680.csv"
    assert config.use_lightning
    assert config.lightning_precision == "16-mixed"


def test_function_screen_plan_contains_all_nine_variants(tmp_path: Path) -> None:
    exit_code = main(["mode=plan", f"output_dir={tmp_path}"])
    rows = json.loads((tmp_path / "screen_plan.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert [row["variant"] for row in rows] == list(VARIANTS)
    assert all("train_lightning.py" in row["command"] for row in rows)
    assert all("seed=42" in row["command"] for row in rows)


def test_reduced_action_scopes_exclude_unused_heads_from_trainable_parameters() -> None:
    _full_total, full_trainable = _parameter_counts(
        VARIANTS["full_joint"],
    )
    _station_total, station_trainable = _parameter_counts(
        VARIANTS["operation_station"],
    )
    _operation_total, operation_trainable = _parameter_counts(
        VARIANTS["operation_only"],
    )

    assert operation_trainable < station_trainable < full_trainable
