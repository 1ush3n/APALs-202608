from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from configs import configs
from data_loader import load_data
from scripts.validate_initial_schedule import validate_schedule


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "283.csv"


@pytest.fixture(autouse=True)
def _restore_global_config_after_each_case() -> Iterator[None]:
    original = configs.to_flat_dict()
    try:
        yield
    finally:
        configs.update_from_dict(original)


def test_initial_schedule_uses_central_station_slot_examples(
    tmp_path: Path,
) -> None:
    task_df = load_data(DATA_PATH)["task_df"]
    task_ids = [
        int(task_id)
        for task_id in task_df.loc[task_df["duration"] > 0.0, "internal_id"].head(4)
    ]
    schedule_path = tmp_path / "overlapping.csv"
    pd.DataFrame(
        [
            {
                "TaskID": task_id,
                "StationID": 1,
                "Team": "[]",
                "Start": 0.0,
                "End": 10.0,
                "Duration": 10.0,
            }
            for task_id in task_ids
        ]
    ).to_csv(schedule_path, index=False)

    configs.n_m = 5
    configs.n_w = 80
    configs.max_slots_per_station = 3
    configs.enable_dynamic_events = False
    report = validate_schedule(
        data_path=DATA_PATH,
        schedule_path=schedule_path,
        config_obj=configs,
        task_id_mode="internal",
    )

    assert report["violations"]["station_slot_violation_count"] == 1
    example = report["examples"]["station_slot"][0]
    assert example["station_id"] == 0
    assert example["capacity"] == 3
    assert example["active_task_ids"] == sorted(task_ids)
