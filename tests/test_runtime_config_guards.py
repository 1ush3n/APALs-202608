from __future__ import annotations

from typing import Any

import pytest

from configs import Config
from runtime.configuration import validate_runtime_config


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    (
        ("update_every_episodes", 2, "update_every_episodes=1"),
        ("n_m", 8, "n_m.*最大为 7"),
        ("sample_temperature", 0.8, "sample_temperature=1"),
    ),
)
def test_runtime_config_rejects_unsupported_training_values(
    field_name: str,
    invalid_value: Any,
    message: str,
) -> None:
    config = Config()
    setattr(config, field_name, invalid_value)

    with pytest.raises(ValueError, match=message):
        validate_runtime_config(config)
