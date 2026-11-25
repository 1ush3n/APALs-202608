from __future__ import annotations

import math

import pytest
import torch
from torch_geometric.data import HeteroData

from configs import Config
from core.action_completion import (
    EarliestAvailabilityActionCompleter,
    EarliestFinishActionCompleter,
    build_action_completer,
)
from runtime.checkpoints import build_checkpoint_metadata, validate_checkpoint_training_spec
from runtime.configuration import validate_runtime_config
from worker_feature_layout import resolve_worker_feature_layout


def _make_observation(
    *,
    station_waits: tuple[float, ...] = (0.0,),
    workers: tuple[tuple[float, float, int, int], ...] = ((0.0, 1.0, 0, 0),),
    demand: int = 1,
    duration: float = 10.0,
) -> HeteroData:
    config = Config()
    layout = resolve_worker_feature_layout(config)
    task_x = torch.zeros((1, 18), dtype=torch.float32)
    task_x[0, 0] = duration
    task_x[0, 5] = 1.0
    task_x[0, 16] = float(demand)

    worker_x = torch.zeros((len(workers), layout.total_dim), dtype=torch.float32)
    for worker_id, (wait, efficiency, skill, lock) in enumerate(workers):
        worker_x[worker_id, layout.efficiency_idx] = efficiency
        worker_x[worker_id, layout.skill_start + skill] = 1.0
        worker_x[worker_id, layout.wait_idx] = math.log1p(wait)
        worker_x[worker_id, layout.lock_start + lock] = 1.0
        worker_x[worker_id, layout.fatigue_idx] = 1.0

    station_x = torch.zeros((len(station_waits), 15), dtype=torch.float32)
    station_x[:, 4] = torch.tensor(
        [math.log1p(wait) for wait in station_waits], dtype=torch.float32
    )

    observation = HeteroData()
    observation["task"].x = task_x
    observation["worker"].x = worker_x
    observation["station"].x = station_x
    return observation


def _complete(
    observation: HeteroData,
    *,
    station_mask: torch.Tensor | None = None,
    worker_mask: torch.Tensor | None = None,
    selected_station: int | None = None,
):
    return EarliestAvailabilityActionCompleter(Config()).complete(
        observation,
        task_id=0,
        station_mask=station_mask,
        worker_mask=worker_mask,
        selected_station=selected_station,
    )


def test_ea_chooses_station_with_shorter_wait() -> None:
    result = _complete(_make_observation(station_waits=(5.0, 1.0)))

    assert result is not None
    assert result.station_id == 1
    assert result.team == (0,)


def test_ea_breaks_equal_station_wait_by_station_id() -> None:
    result = _complete(_make_observation(station_waits=(2.0, 2.0)))

    assert result is not None
    assert result.station_id == 0


def test_ea_skips_earliest_station_without_a_legal_team() -> None:
    observation = _make_observation(
        station_waits=(0.0, 10.0),
        workers=((0.0, 1.0, 0, 2),),
    )

    result = _complete(observation)

    assert result is not None
    assert result.station_id == 1


def test_ea_worker_wait_has_priority_over_efficiency() -> None:
    observation = _make_observation(
        workers=((0.0, 0.5, 0, 0), (1.0, 100.0, 0, 0)),
    )

    result = _complete(observation)

    assert result is not None
    assert result.team == (0,)


def test_ea_worker_efficiency_breaks_equal_wait() -> None:
    observation = _make_observation(
        workers=((1.0, 0.5, 0, 0), (1.0, 100.0, 0, 0)),
    )

    result = _complete(observation)

    assert result is not None
    assert result.team == (1,)


def test_ea_worker_id_breaks_equal_wait_and_efficiency() -> None:
    observation = _make_observation(
        workers=((1.0, 2.0, 0, 0), (1.0, 2.0, 0, 0)),
    )

    result = _complete(observation)

    assert result is not None
    assert result.team == (0,)


def test_ea_filters_worker_mask_skill_and_station_lock() -> None:
    observation = _make_observation(
        workers=(
            (0.0, 100.0, 0, 0),
            (0.0, 100.0, 1, 0),
            (0.0, 100.0, 0, 2),
            (0.0, 1.0, 0, 0),
        ),
    )

    result = _complete(
        observation,
        worker_mask=torch.tensor([True, False, False, False]),
    )

    assert result is not None
    assert result.team == (3,)


def test_ea_selected_station_is_never_replaced() -> None:
    result = _complete(
        _make_observation(station_waits=(0.0, 10.0)),
        selected_station=1,
    )

    assert result is not None
    assert result.station_id == 1


def test_ea_returns_none_instead_of_replacing_an_illegal_selected_station() -> None:
    observation = _make_observation(
        station_waits=(0.0, 10.0),
        workers=((0.0, 1.0, 0, 2),),
    )

    assert _complete(observation, selected_station=0) is None


def test_ea_selects_exact_worker_top_k_in_sorted_order() -> None:
    observation = _make_observation(
        workers=(
            (2.0, 10.0, 0, 0),
            (1.0, 1.0, 0, 0),
            (1.0, 5.0, 0, 0),
            (0.0, 0.5, 0, 0),
        ),
        demand=3,
    )

    result = _complete(observation)

    assert result is not None
    assert result.team == (3, 2, 1)


def test_ea_returns_none_when_legal_worker_count_is_insufficient() -> None:
    observation = _make_observation(
        workers=((0.0, 1.0, 0, 0),),
        demand=2,
    )

    assert _complete(observation) is None


def test_ea_allows_waiting_resources_when_masks_allow_them() -> None:
    observation = _make_observation(station_waits=(4.0,), workers=((3.0, 1.0, 0, 0),))

    result = _complete(
        observation,
        station_mask=torch.tensor([False]),
        worker_mask=torch.tensor([False]),
    )

    assert result is not None
    assert result.station_id == 0
    assert result.team == (0,)


def test_ea_choice_is_independent_of_duration_fatigue_and_station_load() -> None:
    base = _make_observation(
        station_waits=(2.0, 1.0),
        workers=((0.0, 1.0, 0, 0), (1.0, 100.0, 0, 0)),
    )
    altered = base.clone()
    altered["task"].x[0, 0] = 10000.0
    altered["worker"].x[:, resolve_worker_feature_layout(Config()).fatigue_idx] = 0.01
    altered["station"].x[:, 0] = torch.tensor([10000.0, 0.0])

    assert _complete(base) == _complete(altered)


def test_ea_does_not_call_earliest_finish_team_score() -> None:
    completer = EarliestAvailabilityActionCompleter(Config())

    def fail_if_called(**_kwargs: object) -> tuple[float, float, int, tuple[int, ...]]:
        raise AssertionError("EA 不得调用 EFT team score")

    completer._team_score = fail_if_called  # type: ignore[method-assign]
    result = completer.complete(
        _make_observation(),
        task_id=0,
        station_mask=None,
        worker_mask=None,
    )

    assert result is not None


def test_ea_virtual_task_uses_empty_resource_action() -> None:
    observation = _make_observation()
    observation["task"].x[0, 0] = 0.0
    observation["task"].x[0, 5] = 0.0

    result = _complete(observation)

    assert result is not None
    assert result.station_id == -1
    assert result.team == ()


def test_action_completer_builder_defaults_to_eft_and_supports_ea() -> None:
    assert isinstance(build_action_completer(Config()), EarliestFinishActionCompleter)

    config = Config()
    config.policy_action_scope = "operation"
    config.action_completion_mode = "earliest_availability"
    assert isinstance(build_action_completer(config), EarliestAvailabilityActionCompleter)


def test_invalid_action_completion_mode_is_rejected() -> None:
    config = Config()
    config.action_completion_mode = "not_a_completion_mode"

    with pytest.raises(ValueError, match="action_completion_mode"):
        validate_runtime_config(config)


@pytest.mark.parametrize(
    "scope",
    ("operation_station_worker", "operation_station_gated_team", "operation_station_anchor_proposal_team"),
)
def test_ea_is_rejected_for_non_resource_completion_scope(scope: str) -> None:
    config = Config()
    config.policy_action_scope = scope
    config.action_completion_mode = "earliest_availability"

    with pytest.raises(ValueError, match="earliest_availability"):
        validate_runtime_config(config)


def test_old_checkpoint_without_completion_mode_means_eft() -> None:
    metadata = build_checkpoint_metadata(Config())
    metadata["config"].pop("action_completion_mode")

    validate_checkpoint_training_spec(Config(), metadata)


def test_resume_rejects_completion_mode_conflict() -> None:
    metadata = build_checkpoint_metadata(Config())
    config = Config()
    config.action_completion_mode = "earliest_availability"

    with pytest.raises(ValueError, match="action_completion_mode"):
        validate_checkpoint_training_spec(config, metadata)
