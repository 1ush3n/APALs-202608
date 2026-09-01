from __future__ import annotations

import numpy as np
from pathlib import Path
import pytest
import torch
from torch_geometric.data import HeteroData

from configs import Config, configs
from environment import AirLineEnv_Graph
from models.hb_gat_pn import StationSelector, WorkerPointer
from models.worker_pointer_context import build_worker_pressure_context
from runtime.checkpoints import (
    apply_checkpoint_model_spec,
    build_checkpoint_metadata,
    build_model_spec,
)
from runtime.configuration import STRUCTURAL_FIELDS, validate_runtime_config
from tests.runtime_safety import temporary_config
from tests.test_joint_experiment_architecture import (
    _advance_to_ready_physical_task,
    _small_overrides,
)
from ppo_agent import PPOAgent
from models.hb_gat_pn import HBGATPN
from utils.reschedule import BaselineSchedule, BaselineTask
from utils.baseline_identity import (
    build_baseline_team_edge_index,
    build_station_baseline_match,
    build_worker_baseline_membership,
    offset_baseline_team_edge_index,
)
from training.v2_fast_exact_batch import CPUExactBatchBuilder
from training.v2_fast_exact_batch import GPUExactBatchBuilder
from training.memory import Memory
from training.rollout_service import APALRolloutService


RESCHEDULE_DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "r5_task_delay_v1"
    / "instances"
    / "validation"
    / "validation_0001.csv"
)


def _baseline() -> BaselineSchedule:
    return BaselineSchedule(
        tasks={
            0: BaselineTask(0, 2, (1, 4, 7), 0.0, 1.0, 1.0),
            1: BaselineTask(1, 0, (2,), 0.0, 1.0, 1.0),
        },
        makespan=1.0,
    )


def _v2_config(*, bic: bool) -> Config:
    config = Config()
    config.hidden_dim = 8
    config.team_selection_mode = "autoregressive_pressure_v2"
    config.policy_action_scope = "operation_station_worker"
    config.actor_context_mode = "attention"
    config.reschedule_baseline_identity_conditioning = bic
    return config


def _pressure_context() -> object:
    task_features = torch.zeros((1, 2, 18))
    task_features[..., 0] = 2.0
    task_features[..., 5] = 1.0
    worker_features = torch.zeros((1, 3, 17))
    worker_features[..., 1] = 1.0
    return build_worker_pressure_context(
        task_features=task_features,
        worker_features=worker_features,
        task_present=torch.ones((1, 2), dtype=torch.bool),
        task_action_invalid=torch.zeros((1, 2), dtype=torch.bool),
        worker_present=torch.ones((1, 3), dtype=torch.bool),
        worker_queue_invalid=torch.zeros((1, 3), dtype=torch.bool),
        temperature=1.0,
        supply_epsilon=1.0e-6,
    )


def test_station_baseline_match_flag_is_candidate_specific() -> None:
    baseline = _baseline()
    stations = torch.tensor([0, 1, 2, 3, 4])
    flags = build_station_baseline_match(
        [baseline.tasks[0].station_id],
        selected_tasks=torch.tensor([0]),
        candidate_station_ids=stations,
        enabled=True,
    )
    assert flags.shape == (1, 5, 1)
    torch.testing.assert_close(flags[0, :, 0], torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]))


def test_worker_membership_uses_sparse_task_worker_relation_and_offsets() -> None:
    edge_index = build_baseline_team_edge_index(_baseline(), num_tasks=2)
    flags = build_worker_baseline_membership(
        edge_index,
        selected_tasks=torch.tensor([0, 1]),
        num_workers=8,
        enabled=True,
    )
    assert flags.shape == (2, 8, 1)
    torch.testing.assert_close(
        flags[0, :, 0],
        torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
    )
    assert flags[1, 1, 0].item() == 0.0
    assert flags[1, 2, 0].item() == 1.0

    offset_edge_index = offset_baseline_team_edge_index(
        edge_index, task_offset=2, worker_offset=10
    )
    offset_flags = build_worker_baseline_membership(
        offset_edge_index,
        selected_tasks=torch.tensor([2]),
        num_workers=8,
        candidate_worker_offsets=torch.tensor([10]),
        enabled=True,
    )
    assert offset_flags[0, 1, 0].item() == 1.0


def test_no_baseline_or_disabled_bic_returns_zero_flags() -> None:
    baseline = _baseline()
    station_flags = build_station_baseline_match(
        [baseline.tasks[0].station_id],
        selected_tasks=[0],
        candidate_station_ids=[0, 1, 2],
        enabled=False,
    )
    worker_flags = build_worker_baseline_membership(
        None,
        selected_tasks=[0],
        num_workers=8,
        enabled=True,
    )
    assert torch.count_nonzero(station_flags).item() == 0
    assert torch.count_nonzero(worker_flags).item() == 0


def test_bic_is_structural_and_recorded_in_checkpoint_metadata() -> None:
    config = _v2_config(bic=True)
    spec = build_model_spec(config)
    metadata = build_checkpoint_metadata(config)
    assert spec.reschedule_baseline_identity_conditioning is True
    assert metadata["model_spec"]["reschedule_baseline_identity_conditioning"] is True
    assert metadata["config"]["reschedule_baseline_identity_conditioning"] is True
    assert "reschedule_baseline_identity_conditioning" in STRUCTURAL_FIELDS
    incompatible = _v2_config(bic=False)
    with pytest.raises(ValueError, match="reschedule_baseline_identity_conditioning"):
        apply_checkpoint_model_spec(incompatible, spec)


def test_bic_requires_reschedule_worker_pointer_v2_mode() -> None:
    config = _v2_config(bic=True)
    with pytest.raises(ValueError, match="enable_reschedule_mode"):
        validate_runtime_config(config)


def test_zero_initialized_station_adapter_preserves_logits() -> None:
    old_config = _v2_config(bic=False)
    bic_config = _v2_config(bic=True)
    torch.manual_seed(1234)
    old_head = StationSelector(old_config)
    torch.manual_seed(1234)
    bic_head = StationSelector(bic_config)
    task_emb = torch.randn((1, 8))
    station_embs = torch.randn((1, 5, 8))
    old_logits = old_head(task_emb, station_embs)
    new_logits = bic_head(
        task_emb,
        station_embs,
        station_baseline_match=torch.tensor([[[0.0], [0.0], [1.0], [0.0], [0.0]]]),
    )
    assert torch.count_nonzero(bic_head.baseline_station_proj.weight).item() == 0
    torch.testing.assert_close(new_logits, old_logits, atol=1.0e-4, rtol=0.0)


def test_zero_initialized_worker_adapter_preserves_logits() -> None:
    old_config = _v2_config(bic=False)
    bic_config = _v2_config(bic=True)
    torch.manual_seed(1234)
    old_head = WorkerPointer(old_config)
    torch.manual_seed(1234)
    bic_head = WorkerPointer(bic_config)
    common = {
        "task_emb": torch.randn((1, 8)),
        "station_emb": torch.randn((1, 8)),
        "global_context": torch.randn((1, 24)),
        "worker_embs": torch.randn((1, 3, 8)),
        "pressure_context": _pressure_context(),
        "demand": torch.tensor([2.0]),
        "mask": torch.zeros((1, 3), dtype=torch.bool),
    }
    old_state = old_head.initialize_v2_state(batch_size=1, device=torch.device("cpu"))
    bic_state = bic_head.initialize_v2_state(batch_size=1, device=torch.device("cpu"))
    old_logits = old_head.forward_choice_v2(**common, team_state=old_state)
    new_logits = bic_head.forward_choice_v2(
        **common,
        team_state=bic_state,
        worker_baseline_member=torch.zeros((1, 3, 1)),
    )
    assert torch.count_nonzero(bic_head.baseline_worker_proj.weight).item() == 0
    torch.testing.assert_close(new_logits, old_logits, atol=1.0e-4, rtol=0.0)


def test_fast_exact_auxiliary_relation_applies_task_worker_offsets() -> None:
    config = Config(task_feat_dim=18, worker_feat_dim=17, station_feat_dim=15)
    config.use_skill_hub = False

    class _Env:
        @staticmethod
        def rebuild_state_from_snapshot(snapshot: dict) -> HeteroData:
            data = HeteroData()
            data["task"].x = torch.zeros((2, 18))
            data["worker"].x = torch.zeros((int(snapshot["num_workers"]), 17))
            data["station"].x = torch.zeros((2, 15))
            return data

    snapshots = [
        {
            "num_workers": 3,
            "baseline_station": np.array([2, 0]),
            "baseline_team_edge_index": np.array([[0], [1]], dtype=np.int64),
        },
        {
            "num_workers": 2,
            "baseline_station": np.array([1, 0]),
            "baseline_team_edge_index": np.array([[1], [0]], dtype=np.int64),
        },
    ]
    result = CPUExactBatchBuilder(
        config=config, env=_Env(), device=torch.device("cpu")
    ).build(snapshots)
    torch.testing.assert_close(
        result.baseline_team_edge_index,
        torch.tensor([[0, 3], [1, 3]], dtype=torch.long),
    )
    assert result.baseline_station_slices[0].tolist() == [2, 0]
    assert result.baseline_station_slices[1].tolist() == [1, 0]


def test_bic_rollout_and_fast_exact_replay_inputs_have_identical_logprob() -> None:
    overrides = _small_overrides(
        enable_reschedule_mode=True,
        reschedule_baseline_identity_conditioning=True,
        reschedule_baseline_schedule_path=str(
            RESCHEDULE_DATA_PATH.parents[4]
            / "data"
            / "r5_task_delay_v1"
            / "baselines"
            / "validation"
            / "validation_0001_schedule.csv"
        ),
        team_selection_mode="autoregressive_pressure_v2_fast_exact",
        policy_action_scope="operation_station_worker",
        actor_context_mode="attention",
        worker_pointer_v2_replay_mode="behavior_group_exact_gpu_template_v2",
    )
    with temporary_config(configs, overrides):
        env = AirLineEnv_Graph(RESCHEDULE_DATA_PATH, seed=42)
        obs, masks = _advance_to_ready_physical_task(env)
        env._ensure_baseline_schedule()
        snapshot = env.get_state_snapshot()
        obs = env.rebuild_state_from_snapshot(snapshot)
        agent = PPOAgent(
            HBGATPN(configs),
            lr=1.0e-4,
            gamma=0.99,
            k_epochs=1,
            eps_clip=0.2,
            device=torch.device("cpu"),
            batch_size=1,
            total_timesteps=1,
            config=configs,
        )
        obs_result = agent.select_actions_batch(
            [obs],
            [masks[0]],
            [masks[1]],
            [masks[2]],
            deterministic=True,
            baseline_snapshots=[snapshot],
        )[0]
        behavior_obs = list(agent.last_v2_behavior_logprobs)
        fast_result = agent.select_actions_batch(
            [],
            [masks[0]],
            [masks[1]],
            [masks[2]],
            deterministic=True,
            snapshots=[snapshot],
            fast_exact_builder=GPUExactBatchBuilder(
                config=configs, env=env, device=torch.device("cpu")
            ),
        )[0]
        behavior_fast = list(agent.last_v2_behavior_logprobs)
        assert obs_result[0] == fast_result[0]
        assert obs_result[1] == fast_result[1]
        assert len(behavior_obs) == len(behavior_fast)
        for left, right in zip(behavior_obs, behavior_fast):
            assert left == right


def test_bic_diagnostics_use_selected_tokens_and_baseline_identity() -> None:
    memory = Memory()
    memory.states.append(
        {
            "baseline_station": np.array([2]),
            "baseline_team_edge_index": np.array([[0, 0], [1, 4]], dtype=np.int64),
        }
    )
    memory.actions.append((0, 2, [1, 3]))
    metrics = APALRolloutService._baseline_identity_metrics([memory])
    assert metrics["BaselineIdentity/SelectedStationPreservedRate"] == 1.0
    assert metrics["BaselineIdentity/SelectedWorkerMemberRate"] == 0.5
