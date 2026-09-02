from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch_geometric.data import HeteroData

from configs import Config, configs
from core.action_completion import MinWaitActionCompleter, build_action_completer
from core.action_masker import ActionMasker
from models.hb_gat_pn import (
    HBGATPN,
    StrictHomogeneousGraphSAGEEncoder,
)
from runtime.checkpoints import build_checkpoint_metadata, build_model_spec
from worker_feature_layout import resolve_worker_feature_layout


def _strict_observation() -> HeteroData:
    config = Config()
    layout = resolve_worker_feature_layout(config)
    observation = HeteroData()

    task_x = torch.zeros((2, 24), dtype=torch.float32)
    task_x[:, 0] = torch.tensor([1.0, 2.0])
    task_x[:, 1] = 1.0
    task_x[:, 5] = 1.0
    task_x[:, 16] = 1.0
    task_x[:, 18:24] = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    observation["task"].x = task_x

    worker_x = torch.zeros((2, layout.total_dim), dtype=torch.float32)
    worker_x[:, layout.efficiency_idx] = torch.tensor([0.5, 2.0])
    worker_x[:, layout.skill_start] = 1.0
    worker_x[:, layout.wait_idx] = 0.0
    worker_x[:, layout.lock_start] = 1.0
    worker_x[:, layout.fatigue_idx] = 1.0
    observation["worker"].x = worker_x
    observation["station"].x = torch.zeros((2, 15), dtype=torch.float32)
    observation["task", "precedes", "task"].edge_index = torch.empty(
        (2, 0), dtype=torch.long
    )
    return observation


def test_min_wait_ignores_efficiency_and_worker_queue_mask() -> None:
    config = Config(action_completion_mode="min_wait")
    completer = build_action_completer(config)
    assert isinstance(completer, MinWaitActionCompleter)

    observation = _strict_observation()
    layout = resolve_worker_feature_layout(config)
    observation["worker"].x[0, layout.efficiency_idx] = 0.1
    observation["worker"].x[1, layout.efficiency_idx] = 100.0
    worker_mask = torch.tensor([True, False], dtype=torch.bool)

    result = completer.complete(
        observation,
        task_id=0,
        station_mask=torch.tensor([False, True]),
        worker_mask=worker_mask,
    )

    assert result is not None
    assert result.station_id == 0
    assert result.team == (0,)
    candidates = completer.enumerate_team_candidates(
        observation,
        task_id=0,
        station_id=0,
        worker_mask=worker_mask,
    )
    assert candidates is not None
    assert candidates.teams == ((0,),)


def test_task_scope_filters_resource_and_baseline_task_features_for_actor_and_critic() -> None:
    config = Config(
        hidden_dim=8,
        task_feat_dim=24,
        worker_feat_dim=17,
        station_feat_dim=15,
        use_skill_hub=False,
        graph_encoder_mode="none",
        policy_action_scope="operation",
        policy_observation_scope="task",
    )
    config.critic_observation_scope = "match_policy"
    config.task_feature_scope = "intrinsic"
    observation = _strict_observation()
    altered = observation.clone()
    altered["task"].x[:, 18:24] = 10000.0
    altered["station"].x = altered["station"].x + 10000.0
    altered["worker"].x = altered["worker"].x - 10000.0

    model = HBGATPN(config).eval()
    with torch.inference_mode():
        encoded, context = model(observation)
        altered_encoded, altered_context = model(altered)
        logits = model.task_head(encoded["task"], context)
        altered_logits = model.task_head(altered_encoded["task"], altered_context)
        value = model.get_value(observation, actor_x_dict_encoded=encoded)
        altered_value = model.get_value(
            altered,
            actor_x_dict_encoded=altered_encoded,
        )

    assert set(encoded) == {"task"}
    torch.testing.assert_close(encoded["task"], altered_encoded["task"])
    torch.testing.assert_close(context, altered_context)
    torch.testing.assert_close(logits, altered_logits)
    torch.testing.assert_close(value, altered_value)


def test_strict_mask_keeps_ready_task_and_only_applies_structural_station_rules(
    monkeypatch,
) -> None:
    monkeypatch.setattr(configs, "task_mask_mode", "precedence_release_only", raising=False)
    monkeypatch.setattr(configs, "station_mask_mode", "structural_only", raising=False)
    monkeypatch.setattr(configs, "enable_shadow_mask_verification", False, raising=False)
    monkeypatch.setattr(configs, "enable_worker_queue_mask", True, raising=False)
    monkeypatch.setattr(configs, "enable_gpu_tensor_masking", True, raising=False)

    env = SimpleNamespace(
        num_tasks=1,
        num_stations=2,
        num_workers=2,
        current_time=0.0,
        mean_task_time=1.0,
        task_status=np.array([1], dtype=np.int64),
        task_material_ready=np.array([0.0], dtype=np.float64),
        task_static_feat=torch.tensor([[10.0, 0.0, 2.0]]),
        worker_skill_matrix=torch.zeros((2, 5)),
        worker_locks=np.array([2, 2], dtype=np.int64),
        worker_free_time=np.array([100.0, 100.0]),
        fixed_stations=np.array([1], dtype=np.int64),
        max_allowed_stations=np.array([1], dtype=np.int64),
        task_station_map={},
        constraint_engine=SimpleNamespace(minimum_station=lambda *_args: 0),
        station_available_slots=np.array([0, 0], dtype=np.int64),
        station_task_finish_times=[[1.0], [1.0]],
    )

    task_mask, station_mask, worker_mask = ActionMasker(env).get_masks()

    assert task_mask.tolist() == [False]
    assert station_mask.tolist() == [[True, False]]
    assert worker_mask.tolist() == [False, False]


class _CaptureConv(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_index: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        self.edge_index = edge_index.detach().clone()
        assert x.ndim == 2 and x.size(1) == self.hidden_dim
        return torch.zeros_like(x)


def test_strict_homogeneous_graphsage_merges_relations_without_type_embedding() -> None:
    config = Config(hidden_dim=4, num_gat_layers=1)
    encoder = StrictHomogeneousGraphSAGEEncoder(config)
    assert not hasattr(encoder, "type_embedding")

    capture = _CaptureConv(hidden_dim=4)
    encoder.layers[0] = capture
    x_dict = {
        "task": torch.ones((1, 4)),
        "station": torch.ones((1, 4)),
    }
    edge_index_dict = {
        ("task", "precedes", "task"): torch.tensor([[0], [0]]),
        ("task", "assigned_to", "station"): torch.tensor([[0], [0]]),
    }

    result = encoder(x_dict, edge_index_dict)

    assert set(result) == {"task", "station"}
    assert capture.edge_index is not None
    assert capture.edge_index.shape == (2, 2)
    assert sorted(map(tuple, capture.edge_index.t().tolist())) == [(0, 0), (0, 1)]


def test_strict_fields_are_recorded_in_checkpoint_metadata() -> None:
    config = Config()
    config.critic_observation_scope = "match_policy"
    config.task_feature_scope = "intrinsic"
    config.task_mask_mode = "precedence_release_only"
    config.station_mask_mode = "structural_only"
    config.action_completion_mode = "min_wait"
    config.graph_encoder_mode = "homogeneous_graphsage_strict"
    config.homogeneous_use_type_embedding = False
    config.homogeneous_shared_input_projection = False

    spec = build_model_spec(config)
    metadata = build_checkpoint_metadata(config)

    assert spec.action_completion_mode == "min_wait"
    assert spec.task_feature_scope == "intrinsic"
    assert spec.task_mask_mode == "precedence_release_only"
    assert spec.station_mask_mode == "structural_only"
    assert spec.graph_encoder_mode == "homogeneous_graphsage_strict"
    assert metadata["model_spec"]["reschedule_baseline_identity_conditioning"] is False
