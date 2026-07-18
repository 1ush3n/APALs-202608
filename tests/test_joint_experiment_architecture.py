from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from configs import Config, configs
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN, WorkerPointer
from ppo_agent import PPOAgent
from runtime.checkpoints import FORMAT_VERSION, build_model_spec
from runtime.configuration import validate_runtime_config
from tests.runtime_safety import temporary_config
from worker_feature_layout import resolve_worker_feature_layout


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "283.csv"


def _small_overrides(**extra):
    values = {
        "hidden_dim": 32,
        "num_gat_layers": 1,
        "num_heads": 2,
        "use_shared_trunk": True,
        "use_schedule_free": False,
        "use_ema": False,
        "enable_dynamic_events": False,
        "randomize_durations": False,
        "n_w": 80,
        "batch_size": 4,
        "accumulation_steps": 1,
        "k_epochs": 1,
    }
    values.update(extra)
    return values


def test_new_experiment_modes_are_validated_and_legacy_switches_are_rejected() -> None:
    cfg = Config()
    cfg.policy_action_scope = "operation_station"
    cfg.workforce_binding_mode = "preallocated"
    cfg.workforce_preallocation_ratio = 1.0
    cfg.team_selection_mode = "static_topq"
    cfg.graph_encoder_mode = "homogeneous_graphsage"
    cfg.actor_context_mode = "local_only"
    validate_runtime_config(cfg)

    cfg.ablation_no_gat = True
    with pytest.raises(ValueError, match="graph_encoder_mode=none"):
        validate_runtime_config(cfg)


@pytest.mark.parametrize(
    ("graph_mode", "context_mode", "context_width"),
    (
        ("hetero_gat", "attention", 3),
        ("homogeneous_graphsage", "mean_max", 6),
        ("none", "local_only", 3),
    ),
)
def test_graph_and_actor_context_modes_preserve_node_shapes(
    graph_mode: str,
    context_mode: str,
    context_width: int,
) -> None:
    overrides = _small_overrides(
        graph_encoder_mode=graph_mode,
        actor_context_mode=context_mode,
    )
    with temporary_config(configs, overrides):
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
        obs = env.reset(seed=42)
        model = HBGATPN(configs)
        encoded, context = model(obs)
        assert encoded["task"].shape == (env.num_tasks, configs.hidden_dim)
        assert encoded["worker"].shape == (env.num_workers, configs.hidden_dim)
        assert encoded["station"].shape == (env.num_stations, configs.hidden_dim)
        assert context.shape == (1, configs.hidden_dim * context_width)
        value = model.get_value(obs, actor_x_dict_encoded=encoded)
        assert value.shape == (1, 1)
        assert torch.isfinite(value).all()


@pytest.mark.parametrize(
    "action_scope",
    ("operation", "operation_station", "operation_station_worker"),
)
def test_each_action_scope_emits_a_complete_legal_environment_action(
    action_scope: str,
) -> None:
    overrides = _small_overrides(policy_action_scope=action_scope)
    with temporary_config(configs, overrides):
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
        obs = env.reset(seed=42)
        masks = env.get_masks()
        model = HBGATPN(configs)
        agent = PPOAgent(
            model,
            lr=1e-4,
            gamma=0.99,
            k_epochs=1,
            eps_clip=0.2,
            device=torch.device("cpu"),
            batch_size=4,
            total_timesteps=1,
            config=configs,
        )
        action, logprob, _value, _station_mask, invalid = agent.select_action(
            obs,
            mask_task=masks[0],
            mask_station_matrix=masks[1],
            mask_worker=masks[2],
            deterministic=False,
            temperature=1.0,
        )
        assert action is not None
        assert not invalid
        assert torch.isfinite(torch.tensor(logprob))
        _next_obs, _reward, _done, info = env.step(action)
        assert not info.get("invalid_action", False)


def test_static_topq_worker_scores_do_not_depend_on_selected_team() -> None:
    cfg = Config()
    cfg.hidden_dim = 16
    cfg.team_selection_mode = "static_topq"
    with temporary_config(configs, {"hidden_dim": 16, "team_selection_mode": "static_topq"}):
        head = WorkerPointer(cfg)
        task = torch.randn(1, 16)
        workers = torch.randn(1, 8, 16)
        first = head.forward_choice(task, workers, current_team_emb=torch.randn(1, 16))
        second = head.forward_choice(task, workers, current_team_emb=torch.randn(1, 16))
        torch.testing.assert_close(first, second)


def test_full_preallocation_binds_every_worker_and_updates_observation() -> None:
    overrides = _small_overrides(
        workforce_binding_mode="preallocated",
        workforce_preallocation_ratio=1.0,
    )
    with temporary_config(configs, overrides):
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
        obs = env.reset(seed=42)
        diagnostics = env.workforce_preallocation_diagnostics
        assert diagnostics["assigned"] == env.num_workers
        assert diagnostics["mobile_workers"] == 0
        assert int((env.worker_locks > 0).sum()) == env.num_workers
        layout = resolve_worker_feature_layout(configs)
        observed_locks = torch.argmax(obs["worker"].x[:, layout.lock_slice], dim=1)
        assert bool((observed_locks > 0).all())


def test_checkpoint_model_spec_records_all_experiment_semantics() -> None:
    cfg = Config()
    cfg.policy_action_scope = "operation"
    cfg.team_selection_mode = "static_topq"
    cfg.graph_encoder_mode = "none"
    cfg.actor_context_mode = "local_only"
    spec = build_model_spec(cfg)
    assert FORMAT_VERSION == 2
    assert spec.policy_action_scope == "operation"
    assert spec.team_selection_mode == "static_topq"
    assert spec.graph_encoder_mode == "none"
    assert spec.actor_context_mode == "local_only"
