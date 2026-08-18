from __future__ import annotations

from pathlib import Path

import pytest
import torch

from configs import configs
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from runtime.checkpoints import build_model_spec
from tests.runtime_safety import temporary_config
from training.memory import Memory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "283.csv"


def _small_overrides(**extra: object) -> dict[str, object]:
    values: dict[str, object] = {
        "hidden_dim": 32,
        "num_gat_layers": 1,
        "num_heads": 2,
        "use_shared_trunk": True,
        "use_schedule_free": False,
        "use_ema": False,
        "enable_dynamic_events": False,
        "randomize_durations": False,
        "n_w": 80,
        "batch_size": 1,
        "accumulation_steps": 1,
        "k_epochs": 1,
        "policy_action_scope": "operation",
    }
    values.update(extra)
    return values


def _make_env(seed: int = 42) -> AirLineEnv_Graph:
    return AirLineEnv_Graph(DATA_PATH, seed=seed)


@pytest.mark.parametrize(
    ("scope", "expected_node_types"),
    (
        ("task", {"task"}),
        ("task_station", {"task", "station"}),
        ("full", {"task", "station", "worker", "skill"}),
    ),
)
def test_policy_observation_scope_controls_encoded_node_types(
    scope: str,
    expected_node_types: set[str],
) -> None:
    with temporary_config(
        configs,
        _small_overrides(policy_observation_scope=scope),
    ):
        env = _make_env()
        observation = env.reset(seed=42)
        model = HBGATPN(configs).eval()

        with torch.inference_mode():
            encoded, global_context = model(observation)

        assert set(encoded) == expected_node_types
        assert global_context.shape == (1, configs.hidden_dim * 3)


def test_task_scope_does_not_encode_station_or_worker_features() -> None:
    with temporary_config(configs, _small_overrides(policy_observation_scope="task")):
        env = _make_env()
        observation = env.reset(seed=42)
        altered = observation.clone()
        altered["station"].x = altered["station"].x + 100.0
        altered["worker"].x = altered["worker"].x - 100.0
        model = HBGATPN(configs).eval()

        with torch.inference_mode():
            first_encoded, first_context = model(observation)
            second_encoded, second_context = model(altered)

        torch.testing.assert_close(first_encoded["task"], second_encoded["task"])
        torch.testing.assert_close(first_context, second_context)


def test_task_station_scope_does_not_encode_worker_features() -> None:
    with temporary_config(
        configs,
        _small_overrides(
            policy_action_scope="operation_station",
            policy_observation_scope="task_station",
        ),
    ):
        env = _make_env()
        observation = env.reset(seed=42)
        altered = observation.clone()
        altered["worker"].x = altered["worker"].x + 100.0
        model = HBGATPN(configs).eval()

        with torch.inference_mode():
            first_encoded, first_context = model(observation)
            second_encoded, second_context = model(altered)

        torch.testing.assert_close(first_encoded["task"], second_encoded["task"])
        torch.testing.assert_close(first_encoded["station"], second_encoded["station"])
        torch.testing.assert_close(first_context, second_context)


@pytest.mark.parametrize(
    ("action_scope", "observation_scope"),
    (
        ("operation", "task"),
        ("operation_station", "task_station"),
    ),
)
def test_restricted_policy_scope_keeps_full_resource_completion_legal(
    action_scope: str,
    observation_scope: str,
) -> None:
    with temporary_config(
        configs,
        _small_overrides(
            policy_action_scope=action_scope,
            policy_observation_scope=observation_scope,
        ),
    ):
        env = _make_env()
        observation = env.reset(seed=42)
        task_mask, station_mask, worker_mask = env.get_masks()
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

        action, logprob, value, _station_mask, invalid = agent.select_action(
            observation,
            mask_task=task_mask,
            mask_station_matrix=station_mask,
            mask_worker=worker_mask,
            deterministic=True,
            temperature=0.0,
        )

        assert action is not None
        assert not invalid
        assert torch.isfinite(torch.tensor(logprob))
        assert torch.isfinite(torch.tensor(value))
        _next_observation, _reward, _done, info = env.step(action)
        assert not info.get("invalid_action", False)


@pytest.mark.parametrize(
    ("action_scope", "observation_scope"),
    (
        ("operation", "task"),
        ("operation_station", "task_station"),
    ),
)
def test_restricted_policy_scope_supports_finite_ppo_update(
    action_scope: str,
    observation_scope: str,
) -> None:
    with temporary_config(
        configs,
        _small_overrides(
            policy_action_scope=action_scope,
            policy_observation_scope=observation_scope,
        ),
    ):
        env = _make_env()
        observation = env.reset(seed=42)
        masks = env.get_masks()
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
        action, logprob, value, _station_mask, invalid = agent.select_action(
            observation,
            mask_task=masks[0],
            mask_station_matrix=masks[1],
            mask_worker=masks[2],
            deterministic=True,
            temperature=0.0,
        )
        assert action is not None and not invalid

        memory = Memory()
        memory.states.append(env.get_state_snapshot())
        memory.actions.append(action)
        memory.logprobs.append(logprob)
        memory.values.append(value)
        memory.masks.append(masks)
        _next_observation, reward, done, info = env.step(action)
        assert not info.get("invalid_action", False)
        memory.rewards.append(float(reward))
        memory.is_terminals.append(bool(done))

        metrics = agent.update(memory, env, current_ep=1)

        assert torch.isfinite(torch.tensor(metrics["Loss/Total"]))


def test_checkpoint_metadata_records_policy_observation_scope() -> None:
    with temporary_config(configs, _small_overrides(policy_observation_scope="task")):
        spec = build_model_spec(configs)
        assert spec.policy_observation_scope == "task"
