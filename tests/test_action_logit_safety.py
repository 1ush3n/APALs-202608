from pathlib import Path

import pytest
import torch
from torch.distributions import Categorical

from configs import configs
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from tests.runtime_safety import temporary_config


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "283.csv"

from ppo_agent import _finalize_action_logits


def test_finalize_action_logits_excludes_invalid_candidates() -> None:
    logits = torch.tensor([[2.0, 100.0, 1.0]], dtype=torch.float16)
    invalid_mask = torch.tensor([[False, True, False]])

    finalized, usable = _finalize_action_logits(
        logits,
        invalid_mask,
        decision="task",
    )

    assert finalized.dtype == torch.float32
    assert usable.tolist() == [True]
    assert finalized[0, 0].item() == 2.0
    assert finalized[0, 2].item() == 1.0
    assert torch.isfinite(finalized[0, 1]) and finalized[0, 1] < -1000.0
    assert Categorical(logits=finalized).probs[0, 1].item() == 0.0
    assert int(finalized.argmax(dim=-1).item()) == 0


def test_finalize_action_logits_marks_all_invalid_row_unusable() -> None:
    finalized, usable = _finalize_action_logits(
        torch.tensor([[3.0, 4.0]]),
        torch.tensor([[True, True]]),
        decision="station",
    )

    assert usable.tolist() == [False]
    assert torch.isfinite(finalized).all() and (finalized < -1000.0).all()


@pytest.mark.parametrize(
    "bad_value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_finalize_action_logits_rejects_nonfinite_legal_candidate(
    bad_value: float,
) -> None:
    with pytest.raises(FloatingPointError, match="worker"):
        _finalize_action_logits(
            torch.tensor([[0.0, bad_value]]),
            torch.tensor([[False, False]]),
            decision="worker",
        )


def test_finalize_action_logits_ignores_nonfinite_invalid_candidate() -> None:
    finalized, usable = _finalize_action_logits(
        torch.tensor([[0.0, float("nan")]]),
        torch.tensor([[False, True]]),
        decision="worker",
    )

    assert usable.tolist() == [True]
    assert finalized[0, 0] == 0.0 and torch.isfinite(finalized[0, 1]) and finalized[0, 1] < -1000.0


def test_finalize_action_logits_rejects_implicit_mask_broadcast() -> None:
    with pytest.raises(ValueError, match="shape"):
        _finalize_action_logits(
            torch.zeros((1, 3)),
            torch.zeros(3, dtype=torch.bool),
            decision="task",
        )


@pytest.fixture
def main_agent_state():
    overrides = {
        "hidden_dim": 32,
        "num_gat_layers": 1,
        "num_heads": 2,
        "use_shared_trunk": True,
        "use_schedule_free": False,
        "use_ema": False,
        "enable_dynamic_events": False,
        "randomize_durations": False,
        "n_w": 80,
        "batch_size": 2,
        "k_epochs": 1,
    }
    with temporary_config(configs, overrides):
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
        state = env.reset(seed=42)
        agent = PPOAgent(
            HBGATPN(configs),
            lr=1.0e-4,
            gamma=0.99,
            k_epochs=1,
            eps_clip=0.2,
            device=torch.device("cpu"),
            batch_size=2,
            total_timesteps=1,
            config=configs,
        )
        yield agent, env, state, env.get_masks()


@pytest.mark.parametrize("deterministic", [False, True])
def test_select_action_never_forces_task_zero_when_all_tasks_invalid(
    main_agent_state,
    deterministic: bool,
) -> None:
    agent, _env, state, (_task_mask, station_mask, worker_mask) = main_agent_state
    result = agent.select_action(
        state,
        mask_task=torch.ones(state["task"].num_nodes, dtype=torch.bool),
        mask_station_matrix=station_mask,
        mask_worker=worker_mask,
        deterministic=deterministic,
        compute_value=False,
    )

    assert result[0] is None
    assert result[-1] is True


def test_select_actions_batch_never_forces_task_zero_when_all_tasks_invalid(
    main_agent_state,
) -> None:
    agent, _env, state, (_task_mask, station_mask, worker_mask) = main_agent_state
    result = agent.select_actions_batch(
        obs_list=[state],
        mask_task_list=[torch.ones(state["task"].num_nodes, dtype=torch.bool)],
        mask_station_matrix_list=[station_mask],
        mask_worker_list=[worker_mask],
        deterministic=True,
    )[0]

    assert result[0] is None
    assert result[-1] is True

def _advance_to_ready_physical_task(env, state):
    for _ in range(env.num_tasks):
        masks = env.get_masks()
        ready = torch.nonzero(~masks[0], as_tuple=False).reshape(-1).tolist()
        physical = [
            int(task_id)
            for task_id in ready
            if bool(env.constraint_engine.physical_mask[int(task_id)])
        ]
        if physical:
            return state, masks, min(physical)
        assert ready
        state, _reward, done, _info = env.step((min(ready), -1, []))
        assert not done
    raise AssertionError("未找到可调度物理工序")


def test_select_action_returns_no_action_when_selected_task_has_no_station(
    main_agent_state,
) -> None:
    agent, env, state, _masks = main_agent_state
    state, (task_mask, station_mask, worker_mask), task_id = (
        _advance_to_ready_physical_task(env, state)
    )
    forced_task_mask = torch.ones_like(task_mask)
    forced_task_mask[task_id] = False
    forced_station_mask = station_mask.clone()
    forced_station_mask[task_id] = True

    result = agent.select_action(
        state,
        mask_task=forced_task_mask,
        mask_station_matrix=forced_station_mask,
        mask_worker=worker_mask,
        deterministic=True,
        is_eval=True,
        compute_value=False,
    )

    assert result[0] is None
    assert result[-1] is True

def test_finalize_action_logits_keeps_amp_scaled_gradients_finite() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    finalized, usable = _finalize_action_logits(
        logits,
        torch.tensor([[False, True, False]]),
        decision="worker",
    )
    assert usable.tolist() == [True]

    dist = Categorical(logits=finalized)
    scaled_loss = -(
        dist.log_prob(torch.tensor([0])) + 0.1 * dist.entropy()
    ).mean() * 65536.0
    scaled_loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
