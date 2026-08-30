from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch, HeteroData

from configs import Config
from models.hb_gat_pn import HBGATPN, WorkerPointer
from models.worker_pointer_context import (
    PHYSICAL_PREDECESSOR_EDGE,
    build_worker_pressure_context,
)
from ppo_agent import PPOAgent
from runtime.hydra_config import initialize_hydra_runtime
from tests.runtime_safety import temporary_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _pressure_inputs(task_count: int = 4) -> dict[str, torch.Tensor]:
    task_features = torch.zeros((1, task_count, 18), dtype=torch.float32)
    task_features[0, 0, 0] = 2.0
    task_features[0, 0, 2] = 1.0
    task_features[0, 0, 5] = 1.0
    task_features[0, 0, 16] = 2.0
    task_features[0, 1, 0] = 3.0
    task_features[0, 1, 1] = 1.0
    task_features[0, 1, 5] = 1.0
    task_features[0, 1, 16] = 2.0
    worker_features = torch.zeros((1, 3, 17), dtype=torch.float32)
    worker_features[0, 0, 1] = 1.0
    worker_features[0, 1, 1] = 1.0
    worker_features[0, 2, 1] = 1.0
    return {
        "task_features": task_features,
        "worker_features": worker_features,
        "task_present": torch.ones((1, task_count), dtype=torch.bool),
        "task_action_invalid": torch.tensor(
            [[False, True] + [True] * (task_count - 2)], dtype=torch.bool
        ),
        "worker_present": torch.ones((1, 3), dtype=torch.bool),
        "worker_queue_invalid": torch.zeros((1, 3), dtype=torch.bool),
    }


def _pointer_config(**overrides: object) -> Config:
    config = Config()
    config.hidden_dim = 8
    config.team_selection_mode = "autoregressive_pressure_v2"
    config.policy_action_scope = "operation_station_worker"
    config.actor_context_mode = "attention"
    config.seed = 42
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def _build_context(config: Config) -> object:
    inputs = _pressure_inputs()
    return build_worker_pressure_context(
        **inputs,
        temperature=1.0,
        supply_epsilon=1.0e-6,
        physical_predecessor_edges=[torch.tensor([[0], [1]], dtype=torch.long)]
        if config.worker_pointer_v2_next_frontier_pressure
        else None,
    )


def _forward_worker(config: Config) -> tuple[WorkerPointer, torch.Tensor]:
    inputs = _pressure_inputs()
    head = WorkerPointer(config)
    context = _build_context(config)
    state = head.initialize_v2_state(batch_size=1, device=torch.device("cpu"))
    logits = head.forward_choice_v2(
        task_emb=torch.randn((1, 8)),
        station_emb=torch.randn((1, 8)),
        global_context=torch.randn((1, 24)),
        worker_embs=torch.randn((1, 3, 8)),
        pressure_context=context,
        team_state=state,
        demand=torch.tensor([2.0]),
        mask=torch.tensor([[False, False, True]]),
        candidate_skills=inputs["worker_features"][..., 1:6],
        task_required_skills=torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]]),
    )
    return head, logits


@pytest.mark.parametrize(
    ("variant", "expected"),
    (
        ("v0", (False, False, False, False, "off")),
        ("v1", (True, False, False, False, "off")),
        ("v2", (True, True, False, False, "off")),
        ("v3", (True, True, True, False, "off")),
        ("b0", (False, False, False, False, "diagnostic")),
        ("b1", (False, False, False, False, "factorized")),
        ("c1", (False, False, False, True, "off")),
        ("full_smoke", (True, True, True, True, "factorized")),
    ),
)
def test_all_worker_pointer_v2_variants_load_without_final_config(
    variant: str, expected: tuple[bool, bool, bool, bool, str]
) -> None:
    config = Config()
    initialize_hydra_runtime(
        [f"experiment=initial_worker_pointer_v2_{variant}"],
        target=config,
        project_root=PROJECT_ROOT,
        system_name="Linux",
        create_run_context=False,
    )

    assert (
        config.worker_pointer_v2_explicit_team_state,
        config.worker_pointer_v2_marginal_scarcity,
        config.worker_pointer_v2_interaction_residual,
        config.worker_pointer_v2_next_frontier_pressure,
        config.conditional_head_baseline_mode,
    ) == expected
    assert not (PROJECT_ROOT / "conf" / "experiment" / "initial_worker_pointer_v2_final.yaml").exists()


def test_nested_step_zero_invariance_and_full_smoke_forward_are_exact() -> None:
    torch.manual_seed(1234)
    _, v0_logits = _forward_worker(_pointer_config())
    for overrides in (
        {"worker_pointer_v2_explicit_team_state": True},
        {
            "worker_pointer_v2_explicit_team_state": True,
            "worker_pointer_v2_marginal_scarcity": True,
        },
        {
            "worker_pointer_v2_explicit_team_state": True,
            "worker_pointer_v2_marginal_scarcity": True,
            "worker_pointer_v2_interaction_residual": True,
        },
        {"worker_pointer_v2_next_frontier_pressure": True},
        {
            "worker_pointer_v2_explicit_team_state": True,
            "worker_pointer_v2_marginal_scarcity": True,
            "worker_pointer_v2_interaction_residual": True,
            "worker_pointer_v2_next_frontier_pressure": True,
        },
    ):
        torch.manual_seed(1234)
        _, logits = _forward_worker(_pointer_config(**overrides))
        torch.testing.assert_close(logits, v0_logits, atol=1.0e-6, rtol=0.0)


@pytest.mark.parametrize("mode", ("diagnostic", "factorized"))
def test_conditional_modes_have_finite_values_and_isolated_context_contract(
    mode: str,
) -> None:
    config = _pointer_config(conditional_head_baseline_mode=mode)
    model = HBGATPN(config)
    values = model.compute_conditional_values(
        critic_context=torch.randn((2, 24)),
        critic_task_emb=torch.randn((2, 8)),
        critic_station_emb=torch.randn((2, 8)),
        virtual_station=torch.tensor([False, True]),
    )

    assert set(values) == {"task", "station", "worker"}
    assert all(value.shape == (2, 1) for value in values.values())
    assert all(torch.isfinite(value).all() for value in values.values())


def test_full_upgrade_actor_and_conditional_smoke_gradients_are_finite() -> None:
    config = _pointer_config(
        worker_pointer_v2_explicit_team_state=True,
        worker_pointer_v2_marginal_scarcity=True,
        worker_pointer_v2_interaction_residual=True,
        worker_pointer_v2_next_frontier_pressure=True,
        conditional_head_baseline_mode="factorized",
    )
    head, logits = _forward_worker(config)
    model = HBGATPN(config)
    values = model.compute_conditional_values(
        critic_context=torch.randn((1, 24)),
        critic_task_emb=torch.randn((1, 8)),
        critic_station_emb=torch.randn((1, 8)),
    )
    loss = logits[:, :2].sum() + sum(value.sum() for value in values.values())
    loss.backward()

    gradients = [parameter.grad for parameter in head.parameters() if parameter.grad is not None]
    gradients.extend(
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert torch.isfinite(logits).all()


def test_c1_large_sparse_graph_never_materializes_dense_task_adjacency() -> None:
    context = build_worker_pressure_context(
        **_pressure_inputs(task_count=3182),
        temperature=1.0,
        supply_epsilon=1.0e-6,
        physical_predecessor_edges=[
            torch.stack(
                [
                    torch.arange(3181, dtype=torch.long),
                    torch.arange(1, 3182, dtype=torch.long),
                ]
            )
        ],
    )

    assert context.next_frontier_mask is not None
    assert context.pressure_next_frontier is not None
    assert context.next_frontier_mask.shape == (1, 3182)
    assert context.pressure_next_frontier.shape == (1, 5)
    assert all(
        tensor.shape != (3182, 3182)
        for tensor in (
            context.next_frontier_mask,
            context.pressure_next_frontier,
            context.unfinished_physical_predecessor_count,
            context.remaining_physical_predecessor_count,
        )
    )


def test_c1_batched_sparse_edges_round_trip_to_local_graphs() -> None:
    graphs: list[HeteroData] = []
    for _ in range(2):
        graph = HeteroData()
        graph["task"].x = torch.zeros((3, 18))
        graph[PHYSICAL_PREDECESSOR_EDGE].edge_index = torch.tensor(
            [[0, 1], [1, 2]], dtype=torch.long
        )
        graphs.append(graph)
    batch = Batch.from_data_list(graphs)

    local_edges = PPOAgent._physical_predecessor_edge_list(batch, batch_size=2)

    assert [edge.tolist() for edge in local_edges] == [
        [[0, 1], [1, 2]],
        [[0, 1], [1, 2]],
    ]
    assert all(edge.numel() < 3182 * 3182 for edge in local_edges)


def test_factorized_sample_weight_and_value_loss_contract_remains_finite() -> None:
    returns = torch.tensor([1.0, 3.0])
    old_values = torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    active = torch.tensor([[True, True, True], [True, False, False]])
    advantages = PPOAgent.compute_factorized_component_advantages(
        returns=returns,
        old_values=old_values,
        active_mask=active,
    )
    loss, ratios, _ = PPOAgent.compute_factorized_clipped_surrogate(
        current_logprobs=torch.zeros((2, 3), requires_grad=True),
        old_logprobs=torch.zeros((2, 3)),
        advantages=advantages,
        active_mask=active,
        clip_range=0.2,
    )
    value_loss = PPOAgent.compute_factorized_value_loss(
        current_values=old_values,
        old_values=old_values,
        targets=returns,
        active_mask=active,
        clip_range=0.2,
    )

    assert torch.allclose(ratios, torch.ones_like(ratios))
    assert torch.isfinite(loss)
    assert torch.isfinite(value_loss)


@pytest.mark.parametrize(
    ("variant", "conditional_mode"),
    (
        ("v0", "off"),
        ("v1", "off"),
        ("v2", "off"),
        ("v3", "off"),
        ("b0", "diagnostic"),
        ("b1", "factorized"),
        ("c1", "off"),
        ("full_smoke", "factorized"),
    ),
)
def test_short_local_rollout_and_ppo_update_smoke(
    variant: str, conditional_mode: str
) -> None:
    from configs import configs
    from environment import AirLineEnv_Graph
    from tests.test_joint_experiment_architecture import (
        DATA_PATH,
        _small_overrides,
    )
    from tests.test_worker_pointer_v2_behavior_replay import _rollout_single_step

    flags = {
        "worker_pointer_v2_explicit_team_state": variant in {"v1", "v2", "v3", "full_smoke"},
        "worker_pointer_v2_marginal_scarcity": variant in {"v2", "v3", "full_smoke"},
        "worker_pointer_v2_interaction_residual": variant in {"v3", "full_smoke"},
        "worker_pointer_v2_next_frontier_pressure": variant in {"c1", "full_smoke"},
        "conditional_head_baseline_mode": conditional_mode,
    }
    overrides = _small_overrides(
        team_selection_mode="autoregressive_pressure_v2",
        policy_action_scope="operation_station_worker",
        actor_context_mode="attention",
        batch_size=1,
        worker_pointer_v2_behavior_replay=True,
        worker_pointer_v2_logical_batch_cap=1,
        **flags,
    )
    with temporary_config(configs, overrides):
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
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
        memory, b_task, b_station, b_team, old_logprobs, rewards, advantages = (
            _rollout_single_step(agent, env)
        )
        if conditional_mode == "factorized":
            component_logprobs = agent.last_v2_behavior_logprobs[0]
            conditional_values = agent.last_v2_behavior_values[0]
            assert component_logprobs is not None
            assert conditional_values is not None
            memory.old_task_logprob.append(float(component_logprobs[0]))
            memory.old_station_logprob.append(float(component_logprobs[1]))
            memory.old_team_logprob.append(float(component_logprobs[2]))
            memory.old_V_task.append(float(conditional_values[0]))
            memory.old_V_station.append(float(conditional_values[1]))
            memory.old_V_worker.append(float(conditional_values[2]))
        metrics = agent._run_v2_behavior_replay_update(
            memory,
            env,
            current_ep=1,
            advantages=advantages,
            rewards=rewards,
            old_logprobs=old_logprobs,
            b_task=b_task,
            b_station=b_station,
            b_team=b_team,
            action_scope="operation_station_worker",
        )
        if variant in {"b0", "c1", "full_smoke"}:
            diagnostics = agent.worker_pointer_v2_diagnostics.finalize(
                require_coverage=False
            )
            assert diagnostics
        assert metrics["PPO/UpdateSteps"] == 1.0
        assert metrics["PPO/GradientsFinite"] == 1.0
        assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
