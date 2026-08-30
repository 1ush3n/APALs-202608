from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from configs import Config
from models.hb_gat_pn import WorkerPointer
from runtime.checkpoints import apply_checkpoint_model_spec, build_model_spec
from tests.runtime_safety import temporary_config


def _pressure_inputs() -> dict[str, torch.Tensor]:
    # task shape: [B=1, T=4, F=18]
    task = torch.zeros((1, 4, 18), dtype=torch.float32)
    task[0, 0, 0] = 2.0
    task[0, 0, 2] = 1.0  # ready
    task[0, 0, 5] = 1.0  # skill 0
    task[0, 0, 16] = 2.0
    task[0, 1, 0] = 3.0
    task[0, 1, 1] = 1.0  # not ready
    task[0, 1, 6] = 1.0  # skill 1
    task[0, 1, 16] = 1.0
    task[0, 2, 0] = 5.0
    task[0, 2, 3] = 1.0  # scheduled
    task[0, 2, 5] = 1.0
    task[0, 2, 16] = 4.0
    # task 3 is padding/virtual and stays zero.

    # worker shape: [B=1, N=3, F=17]
    worker = torch.zeros((1, 3, 17), dtype=torch.float32)
    worker[0, 0, 1] = 1.0  # skill 0
    worker[0, 1, 1:3] = 1.0  # skills 0 and 1
    worker[0, 1, 6] = torch.log1p(torch.tensor(1.0))
    # worker 2 is padding.
    return {
        "task_features": task,
        "worker_features": worker,
        "task_present": torch.tensor([[True, True, True, False]]),
        "task_action_invalid": torch.tensor([[False, True, True, True]]),
        "worker_present": torch.tensor([[True, True, False]]),
        "worker_queue_invalid": torch.tensor([[False, False, True]]),
    }


def test_v2_benchmark_summary_reports_ordered_quantiles() -> None:
    from scripts.benchmark_worker_pointer_v2_decode import (
        DECODE_VARIANTS,
        summarize_samples,
    )

    summary = summarize_samples([3.0, 1.0, 2.0, 4.0])

    assert summary == {"p10": 1.3, "p50": 2.5, "p90": 3.7, "mean": 2.5}
    assert DECODE_VARIANTS == (
        "decode_uncached_with_legacy_mean",
        "decode_uncached_without_legacy_mean",
        "decode_cached_with_legacy_mean",
        "decode_cached_without_legacy_mean",
    )


def test_v2_benchmark_script_runs_from_project_root() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-B", "scripts/benchmark_worker_pointer_v2_decode.py", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--manifest" in completed.stdout


def test_worker_pressure_context_uses_normalized_work_and_physical_wait_discount() -> None:
    from models.worker_pointer_context import build_worker_pressure_context

    context = build_worker_pressure_context(
        **_pressure_inputs(),
        temperature=1.0,
        supply_epsilon=1.0e-6,
    )

    expected_all_demand = torch.tensor([[4.0, 3.0, 0.0, 0.0, 0.0]])
    expected_near_demand = torch.tensor([[4.0, 0.0, 0.0, 0.0, 0.0]])
    expected_all_supply = torch.tensor([[2.0, 1.0, 0.0, 0.0, 0.0]])
    expected_near_supply = torch.tensor([[1.0 + torch.exp(torch.tensor(-1.0)), torch.exp(torch.tensor(-1.0)), 0.0, 0.0, 0.0]])
    torch.testing.assert_close(context.demand_all, expected_all_demand)
    torch.testing.assert_close(context.demand_near, expected_near_demand)
    torch.testing.assert_close(context.supply_all, expected_all_supply)
    torch.testing.assert_close(context.supply_near, expected_near_supply)
    torch.testing.assert_close(
        context.pressure_all,
        torch.log1p(expected_all_demand / expected_all_supply.clamp_min(1.0e-6)),
    )
    assert context.candidate_exposure.shape == (1, 3, 10)
    assert context.candidate_max_exposure.shape == (1, 3, 2)
    assert torch.isfinite(context.candidate_exposure).all()
    assert context.zero_supply_all[0, 2:].all()


def test_worker_pressure_context_near_demand_respects_action_mask_without_changing_long_term() -> None:
    from models.worker_pointer_context import build_worker_pressure_context

    inputs = _pressure_inputs()
    masked = build_worker_pressure_context(
        **inputs, temperature=1.0, supply_epsilon=1.0e-6
    )
    unmasked_inputs = dict(inputs)
    unmasked_inputs["task_action_invalid"] = torch.tensor([[False, False, True, True]])
    unmasked = build_worker_pressure_context(
        **unmasked_inputs, temperature=1.0, supply_epsilon=1.0e-6
    )

    torch.testing.assert_close(masked.pressure_all, unmasked.pressure_all)
    assert masked.demand_near[0, 1].item() == 0.0
    assert unmasked.demand_near[0, 1].item() == 0.0  # not-ready task remains excluded


def test_v2_gather_selected_task_skills_rejects_virtual_or_multi_skill_tasks() -> None:
    from models.worker_pointer_context import gather_selected_task_skills

    task_features = torch.zeros((2, 3, 18), dtype=torch.float32)
    task_features[0, 1, 5] = 1.0
    task_features[1, 2, 7] = 1.0
    selected_task = torch.tensor([1, 2])

    selected = gather_selected_task_skills(task_features, selected_task)

    assert selected.shape == (2, 5)
    torch.testing.assert_close(selected, task_features[[0, 1], selected_task, 5:10])

    task_features[0, 1, 6] = 1.0
    with pytest.raises(AssertionError):
        gather_selected_task_skills(task_features, selected_task)

    task_features[0, 1, 6] = 0.0
    task_features[1, 2, 7] = 0.0
    with pytest.raises(AssertionError):
        gather_selected_task_skills(task_features, selected_task)


def test_v2_marginal_reserve_scarcity_excludes_required_skill_and_reacts_to_supply() -> None:
    from models.worker_pointer_context import build_v2_marginal_reserve_scarcity

    demand_all = torch.tensor([[20.0, 0.0, 0.0, 0.0, 20.0]])
    supply_all = torch.tensor([[20.0, 0.0, 0.0, 0.0, 2.0]])
    selected_skill_sum = torch.zeros((1, 5))
    candidate_skills = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0]]]
    )
    task_required_skills = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])

    total, extra = build_v2_marginal_reserve_scarcity(
        demand_all=demand_all,
        supply_all=supply_all,
        selected_skill_sum=selected_skill_sum,
        candidate_skills=candidate_skills,
        task_required_skills=task_required_skills,
        epsilon=1.0e-6,
        clip=10.0,
    )

    assert total.shape == extra.shape == (1, 2, 1)
    assert total[0, 0].item() > total[0, 1].item()
    assert extra[0, 0].item() > 0.0
    assert extra[0, 1].item() == pytest.approx(0.0)
    assert torch.all(extra <= total + 1.0e-6)

    abundant_supply = supply_all.clone()
    abundant_supply[0, 4] = 100.0
    _, abundant_extra = build_v2_marginal_reserve_scarcity(
        demand_all=demand_all,
        supply_all=abundant_supply,
        selected_skill_sum=selected_skill_sum,
        candidate_skills=candidate_skills,
        task_required_skills=task_required_skills,
        epsilon=1.0e-6,
        clip=10.0,
    )
    assert abundant_extra[0, 0].item() < extra[0, 0].item()


def test_v2_marginal_reserve_scarcity_reflects_partial_team_consumption() -> None:
    from models.worker_pointer_context import build_v2_marginal_reserve_scarcity

    kwargs = {
        "demand_all": torch.tensor([[20.0, 0.0, 0.0, 0.0, 20.0]]),
        "supply_all": torch.tensor([[20.0, 0.0, 0.0, 0.0, 2.0]]),
        "candidate_skills": torch.tensor([[[1.0, 0.0, 0.0, 0.0, 1.0]]]),
        "task_required_skills": torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]]),
        "epsilon": 1.0e-6,
        "clip": 10.0,
    }
    _, before = build_v2_marginal_reserve_scarcity(
        selected_skill_sum=torch.zeros((1, 5)), **kwargs
    )
    _, after = build_v2_marginal_reserve_scarcity(
        selected_skill_sum=torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]]), **kwargs
    )

    assert after.item() > before.item()


def test_normalized_entropy_uses_each_sample_legal_action_count() -> None:
    from ppo_agent import PPOAgent

    entropy = torch.log(torch.tensor([4.0, 3.0, 2.0, 1.0]))
    invalid_mask = torch.tensor(
        [
            [False, False, False, False],
            [False, False, False, True],
            [False, False, True, True],
            [False, True, True, True],
        ]
    )

    normalized = PPOAgent._normalized_categorical_entropy(entropy, invalid_mask)

    torch.testing.assert_close(normalized, torch.tensor([1.0, 1.0, 1.0, 0.0]))


def test_factorized_surrogate_averages_active_components_per_sample() -> None:
    from ppo_agent import PPOAgent

    current = torch.log(
        torch.tensor([[1.2, 1.2, 1.2], [1.1, 1.1, 1.1]])
    ).requires_grad_()
    old = torch.zeros_like(current)
    advantages = torch.tensor([[1.0, 1.0, 1.0], [2.0, 0.0, 0.0]])
    active = torch.tensor([[True, True, True], [True, False, False]])

    loss, ratios, component_losses = PPOAgent.compute_factorized_clipped_surrogate(
        current_logprobs=current,
        old_logprobs=old,
        advantages=advantages,
        active_mask=active,
        clip_range=0.2,
    )

    expected = torch.tensor(-(1.2 + 2.2) / 2.0)
    torch.testing.assert_close(loss, expected)
    assert ratios.shape == component_losses.shape == (2, 3)
    loss.backward()
    assert current.grad is not None


def test_factorized_component_advantages_normalize_only_active_samples() -> None:
    from ppo_agent import PPOAgent

    returns = torch.tensor([10.0, 20.0, 30.0])
    old_values = torch.tensor(
        [[9.0, 100.0, 100.0], [17.0, 100.0, 100.0], [100.0, 1.0, 100.0]]
    )
    active = torch.tensor(
        [[True, False, False], [True, False, False], [False, True, False]]
    )

    advantages = PPOAgent.compute_factorized_component_advantages(
        returns=returns,
        old_values=old_values,
        active_mask=active,
    )

    torch.testing.assert_close(advantages[:, 0], torch.tensor([-1.0, 1.0, 0.0]))
    torch.testing.assert_close(advantages[:, 1], torch.tensor([0.0, 0.0, 0.0]))
    assert torch.isfinite(advantages).all()


def test_factorized_value_loss_averages_active_components_and_clips_old_values() -> None:
    from ppo_agent import PPOAgent

    current = torch.tensor([[1.5, 4.0, 6.0], [2.0, 8.0, 9.0]])
    old = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    targets = torch.tensor([1.0, 2.0])
    active = torch.tensor([[True, True, False], [True, False, False]])

    loss = PPOAgent.compute_factorized_value_loss(
        current_values=current,
        old_values=old,
        targets=targets,
        active_mask=active,
        clip_range=0.2,
    )

    expected = torch.tensor(((0.5**2 + 3.0**2) / 2.0 + 0.0) / 2.0)
    torch.testing.assert_close(loss, expected)


def test_factorized_value_loss_keeps_base_critic_scale() -> None:
    from ppo_agent import PPOAgent

    combined = PPOAgent.combine_factorized_value_loss(
        base_value_loss=torch.tensor(2.0),
        conditional_value_loss=torch.tensor(4.0),
        coefficient=1.0,
    )

    torch.testing.assert_close(combined, torch.tensor(3.0))


def test_next_frontier_pressure_uses_sparse_physical_predecessors() -> None:
    from models.worker_pointer_context import build_worker_pressure_context

    task_features = torch.zeros((1, 4, 18), dtype=torch.float32)
    task_features[0, :, 0] = 1.0
    task_features[0, 0, 2] = 1.0
    task_features[0, 1, 1] = 1.0
    task_features[0, 2, 1] = 1.0
    task_features[0, 3, 1] = 1.0
    task_features[0, 0:3, 5] = 1.0
    task_features[0, 2, 16] = 1.0
    worker_features = torch.zeros((1, 2, 17), dtype=torch.float32)
    worker_features[0, :, 1] = 1.0
    present = torch.ones((1, 4), dtype=torch.bool)
    worker_present = torch.ones((1, 2), dtype=torch.bool)
    edge_index = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)

    context = build_worker_pressure_context(
        task_features=task_features,
        worker_features=worker_features,
        task_present=present,
        task_action_invalid=torch.zeros_like(present),
        worker_present=worker_present,
        worker_queue_invalid=torch.zeros_like(worker_present),
        temperature=1.0,
        supply_epsilon=1.0e-6,
        physical_predecessor_edges=[edge_index],
    )

    assert context.pressure_next_frontier is not None
    assert context.next_frontier_mask is not None
    assert context.unfinished_physical_predecessor_count is not None
    assert context.remaining_physical_predecessor_count is not None
    assert context.next_frontier_mask.shape == (1, 4)
    assert context.unfinished_physical_predecessor_count.shape == (1, 4)
    assert context.remaining_physical_predecessor_count.shape == (1, 4)
    assert not bool(context.next_frontier_mask[0, 2])

    task_features[0, 1, 1] = 0.0
    task_features[0, 1, 2] = 1.0
    context = build_worker_pressure_context(
        task_features=task_features,
        worker_features=worker_features,
        task_present=present,
        task_action_invalid=torch.zeros_like(present),
        worker_present=worker_present,
        worker_queue_invalid=torch.zeros_like(worker_present),
        temperature=1.0,
        supply_epsilon=1.0e-6,
        physical_predecessor_edges=[edge_index],
    )
    assert bool(context.next_frontier_mask[0, 2])
    assert float(context.pressure_next_frontier[0, 0]) > 0.0
    assert context.pressure_next_frontier.shape == (1, 5)


def test_v2_diagnostics_report_next_frontier_task_demand_and_pressure() -> None:
    from models.worker_pointer_context import build_worker_pressure_context
    from training.worker_pointer_v2_diagnostics import WorkerPointerV2Diagnostics

    inputs = _pressure_inputs()
    task_features = inputs["task_features"].clone()
    task_features[0, 1, 0] = 2.0
    task_features[0, 1, 16] = 3.0
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    context = build_worker_pressure_context(
        **{**inputs, "task_features": task_features},
        temperature=1.0,
        supply_epsilon=1.0e-6,
        physical_predecessor_edges=[edge_index],
    )
    diagnostics = WorkerPointerV2Diagnostics(num_skills=5)
    diagnostics.record_context(context, host_elapsed_ms=0.0)

    metrics = diagnostics.finalize(require_coverage=False)

    assert metrics["PointerV2/NextFrontier/TaskCountMean"] == pytest.approx(1.0)
    assert metrics["PointerV2/NextFrontier/DemandMean"] == pytest.approx(6.0)
    assert metrics["PointerV2/NextFrontier/PressureMean"] > 0.0


def test_c1_sparse_predecessor_edges_keep_pyg_batch_offsets() -> None:
    from torch_geometric.data import Batch, HeteroData

    from models.worker_pointer_context import PHYSICAL_PREDECESSOR_EDGE
    from ppo_agent import PPOAgent

    graphs: list[HeteroData] = []
    for _ in range(2):
        graph = HeteroData()
        graph["task"].x = torch.zeros((2, 18))
        graph[PHYSICAL_PREDECESSOR_EDGE].edge_index = torch.tensor(
            [[0], [1]], dtype=torch.long
        )
        graphs.append(graph)

    batch = Batch.from_data_list(graphs)
    assert batch[PHYSICAL_PREDECESSOR_EDGE].edge_index.tolist() == [
        [0, 2],
        [1, 3],
    ]
    local_edges = PPOAgent._physical_predecessor_edge_list(batch, batch_size=2)

    assert [edge.tolist() for edge in local_edges] == [[[0], [1]], [[0], [1]]]
    assert max(int(edge.max()) for edge in local_edges) < 2


def test_v2_diagnostics_summarize_action_space_and_eft_rank() -> None:
    from models.worker_pointer_context import build_worker_pressure_context
    from training.worker_pointer_v2_diagnostics import WorkerPointerV2Diagnostics

    diagnostics = WorkerPointerV2Diagnostics(num_skills=5)
    diagnostics.record_context(
        build_worker_pressure_context(
            **_pressure_inputs(), temperature=1.0, supply_epsilon=1.0e-6
        ),
        host_elapsed_ms=1.0,
    )
    diagnostics.record_action_space(
        ready_task_count=torch.tensor([2.0]),
        legal_station_count=torch.tensor([3.0]),
        legal_worker_count=torch.tensor([4.0]),
    )
    diagnostics.record_selection(
        selected_exposure=torch.zeros((1, 12)),
        entropy=torch.tensor([0.5]),
        dynamic_eft_features=torch.tensor([[[0.0, -1.0], [0.2, 0.0], [0.8, 1.0]]]),
        selected_worker_index=1,
        worker_invalid_mask=torch.tensor([[False, False, False]]),
    )

    metrics = diagnostics.finalize(require_coverage=False)

    assert metrics["PointerV2/ActionSpace/ReadyTaskMean"] == 2.0
    assert metrics["PointerV2/ActionSpace/LegalStationMean"] == 3.0
    assert metrics["PointerV2/ActionSpace/LegalWorkerMean"] == 4.0
    assert metrics["PointerV2/ActionSpace/LegalWorkerP10"] == 4.0
    assert metrics["PointerV2/EFT/SelectedRankPercentileMean"] == 0.5


def test_v2_diagnostics_summarize_partial_team_operational_state() -> None:
    from models.worker_pointer_context import build_worker_pressure_context
    from training.worker_pointer_v2_diagnostics import WorkerPointerV2Diagnostics

    diagnostics = WorkerPointerV2Diagnostics(num_skills=5)
    diagnostics.record_context(
        build_worker_pressure_context(
            **_pressure_inputs(), temperature=1.0, supply_epsilon=1.0e-6
        ),
        host_elapsed_ms=1.0,
    )
    diagnostics.record_team_state(
        selected_max_wait=torch.tensor([[0.0], [3.0]]),
        selected_capacity_sum=torch.tensor([[0.0], [4.0]]),
    )

    metrics = diagnostics.finalize(require_coverage=False)

    assert metrics["PointerV2/TeamState/MaxWaitMean"] == pytest.approx(1.5)
    assert metrics["PointerV2/TeamState/MaxWaitP95"] == pytest.approx(2.85)
    assert metrics["PointerV2/TeamState/CapacityMean"] == pytest.approx(2.0)
    assert metrics["PointerV2/TeamState/CapacityP95"] == pytest.approx(3.8)


def test_v2_diagnostics_summarize_marginal_reserve_scarcity() -> None:
    from models.worker_pointer_context import build_worker_pressure_context
    from training.worker_pointer_v2_diagnostics import WorkerPointerV2Diagnostics

    diagnostics = WorkerPointerV2Diagnostics(num_skills=5)
    diagnostics.record_context(
        build_worker_pressure_context(
            **_pressure_inputs(), temperature=1.0, supply_epsilon=1.0e-6
        ),
        host_elapsed_ms=1.0,
    )
    diagnostics.record_scarcity(
        marginal_total=torch.tensor([[[2.0], [4.0], [10.0]]]),
        marginal_extra=torch.tensor([[[0.0], [1.0], [3.0]]]),
        worker_invalid_mask=torch.tensor([[False, False, True]]),
        selected_worker_index=1,
    )

    metrics = diagnostics.finalize(require_coverage=False)

    assert metrics["PointerV2/Scarcity/ReserveTotalLegalMean"] == pytest.approx(3.0)
    assert metrics["PointerV2/Scarcity/ReserveTotalLegalP95"] == pytest.approx(3.9)
    assert metrics["PointerV2/Scarcity/ReserveExtraLegalMean"] == pytest.approx(0.5)
    assert metrics["PointerV2/Scarcity/ReserveExtraLegalP95"] == pytest.approx(0.95)
    assert metrics["PointerV2/Scarcity/ReserveExtraLegalMax"] == pytest.approx(1.0)
    assert metrics["PointerV2/Scarcity/ReserveExtraSelectedMean"] == pytest.approx(1.0)
    assert metrics["PointerV2/Scarcity/ReserveExtraTotalRatio"] == pytest.approx(1.0 / 6.0)


def test_v2_diagnostics_summarize_legal_eft_feature_quantiles() -> None:
    from models.worker_pointer_context import build_worker_pressure_context
    from training.worker_pointer_v2_diagnostics import WorkerPointerV2Diagnostics

    diagnostics = WorkerPointerV2Diagnostics(num_skills=5)
    diagnostics.record_context(
        build_worker_pressure_context(
            **_pressure_inputs(), temperature=1.0, supply_epsilon=1.0e-6
        ),
        host_elapsed_ms=1.0,
    )
    diagnostics.record_selection(
        selected_exposure=torch.zeros((1, 12)),
        entropy=torch.tensor([0.5]),
        dynamic_eft_features=torch.tensor(
            [[[0.0, -1.0], [0.2, 0.0], [9.0, 9.0]]]
        ),
        selected_worker_index=1,
        worker_invalid_mask=torch.tensor([[False, False, True]]),
    )

    metrics = diagnostics.finalize(require_coverage=False)

    assert metrics["RelativeEFT/LegalMean"] == pytest.approx(0.1)
    assert metrics["RelativeEFT/LegalP50"] == pytest.approx(0.1)
    assert metrics["RelativeEFT/LegalP90"] == pytest.approx(0.18)
    assert metrics["RelativeEFT/LegalP99"] == pytest.approx(0.198)
    assert metrics["RelativeEFT/LegalMax"] == pytest.approx(0.2)
    assert metrics["ZScoreEFT/LegalMean"] == pytest.approx(-0.5)
    assert metrics["ZScoreEFT/LegalP50"] == pytest.approx(-0.5)
    assert metrics["ZScoreEFT/LegalP90"] == pytest.approx(-0.1)
    assert metrics["ZScoreEFT/LegalP99"] == pytest.approx(-0.01)
    assert metrics["ZScoreEFT/LegalMax"] == pytest.approx(0.0)


def test_v2_gradient_diagnostics_separate_dynamic_eft_projection() -> None:
    from ppo_agent import PPOAgent

    eft_projection = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    eft_projection.grad = torch.tensor([3.0, 4.0])
    v2_attention = torch.nn.Parameter(torch.tensor([3.0]))
    v2_attention.grad = torch.tensor([4.0])

    metrics = PPOAgent._collect_gradient_diagnostics(
        (
            ("worker_head.v2_eft_proj.weight", eft_projection),
            ("worker_head.v2_attn.weight", v2_attention),
        )
    )

    assert metrics["dynamic_eft_projection_grad_norm"] == pytest.approx(5.0)
    assert metrics["dynamic_eft_projection_param_norm"] == pytest.approx(2.0**0.5)
    assert metrics["dynamic_eft_projection_grad_to_param"] == pytest.approx(5.0 / 2.0**0.5)
    assert metrics["worker_v2_grad_to_param"] == pytest.approx(41.0**0.5 / 11.0**0.5)


def test_v2_model_spec_records_pressure_semantics_and_rejects_cross_mode_resume() -> None:
    cfg = Config()
    cfg.team_selection_mode = "autoregressive_pressure_v2"
    cfg.policy_action_scope = "operation_station_worker"
    cfg.actor_context_mode = "attention"
    cfg.worker_pointer_context_version = "pressure_v2_normwork_headcount_physicalwait_v1"
    cfg.worker_pointer_pressure_temperature = 1.0
    cfg.worker_pointer_supply_epsilon = 1.0e-6
    cfg.worker_pointer_wait_discount_mode = "physical_wait_exponential_v1"
    spec = build_model_spec(cfg)

    assert spec.team_selection_mode == "autoregressive_pressure_v2"
    assert spec.worker_pointer_context_version == cfg.worker_pointer_context_version
    assert spec.worker_pointer_pressure_temperature == 1.0
    assert spec.worker_pointer_supply_epsilon == 1.0e-6
    assert spec.worker_pointer_wait_discount_mode == "physical_wait_exponential_v1"
    assert spec.worker_pointer_v2_dynamic_eft_features is False

    legacy = Config()
    with pytest.raises(ValueError, match="team_selection_mode"):
        apply_checkpoint_model_spec(legacy, spec)

    v2 = Config()
    v2.team_selection_mode = "autoregressive_pressure_v2"
    v2.policy_action_scope = "operation_station_worker"
    v2.actor_context_mode = "attention"
    legacy_spec = replace(spec, team_selection_mode="autoregressive")
    with pytest.raises(ValueError, match="team_selection_mode"):
        apply_checkpoint_model_spec(v2, legacy_spec)


def test_v2_dynamic_eft_checkpoint_semantics_reject_cross_mode_resume() -> None:
    cfg = Config()
    cfg.team_selection_mode = "autoregressive_pressure_v2"
    cfg.policy_action_scope = "operation_station_worker"
    cfg.actor_context_mode = "attention"
    cfg.worker_pointer_v2_dynamic_eft_features = True
    cfg.worker_pointer_v2_dynamic_eft_feature_clip = 10.0
    spec = build_model_spec(cfg)

    assert spec.worker_pointer_v2_dynamic_eft_features is True
    assert spec.worker_pointer_v2_dynamic_eft_feature_clip == 10.0

    incompatible = Config()
    incompatible.team_selection_mode = "autoregressive_pressure_v2"
    incompatible.policy_action_scope = "operation_station_worker"
    incompatible.actor_context_mode = "attention"
    with pytest.raises(ValueError, match="dynamic_eft"):
        apply_checkpoint_model_spec(incompatible, spec)


def test_runtime_validation_accepts_only_attention_worker_scope_for_v2() -> None:
    from runtime.configuration import validate_runtime_config

    valid = Config()
    valid.team_selection_mode = "autoregressive_pressure_v2"
    valid.policy_action_scope = "operation_station_worker"
    valid.actor_context_mode = "attention"
    valid.worker_pointer_v2_behavior_replay = True
    valid.worker_pointer_v2_replay_mode = "behavior_group_exact_v1"
    validate_runtime_config(valid)

    wrong_scope = Config()
    wrong_scope.team_selection_mode = "autoregressive_pressure_v2"
    wrong_scope.policy_action_scope = "operation_station_anchor_proposal_team"
    wrong_scope.actor_context_mode = "attention"
    with pytest.raises(ValueError, match="operation_station_worker"):
        validate_runtime_config(wrong_scope)

    local_only = Config()
    local_only.team_selection_mode = "autoregressive_pressure_v2"
    local_only.policy_action_scope = "operation_station_worker"
    local_only.actor_context_mode = "local_only"
    validate_runtime_config(local_only)

    wrong_context = Config()
    wrong_context.team_selection_mode = "autoregressive_pressure_v2"
    wrong_context.policy_action_scope = "operation_station_worker"
    wrong_context.actor_context_mode = "mean_max"
    with pytest.raises(ValueError, match="actor_context_mode"):
        validate_runtime_config(wrong_context)

def _batched_v2_config() -> Config:
    cfg = Config()
    cfg.team_selection_mode = "autoregressive_pressure_v2"
    cfg.policy_action_scope = "operation_station_worker"
    cfg.actor_context_mode = "attention"
    cfg.worker_pointer_v2_behavior_replay = False
    cfg.worker_pointer_v2_replay_mode = "batched_vectorized_v2"
    return cfg


def test_runtime_validation_accepts_batched_vectorized_v2_without_behavior_trace() -> None:
    from runtime.configuration import validate_runtime_config

    validate_runtime_config(_batched_v2_config())


def test_runtime_validation_rejects_batched_v2_with_behavior_trace_enabled() -> None:
    from runtime.configuration import validate_runtime_config

    cfg = _batched_v2_config()
    cfg.worker_pointer_v2_behavior_replay = True
    with pytest.raises(ValueError, match="batched_vectorized_v2"):
        validate_runtime_config(cfg)


def test_runtime_validation_rejects_group_exact_without_behavior_trace() -> None:
    from runtime.configuration import validate_runtime_config

    cfg = _batched_v2_config()
    cfg.worker_pointer_v2_replay_mode = "behavior_group_exact_v1"
    with pytest.raises(ValueError, match="behavior_group_exact_v1"):
        validate_runtime_config(cfg)


def test_runtime_validation_scopes_conditional_head_baseline_modes() -> None:
    from runtime.configuration import validate_runtime_config

    valid = _batched_v2_config()
    valid.conditional_head_baseline_mode = "diagnostic"
    validate_runtime_config(valid)

    wrong_team_mode = Config()
    wrong_team_mode.conditional_head_baseline_mode = "factorized"
    with pytest.raises(ValueError, match="autoregressive_pressure_v2"):
        validate_runtime_config(wrong_team_mode)

    shared_trunk = _batched_v2_config()
    shared_trunk.conditional_head_baseline_mode = "factorized"
    shared_trunk.use_shared_trunk = True
    with pytest.raises(ValueError, match="use_shared_trunk"):
        validate_runtime_config(shared_trunk)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("conditional_head_baseline_mode", "unsupported", "conditional_head_baseline_mode"),
        ("worker_pointer_v2_marginal_scarcity_clip", float("inf"), "marginal_scarcity_clip"),
        ("conditional_head_value_coef", 0.0, "conditional_head_value_coef"),
    ],
)
def test_runtime_validation_rejects_invalid_worker_pointer_v2_architecture_config(
    field_name: str,
    value: object,
    message: str,
) -> None:
    from runtime.configuration import validate_runtime_config

    cfg = _batched_v2_config()
    setattr(cfg, field_name, value)

    with pytest.raises(ValueError, match=message):
        validate_runtime_config(cfg)


def test_legacy_mode_never_uses_v2_behavior_group_exact_replay() -> None:
    from runtime.modes import uses_behavior_group_exact_replay

    cfg = Config()
    cfg.team_selection_mode = "autoregressive"
    cfg.worker_pointer_v2_replay_mode = "behavior_group_exact_v1"

    assert uses_behavior_group_exact_replay(cfg) is False


def _pointer_config(mode: str) -> Config:
    cfg = Config()
    cfg.hidden_dim = 8
    cfg.team_selection_mode = mode
    cfg.policy_action_scope = "operation_station_worker"
    cfg.actor_context_mode = "attention"
    cfg.seed = 42
    cfg.worker_pointer_v2_init_seed_offset = 1009
    return cfg


def test_worker_pointer_v2_deepsets_state_is_order_invariant() -> None:
    head = WorkerPointer(_pointer_config("autoregressive_pressure_v2"))
    embeddings = torch.tensor(
        [[[1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.5, -1.0],
          [0.0, 3.0, 1.0, 2.0, 0.0, 1.0, -0.5, 1.0]]]
    )
    skills = torch.tensor([[[1.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 1.0, 1.0, 0.0, 0.0]]])

    first = head.initialize_v2_state(batch_size=1, device=embeddings.device)
    first = head.advance_v2_state(first, embeddings[:, 0], skills[:, 0])
    first = head.advance_v2_state(first, embeddings[:, 1], skills[:, 1])
    second = head.initialize_v2_state(batch_size=1, device=embeddings.device)
    second = head.advance_v2_state(second, embeddings[:, 1], skills[:, 1])
    second = head.advance_v2_state(second, embeddings[:, 0], skills[:, 0])

    torch.testing.assert_close(first.mapped_sum, second.mapped_sum)
    torch.testing.assert_close(first.mapped_max, second.mapped_max)
    torch.testing.assert_close(first.selected_skill_sum, second.selected_skill_sum)
    torch.testing.assert_close(
        head.v2_team_representation(first), head.v2_team_representation(second)
    )


def test_worker_pointer_v2_scores_enriched_context_and_preserves_mask() -> None:
    from models.worker_pointer_context import build_worker_pressure_context

    head = WorkerPointer(_pointer_config("autoregressive_pressure_v2"))
    context = build_worker_pressure_context(
        **_pressure_inputs(), temperature=1.0, supply_epsilon=1.0e-6
    )
    state = head.initialize_v2_state(batch_size=1, device=torch.device("cpu"))
    logits = head.forward_choice_v2(
        task_emb=torch.randn(1, 8),
        station_emb=torch.randn(1, 8),
        global_context=torch.randn(1, 24),
        worker_embs=torch.randn(1, 3, 8),
        pressure_context=context,
        team_state=state,
        demand=torch.tensor([2.0]),
        mask=torch.tensor([[False, True, True]]),
    )

    assert logits.shape == (1, 3)
    assert torch.isfinite(logits[:, :1]).all()
    assert logits[0, 1].item() == pytest.approx(-1.0e4)
    assert logits[0, 2].item() == pytest.approx(-1.0e4)


def test_worker_pointer_v2_6h_context_uses_only_first_head() -> None:
    from models.worker_pointer_context import build_worker_pressure_context

    head = WorkerPointer(_pointer_config("autoregressive_pressure_v2"))
    context = build_worker_pressure_context(
        **_pressure_inputs(), temperature=1.0, supply_epsilon=1.0e-6
    )
    torch.manual_seed(1234)
    kwargs = {
        "task_emb": torch.randn(1, 8),
        "station_emb": torch.randn(1, 8),
        "worker_embs": torch.randn(1, 3, 8),
        "pressure_context": context,
        "team_state": head.initialize_v2_state(batch_size=1, device=torch.device("cpu")),
        "demand": torch.tensor([2.0]),
        "mask": torch.tensor([[False, False, True]]),
    }
    context_first = torch.randn(1, 24)
    context_second = torch.randn(1, 24)
    cache_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in {"team_state", "mask"}
    }
    cache_first = head.build_v2_decode_cache(
        **cache_kwargs, global_context=context_first
    )
    cache_dual = head.build_v2_decode_cache(
        **cache_kwargs, global_context=torch.cat([context_first, context_second], dim=-1)
    )
    torch.testing.assert_close(cache_dual.query_prefix, cache_first.query_prefix, atol=0.0, rtol=0.0)
    logits_first = head.forward_choice_v2(
        **kwargs, global_context=context_first, decode_cache=cache_first
    )
    logits_dual = head.forward_choice_v2(
        **kwargs,
        global_context=torch.cat([context_first, context_second], dim=-1),
        decode_cache=cache_dual,
    )
    torch.testing.assert_close(logits_dual, logits_first, atol=0.0, rtol=0.0)


def test_worker_pointer_v2_a1_nested_initialization_preserves_v0_parameters() -> None:
    v0_config = _pointer_config("autoregressive_pressure_v2")
    v1_config = _pointer_config("autoregressive_pressure_v2")
    v1_config.worker_pointer_v2_explicit_team_state = True
    torch.manual_seed(1234)
    v0 = WorkerPointer(v0_config)
    torch.manual_seed(1234)
    v1 = WorkerPointer(v1_config)

    v0_state = v0.state_dict()
    v1_state = v1.state_dict()
    query_weight = "v2_query_proj.weight"
    assert set(v0_state) == set(v1_state)
    assert v0.v2_query_proj.in_features == 8 * 6 + 17
    assert v1.v2_query_proj.in_features == 8 * 6 + 19
    torch.testing.assert_close(v1_state[query_weight][:, : 8 * 6 + 17], v0_state[query_weight])
    torch.testing.assert_close(v1_state[query_weight][:, 8 * 6 + 17 :], torch.zeros((8, 2)))
    for key in v0_state:
        if key != query_weight:
            torch.testing.assert_close(v1_state[key], v0_state[key], atol=0.0, rtol=0.0)


def test_worker_pointer_v2_a2_nested_initialization_and_step_zero_are_invariant() -> None:
    from models.worker_pointer_context import build_worker_pressure_context

    v1_config = _pointer_config("autoregressive_pressure_v2")
    v1_config.worker_pointer_v2_explicit_team_state = True
    v2_config = _pointer_config("autoregressive_pressure_v2")
    v2_config.worker_pointer_v2_explicit_team_state = True
    v2_config.worker_pointer_v2_marginal_scarcity = True
    torch.manual_seed(1234)
    v1 = WorkerPointer(v1_config)
    torch.manual_seed(1234)
    v2 = WorkerPointer(v2_config)

    v1_state = v1.state_dict()
    v2_state = v2.state_dict()
    for key in v1_state:
        torch.testing.assert_close(v2_state[key], v1_state[key], atol=0.0, rtol=0.0)
    assert torch.count_nonzero(v2.v2_marginal_proj.weight).item() == 0

    context = build_worker_pressure_context(
        **_pressure_inputs(), temperature=1.0, supply_epsilon=1.0e-6
    )
    common = {
        "task_emb": torch.randn((1, 8)),
        "station_emb": torch.randn((1, 8)),
        "global_context": torch.randn((1, 24)),
        "worker_embs": torch.randn((1, 3, 8)),
        "pressure_context": context,
        "demand": torch.tensor([2.0]),
        "mask": torch.tensor([[False, False, True]]),
        "candidate_skills": _pressure_inputs()["worker_features"][..., 1:6],
        "task_required_skills": torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]]),
    }
    cache_common = {
        key: value for key, value in common.items() if key != "mask"
    }
    v1_cache = v1.build_v2_decode_cache(**cache_common)
    v2_cache = v2.build_v2_decode_cache(**cache_common)
    state_v1 = v1.initialize_v2_state(batch_size=1, device=torch.device("cpu"))
    state_v2 = v2.initialize_v2_state(batch_size=1, device=torch.device("cpu"))
    logits_v1 = v1.forward_choice_v2(
        **common, team_state=state_v1, decode_cache=v1_cache
    )
    logits_v2 = v2.forward_choice_v2(
        **common, team_state=state_v2, decode_cache=v2_cache
    )

    torch.testing.assert_close(logits_v2, logits_v1, atol=1.0e-6, rtol=0.0)
    assert v2._last_v2_marginal_total is not None
    assert v2._last_v2_marginal_extra is not None
    assert torch.isfinite(v2._last_v2_marginal_total).all()
    assert torch.isfinite(v2._last_v2_marginal_extra).all()


def test_worker_pointer_v2_a3_zero_init_preserves_v2_logits_and_public_parameters() -> None:
    from models.worker_pointer_context import build_worker_pressure_context

    v2_config = _pointer_config("autoregressive_pressure_v2")
    v2_config.worker_pointer_v2_explicit_team_state = True
    v2_config.worker_pointer_v2_marginal_scarcity = True
    v3_config = _pointer_config("autoregressive_pressure_v2")
    v3_config.worker_pointer_v2_explicit_team_state = True
    v3_config.worker_pointer_v2_marginal_scarcity = True
    v3_config.worker_pointer_v2_interaction_residual = True
    torch.manual_seed(1234)
    v2 = WorkerPointer(v2_config)
    torch.manual_seed(1234)
    v3 = WorkerPointer(v3_config)

    for key, value in v2.state_dict().items():
        torch.testing.assert_close(v3.state_dict()[key], value, atol=0.0, rtol=0.0)
    assert torch.count_nonzero(v3.v2_interaction_mlp[-1].weight).item() == 0
    assert torch.count_nonzero(v3.v2_interaction_mlp[-1].bias).item() == 0


    context = build_worker_pressure_context(
        **_pressure_inputs(), temperature=1.0, supply_epsilon=1.0e-6
    )
    common = {
        "task_emb": torch.randn((1, 8)),
        "station_emb": torch.randn((1, 8)),
        "global_context": torch.randn((1, 24)),
        "worker_embs": torch.randn((1, 3, 8)),
        "pressure_context": context,
        "demand": torch.tensor([2.0]),
        "mask": torch.tensor([[False, False, True]]),
        "candidate_skills": _pressure_inputs()["worker_features"][..., 1:6],
        "task_required_skills": torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]]),
    }
    cache_v2 = v2.build_v2_decode_cache(
        **{key: value for key, value in common.items() if key != "mask"}
    )
    cache_v3 = v3.build_v2_decode_cache(
        **{key: value for key, value in common.items() if key != "mask"}
    )
    logits_v2 = v2.forward_choice_v2(
        **common,
        team_state=v2.initialize_v2_state(batch_size=1, device=torch.device("cpu")),
        decode_cache=cache_v2,
    )
    logits_v3 = v3.forward_choice_v2(
        **common,
        team_state=v3.initialize_v2_state(batch_size=1, device=torch.device("cpu")),
        decode_cache=cache_v3,
    )

    torch.testing.assert_close(logits_v3, logits_v2, atol=1.0e-6, rtol=0.0)


def test_worker_pointer_v2_c1_zero_init_preserves_v2_logits_and_public_parameters() -> None:
    from models.worker_pointer_context import build_worker_pressure_context

    v3_config = _pointer_config("autoregressive_pressure_v2")
    v3_config.worker_pointer_v2_explicit_team_state = True
    v3_config.worker_pointer_v2_marginal_scarcity = True
    v3_config.worker_pointer_v2_interaction_residual = True
    c1_config = _pointer_config("autoregressive_pressure_v2")
    c1_config.worker_pointer_v2_explicit_team_state = True
    c1_config.worker_pointer_v2_marginal_scarcity = True
    c1_config.worker_pointer_v2_interaction_residual = True
    c1_config.worker_pointer_v2_next_frontier_pressure = True
    torch.manual_seed(1234)
    v3 = WorkerPointer(v3_config)
    torch.manual_seed(1234)
    c1 = WorkerPointer(c1_config)

    for key, value in v3.state_dict().items():
        torch.testing.assert_close(c1.state_dict()[key], value, atol=0.0, rtol=0.0)
    assert torch.count_nonzero(c1.v2_next_frontier_query_proj.weight).item() == 0
    assert torch.count_nonzero(c1.v2_next_frontier_key_proj.weight).item() == 0

    inputs = _pressure_inputs()
    context = build_worker_pressure_context(
        **inputs,
        temperature=1.0,
        supply_epsilon=1.0e-6,
        physical_predecessor_edges=[torch.tensor([[0], [1]], dtype=torch.long)],
    )
    common = {
        "task_emb": torch.randn((1, 8)),
        "station_emb": torch.randn((1, 8)),
        "global_context": torch.randn((1, 24)),
        "worker_embs": torch.randn((1, 3, 8)),
        "pressure_context": context,
        "demand": torch.tensor([2.0]),
        "mask": torch.tensor([[False, False, True]]),
        "candidate_skills": inputs["worker_features"][..., 1:6],
        "task_required_skills": torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]]),
    }
    v3_logits = v3.forward_choice_v2(
        **common,
        team_state=v3.initialize_v2_state(batch_size=1, device=torch.device("cpu")),
    )
    c1_logits = c1.forward_choice_v2(
        **common,
        team_state=c1.initialize_v2_state(batch_size=1, device=torch.device("cpu")),
    )

    torch.testing.assert_close(c1_logits, v3_logits, atol=1.0e-6, rtol=0.0)
    assert context.pressure_next_frontier is not None
    assert torch.isfinite(context.pressure_next_frontier).all()


def test_worker_pointer_v2_a3_interaction_residual_receives_gradient() -> None:
    from models.worker_pointer_context import build_worker_pressure_context

    config = _pointer_config("autoregressive_pressure_v2")
    config.worker_pointer_v2_interaction_residual = True
    head = WorkerPointer(config)
    context = build_worker_pressure_context(
        **_pressure_inputs(), temperature=1.0, supply_epsilon=1.0e-6
    )
    common = {
        "task_emb": torch.randn((1, 8)),
        "station_emb": torch.randn((1, 8)),
        "global_context": torch.randn((1, 24)),
        "worker_embs": torch.randn((1, 3, 8)),
        "pressure_context": context,
        "team_state": head.initialize_v2_state(batch_size=1, device=torch.device("cpu")),
        "demand": torch.tensor([2.0]),
        "mask": torch.tensor([[False, False, True]]),
        "candidate_skills": _pressure_inputs()["worker_features"][..., 1:6],
        "task_required_skills": torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]]),
    }
    logits = head.forward_choice_v2(**common)
    logits[:, :2].sum().backward()

    assert head.v2_interaction_mlp[-1].weight.grad is not None
    assert torch.isfinite(head.v2_interaction_mlp[-1].weight.grad).all()


def test_conditional_critic_diagnostic_heads_detach_shared_inputs() -> None:
    from models.hb_gat_pn import HBGATPN

    config = Config()
    config.hidden_dim = 8
    config.conditional_head_baseline_mode = "diagnostic"
    model = HBGATPN(config)
    critic_context = torch.randn((2, 24), requires_grad=True)
    critic_task_emb = torch.randn((2, 8), requires_grad=True)
    critic_station_emb = torch.randn((2, 8), requires_grad=True)
    virtual_station = torch.tensor([False, True])

    values = model.compute_conditional_values(
        critic_context=critic_context,
        critic_task_emb=critic_task_emb,
        critic_station_emb=critic_station_emb,
        virtual_station=virtual_station,
    )
    assert set(values) == {"task", "station", "worker"}
    assert all(value.shape == (2, 1) for value in values.values())
    values["task"].sum().backward()

    assert model.critic_task_cond[-1].weight.grad is not None
    assert torch.isfinite(model.critic_task_cond[-1].weight.grad).all()
    assert critic_context.grad is None
    assert critic_task_emb.grad is None
    assert critic_station_emb.grad is None
    assert all(parameter.grad is None for parameter in model.critic.parameters())

    torch.testing.assert_close(
        values["station"][1],
        model.compute_conditional_values(
            critic_context=critic_context.detach(),
            critic_task_emb=critic_task_emb.detach(),
            critic_station_emb=torch.zeros_like(critic_station_emb),
            virtual_station=torch.tensor([False, True]),
        )["station"][1],
    )


def test_memory_has_component_behavior_fields_aligned_with_state_storage() -> None:
    from training.memory import Memory

    memory = Memory()

    assert memory.old_task_logprob == []
    assert memory.old_station_logprob == []
    assert memory.old_team_logprob == []
    assert memory.old_V_task == []
    assert memory.old_V_station == []
    assert memory.old_V_worker == []

    memory.states.append({"step": 0})
    memory.old_task_logprob.append(0.1)
    memory.old_station_logprob.append(0.2)
    memory.old_team_logprob.append(0.3)
    memory.old_V_task.append(1.0)
    memory.old_V_station.append(2.0)
    memory.old_V_worker.append(3.0)
    assert len(memory.old_task_logprob) == len(memory.states)
    assert len(memory.old_V_worker) == len(memory.states)

    memory.clear()
    assert not memory.old_task_logprob
    assert not memory.old_V_worker


def test_rollout_memory_append_keeps_component_behavior_data_on_cpu() -> None:
    from training.memory import Memory
    from training.rollout_service import APALRolloutService

    class StubRolloutService:
        device = torch.device("cpu")

    memory = Memory()
    APALRolloutService._append_action(
        StubRolloutService(),
        memory,
        state={"step": 0},
        action=(1, 0, [2]),
        logprob=-0.6,
        value=0.4,
        masks=(None, None, None),
        component_behavior_logprobs=(0.1, 0.2, 0.3),
        conditional_values=(1.0, 2.0, 3.0),
    )

    assert memory.old_task_logprob == [0.1]
    assert memory.old_station_logprob == [0.2]
    assert memory.old_team_logprob == [0.3]
    assert memory.old_V_task == [1.0]
    assert memory.old_V_station == [2.0]
    assert memory.old_V_worker == [3.0]
    assert all(isinstance(value, float) for value in memory.old_V_task)


def test_worker_pointer_v2_a1_uses_log1p_partial_team_operational_state() -> None:
    from models.worker_pointer_context import build_worker_pressure_context

    config = _pointer_config("autoregressive_pressure_v2")
    config.worker_pointer_v2_explicit_team_state = True
    head = WorkerPointer(config)
    context = build_worker_pressure_context(
        **_pressure_inputs(), temperature=1.0, supply_epsilon=1.0e-6
    )
    query_inputs: list[torch.Tensor] = []
    head.v2_query_proj.register_forward_pre_hook(
        lambda _module, inputs: query_inputs.append(inputs[0].detach().clone())
    )
    kwargs = {
        "task_emb": torch.zeros((1, 8)),
        "station_emb": torch.zeros((1, 8)),
        "global_context": torch.zeros((1, 24)),
        "worker_embs": torch.zeros((1, 3, 8)),
        "pressure_context": context,
        "demand": torch.tensor([2.0]),
        "mask": torch.tensor([[False, False, True]]),
    }
    state = head.initialize_v2_state(batch_size=1, device=torch.device("cpu"))
    head.forward_choice_v2(**kwargs, team_state=state)
    state = head.advance_v2_state(
        state,
        torch.zeros((1, 8)),
        torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]]),
        selected_wait=torch.tensor([3.0]),
        selected_capacity=torch.tensor([4.0]),
    )
    head.forward_choice_v2(**kwargs, team_state=state)

    assert len(query_inputs) == 2
    assert query_inputs[0].shape == (1, 8 * 6 + 19)
    torch.testing.assert_close(query_inputs[0][0, -2:], torch.zeros(2))
    torch.testing.assert_close(
        query_inputs[1][0, -2:], torch.log1p(torch.tensor([3.0, 4.0]))
    )


def test_worker_pointer_v2_a1_step_zero_logits_and_mask_match_v0() -> None:
    from models.worker_pointer_context import build_worker_pressure_context

    v0 = WorkerPointer(_pointer_config("autoregressive_pressure_v2"))
    v1_config = _pointer_config("autoregressive_pressure_v2")
    v1_config.worker_pointer_v2_explicit_team_state = True
    v1 = WorkerPointer(v1_config)
    context = build_worker_pressure_context(
        **_pressure_inputs(), temperature=1.0, supply_epsilon=1.0e-6
    )
    common = {
        "task_emb": torch.randn((1, 8)),
        "station_emb": torch.randn((1, 8)),
        "global_context": torch.randn((1, 24)),
        "worker_embs": torch.randn((1, 3, 8)),
        "pressure_context": context,
        "demand": torch.tensor([2.0]),
        "mask": torch.tensor([[False, False, True]]),
    }
    v0_logits = v0.forward_choice_v2(
        **common,
        team_state=v0.initialize_v2_state(batch_size=1, device=torch.device("cpu")),
    )
    v1_logits = v1.forward_choice_v2(
        **common,
        team_state=v1.initialize_v2_state(batch_size=1, device=torch.device("cpu")),
    )

    torch.testing.assert_close(v1_logits[:, :2], v0_logits[:, :2], atol=1.0e-6, rtol=0.0)
    assert v0_logits[0, 2].item() == pytest.approx(-1.0e4)
    assert v1_logits[0, 2].item() == pytest.approx(-1.0e4)


def test_worker_pointer_v2_rejects_context_width_other_than_3h_or_6h() -> None:
    from models.worker_pointer_context import build_worker_pressure_context

    head = WorkerPointer(_pointer_config("autoregressive_pressure_v2"))
    context = build_worker_pressure_context(
        **_pressure_inputs(), temperature=1.0, supply_epsilon=1.0e-6
    )
    with pytest.raises(AssertionError, match="global_context"):
        head.build_v2_decode_cache(
            task_emb=torch.randn(1, 8),
            station_emb=torch.randn(1, 8),
            global_context=torch.randn(1, 32),
            worker_embs=torch.randn(1, 3, 8),
            pressure_context=context,
            demand=torch.tensor([2.0]),
        )


def test_v2_dynamic_eft_features_rank_legal_workers_and_zero_masked() -> None:
    from models.worker_pointer_context import build_worker_eft_features

    head = WorkerPointer(_pointer_config("autoregressive_pressure_v2"))
    state = head.initialize_v2_state(batch_size=1, device=torch.device("cpu"))
    state = head.advance_v2_state(
        state,
        torch.zeros((1, 8)),
        torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]]),
        selected_wait=torch.tensor([0.5]),
        selected_capacity=torch.tensor([1.0]),
    )
    features = build_worker_eft_features(
        team_state=state,
        worker_wait=torch.tensor([[0.0, 2.0, 1.0]]),
        worker_capacity=torch.tensor([[1.0, 0.5, 2.0]]),
        station_wait=torch.tensor([1.0]),
        task_duration=torch.tensor([4.0]),
        demand=torch.tensor([2.0]),
        mask=torch.tensor([[False, False, True]]),
        clip=10.0,
    )

    assert features.shape == (1, 3, 2)
    torch.testing.assert_close(features[0, 0], torch.tensor([0.0, -1.0]))
    torch.testing.assert_close(
        features[0, 1], torch.tensor([0.6008772, 1.0]), atol=1.0e-6, rtol=0.0
    )
    torch.testing.assert_close(features[0, 2], torch.zeros(2))


def test_v2_dynamic_eft_ignores_virtual_station_rows_in_mixed_ppo_batch() -> None:
    from ppo_agent import PPOAgent

    cfg = _pointer_config("autoregressive_pressure_v2")
    cfg.worker_pointer_v2_dynamic_eft_features = True
    agent = object.__new__(PPOAgent)
    agent.config = cfg
    agent.device = torch.device("cpu")
    head = WorkerPointer(cfg)
    team_state = head.initialize_v2_state(batch_size=2, device=torch.device("cpu"))
    worker_features = torch.zeros((2, 3, 17))
    worker_features[:, :, 0] = 1.0
    worker_features[:, :, 16] = 1.0
    features = agent._build_v2_dynamic_eft_features(
        team_state=team_state,
        worker_features=worker_features,
        station_features=torch.zeros((2, 5, 15)),
        selected_station=torch.tensor([-1, 4]),
        task_duration=torch.tensor([0.0, 4.0]),
        demand=torch.tensor([0.0, 1.0]),
        mask=torch.zeros((2, 3), dtype=torch.bool),
    )

    assert features is not None
    torch.testing.assert_close(features[0], torch.zeros_like(features[0]))
    assert torch.isfinite(features[1]).all()


def test_v2_dynamic_eft_late_fusion_is_zero_initialized_and_trainable() -> None:
    from models.worker_pointer_context import build_worker_pressure_context

    cfg = _pointer_config("autoregressive_pressure_v2")
    cfg.worker_pointer_v2_dynamic_eft_features = True
    head = WorkerPointer(cfg)
    context = build_worker_pressure_context(
        **_pressure_inputs(), temperature=1.0, supply_epsilon=1.0e-6
    )
    state = head.initialize_v2_state(batch_size=1, device=torch.device("cpu"))
    kwargs = {
        "task_emb": torch.randn(1, 8),
        "station_emb": torch.randn(1, 8),
        "global_context": torch.randn(1, 24),
        "worker_embs": torch.randn(1, 3, 8),
        "pressure_context": context,
        "team_state": state,
        "demand": torch.tensor([2.0]),
        "mask": torch.tensor([[False, False, True]]),
    }
    dynamic_eft = torch.tensor([[[0.0, -1.0], [1.0, 1.0], [0.0, 0.0]]])

    baseline = head.forward_choice_v2(**kwargs)
    fused = head.forward_choice_v2(**kwargs, dynamic_eft_features=dynamic_eft)

    torch.testing.assert_close(fused, baseline, atol=0.0, rtol=0.0)
    fused[:, :2].sum().backward()
    assert head.v2_eft_proj.weight.grad is not None
    assert torch.isfinite(head.v2_eft_proj.weight.grad).all()


def test_v2_decode_cache_preserves_logits_and_projection_gradients() -> None:
    from models.worker_pointer_context import build_worker_pressure_context

    torch.manual_seed(314)
    head = WorkerPointer(_pointer_config("autoregressive_pressure_v2"))
    context = build_worker_pressure_context(
        **_pressure_inputs(), temperature=1.0, supply_epsilon=1.0e-6
    )
    task_emb = torch.randn(1, 8)
    station_emb = torch.randn(1, 8)
    global_context = torch.randn(1, 24)
    worker_embs = torch.randn(1, 3, 8)
    demand = torch.tensor([2.0])
    mask = torch.tensor([[False, True, False]])
    state = head.initialize_v2_state(batch_size=1, device=torch.device("cpu"))
    state = head.advance_v2_state(
        state,
        worker_embs[:, 0],
        torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]]),
    )

    uncached = head.forward_choice_v2(
        task_emb=task_emb,
        station_emb=station_emb,
        global_context=global_context,
        worker_embs=worker_embs,
        pressure_context=context,
        team_state=state,
        demand=demand,
        mask=mask,
    )
    cache = head.build_v2_decode_cache(
        task_emb=task_emb,
        station_emb=station_emb,
        global_context=global_context,
        worker_embs=worker_embs,
        pressure_context=context,
        demand=demand,
    )
    cached = head.forward_choice_v2(
        task_emb=task_emb,
        station_emb=station_emb,
        global_context=global_context,
        worker_embs=worker_embs,
        pressure_context=context,
        team_state=state,
        demand=demand,
        mask=mask,
        decode_cache=cache,
    )

    torch.testing.assert_close(cached, uncached, atol=1.0e-6, rtol=0.0)

    uncached.sum().backward(retain_graph=True)
    uncached_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in head.named_parameters()
        if name.startswith("v2_") and parameter.grad is not None
    }
    head.zero_grad(set_to_none=True)
    cached.sum().backward()
    cached_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in head.named_parameters()
        if name.startswith("v2_") and parameter.grad is not None
    }
    assert uncached_grads.keys() == cached_grads.keys()
    for name in uncached_grads:
        torch.testing.assert_close(cached_grads[name], uncached_grads[name])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 验证 bf16 数值合同")
def test_v2_pointer_logits_stay_float32_inside_bf16_autocast() -> None:
    from models.worker_pointer_context import build_worker_pressure_context

    device = torch.device("cuda")
    head = WorkerPointer(_pointer_config("autoregressive_pressure_v2")).to(device)
    context = build_worker_pressure_context(
        **{key: value.to(device) for key, value in _pressure_inputs().items()},
        temperature=1.0,
        supply_epsilon=1.0e-6,
    )
    state = head.initialize_v2_state(batch_size=1, device=device)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = head.forward_choice_v2(
            task_emb=torch.randn(1, 8, device=device),
            station_emb=torch.randn(1, 8, device=device),
            global_context=torch.randn(1, 24, device=device),
            worker_embs=torch.randn(1, 3, 8, device=device),
            pressure_context=context,
            team_state=state,
            demand=torch.tensor([2.0], device=device),
            mask=torch.tensor([[False, False, True]], device=device),
        )

    assert logits.dtype == torch.float32
    assert logits[0, 2].item() == pytest.approx(-1.0e4)


def test_v2_initialization_does_not_advance_legacy_rng_or_change_legacy_keys() -> None:
    torch.manual_seed(123)
    legacy = WorkerPointer(_pointer_config("autoregressive"))
    legacy_rng = torch.get_rng_state().clone()
    legacy_keys = tuple(legacy.state_dict())

    torch.manual_seed(123)
    v2 = WorkerPointer(_pointer_config("autoregressive_pressure_v2"))
    v2_rng = torch.get_rng_state().clone()

    assert torch.equal(legacy_rng, v2_rng)
    assert not any(key.startswith("v2_") for key in legacy_keys)
    assert any(key.startswith("v2_") for key in v2.state_dict())
    assert not any(key.startswith("v2_eft_proj") for key in v2.state_dict())

    guided_cfg = _pointer_config("autoregressive_pressure_v2")
    guided_cfg.worker_pointer_v2_dynamic_eft_features = True
    guided = WorkerPointer(guided_cfg)
    assert any(key.startswith("v2_eft_proj") for key in guided.state_dict())


def test_v2_dynamic_eft_does_not_change_existing_v2_initialization() -> None:
    baseline = WorkerPointer(_pointer_config("autoregressive_pressure_v2"))
    guided_cfg = _pointer_config("autoregressive_pressure_v2")
    guided_cfg.worker_pointer_v2_dynamic_eft_features = True
    guided = WorkerPointer(guided_cfg)

    baseline_state = baseline.state_dict()
    guided_state = guided.state_dict()
    for name, value in baseline_state.items():
        if name.startswith("v2_"):
            torch.testing.assert_close(guided_state[name], value, atol=0.0, rtol=0.0)


def test_v2_dynamic_eft_keeps_checkpoint_optimizer_parameter_order() -> None:
    guided_cfg = _pointer_config("autoregressive_pressure_v2")
    guided_cfg.worker_pointer_v2_dynamic_eft_features = True
    guided = WorkerPointer(guided_cfg)

    parameter_names = [name for name, _parameter in guided.named_parameters()]

    assert parameter_names.index("v2_attn.weight") < parameter_names.index("v2_eft_proj.weight")


def _v2_agent_and_ready_env(
    device: torch.device | None = None,
) -> tuple[object, object, tuple[torch.Tensor, ...]]:
    from environment import AirLineEnv_Graph
    from models.hb_gat_pn import HBGATPN
    from ppo_agent import PPOAgent
    from tests.test_joint_experiment_architecture import (
        DATA_PATH,
        _advance_to_ready_physical_task,
    )

    env = AirLineEnv_Graph(DATA_PATH, seed=42)
    obs, masks = _advance_to_ready_physical_task(env)
    agent = PPOAgent(
        HBGATPN(__import__("configs").configs),
        lr=1.0e-4,
        gamma=0.99,
        k_epochs=1,
        eps_clip=0.2,
        device=device or torch.device("cpu"),
        batch_size=1,
        total_timesteps=1,
        config=__import__("configs").configs,
    )
    return agent, env, (obs, *masks)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 验证批量路径原始特征复用")
def test_v2_batch_path_passes_uploaded_raw_features_to_pressure_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from configs import configs
    from tests.test_joint_experiment_architecture import _small_overrides

    devices: list[tuple[torch.device, torch.device]] = []
    original = __import__("ppo_agent").PPOAgent._build_v2_pressure_context

    def spy(self: object, **kwargs: object) -> object:
        devices.append(
            (kwargs["task_features"].device, kwargs["worker_features"].device)
        )
        return original(self, **kwargs)

    monkeypatch.setattr(
        __import__("ppo_agent").PPOAgent, "_build_v2_pressure_context", spy
    )
    overrides = _small_overrides(
        team_selection_mode="autoregressive_pressure_v2",
        policy_action_scope="operation_station_worker",
        actor_context_mode="attention",
        batch_size=1,
        lightning_precision="bf16-mixed",
    )
    with temporary_config(configs, overrides):
        agent, _env, prepared = _v2_agent_and_ready_env(torch.device("cuda"))
        obs, task_mask, station_mask, worker_mask = prepared
        results = agent.select_actions_batch(
            [obs], [task_mask], [station_mask], [worker_mask], deterministic=False
        )

    assert results[0][0] is not None and not results[0][4]
    assert devices and all(
        task_device.type == worker_device.type == "cuda"
        for task_device, worker_device in devices
    )


@pytest.mark.parametrize("deterministic", [False, True])
def test_v2_single_and_deterministic_paths_call_v2_pointer(
    monkeypatch: pytest.MonkeyPatch,
    deterministic: bool,
) -> None:
    from configs import configs
    from tests.test_joint_experiment_architecture import _small_overrides

    calls: list[object] = []
    cache_build_calls: list[object] = []
    original = WorkerPointer.forward_choice_v2
    original_build_cache = WorkerPointer.build_v2_decode_cache

    def spy(self: WorkerPointer, **kwargs: object) -> torch.Tensor:
        calls.append(kwargs.get("decode_cache"))
        return original(self, **kwargs)

    def cache_spy(self: WorkerPointer, **kwargs: object) -> object:
        cache = original_build_cache(self, **kwargs)
        cache_build_calls.append(cache)
        return cache

    monkeypatch.setattr(WorkerPointer, "forward_choice_v2", spy)
    monkeypatch.setattr(WorkerPointer, "build_v2_decode_cache", cache_spy)
    overrides = _small_overrides(
        team_selection_mode="autoregressive_pressure_v2",
        policy_action_scope="operation_station_worker",
        actor_context_mode="attention",
        batch_size=1,
        worker_pointer_v2_behavior_replay=True,
        worker_pointer_v2_replay_mode="behavior_group_exact_v1",
        worker_pointer_v2_logical_batch_cap=1,
    )
    with temporary_config(configs, overrides):
        agent, _env, prepared = _v2_agent_and_ready_env()
        obs, task_mask, station_mask, worker_mask = prepared
        action, logprob, _value, _smask, invalid = agent.select_action(
            obs,
            mask_task=task_mask,
            mask_station_matrix=station_mask,
            mask_worker=worker_mask,
            deterministic=deterministic,
            temperature=1.0,
        )
    assert action is not None and not invalid
    assert torch.isfinite(torch.tensor(logprob))
    assert calls and all(cache is not None for cache in calls)
    assert len(cache_build_calls) == 1


def test_v2_batch_path_calls_v2_pointer(monkeypatch: pytest.MonkeyPatch) -> None:
    from configs import configs
    from tests.test_joint_experiment_architecture import _small_overrides

    calls: list[object] = []
    cache_build_calls: list[object] = []
    original = WorkerPointer.forward_choice_v2
    original_build_cache = WorkerPointer.build_v2_decode_cache

    def spy(self: WorkerPointer, **kwargs: object) -> torch.Tensor:
        calls.append(kwargs.get("decode_cache"))
        return original(self, **kwargs)

    def cache_spy(self: WorkerPointer, **kwargs: object) -> object:
        cache = original_build_cache(self, **kwargs)
        cache_build_calls.append(cache)
        return cache

    monkeypatch.setattr(WorkerPointer, "forward_choice_v2", spy)
    monkeypatch.setattr(WorkerPointer, "build_v2_decode_cache", cache_spy)
    overrides = _small_overrides(
        team_selection_mode="autoregressive_pressure_v2",
        policy_action_scope="operation_station_worker",
        actor_context_mode="attention",
        batch_size=1,
        worker_pointer_v2_behavior_replay=True,
        worker_pointer_v2_replay_mode="behavior_group_exact_v1",
        worker_pointer_v2_logical_batch_cap=1,
    )
    with temporary_config(configs, overrides):
        agent, _env, prepared = _v2_agent_and_ready_env()
        obs, task_mask, station_mask, worker_mask = prepared
        results = agent.select_actions_batch(
            [obs], [task_mask], [station_mask], [worker_mask], deterministic=False
        )
    assert results[0][0] is not None and not results[0][4]
    assert calls and all(cache is not None for cache in calls)
    assert len(cache_build_calls) == 1


def test_v2_ppo_recompute_calls_v2_pointer(monkeypatch: pytest.MonkeyPatch) -> None:
    from configs import configs
    from tests.test_joint_experiment_architecture import _small_overrides
    from training.memory import Memory

    overrides = _small_overrides(
        team_selection_mode="autoregressive_pressure_v2",
        policy_action_scope="operation_station_worker",
        actor_context_mode="attention",
        batch_size=1,
        worker_pointer_v2_behavior_replay=True,
        worker_pointer_v2_replay_mode="behavior_group_exact_v1",
        worker_pointer_v2_logical_batch_cap=1,
    )
    with temporary_config(configs, overrides):
        agent, env, prepared = _v2_agent_and_ready_env()
        obs, task_mask, station_mask, worker_mask = prepared
        action, logprob, value, _smask, invalid = agent.select_actions_batch(
            [obs],
            [task_mask],
            [station_mask],
            [worker_mask],
            deterministic=False,
        )[0]
        assert action is not None and not invalid
        memory = Memory()
        memory.states.append(env.get_state_snapshot())
        memory.actions.append(action)
        memory.logprobs.append(logprob)
        memory.values.append(value)
        memory.masks.append((task_mask, station_mask, worker_mask))
        from training.worker_pointer_v2_behavior import make_behavior_traces

        behavior_traces = make_behavior_traces(
            group_id=(0, 0),
            env_indices=[0],
            behavior_logprobs=agent.last_v2_behavior_logprobs,
        )
        assert len(behavior_traces) == 1
        memory.worker_pointer_v2_behavior_traces.append(behavior_traces[0])
        _obs, reward, done, info = env.step(action)
        assert not info.get("invalid_action", False)
        memory.rewards.append(float(reward))
        memory.is_terminals.append(bool(done))

        calls: list[object] = []
        cache_build_calls: list[object] = []
        original = WorkerPointer.forward_choice_v2
        original_build_cache = WorkerPointer.build_v2_decode_cache

        def spy(self: WorkerPointer, **kwargs: object) -> torch.Tensor:
            calls.append(kwargs.get("decode_cache"))
            return original(self, **kwargs)

        def cache_spy(self: WorkerPointer, **kwargs: object) -> object:
            cache = original_build_cache(self, **kwargs)
            cache_build_calls.append(cache)
            return cache

        monkeypatch.setattr(WorkerPointer, "forward_choice_v2", spy)
        monkeypatch.setattr(WorkerPointer, "build_v2_decode_cache", cache_spy)
        optimizer_parameter_ids = {
            id(parameter)
            for group in agent.optimizer.param_groups
            for parameter in group["params"]
        }
        v2_parameters = [
            parameter
            for name, parameter in agent.policy.named_parameters()
            if name.startswith("worker_head.v2_")
        ]
        assert v2_parameters
        assert all(id(parameter) in optimizer_parameter_ids for parameter in v2_parameters)
        metrics = agent.update(memory, env, current_ep=1)
    assert torch.isfinite(torch.tensor(metrics["PPO/Loss"]))
    assert torch.isfinite(torch.tensor(metrics["Gradient/V2Norm"]))
    assert metrics["Gradient/V2Norm"] > 0.0
    assert metrics["Gradient/V2Coverage"] > 0.0
    assert metrics["V2/FirstContractTotalMaxAE"] <= 1.0e-4
    assert calls and all(cache is not None for cache in calls)
    assert len(cache_build_calls) >= 3


def test_v2_dynamic_eft_features_are_forwarded_to_rollout_and_recompute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from configs import configs
    from tests.test_joint_experiment_architecture import _small_overrides
    from training.memory import Memory
    from training.worker_pointer_v2_behavior import make_behavior_traces

    forwarded: list[torch.Tensor | None] = []
    original = WorkerPointer.forward_choice_v2

    def spy(self: WorkerPointer, **kwargs: object) -> torch.Tensor:
        forwarded.append(kwargs.get("dynamic_eft_features"))
        return original(self, **kwargs)

    monkeypatch.setattr(WorkerPointer, "forward_choice_v2", spy)
    overrides = _small_overrides(
        team_selection_mode="autoregressive_pressure_v2",
        policy_action_scope="operation_station_worker",
        actor_context_mode="attention",
        batch_size=1,
        worker_pointer_v2_behavior_replay=True,
        worker_pointer_v2_replay_mode="behavior_group_exact_v1",
        worker_pointer_v2_logical_batch_cap=1,
        worker_pointer_v2_dynamic_eft_features=True,
    )
    with temporary_config(configs, overrides):
        agent, env, prepared = _v2_agent_and_ready_env()
        obs, task_mask, station_mask, worker_mask = prepared
        action, logprob, value, _smask, invalid = agent.select_actions_batch(
            [obs], [task_mask], [station_mask], [worker_mask], deterministic=False
        )[0]
        assert action is not None and not invalid
        memory = Memory()
        memory.states.append(env.get_state_snapshot())
        memory.actions.append(action)
        memory.logprobs.append(logprob)
        memory.values.append(value)
        memory.masks.append((task_mask, station_mask, worker_mask))
        traces = make_behavior_traces(
            group_id=(0, 0), env_indices=[0], behavior_logprobs=agent.last_v2_behavior_logprobs
        )
        memory.worker_pointer_v2_behavior_traces.append(traces[0])
        _obs, reward, done, info = env.step(action)
        assert not info.get("invalid_action", False)
        memory.rewards.append(float(reward))
        memory.is_terminals.append(bool(done))
        metrics = agent.update(memory, env, current_ep=1)

    assert torch.isfinite(torch.tensor(metrics["PPO/Loss"]))
    assert forwarded and all(features is not None for features in forwarded)
    assert all(features.shape[-1] == 2 and torch.isfinite(features).all() for features in forwarded)


def test_v2_dynamic_eft_features_are_forwarded_to_batched_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from configs import configs
    from environment import AirLineEnv_Graph
    from models.hb_gat_pn import HBGATPN
    from ppo_agent import PPOAgent
    from tests.test_joint_experiment_architecture import DATA_PATH, _advance_to_ready_physical_task, _small_overrides
    from training.memory import Memory

    forwarded: list[torch.Tensor | None] = []
    normalization_masks: list[torch.Tensor] = []
    original = WorkerPointer.forward_choice_v2
    original_normalized_entropy = PPOAgent._normalized_categorical_entropy

    def spy(self: WorkerPointer, **kwargs: object) -> torch.Tensor:
        forwarded.append(kwargs.get("dynamic_eft_features"))
        return original(self, **kwargs)

    def normalization_spy(
        entropy: torch.Tensor, invalid_mask: torch.Tensor
    ) -> torch.Tensor:
        normalization_masks.append(invalid_mask.detach().cpu())
        return original_normalized_entropy(entropy, invalid_mask)

    monkeypatch.setattr(WorkerPointer, "forward_choice_v2", spy)
    monkeypatch.setattr(
        PPOAgent,
        "_normalized_categorical_entropy",
        staticmethod(normalization_spy),
    )
    overrides = _small_overrides(
        team_selection_mode="autoregressive_pressure_v2",
        policy_action_scope="operation_station_worker",
        actor_context_mode="attention",
        batch_size=2,
        accumulation_steps=1,
        worker_pointer_v2_behavior_replay=False,
        worker_pointer_v2_replay_mode="batched_vectorized_v2",
        worker_pointer_v2_dynamic_eft_features=True,
    )
    with temporary_config(configs, overrides):
        environments = [AirLineEnv_Graph(DATA_PATH, seed=seed) for seed in (42, 43)]
        prepared = [_advance_to_ready_physical_task(env) for env in environments]
        agent = PPOAgent(
            HBGATPN(configs), lr=1.0e-4, gamma=0.99, k_epochs=1, eps_clip=0.2,
            device=torch.device("cpu"), batch_size=2, total_timesteps=2, config=configs,
        )
        results = agent.select_actions_batch(
            [item[0] for item in prepared],
            [item[1][0] for item in prepared],
            [item[1][1] for item in prepared],
            [item[1][2] for item in prepared],
            deterministic=False,
        )
        memory = Memory()
        for env, item, result in zip(environments, prepared, results):
            action, logprob, value, _station_mask, invalid = result
            assert action is not None and not invalid
            if isinstance(logprob, (list, tuple)):
                logprob = logprob[0]
            if isinstance(value, (list, tuple)):
                value = value[0]
            task_mask, station_mask, worker_mask = item[1]
            memory.states.append(env.get_state_snapshot())
            memory.actions.append(action)
            memory.logprobs.append(float(logprob))
            memory.values.append(float(value))
            memory.masks.append((task_mask, station_mask, worker_mask))
            memory.rewards.append(0.0)
            memory.is_terminals.append(True)
            memory.is_truncated.append(False)
        metrics = agent.update(memory, env=environments[0], current_ep=1)

    assert metrics["PPO/GradientsFinite"] == 1.0
    assert forwarded and all(features is not None for features in forwarded)
    assert all(features.shape[-1] == 2 and torch.isfinite(features).all() for features in forwarded)
    assert len(normalization_masks) >= 3
    assert any(mask.shape == (2, 80) for mask in normalization_masks)


@pytest.mark.parametrize(
    ("precision", "device_type", "enabled", "dtype", "use_scaler"),
    [
        ("16-mixed", "cuda", True, torch.float16, True),
        ("bf16-mixed", "cuda", True, torch.bfloat16, False),
        ("32-true", "cuda", False, None, False),
        ("bf16-mixed", "cpu", False, None, False),
    ],
)
def test_amp_settings_follow_lightning_precision(
    precision: str,
    device_type: str,
    enabled: bool,
    dtype: torch.dtype | None,
    use_scaler: bool,
) -> None:
    from ppo_agent import PPOAgent

    settings = PPOAgent.resolve_amp_settings(precision, device_type)
    assert settings == (enabled, dtype, use_scaler)


def test_v2_experiment_config_is_isolated_and_explicit() -> None:
    from configs import load_config_files

    root = Path(__file__).resolve().parents[1]
    cfg = Config()
    load_config_files(
        [root / "conf" / "experiment" / "initial_worker_pointer_v2_exploratory.yaml"],
        target=cfg,
    )
    assert cfg.experiment_name == "initial_worker_pointer_v2_exploratory"
    assert cfg.runs_root == "results/01_initial_main"
    assert cfg.team_selection_mode == "autoregressive_pressure_v2"
    assert cfg.policy_action_scope == "operation_station_worker"
    assert cfg.actor_context_mode == "attention"
    assert cfg.lightning_precision == "bf16-mixed"
    assert cfg.num_envs == 4
    assert cfg.batch_size == 256
    assert cfg.accumulation_steps == 16
    assert cfg.worker_pointer_v2_behavior_replay is True
    assert cfg.worker_pointer_v2_replay_mode == "behavior_group_exact_v1"
    assert cfg.worker_pointer_v2_logical_batch_cap == 256
    assert cfg.worker_pointer_v2_rollout_group_upper_bound == 4
    assert cfg.evaluation_protocol == "training_auto_eval_only"


def test_run_manifest_records_model_and_runtime_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from runtime.artifacts import build_run_manifest_payload
    from runtime.seed import set_seed

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_seed(42)
    cfg = _pointer_config("autoregressive_pressure_v2")
    cfg.experiment_name = "initial_worker_pointer_v2_exploratory"
    cfg.evaluation_protocol = "training_auto_eval_only"
    cfg.lightning_precision = "bf16-mixed"
    cfg.num_envs = 4
    cfg.batch_size = 256
    cfg.accumulation_steps = 16
    cfg.worker_pointer_v2_behavior_replay = True
    cfg.worker_pointer_v2_replay_mode = "behavior_group_exact_v1"
    payload = build_run_manifest_payload(cfg, command="pytest")

    assert payload["model_spec"]["team_selection_mode"] == "autoregressive_pressure_v2"
    assert payload["evaluation_protocol"] == "training_auto_eval_only"
    assert payload["runtime"]["num_envs"] == 4
    assert payload["runtime"]["autocast_dtype"] == "bfloat16"
    assert payload["runtime"]["grad_scaler_enabled"] is False
    assert payload["runtime"]["worker_pointer_v2_replay_mode"] == "behavior_group_exact_v1"
    assert payload["runtime"]["requested_logical_batch_cap"] == 256
    assert payload["runtime"]["effective_logical_batch_cap"] == 256
    assert payload["runtime"]["rollout_group_upper_bound"] == 4
    assert payload["runtime"]["target_max_samples_per_optimizer_step"] == 4096
    assert payload["runtime"]["cublas_workspace_config"] == ":4096:8"
    assert payload["runtime"]["deterministic_algorithms_warn_only"] is True


def test_fast_exact_run_manifest_records_v2_runtime_semantics() -> None:
    from runtime.artifacts import build_run_manifest_payload

    cfg = _pointer_config("autoregressive_pressure_v2_fast_exact")
    cfg.worker_pointer_v2_replay_mode = "behavior_group_exact_gpu_template_v2"
    cfg.worker_pointer_v2_rollout_group_upper_bound = 16
    cfg.batch_size = 256
    cfg.accumulation_steps = 16
    cfg.num_envs = 16
    cfg.async_eval_enabled = True
    cfg.async_eval_device = "cuda"
    cfg.async_eval_worker_count = 2
    cfg.async_eval_queue_capacity = 4
    cfg.async_eval_submit_every_episodes = 2
    cfg.async_eval_wait_on_finish = True

    payload = build_run_manifest_payload(cfg, command="pytest")

    assert payload["runtime"]["worker_pointer_v2_replay_mode"] == (
        "behavior_group_exact_gpu_template_v2"
    )
    assert payload["runtime"]["requested_logical_batch_cap"] == 256
    assert payload["runtime"]["effective_logical_batch_cap"] == 256
    assert payload["runtime"]["rollout_group_upper_bound"] == 16
    assert payload["runtime"]["target_max_samples_per_optimizer_step"] == 4096
    assert payload["runtime"]["worker_pointer_v2_init_seed"] == 1051
    assert payload["runtime"]["async_eval_enabled"] is True
    assert payload["runtime"]["async_eval_device"] == "cuda"
    assert payload["runtime"]["async_eval_worker_count"] == 2
    assert payload["runtime"]["async_eval_queue_capacity"] == 4
    assert payload["runtime"]["async_eval_submit_every_episodes"] == 2
    assert payload["runtime"]["async_eval_wait_on_finish"] is True


def test_training_entry_sets_cublas_workspace_before_framework_import() -> None:
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; import train; print(os.environ.get('CUBLAS_WORKSPACE_CONFIG'))",
        ],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip().splitlines()[-1] == ":4096:8"


def test_v2_diagnostics_compute_exact_quantiles_and_clear() -> None:
    from models.worker_pointer_context import WorkerPressureContext
    from training.worker_pointer_v2_diagnostics import WorkerPointerV2Diagnostics

    values = torch.arange(1, 11, dtype=torch.float32).reshape(2, 5)
    context = WorkerPressureContext(
        demand_all=torch.ones((2, 5)),
        demand_near=torch.ones((2, 5)),
        supply_all=torch.ones((2, 5)),
        supply_near=torch.ones((2, 5)),
        pressure_all=values,
        pressure_near=values * 2.0,
        zero_supply_all=torch.zeros((2, 5), dtype=torch.bool),
        zero_supply_near=torch.zeros((2, 5), dtype=torch.bool),
        candidate_exposure=torch.ones((2, 3, 10)),
        candidate_max_exposure=torch.ones((2, 3, 2)),
    )
    diagnostics = WorkerPointerV2Diagnostics(num_skills=5)
    diagnostics.record_context(context, host_elapsed_ms=2.5)
    diagnostics.record_selection(
        selected_exposure=torch.full((2, 12), 3.0),
        entropy=torch.tensor([0.5, 1.5]),
    )
    diagnostics.record_team(torch.full((2, 5), 0.25))

    metrics = diagnostics.finalize(require_coverage=True)
    assert metrics["PointerV2/PressureAll/Skill0/P50"] == pytest.approx(3.5)
    assert metrics["PointerV2/PressureNear/Skill4/Max"] == pytest.approx(20.0)
    assert metrics["PointerV2/ContextHostMs"] == pytest.approx(2.5)
    assert metrics["PointerV2/WorkerEntropyMean"] == pytest.approx(1.0)
    assert metrics["PointerV2/SelectedExposureMean"] == pytest.approx(3.0)
    assert diagnostics.buffered_element_count == 0


def test_v2_coverage_gate_fails_for_near_demand_without_supply() -> None:
    from models.worker_pointer_context import WorkerPressureContext
    from training.worker_pointer_v2_diagnostics import WorkerPointerV2Diagnostics

    ones = torch.ones((1, 5))
    context = WorkerPressureContext(
        demand_all=ones,
        demand_near=ones,
        supply_all=ones,
        supply_near=torch.zeros((1, 5)),
        pressure_all=ones,
        pressure_near=ones,
        zero_supply_all=torch.zeros((1, 5), dtype=torch.bool),
        zero_supply_near=torch.ones((1, 5), dtype=torch.bool),
        candidate_exposure=torch.ones((1, 1, 10)),
        candidate_max_exposure=torch.ones((1, 1, 2)),
    )
    diagnostics = WorkerPointerV2Diagnostics(num_skills=5)
    diagnostics.record_context(context, host_elapsed_ms=0.1)
    with pytest.raises(RuntimeError, match="近期需求.*零有效供给"):
        diagnostics.finalize(require_coverage=True)
