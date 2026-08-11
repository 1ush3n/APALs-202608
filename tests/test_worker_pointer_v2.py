from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

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


def test_runtime_validation_accepts_only_attention_worker_scope_for_v2() -> None:
    from runtime.configuration import validate_runtime_config

    valid = Config()
    valid.team_selection_mode = "autoregressive_pressure_v2"
    valid.policy_action_scope = "operation_station_worker"
    valid.actor_context_mode = "attention"
    validate_runtime_config(valid)

    wrong_scope = Config()
    wrong_scope.team_selection_mode = "autoregressive_pressure_v2"
    wrong_scope.policy_action_scope = "operation_station_anchor_proposal_team"
    wrong_scope.actor_context_mode = "attention"
    with pytest.raises(ValueError, match="operation_station_worker"):
        validate_runtime_config(wrong_scope)

    wrong_context = Config()
    wrong_context.team_selection_mode = "autoregressive_pressure_v2"
    wrong_context.policy_action_scope = "operation_station_worker"
    wrong_context.actor_context_mode = "mean_max"
    with pytest.raises(ValueError, match="actor_context_mode=attention"):
        validate_runtime_config(wrong_context)


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


def _v2_agent_and_ready_env() -> tuple[object, object, tuple[torch.Tensor, ...]]:
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
        device=torch.device("cpu"),
        batch_size=1,
        total_timesteps=1,
        config=__import__("configs").configs,
    )
    return agent, env, (obs, *masks)


@pytest.mark.parametrize("deterministic", [False, True])
def test_v2_single_and_deterministic_paths_call_v2_pointer(
    monkeypatch: pytest.MonkeyPatch,
    deterministic: bool,
) -> None:
    from configs import configs
    from tests.test_joint_experiment_architecture import _small_overrides

    calls: list[int] = []
    original = WorkerPointer.forward_choice_v2

    def spy(self: WorkerPointer, **kwargs: object) -> torch.Tensor:
        calls.append(1)
        return original(self, **kwargs)

    monkeypatch.setattr(WorkerPointer, "forward_choice_v2", spy)
    overrides = _small_overrides(
        team_selection_mode="autoregressive_pressure_v2",
        policy_action_scope="operation_station_worker",
        actor_context_mode="attention",
        batch_size=1,
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
    assert calls


def test_v2_batch_path_calls_v2_pointer(monkeypatch: pytest.MonkeyPatch) -> None:
    from configs import configs
    from tests.test_joint_experiment_architecture import _small_overrides

    calls: list[int] = []
    original = WorkerPointer.forward_choice_v2

    def spy(self: WorkerPointer, **kwargs: object) -> torch.Tensor:
        calls.append(1)
        return original(self, **kwargs)

    monkeypatch.setattr(WorkerPointer, "forward_choice_v2", spy)
    overrides = _small_overrides(
        team_selection_mode="autoregressive_pressure_v2",
        policy_action_scope="operation_station_worker",
        actor_context_mode="attention",
        batch_size=1,
    )
    with temporary_config(configs, overrides):
        agent, _env, prepared = _v2_agent_and_ready_env()
        obs, task_mask, station_mask, worker_mask = prepared
        results = agent.select_actions_batch(
            [obs], [task_mask], [station_mask], [worker_mask], deterministic=False
        )
    assert results[0][0] is not None and not results[0][4]
    assert calls


def test_v2_ppo_recompute_calls_v2_pointer(monkeypatch: pytest.MonkeyPatch) -> None:
    from configs import configs
    from tests.test_joint_experiment_architecture import _small_overrides
    from training.memory import Memory

    overrides = _small_overrides(
        team_selection_mode="autoregressive_pressure_v2",
        policy_action_scope="operation_station_worker",
        actor_context_mode="attention",
        batch_size=1,
    )
    with temporary_config(configs, overrides):
        agent, env, prepared = _v2_agent_and_ready_env()
        obs, task_mask, station_mask, worker_mask = prepared
        action, logprob, value, _smask, invalid = agent.select_action(
            obs,
            mask_task=task_mask,
            mask_station_matrix=station_mask,
            mask_worker=worker_mask,
            deterministic=False,
        )
        assert action is not None and not invalid
        memory = Memory()
        memory.states.append(env.get_state_snapshot())
        memory.actions.append(action)
        memory.logprobs.append(logprob)
        memory.values.append(value)
        memory.masks.append((task_mask, station_mask, worker_mask))
        _obs, reward, done, info = env.step(action)
        assert not info.get("invalid_action", False)
        memory.rewards.append(float(reward))
        memory.is_terminals.append(bool(done))

        calls: list[int] = []
        original = WorkerPointer.forward_choice_v2

        def spy(self: WorkerPointer, **kwargs: object) -> torch.Tensor:
            calls.append(1)
            return original(self, **kwargs)

        monkeypatch.setattr(WorkerPointer, "forward_choice_v2", spy)
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
    assert torch.isfinite(torch.tensor(metrics["Loss/Total"]))
    assert torch.isfinite(torch.tensor(metrics["PointerV2/GradientNorm"]))
    assert metrics["PointerV2/GradientNorm"] > 0.0
    assert metrics["PointerV2/GradientCoverage"] > 0.0
    assert metrics["PointerV2/PPOFirstRecomputeMaxAE"] <= 1.0e-4
    assert calls


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
    assert cfg.evaluation_protocol == "training_auto_eval_only"


def test_run_manifest_records_model_and_runtime_semantics() -> None:
    from runtime.artifacts import build_run_manifest_payload

    cfg = _pointer_config("autoregressive_pressure_v2")
    cfg.experiment_name = "initial_worker_pointer_v2_exploratory"
    cfg.evaluation_protocol = "training_auto_eval_only"
    cfg.lightning_precision = "bf16-mixed"
    cfg.num_envs = 4
    payload = build_run_manifest_payload(cfg, command="pytest")

    assert payload["model_spec"]["team_selection_mode"] == "autoregressive_pressure_v2"
    assert payload["evaluation_protocol"] == "training_auto_eval_only"
    assert payload["runtime"]["num_envs"] == 4
    assert payload["runtime"]["autocast_dtype"] == "bfloat16"
    assert payload["runtime"]["grad_scaler_enabled"] is False


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
