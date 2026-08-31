from __future__ import annotations

from pathlib import Path

import pytest
import torch

from configs import Config, configs
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN, WorkerPointer
from ppo_agent import PPOAgent
from runtime.checkpoints import FORMAT_VERSION, build_model_spec
from runtime.configuration import validate_runtime_config
from tests.runtime_safety import temporary_config
from training.memory import Memory
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


def _advance_to_ready_physical_task(
    env: AirLineEnv_Graph,
) -> tuple[object, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    obs = env.reset(seed=42)
    for _ in range(env.num_tasks):
        masks = env.get_masks()
        ready = torch.nonzero(~masks[0], as_tuple=False).reshape(-1).tolist()
        physical = [
            int(task_id)
            for task_id in ready
            if int(env.task_static_feat[int(task_id), 1].item()) >= 0
        ]
        if physical:
            selected = min(physical)
            forced_task_mask = torch.ones_like(masks[0])
            forced_task_mask[selected] = False
            return obs, (forced_task_mask, masks[1], masks[2])
        assert ready, "推进虚拟节点时不应出现资源等待"
        obs, _reward, done, info = env.step((min(ready), -1, []))
        assert not done
        assert info.get("virtual_task", False)
    raise AssertionError("未能推进到首个可调度物理工序")


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
    monkeypatch: pytest.MonkeyPatch,
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
        if action_scope == "operation_station_worker":
            def fail_if_called(*_args, **_kwargs):
                raise AssertionError("Full 路径不得调用动作补全器")

            monkeypatch.setattr(agent.action_completer, "complete", fail_if_called)
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


@pytest.mark.parametrize(
    "action_scope",
    ("operation", "operation_station", "operation_station_worker"),
)
def test_batched_action_decoding_preserves_legality_for_every_scope(
    action_scope: str,
) -> None:
    overrides = _small_overrides(policy_action_scope=action_scope)
    with temporary_config(configs, overrides):
        envs = [AirLineEnv_Graph(DATA_PATH, seed=42) for _ in range(2)]
        prepared = [_advance_to_ready_physical_task(env) for env in envs]
        observations = [item[0] for item in prepared]
        masks = [item[1] for item in prepared]
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
        results = agent.select_actions_batch(
            observations,
            [item[0] for item in masks],
            [item[1] for item in masks],
            [item[2] for item in masks],
            deterministic=False,
            temperature=1.0,
        )
        assert len(results) == 2
        for env, result in zip(envs, results, strict=True):
            action, logprob, _value, _station_mask, invalid = result
            assert action is not None
            assert not invalid
            assert torch.isfinite(torch.tensor(logprob))
            _obs, _reward, _done, info = env.step(action)
            assert not info.get("invalid_action", False)


@pytest.mark.parametrize(
    "action_scope",
    (
        "operation",
        "operation_station",
        "operation_station_worker",
        "operation_station_anchor_proposal_team",
    ),
)
def test_ppo_update_is_finite_for_every_action_scope(action_scope: str) -> None:
    overrides = _small_overrides(policy_action_scope=action_scope, batch_size=1)
    with temporary_config(configs, overrides):
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
        obs, masks = _advance_to_ready_physical_task(env)
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
            obs,
            mask_task=masks[0],
            mask_station_matrix=masks[1],
            mask_worker=masks[2],
            deterministic=False,
            temperature=1.0,
        )
        assert action is not None and not invalid
        memory = Memory()
        memory.states.append(env.get_state_snapshot())
        memory.actions.append(action)
        memory.logprobs.append(logprob)
        memory.values.append(value)
        memory.masks.append(masks)
        if action_scope == "operation_station_anchor_proposal_team":
            memory.anchor_proposal_traces.append(agent.last_anchor_proposal_trace)
        _obs, reward, done, info = env.step(action)
        assert not info.get("invalid_action", False)
        memory.rewards.append(float(reward))
        memory.is_terminals.append(bool(done))
        metrics = agent.update(memory, env, current_ep=1)
        assert torch.isfinite(torch.tensor(metrics["Loss/Total"]))


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


def test_infeasible_preallocation_fails_before_rollout() -> None:
    overrides = _small_overrides(
        workforce_binding_mode="preallocated",
        workforce_preallocation_ratio=0.01,
    )
    with temporary_config(configs, overrides):
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
        with pytest.raises(RuntimeError, match="workforce_preallocation_infeasible"):
            env.reset(seed=42)


def test_actor_attention_parameters_remain_in_actor_optimizer_group() -> None:
    with temporary_config(configs, _small_overrides()):
        model = HBGATPN(configs)
        agent = PPOAgent(
            model,
            lr=1.0e-4,
            gamma=0.99,
            k_epochs=1,
            eps_clip=0.2,
            device=torch.device("cpu"),
            batch_size=1,
            total_timesteps=1,
            config=configs,
        )
        actor_ids = {id(parameter) for parameter in agent.actor_parameters}
        critic_ids = {id(parameter) for parameter in agent.critic_parameters}
        for name, parameter in model.named_parameters():
            if name.startswith("actor_"):
                assert id(parameter) in actor_ids
                assert id(parameter) not in critic_ids


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
