from __future__ import annotations

import copy
from pathlib import Path

import torch

from configs import Config, configs
from core.action_completion import EarliestFinishActionCompleter
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from runtime.checkpoints import build_model_spec, infer_model_spec
from runtime.configuration import validate_runtime_config
from tests.runtime_safety import temporary_config
from training.memory import Memory
from utils.gpu_graph_manager import GPUBatchGraphManager
from worker_feature_layout import resolve_worker_feature_layout


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
        "policy_action_scope": "operation_station_gated_team",
    }
    values.update(extra)
    return values


def _first_physical_action_state(
    env: AirLineEnv_Graph,
) -> tuple[object, tuple[torch.Tensor, torch.Tensor, torch.Tensor], int]:
    obs = env.reset(seed=42)
    for _ in range(env.num_tasks):
        masks = env.get_masks()
        physical = [
            int(task_id)
            for task_id in torch.nonzero(~masks[0], as_tuple=False).reshape(-1).tolist()
            if int(env.task_static_feat[int(task_id), 1].item()) >= 0
        ]
        if physical:
            task_id = min(physical)
            task_mask = torch.ones_like(masks[0])
            task_mask[task_id] = False
            return obs, (task_mask, masks[1], masks[2]), task_id
        ready = torch.nonzero(~masks[0], as_tuple=False).reshape(-1).tolist()
        assert ready
        obs, _reward, done, info = env.step((min(ready), -1, []))
        assert not done and info.get("virtual_task", False)
    raise AssertionError("未找到可调度的 APAL 物理工序")


def _first_legal_station(station_mask: torch.Tensor, task_id: int) -> int:
    available = torch.nonzero(~station_mask[task_id], as_tuple=False).reshape(-1)
    assert available.numel() > 0
    return int(available[0].item())


def test_team_candidates_are_deterministic_legal_and_anchor_old_completer() -> None:
    with temporary_config(configs, _small_overrides()):
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
        obs, masks, task_id = _first_physical_action_state(env)
        station_id = _first_legal_station(masks[1], task_id)
        completer = EarliestFinishActionCompleter(configs)
        base = completer.complete(
            obs,
            task_id=task_id,
            station_mask=masks[1][task_id],
            worker_mask=masks[2],
            selected_station=station_id,
        )
        first = completer.enumerate_team_candidates(
            obs,
            task_id=task_id,
            station_id=station_id,
            worker_mask=masks[2],
        )
        second = completer.enumerate_team_candidates(
            obs,
            task_id=task_id,
            station_id=station_id,
            worker_mask=masks[2],
        )
        assert base is not None and first is not None and second is not None
        assert first.teams == second.teams
        torch.testing.assert_close(first.gate_features, second.gate_features)
        assert first.teams[0] == base.team
        assert 1 <= len(first.teams) <= configs.conditional_team_max_candidates

        layout = resolve_worker_feature_layout(configs)
        demand = int(env.task_static_feat[task_id, 2].item())
        required_skill = int(env.task_static_feat[task_id, 1].item())
        locks = torch.argmax(obs["worker"].x[:, layout.lock_slice], dim=1)
        for team in first.teams:
            assert len(team) == demand
            assert len(set(team)) == demand
            for worker_id in team:
                assert not bool(masks[2][worker_id].item())
                assert bool(obs["worker"].x[worker_id, layout.skill_slice][required_skill].item() > 0.5)
                assert int(locks[worker_id].item()) in {0, station_id + 1}


def test_team_candidate_generator_degrades_to_single_base_team() -> None:
    with temporary_config(configs, _small_overrides()):
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
        obs, masks, task_id = _first_physical_action_state(env)
        station_id = _first_legal_station(masks[1], task_id)
        completer = EarliestFinishActionCompleter(configs)
        base = completer.complete(
            obs,
            task_id=task_id,
            station_mask=masks[1][task_id],
            worker_mask=masks[2],
            selected_station=station_id,
        )
        assert base is not None
        restricted_mask = torch.ones_like(masks[2], dtype=torch.bool)
        restricted_mask[list(base.team)] = False
        candidates = completer.enumerate_team_candidates(
            obs,
            task_id=task_id,
            station_id=station_id,
            worker_mask=restricted_mask,
        )
        assert candidates is not None
        assert candidates.teams == (base.team,)


def test_zero_initialized_gate_matches_operation_station_at_temperature_zero() -> None:
    overrides = _small_overrides()
    with temporary_config(configs, overrides):
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
        obs, masks, _task_id = _first_physical_action_state(env)

        base_config = copy.deepcopy(configs)
        base_config.policy_action_scope = "operation_station"
        gated_config = copy.deepcopy(configs)
        gated_config.policy_action_scope = "operation_station_gated_team"
        torch.manual_seed(1234)
        base_model = HBGATPN(base_config)
        torch.manual_seed(1234)
        gated_model = HBGATPN(gated_config)
        gated_model.load_state_dict(base_model.state_dict(), strict=False)
        base_agent = PPOAgent(
            base_model, 1.0e-4, 0.99, 1, 0.2, torch.device("cpu"), 1, 1, base_config
        )
        gated_agent = PPOAgent(
            gated_model, 1.0e-4, 0.99, 1, 0.2, torch.device("cpu"), 1, 1, gated_config
        )
        base_action, *_base_rest = base_agent.select_action(
            obs, masks[0], masks[1], masks[2], deterministic=True, temperature=0.0
        )
        gated_action, *_gated_rest = gated_agent.select_action(
            obs, masks[0], masks[1], masks[2], deterministic=True, temperature=0.0
        )
        assert gated_action == base_action


def test_gated_scope_configuration_and_checkpoint_spec_are_explicit() -> None:
    cfg = Config()
    cfg.policy_action_scope = "operation_station_gated_team"
    cfg.conditional_team_max_candidates = 4
    cfg.conditional_team_gate_bias = -4.0
    cfg.conditional_team_nonbaseline_logit = -8.0
    validate_runtime_config(cfg)
    spec = build_model_spec(cfg)
    assert spec.policy_action_scope == "operation_station_gated_team"
    assert spec.conditional_team_max_candidates == 4
    assert spec.conditional_team_gate_bias == -4.0
    assert spec.conditional_team_nonbaseline_logit == -8.0
    inferred = infer_model_spec(HBGATPN(cfg).state_dict())
    assert inferred.policy_action_scope == "operation_station_gated_team"


def test_gated_team_batch_decoding_preserves_apal_legality() -> None:
    with temporary_config(configs, _small_overrides()):
        envs = [AirLineEnv_Graph(DATA_PATH, seed=42) for _ in range(2)]
        prepared = [_first_physical_action_state(env) for env in envs]
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
            [item[0] for item in prepared],
            [item[1][0] for item in prepared],
            [item[1][1] for item in prepared],
            [item[1][2] for item in prepared],
            deterministic=False,
            temperature=1.0,
        )
        assert len(results) == 2
        for env, result in zip(envs, results, strict=True):
            action, logprob, _value, _station_mask, invalid = result
            assert action is not None and not invalid
            assert torch.isfinite(torch.tensor(logprob))
            _obs, _reward, _done, info = env.step(action)
            assert not info.get("invalid_action", False)


def test_snapshot_rebuild_preserves_each_vector_env_duration_perturbation() -> None:
    with temporary_config(configs, _small_overrides(randomize_durations=True)):
        primary_env = AirLineEnv_Graph(DATA_PATH, seed=42)
        secondary_env = AirLineEnv_Graph(DATA_PATH, seed=43)
        primary_env.reset(randomize_duration=True, seed=42)
        secondary_obs = secondary_env.reset(randomize_duration=True, seed=43)
        snapshot = secondary_env.get_state_snapshot()
        rebuilt = primary_env.rebuild_state_from_snapshot(snapshot)
        torch.testing.assert_close(
            rebuilt["task"].x[:, 0],
            secondary_obs["task"].x[:, 0],
        )


def test_gpu_batch_rebuild_preserves_each_snapshot_duration_perturbation() -> None:
    """GPU 批量重建必须保留每个 APAL 环境各自的任务工时底座。"""
    with temporary_config(configs, _small_overrides(randomize_durations=True)):
        first_env = AirLineEnv_Graph(DATA_PATH, seed=42)
        second_env = AirLineEnv_Graph(DATA_PATH, seed=43)
        first_env.reset(randomize_duration=True, seed=42)
        second_env.reset(randomize_duration=True, seed=43)
        snapshots = [first_env.get_state_snapshot(), second_env.get_state_snapshot()]
        manager = GPUBatchGraphManager(torch.device("cpu"), config=configs)
        batch = manager.batched_rebuild_on_gpu(snapshots, first_env)
        task_count = first_env.num_tasks
        for batch_index, snapshot in enumerate(snapshots):
            start = batch_index * task_count
            end = start + task_count
            torch.testing.assert_close(
                batch["task"].x[start:end, 0].cpu(),
                snapshot["base_task_x"][:, 0].cpu(),
            )


def test_gated_team_action_is_legal_in_single_batch_and_ppo_update() -> None:
    with temporary_config(configs, _small_overrides()):
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
        obs, masks, _task_id = _first_physical_action_state(env)
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
            masks[0],
            masks[1],
            masks[2],
            deterministic=False,
            temperature=1.0,
        )
        assert action is not None and not invalid and torch.isfinite(torch.tensor(logprob))
        snapshot = env.get_state_snapshot()
        _obs, reward, done, info = env.step(action)
        assert not info.get("invalid_action", False)

        memory = Memory()
        memory.states.append(snapshot)
        memory.actions.append(action)
        memory.logprobs.append(logprob)
        memory.values.append(value)
        memory.masks.append(masks)
        memory.rewards.append(float(reward))
        memory.is_terminals.append(bool(done))
        metrics = agent.update(memory, env, current_ep=1)
        assert torch.isfinite(torch.tensor(metrics["Loss/Total"]))
