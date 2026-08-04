from __future__ import annotations

import copy
from pathlib import Path

import torch

from configs import Config, configs
from core.action_completion import EarliestFinishActionCompleter
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from runtime.checkpoints import build_checkpoint_metadata, build_model_spec, infer_model_spec
from runtime.configuration import validate_runtime_config
from runtime.initial_checkpoint_selection import load_initial_checkpoint_selection_manifest
from tests.runtime_safety import temporary_config
from training.best_anchor_teacher import BestAnchorTeacherManager
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
    cfg.async_eval_enabled = True
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


def test_best_anchor_switch_is_default_off_and_rejects_non_gated_scope() -> None:
    cfg = Config()
    validate_runtime_config(cfg)
    assert not cfg.best_anchor_distill_enabled

    cfg.best_anchor_distill_enabled = True
    with torch.no_grad():
        cfg.policy_action_scope = "operation_station"
    try:
        validate_runtime_config(cfg)
    except ValueError as exc:
        assert "operation_station_gated_team" in str(exc)
    else:
        raise AssertionError("best-anchor 蒸馏不应作用于非门控动作范围")


def test_best_anchor_external_teacher_is_strictly_verified(tmp_path: Path) -> None:
    cfg = Config()
    cfg.hidden_dim = 32
    cfg.num_gat_layers = 1
    cfg.num_heads = 2
    cfg.use_shared_trunk = True
    cfg.use_schedule_free = False
    cfg.policy_action_scope = "operation_station_gated_team"
    cfg.async_eval_enabled = True
    cfg.checkpoint_selection_protocol = "multiscale_manifest"
    cfg.checkpoint_selection_manifest_path = (
        "data/initial_selection_manifests/real_four_instances_temperature0_v1.json"
    )
    manifest = load_initial_checkpoint_selection_manifest(
        cfg.checkpoint_selection_manifest_path
    )
    checkpoint_path = tmp_path / "external_teacher.ckpt"
    teacher_model = HBGATPN(cfg)
    torch.save(
        {
            "state_dict": {
                f"policy.{name}": value.detach().clone()
                for name, value in teacher_model.state_dict().items()
            },
            "apal_metadata": build_checkpoint_metadata(cfg, episode=1),
        },
        checkpoint_path,
    )
    cfg.best_anchor_distill_enabled = True
    cfg.best_anchor_distill_external_checkpoint_path = str(checkpoint_path)
    cfg.best_anchor_distill_external_selection_score = 0.9
    cfg.best_anchor_distill_external_protocol_id = manifest.protocol_id
    cfg.best_anchor_distill_external_manifest_sha256 = manifest.sha256
    validate_runtime_config(cfg)
    manager = BestAnchorTeacherManager(
        config=cfg,
        device=torch.device("cpu"),
        model_factory=lambda: HBGATPN(copy.deepcopy(cfg)),
        checkpoint_dir=tmp_path / "run_checkpoints",
        make_schedulefree_optimizer=None,
        use_schedule_free=False,
    )
    assert manager.active
    assert manager.state is not None and manager.state.source == "external"
    first = manager.on_update_started()
    assert first["Distill/Enabled"] == 1.0
    assert manager.current_lambda() == cfg.best_anchor_distill_lambda_start

    cfg.best_anchor_distill_external_manifest_sha256 = "0" * 64
    try:
        BestAnchorTeacherManager(
            config=cfg,
            device=torch.device("cpu"),
            model_factory=lambda: HBGATPN(copy.deepcopy(cfg)),
            checkpoint_dir=tmp_path / "run_checkpoints",
            make_schedulefree_optimizer=None,
            use_schedule_free=False,
        )
    except ValueError as exc:
        assert "manifest" in str(exc)
    else:
        raise AssertionError("协议不一致的外部教师不应被加载")


def test_best_anchor_distillation_completes_two_ppo_updates(tmp_path: Path) -> None:
    """低资源 smoke：教师前向、KL 损失和两次 PPO 更新必须同时有限。"""
    manifest_path = "data/initial_selection_manifests/real_four_instances_temperature0_v1.json"
    manifest = load_initial_checkpoint_selection_manifest(manifest_path)
    checkpoint_path = tmp_path / "teacher.ckpt"
    overrides = _small_overrides(
        checkpoint_selection_protocol="multiscale_manifest",
        checkpoint_selection_manifest_path=manifest_path,
        async_eval_enabled=True,
        best_anchor_distill_enabled=True,
        best_anchor_distill_external_checkpoint_path=str(checkpoint_path),
        best_anchor_distill_external_selection_score=0.9,
        best_anchor_distill_external_protocol_id=manifest.protocol_id,
        best_anchor_distill_external_manifest_sha256=manifest.sha256,
    )
    with temporary_config(configs, overrides):
        teacher_model = HBGATPN(configs)
        torch.save(
            {
                "state_dict": {
                    f"policy.{name}": value.detach().clone()
                    for name, value in teacher_model.state_dict().items()
                },
                "apal_metadata": build_checkpoint_metadata(configs, episode=1),
            },
            checkpoint_path,
        )
        validate_runtime_config(configs)
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
        agent = PPOAgent(
            HBGATPN(configs),
            lr=1.0e-4,
            gamma=0.99,
            k_epochs=1,
            eps_clip=0.2,
            device=torch.device("cpu"),
            batch_size=1,
            total_timesteps=2,
            config=configs,
            teacher_model_factory=lambda: HBGATPN(copy.deepcopy(configs)),
            teacher_checkpoint_dir=tmp_path / "run_checkpoints",
        )
        for episode in (1, 2):
            obs, masks, _task_id = _first_physical_action_state(env)
            action, logprob, value, _station_mask, invalid = agent.select_action(
                obs,
                masks[0],
                masks[1],
                masks[2],
                deterministic=False,
                temperature=1.0,
            )
            assert action is not None and not invalid
            snapshot = env.get_state_snapshot()
            _obs, reward, done, info = env.step(action)
            assert not info.get("invalid_action", False)
            memory = Memory()
            memory.states.append(snapshot)
            memory.actions.append(action)
            memory.gated_team_traces.append(agent.last_gated_team_trace)
            memory.logprobs.append(logprob)
            memory.values.append(value)
            memory.masks.append(masks)
            memory.rewards.append(float(reward))
            memory.is_terminals.append(bool(done))
            metrics = agent.update(memory, env, current_ep=episode)
            assert metrics["Distill/Enabled"] == 1.0
            assert torch.isfinite(torch.tensor(metrics["Distill/KLTask"]))
            assert torch.isfinite(torch.tensor(metrics["Distill/KLStation"]))
            assert torch.isfinite(torch.tensor(metrics["Loss/Total"]))


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
        memory.gated_team_traces.append(agent.last_gated_team_trace)
        memory.logprobs.append(logprob)
        memory.values.append(value)
        memory.masks.append(masks)
        memory.rewards.append(float(reward))
        memory.is_terminals.append(bool(done))
        metrics = agent.update(memory, env, current_ep=1)
        assert torch.isfinite(torch.tensor(metrics["Loss/Total"]))


def test_gated_team_ppo_uses_frozen_candidates_when_rebuild_features_change() -> None:
    """PPO 必须沿用采样时的 APAL 团队动作空间，而非事后重枚举。"""
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
        assert action is not None and not invalid
        trace = agent.last_gated_team_trace
        assert trace is not None
        assert tuple(action[2]) == trace.teams[trace.selected_index]

        snapshot = env.get_state_snapshot()
        _obs, reward, done, info = env.step(action)
        assert not info.get("invalid_action", False)

        # 故意改变重建时的工人等待特征；旧实现可能改变候选 0，冻结轨迹不得受影响。
        for worker_id in action[2]:
            snapshot["worker_free_time"][worker_id] += 1000.0

        memory = Memory()
        memory.states.append(snapshot)
        memory.actions.append(action)
        memory.gated_team_traces.append(trace)
        memory.logprobs.append(logprob)
        memory.values.append(value)
        memory.masks.append(masks)
        memory.rewards.append(float(reward))
        memory.is_terminals.append(bool(done))
        metrics = agent.update(memory, env, current_ep=1)
        assert torch.isfinite(torch.tensor(metrics["Loss/Total"]))


def test_gated_team_three_small_ppo_updates_remain_legal() -> None:
    """低资源回归：连续三次采样、合法执行与 PPO 更新均应完成。"""
    with temporary_config(configs, _small_overrides()):
        env = AirLineEnv_Graph(DATA_PATH, seed=42)
        agent = PPOAgent(
            HBGATPN(configs),
            lr=1.0e-4,
            gamma=0.99,
            k_epochs=1,
            eps_clip=0.2,
            device=torch.device("cpu"),
            batch_size=1,
            total_timesteps=3,
            config=configs,
        )
        for episode in range(1, 4):
            obs, masks, _task_id = _first_physical_action_state(env)
            action, logprob, value, _station_mask, invalid = agent.select_action(
                obs,
                masks[0],
                masks[1],
                masks[2],
                deterministic=False,
                temperature=1.0,
            )
            assert action is not None and not invalid
            assert agent.last_gated_team_trace is not None
            snapshot = env.get_state_snapshot()
            _obs, reward, done, info = env.step(action)
            assert not info.get("invalid_action", False)

            memory = Memory()
            memory.states.append(snapshot)
            memory.actions.append(action)
            memory.gated_team_traces.append(agent.last_gated_team_trace)
            memory.logprobs.append(logprob)
            memory.values.append(value)
            memory.masks.append(masks)
            memory.rewards.append(float(reward))
            memory.is_terminals.append(bool(done))
            metrics = agent.update(memory, env, current_ep=episode)
            assert torch.isfinite(torch.tensor(metrics["Loss/Total"]))
