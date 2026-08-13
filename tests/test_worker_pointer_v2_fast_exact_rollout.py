# -*- coding: utf-8 -*-
"""WorkerPointer v2 Fast-Exact 阶段二A：rollout 预构建 batch 接口测试。

- Fast-Exact 模式必须创建 v2 worker head（统一 is_worker_pointer_v2_mode 判断）；
- select_actions_batch 支持 snapshots + fast_exact_builder 快路径；
- snapshots 快路径与 obs_list 路径在确定性模式下动作 / value / 行为 log-prob 等价；
- snapshots 未提供 builder 时 fail-closed。
"""

from __future__ import annotations

from pathlib import Path

import torch

from configs import Config, configs as global_configs
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from tests.runtime_safety import temporary_config
from training.rollout_service import APALRolloutService
from training.v2_fast_exact_batch import GPUExactBatchBuilder


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "283.csv"


def _fast_exact_config() -> Config:
    cfg = Config()
    cfg.hidden_dim = 32
    cfg.num_gat_layers = 1
    cfg.num_heads = 2
    cfg.n_w = 40
    cfg.n_m = 5
    cfg.team_selection_mode = "autoregressive_pressure_v2_fast_exact"
    cfg.policy_action_scope = "operation_station_worker"
    cfg.actor_context_mode = "attention"
    cfg.worker_pointer_v2_replay_mode = "behavior_group_exact_gpu_template_v2"
    cfg.worker_pointer_v2_rollout_group_upper_bound = 16
    cfg.num_envs = 4
    cfg.seed = 42
    return cfg


def _make_env(cfg: Config, seed: int = 42) -> AirLineEnv_Graph:
    env = AirLineEnv_Graph(str(DATA_PATH), seed=seed)
    env.reset(seed=seed)
    return env


def _make_agent(cfg: Config, device: torch.device) -> PPOAgent:
    return PPOAgent(
        HBGATPN(cfg),
        lr=1.0e-4,
        gamma=0.99,
        k_epochs=1,
        eps_clip=0.2,
        device=device,
        batch_size=1,
        total_timesteps=1,
        config=cfg,
    )


def test_fast_exact_model_creates_v2_worker_head() -> None:
    cfg = _fast_exact_config()
    model = HBGATPN(cfg)
    worker_head = model.worker_head
    assert hasattr(worker_head, "v2_member_proj")
    assert hasattr(worker_head, "v2_query_proj")


def test_select_actions_batch_snapshots_requires_builder() -> None:
    with temporary_config(global_configs, {}):
        cfg = _fast_exact_config()
        env = _make_env(cfg)
        snapshot = env.get_state_snapshot()
        masks = env.get_masks()
        agent = _make_agent(cfg, torch.device("cpu"))
        try:
            agent.select_actions_batch(
                [],
                [masks[0]],
                [masks[1]],
                [masks[2]],
                deterministic=True,
                snapshots=[snapshot],
            )
        except RuntimeError as exc:
            assert "fast_exact_builder" in str(exc)
        else:
            raise AssertionError("snapshots 模式缺少 builder 应抛错")


def test_select_actions_batch_snapshots_matches_obs_path() -> None:
    with temporary_config(global_configs, {}):
        cfg = _fast_exact_config()
        env = _make_env(cfg)
        snapshot = env.get_state_snapshot()
        obs = env.rebuild_state_from_snapshot(snapshot)
        masks = env.get_masks()
        device = torch.device("cpu")
        agent = _make_agent(cfg, device)
        builder = GPUExactBatchBuilder(config=cfg, env=env, device=device)

        obs_result = agent.select_actions_batch(
            [obs],
            [masks[0]],
            [masks[1]],
            [masks[2]],
            deterministic=True,
        )
        behavior_before = list(agent.last_v2_behavior_logprobs)
        snap_result = agent.select_actions_batch(
            [],
            [masks[0]],
            [masks[1]],
            [masks[2]],
            deterministic=True,
            snapshots=[snapshot],
            fast_exact_builder=builder,
        )
        behavior_after = list(agent.last_v2_behavior_logprobs)

        (action_a, lp_a, value_a, _mask_a, invalid_a) = obs_result[0]
        (action_b, lp_b, value_b, _mask_b, invalid_b) = snap_result[0]
        assert invalid_a == invalid_b
        assert action_a[0] == action_b[0]
        assert action_a[1] == action_b[1]
        assert list(action_a[2]) == list(action_b[2])
        torch.testing.assert_close(
            torch.as_tensor(value_a, dtype=torch.float32),
            torch.as_tensor(value_b, dtype=torch.float32),
            atol=1.0e-5,
            rtol=0.0,
        )
        for (la, sa, ta), (lb, sb, tb) in zip(behavior_before, behavior_after):
            assert la == lb and sa == sb and ta == tb


def test_rollout_service_fast_exact_creates_gpu_builder() -> None:
    """Fast-Exact 模式下 rollout service 必须持有 GPUExactBatchBuilder。"""
    cfg = _fast_exact_config()

    class _FakeEnv:
        pass

    class _FakeVectorEnv:
        num_envs = 4
        envs = [_FakeEnv()]

    service = APALRolloutService(
        agent=object(),
        vector_env=_FakeVectorEnv(),
        eval_env=object(),
        config=cfg,
        device=torch.device("cpu"),
    )
    assert service._fast_exact_builder is not None
    assert isinstance(service._fast_exact_builder, GPUExactBatchBuilder)
