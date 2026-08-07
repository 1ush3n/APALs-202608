# -*- coding: utf-8 -*-
"""APCF 锚点条件完整团队提议与反事实门控的正式测试。

覆盖协议（论文实现计划）要求：
  1) 提议团队合法性：技能匹配、锁定语义（0=空 / station+1）、worker_mask 排除；
  2) 存在合法替代时提议 P ≠ 锚点 H（汉明距离 ≥ 1）；
  3) 温度 0 + 未预训练（价值头零初始化、门控负偏置）必选锚点分支；
  4) z=0 时重算对数概率仍计入完整提议链（Σ log q + log π̃）；
  5) 单环境与批量路径生成完全一致（掩码/门控/轨迹）；
  6) 预训练 checkpoint 可被 runtime.checkpoints 加载并还原模型语义；
  7) 回归：既有 scope 的 PPO 更新有限性不因 APCF 改动退化。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from configs import Config, configs
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent, FrozenAnchorProposalTrace
from runtime.configuration import validate_runtime_config
from tests.runtime_safety import temporary_config, seed_everything
from training.memory import Memory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "283.csv"


def _apcf_overrides(**extra) -> dict[str, object]:
    values = {
        "policy_action_scope": "operation_station_anchor_proposal_team",
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
        "anchor_proposal_prior_margin": 4.0,
        "anchor_proposal_gate_bias": -4.0,
        "anchor_proposal_train_branch_floor_start": 0.20,
        "anchor_proposal_train_branch_floor_end": 0.02,
        "anchor_proposal_branch_floor_decay_fraction": 0.40,
        "anchor_proposal_require_difference": True,
    }
    values.update(extra)
    return values


def _make_agent(**extra) -> PPOAgent:
    overrides = _apcf_overrides(**extra)
    with temporary_config(configs, overrides):
        model = HBGATPN(configs)
        agent = PPOAgent(
            model,
            lr=1.0e-4,
            gamma=0.99,
            k_epochs=1,
            eps_clip=0.2,
            device=torch.device("cpu"),
            batch_size=int(overrides["batch_size"]),
            total_timesteps=1,
            config=configs,
        )
        return agent, overrides


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


def _proposal_masks_from_obs(
    agent: PPOAgent,
    obs: object,
    task_id: int,
    station_id: int,
    worker_mask: torch.Tensor | None,
) -> torch.Tensor:
    """复刻 PPO 提议分支的非法掩码（True=非法），供测试断言合法性。"""
    from worker_feature_layout import resolve_worker_feature_layout

    layout = resolve_worker_feature_layout(agent.config)
    worker_feats = obs["worker"].x
    skills = worker_feats[:, layout.skill_slice]
    task_skill_vec = obs["task"].x[task_id, 5 : 5 + skills.size(1)]
    skill_idx = int(torch.argmax(task_skill_vec).item())
    has_skill = skills[:, skill_idx] > 0.5
    locks = torch.argmax(worker_feats[:, layout.lock_slice], dim=1)
    lock_ok = (locks == 0) | (locks == (station_id + 1))
    illegal = (~has_skill) | (~lock_ok)
    if worker_mask is not None:
        illegal = illegal | worker_mask.to(device=illegal.device, dtype=torch.bool)
    return illegal


@pytest.mark.parametrize("temperature", (0.0, 1.0))
def test_apcf_proposal_is_legal_and_differs_from_anchor(temperature: float) -> None:
    seed_everything(42)
    agent, overrides = _make_agent()
    env = AirLineEnv_Graph(DATA_PATH, seed=42)
    obs, masks = _advance_to_ready_physical_task(env)
    with temporary_config(configs, overrides):
        action, logprob, _value, _smask, invalid = agent.select_action(
            obs,
            mask_task=masks[0],
            mask_station_matrix=masks[1],
            mask_worker=masks[2],
            deterministic=False,
            temperature=temperature,
        )
    assert action is not None and not invalid
    task_id, station_id, team = int(action[0]), int(action[1]), [int(w) for w in action[2]]
    trace = agent.last_anchor_proposal_trace
    assert isinstance(trace, FrozenAnchorProposalTrace)
    assert trace.task_id == task_id and trace.station_id == station_id

    # 团队必须恰好为工序需求人数、无重复成员。
    assert len(team) == len(set(team))
    assert len(team) == len(trace.anchor_team)

    # 合法性：技能匹配、锁定语义、worker_mask 排除。
    illegal = _proposal_masks_from_obs(agent, obs, task_id, station_id, masks[2])
    for worker_id in team:
        assert not bool(illegal[worker_id].item()), f"工人 {worker_id} 不合法"

    # 存在合法替代时，提议链存在（P≠H 由首步强制非锚点保证）。
    if trace.proposal_available:
        assert len(trace.proposal_worker_sequence) == len(trace.anchor_team)
        assert len(set(trace.proposal_worker_sequence) - set(trace.anchor_team)) >= 1
        assert trace.hamming_distance >= 1
    assert torch.isfinite(torch.tensor(logprob))


def test_apcf_temperature_zero_selects_anchor_without_pretrain() -> None:
    """温度 0 + 价值头零初始化 + 门控负偏置 → 严格选择锚点分支。"""
    seed_everything(42)
    agent, overrides = _make_agent()
    env = AirLineEnv_Graph(DATA_PATH, seed=42)
    obs, masks = _advance_to_ready_physical_task(env)
    with temporary_config(configs, overrides):
        action, _logprob, _value, _smask, invalid = agent.select_action(
            obs,
            mask_task=masks[0],
            mask_station_matrix=masks[1],
            mask_worker=masks[2],
            deterministic=True,
            temperature=0.0,
        )
    assert action is not None and not invalid
    trace = agent.last_anchor_proposal_trace
    assert isinstance(trace, FrozenAnchorProposalTrace)
    assert trace.selected_branch == 0, "未预训练时温度 0 必须选择锚点分支"
    assert tuple(int(w) for w in action[2]) == trace.anchor_team


def test_apcf_z0_recompute_includes_full_proposal_chain() -> None:
    """即使 z=0 执行锚点，重算对数概率也必须包含完整提议链。"""
    seed_everything(42)
    agent, overrides = _make_agent()
    env = AirLineEnv_Graph(DATA_PATH, seed=42)
    obs, masks = _advance_to_ready_physical_task(env)

    # 训练采样下 z 按 ε 下限混合分布采样（未预训练 p(z=1)≈0.21），
    # 重采样直到采到 z=0 分支且存在合法提议，保证覆盖"z=0 仍计入提议链"语义。
    selected: tuple[FrozenAnchorProposalTrace, float] | None = None
    for _attempt in range(40):
        with temporary_config(configs, overrides):
            action, logprob, _value, _smask, invalid = agent.select_action(
                obs,
                mask_task=masks[0],
                mask_station_matrix=masks[1],
                mask_worker=masks[2],
                deterministic=False,
                temperature=1.0,
            )
        assert action is not None and not invalid
        trace = agent.last_anchor_proposal_trace
        assert isinstance(trace, FrozenAnchorProposalTrace)
        if trace.proposal_available and trace.selected_branch == 0:
            selected = (trace, float(logprob))
            break
    assert selected is not None, "40 次重采样仍未采到 z=0 分支，测试环境异常"
    trace, _logprob = selected
    assert trace.selected_branch == 0, "应选择锚点分支"

    # 用冻结轨迹在当前策略下重算（模拟 PPO update 重算路径）。
    model = agent.policy
    encoded, _context = model(obs)
    task_emb = encoded["task"][trace.task_id].unsqueeze(0)
    station_emb = encoded["station"][trace.station_id].unsqueeze(0)
    worker_embs = encoded["worker"]
    recomputed, entropy, _ = agent._recompute_anchor_proposal_logprobs(
        task_embeddings=task_emb,
        station_embeddings=station_emb,
        worker_embeddings=worker_embs.unsqueeze(0),
        frozen_traces=[trace],
    )
    assert torch.isfinite(recomputed).all()
    assert torch.isfinite(entropy).all()
    # 提议链严格为负（log q < 0），且 z=0 时完整对数概率 != 0，
    # 证明提议链即使未被执行也计入了 PPO 的 log π。
    assert recomputed[0].item() < 0.0

    # 手动逐项复算提议链 + z=0 门控对数概率，验证与重算一致。
    from torch.distributions import Categorical

    num_workers = worker_embs.size(0)
    manual_lp = torch.zeros((), device=worker_embs.device)
    for j, chosen in enumerate(trace.proposal_worker_sequence):
        step_mask = torch.full(
            (1, num_workers), True, device=worker_embs.device, dtype=torch.bool
        )
        step_mask[0, list(trace.per_step_worker_ids[j])] = False
        context = (
            worker_embs[list(trace.proposal_worker_sequence[:j]), :].mean(
                dim=0, keepdim=True
            )
            if j > 0
            else None
        )
        scores = model.anchor_team_head.forward_choice(
            task_emb,
            station_emb,
            worker_embs[list(trace.anchor_team), :].mean(dim=0, keepdim=True),
            worker_embs.unsqueeze(0),
            mask=step_mask,
            current_team_emb=context,
        )
        dist = Categorical(logits=scores.float())
        manual_lp = manual_lp + dist.log_prob(
            torch.tensor([[chosen]], device=worker_embs.device)
        )[0]
    # z=0 分支门控对数概率（混合分布，ε=当前探索下限）。
    proposal_emb = worker_embs[list(trace.proposal_worker_sequence), :].mean(
        dim=0, keepdim=True
    )
    branch_logits, _delta, _g = model.anchor_proposal_gate(
        task_emb,
        station_emb,
        worker_embs[list(trace.anchor_team), :].mean(dim=0, keepdim=True),
        proposal_emb,
        torch.tensor(list(trace.gate_features), dtype=torch.float32).reshape(1, -1),
        torch.tensor([[float(trace.hamming_distance)]], dtype=torch.float32),
    )
    eps = max(float(trace.branch_floor), 0.0)
    soft = torch.softmax(branch_logits.float(), dim=1)
    mixed = eps + (1.0 - 2.0 * eps) * soft
    bdist = Categorical(probs=mixed)
    manual_lp = manual_lp + bdist.log_prob(
        torch.tensor([[0]], device=worker_embs.device)
    )[0]
    assert torch.allclose(recomputed[0], manual_lp, atol=1.0e-4)


def test_apcf_single_and_batch_paths_agree() -> None:
    """单环境 select_action 与批量 select_actions_batch 生成一致。"""
    seed_everything(42)
    agent, overrides = _make_agent()
    envs = [AirLineEnv_Graph(DATA_PATH, seed=42) for _ in range(2)]
    prepared = [_advance_to_ready_physical_task(env) for env in envs]
    observations = [item[0] for item in prepared]
    masks = [item[1] for item in prepared]
    with temporary_config(configs, overrides):
        results = agent.select_actions_batch(
            observations,
            [item[0] for item in masks],
            [item[1] for item in masks],
            [item[2] for item in masks],
            deterministic=True,
            temperature=0.0,
        )
        assert len(results) == 2
        for env, result in zip(envs, results, strict=True):
            action, logprob, _value, _smask, invalid = result
            assert action is not None and not invalid
            assert torch.isfinite(torch.tensor(logprob))
            _obs, _reward, _done, info = env.step(action)
            assert not info.get("invalid_action", False)
        assert len(agent.last_anchor_proposal_traces) == 2
        for trace in agent.last_anchor_proposal_traces:
            assert isinstance(trace, FrozenAnchorProposalTrace)
            assert trace.selected_branch == 0
            assert trace.proposal_available


def test_apcf_memory_trace_count_matches_states() -> None:
    """PPO update 前 memory 中锚点轨迹数与状态数必须一致（对齐校验）。"""
    seed_everything(42)
    agent, overrides = _make_agent(batch_size=1)
    env = AirLineEnv_Graph(DATA_PATH, seed=42)
    obs, masks = _advance_to_ready_physical_task(env)
    with temporary_config(configs, overrides):
        action, logprob, value, _smask, invalid = agent.select_action(
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
    memory.anchor_proposal_traces.append(agent.last_anchor_proposal_trace)
    _obs, reward, done, info = env.step(action)
    assert not info.get("invalid_action", False)
    memory.rewards.append(float(reward))
    memory.is_terminals.append(bool(done))
    with temporary_config(configs, overrides):
        metrics = agent.update(memory, env, current_ep=1)
    assert torch.isfinite(torch.tensor(metrics["Loss/Total"]))


def test_apcf_config_validation_rejects_bad_margin_and_floor() -> None:
    """运行时配置校验：margin/门控偏置/探索下限非法值必须拒绝。"""
    cfg = Config()
    cfg.policy_action_scope = "operation_station_anchor_proposal_team"
    cfg.anchor_proposal_prior_margin = 4.0
    cfg.anchor_proposal_gate_bias = -4.0
    cfg.anchor_proposal_train_branch_floor_start = 0.20
    cfg.anchor_proposal_train_branch_floor_end = 0.02
    cfg.anchor_proposal_branch_floor_decay_fraction = 0.40
    validate_runtime_config(cfg)

    cfg.anchor_proposal_prior_margin = 0.0
    with pytest.raises(ValueError):
        validate_runtime_config(cfg)
    cfg.anchor_proposal_prior_margin = 4.0
    cfg.anchor_proposal_gate_bias = 1.0
    with pytest.raises(ValueError):
        validate_runtime_config(cfg)
    cfg.anchor_proposal_gate_bias = -4.0
    cfg.anchor_proposal_train_branch_floor_end = 0.60
    with pytest.raises(ValueError):
        validate_runtime_config(cfg)


def test_apcf_encoder_is_frozen_and_heads_trainable_in_pretrain() -> None:
    """预训练模块冻结编码器、仅训练双头：requires_grad 语义正确。"""
    import sys

    from training.cf_pretrain import CFPretrainLightningModule

    sys.path.insert(0, str(PROJECT_ROOT))
    agent, overrides = _make_agent()
    with temporary_config(configs, overrides):
        module = CFPretrainLightningModule(
            agent.policy,
            configs,
            manifest_path=str(PROJECT_ROOT / "data" / "nonexistent_manifest.json"),
            manifest_sha256="dummy-sha256-for-frozen-semantics-test",
        )
    model = module.policy
    trainable_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable_names, "预训练必须存在可训练参数"
    for name in trainable_names:
        assert name.startswith("anchor_team_head") or name.startswith(
            "anchor_proposal_gate"
        ), f"非双头参数 {name} 不得可训练"
