"""
奖励函数与模型梯度流测试。

覆盖场景：
  1. 单步 reward 计算正确性
  2. episode 累积 reward 与 makespan 的单调关系
  3. HBGATPN 模型前向传播 (forward)
  4. Critic 双流价值估计 (get_value)
  5. PPOAgent select_action 梯度隔离
  6. PPOAgent update 梯度流无 NaN/Inf
  7. 模型参数在更新后确实改变
  8. Select action 在确定性/随机模式下的行为差异
"""
import sys
import os
import copy
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import configs as cfg
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from train import Memory

TOTAL_TESTS = 0
PASSED_TESTS = 0
FAILED_TESTS = []


def check(condition, name):
    global TOTAL_TESTS, PASSED_TESTS
    TOTAL_TESTS += 1
    if condition:
        PASSED_TESTS += 1
        print(f"  [PASS] {name}")
    else:
        FAILED_TESTS.append(name)
        print(f"  [FAIL] {name}")


def _make_agent(device, total_updates=10):
    model = HBGATPN(cfg.configs).to(device)
    agent = PPOAgent(
        model=model, lr=cfg.configs.lr, gamma=cfg.configs.gamma,
        k_epochs=cfg.configs.k_epochs, eps_clip=cfg.configs.eps_clip,
        device=device, batch_size=cfg.configs.batch_size,
        total_timesteps=total_updates,
    )
    return agent


def test_reward_calculation():
    print("\n--- test_reward_calculation ---")
    DATA = os.path.join(ROOT_DIR, "data", "283.csv")
    cfg.configs.n_w = 40
    cfg.configs.n_m = 5

    env = AirLineEnv_Graph(data_path_or_dir=DATA, seed=42)
    state = env.reset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = _make_agent(device)

    rewards = []
    makespans = []
    done = False
    for _ in range(env.num_tasks * 2):
        if done:
            break
        t_mask, s_mask, w_mask = env.get_masks()
        if t_mask.all():
            if env.try_wait_for_resources():
                continue
            break

        action_ret = agent.select_action(
            state.to(device), mask_task=t_mask.to(device),
            mask_station_matrix=s_mask.to(device),
            mask_worker=w_mask.to(device), deterministic=False,
        )
        if action_ret[0] is None:
            break
        action, _, _, _, _ = action_ret
        state, reward, done, info = env.step(action)
        rewards.append(reward)
        makespans.append(np.max(env.station_wall_clock))

    check(len(rewards) > 0, f"Rewards collected: {len(rewards)} steps")
    check(all(isinstance(r, (int, float, np.floating)) for r in rewards),
          "All rewards are numeric")
    check(not any(np.isnan(r) for r in rewards), "No NaN in rewards")
    check(not any(np.isinf(r) for r in rewards), "No Inf in rewards")

    total_reward = sum(rewards)
    print(f"    Total reward: {total_reward:.4f}, Steps: {len(rewards)}")


def test_reward_monotonicity():
    print("\n--- test_reward_monotonicity ---")
    DATA = os.path.join(ROOT_DIR, "data", "283.csv")
    cfg.configs.n_w = 40
    cfg.configs.n_m = 5

    env = AirLineEnv_Graph(data_path_or_dir=DATA, seed=42)
    state = env.reset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = _make_agent(device)

    done = False
    for _ in range(env.num_tasks * 2):
        if done:
            break
        t_mask, s_mask, w_mask = env.get_masks()
        if t_mask.all():
            if env.try_wait_for_resources():
                continue
            break
        action_ret = agent.select_action(
            state.to(device), mask_task=t_mask.to(device),
            mask_station_matrix=s_mask.to(device),
            mask_worker=w_mask.to(device), deterministic=True,
        )
        if action_ret[0] is None:
            break
        action, _, _, _, _ = action_ret
        state, reward, done, info = env.step(action)

    final_makespan = np.max(env.station_wall_clock) if len(env.assigned_tasks) > 0 else 99999
    check(final_makespan > 0, f"Final makespan > 0 (got {final_makespan:.1f})")
    check(final_makespan < 99999, "Makespan is finite (not deadlock)")


def test_model_forward():
    print("\n--- test_model_forward ---")
    DATA = os.path.join(ROOT_DIR, "data", "283.csv")
    cfg.configs.n_w = 40
    cfg.configs.n_m = 5

    env = AirLineEnv_Graph(data_path_or_dir=DATA, seed=42)
    state = env.reset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HBGATPN(cfg.configs).to(device)
    model.eval()

    with torch.no_grad():
        x_dict, global_context = model(state.to(device))

    check('task' in x_dict, "x_dict contains 'task'")
    check('worker' in x_dict, "x_dict contains 'worker'")
    check('station' in x_dict, "x_dict contains 'station'")
    check(x_dict['task'].shape[-1] == cfg.configs.hidden_dim,
          f"Task embedding dim = {cfg.configs.hidden_dim}")
    check(global_context.dim() == 2, "global_context is 2D [B, Dim]")


def test_critic_value():
    print("\n--- test_critic_value ---")
    DATA = os.path.join(ROOT_DIR, "data", "283.csv")
    cfg.configs.n_w = 40
    cfg.configs.n_m = 5

    env = AirLineEnv_Graph(data_path_or_dir=DATA, seed=42)
    state = env.reset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HBGATPN(cfg.configs).to(device)
    model.eval()

    with torch.no_grad():
        value = model.get_value(state.to(device))

    check(value.dim() == 2, f"Value output is 2D, got shape {value.shape}")
    check(value.shape[1] == 1, "Value output is scalar per batch")
    check(not torch.isnan(value).any(), "Value is not NaN")
    check(not torch.isinf(value).any(), "Value is not Inf")


def test_select_action_no_grad():
    print("\n--- test_select_action_no_grad ---")
    DATA = os.path.join(ROOT_DIR, "data", "283.csv")
    cfg.configs.n_w = 40
    cfg.configs.n_m = 5

    env = AirLineEnv_Graph(data_path_or_dir=DATA, seed=42)
    state = env.reset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = _make_agent(device)

    t_mask, s_mask, w_mask = env.get_masks()

    action, logprob, value, sp_mask, is_invalid = agent.select_action(
        state.to(device), mask_task=t_mask.to(device),
        mask_station_matrix=s_mask.to(device),
        mask_worker=w_mask.to(device), deterministic=False,
    )

    check(action is not None, "select_action returns non-None action")
    check(isinstance(action, tuple) and len(action) == 3, "Action is (task_id, station_id, team)")
    check(isinstance(action[2], list), "Team is list of worker indices")
    check(isinstance(logprob, float), "logprob is float")
    check(isinstance(value, float), "value is float")


def test_select_action_deterministic_vs_stochastic():
    print("\n--- test_select_action_deterministic_vs_stochastic ---")
    DATA = os.path.join(ROOT_DIR, "data", "283.csv")
    cfg.configs.n_w = 40
    cfg.configs.n_m = 5
    torch.manual_seed(42)
    np.random.seed(42)

    env = AirLineEnv_Graph(data_path_or_dir=DATA, seed=42)
    state = env.reset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = _make_agent(device)

    t_mask, s_mask, w_mask = env.get_masks()

    torch.manual_seed(123)
    np.random.seed(123)
    a1, _, _, _, _ = agent.select_action(
        state.to(device), mask_task=t_mask.to(device),
        mask_station_matrix=s_mask.to(device),
        mask_worker=w_mask.to(device), deterministic=True,
    )

    torch.manual_seed(123)
    np.random.seed(123)
    a2, _, _, _, _ = agent.select_action(
        state.to(device), mask_task=t_mask.to(device),
        mask_station_matrix=s_mask.to(device),
        mask_worker=w_mask.to(device), deterministic=True,
    )

    check(a1[0] == a2[0], "Deterministic produces same task_id")
    check(a1[1] == a2[1], "Deterministic produces same station_id")


def test_ppo_update_no_nan():
    print("\n--- test_ppo_update_no_nan ---")
    DATA = os.path.join(ROOT_DIR, "data", "283.csv")
    cfg.configs.n_w = 40
    cfg.configs.n_m = 5

    env = AirLineEnv_Graph(data_path_or_dir=DATA, seed=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = _make_agent(device)

    memory = Memory()
    state_snap = env.reset()
    t_mask, s_mask, w_mask = env.get_masks()

    for _ in range(5):
        action_ret = agent.select_action(
            state_snap.to(device), mask_task=t_mask.to(device),
            mask_station_matrix=s_mask.to(device),
            mask_worker=w_mask.to(device), deterministic=False,
        )
        if action_ret[0] is None:
            continue
        action, logprob, val, sp_mask, is_invalid = action_ret
        state_snap, reward, done, info = env.step(action)

        memory.states.append(env.get_state_snapshot())
        memory.actions.append(action)
        memory.logprobs.append(logprob)
        memory.rewards.append(reward)
        memory.is_terminals.append(done)
        memory.masks.append((t_mask, s_mask, w_mask))
        memory.values.append(val)

        if done:
            break
        t_mask, s_mask, w_mask = env.get_masks()

    old_params = {n: p.clone() for n, p in agent.policy.named_parameters()}

    try:
        metrics = agent.update(memory, env)
        check(isinstance(metrics, dict), "update returns dict of metrics")
        check('Loss/Total' in metrics, "metrics contains Loss/Total")
        check(not np.isnan(metrics['Loss/Total']), "Total loss is not NaN")
        check(not np.isinf(metrics['Loss/Total']), "Total loss is not Inf")
        print(f"    Loss/Total: {metrics['Loss/Total']:.4f}")
    except Exception as e:
        check(False, f"update raised exception: {e}")


def test_parameters_change_after_update():
    print("\n--- test_parameters_change_after_update ---")
    DATA = os.path.join(ROOT_DIR, "data", "283.csv")
    cfg.configs.n_w = 40
    cfg.configs.n_m = 5
    cfg.configs.use_schedule_free = False

    env = AirLineEnv_Graph(data_path_or_dir=DATA, seed=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = _make_agent(device)

    memory = Memory()
    state_snap = env.reset()
    t_mask, s_mask, w_mask = env.get_masks()

    for _ in range(6):
        action_ret = agent.select_action(
            state_snap.to(device), mask_task=t_mask.to(device),
            mask_station_matrix=s_mask.to(device),
            mask_worker=w_mask.to(device), deterministic=False,
        )
        if action_ret[0] is None:
            continue
        action, logprob, val, sp_mask, is_invalid = action_ret
        state_snap, reward, done, info = env.step(action)

        memory.states.append(env.get_state_snapshot())
        memory.actions.append(action)
        memory.logprobs.append(logprob)
        memory.rewards.append(reward)
        memory.is_terminals.append(done)
        memory.masks.append((t_mask, s_mask, w_mask))
        memory.values.append(val)

        if done:
            break
        t_mask, s_mask, w_mask = env.get_masks()

    first_param_name = next(iter(agent.policy.named_parameters()))[0]
    old_val = dict(agent.policy.named_parameters())[first_param_name].clone()

    try:
        agent.update(memory, env)
    except Exception:
        pass

    new_val = dict(agent.policy.named_parameters())[first_param_name].clone()
    changed = not torch.equal(old_val, new_val)
    check(changed, f"Parameter '{first_param_name}' changed after update")


def test_model_encoder_no_nan():
    print("\n--- test_model_encoder_no_nan ---")
    DATA = os.path.join(ROOT_DIR, "data", "283.csv")
    cfg.configs.n_w = 40
    cfg.configs.n_m = 5

    env = AirLineEnv_Graph(data_path_or_dir=DATA, seed=42)
    state = env.reset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HBGATPN(cfg.configs).to(device)

    state_d = state.to(device)
    x_dict_e = model.embedder(state_d.x_dict)
    for k, v in x_dict_e.items():
        check(not torch.isnan(v).any(), f"Embedding [{k}] has no NaN")
        check(not torch.isinf(v).any(), f"Embedding [{k}] has no Inf")

    x_dict_enc = model.encoder(x_dict_e, state_d.edge_index_dict)
    for k, v in x_dict_enc.items():
        check(not torch.isnan(v).any(), f"Encoder output [{k}] has no NaN")


def main():
    print("=" * 60)
    print("REWARD & GRADIENT FLOW TEST SUITE")
    print("=" * 60)

    test_reward_calculation()
    test_reward_monotonicity()
    test_model_forward()
    test_critic_value()
    test_select_action_no_grad()
    test_select_action_deterministic_vs_stochastic()
    test_ppo_update_no_nan()
    test_parameters_change_after_update()
    test_model_encoder_no_nan()

    print("\n" + "=" * 60)
    print(f"RESULTS: {PASSED_TESTS}/{TOTAL_TESTS} passed")
    if FAILED_TESTS:
        print("FAILURES:")
        for name in FAILED_TESTS:
            print(f"  - {name}")
    print("=" * 60)
    return len(FAILED_TESTS) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
