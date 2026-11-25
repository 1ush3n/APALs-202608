# -*- coding: utf-8 -*-
"""临时复现脚本 v3：多步 rollout 多组重放 backward 错误定位（用完即删）。"""
import sys

sys.path.insert(0, ".")

import torch

from configs import configs
from environment import AirLineEnv_Graph
from tests.runtime_safety import temporary_config
from tests.test_joint_experiment_architecture import DATA_PATH
from tests.test_worker_pointer_v2_fast_exact_replay import (
    _fast_exact_overrides,
    _make_agent,
)
from training.memory import Memory
from training.v2_fast_exact_batch import GPUExactBatchBuilder
from training.worker_pointer_v2_behavior import make_behavior_traces

overrides = _fast_exact_overrides()
overrides["hidden_dim"] = 128
overrides["num_heads"] = 4
overrides["k_epochs"] = 2
overrides["batch_size"] = 256
overrides["accumulation_steps"] = 16
DATA_PATH = "data/scale_400_800_datasets/syn_403_77.csv"
with temporary_config(configs, overrides):
    agent = _make_agent()
    device = torch.device("cuda")
    envs = [
        AirLineEnv_Graph(DATA_PATH, seed=40 + index) for index in range(4)
    ]
    for index, env in enumerate(envs):
        env.reset(seed=40 + index, randomize_workers=True)
    memory = Memory()
    b_task, b_station, b_team = [], [], []
    old_lp, rewards, advantages = [], [], []
    max_team = int(getattr(configs, "max_team_size", 5))
    rollout_steps = 130
    active_counts = [4] * 130
    active = list(range(4))
    builder = GPUExactBatchBuilder(config=configs, env=envs[0], device=device)
    for step_index in range(rollout_steps):
        active = list(range(active_counts[step_index]))
        masks_list = [envs[index].get_masks() for index in active]
        results = agent.select_actions_batch(
            [], [m[0] for m in masks_list], [m[1] for m in masks_list],
            [m[2] for m in masks_list],
            deterministic=False, temperature=1.0, is_eval=False,
            snapshots=[envs[i].get_state_snapshot() for i in active],
            fast_exact_builder=builder,
        )
        behavior_lps = agent.last_v2_behavior_logprobs
        traces = make_behavior_traces(
            group_id=(0, step_index),
            env_indices=list(range(len(active))),
            behavior_logprobs=behavior_lps,
        )
        for local_index, (action, logprob, value, _, invalid) in enumerate(results):
            assert not invalid
            env_index = active[local_index]
            memory.states.append(envs[env_index].get_state_snapshot())
            memory.actions.append(action)
            memory.logprobs.append(
                float(logprob[0]) if isinstance(logprob, (list, tuple)) else float(logprob)
            )
            memory.values.append(float(value))
            memory.masks.append(masks_list[local_index])
            memory.rewards.append(0.0)
            memory.is_terminals.append(False)
            memory.worker_pointer_v2_behavior_traces.append(traces[local_index])
            team = list(action[2])
            padded = team + [-1] * (max_team - len(team))
            b_task.append(int(action[0]))
            b_station.append(int(action[1]))
            b_team.append(padded[:max_team])
            old_lp.append(
                float(logprob[0]) if isinstance(logprob, (list, tuple)) else float(logprob)
            )
            rewards.append(0.0)
            advantages.append(1.0)

    builder = GPUExactBatchBuilder(config=configs, env=envs[0], device=device)
    with torch.autograd.detect_anomaly():
        metrics = agent._run_v2_fast_exact_replay_update(
            memory,
            envs[0],
            current_ep=1,
            advantages=torch.tensor(advantages, dtype=torch.float32),
            rewards=torch.tensor(rewards, dtype=torch.float32),
            old_logprobs=torch.tensor(old_lp, dtype=torch.float32),
            b_task=torch.tensor(b_task, dtype=torch.long),
            b_station=torch.tensor(b_station, dtype=torch.long),
            b_team=torch.tensor(b_team, dtype=torch.long),
            action_scope="operation_station_worker",
            fast_exact_builder=builder,
        )
    print("OK samples=", len(memory.states), "MaxAE=", metrics["V2/FirstContractTotalMaxAE"])
