# -*- coding: utf-8 -*-
"""临时诊断：torch.profiler 分解一次同形 replay 的耗时构成。

输出 GAT 编码、task/station 头、worker 团队解码、backward 的耗时占比，
为 replay 性能优化提供数据依据。

用法：
    python scripts/_profile_replay.py --envs 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import configs
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from runtime.seed import set_seed
from training.v2_fast_exact_batch import GPUExactBatchBuilder


def _pick_legal_targets(env, max_team: int) -> tuple[int, int, list[int]]:
    task_mask, station_mask, worker_mask = env.get_masks()
    task_id = int(torch.nonzero(~task_mask)[0].item())
    station_id = int(torch.nonzero(station_mask[task_id])[0].item()) - 1
    worker_ids = torch.nonzero(~worker_mask).reshape(-1)[:max_team].tolist()
    return task_id, station_id, worker_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--data", type=str, default="data/680.csv")
    args = parser.parse_args()

    configs.team_selection_mode = "autoregressive_pressure_v2_fast_exact"
    configs.num_envs = int(args.envs)
    configs.lightning_precision = "bf16-mixed"
    configs.batch_size = 256
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA 不可用，跳过")
        return 0

    envs = [
        AirLineEnv_Graph(data_path_or_dir=args.data, seed=42 + index)
        for index in range(int(args.envs))
    ]
    for env in envs:
        env.reset(randomize_duration=False, randomize_workers=False)
    builder = GPUExactBatchBuilder(config=configs, env=envs[0], device=device)
    model = HBGATPN(configs).to(device)
    agent = PPOAgent(
        model=model,
        lr=float(configs.lr),
        gamma=float(configs.gamma),
        k_epochs=1,
        eps_clip=float(configs.eps_clip),
        device=device,
        batch_size=int(configs.batch_size),
        total_timesteps=3,
        config=configs,
    )

    max_team = 4
    group_size = int(args.envs)
    snap_group = [env.get_state_snapshot() for env in envs]
    mask_group = [env.get_masks() for env in envs]
    fast_batch = builder.build(
        snap_group, masks=mask_group, memory_indices=list(range(group_size))
    )
    batch = fast_batch.batch
    task_ids, station_ids, team_rows = [], [], []
    for env in envs:
        t_id, s_id, team = _pick_legal_targets(env, max_team)
        task_ids.append(t_id)
        station_ids.append(s_id)
        team_rows.append(team + [-1] * (max_team - len(team)))
    batch.y_task = torch.tensor(task_ids, dtype=torch.long, device=device)
    batch.y_station = torch.tensor(station_ids, dtype=torch.long, device=device)
    batch.y_team = torch.tensor(team_rows, dtype=torch.long, device=device)

    # 预热一次
    agent._replay_v2_fast_exact_group(fast_batch)
    agent.optimizer.zero_grad()

    from torch.profiler import ProfilerActivity, profile

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        outputs = agent._replay_v2_fast_exact_group(fast_batch)
        loss = sum(output["team"].sum() for output in outputs)
        loss.backward()
        torch.cuda.synchronize(device)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
