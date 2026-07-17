from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import configs
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from runtime.seed import set_seed
from training.rollout_service import APALRolloutService
from utils.vector_env import EnvCreator, VectorEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="APAL rollout 可复现性能基准")
    parser.add_argument("--data", type=Path, default=Path("data") / "680.csv")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ipc-fusion", action="store_true")
    parser.add_argument("--detailed-profile", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = args.data if args.data.is_absolute() else PROJECT_ROOT / args.data
    data_path = data_path.resolve()
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    num_envs = int(args.num_envs or configs.num_envs)
    if num_envs < 1 or args.max_steps < 1 or args.repeats < 1:
        raise ValueError("num_envs、max_steps 和 repeats 必须大于 0")

    overrides = {
        "data_file_path": str(data_path),
        "train_data_path_or_dir": str(data_path),
        "num_envs": num_envs,
        "rollout_max_steps": int(args.max_steps),
        "seed": int(args.seed),
        "randomize_durations": False,
        "enable_dynamic_events": False,
        "enable_station_breakdown": False,
        "enable_material_delay": False,
        "rollout_heartbeat_interval_sec": 0.0,
        "enable_rollout_ipc_fusion": bool(args.ipc_fusion),
        "enable_rollout_detailed_profiler": bool(args.detailed_profile),
        "rollout_profile_interval": 1,
    }
    configs.update_from_dict(overrides)
    set_seed(int(args.seed))

    start_method = "forkserver" if platform.system() == "Linux" else "spawn"
    vector_env = VectorEnv(
        EnvCreator(str(data_path), seed_offset=int(args.seed)),
        num_envs=num_envs,
        start_method=start_method,
        worker_threads=1,
        init_timeout_sec=float(configs.vector_env_init_timeout_sec),
        command_timeout_sec=float(configs.vector_env_command_timeout_sec),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HBGATPN(configs).to(device)
    agent = PPOAgent(
        model=model,
        lr=float(configs.lr),
        gamma=float(configs.gamma),
        k_epochs=int(configs.k_epochs),
        eps_clip=float(configs.eps_clip),
        device=device,
        batch_size=int(configs.batch_size),
        total_timesteps=max(1, int(args.repeats)),
        config=configs,
    )
    service = APALRolloutService(
        agent=agent,
        vector_env=vector_env,
        eval_env=vector_env.envs[0],
        config=configs,
        device=device,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    measurements: list[dict[str, float | int]] = []
    try:
        for repeat in range(int(args.repeats)):
            _, metrics = service._collect_episode(repeat + 1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            record: dict[str, float | int] = {
                "repeat": repeat + 1,
                "environment_steps": int(metrics.environment_steps),
                "total_seconds": float(metrics.total_seconds),
                "sps": float(metrics.steps_per_second),
                "ipc_mask_ms": float(metrics.ipc_mask_ms),
                "forward_ms": float(metrics.forward_ms),
                "rebuild_ms": float(metrics.rebuild_ms),
                "environment_step_ms": float(metrics.environment_step_ms),
            }
            measurements.append(record)
            if metrics.extra_metrics:
                record.update(
                    {
                        key: float(value)
                        for key, value in metrics.extra_metrics.items()
                        if key.startswith("Rollout/Profile/")
                    }
                )
            print(json.dumps(record, ensure_ascii=False), flush=True)
    finally:
        service.close()

    summary = {
        "data": str(data_path),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "num_envs": num_envs,
        "max_steps": int(args.max_steps),
        "ipc_fusion": bool(args.ipc_fusion),
        "detailed_profile": bool(args.detailed_profile),
        "mean_sps": sum(float(item["sps"]) for item in measurements) / len(measurements),
        "peak_cuda_allocated_mb": (
            float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)
            if device.type == "cuda"
            else 0.0
        ),
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
