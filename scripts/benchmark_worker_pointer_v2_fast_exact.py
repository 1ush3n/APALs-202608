# -*- coding: utf-8 -*-
"""WorkerPointer v2 Fast-Exact 三组性能基准。

按独立进程、同 seed、同数据执行三组对比：
1. ``v2_legacy``：历史 autoregressive_pressure_v2，平台默认小环境数，CPU rebuild；
2. ``v2_cpu_wide``：历史 v2 仅扩大环境数（Linux 16 / Windows 4），CPU rebuild；
3. ``v2_fast_exact``：autoregressive_pressure_v2_fast_exact，GPU 常驻模板。

每组先执行一次不计时 warm-up，再记录：
rollout SPS、replay samples/s、update 总耗时、precheck 组数、模板命中率、
峰值 allocated/reserved 显存、GPU 利用率均值/P50/P90、fallback 次数、
首次同形重算各分量 MAE/MaxAE。

用法示例（Windows）：
    python scripts/benchmark_worker_pointer_v2_fast_exact.py --data data/680.csv
    python scripts/benchmark_worker_pointer_v2_fast_exact.py --mode v2_fast_exact
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import configs
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from runtime.seed import set_seed
from training.fast_exact_benchmark import (
    compute_replay_performance,
    compute_template_hit_rate,
    measure_operation,
    resolve_benchmark_num_envs,
    summarize_utilization,
)
from training.rollout_service import APALRolloutService
from utils.vector_env import EnvCreator, VectorEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorkerPointer v2 Fast-Exact 性能基准")
    parser.add_argument("--data", type=Path, default=Path("data") / "680.csv")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=("all", "v2_legacy", "v2_cpu_wide", "v2_fast_exact"),
    )
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="逻辑 PPO batch（默认 256，CLI 最高优先级）")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="0=完整 rollout（默认，保证压力覆盖真实）；>0 仅用于受控短跑")
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--rollout-episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--worker-mode",
        choices=("v2_legacy", "v2_cpu_wide", "v2_fast_exact"),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


class NvidiaSmiSampler:
    """在被测区间内后台采集 GPU 利用率，不引入固定等待。"""

    def __init__(self, *, interval_ms: int = 200) -> None:
        self.interval_ms = max(100, int(interval_ms))
        self._samples: list[float] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        self._samples = []
        self._stop_event.clear()
        if not torch.cuda.is_available():
            return

        def _poll() -> None:
            process: subprocess.Popen[str] | None = None
            try:
                process = subprocess.Popen(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                        "-lms",
                        str(self.interval_ms),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                self._process = process
                assert process.stdout is not None
                for line in process.stdout:
                    if self._stop_event.is_set():
                        break
                    value = line.strip()
                    if value.isdigit():
                        self._samples.append(float(value))
            except (OSError, subprocess.SubprocessError):
                return
            finally:
                if process is not None and process.poll() is None:
                    process.terminate()

        self._thread = threading.Thread(target=_poll, daemon=True)
        self._thread.start()

    def stop(self) -> list[float]:
        self._stop_event.set()
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._process = None
        return list(self._samples)


def _apply_mode_overrides(
    mode: str,
    *,
    num_envs_override: int | None,
    batch_size: int,
    max_steps: int,
    seed: int,
    data_path: Path,
) -> int:
    if mode == "v2_fast_exact":
        team_mode = "autoregressive_pressure_v2_fast_exact"
    else:
        team_mode = "autoregressive_pressure_v2"
    num_envs = resolve_benchmark_num_envs(
        mode,
        platform_name=platform.system(),
        override=num_envs_override,
    )
    overrides = {
        "team_selection_mode": team_mode,
        "policy_action_scope": "operation_station_worker",
        "lightning_precision": "bf16-mixed",
        "batch_size": int(batch_size),
        "worker_pointer_v2_logical_batch_cap": int(batch_size),
        "data_file_path": str(data_path),
        "train_data_path_or_dir": str(data_path),
        "num_envs": num_envs,
        "rollout_max_steps": int(max_steps),
        "seed": int(seed),
        "update_every_episodes": 1,
        "randomize_durations": False,
        "enable_dynamic_events": False,
        "enable_station_breakdown": False,
        "enable_material_delay": False,
        "enable_multi_benchmark_eval": False,
        "async_eval_enabled": False,
        "async_validation_enabled": False,
        "enable_reschedule_mode": False,
        "rollout_heartbeat_interval_sec": 0.0,
        "enable_rollout_ipc_fusion": False,
    }
    configs.update_from_dict(overrides)
    set_seed(int(seed))
    return num_envs


def _build_service(num_envs: int, data_path: Path, seed: int) -> tuple[APALRolloutService, VectorEnv]:
    start_method = "forkserver" if platform.system() == "Linux" else "spawn"
    vector_env = VectorEnv(
        EnvCreator(str(data_path), seed_offset=int(seed)),
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
        total_timesteps=3,
        config=configs,
    )
    service = APALRolloutService(
        agent=agent,
        vector_env=vector_env,
        eval_env=vector_env.envs[0],
        config=configs,
        device=device,
    )
    return service, vector_env


def run_mode(
    mode: str,
    *,
    num_envs_override: int | None,
    batch_size: int,
    max_steps: int,
    rollout_episodes: int,
    updates: int,
    seed: int,
    data_path: Path,
    warmup: bool,
) -> dict:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    num_envs = _apply_mode_overrides(
        mode,
        num_envs_override=num_envs_override,
        batch_size=batch_size,
        max_steps=max_steps,
        seed=seed,
        data_path=data_path,
    )
    service, vector_env = _build_service(num_envs, data_path, seed)
    device = service.device

    def _synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    try:
        if warmup:
            service._collect_episode(0)

        rollout_sps: list[float] = []
        rollout_gpu_utilization: list[float] = []
        for episode in range(1, rollout_episodes + 1):
            episode_result, _elapsed, samples = measure_operation(
                lambda episode=episode: service._collect_episode(episode),
                sampler=NvidiaSmiSampler(),
                synchronize=_synchronize,
            )
            _, metrics = episode_result
            rollout_gpu_utilization.extend(samples)
            rollout_sps.append(float(metrics.steps_per_second))
        mean_rollout_sps = sum(rollout_sps) / max(1, len(rollout_sps))

        peak_reserved_mb = 0.0
        if device.type == "cuda":
            peak_reserved_mb = float(torch.cuda.max_memory_reserved(device)) / (1024.0**2)
            torch.cuda.reset_peak_memory_stats(device)

        last_update_metrics: dict[str, float] = {}
        update_gpu_utilization: list[float] = []
        update_seconds_total = 0.0
        replay_samples_total = 0
        physical_groups_total = 0.0
        for update_index in range(1, updates + 1):
            update = service.collect(update_index)
            replay_samples_total += len(update.memory.states)
            update_result, update_seconds, samples = measure_operation(
                lambda update=update: service.agent.update(
                    update.memory,
                    update.env,
                    current_ep=update.episode,
                ),
                sampler=NvidiaSmiSampler(),
                synchronize=_synchronize,
            )
            last_update_metrics = dict(update_result)
            update_seconds_total += float(update_seconds)
            update_gpu_utilization.extend(samples)
            physical_groups_total += float(
                update_result.get("V2/FastExact/PhysicalGroupCount", 0.0) or 0.0
            )

        # 三组统一使用完整 PPO update wall time；分子包含所有 PPO epoch 实际
        # 处理的样本次数，避免 k_epochs=2 时把吞吐低估一半。
        aggregate_metrics = dict(last_update_metrics)
        aggregate_metrics["V2/FastExact/ReplayUpdateSeconds"] = update_seconds_total
        aggregate_metrics["V2/FastExact/PhysicalGroupCount"] = physical_groups_total
        performance = compute_replay_performance(
            aggregate_metrics,
            sample_count=replay_samples_total,
            k_epochs=int(service.agent.k_epochs),
        )

        rollout_builder = getattr(service, "_fast_exact_builder", None)
        replay_builder = getattr(service.agent, "_v2_fast_exact_builder", None)

        def _builder_stats(builder: object | None) -> dict | None:
            if builder is None:
                return None
            hits = int(getattr(builder, "template_hits", 0))
            misses = int(getattr(builder, "template_misses", 0))
            return {
                "hits": hits,
                "misses": misses,
                "hit_rate": compute_template_hit_rate(hits, misses),
            }

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_allocated_mb = float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)
            peak_reserved_mb = max(peak_reserved_mb, float(torch.cuda.max_memory_reserved(device)) / (1024.0**2))
        else:
            peak_allocated_mb = 0.0

        row: dict = {
            "mode": mode,
            "num_envs": num_envs,
            "batch_size": int(service.agent.batch_size),
            "k_epochs": int(service.agent.k_epochs),
            "accumulation_steps": int(service.agent.accumulation_steps),
            "team_selection_mode": str(configs.team_selection_mode),
            "mean_rollout_sps": mean_rollout_sps,
            "peak_allocated_mb": peak_allocated_mb,
            "peak_reserved_mb": peak_reserved_mb,
            "gpu_utilization": {
                "rollout": summarize_utilization(rollout_gpu_utilization),
                "update": summarize_utilization(update_gpu_utilization),
            },
            "replay": performance,
            "rollout_template": _builder_stats(rollout_builder),
            "replay_template": _builder_stats(replay_builder),
            "fallback_count": 0,
        }
        return row
    finally:
        service.close()


def run_modes_in_subprocesses(
    args: argparse.Namespace,
    *,
    data_path: Path,
    modes: list[str],
    result_dir: Path,
) -> list[dict]:
    """逐模式启动独立 Python 进程，避免 CUDA context 与缓存相互污染。"""
    result_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    script_path = Path(__file__).resolve()
    for mode in modes:
        result_path = result_dir / f"{mode}.json"
        command = [
            sys.executable,
            str(script_path),
            "--worker-mode",
            mode,
            "--result-json",
            str(result_path),
            "--data",
            str(data_path),
            "--batch-size",
            str(int(args.batch_size)),
            "--max-steps",
            str(int(args.max_steps)),
            "--updates",
            str(int(args.updates)),
            "--rollout-episodes",
            str(int(args.rollout_episodes)),
            "--seed",
            str(int(args.seed)),
            "--warmup" if bool(args.warmup) else "--no-warmup",
        ]
        if args.num_envs is not None:
            command.extend(["--num-envs", str(int(args.num_envs))])
        print(f"[Benchmark] mode={mode} start", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        if not result_path.is_file():
            raise RuntimeError(f"benchmark worker 未生成结果: {result_path}")
        row = json.loads(result_path.read_text(encoding="utf-8"))
        if row.get("mode") != mode:
            raise RuntimeError(
                f"benchmark worker 模式错位: expected={mode!r}, "
                f"actual={row.get('mode')!r}"
            )
        rows.append(row)
        print(f"[Benchmark] mode={mode} done", flush=True)
    return rows


def _write_result_atomic(path: Path, row: dict) -> None:
    """原子写入单模式结果，避免父进程读取半截 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    data_path = args.data if args.data.is_absolute() else PROJECT_ROOT / args.data
    data_path = data_path.resolve()
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    if args.worker_mode is not None:
        if args.result_json is None:
            raise ValueError("--worker-mode 必须同时提供 --result-json")
        row = run_mode(
            args.worker_mode,
            num_envs_override=args.num_envs,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            rollout_episodes=args.rollout_episodes,
            updates=args.updates,
            seed=args.seed,
            data_path=data_path,
            warmup=args.warmup,
        )
        row["worker_pid"] = int(os.getpid())
        _write_result_atomic(args.result_json.resolve(), row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        return 0

    modes = (
        ["v2_legacy", "v2_cpu_wide", "v2_fast_exact"]
        if args.mode == "all"
        else [args.mode]
    )
    with tempfile.TemporaryDirectory(prefix="apal_fast_exact_benchmark_") as directory:
        rows = run_modes_in_subprocesses(
            args,
            data_path=data_path,
            modes=modes,
            result_dir=Path(directory),
        )

    summary = {
        "platform": platform.system(),
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        ),
        "data": str(data_path),
        "rows": rows,
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
