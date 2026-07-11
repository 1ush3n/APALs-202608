from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.literature.common import resolve_project_path
from baselines.literature_dqn.train_graph_ddqn_apal import GraphDDQNAgent
from configs import configs
from environment import AirLineEnv_Graph
from env_wrapper import standardize_env_step
from runtime.hydra_config import (
    ExtraArgument,
    HydraCliError,
    hydra_help,
    initialize_hydra_runtime,
    should_show_help,
)
from runtime.seed import set_seed
from training.observation import refresh_env_observation
from utils.device_utils import clear_torch_cache, get_available_device


EXTRA_ARGS = {
    "benchmark_data_path": ExtraArgument(default="data/283.csv", help="固定基准实例"),
    "benchmark_transitions": ExtraArgument(default=64, help="预生成 transition 数"),
    "benchmark_updates": ExtraArgument(default=5, help="每次重复测量 replay update 数"),
    "benchmark_repeats": ExtraArgument(default=5, help="配对重复次数"),
    "output_dir": ExtraArgument(
        default="results/05_efficiency_and_logs/ddqn_performance_optimization",
        help="基准结果目录",
    ),
}


def _clone_args(args: Any, *, batched: bool, batch_size: int) -> SimpleNamespace:
    values = dict(vars(args))
    values.update(
        {
            "epsilon": 1.0,
            "epsilon_min": 0.05,
            "epsilon_decay": 0.995,
            "memory_size": max(int(values.get("benchmark_transitions", 64)) * 2, batch_size),
            "replay_start_size": batch_size,
            "ddqn_enable_batched_replay": bool(batched),
            "ddqn_enable_profiler": False,
            "ddqn_enable_gpu_batch_rebuild": False,
        }
    )
    return SimpleNamespace(**values)


def _collect_transitions(
    env: AirLineEnv_Graph,
    agent: GraphDDQNAgent,
    count: int,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    state = env.reset(randomize_duration=False, randomize_workers=False, seed=seed)
    transitions: list[dict[str, Any]] = []
    while len(transitions) < count:
        masks = env.get_masks()
        while bool(masks[0].all()):
            if not env.try_wait_for_resources():
                state = env.reset(
                    randomize_duration=False,
                    randomize_workers=False,
                    seed=seed + len(transitions) + 1,
                )
                masks = env.get_masks()
                break
            state = refresh_env_observation(env)
            masks = env.get_masks()
        snapshot = env.get_state_snapshot()
        action = agent.random_action(state, masks)
        if action is None:
            state = env.reset(
                randomize_duration=False,
                randomize_workers=False,
                seed=seed + len(transitions) + 1,
            )
            continue
        state, reward, done, _info = standardize_env_step(env, action)
        transitions.append(
            {
                "state_snapshot": snapshot,
                "action": action,
                "reward": reward,
                "next_snapshot": env.get_state_snapshot(),
                "done": done,
                "masks": masks,
                "next_masks": env.get_masks(),
            }
        )
        if done:
            state = env.reset(
                randomize_duration=False,
                randomize_workers=False,
                seed=seed + len(transitions) + 1,
            )
    return transitions


def _fill(agent: GraphDDQNAgent, transitions: list[dict[str, Any]]) -> None:
    for transition in transitions:
        agent.remember(**transition)


def _measure(
    args: Any,
    env: AirLineEnv_Graph,
    transitions: list[dict[str, Any]],
    model_state: dict[str, torch.Tensor],
    *,
    batched: bool,
    batch_size: int,
    updates: int,
    device: torch.device,
) -> dict[str, float | str]:
    agent = GraphDDQNAgent(
        _clone_args(args, batched=batched, batch_size=batch_size),
        device,
    )
    agent.model.load_state_dict(model_state)
    agent.target_model.load_state_dict(model_state)
    _fill(agent, transitions)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    losses = [
        agent.replay(env, batch_size, dataset_idx=0)
        for _ in range(updates)
    ]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_vram = (
        float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)
        if device.type == "cuda"
        else 0.0
    )
    clear_torch_cache()
    return {
        "mode": "batched" if batched else "serial",
        "elapsed_sec": elapsed,
        "updates_per_sec": float(updates) / max(elapsed, 1e-12),
        "mean_loss": statistics.fmean(losses),
        "peak_vram_mb": peak_vram,
    }


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if should_show_help(raw_args):
        print(hydra_help(EXTRA_ARGS))
        return 0
    try:
        args = initialize_hydra_runtime(
            raw_args,
            target=configs,
            project_root=PROJECT_ROOT,
            default_experiment="initial_schedule_283",
            extra_arguments=EXTRA_ARGS,
            create_run_context=False,
        )
        seed = int(configs.seed)
        set_seed(seed)
        configs.enable_dynamic_events = False
        configs.randomize_durations = False
        data_path = resolve_project_path(args.benchmark_data_path)
        output_dir = resolve_project_path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        batch_size = int(configs.batch_size)
        transition_count = max(batch_size, int(args.benchmark_transitions))
        updates = max(1, int(args.benchmark_updates))
        repeats = max(1, int(args.benchmark_repeats))
        device = get_available_device()
        env = AirLineEnv_Graph(str(data_path), seed=seed)

        collector = GraphDDQNAgent(
            _clone_args(args, batched=True, batch_size=batch_size),
            device,
        )
        transitions = _collect_transitions(
            env,
            collector,
            transition_count,
            seed=seed,
        )
        model_state = {
            key: value.detach().cpu().clone()
            for key, value in collector.model.state_dict().items()
        }
        del collector
        clear_torch_cache()

        rows: list[dict[str, float | str | int]] = []
        for repeat in range(repeats):
            order = (False, True) if repeat % 2 == 0 else (True, False)
            for batched in order:
                result = _measure(
                    args,
                    env,
                    transitions,
                    model_state,
                    batched=batched,
                    batch_size=batch_size,
                    updates=updates,
                    device=device,
                )
                rows.append({"repeat": repeat, **result})
                print(
                    f"[DDQNBenchmark] repeat={repeat} mode={result['mode']} "
                    f"updates_per_sec={float(result['updates_per_sec']):.4f} "
                    f"peak_vram_mb={float(result['peak_vram_mb']):.1f}",
                    flush=True,
                )

        by_mode = {
            mode: [float(row["updates_per_sec"]) for row in rows if row["mode"] == mode]
            for mode in ("serial", "batched")
        }
        serial_median = statistics.median(by_mode["serial"])
        batched_median = statistics.median(by_mode["batched"])
        summary = {
            "data_path": str(data_path),
            "seed": seed,
            "device": str(device),
            "batch_size": batch_size,
            "transitions": transition_count,
            "updates_per_repeat": updates,
            "repeats": repeats,
            "serial_updates_per_sec_median": serial_median,
            "batched_updates_per_sec_median": batched_median,
            "speedup": batched_median / max(serial_median, 1e-12),
        }
        with (output_dir / "ddqn_replay_benchmark.csv").open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (output_dir / "ddqn_replay_benchmark_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (HydraCliError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"[CLI] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
