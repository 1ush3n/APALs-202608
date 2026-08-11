from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch_geometric.data import Batch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DECODE_VARIANTS = (
    "decode_uncached_with_legacy_mean",
    "decode_uncached_without_legacy_mean",
    "decode_cached_with_legacy_mean",
    "decode_cached_without_legacy_mean",
)

from configs import Config
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent


def summarize_samples(samples: list[float]) -> dict[str, float]:
    if not samples:
        raise ValueError("样本不能为空")
    values = np.asarray(samples, dtype=np.float64)
    return {
        "p10": float(np.quantile(values, 0.1)),
        "p50": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
        "mean": float(values.mean()),
    }


def _advance_to_physical_task(env: AirLineEnv_Graph) -> tuple[object, tuple[torch.Tensor, ...]]:
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
            chosen = min(physical)
            forced_mask = torch.ones_like(masks[0])
            forced_mask[chosen] = False
            return obs, (forced_mask, masks[1], masks[2])
        if not ready:
            raise RuntimeError("冻结状态中不存在可推进任务")
        obs, _reward, done, info = env.step((min(ready), -1, []))
        if done or not info.get("virtual_task", False):
            raise RuntimeError("未能推进到可调度物理工序")
    raise RuntimeError("超过工序数仍未找到物理工序")


def _time_cuda(
    fn: Callable[[], None], *, warmup: int, repeats: int, device: torch.device
) -> dict[str, dict[str, float]]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    gpu_ms: list[float] = []
    wall_ms: list[float] = []
    for _ in range(repeats):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        wall_started = time.perf_counter()
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize(device)
        wall_ms.append((time.perf_counter() - wall_started) * 1000.0)
        gpu_ms.append(float(start_event.elapsed_time(end_event)))
    return {"cuda_event_ms": summarize_samples(gpu_ms), "wall_sync_ms": summarize_samples(wall_ms)}


def _select_manifest_paths(manifest_path: Path) -> list[Path]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = payload.get("files", [])
    if len(files) < 4:
        raise RuntimeError("真实训练 manifest 至少需要四个声明 CSV")
    offsets = np.linspace(0, len(files) - 1, num=4, dtype=int).tolist()
    return [manifest_path.parent / str(files[index]["file"]) for index in offsets]


def benchmark_worker_pointer_v2_decode(
    *, manifest_path: Path, warmup: int, repeats: int, seed: int
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("该 benchmark 必须在 CUDA 上执行")
    if warmup < 1 or repeats < 3:
        raise ValueError("warmup 至少为 1，repeats 至少为 3")
    device = torch.device("cuda")
    torch.manual_seed(seed)
    config = Config()
    config.team_selection_mode = "autoregressive_pressure_v2"
    config.policy_action_scope = "operation_station_worker"
    config.actor_context_mode = "attention"
    config.lightning_precision = "bf16-mixed"
    config.seed = seed
    model = HBGATPN(config).to(device).eval()
    agent = PPOAgent(
        model,
        lr=1.0e-4,
        gamma=0.99,
        k_epochs=1,
        eps_clip=0.2,
        device=device,
        batch_size=1,
        total_timesteps=1,
        config=config,
    )
    environments = [AirLineEnv_Graph(path, seed=seed) for path in _select_manifest_paths(manifest_path)]
    prepared = [_advance_to_physical_task(env) for env in environments]
    observations = [item[0] for item in prepared]
    masks = [item[1] for item in prepared]
    batch = Batch.from_data_list(observations).to(device)
    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        embeddings, global_context = model(batch)
    task_ptr = batch["task"].ptr.tolist()
    station_ptr = batch["station"].ptr.tolist()
    worker_ptr = batch["worker"].ptr.tolist()
    graph_index = 0
    task_start, task_end = task_ptr[graph_index], task_ptr[graph_index + 1]
    station_start, station_end = station_ptr[graph_index], station_ptr[graph_index + 1]
    worker_start, worker_end = worker_ptr[graph_index], worker_ptr[graph_index + 1]
    task_mask, station_mask, worker_mask = masks[graph_index]
    task_id = int(torch.nonzero(~task_mask, as_tuple=False)[0].item())
    station_id = int(torch.nonzero(~station_mask[task_id], as_tuple=False)[0].item())
    raw_task_gpu = batch["task"].x[task_start:task_end]
    raw_worker_gpu = batch["worker"].x[worker_start:worker_end]
    raw_task_cpu = observations[graph_index]["task"].x
    raw_worker_cpu = observations[graph_index]["worker"].x
    pressure = agent._build_v2_pressure_context(
        task_features=raw_task_gpu,
        worker_features=raw_worker_gpu,
        task_present=None,
        task_action_invalid=task_mask.to(device),
        worker_present=None,
        worker_queue_invalid=worker_mask.to(device),
    )
    task_emb = embeddings["task"][task_start + task_id].unsqueeze(0)
    station_emb = embeddings["station"][station_start + station_id].unsqueeze(0)
    worker_embs = embeddings["worker"][worker_start:worker_end].unsqueeze(0)
    demand = raw_task_gpu[task_id, 16].reshape(1).clamp_min(1.0)
    skills = raw_worker_gpu[:, 1:6]
    mask = torch.zeros((1, worker_embs.size(1)), device=device, dtype=torch.bool)
    worker_head = model.worker_head
    steps = min(5, max(1, int(demand.item())))

    def decode(cache_enabled: bool, keep_legacy_mean: bool) -> None:
        state = worker_head.initialize_v2_state(batch_size=1, device=device)
        cache = (
            worker_head.build_v2_decode_cache(
                task_emb=task_emb,
                station_emb=station_emb,
                global_context=global_context[graph_index].unsqueeze(0),
                worker_embs=worker_embs,
                pressure_context=pressure,
                demand=demand,
            )
            if cache_enabled
            else None
        )
        team: list[int] = []
        current_mask = mask.clone()
        for step in range(steps):
            worker_head.forward_choice_v2(
                task_emb=task_emb,
                station_emb=station_emb,
                global_context=global_context[graph_index].unsqueeze(0),
                worker_embs=worker_embs,
                pressure_context=pressure,
                team_state=state,
                demand=demand,
                mask=current_mask,
                decode_cache=cache,
            )
            worker_id = step % worker_embs.size(1)
            team.append(worker_id)
            state = worker_head.advance_v2_state(
                state, worker_embs[:, worker_id, :], skills[worker_id].unsqueeze(0)
            )
            if keep_legacy_mean:
                worker_embs[0, team, :].mean(dim=0, keepdim=True)
            current_mask = current_mask.clone()
            current_mask[0, worker_id] = True

    def cpu_pressure() -> None:
        agent._build_v2_pressure_context(
            task_features=raw_task_cpu,
            worker_features=raw_worker_cpu,
            task_present=None,
            task_action_invalid=task_mask,
            worker_present=None,
            worker_queue_invalid=worker_mask,
        )

    def gpu_pressure() -> None:
        agent._build_v2_pressure_context(
            task_features=raw_task_gpu,
            worker_features=raw_worker_gpu,
            task_present=None,
            task_action_invalid=task_mask.to(device),
            worker_present=None,
            worker_queue_invalid=worker_mask.to(device),
        )

    with torch.inference_mode():
        timings = {
            "pressure_cpu_upload": _time_cuda(cpu_pressure, warmup=warmup, repeats=repeats, device=device),
            "pressure_gpu_reuse": _time_cuda(gpu_pressure, warmup=warmup, repeats=repeats, device=device),
            "decode_uncached_with_legacy_mean": _time_cuda(lambda: decode(False, True), warmup=warmup, repeats=repeats, device=device),
            "decode_uncached_without_legacy_mean": _time_cuda(lambda: decode(False, False), warmup=warmup, repeats=repeats, device=device),
            "decode_cached_with_legacy_mean": _time_cuda(lambda: decode(True, True), warmup=warmup, repeats=repeats, device=device),
            "decode_cached_without_legacy_mean": _time_cuda(lambda: decode(True, False), warmup=warmup, repeats=repeats, device=device),
        }
    return {
        "status": "passed",
        "manifest": str(manifest_path.resolve()),
        "seed": seed,
        "warmup": warmup,
        "repeats": repeats,
        "graphs": [str(path.resolve()) for path in _select_manifest_paths(manifest_path)],
        "team_steps": steps,
        "timing": timings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    report = benchmark_worker_pointer_v2_decode(
        manifest_path=manifest_path, warmup=args.warmup, repeats=args.repeats, seed=args.seed
    )
    output_dir = args.output_dir or (
        PROJECT_ROOT / ".pytest_tmp_v2" / f"worker_pointer_v2_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(report_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
