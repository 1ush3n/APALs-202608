# -*- coding: utf-8 -*-
"""WorkerPointer v2 FULL-X 自适应 GPU MAX-FILL 调优扫描器。

三阶段扫描架构：
- G1: 自适应 Encoder Batch Cap (16 -> 32 -> 64 -> 128 -> 256) + Worst-case OOM 压力门禁 (Peak Reserved <= 85%)；
- G2: 自适应 PPO Logical Batch Size (256/16 -> 512/8 -> 1024/4 -> 2048/2 -> 4096/1)；
- G3: 自适应 Rollout num_envs (4, 8, 16)；

每个 Candidate 均在独立的 subprocess / 独立 CUDA context 中执行以防显存污染。
第一选择准则：End-to-End Samples/sec 最高的安全配置。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import Config, configs
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from runtime.seed import set_seed
from runtime.modes import FAST_EXACT_REPLAY_MODE
from training.fast_exact_benchmark import (
    compute_replay_performance,
    compute_template_hit_rate,
    measure_operation,
    summarize_group_sizes,
    summarize_utilization,
)
from training.rollout_service import APALRolloutService
from utils.vector_env import EnvCreator, VectorEnv


class NvidiaSmiSampler:
    """在被测区间内后台独立采样 GPU 利用率。"""

    def __init__(self, *, interval_ms: int = 100) -> None:
        self.interval_ms = max(50, int(interval_ms))
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


def _apply_full_x_overrides(
    *,
    data_path: Path,
    encoder_batch_cap: int,
    batch_size: int,
    accumulation_steps: int,
    num_envs: int,
    seed: int = 42,
) -> None:
    overrides = {
        "team_selection_mode": "autoregressive_pressure_v2_fast_exact",
        "policy_action_scope": "operation_station_worker",
        "actor_context_mode": "attention",
        "use_shared_trunk": False,
        "worker_pointer_v2_dynamic_eft_features": True,
        "worker_pointer_v2_dynamic_eft_feature_clip": 10.0,
        "worker_pointer_v2_explicit_team_state": True,
        "worker_pointer_v2_marginal_scarcity": True,
        "worker_pointer_v2_marginal_scarcity_clip": 10.0,
        "worker_pointer_v2_interaction_residual": True,
        "worker_pointer_v2_next_frontier_pressure": True,
        "conditional_head_baseline_mode": "factorized",
        "conditional_head_value_coef": 1.0,
        "lightning_precision": "bf16-mixed",
        "worker_pointer_v2_replay_mode": FAST_EXACT_REPLAY_MODE,
        "worker_pointer_v2_behavior_replay": True,
        "worker_pointer_v2_strict_gpu_replay": True,
        "worker_pointer_v2_fast_replay_batching": "logical_batch_v1",
        "worker_pointer_v2_fast_replay_encoder_batch_cap": int(encoder_batch_cap),
        "worker_pointer_v2_logical_batch_cap": int(batch_size),
        "worker_pointer_v2_rollout_group_upper_bound": int(num_envs),
        "batch_size": int(batch_size),
        "accumulation_steps": int(accumulation_steps),
        "num_envs": int(num_envs),
        "data_file_path": str(data_path),
        "train_data_path_or_dir": str(data_path),
        "worker_pointer_v2_fast_exact_profile": True,
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


def run_worker_candidate(
    *,
    data_path: Path,
    encoder_batch_cap: int,
    batch_size: int,
    accumulation_steps: int,
    num_envs: int,
    rollout_episodes: int = 2,
    seed: int = 42,
    warmup: bool = True,
) -> dict[str, Any]:
    """在独立子进程中执行单组 candidate 评估，返回高保真基准指标。"""
    _apply_full_x_overrides(
        data_path=data_path,
        encoder_batch_cap=encoder_batch_cap,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        num_envs=num_envs,
        seed=seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_method = "forkserver" if platform.system() == "Linux" else "spawn"

    vector_env = VectorEnv(
        EnvCreator(str(data_path), seed_offset=int(seed)),
        num_envs=int(num_envs),
        start_method=start_method,
        worker_threads=1,
        init_timeout_sec=float(configs.vector_env_init_timeout_sec),
        command_timeout_sec=float(configs.vector_env_command_timeout_sec),
    )
    model = HBGATPN(configs).to(device)
    agent = PPOAgent(
        model=model,
        lr=float(configs.lr),
        gamma=float(configs.gamma),
        k_epochs=int(configs.k_epochs),
        eps_clip=float(configs.eps_clip),
        device=device,
        batch_size=int(batch_size),
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

    try:
        # Warm-up (不计入评测)
        if warmup:
            upd = service.collect(1)
            service.agent.update(upd.memory, upd.env, current_ep=upd.episode)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        sampler = NvidiaSmiSampler(interval_ms=100)
        sampler.start()

        total_start = time.perf_counter()
        rollout_start = time.perf_counter()
        upd = service.collect(2)
        rollout_sec = time.perf_counter() - rollout_start

        update_start = time.perf_counter()
        update_result = service.agent.update(upd.memory, upd.env, current_ep=upd.episode)
        update_sec = time.perf_counter() - update_start
        total_sec = time.perf_counter() - total_start

        util_samples = sampler.stop()
        util_stats = summarize_utilization(util_samples)

        metrics = dict(update_result) if update_result is not None else {}
        sample_count = len(upd.memory.states)
        e2e_sps = sample_count / max(total_sec, 1e-6)
        replay_perf = compute_replay_performance(
            update_metrics=metrics,
            sample_count=sample_count,
            k_epochs=int(configs.k_epochs),
        )

        total_vram_mb = (
            torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)
            if torch.cuda.is_available()
            else 0.0
        )
        peak_allocated_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            if torch.cuda.is_available()
            else 0.0
        )
        peak_reserved_mb = (
            torch.cuda.max_memory_reserved(device) / (1024 * 1024)
            if torch.cuda.is_available()
            else 0.0
        )
        peak_reserved_pct = (
            (peak_reserved_mb / total_vram_mb * 100.0) if total_vram_mb > 0 else 0.0
        )

        return {
            "status": "SUCCESS",
            "encoder_batch_cap": int(encoder_batch_cap),
            "batch_size": int(batch_size),
            "accumulation_steps": int(accumulation_steps),
            "num_envs": int(num_envs),
            "data": str(data_path),
            "sample_count": sample_count,
            "total_wall_sec": total_sec,
            "rollout_wall_sec": rollout_sec,
            "update_wall_sec": update_sec,
            "e2e_samples_per_sec": e2e_sps,
            "replay_samples_per_sec": replay_perf.get("replay_samples_per_sec", 0.0),
            "peak_allocated_mb": peak_allocated_mb,
            "peak_reserved_mb": peak_reserved_mb,
            "total_vram_mb": total_vram_mb,
            "peak_reserved_pct": peak_reserved_pct,
            "gpu_util_mean": util_stats.get("mean"),
            "gpu_util_p50": util_stats.get("p50"),
            "gpu_util_p90": util_stats.get("p90"),
            "EncoderMs": metrics.get("V2/FastExact/Profile/EncoderMs", 0.0),
            "ActionHeadMs": metrics.get("V2/FastExact/Profile/ActionHeadMs", 0.0),
            "WorkerPointerMs": metrics.get("V2/FastExact/Profile/WorkerPointerMs", 0.0),
            "BackwardMs": metrics.get("V2/FastExact/Profile/BackwardMs", 0.0),
            "OptimizerMs": metrics.get("V2/FastExact/Profile/OptimizerMs", 0.0),
            "BuilderCalls": metrics.get("V2/FastExact/Profile/BuilderCalls", 0.0),
            "FormalReplayCalls": metrics.get("V2/FastExact/Profile/FormalReplayCalls", 0.0),
            "ActualSamplesPerUpdate": sample_count,
            "ActualLogicalBatchCount": metrics.get("V2/FastExact/PhysicalGroupCount", 0.0),
            "ActualMeanLogicalBatchSize": metrics.get("V2/FastExact/PhysicalGroupMeanSize", 0.0),
            "ActualOptimizerSteps": metrics.get("PPO/UpdateSteps", 0.0),
            "FirstContractTotalMaxAE": metrics.get("V2/FirstContractTotalMaxAE", 0.0),
            "GradientsFinite": metrics.get("Gradient/Finite", 1.0),
            "FallbackCount": metrics.get("V2/FastExact/FallbackCount", 0.0),
            "OOM": 0,
            "NonFinite": 0 if math.isfinite(metrics.get("Loss/Total", 0.0)) else 1,
        }
    except torch.cuda.OutOfMemoryError as oom_err:
        return {
            "status": "OOM",
            "encoder_batch_cap": int(encoder_batch_cap),
            "batch_size": int(batch_size),
            "accumulation_steps": int(accumulation_steps),
            "num_envs": int(num_envs),
            "data": str(data_path),
            "OOM": 1,
            "error": str(oom_err),
        }
    except Exception as exc:
        return {
            "status": "ERROR",
            "encoder_batch_cap": int(encoder_batch_cap),
            "batch_size": int(batch_size),
            "accumulation_steps": int(accumulation_steps),
            "num_envs": int(num_envs),
            "data": str(data_path),
            "OOM": 0,
            "error": str(exc),
        }
    finally:
        try:
            vector_env.close()
        except Exception:
            pass


def _exec_candidate_in_subprocess(
    *,
    script_path: Path,
    data_path: Path,
    encoder_batch_cap: int,
    batch_size: int,
    accumulation_steps: int,
    num_envs: int,
    rollout_episodes: int = 2,
    seed: int = 42,
    warmup: bool = True,
) -> dict[str, Any]:
    """通过子进程调用保证完全干净的独立 CUDA 上下文。"""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        res_json = Path(tmp.name)
    try:
        cmd = [
            sys.executable,
            str(script_path),
            "--worker-candidate",
            "--result-json",
            str(res_json),
            "--data",
            str(data_path),
            "--encoder-batch-cap",
            str(encoder_batch_cap),
            "--batch-size",
            str(batch_size),
            "--accumulation-steps",
            str(accumulation_steps),
            "--num-envs",
            str(num_envs),
            "--rollout-episodes",
            str(rollout_episodes),
            "--seed",
            str(seed),
            "--warmup" if warmup else "--no-warmup",
        ]
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        if res_json.is_file() and res_json.stat().st_size > 0:
            return json.loads(res_json.read_text(encoding="utf-8"))
        return {
            "status": "SUBPROCESS_FAIL",
            "encoder_batch_cap": encoder_batch_cap,
            "batch_size": batch_size,
            "accumulation_steps": accumulation_steps,
            "num_envs": num_envs,
            "stdout": proc.stdout[-1000:],
            "stderr": proc.stderr[-1000:],
            "returncode": proc.returncode,
        }
    finally:
        if res_json.is_file():
            try:
                res_json.unlink()
            except OSError:
                pass


def run_adaptive_max_fill_sweep(
    *,
    base_data: Path = Path("data/680.csv"),
    worst_case_datas: list[Path] = [
        Path("data/scale_400_800_datasets/variant_53_tasks_793_template_680.csv"),
        Path("data/scale_400_800_datasets/variant_45_tasks_777_template_680.csv"),
    ],
    seed: int = 42,
) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    print("=" * 80)
    print("🚀 [MAX-FILL] 启动 FULL-X 自适应 GPU MAX-FILL 极限负载调优扫描")
    print(f"   基础评测实例: {base_data}")
    print(f"   Worst-Case 门禁实例: {[str(p) for p in worst_case_datas]}")
    print("=" * 80)

    # ---------------------------------------------------------
    # G1: 自适应 Encoder Batch Cap 扫描
    # ---------------------------------------------------------
    print("\n--- [Stage G1] 自适应 Encoder Batch Cap 压力扫描 (16 -> 32 -> 64 -> 128 -> 256) ---")
    g1_candidates = [16, 32, 64, 128, 256]
    g1_results: list[dict[str, Any]] = []
    consecutive_drops = 0
    best_g1_sps = -1.0

    for cap in g1_candidates:
        print(f"  > [G1 Candidate] encoder_batch_cap={cap} (base envs=4, batch=256, accum=16) ...", flush=True)
        res = _exec_candidate_in_subprocess(
            script_path=script_path,
            data_path=base_data,
            encoder_batch_cap=cap,
            batch_size=256,
            accumulation_steps=16,
            num_envs=4,
            rollout_episodes=1,
            seed=seed,
            warmup=False,
        )
        g1_results.append(res)
        if res.get("status") == "OOM":
            print(f"    ❌ OOM detected at cap={cap}, 阻断更大 cap 扫描。")
            break
        if res.get("status") != "SUCCESS":
            print(f"    ⚠️ 候选执行失败: {res.get('status')}")
            break

        sps = float(res.get("e2e_samples_per_sec", 0.0))
        peak_res_pct = float(res.get("peak_reserved_pct", 0.0))
        peak_res_mb = float(res.get("peak_reserved_mb", 0.0))
        util_mean = res.get("gpu_util_mean")
        print(f"    ✅ cap={cap} | End-to-End SPS: {sps:.2f} | Peak Reserved: {peak_res_mb:.1f}MB ({peak_res_pct:.1f}%) | GPU Util: {util_mean}%")

        if sps > best_g1_sps:
            best_g1_sps = sps
            consecutive_drops = 0
        else:
            consecutive_drops += 1
            if consecutive_drops >= 2:
                print(f"    ⏹️ 连续 2 档吞吐未见提升 (当前 {sps:.2f} < 最佳 {best_g1_sps:.2f})，触发自适应早停。")
                break

    valid_g1 = [r for r in g1_results if r.get("status") == "SUCCESS"]
    sorted_g1 = sorted(valid_g1, key=lambda x: x.get("e2e_samples_per_sec", 0.0), reverse=True)
    top_candidates = sorted_g1[:2]

    best_safe_cap = 16
    best_safe_g1_res = None

    print("\n--- [Stage G1 Gate] Worst-Case OOM 压力门禁验证 (要求 Peak Reserved <= 85%) ---")
    for cand in top_candidates:
        cap = cand["encoder_batch_cap"]
        cand_passed = True
        gate_details = []
        for wc_data in worst_case_datas:
            print(f"  > [G1 Worst-Case Gate] cap={cap} on {wc_data.name} ...", flush=True)
            wc_res = _exec_candidate_in_subprocess(
                script_path=script_path,
                data_path=wc_data,
                encoder_batch_cap=cap,
                batch_size=256,
                accumulation_steps=16,
                num_envs=4,
                rollout_episodes=1,
                seed=seed,
                warmup=False,
            )
            gate_details.append(wc_res)
            if wc_res.get("status") != "SUCCESS" or wc_res.get("OOM", 0) == 1:
                cand_passed = False
                print(f"    ❌ Worst-case 出现失败/OOM: {wc_res.get('status')}")
                break
            peak_pct = float(wc_res.get("peak_reserved_pct", 0.0))
            max_ae = float(wc_res.get("FirstContractTotalMaxAE", 1.0))
            if peak_pct > 85.0:
                cand_passed = False
                print(f"    ❌ Peak Reserved {peak_pct:.1f}% 超出 85% 上限！")
                break
            if max_ae > 1e-3:
                cand_passed = False
                print(f"    ❌ FirstContractTotalMaxAE {max_ae:.6f} 超出 1e-3 阈值！")
                break
            print(f"    ✅ 通过 {wc_data.name}: Peak Reserved {peak_pct:.1f}% <= 85%, MaxAE={max_ae:.6f}")

        if cand_passed:
            best_safe_cap = cap
            best_safe_g1_res = cand
            print(f"  🏆 G1 选定最佳安全 Encoder Cap: {best_safe_cap} (基准 SPS: {cand['e2e_samples_per_sec']:.2f})")
            break

    if best_safe_g1_res is None:
        best_safe_cap = 16
        print("  ⚠️ 前序候选均未通过 Worst-case，默认回退安全基准 cap=16")

    # ---------------------------------------------------------
    # G2: 自适应 PPO Logical Batch Size 扫描
    # ---------------------------------------------------------
    print(f"\n--- [Stage G2] 自适应 PPO Logical Batch Size 扫描 (锁定 encoder_batch_cap={best_safe_cap}) ---")
    g2_candidates = [
        (256, 16),
        (512, 8),
        (1024, 4),
        (2048, 2),
        (4096, 1),
    ]
    g2_results: list[dict[str, Any]] = []
    best_g2_sps = -1.0
    best_g2_pair = (256, 16)
    best_g2_res = None
    g2_drops = 0

    for batch, accum in g2_candidates:
        print(f"  > [G2 Candidate] batch={batch}, accumulation={accum} (effective ~4096) ...", flush=True)
        res = _exec_candidate_in_subprocess(
            script_path=script_path,
            data_path=base_data,
            encoder_batch_cap=best_safe_cap,
            batch_size=batch,
            accumulation_steps=accum,
            num_envs=4,
            rollout_episodes=1,
            seed=seed,
            warmup=False,
        )
        g2_results.append(res)
        if res.get("status") == "OOM":
            print(f"    ❌ OOM detected at batch={batch}, 阻断更大 batch 扫描。")
            break
        if res.get("status") != "SUCCESS":
            print(f"    ⚠️ 候选执行失败: {res.get('status')}")
            break

        sps = float(res.get("e2e_samples_per_sec", 0.0))
        peak_res_pct = float(res.get("peak_reserved_pct", 0.0))
        util_mean = res.get("gpu_util_mean")
        print(f"    ✅ batch={batch}, accum={accum} | End-to-End SPS: {sps:.2f} | Peak Reserved: {peak_res_pct:.1f}% | GPU Util: {util_mean}%")

        if sps > best_g2_sps:
            best_g2_sps = sps
            best_g2_pair = (batch, accum)
            best_g2_res = res
            g2_drops = 0
        else:
            g2_drops += 1
            if g2_drops >= 2:
                print(f"    ⏹️ 连续 2 档吞吐持续下降 (当前 {sps:.2f} < 最佳 {best_g2_sps:.2f})，触发自适应早停。")
                break

    print(f"  🏆 G2 选定最优 Batch 组合: batch={best_g2_pair[0]}, accumulation={best_g2_pair[1]} (SPS: {best_g2_sps:.2f})")

    # ---------------------------------------------------------
    # G3: 自适应 Rollout num_envs 扫描
    # ---------------------------------------------------------
    print(f"\n--- [Stage G3] 自适应 Rollout num_envs 扫描 (锁定 cap={best_safe_cap}, batch={best_g2_pair[0]}, accum={best_g2_pair[1]}) ---")
    g3_candidates = [4, 8, 16]
    g3_results: list[dict[str, Any]] = []
    best_g3_sps = -1.0
    best_envs = 4
    best_final_res = None

    for envs in g3_candidates:
        print(f"  > [G3 Candidate] num_envs={envs} ...", flush=True)
        res = _exec_candidate_in_subprocess(
            script_path=script_path,
            data_path=base_data,
            encoder_batch_cap=best_safe_cap,
            batch_size=best_g2_pair[0],
            accumulation_steps=best_g2_pair[1],
            num_envs=envs,
            rollout_episodes=1,
            seed=seed,
            warmup=False,
        )
        g3_results.append(res)
        if res.get("status") != "SUCCESS":
            print(f"    ⚠️ 候选执行失败: {res.get('status')}")
            continue

        sps = float(res.get("e2e_samples_per_sec", 0.0))
        rollout_time = float(res.get("rollout_wall_sec", 0.0))
        update_time = float(res.get("update_wall_sec", 0.0))
        peak_res_pct = float(res.get("peak_reserved_pct", 0.0))
        util_mean = res.get("gpu_util_mean")
        print(f"    ✅ num_envs={envs} | End-to-End SPS: {sps:.2f} | Rollout: {rollout_time:.2f}s, Update: {update_time:.2f}s | Peak Reserved: {peak_res_pct:.1f}% | GPU Util: {util_mean}%")

        if sps > best_g3_sps:
            best_g3_sps = sps
            best_envs = envs
            best_final_res = res

    print(f"  🏆 G3 选定最优环境数: num_envs={best_envs} (最终端到端 SPS: {best_g3_sps:.2f})")

    # ---------------------------------------------------------
    # 汇总决策与 P3/P4 评估
    # ---------------------------------------------------------
    final_cap = best_safe_cap
    final_batch, final_accum = best_g2_pair
    final_envs = best_envs

    p3_p4_recommended = False
    p3_p4_reason = ""
    if best_final_res is not None:
        util = float(best_final_res.get("gpu_util_mean") or 0.0)
        upd_sec = float(best_final_res.get("update_wall_sec", 1.0))
        action_head_ms = float(best_final_res.get("ActionHeadMs", 0.0))
        wp_ms = float(best_final_res.get("WorkerPointerMs", 0.0))
        head_ratio = ((action_head_ms + wp_ms) / 1000.0) / max(upd_sec, 1e-6)

        if util < 60.0 and head_ratio > 0.50:
            p3_p4_recommended = True
            p3_p4_reason = f"GPU Util={util:.1f}% < 60% 且 (ActionHead+WorkerPointer) 占比 {head_ratio*100:.1f}% > 50% update wall time"
        else:
            p3_p4_reason = f"GPU Util={util:.1f}% (阈值 60%) 或 Head 耗时占比 {head_ratio*100:.1f}% (阈值 50%) 未同时满足触发条件，保持现有代码稳定性，坚决不实施 P3/P4。"

    summary = {
        "final_parameters": {
            "encoder_batch_cap": final_cap,
            "batch_size": final_batch,
            "accumulation_steps": final_accum,
            "num_envs": final_envs,
            "effective_batch_size": final_batch * final_accum,
        },
        "best_performance": best_final_res,
        "p3_p4_decision": {
            "recommended": p3_p4_recommended,
            "reason": p3_p4_reason,
        },
        "g1_results": g1_results,
        "g2_results": g2_results,
        "g3_results": g3_results,
    }

    print("\n" + "=" * 80)
    print("🎯 [MAX-FILL] 最终参数决选结果：")
    print(f"   • encoder_batch_cap: {final_cap}")
    print(f"   • batch_size: {final_batch}")
    print(f"   • accumulation_steps: {final_accum}")
    print(f"   • num_envs: {final_envs}")
    print(f"   • End-to-End SPS: {best_g3_sps:.2f}")
    print(f"   • P3/P4 建议: {'实施' if p3_p4_recommended else '不实施'} ({p3_p4_reason})")
    print("=" * 80)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FULL-X GPU MAX-FILL 自适应调优扫描器")
    parser.add_argument("--worker-candidate", action="store_true", help="单 candidate 子进程执行模式")
    parser.add_argument("--result-json", type=Path, default=None, help="写入 candidate 结果路径")
    parser.add_argument("--data", type=Path, default=Path("data/680.csv"))
    parser.add_argument("--encoder-batch-cap", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--accumulation-steps", type=int, default=16)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-summary-json", type=Path, default=Path("results/max_fill_summary.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker_candidate:
        if args.result_json is None:
            raise ValueError("--worker-candidate 模式必须提供 --result-json")
        res = run_worker_candidate(
            data_path=args.data,
            encoder_batch_cap=args.encoder_batch_cap,
            batch_size=args.batch_size,
            accumulation_steps=args.accumulation_steps,
            num_envs=args.num_envs,
            rollout_episodes=args.rollout_episodes,
            seed=args.seed,
            warmup=args.warmup,
        )
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    summary = run_adaptive_max_fill_sweep(
        base_data=args.data,
        seed=args.seed,
    )
    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[MAX-FILL] 完整调优报告已保存至: {args.output_summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
