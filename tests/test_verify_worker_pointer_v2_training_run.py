from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


pytest.importorskip("tensorboard")


def _write_run(
    run_dir: Path,
    *,
    mode: str,
    updates: int = 3,
    completion: float = 1.0,
    sps: float = 100.0,
    peak_memory_mib: float = 1024.0,
    async_eval: bool = False,
    batch_size: int = 256,
    replay_mode: str | None = None,
) -> None:
    import torch
    import yaml
    from torch.utils.tensorboard import SummaryWriter

    project_root = run_dir.parents[2]
    manifest_path = project_root / "data" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {"files": [{"file": f"variant_{index:02d}.csv"} for index in range(80)]}
        ),
        encoding="utf-8",
    )
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    config_dir = run_dir / "configs"
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs" / "tensorboard"
    config_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    resolved = {
        "max_episodes": updates,
        "kl_early_stop": 0.02,
        "async_eval_enabled": async_eval,
        "async_eval_submit_every_episodes": 2,
        "batch_size": batch_size,
        "accumulation_steps": 16,
        "num_envs": 16 if mode == "autoregressive_pressure_v2_fast_exact" else 4,
        "worker_pointer_v2_replay_mode": replay_mode or (
            "behavior_group_exact_gpu_template_v2"
            if mode == "autoregressive_pressure_v2_fast_exact"
            else "behavior_group_exact_v1"
        ),
        "worker_pointer_v2_rollout_group_upper_bound": (
            16 if mode == "autoregressive_pressure_v2_fast_exact" else 4
        ),
    }
    (config_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved), encoding="utf-8"
    )
    model_spec = {
        "team_selection_mode": mode,
        "policy_action_scope": "operation_station_worker",
        "worker_pointer_context_version": "pressure_v2_normwork_headcount_physicalwait_v1",
        "worker_pointer_pressure_temperature": 1.0,
        "worker_pointer_supply_epsilon": 1.0e-6,
        "worker_pointer_wait_discount_mode": "physical_wait_exponential_v1",
    }
    is_v2 = mode in {
        "autoregressive_pressure_v2",
        "autoregressive_pressure_v2_fast_exact",
    }
    is_fast_exact = mode == "autoregressive_pressure_v2_fast_exact"
    (config_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "evaluation_protocol": "training_auto_eval_only",
                "model_spec": model_spec,
                "training_manifest_path": str(manifest_path),
                "training_manifest_sha256": manifest_sha256,
                "runtime": {
                    "num_envs": 16 if is_fast_exact else 4,
                    "batch_size": batch_size,
                    "accumulation_steps": 16,
                    "lightning_precision": "bf16-mixed",
                    "autocast_dtype": "bfloat16",
                    "grad_scaler_enabled": False,
                    "worker_pointer_v2_replay_mode": replay_mode or (
                        "behavior_group_exact_gpu_template_v2"
                        if is_fast_exact
                        else "behavior_group_exact_v1"
                        if is_v2
                        else None
                    ),
                    "requested_logical_batch_cap": (
                        batch_size if is_v2 else None
                    ),
                    "effective_logical_batch_cap": (
                        batch_size if is_v2 else None
                    ),
                    "rollout_group_upper_bound": (
                        16 if is_fast_exact else 4 if is_v2 else None
                    ),
                    "target_max_samples_per_optimizer_step": (
                        batch_size * 16 if is_v2 else None
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    torch.save({"apal_metadata": {"model_spec": model_spec}}, checkpoint_dir / "last.ckpt")
    writer = SummaryWriter(log_dir)
    for step in range(updates):
        writer.add_scalar("Rollout/CompletionRate", completion, step)
        if not async_eval:
            writer.add_scalar("Eval/completion_rate", completion, step)
        writer.add_scalar("Rollout/StepsPerSecond", sps, step)
        writer.add_scalar("Memory/PeakAllocatedMiB", peak_memory_mib, step)
        writer.add_scalar("Policy/ApproxKL", 0.01, step)
        if is_v2:
            for tag, value in {
                "PPO/GradientsFinite": 1.0,
                "PointerV2/GradientNorm": 1.0,
                "PointerV2/GradientCoverage": 0.5,
                "PointerV2/AutocastEnabled": 1.0,
                "PointerV2/AutocastBF16": 1.0,
                "PointerV2/GradScalerEnabled": 0.0,
                "PointerV2/NonFiniteCount": 0.0,
            }.items():
                writer.add_scalar(tag, value, step)
            if replay_mode != "batched_vectorized_v2":
                writer.add_scalar("PointerV2/PPOFirstRecomputeMaxAE", 1.0e-4, step)
                writer.add_scalar("PointerV2/PPOFirstRecomputeMAE", 1.0e-5, step)
            else:
                writer.add_scalar("V2/BatchedReplayUpdateSeconds", 1.0, step)
    writer.flush()
    writer.close()
    if async_eval:
        async_root = checkpoint_dir / "async_eval"
        done_dir = async_root / "queue" / "done"
        result_dir = async_root / "results"
        done_dir.mkdir(parents=True)
        result_dir.mkdir(parents=True)
        result = {
            "episode": updates,
            "eligible": completion,
            "complete": completion,
            "selection_score": 100.0,
            "makespan": 100.0,
        }
        (result_dir / f"episode_{updates:06d}.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
        (done_dir / f"episode_{updates:06d}.json").write_text(
            json.dumps({"episode": updates, "result": result}), encoding="utf-8"
        )


def test_training_gate_report_passes_and_compares_legacy(tmp_path: Path) -> None:
    from scripts.verify_worker_pointer_v2_training_run import evaluate_training_run

    v2_run = tmp_path / "results" / "initial_worker_pointer_v2_exploratory" / "v2"
    legacy_run = tmp_path / "results" / "initial_worker_pointer_v2_exploratory" / "legacy"
    _write_run(v2_run, mode="autoregressive_pressure_v2", sps=90.0, peak_memory_mib=1300.0)
    _write_run(legacy_run, mode="autoregressive", sps=100.0, peak_memory_mib=1000.0)

    report = evaluate_training_run(v2_run, legacy_run_dir=legacy_run)

    assert report["status"] == "passed"
    assert report["checks"]["legacy_performance_comparison"]["passed"] is True
    report_path = v2_run / "artifacts" / "training_gate_report.json"
    assert report_path.is_file()
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "passed"


def test_training_gate_report_fails_closed_for_incomplete_evaluation(tmp_path: Path) -> None:
    from scripts.verify_worker_pointer_v2_training_run import evaluate_training_run

    v2_run = tmp_path / "results" / "initial_worker_pointer_v2_exploratory" / "v2"
    _write_run(v2_run, mode="autoregressive_pressure_v2", completion=0.5)

    report = evaluate_training_run(v2_run)

    assert report["status"] == "failed"
    assert report["checks"]["scalar_contract"]["passed"] is False


def test_training_gate_report_fails_closed_for_missing_group_replay_semantics(
    tmp_path: Path,
) -> None:
    from scripts.verify_worker_pointer_v2_training_run import evaluate_training_run

    v2_run = tmp_path / "results" / "initial_worker_pointer_v2_exploratory" / "v2"
    _write_run(v2_run, mode="autoregressive_pressure_v2")
    manifest_path = v2_run / "configs" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime"].pop("worker_pointer_v2_replay_mode")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = evaluate_training_run(v2_run)

    assert report["status"] == "failed"
    assert report["checks"]["runtime"]["passed"] is False


def test_fast_exact_async_training_gate_uses_drained_queue_results(
    tmp_path: Path,
) -> None:
    from scripts.verify_worker_pointer_v2_training_run import evaluate_training_run

    run_dir = tmp_path / "results" / "initial_worker_pointer_v2_fast_exact" / "run"
    _write_run(
        run_dir,
        mode="autoregressive_pressure_v2_fast_exact",
        async_eval=True,
    )

    report = evaluate_training_run(run_dir)

    assert report["status"] == "passed"
    assert report["checks"]["model_mode"]["passed"] is True
    assert report["checks"]["runtime"]["passed"] is True
    assert report["checks"]["async_evaluation"]["passed"] is True


def test_fast_exact_training_gate_respects_resolved_cli_batch_size(
    tmp_path: Path,
) -> None:
    from scripts.verify_worker_pointer_v2_training_run import evaluate_training_run

    run_dir = tmp_path / "results" / "initial_worker_pointer_v2_fast_exact" / "run"
    _write_run(
        run_dir,
        mode="autoregressive_pressure_v2_fast_exact",
        async_eval=True,
        batch_size=128,
    )

    report = evaluate_training_run(run_dir)

    assert report["status"] == "passed"
    assert report["checks"]["runtime"]["passed"] is True


def test_training_gate_accepts_batched_v2_without_group_contract(tmp_path: Path) -> None:
    from scripts.verify_worker_pointer_v2_training_run import evaluate_training_run

    run_dir = tmp_path / "results" / "batched_v2" / "run"
    _write_run(
        run_dir,
        mode="autoregressive_pressure_v2",
        replay_mode="batched_vectorized_v2",
    )

    report = evaluate_training_run(run_dir)

    assert report["status"] == "passed"
    assert report["checks"]["v2_numerical_contract"]["passed"] is True
