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
    resolved = {"max_episodes": updates, "kl_early_stop": 0.02}
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
    (config_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "evaluation_protocol": "training_auto_eval_only",
                "model_spec": model_spec,
                "training_manifest_path": str(manifest_path),
                "training_manifest_sha256": manifest_sha256,
                "runtime": {
                    "num_envs": 4,
                    "batch_size": 256,
                    "accumulation_steps": 16,
                    "lightning_precision": "bf16-mixed",
                    "autocast_dtype": "bfloat16",
                    "grad_scaler_enabled": False,
                    "worker_pointer_v2_replay_mode": (
                        "behavior_group_exact_v1"
                        if mode == "autoregressive_pressure_v2"
                        else None
                    ),
                    "requested_logical_batch_cap": (
                        256 if mode == "autoregressive_pressure_v2" else None
                    ),
                    "effective_logical_batch_cap": (
                        256 if mode == "autoregressive_pressure_v2" else None
                    ),
                    "rollout_group_upper_bound": (
                        4 if mode == "autoregressive_pressure_v2" else None
                    ),
                    "target_max_samples_per_optimizer_step": (
                        4096 if mode == "autoregressive_pressure_v2" else None
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
        writer.add_scalar("Eval/completion_rate", completion, step)
        writer.add_scalar("Rollout/StepsPerSecond", sps, step)
        writer.add_scalar("Memory/PeakAllocatedMiB", peak_memory_mib, step)
        writer.add_scalar("Policy/ApproxKL", 0.01, step)
        if mode == "autoregressive_pressure_v2":
            for tag, value in {
                "PPO/GradientsFinite": 1.0,
                "PointerV2/GradientNorm": 1.0,
                "PointerV2/GradientCoverage": 0.5,
                "PointerV2/PPOFirstRecomputeMaxAE": 1.0e-4,
                "PointerV2/PPOFirstRecomputeMAE": 1.0e-5,
                "PointerV2/AutocastEnabled": 1.0,
                "PointerV2/AutocastBF16": 1.0,
                "PointerV2/GradScalerEnabled": 0.0,
                "PointerV2/NonFiniteCount": 0.0,
            }.items():
                writer.add_scalar(tag, value, step)
    writer.flush()
    writer.close()


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
