"""WorkerPointer v2 训练运行的 fail-closed 产物审计。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml
from tensorboard.backend.event_processing import event_accumulator


V2_MODE = "autoregressive_pressure_v2"
FAST_EXACT_MODE = "autoregressive_pressure_v2_fast_exact"
V2_MODES = frozenset({V2_MODE, FAST_EXACT_MODE})
LEGACY_MODE = "autoregressive"
REQUIRED_V2_SCALARS = (
    "Rollout/CompletionRate",
    "Rollout/StepsPerSecond",
    "Memory/PeakAllocatedMiB",
    "Policy/ApproxKL",
    "PPO/GradientsFinite",
    "PointerV2/GradientNorm",
    "PointerV2/GradientCoverage",
    "PointerV2/PPOFirstRecomputeMAE",
    "PointerV2/PPOFirstRecomputeMaxAE",
    "PointerV2/AutocastEnabled",
    "PointerV2/AutocastBF16",
    "PointerV2/GradScalerEnabled",
    "PointerV2/NonFiniteCount",
)


def _check_async_evaluation(run_dir: Path) -> dict[str, Any]:
    """以持久化队列产物审计后台验证，不依赖训练进程内的同步标量。"""
    root = run_dir / "checkpoints" / "async_eval"
    failed_paths = sorted((root / "queue" / "failed").glob("*.json"))
    pending_paths = sorted((root / "queue" / "pending").glob("*.json"))
    running_paths = sorted((root / "queue" / "running").glob("*.json"))
    done_paths = sorted((root / "queue" / "done").glob("*.json"))
    invalid_results: list[str] = []
    episodes: list[int] = []
    for done_path in done_paths:
        payload = json.loads(done_path.read_text(encoding="utf-8-sig"))
        result = payload.get("result") if isinstance(payload, Mapping) else None
        if not isinstance(result, Mapping):
            invalid_results.append(done_path.name)
            continue
        episode = int(payload.get("episode", result.get("episode", -1)))
        eligible = float(result.get("eligible", result.get("complete", 0.0)))
        score = float(result.get("selection_score", float("inf")))
        result_path = root / "results" / f"episode_{episode:06d}.json"
        if episode < 0 or eligible < 1.0 - 1.0e-9 or not math.isfinite(score) or not result_path.is_file():
            invalid_results.append(done_path.name)
            continue
        episodes.append(episode)
    passed = bool(
        not failed_paths
        and not pending_paths
        and not running_paths
        and done_paths
        and not invalid_results
        and len(episodes) == len(set(episodes))
    )
    return _check(
        passed,
        "异步验证队列必须排空，且所有已提交任务均成功、合法并发布结果",
        done_episodes=episodes,
        failed_jobs=[path.name for path in failed_paths],
        pending_jobs=[path.name for path in pending_paths],
        running_jobs=[path.name for path in running_paths],
        invalid_results=invalid_results,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float_values(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values]


def _median_last_two(values: Sequence[float]) -> float:
    tail = sorted(_as_float_values(values)[-2:])
    if not tail:
        raise ValueError("缺少用于性能比较的标量")
    return float(sum(tail) / len(tail))


def _load_scalars(run_dir: Path) -> dict[str, list[float]]:
    event_paths = sorted((run_dir / "logs" / "tensorboard").rglob("events.out.tfevents.*"))
    if not event_paths:
        raise FileNotFoundError("未找到 TensorBoard event 文件")
    accumulator = event_accumulator.EventAccumulator(
        str(event_paths[0]),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()
    return {
        tag: [float(event.value) for event in accumulator.Scalars(tag)]
        for tag in accumulator.Tags().get("scalars", [])
    }


def _check(condition: bool, detail: str, **values: Any) -> dict[str, Any]:
    return {"passed": bool(condition), "detail": detail, **values}


def _resolve_manifest_path(run_dir: Path, value: object) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    # run manifest 记录的是训练启动目录相对路径；审计器首先按当前项目根解析，
    # 仅在该路径不存在时回退到 run 目录，避免把 data/... 错解为 run/data/...。
    project_relative = (Path.cwd() / path).resolve()
    return project_relative if project_relative.exists() else (run_dir / path).resolve()


def _load_run_inputs(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[float]]]:
    manifest_path = run_dir / "configs" / "run_manifest.json"
    config_path = run_dir / "configs" / "resolved_config.yaml"
    checkpoint_path = run_dir / "checkpoints" / "last.ckpt"
    missing = [str(path) for path in (manifest_path, config_path, checkpoint_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少运行必需产物: " + ", ".join(missing))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(manifest, dict) or not isinstance(config, dict):
        raise ValueError("运行 manifest 或 resolved config 格式无效")
    return manifest, config, _load_scalars(run_dir)


def _run_checks(
    run_dir: Path,
    *,
    expected_mode: str,
    require_v2: bool,
) -> tuple[dict[str, Any], dict[str, list[float]], int]:
    manifest, config, scalars = _load_run_inputs(run_dir)
    checks: dict[str, Any] = {}
    model_spec = manifest.get("model_spec")
    runtime = manifest.get("runtime")
    if not isinstance(model_spec, Mapping) or not isinstance(runtime, Mapping):
        raise ValueError("run manifest 缺少 model_spec 或 runtime")
    checks["evaluation_protocol"] = _check(
        manifest.get("evaluation_protocol") == "training_auto_eval_only",
        "运行必须仅标记为 training_auto_eval_only",
        observed=manifest.get("evaluation_protocol"),
    )
    checks["model_mode"] = _check(
        model_spec.get("team_selection_mode") == expected_mode
        and model_spec.get("policy_action_scope") == "operation_station_worker",
        "模型模式或动作作用域不符合训练门要求",
        observed_mode=model_spec.get("team_selection_mode"),
        observed_scope=model_spec.get("policy_action_scope"),
    )
    configured_num_envs = int(config.get("num_envs", runtime.get("num_envs", 0)))
    runtime_ok = (
        int(runtime.get("num_envs", 0)) == configured_num_envs
        and configured_num_envs > 0
        and runtime.get("lightning_precision") == "bf16-mixed"
        and runtime.get("autocast_dtype") == "bfloat16"
        and runtime.get("grad_scaler_enabled") is False
    )
    if require_v2:
        is_fast_exact = expected_mode == FAST_EXACT_MODE
        configured_batch_size = int(
            config.get("batch_size", runtime.get("batch_size", 0))
        )
        configured_accumulation = int(
            config.get("accumulation_steps", runtime.get("accumulation_steps", 0))
        )
        expected_replay_mode = str(
            config.get(
                "worker_pointer_v2_replay_mode",
                "behavior_group_exact_gpu_template_v2"
                if is_fast_exact
                else "behavior_group_exact_v1",
            )
        )
        expected_group_bound = int(
            config.get(
                "worker_pointer_v2_rollout_group_upper_bound",
                16 if is_fast_exact else 4,
            )
        )
        runtime_ok = runtime_ok and (
            configured_batch_size > 0
            and configured_accumulation > 0
            and int(runtime.get("batch_size", 0)) == configured_batch_size
            and int(runtime.get("accumulation_steps", 0)) == configured_accumulation
            and runtime.get("worker_pointer_v2_replay_mode")
            == expected_replay_mode
            and int(runtime.get("requested_logical_batch_cap", 0))
            == configured_batch_size
            and int(runtime.get("effective_logical_batch_cap", 0))
            == configured_batch_size
            and int(runtime.get("rollout_group_upper_bound", 0)) == expected_group_bound
            and int(runtime.get("target_max_samples_per_optimizer_step", 0))
            == configured_batch_size * configured_accumulation
        )
    checks["runtime"] = _check(
        runtime_ok,
        "运行参数、bf16、GradScaler 与所选 v2 重放模式必须一致",
        observed=dict(runtime),
    )
    manifest_file = _resolve_manifest_path(run_dir, manifest.get("training_manifest_path"))
    manifest_sha = manifest.get("training_manifest_sha256")
    manifest_ok = bool(
        manifest_file is not None
        and manifest_file.is_file()
        and isinstance(manifest_sha, str)
        and _sha256(manifest_file) == manifest_sha
    )
    checks["training_manifest"] = _check(
        manifest_ok,
        "训练 manifest 路径或 SHA-256 不一致",
        path=str(manifest_file) if manifest_file else None,
        expected_sha256=manifest_sha,
        observed_sha256=_sha256(manifest_file) if manifest_file and manifest_file.is_file() else None,
    )
    if require_v2 and manifest_ok:
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
        files = payload.get("files") if isinstance(payload, Mapping) else None
        names = [item.get("file") for item in files] if isinstance(files, list) else []
        checks["manifest_dataset_pool"] = _check(
            len(names) == 80 and all(isinstance(name, str) and name.startswith("variant_") for name in names),
            "真实训练池必须恰为 80 个 manifest 声明的 variant CSV",
            declared_count=len(names),
            non_variant_count=sum(not isinstance(name, str) or not name.startswith("variant_") for name in names),
        )
    checkpoint = torch.load(run_dir / "checkpoints" / "last.ckpt", map_location="cpu")
    metadata = checkpoint.get("apal_metadata") if isinstance(checkpoint, Mapping) else None
    checkpoint_spec = metadata.get("model_spec") if isinstance(metadata, Mapping) else None
    checks["checkpoint_metadata"] = _check(
        isinstance(checkpoint_spec, Mapping) and dict(checkpoint_spec) == dict(model_spec),
        "last.ckpt 的 apal_metadata.model_spec 缺失或与 run manifest 不一致",
    )
    updates = int(config.get("max_episodes", 0))
    checks["expected_updates"] = _check(
        updates > 0,
        "resolved config 缺少正数 max_episodes",
        expected_updates=updates,
    )
    async_eval_enabled = bool(config.get("async_eval_enabled", False))
    required = REQUIRED_V2_SCALARS if require_v2 else (
        "Rollout/CompletionRate",
        "Rollout/StepsPerSecond",
        "Memory/PeakAllocatedMiB",
    )
    if not async_eval_enabled:
        required = (*required, "Eval/completion_rate")
    missing = [tag for tag in required if len(scalars.get(tag, [])) < updates]
    finite_tags = [tag for tag in required if tag in scalars and not all(math.isfinite(value) for value in scalars[tag])]
    scalar_ok = updates > 0 and not missing and not finite_tags
    if scalar_ok:
        rollout_completion = scalars["Rollout/CompletionRate"][-updates:]
        eval_completion = (
            []
            if async_eval_enabled
            else scalars["Eval/completion_rate"][-updates:]
        )
        scalar_ok = all(value == 1.0 for value in rollout_completion + eval_completion)
    checks["scalar_contract"] = _check(
        scalar_ok,
        "标量缺失、非有限或 rollout/eval 完成率非 100%",
        missing=missing,
        non_finite_tags=finite_tags,
        observed_rollout_completion=scalars.get("Rollout/CompletionRate", [])[-updates:],
        observed_eval_completion=scalars.get("Eval/completion_rate", [])[-updates:],
    )
    if async_eval_enabled:
        checks["async_evaluation"] = _check_async_evaluation(run_dir)
    if require_v2 and scalar_ok:
        kl_limit = float(config.get("kl_early_stop", 0.0))
        v2_ok = (
            all(value == 1.0 for value in scalars["PPO/GradientsFinite"][-updates:])
            and all(value > 0.0 for value in scalars["PointerV2/GradientNorm"][-updates:])
            and all(value > 0.0 for value in scalars["PointerV2/GradientCoverage"][-updates:])
            and max(scalars["PointerV2/PPOFirstRecomputeMaxAE"][-updates:]) <= 1.0e-3
            and all(value == 1.0 for value in scalars["PointerV2/AutocastEnabled"][-updates:])
            and all(value == 1.0 for value in scalars["PointerV2/AutocastBF16"][-updates:])
            and all(value == 0.0 for value in scalars["PointerV2/GradScalerEnabled"][-updates:])
            and all(value == 0.0 for value in scalars["PointerV2/NonFiniteCount"][-updates:])
            and max(scalars["Policy/ApproxKL"][-updates:]) <= kl_limit
        )
        checks["v2_numerical_contract"] = _check(
            v2_ok,
            "v2 梯度、重算、AMP、KL 或非有限计数不符合门槛",
            kl_early_stop=kl_limit,
            max_kl=max(scalars["Policy/ApproxKL"][-updates:]),
            max_recompute_error=max(scalars["PointerV2/PPOFirstRecomputeMaxAE"][-updates:]),
        )
    return checks, scalars, updates


def evaluate_training_run(run_dir: Path, *, legacy_run_dir: Path | None = None) -> dict[str, Any]:
    """审计 v2 训练产物，并将唯一结论写入运行目录 artifacts。"""
    run_dir = Path(run_dir).resolve()
    report_path = run_dir / "artifacts" / "training_gate_report.json"
    checks: dict[str, Any] = {}
    try:
        manifest_path = run_dir / "configs" / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        model_spec = manifest.get("model_spec") if isinstance(manifest, Mapping) else None
        observed_mode = (
            str(model_spec.get("team_selection_mode", ""))
            if isinstance(model_spec, Mapping)
            else ""
        )
        if observed_mode not in V2_MODES:
            raise ValueError(f"不支持的 WorkerPointer v2 模式: {observed_mode!r}")
        checks, scalars, updates = _run_checks(
            run_dir, expected_mode=observed_mode, require_v2=True
        )
        if legacy_run_dir is not None:
            legacy_checks, legacy_scalars, legacy_updates = _run_checks(
                Path(legacy_run_dir).resolve(), expected_mode=LEGACY_MODE, require_v2=False
            )
            legacy_ok = all(item["passed"] for item in legacy_checks.values())
            v2_sps = _median_last_two(scalars["Rollout/StepsPerSecond"])
            legacy_sps = _median_last_two(legacy_scalars["Rollout/StepsPerSecond"])
            v2_peak = max(scalars["Memory/PeakAllocatedMiB"])
            legacy_peak = max(legacy_scalars["Memory/PeakAllocatedMiB"])
            comparison_ok = (
                legacy_ok
                and updates >= 3
                and legacy_updates >= 3
                and v2_sps >= legacy_sps * 0.85
                and v2_peak <= legacy_peak + 512.0
                and v2_peak < 7680.0
            )
            checks["legacy_performance_comparison"] = _check(
                comparison_ok,
                "v2 与 legacy 的 SPS 或峰值显存比较未达标",
                v2_median_last_two_sps=v2_sps,
                legacy_median_last_two_sps=legacy_sps,
                v2_peak_memory_mib=v2_peak,
                legacy_peak_memory_mib=legacy_peak,
                v2_updates=updates,
                legacy_updates=legacy_updates,
                legacy_checks=legacy_checks,
            )
        else:
            checks["absolute_peak_memory"] = _check(
                max(scalars["Memory/PeakAllocatedMiB"]) < 7680.0,
                "峰值显存必须低于 7.5 GiB",
                peak_memory_mib=max(scalars["Memory/PeakAllocatedMiB"]),
            )
    except Exception as exc:  # 审计器必须把任何解析异常收敛为 fail-closed 报告。
        checks["artifact_readability"] = _check(False, f"无法验证训练产物: {type(exc).__name__}: {exc}")
    status = "passed" if checks and all(item["passed"] for item in checks.values()) else "failed"
    report = {
        "status": status,
        "run_dir": str(run_dir),
        "legacy_run_dir": str(Path(legacy_run_dir).resolve()) if legacy_run_dir else None,
        "checks": checks,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="验证 WorkerPointer v2 PPO 训练运行")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--legacy-run-dir", type=Path)
    args = parser.parse_args()
    report = evaluate_training_run(args.run_dir, legacy_run_dir=args.legacy_run_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
