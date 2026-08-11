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
LEGACY_MODE = "autoregressive"
REQUIRED_V2_SCALARS = (
    "Rollout/CompletionRate",
    "Eval/completion_rate",
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
    checks["runtime"] = _check(
        int(runtime.get("num_envs", 0)) == 4
        and runtime.get("lightning_precision") == "bf16-mixed"
        and runtime.get("autocast_dtype") == "bfloat16"
        and runtime.get("grad_scaler_enabled") is False,
        "运行必须为 4 环境 bf16 且禁用 GradScaler",
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
    required = REQUIRED_V2_SCALARS if require_v2 else (
        "Rollout/CompletionRate",
        "Eval/completion_rate",
        "Rollout/StepsPerSecond",
        "Memory/PeakAllocatedMiB",
    )
    missing = [tag for tag in required if len(scalars.get(tag, [])) < updates]
    finite_tags = [tag for tag in required if tag in scalars and not all(math.isfinite(value) for value in scalars[tag])]
    scalar_ok = updates > 0 and not missing and not finite_tags
    if scalar_ok:
        rollout_completion = scalars["Rollout/CompletionRate"][-updates:]
        eval_completion = scalars["Eval/completion_rate"][-updates:]
        scalar_ok = all(value == 1.0 for value in rollout_completion + eval_completion)
    checks["scalar_contract"] = _check(
        scalar_ok,
        "标量缺失、非有限或 rollout/eval 完成率非 100%",
        missing=missing,
        non_finite_tags=finite_tags,
        observed_rollout_completion=scalars.get("Rollout/CompletionRate", [])[-updates:],
        observed_eval_completion=scalars.get("Eval/completion_rate", [])[-updates:],
    )
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
        checks, scalars, updates = _run_checks(
            run_dir, expected_mode=V2_MODE, require_v2=True
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
