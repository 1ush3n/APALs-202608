"""导出 ScheduleFree train_y checkpoint 对应的独立 eval_x checkpoint。"""

from __future__ import annotations

import argparse
import json
import math
import sys
import uuid
from pathlib import Path
from typing import Any

import torch


# 允许以 ``python scripts/...py`` 直接执行，并保持 Windows/Linux 一致的项目根目录解析。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.checkpoints import load_checkpoint
from runtime.initial_checkpoint_selection import sha256_file
from runtime.schedulefree_checkpoint import schedulefree_parameter_mode
from runtime.schedulefree_export import export_schedulefree_eval_payload
from training.async_eval_worker import load_checkpoint_agent_for_evaluation


def optional_finite_score(value: Any) -> float | None:
    """将回调指标转为严格 JSON 可表达的有限浮点数。"""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 ScheduleFree eval_x checkpoint")
    parser.add_argument("--checkpoint", required=True, help="源 Lightning checkpoint（必须是 train_y）")
    parser.add_argument("--output", required=True, help="新 eval_x checkpoint；目标不得已存在")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    audit_path = output.with_name("checkpoint_conversion.json")
    if not source.is_file():
        raise FileNotFoundError(f"源 checkpoint 不存在：{source}")
    if output.exists():
        raise FileExistsError(f"目标 checkpoint 已存在，拒绝覆盖：{output}")
    if audit_path.exists():
        raise FileExistsError(f"转换审计已存在，拒绝覆盖：{audit_path}")

    source_sha256 = sha256_file(source)
    checkpoint, _saved_config, agent = load_checkpoint_agent_for_evaluation(
        {"candidate_path": str(source), "candidate_sha256": source_sha256},
        torch.device("cpu"),
    )
    source_mode = schedulefree_parameter_mode(
        agent.optimizer,
        schedulefree_enabled=bool(getattr(agent, "use_schedule_free", False)),
    )
    if source_mode != "train_y":
        raise ValueError(f"源 checkpoint 参数态必须为 train_y，当前为 {source_mode}")

    exported = export_schedulefree_eval_payload(
        payload=checkpoint.payload,
        policy=agent.policy,
        optimizer=agent.optimizer,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(output, exported)
    output_sha256 = sha256_file(output)

    reloaded = load_checkpoint(output, map_location="cpu")
    payload = reloaded.payload
    optimizer_states = payload.get("optimizer_states", []) if isinstance(payload, dict) else []
    if not optimizer_states:
        raise RuntimeError("导出 checkpoint 缺少 optimizer_states")
    output_modes = [group.get("train_mode") for group in optimizer_states[0]["param_groups"]]
    if not output_modes or any(mode is not False for mode in output_modes):
        raise RuntimeError(f"导出 checkpoint 未保存 eval_x 参数态：{output_modes}")
    expected_state = exported["state_dict"]
    actual_state = payload["state_dict"]
    if expected_state.keys() != actual_state.keys() or any(
        not torch.equal(expected_state[name].cpu(), actual_state[name].cpu())
        for name in expected_state
    ):
        raise RuntimeError("导出 checkpoint 的 state_dict 与捕获的 eval_x 参数不一致")

    callbacks = payload.get("callbacks", {}) if isinstance(payload, dict) else {}
    rollout_state = callbacks.get("RolloutCheckpoint", {}) if isinstance(callbacks, dict) else {}
    source_selection_score = optional_finite_score(rollout_state.get("best_score"))
    metadata = checkpoint.metadata
    _atomic_write_json(
        audit_path,
        {
            "format_version": 1,
            "source_checkpoint": str(source),
            "source_sha256": source_sha256,
            "source_parameter_state": source_mode,
            "output_checkpoint": str(output),
            "output_sha256": output_sha256,
            "output_parameter_state": "eval_x",
            "source_episode": metadata.get("episode"),
            "source_selection_score": source_selection_score,
            "source_selection_score_status": (
                "recorded_in_rollout_callback"
                if source_selection_score is not None
                else "not_recorded_in_rollout_callback_async_selection"
            ),
            "model_spec": metadata.get("model_spec"),
            "conversion_device": "cpu",
        },
    )
    print(f"[ScheduleFreeExport] source={source}")
    print(f"[ScheduleFreeExport] output={output}")
    print(f"[ScheduleFreeExport] sha256={output_sha256}")
    print(f"[ScheduleFreeExport] audit={audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
