"""为单个初始调度训练目录生成可追溯的诊断归档。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar_summary(event_path: Path) -> dict[str, Any]:
    accumulator = EventAccumulator(str(event_path))
    accumulator.Reload()
    values = accumulator.Scalars("Eval/makespan")
    if not values:
        raise ValueError(f"TensorBoard 缺少 Eval/makespan: {event_path}")
    best = min(values, key=lambda item: float(item.value))
    last = values[-1]
    return {
        "event_path": str(event_path),
        "eval_count": len(values),
        "first_episode": int(values[0].step) + 1,
        "first_makespan": float(values[0].value),
        "best_episode": int(best.step) + 1,
        "best_makespan": float(best.value),
        "last_episode": int(last.step) + 1,
        "last_makespan": float(last.value),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="归档初始调度训练诊断")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--classification", required=True)
    parser.add_argument("--decision", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    config_path = run_dir / "configs" / "resolved_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    event_path = next(run_dir.rglob("events.out.tfevents.*"), None)
    if event_path is None:
        raise FileNotFoundError("未找到 TensorBoard event")
    tensorboard = scalar_summary(event_path)
    checkpoints: dict[str, dict[str, Any]] = {}
    for name in ("best.ckpt", "last.ckpt"):
        path = run_dir / "checkpoints" / name
        if path.is_file():
            checkpoints[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    max_episodes = int(config.get("max_episodes", 0))
    payload = {
        "run_dir": str(run_dir),
        "created_at": datetime.now().astimezone().isoformat(),
        "classification": args.classification,
        "decision": args.decision,
        "training": {
            "max_episodes": max_episodes,
            "policy_action_scope": config.get("policy_action_scope"),
            "workforce_binding_mode": config.get("workforce_binding_mode"),
            "workforce_preallocation_ratio": config.get("workforce_preallocation_ratio"),
            "graph_encoder_mode": config.get("graph_encoder_mode"),
            "actor_context_mode": config.get("actor_context_mode"),
            "batch_size": config.get("batch_size"),
            "accumulation_steps": config.get("accumulation_steps"),
            "num_envs": config.get("num_envs"),
            "training_manifest_path": config.get("training_manifest_path"),
        },
        "tensorboard": tensorboard,
        "checkpoints": checkpoints,
        "formal_eval": "未执行；训练期 Eval 不得替代四实例六次正式验证。",
        "training_manifest_evidence": "本地未保存训练时 manifest，不能据此升级为严格主表结果。",
    }
    (run_dir / "training_diagnostic_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "README.md").write_text(
        "# 初始调度训练诊断\n\n"
        f"- 分类：`{args.classification}`。\n"
        f"- 处置：{args.decision}。\n"
        f"- 训练期 Eval：best episode {tensorboard['best_episode']}，makespan={tensorboard['best_makespan']:.6f}；"
        f"last episode {tensorboard['last_episode']}，makespan={tensorboard['last_makespan']:.6f}。\n"
        f"- 配置：scope=`{config.get('policy_action_scope')}`，batch_size=`{config.get('batch_size')}`，"
        f"accumulation_steps=`{config.get('accumulation_steps')}`。\n"
        "- 仅为训练诊断，尚无四实例六次独立合法性验证，不得作为正式质量结论。\n"
        "- 本地未留存训练时五技能 manifest，训练池协议仍待服务器资产核验。\n",
        encoding="utf-8",
    )
    manifest_rows = []
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file() and item.name != "file_manifest.json"):
        manifest_rows.append({"path": str(path.relative_to(run_dir)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    (run_dir / "file_manifest.json").write_text(
        json.dumps({"files": manifest_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
