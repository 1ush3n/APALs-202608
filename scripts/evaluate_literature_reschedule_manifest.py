from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.literature.common import LiteraturePolicyAdapter
from baselines.literature.evaluate_literature_baseline import _build_model, _load_checkpoint
from configs import configs
from environment import AirLineEnv_Graph
from runtime.hydra_config import ExtraArgument, HydraCliError, hydra_help, initialize_hydra_runtime, should_show_help
from runtime.initial_worker_mapping import apply_initial_worker_mapping
from runtime.paths import resolve_workspace_path
from runtime.reschedule_eval import evaluate_reschedule_model
from runtime.reschedule_manifest import REAL_INSTANCE_IDS, load_reschedule_manifest, validate_r5_manifest_assets


EXTRA_ARGS = {
    "model_path": ExtraArgument(required=True, help="Graph-DDQN-APAL 或 Simple-HeteroGAT-PPO checkpoint"),
    "manifest_path": ExtraArgument(required=True, help="r5 manifest"),
    "output_dir": ExtraArgument(default="results/literature_reschedule_manifest_eval", help="输出目录"),
    "temperature": ExtraArgument(default=0.0, help="r5 正式评估必须为 0"),
}


def evaluate_literature_r5_manifest(
    *,
    model_path: Path,
    manifest_path: Path,
    output_dir: Path,
    temperature: float = 0.0,
) -> dict[str, Any]:
    if abs(float(temperature)) > 1e-12:
        raise ValueError("r5 学习型 baseline 最终评估必须使用 temperature=0")
    if not torch.cuda.is_available():
        raise RuntimeError("r5 学习型 baseline 最终评估必须使用 CUDA")

    checkpoint = _load_checkpoint(model_path)
    saved_config = checkpoint.get("config")
    if isinstance(saved_config, dict):
        configs.update_from_dict(saved_config)
    manifest = load_reschedule_manifest(manifest_path)
    validate_r5_manifest_assets(manifest)
    if tuple(entry.instance_id for entry in manifest.filter(split="eval")) != REAL_INSTANCE_IDS:
        raise ValueError(f"r5 最终评估必须使用四个真实实例: {REAL_INSTANCE_IDS}")

    device = torch.device("cuda")
    model = _build_model(checkpoint, device)
    adapter = LiteraturePolicyAdapter(model, device)
    backup = {
        key: getattr(configs, key, None)
        for key in (
            "enable_reschedule_mode",
            "reschedule_async_protocol",
            "reschedule_manifest_path",
            "data_file_path",
            "reschedule_baseline_schedule_path",
            "reschedule_eval_scenario_path",
            "reschedule_eval_instance_id",
            "verbose_reschedule_eval_progress",
        )
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    try:
        configs.enable_reschedule_mode = True
        configs.reschedule_async_protocol = "r5_task_delay_v1"
        configs.reschedule_manifest_path = str(manifest.path)
        configs.verbose_reschedule_eval_progress = False
        for entry in manifest.filter(split="eval"):
            assert entry.scenario_path is not None
            configs.data_file_path = str(entry.data_path)
            configs.reschedule_baseline_schedule_path = str(entry.baseline_schedule_path)
            configs.reschedule_eval_scenario_path = str(entry.scenario_path)
            configs.reschedule_eval_instance_id = entry.instance_id
            apply_initial_worker_mapping(configs, entry.data_path, explicit_fields=set())
            env = AirLineEnv_Graph(str(entry.data_path), seed=int(configs.seed))
            result = evaluate_reschedule_model(
                env,
                adapter,
                num_runs=None,
                temperature=0.0,
                scenario_ids=None,
                skip_value_estimation=True,
            )
            summary = dict(getattr(evaluate_reschedule_model, "last_metrics", {}) or {})
            scenario_rows = list(getattr(evaluate_reschedule_model, "last_scenario_metrics", []) or [])
            for row in scenario_rows:
                rows.append({"method": str(checkpoint.get("algorithm", "literature_baseline")), "instance_id": entry.instance_id, **row})
            summaries[entry.instance_id] = {
                "scenario_count": len(scenario_rows),
                "avg_makespan": float(result[0]),
                "avg_selection_score": float(summary.get("selection_score", 0.0)),
                "eligible_rate": float(summary.get("eligible_rate", 0.0)),
            }
    finally:
        for key, value in backup.items():
            setattr(configs, key, value)

    if len(rows) != 36:
        raise ValueError(f"r5 学习型 baseline 必须输出 36 个场景，实际为 {len(rows)}")
    pd.DataFrame(rows).to_csv(output_dir / "literature_reschedule_eval.csv", index=False)
    payload = {
        "model_path": str(model_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "algorithm": str(checkpoint.get("algorithm", "literature_baseline")),
        "temperature": 0.0,
        "scenario_count": len(rows),
        "rows": rows,
        "summaries": summaries,
    }
    (output_dir / "literature_reschedule_eval_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


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
            default_experiment="reschedule_task_delay_r5",
            extra_arguments=EXTRA_ARGS,
        )
        payload = evaluate_literature_r5_manifest(
            model_path=resolve_workspace_path(args.model_path),
            manifest_path=resolve_workspace_path(args.manifest_path),
            output_dir=resolve_workspace_path(args.output_dir),
            temperature=float(args.temperature),
        )
    except (HydraCliError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"[CLI] {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: value for key, value in payload.items() if key not in {"rows", "summaries"}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
