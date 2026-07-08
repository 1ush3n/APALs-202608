from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from baselines.literature.common import LITERATURE_FEATURE_MODE, evaluate_graph_policy, resolve_project_path
from baselines.literature_dqn.train_graph_ddqn_apal import GraphDDQNAPAL
from baselines.literature_ppo.train_l2d_ppo_apal import L2DPPOAPAL
from configs import configs, load_config_files
from environment import AirLineEnv_Graph
from runtime.artifacts import resolve_run_output_dir, write_run_context_files, write_run_manifest
from runtime.hydra_config import ExtraArgument, HydraCliError, hydra_help, initialize_hydra_runtime, should_show_help
from runtime.seed import set_seed


EXTRA_ARGS = {
    "model_path": ExtraArgument(required=True, help="文献适配 baseline checkpoint"),
    "data_dir": ExtraArgument(default="data", help="数据文件所在目录"),
    "datasets": ExtraArgument(default=["283.csv"], help="数据集列表，例如 datasets=[283.csv,680.csv]"),
    "num_runs": ExtraArgument(default=1, help="重复 deterministic 评估次数"),
    "temperature": ExtraArgument(default=0.0, help="评估采样温度；0 表示确定性"),
    "output_dir": ExtraArgument(default=None, help="可选输出目录；缺省写入 runs artifacts"),
}


def _dataset_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise ValueError(f"无法解析 datasets 参数: {value!r}")


def _load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint 缺少 model_state_dict")
    if checkpoint.get("feature_mode") != LITERATURE_FEATURE_MODE:
        raise ValueError(
            f"checkpoint feature_mode={checkpoint.get('feature_mode')!r}，"
            f"当前只接受 {LITERATURE_FEATURE_MODE!r}"
        )
    return checkpoint


def _apply_model_config(checkpoint: dict[str, Any]) -> None:
    for name in ("hidden_dim", "num_gat_layers", "num_heads"):
        if checkpoint.get(name) is not None:
            setattr(configs, name, int(checkpoint[name]))
    for name in ("use_skill_hub", "skill_hub_bidirectional"):
        if checkpoint.get(name) is not None:
            setattr(configs, name, bool(checkpoint[name]))


def _build_model(checkpoint: dict[str, Any], device: torch.device) -> torch.nn.Module:
    _apply_model_config(checkpoint)
    model_type = str(checkpoint.get("model_type", ""))
    if model_type == "L2DPPOAPAL" or checkpoint.get("algorithm") == "L2D-PPO-APAL":
        model = L2DPPOAPAL(configs)
    elif model_type == "GraphDDQNAPAL" or checkpoint.get("algorithm") == "Graph-DDQN-APAL":
        model = GraphDDQNAPAL(configs)
    else:
        raise ValueError(f"未知文献 baseline checkpoint 类型: {model_type!r}/{checkpoint.get('algorithm')!r}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _save_eval_results(
    output_root: Path,
    *,
    method: str,
    dataset_name: str,
    metrics: dict[str, Any],
    schedule: list[Any],
    run_metrics: list[dict[str, Any]],
) -> None:
    output_dir = output_root / method / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    full_metrics = dict(metrics)
    full_metrics["runs"] = run_metrics
    (output_dir / "metrics.json").write_text(json.dumps(full_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = [
        {
            "TaskID": int(task_id),
            "StationID": int(station_id) + 1,
            "Team": str([int(worker) for worker in team]),
            "Start": float(start),
            "End": float(end),
            "Duration": float(end) - float(start),
        }
        for task_id, station_id, team, start, end in schedule
    ]
    pd.DataFrame(rows).to_csv(output_dir / "schedule.csv", index=False)
    detail_rows = []
    for run_idx, run_metric in enumerate(run_metrics):
        row = {"RunIdx": run_idx + 1, "Seed": int(run_metric.get("seed", int(getattr(configs, "seed", 42)) + run_idx))}
        row.update({key: value for key, value in run_metric.items() if isinstance(value, (int, float)) and key != "seed"})
        detail_rows.append(row)
    pd.DataFrame(detail_rows).to_csv(output_dir / "runs_detail.csv", index=False)


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
        )
        args.datasets = _dataset_list(args.datasets)
        checkpoint_path = resolve_project_path(args.model_path)
        checkpoint = _load_checkpoint(checkpoint_path)
        method = str(checkpoint.get("algorithm", "literature_baseline"))
        output_root, context = resolve_run_output_dir(
            configs,
            PROJECT_ROOT,
            default_legacy_dir="results/literature_baselines",
            run_subdir=Path("baselines") / "literature_eval",
            explicit_dir=getattr(args, "output_dir", None),
            section="artifacts",
        )
        extra = {
            "run_type": "baseline_eval",
            "artifact_kind": "literature_baseline",
            "method": method,
            "checkpoint": str(checkpoint_path.resolve()),
            "datasets": list(args.datasets),
        }
        if context is not None:
            write_run_context_files(context, configs, command="evaluate_literature_baseline", extra=extra)
        else:
            write_run_manifest(output_root, configs, command="evaluate_literature_baseline", extra=extra)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _build_model(checkpoint, device)
        summary_rows: list[dict[str, Any]] = []

        for dataset in args.datasets:
            dataset_path = Path(args.data_dir) / dataset
            dataset_path = resolve_project_path(dataset_path)
            scale_yaml = PROJECT_ROOT / "conf" / "env" / f"initial_bucket_{dataset_path.stem}.yaml"
            if scale_yaml.exists():
                load_config_files([str(scale_yaml)])
            set_seed(int(getattr(configs, "seed", 42)))
            env = AirLineEnv_Graph(data_path_or_dir=str(dataset_path), seed=int(getattr(configs, "seed", 42)))
            metrics, schedule, run_metrics = evaluate_graph_policy(
                model,
                env,
                device,
                seed=int(getattr(configs, "seed", 42)),
                num_runs=int(args.num_runs),
                temperature=float(args.temperature),
            )
            _save_eval_results(
                output_root,
                method=method,
                dataset_name=dataset_path.stem,
                metrics=metrics,
                schedule=schedule,
                run_metrics=run_metrics,
            )
            for run_idx, run_metric in enumerate(run_metrics):
                summary_rows.append(
                    {
                        "Dataset": dataset_path.stem,
                        "Method": method,
                        "Run": run_idx + 1,
                        "Seed": int(run_metric.get("seed", int(configs.seed) + run_idx)),
                        "Makespan": run_metric["makespan"],
                        "BalanceStd": run_metric["workload_balance_std"],
                        "WorkerUtil": run_metric["worker_utilization"],
                        "StationUtil": run_metric["station_utilization"],
                        "Time(s)": run_metric["inference_time"],
                        "Valid": run_metric["valid"],
                    }
                )
            print(f"[Eval][{method}] {dataset_path.stem} mk={metrics['makespan']:.2f} valid={metrics['valid']:.0f}", flush=True)

        summary_path = output_root / f"{method}_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"[*] 文献 baseline 评估汇总已导出: {summary_path}", flush=True)
    except (HydraCliError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"[CLI] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
