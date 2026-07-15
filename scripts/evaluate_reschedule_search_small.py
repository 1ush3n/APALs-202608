# -*- coding: utf-8 -*-
"""在单个 APAL 实例的低/中/高固定场景上运行小规模搜索评估。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import configs
from runtime.hydra_config import (
    ExtraArgument,
    HydraCliError,
    hydra_help,
    initialize_hydra_runtime,
    should_show_help,
)
from runtime.paths import resolve_workspace_path
from runtime.reschedule_manifest import load_reschedule_manifest
from scripts.evaluate_reschedule_rules import evaluate_reschedule_rules_manifest
from utils.reschedule import load_reschedule_scenarios, save_reschedule_scenarios


SEARCH_METHODS = ("Beam", "IG", "SA")
SEARCH_LEVELS = ("low", "medium", "high")

SEARCH_ARGS = {
    "manifest_path": ExtraArgument(required=True, help="包含目标实例、baseline 和60场景文件的 manifest"),
    "instance_id": ExtraArgument(required=True, help="单个目标实例，例如 real_680"),
    "scenario_index": ExtraArgument(default=0, help="每个等级选择同一编号场景；0 表示 *_000"),
    "seed": ExtraArgument(default=42, help="Beam/IG/SA 搜索固定种子"),
    "parallel_workers": ExtraArgument(default=3, help="本地并行进程数；建议16GB内存使用3"),
    "beam_width": ExtraArgument(default=4, help="Beam Search 候选解数量"),
    "beam_branch_factor": ExtraArgument(default=4, help="Beam Search 每个候选解的扰动分支数"),
    "beam_levels": ExtraArgument(default=4, help="Beam Search 最大展开层数"),
    "beam_patience": ExtraArgument(default=2, help="Beam Search 连续无改进提前停止层数"),
    "ig_iterations": ExtraArgument(default=80, help="Iterated Greedy 迭代次数"),
    "ig_destroy_ratio": ExtraArgument(default=0.10, help="Iterated Greedy 每次破坏的任务比例"),
    "ig_noise_sigma": ExtraArgument(default=0.20, help="Iterated Greedy 修复扰动强度"),
    "sa_iterations": ExtraArgument(default=120, help="Simulated Annealing 迭代次数"),
    "sa_initial_temp": ExtraArgument(default=0.05, help="Simulated Annealing 初始温度"),
    "sa_cooling": ExtraArgument(default=0.96, help="Simulated Annealing 降温系数"),
    "sa_min_temp": ExtraArgument(default=1.0e-4, help="Simulated Annealing 最小温度"),
    "verify_static_cache": ExtraArgument(default=False, help="是否校验静态约束缓存等价性"),
    "resume_partial": ExtraArgument(default=True, help="是否从相同 output_dir 的断点继续"),
    "force_rerun": ExtraArgument(default=False, help="是否清除断点并重新运行9个任务"),
    "flush_every": ExtraArgument(default=1, help="每完成多少个任务刷新断点"),
    "progress_interval": ExtraArgument(default=30.0, help="无任务完成时的心跳间隔（秒）"),
    "quiet": ExtraArgument(default=False, help="是否关闭进度输出"),
    "output_dir": ExtraArgument(default=None, help="固定输出目录；复用同一路径即可断点续算"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def select_three_level_scenarios(
    scenario_items: list[tuple[str, Any]],
    *,
    scenario_index: int,
) -> list[tuple[str, Any]]:
    """严格选择 low/medium/high 的同编号固定场景。"""
    index = int(scenario_index)
    if index < 0:
        raise ValueError("scenario_index 不能为负数")
    by_id: dict[str, Any] = {}
    for scenario_id, scenario in scenario_items:
        key = str(scenario_id)
        if key in by_id:
            raise ValueError(f"场景文件包含重复 scenario_id: {key}")
        by_id[key] = scenario
    expected_ids = [f"{level}_{index:03d}" for level in SEARCH_LEVELS]
    missing = [scenario_id for scenario_id in expected_ids if scenario_id not in by_id]
    if missing:
        raise ValueError(f"场景文件缺少小规模搜索所需场景: {missing}")
    return [(scenario_id, by_id[scenario_id]) for scenario_id in expected_ids]


def prepare_small_search_protocol(
    *,
    manifest_path: str | Path,
    instance_id: str,
    scenario_index: int,
    output_dir: Path,
    force_rerun: bool,
) -> tuple[Path, dict[str, Any]]:
    """生成不修改源 manifest 的三场景派生协议。"""
    source_manifest_path = resolve_workspace_path(manifest_path).resolve()
    manifest = load_reschedule_manifest(source_manifest_path)
    entry = manifest.get(str(instance_id))
    if entry.scenario_path is None:
        raise ValueError(f"{instance_id} 没有固定场景文件")

    source_scenario_path = entry.scenario_path.resolve()
    selected = select_three_level_scenarios(
        load_reschedule_scenarios(source_scenario_path),
        scenario_index=int(scenario_index),
    )
    protocol_dir = output_dir / "protocol"
    selected_scenario_path = protocol_dir / f"{instance_id}_search3_scenarios.csv"
    derived_manifest_path = protocol_dir / "search3_manifest.json"
    protocol_path = protocol_dir / "search3_protocol.json"

    provenance = {
        "protocol_version": 1,
        "kind": "apal_reschedule_search_three_scenarios",
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "instance_id": str(instance_id),
        "data_path": str(entry.data_path.resolve()),
        "data_sha256": _sha256(entry.data_path.resolve()),
        "baseline_schedule_path": str(entry.baseline_schedule_path.resolve()),
        "baseline_schedule_sha256": _sha256(entry.baseline_schedule_path.resolve()),
        "source_scenario_path": str(source_scenario_path),
        "source_scenario_sha256": _sha256(source_scenario_path),
        "scenario_index": int(scenario_index),
        "selected_scenario_ids": [scenario_id for scenario_id, _ in selected],
        "methods": list(SEARCH_METHODS),
    }
    if protocol_path.exists() and not bool(force_rerun):
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != provenance:
            raise RuntimeError(
                "output_dir 已包含不同输入的搜索断点；请更换 output_dir，"
                "或显式设置 force_rerun=true"
            )

    protocol_dir.mkdir(parents=True, exist_ok=True)
    save_reschedule_scenarios(selected_scenario_path, selected)
    derived_manifest = {
        "version": 1,
        "kind": "reschedule_dataset_manifest",
        "source_manifest_path": str(source_manifest_path),
        "scenario_protocol": provenance["kind"],
        "instances": [
            {
                "instance_id": str(instance_id),
                "split": "eval",
                "source": "real",
                "data_path": str(entry.data_path.resolve()),
                "baseline_schedule_path": str(entry.baseline_schedule_path.resolve()),
                "scenario_path": str(selected_scenario_path.resolve()),
                "num_tasks": entry.num_tasks,
                "baseline_makespan": entry.baseline_makespan,
                "status": "ready",
            }
        ],
    }
    _write_json_atomic(derived_manifest_path, derived_manifest)
    _write_json_atomic(protocol_path, provenance)
    return derived_manifest_path, provenance


def run_small_search(
    *,
    manifest_path: str | Path,
    instance_id: str,
    scenario_index: int,
    output_dir: Path,
    seed: int,
    parallel_workers: int,
    beam_width: int,
    beam_branch_factor: int,
    beam_levels: int,
    beam_patience: int,
    ig_iterations: int,
    ig_destroy_ratio: float,
    ig_noise_sigma: float,
    sa_iterations: int,
    sa_initial_temp: float,
    sa_cooling: float,
    sa_min_temp: float,
    verify_static_cache: bool,
    resume_partial: bool,
    force_rerun: bool,
    flush_every: int,
    progress_interval: float,
    quiet: bool,
) -> dict[str, Any]:
    derived_manifest_path, provenance = prepare_small_search_protocol(
        manifest_path=manifest_path,
        instance_id=instance_id,
        scenario_index=int(scenario_index),
        output_dir=output_dir,
        force_rerun=bool(force_rerun),
    )
    summary = evaluate_reschedule_rules_manifest(
        manifest_path=derived_manifest_path,
        instance_ids=[str(instance_id)],
        methods=list(SEARCH_METHODS),
        seed=int(seed),
        output_dir=output_dir,
        verbose=not bool(quiet),
        beam_width=int(beam_width),
        beam_branch_factor=int(beam_branch_factor),
        beam_levels=int(beam_levels),
        beam_patience=int(beam_patience),
        ig_iterations=int(ig_iterations),
        ig_destroy_ratio=float(ig_destroy_ratio),
        ig_noise_sigma=float(ig_noise_sigma),
        sa_iterations=int(sa_iterations),
        sa_initial_temp=float(sa_initial_temp),
        sa_cooling=float(sa_cooling),
        sa_min_temp=float(sa_min_temp),
        parallel_workers=int(parallel_workers),
        parallel_backend="process",
        verify_static_cache=bool(verify_static_cache),
        resume=bool(resume_partial),
        force_rerun=bool(force_rerun),
        flush_every=int(flush_every),
        show_progress=not bool(quiet),
        progress_interval=float(progress_interval),
    )
    compact = {
        "protocol": provenance,
        "row_count": int(summary.get("row_count", 0)),
        "summary_by_instance_method": summary.get("summary_by_instance_method", []),
    }
    _write_json_atomic(output_dir / "search3_summary.json", compact)
    return summary


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if should_show_help(raw_args):
        print(hydra_help(SEARCH_ARGS))
        return 0
    try:
        args = initialize_hydra_runtime(
            raw_args,
            target=configs,
            project_root=PROJECT_ROOT,
            default_experiment="reschedule_task_delay",
            extra_arguments=SEARCH_ARGS,
            create_run_context=False,
        )
        instance_id = str(args.instance_id)
        scenario_index = int(args.scenario_index)
        output_dir = (
            resolve_workspace_path(args.output_dir)
            if args.output_dir
            else PROJECT_ROOT
            / "results"
            / "04_reschedule_baselines"
            / "search_small_3scenario"
            / f"{instance_id}_idx{scenario_index:03d}_seed{int(args.seed)}"
        )
        summary = run_small_search(
            manifest_path=args.manifest_path,
            instance_id=instance_id,
            scenario_index=scenario_index,
            output_dir=output_dir,
            seed=int(args.seed),
            parallel_workers=int(args.parallel_workers),
            beam_width=int(args.beam_width),
            beam_branch_factor=int(args.beam_branch_factor),
            beam_levels=int(args.beam_levels),
            beam_patience=int(args.beam_patience),
            ig_iterations=int(args.ig_iterations),
            ig_destroy_ratio=float(args.ig_destroy_ratio),
            ig_noise_sigma=float(args.ig_noise_sigma),
            sa_iterations=int(args.sa_iterations),
            sa_initial_temp=float(args.sa_initial_temp),
            sa_cooling=float(args.sa_cooling),
            sa_min_temp=float(args.sa_min_temp),
            verify_static_cache=bool(args.verify_static_cache),
            resume_partial=bool(args.resume_partial),
            force_rerun=bool(args.force_rerun),
            flush_every=int(args.flush_every),
            progress_interval=float(args.progress_interval),
            quiet=bool(args.quiet),
        )
    except KeyboardInterrupt:
        print("\n[Interrupted] 已保留已完成任务，可使用相同命令继续。", file=sys.stderr)
        return 130
    except (HydraCliError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"[CLI] {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "row_count": int(summary.get("row_count", 0)),
                "expected_rows": 9,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
