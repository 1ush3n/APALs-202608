"""审计初始调度中候选团队的一步反事实改进空间。

该工具不加载策略 checkpoint、不训练、不修改环境或既有实验结果。它在确定性
启发式轨迹上固定工序—工位，只比较候选 0 与合法单人替换团队执行一步后的真实
环境指标。结果仅是局部机会证据，不能替代完整初始调度验证。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import configs, load_config_files
from core.action_completion import EarliestFinishActionCompleter, TeamCandidates
from environment import AirLineEnv_Graph
from runtime.initial_worker_mapping import apply_initial_worker_mapping
from runtime.team_opportunity import evaluate_one_step_candidate


_SHARED_ENV_ATTRIBUTES = (
    "dataset_pool", "raw_data", "base_data", "base_task_x", "base_worker_x",
    "base_station_x", "task_static_feat", "worker_skill_matrix", "predecessors",
    "successors", "num_preds", "fixed_stations", "constraint_engine", "mean_task_time",
    "ideal_station_load", "ideal_makespan", "total_base_workload", "base_durations",
    "max_allowed_stations", "is_critical", "full_worker_efficiency",
    "full_worker_skill_matrix", "worker_efficiency", "worker_static_feat",
    "worker_feature_layout",
)


@dataclass(frozen=True)
class SelectedPair:
    """确定性轨迹中的一个可执行工序—工位与其合法团队候选。"""

    task_id: int
    station_id: int
    candidates: TeamCandidates


def _workspace_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def initial_objective_from_info(
    reward: float,
    done: bool,
    info: dict[str, Any],
) -> dict[str, float]:
    """提取初始调度一步动作的可比较成本；越小越好。"""
    makespan_penalty = float(info.get("makespan_penalty", 0.0))
    balance_penalty = float(info.get("std_penalty", 0.0))
    return {
        "objective": makespan_penalty + balance_penalty,
        "makespan_penalty": makespan_penalty,
        "balance_penalty": balance_penalty,
        "reward": float(reward),
        "done": float(done),
    }


def _configure(data_path: Path) -> None:
    load_config_files(
        [str(PROJECT_ROOT / "conf" / "experiment" / "initial_conditional_team_gate_prior_residual.yaml")],
        target=configs,
    )
    configs.data_file_path = str(data_path)
    configs.train_data_path_or_dir = str(data_path.parent)
    configs.randomize_durations = False
    configs.enable_dynamic_events = False
    configs.enable_station_breakdown = False
    configs.enable_material_delay = False
    configs.enable_online_duration_perturb = False
    configs.enable_worker_fatigue = False
    apply_initial_worker_mapping(configs, data_path, explicit_fields=set())


def _select_pair(
    env: AirLineEnv_Graph,
    obs: Any,
    completer: EarliestFinishActionCompleter,
    *,
    max_candidates: int,
) -> SelectedPair | None:
    task_mask, station_mask, worker_mask = env.get_masks()
    valid_tasks = torch.nonzero(~task_mask, as_tuple=False).reshape(-1).tolist()
    for task_id in sorted(int(value) for value in valid_tasks):
        valid_stations = torch.nonzero(~station_mask[task_id], as_tuple=False).reshape(-1).tolist()
        for station_id in sorted(int(value) for value in valid_stations):
            candidates = completer.enumerate_team_candidates(
                obs,
                task_id=task_id,
                station_id=station_id,
                worker_mask=worker_mask,
                max_candidates=max_candidates,
            )
            if candidates is not None:
                return SelectedPair(task_id, station_id, candidates)
    return None


def summarize_initial_opportunities(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总局部团队替换机会，并按技能瓶颈分层。"""
    if not rows:
        return {"audited_states": 0}
    improved = [row for row in rows if bool(row["has_single_swap_improvement"])]
    bottleneck = [row for row in rows if bool(row["skill_bottleneck"])]
    non_bottleneck = [row for row in rows if not bool(row["skill_bottleneck"])]

    def _mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "audited_states": len(rows),
        "states_with_at_least_two_current_candidates": sum(
            int(int(row["candidate_count"]) >= 2) for row in rows
        ),
        "states_with_better_single_swap": len(improved),
        "better_single_swap_rate": float(len(improved) / len(rows)),
        "best_single_objective_gain_mean": _mean(
            [float(row["best_single_objective_gain"]) for row in improved]
        ),
        "best_single_objective_gain_max": max(
            (float(row["best_single_objective_gain"]) for row in improved), default=0.0
        ),
        "improvement_rate_skill_bottleneck": (
            _mean([float(bool(row["has_single_swap_improvement"])) for row in bottleneck])
            if bottleneck else None
        ),
        "improvement_rate_non_bottleneck": (
            _mean([float(bool(row["has_single_swap_improvement"])) for row in non_bottleneck])
            if non_bottleneck else None
        ),
        "mean_component_gain_when_improved": {
            "makespan": _mean([float(row["best_single_gain_makespan"]) for row in improved]),
            "balance": _mean([float(row["best_single_gain_balance"]) for row in improved]),
        },
        "metric_note": (
            "收益基于同一初始调度状态、固定工序—工位、执行一步后的真实环境代价变化；"
            "不等同于完整 APAL 排程的最终 makespan 改进。"
        ),
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    _configure(args.data_path)
    env = AirLineEnv_Graph(data_path_or_dir=str(args.data_path), seed=int(args.seed))
    obs = env.reset(randomize_duration=False, randomize_workers=False, seed=int(args.seed))
    completer = EarliestFinishActionCompleter(configs)
    rows: list[dict[str, Any]] = []
    trajectory_step = 0

    while trajectory_step < int(args.max_trajectory_steps) and len(rows) < int(args.max_states):
        selected = _select_pair(
            env,
            obs,
            completer,
            max_candidates=int(args.max_candidates),
        )
        if selected is None:
            if not env.try_wait_for_resources():
                break
            obs = env._get_observation()
            continue

        task_mask, _station_mask, worker_mask = env.get_masks()
        del task_mask
        base_team = selected.candidates.teams[0]
        should_audit = (
            len(selected.candidates.teams) > 1
            and trajectory_step % max(1, int(args.state_stride)) == 0
        )
        if should_audit:
            action_prefix = (selected.task_id, selected.station_id)
            base = evaluate_one_step_candidate(
                env,
                action=(*action_prefix, base_team),
                metric_extractor=initial_objective_from_info,
                shared_attribute_names=_SHARED_ENV_ATTRIBUTES,
            )
            alternatives = [
                evaluate_one_step_candidate(
                    env,
                    action=(*action_prefix, team),
                    metric_extractor=initial_objective_from_info,
                    shared_attribute_names=_SHARED_ENV_ATTRIBUTES,
                )
                for team in selected.candidates.teams[1:]
            ]
            best = min(alternatives, key=lambda result: result["objective"])
            requirements = completer._extract_task_requirements(obs["task"].x, selected.task_id)
            assert requirements is not None
            required_skill, demand, _duration = requirements
            legal_workers = completer._legal_worker_ids(
                obs["worker"].x,
                required_skill=required_skill,
                station_id=selected.station_id,
                worker_mask=worker_mask,
            )
            gain = float(base["objective"] - best["objective"])
            rows.append(
                {
                    "trajectory_step": trajectory_step,
                    "task_id": selected.task_id,
                    "station_id": selected.station_id,
                    "required_skill": int(required_skill),
                    "demand": int(demand),
                    "legal_worker_count": len(legal_workers),
                    "skill_slack": len(legal_workers) - int(demand),
                    "skill_bottleneck": len(legal_workers) - int(demand) <= 1,
                    "candidate_count": len(selected.candidates.teams),
                    "base_team": json.dumps(base_team),
                    "best_single_objective_gain": gain,
                    "has_single_swap_improvement": gain > float(args.improvement_epsilon),
                    "best_single_gain_makespan": float(
                        base["makespan_penalty"] - best["makespan_penalty"]
                    ),
                    "best_single_gain_balance": float(
                        base["balance_penalty"] - best["balance_penalty"]
                    ),
                    "base_objective": base["objective"],
                    "best_single_objective": best["objective"],
                }
            )

        obs, _reward, done, info = env.step(
            (selected.task_id, selected.station_id, list(base_team))
        )
        if "error" in info:
            raise RuntimeError(f"审计基准动作被环境拒绝：{info['error']}")
        trajectory_step += 1
        if done:
            break

    args.output_dir.mkdir(parents=True, exist_ok=False)
    fields = list(rows[0]) if rows else []
    with (args.output_dir / "state_opportunities.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    result = {
        "protocol": "initial_schedule_one_step_team_counterfactual_v1",
        "data_path": str(args.data_path),
        "seed": int(args.seed),
        "trajectory_steps": trajectory_step,
        "summary": summarize_initial_opportunities(rows),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/680.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-states", type=int, default=20)
    parser.add_argument("--state-stride", type=int, default=5)
    parser.add_argument("--max-trajectory-steps", type=int, default=400)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--improvement-epsilon", type=float, default=1.0e-9)
    parser.add_argument(
        "--output-dir",
        default="results/90_legacy_and_smoke/initial_team_opportunity_audit_real680_seed42",
    )
    args = parser.parse_args(argv)
    args.data_path = _workspace_path(args.data_path)
    args.output_dir = _workspace_path(args.output_dir)
    return args


if __name__ == "__main__":
    print(json.dumps(run_audit(parse_args()), ensure_ascii=False, indent=2))
