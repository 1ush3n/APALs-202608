"""审计 APAL 重调度中团队候选的即时反事实改进空间。

该脚本不训练、不修改模型，也不覆盖任何实验结果。它从同一重调度状态复制环境，
固定工序和工位，仅替换团队成员，比较统一重调度目标在执行一步后的真实变化。
输出必须被理解为“局部机会审计”，不能替代完整 240 场景正式验证。
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import configs, load_config_files
from core.action_completion import EarliestFinishActionCompleter, TeamCandidates
from environment import AirLineEnv_Graph
from runtime.initial_worker_mapping import apply_initial_worker_mapping
from utils.reschedule import load_reschedule_scenarios


OBJECTIVE_KEYS = (
    "score_makespan",
    "score_balance",
    "score_takt_violation",
    "score_start_stability",
    "score_station_change",
    "score_team_change",
)


@dataclass(frozen=True)
class SelectedPair:
    """诊断轨迹中一个确定性的工序—工位动作。"""

    task_id: int
    station_id: int
    candidates: TeamCandidates


def _workspace_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _configure(args: argparse.Namespace) -> None:
    load_config_files(
        [str(PROJECT_ROOT / "conf" / "experiment" / "reschedule_task_delay.yaml")],
        target=configs,
    )
    overrides: dict[str, Any] = {
        "enable_reschedule_mode": True,
        "reschedule_manifest_path": str(args.manifest),
        "reschedule_eval_instance_id": str(args.instance_id),
        "data_file_path": str(args.data_path),
        "train_data_path_or_dir": str(args.data_path.parent),
        "randomize_durations": False,
        "enable_dynamic_events": False,
        "enable_station_breakdown": False,
        "enable_material_delay": False,
        "enable_online_duration_perturb": False,
        "enable_worker_fatigue": False,
        "conditional_team_max_candidates": int(args.max_candidates),
    }
    for key, value in overrides.items():
        setattr(configs, key, value)
    # 与正式重调度评估一致：真实实例的基准排程使用其固定工人规模。
    apply_initial_worker_mapping(configs, args.data_path, explicit_fields=set())


def _load_forced_scenario(path: Path, scenario_id: str | None) -> Any | None:
    if scenario_id is None:
        return None
    scenarios = load_reschedule_scenarios(path)
    for item_id, scenario in scenarios:
        if str(item_id) == scenario_id:
            return scenario
    available = ", ".join(str(item_id) for item_id, _scenario in scenarios[:10])
    raise ValueError(f"场景 {scenario_id!r} 不存在；前十个场景为：{available}")


def _select_pair(
    env: AirLineEnv_Graph,
    obs: Any,
    completer: EarliestFinishActionCompleter,
) -> SelectedPair | None:
    """用确定性基准顺序生成审计轨迹，避免将未训练策略混入机会统计。"""

    task_mask, station_mask, worker_mask = env.get_masks()
    valid_tasks = torch.nonzero(~task_mask, as_tuple=False).reshape(-1).tolist()
    if not valid_tasks:
        return None

    def task_key(task_id: int) -> tuple[float, int]:
        baseline = getattr(env, "baseline_schedule", None)
        baseline_task = None if baseline is None else baseline.tasks.get(int(task_id))
        baseline_start = float("inf") if baseline_task is None else float(baseline_task.start)
        return baseline_start, int(task_id)

    for task_id in sorted((int(value) for value in valid_tasks), key=task_key):
        valid_stations = torch.nonzero(~station_mask[task_id], as_tuple=False).reshape(-1).tolist()
        for station_id in sorted(int(value) for value in valid_stations):
            candidates = completer.enumerate_team_candidates(
                obs,
                task_id=task_id,
                station_id=station_id,
                worker_mask=worker_mask,
                max_candidates=int(getattr(configs, "conditional_team_max_candidates", 4)),
            )
            if candidates is not None:
                return SelectedPair(task_id, station_id, candidates)
    return None


def _objective_delta(info: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(info.get(f"reschedule_delta_{key}", 0.0))
        for key in OBJECTIVE_KEYS
    }


def _evaluate_team(
    env: AirLineEnv_Graph,
    *,
    task_id: int,
    station_id: int,
    team: tuple[int, ...],
) -> dict[str, float]:
    """从完全相同的 APAL 状态执行一个候选团队，返回真实一步目标增量。"""

    # 环境的数据集上下文和图骨架在一个 episode 内只读；对它们深拷贝会使每个
    # 反事实候选复制整张 APAL 图，既浪费内存也会遮蔽诊断结果。动态排程状态仍由
    # deepcopy 复制，保证候选之间相互隔离。
    static_names = (
        "dataset_pool", "raw_data", "base_data", "base_task_x", "base_worker_x",
        "base_station_x", "task_static_feat", "worker_skill_matrix", "predecessors",
        "successors", "num_preds", "fixed_stations", "constraint_engine", "mean_task_time",
        "ideal_station_load", "ideal_makespan", "total_base_workload", "base_durations",
        "max_allowed_stations", "is_critical", "full_worker_efficiency",
        "full_worker_skill_matrix", "worker_efficiency", "worker_static_feat",
        "worker_feature_layout", "baseline_schedule", "reschedule_scenario",
    )
    memo = {
        id(getattr(env, name)): getattr(env, name)
        for name in static_names
        if hasattr(env, name)
    }
    clone = copy.deepcopy(env, memo=memo)
    try:
        clone.skip_obs_building = True
        _obs, reward, done, info = clone.step((int(task_id), int(station_id), list(team)))
        if "error" in info:
            raise RuntimeError(f"候选团队被环境拒绝：{info['error']}")
        terms = _objective_delta(info)
        return {
            "objective_delta": float(sum(terms.values())),
            "reward": float(reward),
            "done": float(bool(done)),
            **terms,
        }
    finally:
        del clone
        gc.collect()


def _bounded_two_swap_teams_for_task(
    *,
    task_id: int,
    candidates: TeamCandidates,
    obs: Any,
    completer: EarliestFinishActionCompleter,
    worker_mask: torch.Tensor,
    max_teams: int,
) -> list[tuple[int, ...]]:
    """为指定工序生成受限双人替换候选，且不改变原候选集合。"""

    from itertools import combinations, permutations

    base = candidates.teams[0]
    if len(base) < 2 or max_teams <= 0:
        return []
    task_x = obs["task"].x
    worker_x = obs["worker"].x
    station_x = obs["station"].x
    requirements = completer._extract_task_requirements(task_x, int(task_id))
    if requirements is None:
        return []
    required_skill, demand, task_duration = requirements
    legal_workers = completer._legal_worker_ids(
        worker_x,
        required_skill=required_skill,
        station_id=candidates.station_id,
        worker_mask=worker_mask,
    )
    replacement_pool = [worker_id for worker_id in legal_workers if worker_id not in set(base)]
    if len(replacement_pool) < 2:
        return []
    worker_wait = torch.expm1(worker_x[:, completer.worker_layout.wait_idx]).clamp_min(0.0)
    station_wait = torch.expm1(station_x[:, 4]).clamp_min(0.0)
    worker_capacity = (
        worker_x[:, completer.worker_layout.efficiency_idx]
        * worker_x[:, completer.worker_layout.fatigue_idx]
    ).clamp_min(1.0e-6)
    scored: list[tuple[tuple[float, float, int, tuple[int, ...]], tuple[int, ...]]] = []
    seen: set[tuple[int, ...]] = set(candidates.teams)
    for replace_positions in combinations(range(len(base)), 2):
        for replacements in permutations(replacement_pool, 2):
            team = list(base)
            for position, worker_id in zip(replace_positions, replacements, strict=True):
                team[position] = int(worker_id)
            candidate = tuple(team)
            if candidate in seen:
                continue
            seen.add(candidate)
            scored.append(
                (
                    completer._team_score(
                        team=candidate,
                        station_id=candidates.station_id,
                        task_duration=task_duration,
                        demand=demand,
                        worker_wait=worker_wait,
                        worker_capacity=worker_capacity,
                        station_wait=station_wait,
                        station_x=station_x,
                    ),
                    candidate,
                )
            )
    scored.sort(key=lambda item: item[0])
    return [team for _score, team in scored[:max_teams]]


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"audited_states": 0}
    frame = pd.DataFrame(rows)
    improved = frame[frame["has_single_swap_improvement"]]
    bottleneck = frame[frame["skill_bottleneck"]]
    non_bottleneck = frame[~frame["skill_bottleneck"]]
    components = {
        key: float(improved[f"best_single_gain_{key}"].mean()) if len(improved) else 0.0
        for key in OBJECTIVE_KEYS
    }
    return {
        "audited_states": int(len(frame)),
        "states_with_at_least_two_current_candidates": int((frame["candidate_count"] >= 2).sum()),
        "states_with_better_single_swap": int(len(improved)),
        "better_single_swap_rate": float(len(improved) / len(frame)),
        "best_single_objective_gain_mean": float(improved["best_single_objective_gain"].mean()) if len(improved) else 0.0,
        "best_single_objective_gain_max": float(improved["best_single_objective_gain"].max()) if len(improved) else 0.0,
        "improvement_rate_skill_bottleneck": float(bottleneck["has_single_swap_improvement"].mean()) if len(bottleneck) else None,
        "improvement_rate_non_bottleneck": float(non_bottleneck["has_single_swap_improvement"].mean()) if len(non_bottleneck) else None,
        "mean_component_gain_when_improved": components,
        "states_where_bounded_two_swap_beats_all_single_swaps": int(frame["two_swap_beats_single"].sum()),
        "bounded_two_swap_note": "仅搜索每状态估计最早完工的前若干双替换团队，不等同于全组合最优。",
        "metric_note": "所有收益是同一状态、同一工序—工位、执行一步后的统一重调度目标变化；不是完整 episode 的最终收益。",
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    _configure(args)
    env = AirLineEnv_Graph(data_path_or_dir=str(args.data_path), seed=int(args.seed))
    forced_scenario = _load_forced_scenario(args.scenario_path, args.scenario_id)
    if forced_scenario is not None:
        env._forced_reschedule_scenario = forced_scenario
    obs = env.reset(randomize_duration=False, randomize_workers=False, seed=int(args.seed))
    completer = EarliestFinishActionCompleter(configs)
    rows: list[dict[str, Any]] = []
    trajectory_step = 0
    while trajectory_step < int(args.max_trajectory_steps) and len(rows) < int(args.max_states):
        selected = _select_pair(env, obs, completer)
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
            base_eval = _evaluate_team(
                env,
                task_id=selected.task_id,
                station_id=selected.station_id,
                team=base_team,
            )
            alternative_rows: list[tuple[str, tuple[int, ...], dict[str, float]]] = []
            for index, team in enumerate(selected.candidates.teams[1:], start=1):
                alternative_rows.append(
                    (
                        f"single_{index}",
                        team,
                        _evaluate_team(
                            env,
                            task_id=selected.task_id,
                            station_id=selected.station_id,
                            team=team,
                        ),
                    )
                )
            two_swap_teams = _bounded_two_swap_teams_for_task(
                task_id=selected.task_id,
                candidates=selected.candidates,
                obs=obs,
                completer=completer,
                worker_mask=worker_mask,
                max_teams=int(args.max_two_swap_candidates),
            )
            two_swap_evals = [
                _evaluate_team(
                    env,
                    task_id=selected.task_id,
                    station_id=selected.station_id,
                    team=team,
                )
                for team in two_swap_teams
            ]
            best_single = min((item[2] for item in alternative_rows), key=lambda item: item["objective_delta"])
            best_two = min(two_swap_evals, key=lambda item: item["objective_delta"], default=None)
            requirements = completer._extract_task_requirements(obs["task"].x, selected.task_id)
            assert requirements is not None
            required_skill, demand, _duration = requirements
            legal_worker_count = len(
                completer._legal_worker_ids(
                    obs["worker"].x,
                    required_skill=required_skill,
                    station_id=selected.station_id,
                    worker_mask=worker_mask,
                )
            )
            gain = float(base_eval["objective_delta"] - best_single["objective_delta"])
            row: dict[str, Any] = {
                "trajectory_step": int(trajectory_step),
                "task_id": int(selected.task_id),
                "station_id": int(selected.station_id),
                "required_skill": int(required_skill),
                "demand": int(demand),
                "legal_worker_count": int(legal_worker_count),
                "skill_slack": int(legal_worker_count - demand),
                "skill_bottleneck": bool(legal_worker_count - demand <= 1),
                "candidate_count": int(len(selected.candidates.teams)),
                "base_team": json.dumps(base_team),
                "best_single_objective_gain": gain,
                "has_single_swap_improvement": bool(gain > float(args.improvement_epsilon)),
                "two_swap_candidate_count": int(len(two_swap_evals)),
                "two_swap_beats_single": bool(
                    best_two is not None
                    and best_two["objective_delta"] < best_single["objective_delta"] - float(args.improvement_epsilon)
                ),
            }
            for key in OBJECTIVE_KEYS:
                row[f"base_{key}"] = base_eval[key]
                row[f"best_single_{key}"] = best_single[key]
                row[f"best_single_gain_{key}"] = float(base_eval[key] - best_single[key])
                if best_two is not None:
                    row[f"best_two_{key}"] = best_two[key]
            rows.append(row)

        obs, _reward, done, info = env.step(
            (selected.task_id, selected.station_id, list(base_team))
        )
        if "error" in info:
            raise RuntimeError(f"诊断轨迹基准动作被环境拒绝：{info['error']}")
        trajectory_step += 1
        if done:
            break

    result = {
        "protocol": "local_one_step_counterfactual",
        "data_path": str(args.data_path),
        "manifest_path": str(args.manifest),
        "instance_id": str(args.instance_id),
        "scenario_id": args.scenario_id,
        "seed": int(args.seed),
        "trajectory_steps": int(trajectory_step),
        "summary": _summary(rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "state_opportunities.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/680.csv")
    parser.add_argument("--manifest", default="data/r3/m.json")
    parser.add_argument("--instance-id", default="real_680")
    parser.add_argument("--scenario-path", default="data/r3/s/real_680_load_grid_seed20260701.csv")
    parser.add_argument("--scenario-id", default="medium_000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-states", type=int, default=20)
    parser.add_argument("--state-stride", type=int, default=5)
    parser.add_argument("--max-trajectory-steps", type=int, default=400)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--max-two-swap-candidates", type=int, default=8)
    parser.add_argument("--improvement-epsilon", type=float, default=1.0e-9)
    parser.add_argument(
        "--output-dir",
        default="results/90_legacy_and_smoke/r3_team_opportunity_audit_real680_medium000",
    )
    parsed = parser.parse_args(argv)
    parsed.data_path = _workspace_path(parsed.data_path)
    parsed.manifest = _workspace_path(parsed.manifest)
    parsed.scenario_path = _workspace_path(parsed.scenario_path)
    parsed.output_dir = _workspace_path(parsed.output_dir)
    return parsed


if __name__ == "__main__":
    outcome = run_audit(parse_args())
    print(json.dumps(outcome, ensure_ascii=False, indent=2))
