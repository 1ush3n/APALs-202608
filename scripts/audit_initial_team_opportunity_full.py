# -*- coding: utf-8 -*-
"""审计初始调度候选团队替换的完整排程反事实价值（E1 无杠杆判定 + E2 生成器瓶颈判定）。

方法：在确定性候选 0 启发式轨迹的采样状态上，
  1) 枚举全部合法单人替换团队（含生成器 top-4 之外的团队）；
  2) 一步代价差筛选，再对最有希望的子集执行完整 episode 反事实
     （强制该团队一步，之后固定工序—工位、用候选 0 团队跑完），比较最终 makespan；
  3) 判定：任何团队（全 episode 口径）是否优于候选 0？最优团队是否在生成器 top-4 内
     （生成器覆盖 / 瓶颈诊断）？
  4) 可选受限双人替换上界（双人替换是否严格优于全部单人替换）与启发式排名对齐诊断。

注意：反事实仅替换一步团队并固定工序—工位，属于局部机会证据，不等同于完整
初始调度 episode 的全局最优证明。
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import configs
from core.action_completion import EarliestFinishActionCompleter
from environment import AirLineEnv_Graph
from runtime.team_opportunity import evaluate_one_step_candidate

try:  # 以脚本方式直接运行时 scripts/ 目录在 sys.path 中
    from audit_initial_team_opportunity import (
        _configure,
        _select_pair,
        _SHARED_ENV_ATTRIBUTES,
        _workspace_path,
        initial_objective_from_info,
    )
except ImportError:  # 以 python -m 方式运行时通过包路径导入
    from scripts.audit_initial_team_opportunity import (
        _configure,
        _select_pair,
        _SHARED_ENV_ATTRIBUTES,
        _workspace_path,
        initial_objective_from_info,
    )


def _final_makespan(env: AirLineEnv_Graph) -> float:
    """从已跑完/截断的克隆环境读取最终 makespan。"""
    clock = np.asarray(getattr(env, "station_wall_clock", []), dtype=float)
    return float(np.max(clock)) if clock.size else float("nan")


@dataclass(frozen=True)
class _BasePair:
    """轻量工序—工位选择结果：仅含基准团队（延续轨迹只消费候选 0）。"""

    task_id: int
    station_id: int
    team: tuple[int, ...]


def _select_base_pair(
    env: AirLineEnv_Graph,
    obs: Any,
    completer: EarliestFinishActionCompleter,
) -> _BasePair | None:
    """与 _select_pair 严格等价的轻量版：只求第一个可行工序—工位的基准团队。

    等价性依据：_select_pair 的候选 0（candidates.teams[0]）正是 _complete_for_station
    的团队，而其可行性判定（candidates is not None）也正是 _complete_for_station
    非 None。延续轨迹只消费 teams[0]，因此可跳过替代候选枚举、排序与门控特征构造，
    把每步耗时从 ~3.7ms 降到 ~1ms（本脚本为快速验证而生，这是最大的单点加速）。
    """
    task_mask, station_mask, worker_mask = env.get_masks()
    valid_tasks = torch.nonzero(~task_mask, as_tuple=False).reshape(-1).tolist()
    task_x = obs["task"].x
    worker_x = obs["worker"].x
    station_x = obs["station"].x
    for task_id in sorted(int(value) for value in valid_tasks):
        valid_stations = torch.nonzero(
            ~station_mask[int(task_id)], as_tuple=False
        ).reshape(-1).tolist()
        for station_id in sorted(int(value) for value in valid_stations):
            result = completer._complete_for_station(
                task_x=task_x,
                worker_x=worker_x,
                station_x=station_x,
                task_id=int(task_id),
                station_id=int(station_id),
                worker_mask=worker_mask,
            )
            if result is not None:
                return _BasePair(int(task_id), int(station_id), result.team)
    return None


def _heuristic_features(
    completer: EarliestFinishActionCompleter, obs: Any
) -> dict[str, torch.Tensor]:
    """一次性提取启发式打分所需特征张量，供候选循环内复用（避免每队重复 expm1）。"""
    worker_x = obs["worker"].x
    station_x = obs["station"].x
    return {
        "worker_wait": torch.expm1(worker_x[:, completer.worker_layout.wait_idx]).clamp_min(0.0),
        "station_wait": torch.expm1(station_x[:, 4]).clamp_min(0.0),
        "worker_capacity": (
            worker_x[:, completer.worker_layout.efficiency_idx]
            * worker_x[:, completer.worker_layout.fatigue_idx]
        ).clamp_min(1.0e-6),
        "station_x": station_x,
    }


def _heuristic_finish_with(
    completer: EarliestFinishActionCompleter,
    *,
    station_id: int,
    team: tuple[int, ...],
    demand: int,
    task_duration: float,
    features: dict[str, torch.Tensor],
) -> float:
    """在预提取特征上计算启发式最早完工估计（与 _team_score 一致，越小越好）。"""
    return float(
        completer._team_score(
            team=tuple(team),
            station_id=station_id,
            task_duration=task_duration,
            demand=demand,
            worker_wait=features["worker_wait"],
            worker_capacity=features["worker_capacity"],
            station_wait=features["station_wait"],
            station_x=features["station_x"],
        )[0]
    )


def _heuristic_finish(
    completer: EarliestFinishActionCompleter,
    obs: Any,
    *,
    task_id: int,
    station_id: int,
    team: tuple[int, ...],
    demand: int,
    task_duration: float,
) -> float:
    """复刻候选生成器内部的启发式最早完工估计（与 _team_score 一致，越小越好）。"""
    del task_id  # 保留接口兼容；_team_score 不需要任务 ID
    return _heuristic_finish_with(
        completer,
        station_id=station_id,
        team=team,
        demand=demand,
        task_duration=task_duration,
        features=_heuristic_features(completer, obs),
    )


def _rollout_remaining(
    env: AirLineEnv_Graph,
    obs: Any,
    completer: EarliestFinishActionCompleter,
    *,
    max_candidates: int,
    max_episode_steps: int,
) -> dict[str, Any]:
    """从当前状态用候选 0 策略（确定性排序选工序—工位 + 补全器基准团队）跑完 episode。"""
    del max_candidates  # 延续只消费基准团队，候选枚举已由 _select_base_pair 跳过
    steps = 0
    done = False
    while steps < max_episode_steps:
        selected = _select_base_pair(env, obs, completer)
        if selected is None:
            if not env.try_wait_for_resources():
                break
            obs = env._get_observation()
            continue
        obs, _reward, done, info = env.step(
            (selected.task_id, selected.station_id, list(selected.team))
        )
        if "error" in info:
            raise RuntimeError(f"延续轨迹动作被环境拒绝：{info['error']}")
        steps += 1
        if done:
            break
    return {"makespan": _final_makespan(env), "steps": steps, "done": bool(done)}


def _forced_team_full_episode(
    env: AirLineEnv_Graph,
    *,
    task_id: int,
    station_id: int,
    team: tuple[int, ...] | list[int],
    completer: EarliestFinishActionCompleter,
    max_candidates: int,
    max_episode_steps: int,
) -> dict[str, Any]:
    """克隆当前状态，强制该团队一步，之后候选 0 策略跑完，返回完整排程结果。"""
    del max_candidates  # 延续只消费基准团队，候选枚举已由 _select_base_pair 跳过
    memo = {
        id(getattr(env, name)): getattr(env, name)
        for name in _SHARED_ENV_ATTRIBUTES
        if hasattr(env, name)
    }
    clone = copy.deepcopy(env, memo=memo)
    try:
        clone.skip_obs_building = False
        obs, _reward, done, info = clone.step(
            (int(task_id), int(station_id), [int(w) for w in team])
        )
        if "error" in info:
            raise RuntimeError(f"反事实强制团队动作被环境拒绝：{info['error']}")
        steps = 1
        while not done and steps < max_episode_steps:
            selected = _select_base_pair(clone, obs, completer)
            if selected is None:
                if not clone.try_wait_for_resources():
                    break
                obs = clone._get_observation()
                continue
            obs, _reward, done, info = clone.step(
                (selected.task_id, selected.station_id, list(selected.team))
            )
            if "error" in info:
                raise RuntimeError(f"延续轨迹动作被环境拒绝：{info['error']}")
            steps += 1
            if done:
                break
        return {
            "makespan": _final_makespan(clone),
            "steps": steps,
            "done": bool(done),
            "forced_team": json.dumps(list(team)),
        }
    finally:
        del clone
        gc.collect()


def _enumerate_one_swaps(
    base_team: tuple[int, ...] | list[int], legal_workers: list[int]
) -> list[tuple[int, ...]]:
    """全部合法单人替换：仅替换基准团队的一个位置，去重并排序。"""
    teams: set[tuple[int, ...]] = set()
    for pos in range(len(base_team)):
        for worker in legal_workers:
            candidate = list(base_team)
            candidate[pos] = worker
            teams.add(tuple(sorted(candidate)))
    return sorted(teams)


def _bounded_two_swaps(
    base_team: tuple[int, ...] | list[int],
    legal_workers: list[int],
    *,
    completer: EarliestFinishActionCompleter,
    obs: Any,
    task_id: int,
    station_id: int,
    demand: int,
    task_duration: float,
    pool_size: int,
    features: dict[str, torch.Tensor] | None = None,
) -> list[tuple[int, ...]]:
    """受限双人替换：按启发式分数取前 pool_size 个（替换两个不同位置、两名新工人）。"""
    del task_id  # 保留接口兼容；_team_score 不需要任务 ID
    if len(base_team) < 2 or len(legal_workers) < 2:
        return []
    if features is None:
        features = _heuristic_features(completer, obs)
    candidates: set[tuple[int, ...]] = set()
    positions = list(range(len(base_team)))
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            for a in legal_workers:
                for b in legal_workers:
                    if a == b:
                        continue
                    candidate = list(base_team)
                    candidate[positions[i]] = a
                    candidate[positions[j]] = b
                    candidates.add(tuple(sorted(candidate)))
    ranked = sorted(
        candidates,
        key=lambda team: _heuristic_finish_with(
            completer,
            station_id=station_id,
            team=team,
            demand=demand,
            task_duration=task_duration,
            features=features,
        ),
    )
    return ranked[: max(1, int(pool_size))]


def _spearman_rho(xs: list[float], ys: list[float]) -> float | None:
    """纯 Python Spearman 秩相关（处理并列），样本不足返回 None。"""
    n = len(xs)
    if n < 3:
        return None

    def _ranks(values: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = average
            i = j + 1
        return ranks

    rx, ry = _ranks(xs), _ranks(ys)
    denominator = n**3 - n
    if denominator == 0.0:
        return None
    rho = 1.0 - 6.0 * sum((rx[i] - ry[i]) ** 2 for i in range(n)) / denominator
    return float(rho)


def _audit_state(
    env: AirLineEnv_Graph,
    obs: Any,
    completer: EarliestFinishActionCompleter,
    selected: Any,
    *,
    trajectory_step: int,
    max_candidates: int,
    max_episode_steps: int,
    top_k: int,
    enable_two_swap: bool,
    two_swap_pool: int,
    max_two_swaps: int,
    improvement_epsilon: float,
    one_step_filter: bool,
    max_legal_one_swaps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """对一个采样状态执行 E1+E2 审计，返回（状态汇总行，逐团队明细行）。"""
    task_id, station_id = int(selected.task_id), int(selected.station_id)
    base_team = selected.candidates.teams[0]
    task_mask, _station_mask, worker_mask = env.get_masks()
    del task_mask
    requirements = completer._extract_task_requirements(obs["task"].x, task_id)
    if requirements is None:
        return None, []  # 虚拟/零工时任务无团队替换空间，跳过
    required_skill, demand, task_duration = requirements
    legal_workers = [
        int(w)
        for w in completer._legal_worker_ids(
            obs["worker"].x,
            required_skill=required_skill,
            station_id=station_id,
            worker_mask=worker_mask,
        )
        if int(w) not in set(base_team)
    ]
    generated_teams = [tuple(sorted(t)) for t in selected.candidates.teams[1:]]
    full_pool = _enumerate_one_swaps(base_team, legal_workers)
    pool = full_pool
    cap = int(max_legal_one_swaps)
    if cap > 0 and len(full_pool) > cap:
        stride = max(1, math.ceil(len(full_pool) / cap))
        pool = full_pool[::stride][:cap]

    # 启发式分数（纯张量运算、廉价，作为全池排名依据；特征张量只提取一次供循环复用）
    heuristic_features = _heuristic_features(completer, obs)
    heuristic_by_team = {
        team: _heuristic_finish_with(
            completer, station_id=station_id, team=team,
            demand=demand, task_duration=task_duration, features=heuristic_features,
        )
        for team in pool
    }

    # ---- 可选的一步筛选（默认关闭；开启时为全池做真实一步评估以对照既有审计口径）----
    def _one_step(team: tuple[int, ...]) -> dict[str, float]:
        result = evaluate_one_step_candidate(
            env,
            action=(task_id, station_id, list(team)),
            metric_extractor=initial_objective_from_info,
            shared_attribute_names=_SHARED_ENV_ATTRIBUTES,
        )
        return {
            "objective": float(result["objective"]),
            "makespan_penalty": float(result["makespan_penalty"]),
            "balance_penalty": float(result["balance_penalty"]),
        }

    pool_one_step: dict[tuple[int, ...], dict[str, float]] = {}
    if one_step_filter:
        for team in pool:
            pool_one_step[team] = _one_step(team)

    # ---- 完整 episode 反事实候选集 ----
    full_episode_teams: list[tuple[tuple[int, ...], str]] = [(tuple(base_team), "base")]
    for team in generated_teams:
        full_episode_teams.append((team, "generated"))
    if one_step_filter:
        ranked_pool = sorted(
            pool,
            key=lambda team: (
                pool_one_step[team]["makespan_penalty"],
                heuristic_by_team[team],
            ),
        )
    else:
        ranked_pool = sorted(pool, key=lambda team: heuristic_by_team[team])
    for team in ranked_pool[: max(1, int(top_k))]:
        full_episode_teams.append((team, "legal_pool_topk"))
    if enable_two_swap:
        two_swaps = _bounded_two_swaps(
            base_team, legal_workers, completer=completer, obs=obs, task_id=task_id,
            station_id=station_id, demand=demand, task_duration=task_duration,
            pool_size=two_swap_pool, features=heuristic_features,
        )
        for team in two_swaps[: max(1, int(max_two_swaps))]:
            full_episode_teams.append((team, "two_swap"))

    # ---- 执行完整 episode 反事实 ----
    seen: set[tuple[int, ...]] = set()
    results: list[tuple[tuple[int, ...], str, dict[str, Any]]] = []
    for team, source in full_episode_teams:
        if team in seen:
            continue
        seen.add(team)
        outcome = _forced_team_full_episode(
            env, task_id=task_id, station_id=station_id, team=team,
            completer=completer, max_candidates=max_candidates,
            max_episode_steps=max_episode_steps,
        )
        results.append((team, source, outcome))

    # ---- 汇总 ----
    baseline_row = next(r for r in results if r[1] == "base")
    baseline_makespan = float(baseline_row[2]["makespan"])
    baseline_done = bool(baseline_row[2]["done"])
    valid = [r for r in results if r[2]["done"] and r[1] != "base"]
    capped_count = sum(1 for r in results if not r[2]["done"])
    best: tuple[tuple[int, ...], str, dict[str, Any]] | None = min(
        valid, key=lambda r: r[2]["makespan"]
    ) if valid else None
    best_team, best_source, best_outcome = (
        best if best is not None else ((), "base", baseline_row[2])
    )
    gain_pct = (
        (baseline_makespan - float(best_outcome["makespan"])) / baseline_makespan * 100.0
        if baseline_makespan > 0.0 and best is not None
        else 0.0
    )
    any_better = bool(
        baseline_done
        and best is not None
        and float(best_outcome["makespan"]) < baseline_makespan - improvement_epsilon
    )
    best_in_generator = any_better and tuple(sorted(best_team)) in {
        tuple(sorted(t)) for t in generated_teams
    }
    two_swap_beat_single: bool | None = None
    if enable_two_swap:
        single_best = (
            min(
                (r for r in results if r[1] in ("generated", "legal_pool_topk") and r[2]["done"]),
                key=lambda r: r[2]["makespan"],
            )[2]["makespan"]
            if any(r[1] in ("generated", "legal_pool_topk") and r[2]["done"] for r in results)
            else float("inf")
        )
        two_best = (
            min(
                (r for r in results if r[1] == "two_swap" and r[2]["done"]),
                key=lambda r: r[2]["makespan"],
            )[2]["makespan"]
            if any(r[1] == "two_swap" and r[2]["done"] for r in results)
            else float("inf")
        )
        two_swap_beat_single = (
            two_best < single_best - improvement_epsilon
            if (two_best != float("inf") and single_best != float("inf"))
            else None
        )

    # 评估子集的启发式与一步口径（含不在池内的基准/双替换团队）
    for team in seen:
        if team not in heuristic_by_team:
            heuristic_by_team[team] = _heuristic_finish_with(
                completer, station_id=station_id, team=team,
                demand=demand, task_duration=task_duration, features=heuristic_features,
            )
    evaluated_one_step: dict[tuple[int, ...], dict[str, float]] = {}
    for team in seen:
        evaluated_one_step[team] = _one_step(team)
    base_one_step = evaluated_one_step[tuple(base_team)]

    aligned = [r for r in results if r[1] != "base" and r[2]["done"]]
    heuristic_alignment = (
        _spearman_rho(
            [heuristic_by_team[r[0]] for r in aligned],
            [r[2]["makespan"] for r in aligned],
        )
        if len(aligned) >= 3
        else None
    )

    state_row = {
        "trajectory_step": trajectory_step,
        "task_id": task_id,
        "station_id": station_id,
        "required_skill": int(required_skill),
        "demand": int(demand),
        "legal_worker_count": len(legal_workers),
        "skill_slack": len(legal_workers) - int(demand),
        "candidate_count": len(selected.candidates.teams),
        "pool_size_one_swap_full": len(full_pool),
        "pool_size_one_swap_capped": len(pool),
        "one_swap_cap_applied": cap > 0 and len(full_pool) > cap,
        "base_team": json.dumps(list(base_team)),
        "baseline_full_makespan": baseline_makespan,
        "baseline_done": baseline_done,
        "best_full_makespan": float(best_outcome["makespan"]) if best is not None else None,
        "best_full_gain_pct": gain_pct,
        "best_team": json.dumps(list(best_team)),
        "best_team_source": best_source,
        "best_team_in_generator_top4": best_in_generator,
        "any_team_better_full_episode": any_better,
        "two_swap_beats_best_single": two_swap_beat_single,
        "heuristic_alignment_spearman": heuristic_alignment,
        "evaluated_teams": len(results),
        "capped_episodes": capped_count,
        "base_one_step_makespan": base_one_step["makespan_penalty"],
        "base_one_step_balance": base_one_step["balance_penalty"],
    }

    team_rows = []
    for team, source, outcome in results:
        team_rows.append(
            {
                "trajectory_step": trajectory_step,
                "team": json.dumps(list(team)),
                "source": source,
                "in_generator_top4": tuple(sorted(team)) in {
                    tuple(sorted(t)) for t in generated_teams
                },
                "heuristic_finish": heuristic_by_team[team],
                "one_step_makespan": evaluated_one_step[team]["makespan_penalty"],
                "one_step_balance": evaluated_one_step[team]["balance_penalty"],
                "full_makespan": outcome["makespan"],
                "full_delta_vs_baseline": outcome["makespan"] - baseline_makespan,
                "full_improved": outcome["makespan"] < baseline_makespan - improvement_epsilon,
                "steps": outcome["steps"],
                "done": outcome["done"],
            }
        )
    return state_row, team_rows


def summarize_full_episode_audit(state_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总 E1+E2 审计结果（决策口径：完整排程 makespan 反事实）。"""
    if not state_rows:
        return {"audited_states": 0}
    valid_states = [row for row in state_rows if bool(row["baseline_done"])]
    improved = [row for row in valid_states if bool(row["any_team_better_full_episode"])]
    baseline_makespans = [float(row["baseline_full_makespan"]) for row in valid_states]

    def _mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "audited_states": len(state_rows),
        "valid_states": len(valid_states),
        "states_with_better_team_full_episode": len(improved),
        "better_team_rate": float(len(improved) / len(valid_states)) if valid_states else 0.0,
        "best_gain_pct_mean_when_improved": _mean(
            [float(row["best_full_gain_pct"]) for row in improved]
        ),
        "best_gain_pct_max_when_improved": max(
            (float(row["best_full_gain_pct"]) for row in improved), default=0.0
        ),
        "generator_coverage_of_better_teams": (
            _mean(
                [
                    float(bool(row["best_team_in_generator_top4"]))
                    for row in improved
                ]
            )
            if improved
            else None
        ),
        "states_two_swap_beats_best_single": sum(
            1 for row in valid_states if row["two_swap_beats_best_single"] is True
        ),
        "states_two_swap_evaluated": sum(
            1 for row in valid_states if row["two_swap_beats_best_single"] is not None
        ),
        "heuristic_alignment_spearman_mean": _mean(
            [
                float(row["heuristic_alignment_spearman"])
                for row in valid_states
                if row["heuristic_alignment_spearman"] is not None
            ]
        ),
        "heuristic_alignment_states": sum(
            1 for row in valid_states if row["heuristic_alignment_spearman"] is not None
        ),
        "baseline_makespan_mean": _mean(baseline_makespans),
        "capped_episode_states": sum(1 for row in state_rows if row["capped_episodes"] > 0),
        "metric_note": (
            "完整排程反事实：在采样状态强制该团队一步、固定工序—工位、之后候选 0 策略跑完；"
            "比较最终 makespan。结果属局部机会证据，不等同于完整初始调度 episode 的全局最优证明。"
        ),
    }


def run_full_audit(args: argparse.Namespace) -> dict[str, Any]:
    _configure(args.data_path)
    env = AirLineEnv_Graph(data_path_or_dir=str(args.data_path), seed=int(args.seed))
    obs = env.reset(randomize_duration=False, randomize_workers=False, seed=int(args.seed))
    completer = EarliestFinishActionCompleter(configs)
    state_rows: list[dict[str, Any]] = []
    team_rows: list[dict[str, Any]] = []
    trajectory_step = 0

    while (
        trajectory_step < int(args.max_trajectory_steps)
        and len(state_rows) < int(args.max_states)
    ):
        selected = _select_pair(env, obs, completer, max_candidates=int(args.max_candidates))
        if selected is None:
            if not env.try_wait_for_resources():
                break
            obs = env._get_observation()
            continue
        if (
            trajectory_step % max(1, int(args.state_stride)) == 0
            and len(selected.candidates.teams) > 1
        ):
            state_row, team_detail = _audit_state(
                env, obs, completer, selected,
                trajectory_step=trajectory_step,
                max_candidates=int(args.max_candidates),
                max_episode_steps=int(args.max_episode_steps),
                top_k=int(args.top_k),
                enable_two_swap=bool(args.enable_two_swap),
                two_swap_pool=int(args.two_swap_pool),
                max_two_swaps=int(args.max_two_swaps),
                improvement_epsilon=float(args.improvement_epsilon),
                one_step_filter=bool(args.one_step_filter),
                max_legal_one_swaps=int(args.max_legal_one_swaps),
            )
            if state_row is not None:
                state_rows.append(state_row)
                team_rows.extend(team_detail)
                print(
                    f"[audit] step={trajectory_step} 状态 #{len(state_rows)}/"
                    f"{args.max_states} 完成，"
                    f"baseline_makespan={state_row['baseline_full_makespan']:.4f}",
                    file=sys.stderr,
                    flush=True,
                )

        obs, _reward, done, info = env.step(
            (selected.task_id, selected.station_id, list(selected.candidates.teams[0]))
        )
        if "error" in info:
            raise RuntimeError(f"审计基准动作被环境拒绝：{info['error']}")
        trajectory_step += 1
        if done:
            break

    args.output_dir.mkdir(parents=True, exist_ok=False)
    state_fields = list(state_rows[0]) if state_rows else []
    with (args.output_dir / "state_full_episode.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        if state_fields:
            writer = csv.DictWriter(handle, fieldnames=state_fields)
            writer.writeheader()
            writer.writerows(state_rows)
    team_fields = list(team_rows[0]) if team_rows else []
    with (args.output_dir / "team_full_episode.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        if team_fields:
            writer = csv.DictWriter(handle, fieldnames=team_fields)
            writer.writeheader()
            writer.writerows(team_rows)
    result = {
        "protocol": "initial_schedule_full_episode_team_counterfactual_v1",
        "data_path": str(args.data_path),
        "seed": int(args.seed),
        "trajectory_steps": trajectory_step,
        "summary": summarize_full_episode_audit(state_rows),
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
    parser.add_argument("--max-trajectory-steps", type=int, default=1200,
                        help="主轨迹最长步数（默认超过单数据集完整 episode 长度，保证采样覆盖全程）")
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=2000)
    parser.add_argument("--max-legal-one-swaps", type=int, default=0,
                        help="全合法单人替换池上限（0=不限，等距抽样）")
    parser.add_argument("--one-step-filter", action=argparse.BooleanOptionalAction,
                        default=False, help="为全池做真实一步评估（较慢，默认关闭）")
    parser.add_argument("--enable-two-swap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--two-swap-pool", type=int, default=24)
    parser.add_argument("--max-two-swaps", type=int, default=4)
    parser.add_argument("--improvement-epsilon", type=float, default=1.0e-6)
    parser.add_argument(
        "--output-dir",
        default="results/90_legacy_and_smoke/initial_team_opportunity_full_audit_real680_seed42",
    )
    args = parser.parse_args(argv)
    args.data_path = _workspace_path(args.data_path)
    args.output_dir = _workspace_path(args.output_dir)
    return args


if __name__ == "__main__":
    print(json.dumps(run_full_audit(parse_args()), ensure_ascii=False, indent=2))
