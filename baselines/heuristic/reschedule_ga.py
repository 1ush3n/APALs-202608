from __future__ import annotations

import copy
import time
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import configs, load_config_files
from environment import AirLineEnv_Graph
from utils.reschedule import (
    HARD_CONSTRAINT_KEYS,
    RescheduleScenario,
    calculate_reschedule_composite_score,
    calculate_stability_metrics,
    load_reschedule_scenarios,
)


@dataclass
class RescheduleGAResult:
    """单个固定重调度场景下的 GA 结果。"""

    scenario_id: str
    scenario_start_time: float
    delayed_task_count: int
    makespan: float
    balance_std: float
    reward: float
    fitness: float
    duration_sec: float
    assigned_tasks: list[tuple[int, int, list[int], float, float]]
    constraint_metrics: dict[str, float]
    env: AirLineEnv_Graph


def _to_numpy_bool(mask: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(mask):
        return mask.detach().cpu().numpy().astype(bool)
    return np.asarray(mask, dtype=bool)


class RescheduleGeneticAlgorithmScheduler:
    """
    APAL 预测-反应式重调度 GA 基线。

    染色体只表达“任务、站位、工人”的偏好；真正的合法性由同一个
    AirLineEnv_Graph 环境边界检查，因此与 PPO 重调度共享冻结任务、
    release time、紧前紧后、站位槽位、技能和派工人数等硬约束。
    """

    def __init__(
        self,
        *,
        data_path_or_dir: str | Path,
        scenario: RescheduleScenario,
        scenario_id: str,
        pop_size: int = 30,
        max_gen: int = 20,
        cx_pb: float = 0.8,
        mut_pb: float = 0.2,
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        self.data_path_or_dir = Path(data_path_or_dir)
        self.scenario = scenario
        self.scenario_id = scenario_id
        self.pop_size = int(pop_size)
        self.max_gen = int(max_gen)
        self.cx_pb = float(cx_pb)
        self.mut_pb = float(mut_pb)
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.rng = np.random.RandomState(self.seed)

        self.env = AirLineEnv_Graph(data_path_or_dir=str(self.data_path_or_dir), seed=self.seed)
        setattr(self.env, "_forced_reschedule_scenario", scenario)
        self.env.skip_obs_building = True
        self.env.reset(randomize_duration=False, randomize_workers=False, seed=self.seed)

        self.num_tasks = self.env.num_tasks
        self.num_stations = self.env.num_stations
        self.num_workers = self.env.num_workers

    def _create_individual(self) -> dict[str, list[Any]]:
        return {
            "seq_pref": self.rng.rand(self.num_tasks).tolist(),
            "station_pref": self.rng.rand(self.num_tasks, self.num_stations).tolist(),
            "team_pref": self.rng.rand(self.num_tasks, self.num_workers).tolist(),
        }

    def _init_population(self) -> list[dict[str, list[Any]]]:
        return [self._create_individual() for _ in range(self.pop_size)]

    def _select_action(self, individual: dict[str, list[Any]]) -> tuple[int, int, list[int]] | None:
        task_mask_raw, station_mask_raw, worker_mask_raw = self.env.get_masks()
        task_mask = _to_numpy_bool(task_mask_raw)
        station_mask = _to_numpy_bool(station_mask_raw)
        worker_mask = _to_numpy_bool(worker_mask_raw)

        if task_mask.all():
            return None

        seq_prefs = np.asarray(individual["seq_pref"], dtype=float)
        station_prefs = np.asarray(individual["station_pref"], dtype=float)
        team_prefs = np.asarray(individual["team_pref"], dtype=float)

        available_tasks = np.where(~task_mask)[0].tolist()
        available_tasks.sort(key=lambda task_id: seq_prefs[int(task_id)], reverse=True)
        worker_skill_matrix = (
            self.env.worker_skill_matrix.detach().cpu().numpy()
            if torch.is_tensor(self.env.worker_skill_matrix)
            else np.asarray(self.env.worker_skill_matrix)
        )
        worker_locks = np.asarray(self.env.worker_locks, dtype=int)

        for task_id in available_tasks:
            valid_stations = np.where(~station_mask[int(task_id)])[0].tolist()
            valid_stations.sort(key=lambda sid: station_prefs[int(task_id), int(sid)], reverse=True)
            skill_id = int(self.env.task_static_feat[int(task_id), 1].item())
            demand = max(1, int(self.env.task_static_feat[int(task_id), 2].item()))

            for station_id in valid_stations:
                candidates = [
                    int(w)
                    for w in range(self.num_workers)
                    if not bool(worker_mask[w])
                    and worker_skill_matrix[w, skill_id] > 0.5
                    and worker_locks[w] in {0, int(station_id) + 1}
                ]
                if len(candidates) < demand:
                    continue
                candidates.sort(key=lambda w: team_prefs[int(task_id), int(w)], reverse=True)
                return int(task_id), int(station_id), candidates[:demand]
        return None

    def _decode_individual(
        self,
        individual: dict[str, list[Any]],
        *,
        seed_offset: int = 0,
    ) -> tuple[float, dict[str, float], AirLineEnv_Graph]:
        self.env.skip_obs_building = True
        setattr(self.env, "_forced_reschedule_scenario", self.scenario)
        self.env.reset(randomize_duration=False, randomize_workers=False, seed=self.seed + seed_offset)

        done = False
        total_reward = 0.0
        invalid_steps = 0
        max_steps = max(1, self.num_tasks * 3)

        for _ in range(max_steps):
            if done:
                break
            action = self._select_action(individual)
            if action is None:
                if self.env.try_wait_for_resources():
                    continue
                invalid_steps += 1
                break
            _obs, reward, done, info = self.env.step(action)
            total_reward += float(reward)
            if info.get("invalid_action", False):
                invalid_steps += 1
                break

        complete = len(self.env.assigned_tasks) == self.env.num_tasks
        if complete:
            makespan = float(np.max(self.env.station_wall_clock))
            balance_std = float(np.std(self.env.station_loads))
        else:
            makespan = float(self.env.ideal_makespan * 3.0)
            balance_std = float(self.env.ideal_station_load * 3.0)

        stability = {"start_deviation_mean_h": 0.0, "station_change_rate": 0.0, "team_change_rate": 0.0}
        takt = 0.0
        takt_violation = 0.0
        if self.env.baseline_schedule is not None:
            takt = float(self.env.baseline_schedule.makespan)
            takt_violation = max(0.0, makespan - takt)
            stability = calculate_stability_metrics(
                self.env.baseline_schedule,
                self.env.assigned_tasks,
                current_time=float(self.env.reschedule_start_time),
            )

        # 把站位/团队变更率折算到节拍尺度，保证 GA 目标和重调度奖励同量纲。
        stability_scale = max(1.0, takt)
        metrics = {
            "makespan": makespan,
            "balance_std": balance_std,
            "reward": float(total_reward),
            "complete": float(complete),
            "invalid_step_count": float(invalid_steps),
            "takt_h": float(takt),
            "takt_violation_h": float(takt_violation),
            "start_deviation_mean_h": float(stability["start_deviation_mean_h"]),
            "station_change_rate": float(stability["station_change_rate"]),
            "team_change_rate": float(stability["team_change_rate"]),
        }
        for key in HARD_CONSTRAINT_KEYS:
            metrics.setdefault(key, 0.0)
        score_result = calculate_reschedule_composite_score(
            makespan=makespan,
            balance_std=balance_std,
            constraint_metrics=metrics,
            config_obj=configs,
            ideal_station_load=float(getattr(self.env, "ideal_station_load", 1.0)),
        )
        metrics["eligible"] = float(score_result.eligible)
        metrics["composite_score"] = float(score_result.score)
        metrics["selection_score"] = float(score_result.selection_score)
        metrics["fitness"] = float(score_result.selection_score)
        metrics.update(score_result.terms)
        return float(score_result.selection_score), metrics, self.env

    def _crossover(
        self,
        p1: dict[str, list[Any]],
        p2: dict[str, list[Any]],
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
        for key in ("seq_pref", "station_pref", "team_pref"):
            arr1 = np.asarray(c1[key], dtype=float)
            arr2 = np.asarray(c2[key], dtype=float)
            mask = self.rng.rand(*arr1.shape) < 0.5
            new1 = np.where(mask, arr2, arr1)
            new2 = np.where(mask, arr1, arr2)
            c1[key] = new1.tolist()
            c2[key] = new2.tolist()
        return c1, c2

    def _mutate(self, individual: dict[str, list[Any]]) -> dict[str, list[Any]]:
        for key in ("seq_pref", "station_pref", "team_pref"):
            arr = np.asarray(individual[key], dtype=float)
            mask = self.rng.rand(*arr.shape) < self.mut_pb
            arr = arr + mask * self.rng.normal(0.0, 0.2, size=arr.shape)
            individual[key] = arr.tolist()
        return individual

    def run(self) -> RescheduleGAResult:
        if self.verbose:
            print(
                f"--- 启动 APAL 重调度 GA | 场景={self.scenario_id} | "
                f"PopSize={self.pop_size}, MaxGen={self.max_gen} ---"
            )
        population = self._init_population()
        best_individual: dict[str, list[Any]] | None = None
        best_fitness = float("inf")
        best_metrics: dict[str, float] | None = None
        start_wall = time.time()

        for gen_idx in range(self.max_gen):
            evaluated: list[tuple[float, dict[str, float], dict[str, list[Any]]]] = []
            for ind_idx, individual in enumerate(population):
                fitness, metrics, _env = self._decode_individual(individual, seed_offset=gen_idx * self.pop_size + ind_idx)
                evaluated.append((fitness, metrics, individual))
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_metrics = metrics
                    best_individual = copy.deepcopy(individual)

            evaluated.sort(key=lambda item: item[0])
            if self.verbose:
                best_gen = evaluated[0][1]
                print(
                    f"[GA Gen {gen_idx + 1}/{self.max_gen}] "
                    f"Score={best_gen['composite_score']:.4f}, "
                    f"Makespan={best_gen['makespan']:.3f}, "
                    f"TaktViolation={best_gen['takt_violation_h']:.3f}, "
                    f"StartDev={best_gen['start_deviation_mean_h']:.3f}, "
                    f"StationChange={best_gen['station_change_rate']:.3f}, "
                    f"TeamChange={best_gen['team_change_rate']:.3f}, "
                    f"Eligible={int(best_gen['eligible'])}, "
                    f"Complete={int(best_gen['complete'])}"
                )

            elite_count = max(1, int(self.pop_size * 0.1))
            next_population = [copy.deepcopy(item[2]) for item in evaluated[:elite_count]]
            tournament_size = min(3, len(evaluated))
            while len(next_population) < self.pop_size:
                p1 = min(self.rng.choice(len(evaluated), size=tournament_size, replace=False), key=lambda idx: evaluated[int(idx)][0])
                p2 = min(self.rng.choice(len(evaluated), size=tournament_size, replace=False), key=lambda idx: evaluated[int(idx)][0])
                parent1 = evaluated[int(p1)][2]
                parent2 = evaluated[int(p2)][2]
                if self.rng.rand() < self.cx_pb:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
                next_population.append(self._mutate(child1))
                if len(next_population) < self.pop_size:
                    next_population.append(self._mutate(child2))
            population = next_population[: self.pop_size]

        assert best_individual is not None
        final_fitness, final_metrics, final_env = self._decode_individual(best_individual, seed_offset=0)
        duration = time.time() - start_wall
        metrics = best_metrics or final_metrics
        metrics.update(final_metrics)
        return RescheduleGAResult(
            scenario_id=self.scenario_id,
            scenario_start_time=float(self.scenario.start_time),
            delayed_task_count=len(self.scenario.task_release_times),
            makespan=float(final_metrics["makespan"]),
            balance_std=float(final_metrics["balance_std"]),
            reward=float(final_metrics["reward"]),
            fitness=float(final_fitness),
            duration_sec=float(duration),
            assigned_tasks=list(final_env.assigned_tasks),
            constraint_metrics={},
            env=final_env,
        )


def evaluate_reschedule_ga(
    *,
    data_path_or_dir: str | Path | None = None,
    pop_size: int = 30,
    max_gen: int = 20,
    num_runs: int | None = None,
    seed: int | None = None,
    output_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    使用与 PPO 重调度完全相同的 baseline、固定扰动场景和环境约束评估 GA。
    """

    from train import (
        _compute_assignment_utilization,
        _compute_reschedule_constraint_metrics,
        ensure_reschedule_baseline_available,
        ensure_reschedule_eval_scenarios_available,
        resolve_workspace_path,
    )

    baseline_path = ensure_reschedule_baseline_available(configs)
    scenario_path = ensure_reschedule_eval_scenarios_available(configs)
    if baseline_path is None or scenario_path is None:
        raise RuntimeError("GA 重调度评估需要 enable_reschedule_mode=True，并且 baseline/固定场景必须可用。")

    data_path = resolve_workspace_path(data_path_or_dir or getattr(configs, "data_file_path", "data/283.csv"))
    scenario_items = load_reschedule_scenarios(scenario_path)
    if num_runs is not None:
        scenario_items = scenario_items[: max(1, int(num_runs))]

    results: list[RescheduleGAResult] = []
    base_seed = int(seed if seed is not None else getattr(configs, "reschedule_eval_scenario_seed", 42))
    for idx, (scenario_id, scenario) in enumerate(scenario_items):
        solver = RescheduleGeneticAlgorithmScheduler(
            data_path_or_dir=data_path,
            scenario=scenario,
            scenario_id=scenario_id,
            pop_size=pop_size,
            max_gen=max_gen,
            seed=base_seed + idx,
            verbose=verbose,
        )
        result = solver.run()
        constraints = _compute_reschedule_constraint_metrics(result.env)
        constraints["scenario_index"] = float(idx)
        constraints["reschedule_start_time"] = float(scenario.start_time)
        constraints["delayed_task_count"] = float(len(scenario.task_release_times))
        constraints["complete"] = float(len(result.env.assigned_tasks) == result.env.num_tasks)
        score_result = calculate_reschedule_composite_score(
            makespan=result.makespan,
            balance_std=result.balance_std,
            constraint_metrics=constraints,
            config_obj=configs,
            ideal_station_load=float(getattr(result.env, "ideal_station_load", 1.0)),
        )
        constraints["eligible"] = float(score_result.eligible)
        constraints["composite_score"] = float(score_result.score)
        constraints["selection_score"] = float(score_result.selection_score)
        constraints.update(score_result.terms)
        result.fitness = float(score_result.selection_score)
        result.constraint_metrics = constraints
        results.append(result)

    rows: list[dict[str, float | str]] = []
    for result in results:
        worker_util, station_util = _compute_assignment_utilization(result.env, result.makespan)
        row: dict[str, float | str] = {
            "scenario_id": result.scenario_id,
            "makespan": result.makespan,
            "balance_std": result.balance_std,
            "reward": result.reward,
            "score": float(result.constraint_metrics.get("composite_score", result.fitness)),
            "selection_score": result.fitness,
            "fitness": result.fitness,
            "eligible": float(result.constraint_metrics.get("eligible", 0.0)),
            "duration_sec": result.duration_sec,
            "worker_util": worker_util,
            "station_util": station_util,
            "scenario_start_time": result.scenario_start_time,
            "delayed_task_count": float(result.delayed_task_count),
        }
        row.update(result.constraint_metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    summary = {
        "baseline_path": str(Path(baseline_path).resolve()),
        "scenario_path": str(Path(scenario_path).resolve()),
        "data_path": str(Path(data_path).resolve()),
        "scenario_count": int(len(results)),
        "avg_makespan": float(df["makespan"].mean()) if not df.empty else 0.0,
        "avg_balance_std": float(df["balance_std"].mean()) if not df.empty else 0.0,
        "avg_score": float(df["score"].mean()) if not df.empty else 0.0,
        "avg_selection_score": float(df["selection_score"].mean()) if not df.empty else 0.0,
        "avg_fitness": float(df["fitness"].mean()) if not df.empty else 0.0,
        "eligible_rate": float(df["eligible"].mean()) if not df.empty else 0.0,
        "avg_duration_sec": float(df["duration_sec"].mean()) if not df.empty else 0.0,
        "rows": rows,
    }

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "reschedule_ga_eval.csv", index=False)

    return summary


def main() -> None:
    """直接运行本文件时，按 YAML 实验配置启动 GA 重调度评估。"""

    parser = argparse.ArgumentParser(description="评估 APAL 预测-反应式重调度 GA 基线")
    parser.add_argument("--config", type=str, default="conf/experiment/reschedule_task_delay.yaml")
    parser.add_argument("--pop_size", type=int, default=30)
    parser.add_argument("--max_gen", type=int, default=20)
    parser.add_argument("--num_runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/reschedule_ga")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    from train import resolve_workspace_path

    config_path = resolve_workspace_path(args.config)
    load_config_files([str(config_path)])
    summary = evaluate_reschedule_ga(
        pop_size=args.pop_size,
        max_gen=args.max_gen,
        num_runs=args.num_runs,
        seed=args.seed,
        output_dir=resolve_workspace_path(args.output_dir),
        verbose=not args.quiet,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, ensure_ascii=False, indent=2))
    print(f"GA 重调度明细已保存到: {resolve_workspace_path(args.output_dir) / 'reschedule_ga_eval.csv'}")


if __name__ == "__main__":
    main()
