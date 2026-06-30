from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Iterable, Literal

import numpy as np

from configs import configs


SeedRule = Literal["SPT", "LPT", "CPM", "MSL", "Random"]


@dataclass
class PrioritySolution:
    """APAL 调度候选解：用优先级编码工序、工位和工人选择偏好。"""

    task_priority: np.ndarray
    station_priority: np.ndarray
    worker_priority: np.ndarray

    def clone(self) -> "PrioritySolution":
        return PrioritySolution(
            task_priority=self.task_priority.copy(),
            station_priority=self.station_priority.copy(),
            worker_priority=self.worker_priority.copy(),
        )


@dataclass
class DecodeResult:
    fitness: float
    makespan: float
    balance_std: float
    assigned_tasks: list[tuple[int, int, list[int], float, float]]
    complete: bool
    deadlock_count: int


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


def _compute_cpm_times(env: Any) -> tuple[np.ndarray, np.ndarray]:
    durations = _to_numpy(env.task_static_feat[:, 0]).astype(float)
    topo_order = env._topological_sort()
    es = np.zeros(env.num_tasks, dtype=float)
    for task_id in topo_order:
        es[task_id] = max((es[p] + durations[p] for p in env.predecessors[task_id]), default=0.0)

    horizon = float(np.max(es + durations)) if env.num_tasks > 0 else 0.0
    ls = np.full(env.num_tasks, horizon, dtype=float)
    for task_id in reversed(topo_order):
        latest_finish = min((ls[s] for s in env.successors[task_id]), default=horizon)
        ls[task_id] = latest_finish - durations[task_id]
    return es, ls


def _normalize(values: np.ndarray) -> np.ndarray:
    arr = values.astype(float).copy()
    min_v = float(np.min(arr)) if arr.size else 0.0
    max_v = float(np.max(arr)) if arr.size else 0.0
    if max_v - min_v < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - min_v) / (max_v - min_v)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    random.seed(int(seed))


def _resource_metrics(env: Any, assigned_tasks: Iterable[tuple[int, int, list[int], float, float]], makespan: float) -> tuple[float, float]:
    if makespan <= 0:
        return 0.0, 0.0

    worker_busy_time = 0.0
    station_busy_time = np.zeros(env.num_stations, dtype=float)
    for _, sid, team, start, end in assigned_tasks:
        duration = float(end) - float(start)
        worker_busy_time += duration * len(team)
        if sid >= 0:
            station_busy_time[int(sid)] += duration

    worker_util = worker_busy_time / (env.num_workers * makespan) if env.num_workers > 0 else 0.0
    max_slots = getattr(configs, "max_slots_per_station", 3)
    station_util = float(np.sum(station_busy_time)) / (env.num_stations * max_slots * makespan) if env.num_stations > 0 else 0.0
    return float(worker_util), float(station_util)


class PriorityDecoder:
    """使用真实 APAL 环境把优先级候选解解码为完整排程。"""

    def __init__(self, env: Any, balance_weight: float = 1.0, max_steps_factor: int = 4):
        self.env = env
        self.balance_weight = float(balance_weight)
        self.max_steps_factor = int(max_steps_factor)

    def decode(self, solution: PrioritySolution, seed: int) -> DecodeResult:
        _set_seed(seed)
        env = self.env
        env.reset(randomize_duration=False, randomize_workers=False, seed=seed)
        env.skip_obs_building = True

        task_static_feat = _to_numpy(env.task_static_feat)
        worker_skill_matrix = _to_numpy(env.worker_skill_matrix)
        max_steps = max(1, env.num_tasks * self.max_steps_factor)
        done = False
        deadlock_count = 0

        for _ in range(max_steps):
            if len(env.assigned_tasks) == env.num_tasks:
                done = True
                break

            task_mask_raw, station_mask_raw, _ = env.get_masks()
            task_mask = _to_numpy(task_mask_raw).astype(bool)
            station_mask = _to_numpy(station_mask_raw).astype(bool)

            if bool(task_mask.all()):
                if env.try_wait_for_resources():
                    continue
                deadlock_count += 1
                break

            available_tasks = np.where(~task_mask)[0]
            ordered_tasks = sorted(available_tasks, key=lambda t: solution.task_priority[int(t)], reverse=True)
            action = None

            for task_id in ordered_tasks:
                tid = int(task_id)
                valid_stations = np.where(~station_mask[tid])[0]
                if valid_stations.size == 0:
                    continue

                selected_station = int(max(valid_stations, key=lambda s: solution.station_priority[tid, int(s)]))
                task_skill = int(task_static_feat[tid, 1])
                worker_demand = max(1, int(task_static_feat[tid, 2]))
                worker_locks = _to_numpy(env.worker_locks)

                skilled_workers: list[int] = []
                for worker_id in range(env.num_workers):
                    if worker_skill_matrix[worker_id, task_skill] > 0.5:
                        if worker_locks[worker_id] == 0 or worker_locks[worker_id] == selected_station + 1:
                            skilled_workers.append(worker_id)

                if len(skilled_workers) < worker_demand:
                    continue

                skilled_workers.sort(key=lambda w: solution.worker_priority[tid, w], reverse=True)
                action = (tid, selected_station, skilled_workers[:worker_demand])
                break

            if action is None:
                if env.try_wait_for_resources():
                    continue
                deadlock_count += 1
                break

            _, _, done, info = env.step(action)
            if info.get("invalid_action"):
                deadlock_count += 1
                break

        complete = len(env.assigned_tasks) == env.num_tasks
        if complete:
            makespan = float(np.max(env.station_wall_clock))
            balance_std = float(np.std(env.station_loads))
            fitness = makespan + self.balance_weight * balance_std
            assigned_tasks = list(env.assigned_tasks)
        else:
            makespan = float(env.ideal_makespan * 3.0)
            balance_std = float(env.ideal_station_load * 3.0)
            fitness = makespan + self.balance_weight * balance_std
            assigned_tasks = []
            deadlock_count = max(1, deadlock_count)

        return DecodeResult(
            fitness=float(fitness),
            makespan=makespan,
            balance_std=balance_std,
            assigned_tasks=assigned_tasks,
            complete=complete,
            deadlock_count=deadlock_count,
        )


class AdvancedSchedulerBase:
    def __init__(self, env: Any, seed: int = 42, balance_weight: float = 1.0):
        self.env = env
        self.seed = int(seed)
        self.balance_weight = float(balance_weight)
        self.decoder = PriorityDecoder(env, balance_weight=balance_weight)
        self.num_tasks = env.num_tasks
        self.num_stations = env.num_stations
        self.num_workers = env.num_workers

    def _random_solution(self, rng: np.random.Generator) -> PrioritySolution:
        return PrioritySolution(
            task_priority=rng.random(self.num_tasks),
            station_priority=rng.random((self.num_tasks, self.num_stations)),
            worker_priority=rng.random((self.num_tasks, self.num_workers)),
        )

    def _seed_solution(self, rule: SeedRule, seed: int) -> PrioritySolution:
        rng = _rng(seed)
        solution = self._random_solution(rng)
        durations = _to_numpy(self.env.task_static_feat[:, 0]).astype(float)
        es, ls = _compute_cpm_times(self.env)

        if rule == "SPT":
            solution.task_priority = 1.0 - _normalize(durations)
        elif rule == "LPT":
            solution.task_priority = _normalize(durations)
        elif rule == "CPM":
            solution.task_priority = 1.0 - _normalize(ls)
        elif rule == "MSL":
            solution.task_priority = 0.5 * (1.0 - _normalize(ls)) + 0.5 * (1.0 - _normalize(durations))
        else:
            solution.task_priority = rng.random(self.num_tasks)

        # 工位偏好默认轻微倾向低编号工位，并保留随机扰动，避免完全同质。
        station_rank = 1.0 - _normalize(np.arange(self.num_stations, dtype=float))
        solution.station_priority = np.tile(station_rank, (self.num_tasks, 1)) + rng.normal(0.0, 0.03, (self.num_tasks, self.num_stations))
        solution.worker_priority = rng.random((self.num_tasks, self.num_workers))
        return solution

    def _initial_pool(self, count: int) -> list[PrioritySolution]:
        rules: list[SeedRule] = ["CPM", "SPT", "LPT", "MSL", "Random"]
        pool = [self._seed_solution(rule, self.seed + idx) for idx, rule in enumerate(rules[:count])]
        rng = _rng(self.seed + 10_000)
        while len(pool) < count:
            pool.append(self._random_solution(rng))
        return pool

    def _mutate(
        self,
        solution: PrioritySolution,
        rng: np.random.Generator,
        sigma: float = 0.20,
        task_rate: float = 0.05,
        station_rate: float = 0.03,
        worker_rate: float = 0.02,
    ) -> PrioritySolution:
        child = solution.clone()
        task_mask = rng.random(self.num_tasks) < task_rate
        child.task_priority[task_mask] += rng.normal(0.0, sigma, int(np.sum(task_mask)))

        station_mask = rng.random((self.num_tasks, self.num_stations)) < station_rate
        child.station_priority[station_mask] += rng.normal(0.0, sigma, int(np.sum(station_mask)))

        worker_mask = rng.random((self.num_tasks, self.num_workers)) < worker_rate
        child.worker_priority[worker_mask] += rng.normal(0.0, sigma, int(np.sum(worker_mask)))
        return child

    def _result_tuple(self, result: DecodeResult) -> tuple[float, float, list[tuple[int, int, list[int], float, float]]]:
        return result.makespan, result.balance_std, result.assigned_tasks


class BeamSearchScheduler(AdvancedSchedulerBase):
    def __init__(
        self,
        env: Any,
        beam_width: int = 4,
        branch_factor: int = 4,
        levels: int = 8,
        patience: int = 4,
        seed: int = 42,
        balance_weight: float = 1.0,
    ):
        super().__init__(env, seed=seed, balance_weight=balance_weight)
        self.beam_width = int(beam_width)
        self.branch_factor = int(branch_factor)
        self.levels = int(levels)
        self.patience = int(patience)

    def run(self) -> tuple[float, float, list[tuple[int, int, list[int], float, float]]]:
        print(f"--- 启动 Beam Search 基线: width={self.beam_width}, branch={self.branch_factor}, levels={self.levels} ---")
        start_time = time.time()
        rng = _rng(self.seed)
        beam: list[tuple[float, PrioritySolution, DecodeResult]] = []
        best_result: DecodeResult | None = None
        stale_rounds = 0

        for idx, solution in enumerate(self._initial_pool(max(self.beam_width, 5))):
            result = self.decoder.decode(solution, self.seed + idx)
            beam.append((result.fitness, solution, result))
            if best_result is None or result.fitness < best_result.fitness:
                best_result = result

        beam.sort(key=lambda item: item[0])
        beam = beam[: self.beam_width]

        for level in range(self.levels):
            candidates: list[tuple[float, PrioritySolution, DecodeResult]] = list(beam)
            previous_best = best_result.fitness if best_result is not None else float("inf")

            for _, parent, _ in beam:
                for branch_idx in range(self.branch_factor):
                    child = self._mutate(
                        parent,
                        rng,
                        sigma=0.18,
                        task_rate=0.03 + 0.01 * branch_idx,
                        station_rate=0.02,
                        worker_rate=0.015,
                    )
                    result = self.decoder.decode(child, self.seed + 1_000 + level * self.branch_factor + branch_idx)
                    candidates.append((result.fitness, child, result))
                    if best_result is None or result.fitness < best_result.fitness:
                        best_result = result

            candidates.sort(key=lambda item: item[0])
            beam = candidates[: self.beam_width]
            stale_rounds = stale_rounds + 1 if best_result and best_result.fitness >= previous_best - 1e-9 else 0
            print(f"[Beam {level + 1}/{self.levels}] Best Fit={beam[0][0]:.2f}, Makespan={beam[0][2].makespan:.2f}")
            if stale_rounds >= self.patience:
                break

        assert best_result is not None
        print(f"--- Beam Search 结束，耗时 {time.time() - start_time:.1f}s，Best Mk={best_result.makespan:.2f} ---")
        return self._result_tuple(best_result)


class IteratedGreedyScheduler(AdvancedSchedulerBase):
    def __init__(
        self,
        env: Any,
        iterations: int = 80,
        destroy_ratio: float = 0.10,
        noise_sigma: float = 0.25,
        seed: int = 42,
        balance_weight: float = 1.0,
    ):
        super().__init__(env, seed=seed, balance_weight=balance_weight)
        self.iterations = int(iterations)
        self.destroy_ratio = float(destroy_ratio)
        self.noise_sigma = float(noise_sigma)

    def _destroy_repair(self, solution: PrioritySolution, rng: np.random.Generator) -> PrioritySolution:
        child = solution.clone()
        destroy_count = max(1, int(math.ceil(self.num_tasks * self.destroy_ratio)))
        durations = _to_numpy(self.env.task_static_feat[:, 0]).astype(float)
        _, ls = _compute_cpm_times(self.env)
        critical_score = _normalize(durations) + (1.0 - _normalize(ls))
        prob = critical_score + 1e-6
        prob = prob / np.sum(prob)
        task_ids = rng.choice(self.num_tasks, size=destroy_count, replace=False, p=prob)

        child.task_priority[task_ids] = rng.random(destroy_count)
        child.station_priority[task_ids, :] = rng.random((destroy_count, self.num_stations))
        child.worker_priority[task_ids, :] = rng.random((destroy_count, self.num_workers))
        child.task_priority[task_ids] += rng.normal(0.0, self.noise_sigma, destroy_count)
        return child

    def run(self) -> tuple[float, float, list[tuple[int, int, list[int], float, float]]]:
        print(f"--- 启动 Iterated Greedy / Destroy-and-Repair 基线: iter={self.iterations}, destroy={self.destroy_ratio:.2f} ---")
        start_time = time.time()
        rng = _rng(self.seed)
        current = self._seed_solution("CPM", self.seed)
        current_result = self.decoder.decode(current, self.seed)
        best = current.clone()
        best_result = current_result

        for iteration in range(self.iterations):
            candidate = self._destroy_repair(current, rng)
            result = self.decoder.decode(candidate, self.seed + 2_000 + iteration)
            if result.fitness < current_result.fitness:
                current = candidate
                current_result = result
            if result.fitness < best_result.fitness:
                best = candidate.clone()
                best_result = result
            if (iteration + 1) % max(1, self.iterations // 10) == 0:
                print(f"[IG {iteration + 1}/{self.iterations}] Best Fit={best_result.fitness:.2f}, Makespan={best_result.makespan:.2f}")

        _ = best
        print(f"--- IG 结束，耗时 {time.time() - start_time:.1f}s，Best Mk={best_result.makespan:.2f} ---")
        return self._result_tuple(best_result)


class SimulatedAnnealingScheduler(AdvancedSchedulerBase):
    def __init__(
        self,
        env: Any,
        iterations: int = 120,
        initial_temp: float = 0.05,
        cooling: float = 0.96,
        min_temp: float = 1e-4,
        seed: int = 42,
        balance_weight: float = 1.0,
    ):
        super().__init__(env, seed=seed, balance_weight=balance_weight)
        self.iterations = int(iterations)
        self.initial_temp = float(initial_temp)
        self.cooling = float(cooling)
        self.min_temp = float(min_temp)

    def _neighbor(self, solution: PrioritySolution, rng: np.random.Generator) -> PrioritySolution:
        child = solution.clone()
        op = int(rng.integers(0, 4))
        if op == 0:
            task_id = int(rng.integers(0, self.num_tasks))
            child.task_priority[task_id] += rng.normal(0.0, 0.25)
        elif op == 1 and self.num_tasks >= 2:
            a, b = rng.choice(self.num_tasks, size=2, replace=False)
            child.task_priority[a], child.task_priority[b] = child.task_priority[b], child.task_priority[a]
        elif op == 2:
            task_id = int(rng.integers(0, self.num_tasks))
            child.station_priority[task_id, :] += rng.normal(0.0, 0.20, self.num_stations)
        else:
            task_id = int(rng.integers(0, self.num_tasks))
            child.worker_priority[task_id, :] += rng.normal(0.0, 0.20, self.num_workers)
        return child

    def run(self) -> tuple[float, float, list[tuple[int, int, list[int], float, float]]]:
        print(f"--- 启动 Simulated Annealing 基线: iter={self.iterations}, T0={self.initial_temp}, cooling={self.cooling} ---")
        start_time = time.time()
        rng = _rng(self.seed)
        current = self._seed_solution("CPM", self.seed)
        current_result = self.decoder.decode(current, self.seed)
        best = current.clone()
        best_result = current_result
        temp = self.initial_temp

        for iteration in range(self.iterations):
            candidate = self._neighbor(current, rng)
            result = self.decoder.decode(candidate, self.seed + 3_000 + iteration)
            delta_norm = (result.fitness - current_result.fitness) / max(1e-6, float(self.env.ideal_makespan))
            accept = delta_norm < 0.0 or rng.random() < math.exp(-delta_norm / max(temp, 1e-12))
            if accept:
                current = candidate
                current_result = result
            if result.fitness < best_result.fitness:
                best = candidate.clone()
                best_result = result
            temp = max(self.min_temp, temp * self.cooling)
            if (iteration + 1) % max(1, self.iterations // 10) == 0:
                print(f"[SA {iteration + 1}/{self.iterations}] T={temp:.5f}, Best Fit={best_result.fitness:.2f}, Makespan={best_result.makespan:.2f}")

        _ = best
        print(f"--- SA 结束，耗时 {time.time() - start_time:.1f}s，Best Mk={best_result.makespan:.2f} ---")
        return self._result_tuple(best_result)


def build_metrics(env: Any, makespan: float, balance_std: float, assigned_tasks: list[tuple[int, int, list[int], float, float]], inference_time: float) -> dict[str, float]:
    complete = len(assigned_tasks) == env.num_tasks
    if complete:
        worker_util, station_util = _resource_metrics(env, assigned_tasks, makespan)
        return {
            "makespan": float(makespan),
            "workload_balance_std": float(balance_std),
            "worker_utilization": float(worker_util),
            "station_utilization": float(station_util),
            "inference_time": float(inference_time),
            "valid": 1.0,
            "deadlock_count": 0,
            "completion_rate": 1.0,
        }

    return {
        "makespan": float(env.ideal_makespan * 3.0),
        "workload_balance_std": float(env.ideal_station_load * 3.0),
        "worker_utilization": 0.0,
        "station_utilization": 0.0,
        "inference_time": float(inference_time),
        "valid": 0.0,
        "deadlock_count": 1,
        "completion_rate": 0.0,
    }
