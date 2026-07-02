from __future__ import annotations

import time

import numpy as np
import torch

from configs import configs
from environment import AirLineEnv_Graph
from runtime.multiscale import (
    BenchmarkScore,
    parse_reference_makespans,
    score_multi_benchmark,
)
from runtime.paths import resolve_workspace_path
from training.observation import refresh_env_observation


def _get_cpm_earliest_starts(ctx: dict) -> np.ndarray:
    """基于静态工序图计算 CPM 最早开工时间，用于 APAL rollout 诊断。"""
    cached = ctx.get("_diag_cpm_earliest_starts")
    if cached is not None:
        return cached

    task_static_feat = ctx["task_static_feat"]
    durations = (
        task_static_feat[:, 0].detach().cpu().numpy()
        if torch.is_tensor(task_static_feat)
        else np.asarray(task_static_feat)[:, 0]
    )
    num_tasks = len(durations)
    predecessors = ctx["predecessors"]
    successors = ctx["successors"]

    def _neighbors(container, idx: int):
        return container.get(idx, []) if hasattr(container, "get") else container[idx]

    indegree = [len(_neighbors(predecessors, i)) for i in range(num_tasks)]
    queue = [i for i, deg in enumerate(indegree) if deg == 0]
    topo_order = []
    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        for nxt in _neighbors(successors, node):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    earliest = np.zeros(num_tasks, dtype=float)
    for node in topo_order:
        for pred in _neighbors(predecessors, node):
            earliest[node] = max(earliest[node], earliest[pred] + durations[pred])
    ctx["_diag_cpm_earliest_starts"] = earliest
    return earliest


def compute_apal_rollout_diagnostics(env_proxy, snapshot: dict, masks) -> dict:
    """从轻量 snapshot 计算 APAL rollout 诊断，不修改环境状态。"""
    task_mask, _, _ = masks
    current_time = float(snapshot["current_time"])
    worker_free_time = np.asarray(snapshot["worker_free_time"], dtype=float)
    station_slots = np.asarray(
        snapshot.get(
            "station_available_slots",
            np.full(len(snapshot["station_loads"]), configs.max_slots_per_station),
        ),
        dtype=float,
    )

    schedulable_tasks = float((~task_mask).sum().item())
    worker_waits = np.maximum(0.0, worker_free_time - current_time)
    avg_worker_wait_h = float(worker_waits.mean()) if worker_waits.size > 0 else 0.0
    worker_idle_ratio = float(np.mean(worker_free_time <= current_time)) if worker_free_time.size > 0 else 0.0

    active_counts = np.zeros(len(station_slots), dtype=float)
    station_next_finish = [[] for _ in range(len(station_slots))]
    for _, station_id, _, _, finish_time in snapshot["assigned_tasks"]:
        if station_id >= 0 and finish_time > current_time:
            active_counts[station_id] += 1.0
            station_next_finish[station_id].append(float(finish_time))

    free_slots = np.maximum(0.0, station_slots - active_counts)
    station_slot_vacancy_ratio = float(free_slots.sum() / max(1.0, station_slots.sum()))
    station_waits = []
    for sid, finishes in enumerate(station_next_finish):
        if len(finishes) >= max(1, int(station_slots[sid])):
            station_waits.append(max(0.0, min(finishes) - current_time))
        else:
            station_waits.append(0.0)
    avg_station_wait_h = float(np.mean(station_waits)) if station_waits else 0.0

    critical_offset_values = []
    dataset_idx = snapshot.get("dataset_idx", 0)
    pool = getattr(env_proxy, "dataset_pool", [])
    ctx = pool[dataset_idx] if dataset_idx < len(pool) and pool[dataset_idx] is not None else {}
    is_critical = np.asarray(ctx.get("is_critical", []), dtype=bool)
    if is_critical.size > 0:
        cpm_earliest = _get_cpm_earliest_starts(ctx)
        for task_id, _, _, start_time, _ in snapshot["assigned_tasks"]:
            if 0 <= task_id < len(is_critical) and is_critical[task_id]:
                critical_offset_values.append(max(0.0, float(start_time) - float(cpm_earliest[task_id])))

    return {
        "schedulable_tasks": schedulable_tasks,
        "avg_worker_wait_h": avg_worker_wait_h,
        "avg_station_wait_h": avg_station_wait_h,
        "avg_resource_wait_h": avg_worker_wait_h + avg_station_wait_h,
        "station_slot_vacancy_ratio": station_slot_vacancy_ratio,
        "worker_idle_ratio": worker_idle_ratio,
        "critical_start_offset_h": float(np.mean(critical_offset_values)) if critical_offset_values else np.nan,
    }


def evaluate_model(
    env,
    agent,
    num_runs=1,
    temperature=None,
    writer=None,
    current_ep=0,
    scenario_names=None,
):
    """按固定场景评估当前 APAL 初始调度策略。"""
    if temperature is None:
        temperature = getattr(configs, "eval_temperature", 0.0)

    was_training = bool(agent.policy.training)
    agent.policy.eval()
    verbose_eval = bool(getattr(configs, "verbose_eval_progress", writer is None))

    backups = {
        "enable_dynamic_events": getattr(configs, "enable_dynamic_events", False),
        "enable_station_breakdown": getattr(configs, "enable_station_breakdown", False),
        "enable_material_delay": getattr(configs, "enable_material_delay", False),
    }
    scenarios = [
        {"name": "0_Standard", "rand_dur": False, "rand_w": False, "dyn_ev": False, "seed": None},
        {"name": "1_DurationNoise", "rand_dur": True, "rand_w": False, "dyn_ev": False, "seed": None},
        {"name": "2_WorkerNoise", "rand_dur": False, "rand_w": True, "dyn_ev": False, "seed": None},
        {"name": "3_DynamicEvents", "rand_dur": False, "rand_w": False, "dyn_ev": True, "seed": None},
    ]
    if scenario_names is not None:
        aliases = {
            "standard": "0_Standard",
            "duration_noise": "1_DurationNoise",
            "worker_noise": "2_WorkerNoise",
            "dynamic_events": "3_DynamicEvents",
        }
        selected = {aliases.get(str(name).lower(), str(name)) for name in scenario_names}
        scenarios = [scenario for scenario in scenarios if scenario["name"] in selected]
        if not scenarios:
            raise ValueError(f"未选择任何有效评估场景: {scenario_names}")

    base_eval_seed = int(getattr(configs, "seed", 42))
    runs_per_scenario = max(1, int(num_runs))
    for scenario_idx, scenario in enumerate(scenarios):
        scenario["seed"] = base_eval_seed + scenario_idx * runs_per_scenario

    scenario_results = []
    try:
        for sc in scenarios:
            setattr(configs, "enable_dynamic_events", bool(sc["dyn_ev"]))
            setattr(configs, "enable_station_breakdown", bool(sc["dyn_ev"]))
            setattr(configs, "enable_material_delay", bool(sc["dyn_ev"]))

            sc_makespans = []
            sc_balances = []
            sc_rewards = []
            sc_schedules = []
            sc_durations = []
            sc_worker_utils = []
            sc_station_utils = []

            for run_idx in range(runs_per_scenario):
                state = env.reset(randomize_duration=sc["rand_dur"], randomize_workers=sc["rand_w"], seed=sc["seed"] + run_idx)
                done = False
                total_reward = 0.0
                device = agent.device
                start_time = time.time()

                while not done:
                    task_mask, station_mask, worker_mask = env.get_masks()
                    if task_mask.all():
                        if env.try_wait_for_resources():
                            state = refresh_env_observation(env)
                            continue
                        done = True
                        break

                    action_ret = agent.select_action(
                        state.to(device),
                        mask_task=task_mask.to(device),
                        mask_station_matrix=station_mask.to(device),
                        mask_worker=worker_mask.to(device),
                        deterministic=(temperature == 0.0),
                        temperature=temperature,
                        is_eval=True,
                    )
                    if action_ret[0] is None:
                        break
                    action, _, _, _, is_invalid = action_ret
                    if getattr(configs, "ablation_no_mask", False) and is_invalid:
                        break
                    state, reward, done, _info = env.step(action)
                    total_reward += float(reward)

                elapsed = time.time() - start_time
                complete = len(env.assigned_tasks) == env.num_tasks
                if not complete:
                    sc_makespans.append(float(env.ideal_makespan * 3.0))
                    sc_balances.append(float(env.ideal_station_load * 3.0))
                    penalty = configs.deadlock_penalty_constant * configs.r_coef_makespan * configs.reward_scale * 4
                    sc_rewards.append(float(total_reward - penalty))
                    sc_schedules.append([])
                    sc_worker_utils.append(0.0)
                    sc_station_utils.append(0.0)
                else:
                    final_makespan = float(np.max(env.station_wall_clock))
                    sc_makespans.append(final_makespan)
                    sc_balances.append(float(np.std(env.station_loads)))
                    sc_rewards.append(float(total_reward))
                    sc_schedules.append(list(env.assigned_tasks))
                    worker_busy_time = 0.0
                    station_busy_time = np.zeros(env.num_stations)
                    for _tid, sid, team, start, end in env.assigned_tasks:
                        dur = float(end) - float(start)
                        worker_busy_time += dur * len(team)
                        if sid >= 0:
                            station_busy_time[sid] += dur
                    w_util = worker_busy_time / (env.num_workers * final_makespan) if final_makespan > 0 else 0.0
                    max_slots = getattr(configs, "max_slots_per_station", 3)
                    s_util = np.sum(station_busy_time) / (env.num_stations * max_slots * final_makespan) if final_makespan > 0 else 0.0
                    sc_worker_utils.append(float(w_util))
                    sc_station_utils.append(float(s_util))
                sc_durations.append(float(elapsed))
                if verbose_eval:
                    print(
                        f"[Eval][RunResult] scenario={sc['name']} run={run_idx + 1}/{runs_per_scenario} "
                        f"complete={int(complete)} tasks={len(getattr(env, 'assigned_tasks', []))}/{getattr(env, 'num_tasks', '?')} "
                        f"Mk={float(sc_makespans[-1]):.2f} Time={float(sc_durations[-1]):.2f}s",
                        flush=True,
                    )

            best_idx = int(np.argmin(sc_makespans))
            sc_res = {
                "name": sc["name"],
                "makespan": float(np.mean(sc_makespans)),
                "balance": float(np.mean(sc_balances)),
                "reward": float(np.mean(sc_rewards)),
                "schedule": sc_schedules[best_idx],
                "duration": float(np.mean(sc_durations)),
                "w_util": float(np.mean(sc_worker_utils)),
                "s_util": float(np.mean(sc_station_utils)),
            }
            scenario_results.append(sc_res)
            if writer is not None:
                writer.add_scalar(f"Eval_Scenario/{sc['name']}_Makespan", sc_res["makespan"], current_ep)
                writer.add_scalar(f"Eval_Scenario/{sc['name']}_Reward", sc_res["reward"], current_ep)
                writer.add_scalar(f"Eval_Scenario/{sc['name']}_WorkerUtil", sc_res["w_util"], current_ep)
                writer.add_scalar(f"Eval_Scenario/{sc['name']}_StationUtil", sc_res["s_util"], current_ep)
            if verbose_eval:
                print(
                    f"[Eval][ScenarioResult] scenario={sc['name']} Mk={sc_res['makespan']:.2f} "
                    f"Bal={sc_res['balance']:.2f} Reward={sc_res['reward']:.2f} AvgTime={sc_res['duration']:.2f}s",
                    flush=True,
                )
    finally:
        for key, value in backups.items():
            setattr(configs, key, value)
        if was_training:
            agent.policy.train()

    avg_makespan = float(np.mean([r["makespan"] for r in scenario_results]))
    avg_balance = float(np.mean([r["balance"] for r in scenario_results]))
    avg_reward = float(np.mean([r["reward"] for r in scenario_results]))
    avg_duration = float(np.mean([r["duration"] for r in scenario_results]))
    avg_w_util = float(np.mean([r["w_util"] for r in scenario_results]))
    avg_s_util = float(np.mean([r["s_util"] for r in scenario_results]))
    best_sch = scenario_results[0]["schedule"]
    if verbose_eval:
        print(
            f"[Eval][Result] Mk={avg_makespan:.2f} Bal={avg_balance:.2f} Reward={avg_reward:.2f} "
            f"AvgTime={avg_duration:.2f}s WUtil={avg_w_util * 100:.1f}% "
            f"SUtil={avg_s_util * 100:.1f}% BestTasks={len(best_sch)}",
            flush=True,
        )
    return avg_makespan, avg_balance, avg_reward, best_sch, avg_duration, avg_w_util, avg_s_util


def evaluate_initial_multi_benchmark(agent, config_obj=configs, writer=None, current_ep=0):
    """在多个固定初始调度基准集上评估，并计算归一化综合分。"""
    refs = parse_reference_makespans(getattr(config_obj, "multi_benchmark_reference_makespans", {}))
    rows: list[BenchmarkScore] = []
    device = agent.device
    was_training = bool(agent.policy.training)
    paths = list(getattr(config_obj, "multi_benchmark_data_paths", []))
    if not paths:
        raise ValueError("multi_benchmark_data_paths 不能为空")

    backups = {
        "enable_dynamic_events": getattr(config_obj, "enable_dynamic_events", False),
        "enable_station_breakdown": getattr(config_obj, "enable_station_breakdown", False),
        "enable_material_delay": getattr(config_obj, "enable_material_delay", False),
        "enable_online_duration_perturb": getattr(config_obj, "enable_online_duration_perturb", False),
        "enable_worker_fatigue": getattr(config_obj, "enable_worker_fatigue", False),
        "randomize_durations": getattr(config_obj, "randomize_durations", False),
    }
    for key in backups:
        setattr(config_obj, key, False)

    try:
        agent.policy.eval()
        for raw_path in paths:
            data_path = resolve_workspace_path(raw_path)
            benchmark_name = data_path.stem
            if benchmark_name not in refs:
                raise ValueError(f"缺少基准 {benchmark_name} 的 reference makespan")
            benchmark_seed = int(getattr(config_obj, "seed", 42)) + len(rows)
            env = AirLineEnv_Graph(data_path_or_dir=str(data_path), seed=benchmark_seed)
            state = env.reset(randomize_duration=False, randomize_workers=False, seed=benchmark_seed)
            done = False
            invalid_step_count = 0
            start_time = time.time()

            for _ in range(env.num_tasks * 3):
                if done:
                    break
                task_mask, station_mask, worker_mask = env.get_masks()
                if task_mask.all():
                    if env.try_wait_for_resources():
                        state = refresh_env_observation(env)
                        continue
                    invalid_step_count += 1
                    break
                action_ret = agent.select_action(
                    state.to(device),
                    mask_task=task_mask.to(device),
                    mask_station_matrix=station_mask.to(device),
                    mask_worker=worker_mask.to(device),
                    deterministic=True,
                    temperature=0.0,
                    is_eval=True,
                )
                if action_ret[0] is None:
                    invalid_step_count += 1
                    break
                action, _, _, _, is_invalid = action_ret
                if is_invalid:
                    invalid_step_count += 1
                    break
                state, _reward, done, info = env.step(action)
                if info.get("invalid_action", False):
                    invalid_step_count += 1
                    break

            complete = len(env.assigned_tasks) == env.num_tasks
            makespan = float(np.max(env.station_wall_clock)) if complete and invalid_step_count == 0 else float(env.ideal_makespan * 3.0)
            reference = float(refs[benchmark_name])
            row = BenchmarkScore(
                benchmark_name=benchmark_name,
                data_path=str(data_path),
                makespan=makespan,
                reference_makespan=reference,
                normalized_score=float(makespan / reference),
                complete=bool(complete),
                invalid_step_count=int(invalid_step_count),
                inference_time=float(time.time() - start_time),
            )
            rows.append(row)
            if writer is not None:
                writer.add_scalar(f"MultiBenchmark/{benchmark_name}_NormalizedScore", row.normalized_score, current_ep)
                writer.add_scalar(f"MultiBenchmark/{benchmark_name}_Makespan", row.makespan, current_ep)
                writer.add_scalar(f"MultiBenchmark/{benchmark_name}_InvalidSteps", row.invalid_step_count, current_ep)
    finally:
        for key, value in backups.items():
            setattr(config_obj, key, value)
        if was_training:
            agent.policy.train()
    return score_multi_benchmark(rows)


__all__ = [
    "compute_apal_rollout_diagnostics",
    "evaluate_initial_multi_benchmark",
    "evaluate_model",
]
