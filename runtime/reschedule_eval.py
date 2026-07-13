from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from configs import configs
from core.time_comparison import release_time_tolerance, time_reached_scalar
from runtime.checkpoints import load_checkpoint
from runtime.paths import PROJECT_ROOT, resolve_workspace_path
from runtime.reschedule_manifest import resolve_manifest_eval_entry
from utils.reschedule import (
    calculate_reschedule_composite_score,
    calculate_stability_metrics,
    load_baseline_schedule,
    load_reschedule_scenarios,
    sample_task_delay_scenario,
    save_reschedule_scenarios,
)
from training.observation import refresh_env_observation


class MaskEnvironmentMismatchError(RuntimeError):
    """动作掩码允许的动作被环境硬边界拒绝。"""


def _mask_mismatch_message(
    env,
    *,
    action,
    info: dict,
    scenario_id: str,
) -> str:
    task_id, station_id, team = action
    release_time = float(info.get("release_time", np.nan))
    current_time = float(info.get("current_time", getattr(env, "current_time", np.nan)))
    tolerance = release_time_tolerance(configs)
    return (
        "动作掩码与环境边界不一致: "
        f"scenario={scenario_id} reason={info.get('error', info.get('reason', 'unknown'))} "
        f"task={int(task_id)} station={int(station_id)} team={[int(w) for w in team]} "
        f"current_time={current_time:.17g} release_time={release_time:.17g} "
        f"gap_h={release_time - current_time:.17g} tolerance_h={tolerance:.17g} "
        f"release_dtype={getattr(getattr(env, 'task_material_ready', None), 'dtype', 'unknown')}"
    )


def ensure_reschedule_baseline_available(config_obj=configs) -> Path | None:
    """重调度训练前确保 baseline CSV 存在；不存在时用初始模型生成一次。"""
    if not getattr(config_obj, "enable_reschedule_mode", False):
        return None
    manifest_entry = resolve_manifest_eval_entry(config_obj)
    if manifest_entry is not None:
        if not manifest_entry.baseline_schedule_path.exists():
            raise FileNotFoundError(f"manifest 指向的重调度 baseline 不存在: {manifest_entry.baseline_schedule_path}")
        return manifest_entry.baseline_schedule_path
    baseline_path = resolve_workspace_path(getattr(config_obj, "reschedule_baseline_schedule_path", "results/final_schedule.csv"))
    if baseline_path.exists():
        return baseline_path
    model_path = resolve_workspace_path(
        getattr(config_obj, "reschedule_baseline_model_path", "checkpoints/initial_schedule/bestmodel/best_model.pth")
    )
    if not model_path.exists():
        raise FileNotFoundError(f"重调度 baseline 不存在，且找不到用于生成 baseline 的初始模型: {model_path}")

    backup = {
        "enable_reschedule_mode": getattr(config_obj, "enable_reschedule_mode", False),
        "task_feat_dim": getattr(config_obj, "task_feat_dim", 18),
        "randomize_durations": getattr(config_obj, "randomize_durations", False),
        "enable_dynamic_events": getattr(config_obj, "enable_dynamic_events", False),
    }
    try:
        setattr(config_obj, "enable_reschedule_mode", False)
        setattr(config_obj, "task_feat_dim", 18)
        setattr(config_obj, "randomize_durations", False)
        setattr(config_obj, "enable_dynamic_events", False)
        from scripts.generate_schedule import generate_schedule

        df = generate_schedule(model_path=str(model_path))
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        if baseline_path != PROJECT_ROOT / "results" / "final_schedule.csv":
            df.to_csv(baseline_path, index=False)
    finally:
        for key, value in backup.items():
            setattr(config_obj, key, value)
    if not baseline_path.exists():
        raise FileNotFoundError(f"baseline 自动生成后仍未找到: {baseline_path}")
    return baseline_path


def load_warm_start_weights_with_input_expansion(model: torch.nn.Module, model_path: Path, device: torch.device) -> dict[str, int]:
    """加载初始调度模型；当 task 输入维度扩展时复制旧权重的已有列。"""
    checkpoint = load_checkpoint(model_path, map_location=device)
    source_state = checkpoint.state_dict
    target_state = model.state_dict()
    loaded_exact = 0
    loaded_expanded = 0
    skipped = 0

    for key, target_tensor in target_state.items():
        source_tensor = source_state.get(key) if isinstance(source_state, dict) else None
        if source_tensor is None:
            skipped += 1
            continue
        if tuple(source_tensor.shape) == tuple(target_tensor.shape):
            target_state[key] = source_tensor.to(target_tensor.device, dtype=target_tensor.dtype)
            loaded_exact += 1
            continue
        if (
            source_tensor.ndim == target_tensor.ndim == 2
            and source_tensor.shape[0] == target_tensor.shape[0]
            and source_tensor.shape[1] <= target_tensor.shape[1]
        ):
            patched = target_tensor.clone()
            patched[:, : source_tensor.shape[1]] = source_tensor.to(target_tensor.device, dtype=target_tensor.dtype)
            target_state[key] = patched
            loaded_expanded += 1
            continue
        skipped += 1

    model.load_state_dict(target_state, strict=True)
    return {"loaded_exact": loaded_exact, "loaded_expanded": loaded_expanded, "skipped": skipped}


def ensure_reschedule_eval_scenarios_available(config_obj=configs) -> Path | None:
    """确保重调度验证使用固定场景文件，而不是每次临时随机生成。"""
    if not getattr(config_obj, "enable_reschedule_mode", False):
        return None
    manifest_entry = resolve_manifest_eval_entry(config_obj)
    if manifest_entry is not None and manifest_entry.scenario_path is not None:
        if not manifest_entry.scenario_path.exists():
            raise FileNotFoundError(f"manifest 指向的固定重调度场景不存在: {manifest_entry.scenario_path}")
        return manifest_entry.scenario_path
    scenario_path = resolve_workspace_path(getattr(config_obj, "reschedule_eval_scenario_path", "results/reschedule_eval_scenarios.csv"))
    if scenario_path.exists():
        return scenario_path

    baseline_path = ensure_reschedule_baseline_available(config_obj)
    if baseline_path is None:
        return None
    baseline = load_baseline_schedule(baseline_path)
    num_scenarios = max(1, int(getattr(config_obj, "reschedule_eval_num_scenarios", 4)))
    seed = int(getattr(config_obj, "reschedule_eval_scenario_seed", 42))
    scenarios = []
    for idx in range(num_scenarios):
        scenario = sample_task_delay_scenario(
            baseline,
            rng=np.random.RandomState(seed + idx),
            min_start_ratio=float(getattr(config_obj, "reschedule_start_time_min_ratio", 0.15)),
            max_start_ratio=float(getattr(config_obj, "reschedule_start_time_max_ratio", 0.65)),
            task_prob=float(getattr(config_obj, "reschedule_delay_task_prob", 0.08)),
            delay_min=float(getattr(config_obj, "reschedule_delay_min", 5.0)),
            delay_max=float(getattr(config_obj, "reschedule_delay_max", 30.0)),
        )
        scenarios.append((f"eval_{idx:03d}", scenario))
    save_reschedule_scenarios(scenario_path, scenarios)
    print(f"固定重调度验证场景已生成: {scenario_path}")
    return scenario_path


def _compute_assignment_utilization(env, final_makespan: float) -> tuple[float, float]:
    worker_busy_time = 0.0
    station_busy_time = np.zeros(env.num_stations)
    for _tid, sid, team, start, end in env.assigned_tasks:
        dur = max(0.0, float(end) - float(start))
        worker_busy_time += dur * len(team)
        if sid >= 0:
            station_busy_time[sid] += dur
    worker_util = worker_busy_time / (env.num_workers * final_makespan) if final_makespan > 0 else 0.0
    max_slots = getattr(configs, "max_slots_per_station", 3)
    station_util = np.sum(station_busy_time) / (env.num_stations * max_slots * final_makespan) if final_makespan > 0 else 0.0
    return float(worker_util), float(station_util)


def _compute_reschedule_constraint_metrics(env) -> dict[str, float]:
    baseline = getattr(env, "baseline_schedule", None)
    start_time = float(getattr(env, "reschedule_start_time", 0.0))
    assigned_by_task = {int(item[0]): item for item in env.assigned_tasks}
    metrics = {
        "frozen_violation_count": 0.0,
        "release_violation_count": 0.0,
        "precedence_violation_count": 0.0,
        "worker_overlap_violation_count": 0.0,
        "station_slot_violation_count": 0.0,
        "skill_violation_count": 0.0,
        "demand_violation_count": 0.0,
        "fixed_station_violation_count": 0.0,
        "station_range_violation_count": 0.0,
        "physical_station_violation_count": 0.0,
        "worker_station_binding_violation_count": 0.0,
        "duplicate_task_count": 0.0,
        "missing_task_count": 0.0,
        "takt_h": 0.0,
        "takt_violation_h": 0.0,
        "lower_bound_h": float(getattr(env, "reschedule_lower_bound", 0.0)),
        "takt_feasible": float(bool(getattr(env, "reschedule_takt_feasible", True))),
        "start_deviation_mean_h": 0.0,
        "station_change_rate": 0.0,
        "team_change_rate": 0.0,
    }

    task_ids = [int(item[0]) for item in env.assigned_tasks]
    metrics["duplicate_task_count"] = float(len(task_ids) - len(set(task_ids)))
    metrics["missing_task_count"] = float(max(0, env.num_tasks - len(set(task_ids))))

    if baseline is not None:
        metrics["takt_h"] = float(baseline.makespan)
        final_makespan = float(np.max(env.station_wall_clock)) if len(env.station_wall_clock) > 0 else 0.0
        metrics["takt_violation_h"] = max(0.0, final_makespan - float(baseline.makespan))
        stability = calculate_stability_metrics(baseline, env.assigned_tasks, current_time=start_time)
        metrics["start_deviation_mean_h"] = float(stability["start_deviation_mean_h"])
        metrics["station_change_rate"] = float(stability["station_change_rate"])
        metrics["team_change_rate"] = float(stability["team_change_rate"])

        for task_id, base_task in baseline.tasks.items():
            if base_task.start > start_time + 1e-9:
                continue
            assigned = assigned_by_task.get(int(task_id))
            if assigned is None:
                metrics["frozen_violation_count"] += 1.0
                continue
            _tid, sid, team, start, end = assigned
            if (
                int(sid) != int(base_task.station_id)
                or set(int(w) for w in team) != set(base_task.team)
                or abs(float(start) - float(base_task.start)) > 1e-5
                or abs(float(end) - float(base_task.end)) > 1e-5
            ):
                metrics["frozen_violation_count"] += 1.0

    if hasattr(env, "task_material_ready"):
        for task_id, _sid, _team, start, _end in env.assigned_tasks:
            release_time = float(env.task_material_ready[int(task_id)])
            if not time_reached_scalar(
                release_time,
                float(start),
                release_time_tolerance(configs),
            ):
                metrics["release_violation_count"] += 1.0

    if hasattr(env, "raw_data") and "precedence_edges" in env.raw_data:
        edges = env.raw_data["precedence_edges"]
        edges_np = edges.detach().cpu().numpy() if hasattr(edges, "detach") else np.asarray(edges)
        for src, dst in edges_np.T:
            pred = assigned_by_task.get(int(src))
            succ = assigned_by_task.get(int(dst))
            if pred is not None and succ is not None and float(pred[4]) > float(succ[3]) + 1e-5:
                metrics["precedence_violation_count"] += 1.0

    intervals_by_worker: dict[int, list[tuple[float, float]]] = {}
    intervals_by_station: dict[int, list[tuple[float, float]]] = {}
    for task_id, sid, team, start, end in env.assigned_tasks:
        duration = float(end) - float(start)
        if duration <= 1e-8:
            continue
        demand = max(1, int(env.task_static_feat[int(task_id), 2].item()))
        if len(team) < demand:
            metrics["demand_violation_count"] += 1.0
        skill_id = int(env.task_static_feat[int(task_id), 1].item())
        for worker_id in team:
            if worker_id < 0 or worker_id >= env.num_workers or env.worker_skill_matrix[int(worker_id), skill_id] < 0.5:
                metrics["skill_violation_count"] += 1.0
            intervals_by_worker.setdefault(int(worker_id), []).append((float(start), float(end)))
        if sid >= 0:
            intervals_by_station.setdefault(int(sid), []).append((float(start), float(end)))

    for intervals in intervals_by_worker.values():
        intervals.sort()
        for (_, prev_end), (next_start, _) in zip(intervals, intervals[1:]):
            if prev_end > next_start + 1e-5:
                metrics["worker_overlap_violation_count"] += 1.0

    max_slots = int(getattr(configs, "max_slots_per_station", 3))
    for intervals in intervals_by_station.values():
        events: list[tuple[float, int]] = []
        for start, end in intervals:
            events.append((start, 1))
            events.append((end, -1))
        events.sort(key=lambda item: (item[0], item[1]))
        active = 0
        for _time_point, delta in events:
            active += delta
            if active > max_slots:
                metrics["station_slot_violation_count"] += 1.0
                break

    # 统一约束引擎是排程合法性的权威来源；上面的专项统计保留用于兼容旧日志。
    central_report = env.validate_assignments(env.assigned_tasks)
    central = central_report.violations
    for key in (
        "precedence_violation_count",
        "worker_overlap_violation_count",
        "station_slot_violation_count",
        "skill_violation_count",
        "demand_violation_count",
        "fixed_station_violation_count",
        "station_range_violation_count",
        "physical_station_violation_count",
        "worker_station_binding_violation_count",
        "duplicate_task_count",
        "missing_task_count",
    ):
        metrics[key] = float(central[key])

    return metrics


def evaluate_reschedule_model(env, agent, num_runs=4, temperature=None, writer=None, current_ep=0):
    """评估 APAL 预测-反应式重调度策略。"""
    if temperature is None:
        temperature = getattr(configs, "eval_temperature", 0.0)

    was_training = bool(agent.policy.training)
    agent.policy.eval()
    backups = {
        "enable_dynamic_events": getattr(configs, "enable_dynamic_events", False),
        "enable_station_breakdown": getattr(configs, "enable_station_breakdown", False),
        "enable_material_delay": getattr(configs, "enable_material_delay", False),
        "enable_online_duration_perturb": getattr(configs, "enable_online_duration_perturb", False),
        "enable_worker_fatigue": getattr(configs, "enable_worker_fatigue", False),
        "randomize_durations": getattr(configs, "randomize_durations", False),
    }
    for key in backups:
        setattr(configs, key, False)

    makespans, balances, rewards, durations = [], [], [], []
    worker_utils, station_utils = [], []
    schedules = []
    constraint_rows = []
    score_rows = []
    scenario_path = ensure_reschedule_eval_scenarios_available(configs)
    scenario_items = [] if scenario_path is None else load_reschedule_scenarios(scenario_path)
    if num_runs is not None:
        scenario_items = scenario_items[: max(1, int(num_runs))]
    verbose_progress = bool(getattr(configs, "verbose_reschedule_eval_progress", False))
    if verbose_progress:
        print(
            f"[RescheduleEval] scenarios={len(scenario_items)} path={scenario_path} "
            f"temperature={float(temperature):.4g}",
            flush=True,
        )

    try:
        base_seed = int(getattr(configs, "reschedule_eval_scenario_seed", 42))
        for idx, (scenario_id, scenario) in enumerate(scenario_items):
            setattr(env, "_forced_reschedule_scenario", scenario)
            state = env.reset(randomize_duration=False, randomize_workers=False, seed=base_seed + idx)
            done = False
            total_reward = 0.0
            invalid_step_count = 0
            mismatch_recovery_count = 0
            blocked_tasks: set[int] = set()
            blocked_at_time: float | None = None
            mismatch_retries_at_time = 0
            mismatch_policy = str(getattr(configs, "eval_mask_mismatch_policy", "fail")).lower()
            max_mismatch_retries = int(
                getattr(configs, "eval_mask_mismatch_max_retries_per_time", 16)
            )
            start_wall = time.time()

            for _ in range(env.num_tasks * 3):
                if done:
                    break
                task_mask, station_mask, worker_mask = env.get_masks()
                current_time = float(env.current_time)
                if (
                    blocked_at_time is None
                    or abs(current_time - blocked_at_time) > release_time_tolerance(configs)
                ):
                    blocked_tasks.clear()
                    blocked_at_time = current_time
                    mismatch_retries_at_time = 0
                if blocked_tasks:
                    task_mask = task_mask.clone()
                    station_mask = station_mask.clone()
                    blocked = sorted(blocked_tasks)
                    task_mask[blocked] = True
                    station_mask[blocked, :] = True
                if task_mask.all():
                    if env.try_wait_for_resources():
                        state = refresh_env_observation(env)
                        continue
                    break

                action_ret = agent.select_action(
                    state.to(agent.device),
                    mask_task=task_mask.to(agent.device),
                    mask_station_matrix=station_mask.to(agent.device),
                    mask_worker=worker_mask.to(agent.device),
                    deterministic=(temperature == 0.0),
                    temperature=temperature,
                    is_eval=True,
                )
                if action_ret[0] is None:
                    break
                action, _, _, _, is_invalid = action_ret
                if getattr(configs, "ablation_no_mask", False) and is_invalid:
                    break
                state, reward, done, info = env.step(action)
                if info.get("invalid_action", False):
                    mismatch_message = _mask_mismatch_message(
                        env,
                        action=action,
                        info=info,
                        scenario_id=str(scenario_id),
                    )
                    recoverable = str(info.get("error", "")) == "task_release_time_not_reached"
                    if mismatch_policy == "recover" and recoverable:
                        mismatch_recovery_count += 1
                        mismatch_retries_at_time += 1
                        if mismatch_retries_at_time > max_mismatch_retries:
                            raise MaskEnvironmentMismatchError(
                                f"{mismatch_message} retries_at_time={mismatch_retries_at_time}"
                            )
                        blocked_tasks.add(int(action[0]))
                        state = refresh_env_observation(env)
                        continue
                    invalid_step_count += 1
                    raise MaskEnvironmentMismatchError(mismatch_message)
                total_reward += float(reward)

            elapsed = time.time() - start_wall
            complete = len(env.assigned_tasks) == env.num_tasks
            if complete:
                final_makespan = float(np.max(env.station_wall_clock))
                balance = float(np.std(env.station_loads))
                worker_util, station_util = _compute_assignment_utilization(env, final_makespan)
                schedule = list(env.assigned_tasks)
            else:
                final_makespan = float(env.ideal_makespan * 3.0)
                balance = float(env.ideal_station_load * 3.0)
                worker_util, station_util = 0.0, 0.0
                schedule = []

            constraints = _compute_reschedule_constraint_metrics(env)
            constraints["scenario_id"] = scenario_id
            constraints["scenario_index"] = float(idx)
            constraints["reschedule_start_time"] = float(scenario.start_time)
            constraints["delayed_task_count"] = float(len(scenario.task_release_times))
            constraints["invalid_step_count"] = float(invalid_step_count)
            constraints["mask_mismatch_recovery_count"] = float(mismatch_recovery_count)
            constraints["mask_mismatch_recovered"] = float(mismatch_recovery_count > 0)
            constraints["release_time_tolerance_hours"] = float(
                release_time_tolerance(configs)
            )
            constraints["complete"] = float(complete)
            score_result = calculate_reschedule_composite_score(
                makespan=final_makespan,
                balance_std=balance,
                constraint_metrics=constraints,
                config_obj=configs,
                ideal_station_load=float(getattr(env, "ideal_station_load", 1.0)),
            )
            constraints["eligible"] = float(score_result.eligible)
            constraints["composite_score"] = float(score_result.score)
            constraints["selection_score"] = float(score_result.selection_score)
            constraints.update(score_result.terms)
            constraint_rows.append(constraints)
            score_rows.append(score_result)
            makespans.append(final_makespan)
            balances.append(balance)
            rewards.append(total_reward)
            durations.append(elapsed)
            worker_utils.append(worker_util)
            station_utils.append(station_util)
            schedules.append(schedule)
            if verbose_progress:
                print(
                    f"[RescheduleEval] {idx + 1}/{len(scenario_items)} {scenario_id} "
                    f"mk={final_makespan:.2f} score={float(score_result.score):.4f} "
                    f"elig={int(score_result.eligible)} complete={int(complete)} "
                    f"dur={elapsed:.2f}s",
                    flush=True,
                )
    finally:
        if hasattr(env, "_forced_reschedule_scenario"):
            delattr(env, "_forced_reschedule_scenario")
        for key, value in backups.items():
            setattr(configs, key, value)
        if was_training:
            agent.policy.train()

    if score_rows:
        best_idx = int(np.argmin([score.selection_score for score in score_rows]))
    else:
        best_idx = int(np.argmin(makespans)) if makespans else 0
    metric_keys = [
        "frozen_violation_count",
        "release_violation_count",
        "precedence_violation_count",
        "worker_overlap_violation_count",
        "station_slot_violation_count",
        "skill_violation_count",
        "demand_violation_count",
        "fixed_station_violation_count",
        "station_range_violation_count",
        "physical_station_violation_count",
        "worker_station_binding_violation_count",
        "duplicate_task_count",
        "missing_task_count",
        "invalid_step_count",
        "mask_mismatch_recovery_count",
        "mask_mismatch_recovered",
        "release_time_tolerance_hours",
        "complete",
        "reschedule_start_time",
        "delayed_task_count",
        "takt_h",
        "takt_violation_h",
        "lower_bound_h",
        "takt_feasible",
        "start_deviation_mean_h",
        "station_change_rate",
        "team_change_rate",
        "eligible",
        "composite_score",
        "selection_score",
        "score_makespan",
        "score_balance",
        "score_takt_violation",
        "score_start_stability",
        "score_station_change",
        "score_team_change",
    ]
    avg_metrics = {
        key: float(np.mean([row[key] for row in constraint_rows])) if constraint_rows else 0.0
        for key in metric_keys
    }
    avg_metrics["eligible_rate"] = avg_metrics.get("eligible", 0.0)

    avg_makespan = float(np.mean(makespans)) if makespans else 0.0
    avg_balance = float(np.mean(balances)) if balances else 0.0
    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    avg_duration = float(np.mean(durations)) if durations else 0.0
    avg_w_util = float(np.mean(worker_utils)) if worker_utils else 0.0
    avg_s_util = float(np.mean(station_utils)) if station_utils else 0.0
    best_sch = schedules[best_idx] if schedules else []

    if writer is not None:
        writer.add_scalar("RescheduleEval/Makespan", avg_makespan, current_ep)
        writer.add_scalar("RescheduleEval/Reward", avg_reward, current_ep)
        writer.add_scalar("RescheduleEval/WorkerUtil", avg_w_util, current_ep)
        writer.add_scalar("RescheduleEval/StationUtil", avg_s_util, current_ep)
        writer.add_scalar("RescheduleEval/CompositeScore", avg_metrics.get("composite_score", 0.0), current_ep)
        writer.add_scalar("RescheduleEval/EligibleRate", avg_metrics.get("eligible_rate", 0.0), current_ep)
        for key, value in avg_metrics.items():
            writer.add_scalar(f"RescheduleEval/{key}", value, current_ep)

    evaluate_reschedule_model.last_metrics = avg_metrics
    evaluate_reschedule_model.last_scenario_metrics = constraint_rows
    return avg_makespan, avg_balance, avg_reward, best_sch, avg_duration, avg_w_util, avg_s_util


__all__ = [
    "MaskEnvironmentMismatchError",
    "_compute_assignment_utilization",
    "_compute_reschedule_constraint_metrics",
    "ensure_reschedule_baseline_available",
    "ensure_reschedule_eval_scenarios_available",
    "evaluate_reschedule_model",
    "load_warm_start_weights_with_input_expansion",
]
