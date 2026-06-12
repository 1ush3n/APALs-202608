from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import configs, load_config_files
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from scripts.evaluate_reschedule_model import _load_policy_weights
from train import (
    _compute_assignment_utilization,
    _compute_reschedule_constraint_metrics,
    ensure_reschedule_baseline_available,
    ensure_reschedule_eval_scenarios_available,
    refresh_env_observation,
    resolve_workspace_path,
)
from utils.reschedule import (
    calculate_reschedule_composite_score,
    load_baseline_schedule,
    load_reschedule_scenarios,
)


def _parse_team(value: Any) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    text = str(value).strip()
    if not text:
        return ()
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = [part.strip() for part in text.strip("[]").split(",") if part.strip()]
    if isinstance(parsed, int):
        return (int(parsed),)
    return tuple(int(v) for v in parsed)


def _schedule_to_frame(schedule: list[tuple[int, int, list[int], float, float]]) -> pd.DataFrame:
    rows = []
    for task_id, station_id, team, start, end in schedule:
        start_f = float(start)
        end_f = float(end)
        rows.append(
            {
                "TaskID": int(task_id),
                "StationID": int(station_id) + 1 if int(station_id) >= 0 else 0,
                "Team": list(int(w) for w in team),
                "Start": start_f,
                "End": end_f,
                "Duration": max(0.0, end_f - start_f),
            }
        )
    return pd.DataFrame(rows).sort_values(["Start", "StationID", "TaskID"]).reset_index(drop=True)


def _baseline_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    df["TaskID"] = df["TaskID"].astype(int)
    df["StationID"] = df["StationID"].astype(int)
    df["Start"] = df["Start"].astype(float)
    df["End"] = df["End"].astype(float)
    df["Duration"] = df["Duration"].astype(float)
    return df


def _compare_schedules(
    baseline_df: pd.DataFrame,
    reschedule_df: pd.DataFrame,
    *,
    scenario_start: float,
    release_times: dict[int, float],
) -> pd.DataFrame:
    base = baseline_df.set_index("TaskID")
    res = reschedule_df.set_index("TaskID")
    rows = []
    for task_id in sorted(set(base.index) | set(res.index)):
        b = base.loc[task_id] if task_id in base.index else None
        r = res.loc[task_id] if task_id in res.index else None
        b_team = _parse_team(b["Team"]) if b is not None else ()
        r_team = _parse_team(r["Team"]) if r is not None else ()
        b_start = float(b["Start"]) if b is not None else np.nan
        r_start = float(r["Start"]) if r is not None else np.nan
        b_station = int(b["StationID"]) if b is not None else -999
        r_station = int(r["StationID"]) if r is not None else -999
        rows.append(
            {
                "TaskID": int(task_id),
                "baseline_station": b_station,
                "reschedule_station": r_station,
                "baseline_team": list(b_team),
                "reschedule_team": list(r_team),
                "baseline_start": b_start,
                "reschedule_start": r_start,
                "baseline_end": float(b["End"]) if b is not None else np.nan,
                "reschedule_end": float(r["End"]) if r is not None else np.nan,
                "start_deviation_h": float(r_start - b_start) if np.isfinite(b_start) and np.isfinite(r_start) else np.nan,
                "abs_start_deviation_h": float(abs(r_start - b_start)) if np.isfinite(b_start) and np.isfinite(r_start) else np.nan,
                "station_changed": bool(b_station != r_station) if b is not None and r is not None else True,
                "team_changed": bool(set(b_team) != set(r_team)) if b is not None and r is not None else True,
                "frozen": bool(np.isfinite(b_start) and b_start <= scenario_start + 1e-9),
                "delayed_release": int(task_id) in release_times,
                "release_time": float(release_times.get(int(task_id), b_start if np.isfinite(b_start) else scenario_start)),
                "missing_in_reschedule": r is None,
            }
        )
    return pd.DataFrame(rows)


def _draw_schedule_axis(
    ax,
    df: pd.DataFrame,
    *,
    title: str,
    diff_df: pd.DataFrame | None,
    is_reschedule: bool,
    scenario_start: float,
    takt: float,
    makespan: float,
) -> None:
    colors = plt.cm.tab20.colors
    for row in df.itertuples(index=False):
        duration = float(row.Duration)
        if duration <= 1e-9:
            continue
        task_id = int(row.TaskID)
        station_id = int(row.StationID)
        if station_id <= 0:
            continue
        color = colors[task_id % len(colors)]
        edgecolor = "none"
        linewidth = 0.0
        alpha = 0.72
        hatch = None
        if diff_df is not None and task_id in set(diff_df["TaskID"]):
            d = diff_df.loc[diff_df["TaskID"] == task_id].iloc[0]
            if bool(d["delayed_release"]):
                edgecolor = "#2563eb"
                linewidth = 1.0
            if is_reschedule and bool(d["station_changed"]):
                edgecolor = "#dc2626"
                linewidth = 1.2
            elif is_reschedule and bool(d["team_changed"]):
                edgecolor = "#f97316"
                linewidth = 1.0
            if bool(d["frozen"]):
                hatch = "//"
                alpha = 0.55
        ax.barh(
            station_id,
            duration,
            left=float(row.Start),
            height=0.56,
            color=color,
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=linewidth,
            hatch=hatch,
            align="center",
        )
    ax.axvline(scenario_start, color="#2563eb", linestyle="--", linewidth=1.6, label="reschedule start")
    ax.axvline(takt, color="#111827", linestyle=":", linewidth=1.4, label="baseline takt")
    ax.axvline(makespan, color="#dc2626", linestyle="-.", linewidth=1.2, label="makespan")
    ax.set_title(title)
    ax.set_ylabel("Station")
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)


def _plot_comparison(
    *,
    baseline_df: pd.DataFrame,
    reschedule_df: pd.DataFrame,
    diff_df: pd.DataFrame,
    scenario_id: str,
    scenario_start: float,
    takt: float,
    baseline_makespan: float,
    reschedule_makespan: float,
    metrics: dict[str, float],
    output_path: Path,
) -> None:
    max_station = int(max(baseline_df["StationID"].max(), reschedule_df["StationID"].max(), 1))
    x_max = max(float(baseline_df["End"].max()), float(reschedule_df["End"].max()), takt) * 1.03

    fig, axes = plt.subplots(2, 1, figsize=(18, 10), dpi=120, sharex=True)
    _draw_schedule_axis(
        axes[0],
        baseline_df,
        title=f"Baseline schedule | makespan={baseline_makespan:.2f}h",
        diff_df=diff_df,
        is_reschedule=False,
        scenario_start=scenario_start,
        takt=takt,
        makespan=baseline_makespan,
    )
    _draw_schedule_axis(
        axes[1],
        reschedule_df,
        title=f"Rescheduled schedule ({scenario_id}) | makespan={reschedule_makespan:.2f}h",
        diff_df=diff_df,
        is_reschedule=True,
        scenario_start=scenario_start,
        takt=takt,
        makespan=reschedule_makespan,
    )

    for ax in axes:
        ax.set_xlim(0, x_max)
        ax.set_yticks(range(1, max_station + 1))
        ax.set_yticklabels([f"S{i}" for i in range(1, max_station + 1)])

    axes[1].set_xlabel("Time (h)")
    legend_items = [
        mpatches.Patch(facecolor="white", edgecolor="#2563eb", label="delayed/release affected"),
        mpatches.Patch(facecolor="white", edgecolor="#dc2626", label="station changed"),
        mpatches.Patch(facecolor="white", edgecolor="#f97316", label="team changed"),
        mpatches.Patch(facecolor="white", edgecolor="#6b7280", hatch="//", label="frozen task"),
    ]
    axes[0].legend(handles=legend_items, loc="upper right", ncol=2, fontsize=9)

    score = float(metrics.get("composite_score", np.nan))
    subtitle = (
        f"score={score:.4f} | start={scenario_start:.2f}h | takt={takt:.2f}h | "
        f"takt violation={metrics.get('takt_violation_h', 0.0):.2f}h | "
        f"start dev={metrics.get('start_deviation_mean_h', 0.0):.2f}h | "
        f"station change={metrics.get('station_change_rate', 0.0):.3f} | "
        f"team change={metrics.get('team_change_rate', 0.0):.3f}"
    )
    fig.suptitle(f"APAL Baseline vs Reschedule Comparison\n{subtitle}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _run_reschedule_scenario(
    *,
    env: AirLineEnv_Graph,
    agent: PPOAgent,
    scenario_id: str,
    scenario,
    seed: int,
    temperature: float,
) -> tuple[pd.DataFrame, dict[str, float], float, float, float]:
    setattr(env, "_forced_reschedule_scenario", scenario)
    state = env.reset(randomize_duration=False, randomize_workers=False, seed=seed)
    done = False
    total_reward = 0.0
    invalid_step_count = 0
    start_wall = time.time()

    try:
        for _ in range(env.num_tasks * 3):
            if done:
                break
            task_mask, station_mask, worker_mask = env.get_masks()
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
            total_reward += float(reward)
            if info.get("invalid_action", False):
                invalid_step_count += 1
                break
    finally:
        if hasattr(env, "_forced_reschedule_scenario"):
            delattr(env, "_forced_reschedule_scenario")

    elapsed = time.time() - start_wall
    complete = len(env.assigned_tasks) == env.num_tasks
    if complete:
        final_makespan = float(np.max(env.station_wall_clock))
        balance = float(np.std(env.station_loads))
        worker_util, station_util = _compute_assignment_utilization(env, final_makespan)
    else:
        final_makespan = float(env.ideal_makespan * 3.0)
        balance = float(env.ideal_station_load * 3.0)
        worker_util, station_util = 0.0, 0.0

    constraints = _compute_reschedule_constraint_metrics(env)
    constraints["scenario_id"] = scenario_id
    constraints["reschedule_start_time"] = float(scenario.start_time)
    constraints["delayed_task_count"] = float(len(scenario.task_release_times))
    constraints["invalid_step_count"] = float(invalid_step_count)
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
    constraints["reward"] = float(total_reward)
    constraints["duration_sec"] = float(elapsed)
    constraints["worker_util"] = float(worker_util)
    constraints["station_util"] = float(station_util)
    constraints.update(score_result.terms)
    return _schedule_to_frame(list(env.assigned_tasks)), constraints, final_makespan, balance, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="可视化 APAL baseline 与 PPO 重调度结果的偏差")
    parser.add_argument("--config", type=str, default="conf/experiment/reschedule_task_delay.yaml")
    parser.add_argument("--model_path", type=str, default="checkpoints/reschedule_task_delay/bestmodel/best_model.pth")
    parser.add_argument("--baseline", type=str, default=None)
    parser.add_argument("--scenario_path", type=str, default=None)
    parser.add_argument("--scenario_id", type=str, default=None)
    parser.add_argument("--num_scenarios", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="results/reschedule_visual")
    args = parser.parse_args()

    load_config_files([str(resolve_workspace_path(args.config))])
    baseline_path = resolve_workspace_path(args.baseline) if args.baseline else ensure_reschedule_baseline_available(configs)
    scenario_path = resolve_workspace_path(args.scenario_path) if args.scenario_path else ensure_reschedule_eval_scenarios_available(configs)
    if baseline_path is None or scenario_path is None:
        raise RuntimeError("需要可用的 baseline CSV 和固定重调度 scenario CSV")

    baseline_df = _baseline_frame(Path(baseline_path))
    baseline = load_baseline_schedule(Path(baseline_path))
    scenario_items = load_reschedule_scenarios(Path(scenario_path))
    if args.scenario_id:
        scenario_items = [item for item in scenario_items if item[0] == args.scenario_id]
        if not scenario_items:
            raise ValueError(f"找不到 scenario_id={args.scenario_id}")
    else:
        scenario_items = scenario_items[: max(1, int(args.num_scenarios))]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = AirLineEnv_Graph(data_path_or_dir=str(resolve_workspace_path(configs.data_file_path)), seed=int(configs.seed))
    model = HBGATPN(configs).to(device)
    load_stats = _load_policy_weights(model, resolve_workspace_path(args.model_path), device)
    agent = PPOAgent(
        model,
        configs.lr,
        configs.gamma,
        configs.k_epochs,
        configs.eps_clip,
        device,
        batch_size=configs.batch_size,
        total_timesteps=1,
    )
    agent.policy.eval()

    output_dir = resolve_workspace_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    base_seed = int(getattr(configs, "reschedule_eval_scenario_seed", 30300))
    for idx, (scenario_id, scenario) in enumerate(scenario_items):
        reschedule_df, metrics, makespan, balance, elapsed = _run_reschedule_scenario(
            env=env,
            agent=agent,
            scenario_id=scenario_id,
            scenario=scenario,
            seed=base_seed + idx,
            temperature=float(args.temperature),
        )
        diff_df = _compare_schedules(
            baseline_df,
            reschedule_df,
            scenario_start=float(scenario.start_time),
            release_times=scenario.task_release_times,
        )
        schedule_csv = output_dir / f"{scenario_id}_reschedule_schedule.csv"
        diff_csv = output_dir / f"{scenario_id}_diff.csv"
        image_path = output_dir / f"{scenario_id}_baseline_vs_reschedule.png"
        reschedule_df.to_csv(schedule_csv, index=False)
        diff_df.to_csv(diff_csv, index=False)
        _plot_comparison(
            baseline_df=baseline_df,
            reschedule_df=reschedule_df,
            diff_df=diff_df,
            scenario_id=scenario_id,
            scenario_start=float(scenario.start_time),
            takt=float(baseline.makespan),
            baseline_makespan=float(baseline.makespan),
            reschedule_makespan=float(makespan),
            metrics=metrics,
            output_path=image_path,
        )
        row = {
            "scenario_id": scenario_id,
            "image_path": str(image_path),
            "schedule_csv": str(schedule_csv),
            "diff_csv": str(diff_csv),
            "makespan": float(makespan),
            "balance_std": float(balance),
            "duration_sec": float(elapsed),
            "load_mode": load_stats.get("mode", ""),
        }
        row.update(metrics)
        summary_rows.append(row)
        print(
            f"[{scenario_id}] image={image_path} score={metrics.get('composite_score', 0.0):.4f} "
            f"makespan={makespan:.2f} eligible={int(metrics.get('eligible', 0.0))}"
        )

    pd.DataFrame(summary_rows).to_csv(output_dir / "comparison_summary.csv", index=False)
    with (output_dir / "comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)
    print(f"重调度对比可视化已保存到: {output_dir}")


if __name__ == "__main__":
    main()
