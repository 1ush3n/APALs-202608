# -*- coding: utf-8 -*-
"""验证 APAL 预测-反应式重调度的 baseline、冻结和工序延迟硬约束。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import Config, configs, load_config_files
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from tests.runtime_safety import temporary_config
from runtime.reschedule_eval import evaluate_reschedule_model
from utils.gpu_graph_manager import GPUBatchGraphManager
from utils.vector_env import EnvCreator, VectorEnv
from utils.reschedule import load_baseline_schedule, load_reschedule_scenarios
from baselines.heuristic.reschedule_rules import BeamSearchRepairRule


def _base_overrides() -> dict[str, object]:
    return {
        "enable_dynamic_events": False,
        "enable_station_breakdown": False,
        "enable_material_delay": False,
        "enable_online_duration_perturb": False,
        "enable_worker_fatigue": False,
        "enable_reschedule_mode": False,
        "randomize_durations": False,
        "task_feat_dim": 18,
        "n_w": 80,
    }


def _first_valid_action(env: AirLineEnv_Graph) -> tuple[int, int, list[int]]:
    task_mask, station_mask, worker_mask = env.get_masks()
    valid_tasks = torch.where(~task_mask)[0]
    assert len(valid_tasks) > 0
    for task_tensor in valid_tasks:
        task_id = int(task_tensor.item())
        valid_stations = torch.where(~station_mask[task_id])[0]
        if len(valid_stations) == 0:
            continue
        station_id = int(valid_stations[0].item())
        skill_id = int(env.task_static_feat[task_id, 1].item())
        demand = max(1, int(env.task_static_feat[task_id, 2].item()))
        candidates = [
            int(w)
            for w in np.where(env.worker_skill_matrix[:, skill_id].numpy() > 0.5)[0]
            if not bool(worker_mask[w])
            and int(env.worker_locks[w]) in {0, station_id + 1}
        ]
        if len(candidates) >= demand:
            return task_id, station_id, candidates[:demand]
    raise AssertionError("没有找到可执行动作")


def _write_greedy_baseline(path: Path) -> pd.DataFrame:
    with temporary_config(configs, _base_overrides()):
        env = AirLineEnv_Graph(data_path_or_dir=str(PROJECT_ROOT / "data" / "283.csv"), seed=7)
        env.reset(randomize_duration=False, randomize_workers=False, seed=7)
        for _ in range(env.num_tasks * 3):
            if len(env.assigned_tasks) == env.num_tasks:
                break
            masks = env.get_masks()
            if masks[0].all():
                assert env.try_wait_for_resources()
                continue
            _obs, _reward, _done, info = env.step(_first_valid_action(env))
            assert "error" not in info
        assert len(env.assigned_tasks) == env.num_tasks

        rows = [
            {
                "TaskID": int(task_id),
                "StationID": int(station_id) + 1,
                "Team": str([int(w) for w in team]),
                "Start": float(start),
                "End": float(end),
                "Duration": float(end - start),
            }
            for task_id, station_id, team, start, end in env.assigned_tasks
        ]
        df = pd.DataFrame(rows).sort_values("Start")
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return df


def _reschedule_overrides(baseline_path: Path, scenario_path: Path) -> dict[str, object]:
    cfg = Config()
    load_config_files([str(PROJECT_ROOT / "conf" / "experiment" / "reschedule_task_delay.yaml")], target=cfg)
    data = cfg.to_flat_dict()
    data.update(
        {
            "reschedule_baseline_schedule_path": str(baseline_path),
            "reschedule_scenario_path": str(scenario_path),
            "reschedule_eval_scenario_path": str(scenario_path),
            "data_file_path": str(PROJECT_ROOT / "data" / "283.csv"),
            "train_data_path_or_dir": str(PROJECT_ROOT / "data" / "283.csv"),
            "num_envs_windows": 1,
            "num_envs_linux": 1,
        }
    )
    return data


def test_reschedule_config_loads_and_isolates_experiment() -> None:
    cfg = Config()
    load_config_files([str(PROJECT_ROOT / "conf" / "experiment" / "reschedule_task_delay.yaml")], target=cfg)

    assert cfg.experiment_name == "reschedule_task_delay"
    assert cfg.enable_reschedule_mode is True
    assert cfg.task_feat_dim == 24
    assert cfg.enable_dynamic_events is False
    assert cfg.enable_material_delay is False
    assert cfg.reschedule_warm_start is True


def test_baseline_loader_computes_takt_from_schedule(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.csv"
    df = _write_greedy_baseline(baseline_path)

    baseline = load_baseline_schedule(baseline_path)
    assert baseline.makespan == float(df["End"].max())
    assert len(baseline.tasks) == len(df)


def test_reschedule_reset_freezes_started_tasks_and_adds_features(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.csv"
    df = _write_greedy_baseline(baseline_path)
    start_time = float(df["Start"].quantile(0.35))
    delayed_row = df[df["Start"] > start_time].iloc[0]
    delayed_task = int(delayed_row["TaskID"])
    release_time = float(delayed_row["Start"] + 10.0)
    scenario_path = tmp_path / "scenario.csv"
    pd.DataFrame(
        [{"reschedule_start_time": start_time, "TaskID": delayed_task, "release_time": release_time}]
    ).to_csv(scenario_path, index=False)

    with temporary_config(configs, _reschedule_overrides(baseline_path, scenario_path)):
        env = AirLineEnv_Graph(data_path_or_dir=str(PROJECT_ROOT / "data" / "283.csv"), seed=11)
        obs = env.reset(randomize_duration=False, randomize_workers=False, seed=11)

        assert obs["task"].x.size(1) == configs.task_feat_dim
        assert obs["task"].x.size(1) >= 23
        assert env.baseline_schedule is not None
        assert abs(env.baseline_schedule.makespan - float(df["End"].max())) < 1e-6
        frozen_ids = {int(row.TaskID) for row in df.itertuples() if float(row.Start) <= start_time + 1e-9}
        assigned_ids = {int(item[0]) for item in env.assigned_tasks}
        assert frozen_ids <= assigned_ids
        assert obs["task"].x[list(frozen_ids), 21].min().item() == 1.0
        assert env.task_material_ready[delayed_task] == release_time
        assert obs["task"].x[delayed_task, 22].item() == 1.0


def test_reschedule_rule_static_cache_matches_environment_queries(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.csv"
    df = _write_greedy_baseline(baseline_path)
    start_time = float(df["Start"].quantile(0.35))
    delayed_row = df[df["Start"] > start_time].iloc[0]
    scenario_path = tmp_path / "scenario.csv"
    pd.DataFrame(
        [
            {
                "scenario_id": "low_000",
                "reschedule_start_time": start_time,
                "TaskID": int(delayed_row["TaskID"]),
                "release_time": float(delayed_row["Start"] + 10.0),
            }
        ]
    ).to_csv(scenario_path, index=False)
    scenario_id, scenario = load_reschedule_scenarios(scenario_path)[0]

    with temporary_config(configs, _reschedule_overrides(baseline_path, scenario_path)):
        solver = BeamSearchRepairRule(
            data_path_or_dir=PROJECT_ROOT / "data" / "283.csv",
            scenario=scenario,
            scenario_id=scenario_id,
            scenario_level="low",
            seed=21,
            verify_static_cache=True,
            beam_width=1,
            beam_branch_factor=1,
            beam_levels=1,
        )
        ctx = solver.static_context
        assert ctx is not None
        for task_id in range(min(20, solver.env.num_tasks)):
            assert solver._task_duration(task_id) == float(solver.env.task_static_feat[task_id, 0].item())
            assert solver._task_skill(task_id) == int(solver.env.task_static_feat[task_id, 1].item())
            assert solver._task_demand(task_id) == max(1, int(solver.env.task_static_feat[task_id, 2].item()))

        task_id = int(np.where(solver.env.task_status <= 1)[0][0])
        skill_id = solver._task_skill(task_id)
        worker_mask = np.zeros(solver.env.num_workers, dtype=bool)
        station_id = 0
        expected = [
            int(worker_id)
            for worker_id in range(solver.env.num_workers)
            if solver.env.worker_skill_matrix[int(worker_id), skill_id] > 0.5
            and int(solver.env.worker_locks[int(worker_id)]) in {0, station_id + 1}
        ]
        assert solver._valid_workers(task_id, station_id, worker_mask) == expected

        if expected:
            worker_mask[expected[0]] = True
            assert expected[0] not in solver._valid_workers(task_id, station_id, worker_mask)
            worker_mask[expected[0]] = False
            solver.env.worker_locks[expected[0]] = station_id + 2
            assert expected[0] not in solver._valid_workers(task_id, station_id, worker_mask)


def test_delayed_zero_duration_task_is_not_auto_completed_before_release(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.csv"
    df = _write_greedy_baseline(baseline_path)
    zero_rows = df[(df["Duration"].abs() <= 1e-8) & (df["Start"] > 1.0)]
    assert not zero_rows.empty
    delayed_row = zero_rows.iloc[0]
    delayed_task = int(delayed_row["TaskID"])
    start_time = max(0.0, float(delayed_row["Start"]) - 0.5)
    release_time = float(delayed_row["Start"] + 20.0)
    scenario_path = tmp_path / "zero_duration_delay.csv"
    pd.DataFrame(
        [{"reschedule_start_time": start_time, "TaskID": delayed_task, "release_time": release_time}]
    ).to_csv(scenario_path, index=False)

    with temporary_config(configs, _reschedule_overrides(baseline_path, scenario_path)):
        env = AirLineEnv_Graph(data_path_or_dir=str(PROJECT_ROOT / "data" / "283.csv"), seed=12)
        env.reset(randomize_duration=False, randomize_workers=False, seed=12)

        assert env.task_material_ready[delayed_task] == release_time
        assigned = [item for item in env.assigned_tasks if int(item[0]) == delayed_task]
        if assigned:
            assert float(assigned[0][3]) + 1e-8 >= release_time
            assert float(assigned[0][4]) + 1e-8 >= release_time
        else:
            assert int(env.task_status[delayed_task]) != 2


def test_release_time_is_environment_hard_constraint(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.csv"
    df = _write_greedy_baseline(baseline_path)
    start_time = float(df["Start"].quantile(0.25))
    delayed_row = df[df["Start"] > start_time].iloc[0]
    delayed_task = int(delayed_row["TaskID"])
    release_time = float(max(delayed_row["Start"] + 20.0, start_time + 20.0))
    scenario_path = tmp_path / "scenario.csv"
    pd.DataFrame(
        [{"reschedule_start_time": start_time, "TaskID": delayed_task, "release_time": release_time}]
    ).to_csv(scenario_path, index=False)

    with temporary_config(configs, _reschedule_overrides(baseline_path, scenario_path)):
        env = AirLineEnv_Graph(data_path_or_dir=str(PROJECT_ROOT / "data" / "283.csv"), seed=13)
        env.reset(randomize_duration=False, randomize_workers=False, seed=13)
        env.task_status[delayed_task] = 1
        env.current_time = min(env.current_time, release_time - 1.0)
        skill_id = int(env.task_static_feat[delayed_task, 1].item())
        demand = max(1, int(env.task_static_feat[delayed_task, 2].item()))
        team = [int(w) for w in np.where(env.worker_skill_matrix[:, skill_id].numpy() > 0.5)[0]][:demand]
        _obs, _reward, _done, info = env.step((delayed_task, 0, team))

        assert info["invalid_action"] is True
        assert info["error"] == "task_release_time_not_reached"


def test_vector_env_child_process_receives_reschedule_config(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.csv"
    df = _write_greedy_baseline(baseline_path)
    start_time = float(df["Start"].quantile(0.30))
    delayed_row = df[df["Start"] > start_time].iloc[0]
    scenario_path = tmp_path / "scenario.csv"
    pd.DataFrame(
        [
            {
                "reschedule_start_time": start_time,
                "TaskID": int(delayed_row["TaskID"]),
                "release_time": float(delayed_row["Start"] + 8.0),
            }
        ]
    ).to_csv(scenario_path, index=False)

    with temporary_config(configs, _reschedule_overrides(baseline_path, scenario_path)):
        make_env = EnvCreator(str(PROJECT_ROOT / "data" / "283.csv"), seed_offset=100)
        vec_env = VectorEnv(make_env, num_envs=2, start_method="spawn")
        try:
            states = vec_env.reset_all(randomize_duration=False, randomize_workers=False)
            masks, snapshots = vec_env.get_masks_and_snapshots_all()
            assert len(states) == 2
            assert len(masks) == 2
            assert states[0]["task"].x.size(1) == configs.task_feat_dim
            assert states[0]["task"].x.size(1) >= 23
            assert "baseline_start" in snapshots[0]
            rebuilt = vec_env.envs[0].rebuild_state_from_snapshot(snapshots[0])
            assert rebuilt["task"].x.size(1) == configs.task_feat_dim
            assert rebuilt["task"].x.size(1) >= 23
            assert snapshots[0]["baseline_makespan"] == float(df["End"].max())
        finally:
            vec_env.close()


def test_gpu_batch_rebuild_preserves_reschedule_features(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.csv"
    df = _write_greedy_baseline(baseline_path)
    start_time = float(df["Start"].quantile(0.30))
    delayed_row = df[df["Start"] > start_time].iloc[0]
    scenario_path = tmp_path / "scenario.csv"
    pd.DataFrame(
        [
            {
                "reschedule_start_time": start_time,
                "TaskID": int(delayed_row["TaskID"]),
                "release_time": float(delayed_row["Start"] + 8.0),
            }
        ]
    ).to_csv(scenario_path, index=False)

    with temporary_config(configs, _reschedule_overrides(baseline_path, scenario_path)):
        env = AirLineEnv_Graph(data_path_or_dir=str(PROJECT_ROOT / "data" / "283.csv"), seed=17)
        env.reset(randomize_duration=False, randomize_workers=False, seed=17)
        snapshots = [env.get_state_snapshot(), env.get_state_snapshot()]
        batch = GPUBatchGraphManager(torch.device("cpu")).batched_rebuild_on_gpu(snapshots, env)

        assert batch["task"].x.size(1) == configs.task_feat_dim
        assert batch["task"].x.size(1) >= 23
        assert batch["task"].x[:, 21].max().item() == 1.0
        assert batch["task"].x[:, 22].max().item() == 1.0


def test_evaluate_reschedule_model_reports_constraint_metrics(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.csv"
    df = _write_greedy_baseline(baseline_path)
    start_time = float(df["Start"].quantile(0.30))
    delayed_row = df[df["Start"] > start_time].iloc[0]
    scenario_path = tmp_path / "scenario.csv"
    pd.DataFrame(
        [
            {
                "reschedule_start_time": start_time,
                "TaskID": int(delayed_row["TaskID"]),
                "release_time": float(delayed_row["Start"] + 6.0),
            }
        ]
    ).to_csv(scenario_path, index=False)

    overrides = _reschedule_overrides(baseline_path, scenario_path)
    overrides.update(
        {
            "hidden_dim": 32,
            "num_gat_layers": 1,
            "num_heads": 2,
            "batch_size": 4,
            "accumulation_steps": 1,
            "use_schedule_free": False,
        }
    )
    with temporary_config(configs, overrides):
        device = torch.device("cpu")
        env = AirLineEnv_Graph(data_path_or_dir=str(PROJECT_ROOT / "data" / "283.csv"), seed=23)
        agent = PPOAgent(
            HBGATPN(configs).to(device),
            configs.lr,
            configs.gamma,
            configs.k_epochs,
            configs.eps_clip,
            device,
            batch_size=configs.batch_size,
            total_timesteps=1,
        )
        makespan, _balance, _reward, schedule, _duration, _w_util, _s_util = evaluate_reschedule_model(
            env,
            agent,
            num_runs=1,
            temperature=0.0,
        )
        metrics = getattr(evaluate_reschedule_model, "last_metrics")

        assert makespan > 0.0
        assert metrics["frozen_violation_count"] == 0.0
        assert metrics["release_violation_count"] == 0.0
        assert metrics["precedence_violation_count"] == 0.0
        assert metrics["worker_overlap_violation_count"] == 0.0
        assert metrics["station_slot_violation_count"] == 0.0
        assert metrics["skill_violation_count"] == 0.0
        assert metrics["demand_violation_count"] == 0.0
        assert metrics["duplicate_task_count"] == 0.0
        assert metrics["takt_h"] == float(df["End"].max())
        assert "composite_score" in metrics
        assert "eligible_rate" in metrics
        assert "score_makespan" in metrics
        assert "score_balance" in metrics
        assert "score_takt_violation" in metrics
        assert "score_start_stability" in metrics
        assert "score_station_change" in metrics
        assert "score_team_change" in metrics
        assert isinstance(schedule, list)


def test_reschedule_constraint_metrics_detect_precedence_violation(tmp_path: Path) -> None:
    from runtime.reschedule_eval import _compute_reschedule_constraint_metrics

    baseline_path = tmp_path / "baseline.csv"
    df = _write_greedy_baseline(baseline_path)
    start_time = float(df["Start"].quantile(0.10))
    scenario_path = tmp_path / "scenario.csv"
    pd.DataFrame([{"reschedule_start_time": start_time, "TaskID": int(df.iloc[-1]["TaskID"]), "release_time": start_time + 1.0}]).to_csv(
        scenario_path,
        index=False,
    )

    with temporary_config(configs, _reschedule_overrides(baseline_path, scenario_path)):
        env = AirLineEnv_Graph(data_path_or_dir=str(PROJECT_ROOT / "data" / "283.csv"), seed=29)
        env.reset(randomize_duration=False, randomize_workers=False, seed=29)
        edges = env.raw_data["precedence_edges"].detach().cpu().numpy()
        src, dst = int(edges[0, 0]), int(edges[1, 0])
        pred = env.baseline_schedule.tasks[src]
        succ = env.baseline_schedule.tasks[dst]
        env.assigned_tasks = [
            (src, pred.station_id, list(pred.team), 10.0, 20.0),
            (dst, succ.station_id, list(succ.team), 15.0, 25.0),
        ]

        metrics = _compute_reschedule_constraint_metrics(env)
        assert metrics["precedence_violation_count"] >= 1.0
