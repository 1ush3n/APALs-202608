from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from training.async_evaluation import (
    AsyncEvalPaths,
    atomic_write_json,
    process_is_alive,
    restore_interrupted_jobs,
)
from runtime.initial_checkpoint_selection import sha256_file


R5_RESCHEDULE_ASYNC_PROTOCOL = "r5_task_delay_v1"


class _Heartbeat:
    def __init__(self, path: Path, interval_sec: float) -> None:
        self.path = path
        self.interval_sec = max(1.0, float(interval_sec))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="async-eval-heartbeat", daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_sec):
            atomic_write_json(self.path, {"pid": os.getpid(), "updated_at": time.time()})

    def __enter__(self) -> "_Heartbeat":
        atomic_write_json(self.path, {"pid": os.getpid(), "updated_at": time.time()})
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback_obj) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval_sec + 1.0)
        atomic_write_json(self.path, {"pid": os.getpid(), "updated_at": time.time()})


class _ExclusivePidLock:
    """跨平台 PID 文件锁；仅用于短暂的 worker/最佳模型发布临界区。"""

    def __init__(self, path: Path, *, wait_timeout_sec: float = 120.0) -> None:
        self.path = path
        self.wait_timeout_sec = float(wait_timeout_sec)
        self._acquired = False

    def __enter__(self) -> "_ExclusivePidLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.wait_timeout_sec
        while time.monotonic() < deadline:
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    existing_pid = int(self.path.read_text(encoding="ascii").strip())
                except (OSError, ValueError):
                    existing_pid = -1
                if process_is_alive(existing_pid):
                    time.sleep(0.1)
                    continue
                self.path.unlink(missing_ok=True)
                continue
            with os.fdopen(descriptor, "w", encoding="ascii") as handle:
                handle.write(str(os.getpid()))
            self._acquired = True
            return self
        raise RuntimeError(f"获取异步验证锁超时: {self.path}")

    def __exit__(self, exc_type, exc_value, traceback_obj) -> None:
        if self._acquired:
            self.path.unlink(missing_ok=True)


class _WorkerLock(_ExclusivePidLock):
    """每个 worker 使用独立锁，允许同一队列并行消费。"""


# 向后兼容已有单元测试导入名称。
_restore_interrupted_jobs = restore_interrupted_jobs


def _claim_next_job(paths: AsyncEvalPaths) -> tuple[Path, dict[str, Any]] | None:
    for pending_path in sorted(paths.pending.glob("*.json")):
        running_path = paths.running / pending_path.name
        try:
            pending_path.replace(running_path)
        except FileNotFoundError:
            continue
        try:
            payload = json.loads(
                running_path.read_text(encoding="utf-8-sig")
            )
            return running_path, payload
        except OSError:
            # Windows 文件锁竞态：任务已移入 running 但读取被拒（可能正被
            # 其他 worker 的 done/retry/failed 清理路径删除）。原子移回
            # pending 等待后续轮次重试，避免 worker 因该异常整体退出。
            try:
                running_path.replace(pending_path)
            except OSError:
                pass
            continue
    return None


def _write_summary_csv(paths: AsyncEvalPaths) -> None:
    rows: list[dict[str, Any]] = []
    for result_path in sorted(paths.results.glob("episode_*.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8-sig"))
        row = {
            key: value
            for key, value in payload.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        if "instances" in payload:
            row["instances_json"] = json.dumps(payload["instances"], ensure_ascii=False)
        rows.append(row)
    if not rows:
        return
    preferred = [
        "episode",
        "evaluation_kind",
        "instance_id",
        "scenario_id",
        "eligible",
        "selection_score",
        "composite_score",
        "makespan",
        "duration_sec",
        "candidate_path",
    ]
    extra = sorted({key for row in rows for key in row}.difference(preferred))
    fieldnames = [key for key in preferred if any(key in row for row in rows)] + extra
    destination = paths.results / "async_eval_summary.csv"
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)


def _write_schedule(path: Path, schedule: list[Any]) -> None:
    """保留旧单实例异步验证的历史 CSV 格式。"""
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task_id", "station_id", "worker_ids", "start_time", "finish_time"])
        for task_id, station_id, workers, start_time, finish_time in schedule:
            writer.writerow(
                [int(task_id), int(station_id), ";".join(str(int(worker_id)) for worker_id in workers), float(start_time), float(finish_time)]
            )
    temporary.replace(path)


def _write_canonical_initial_schedule(path: Path, schedule: list[Any]) -> None:
    """输出严格审计器要求的初始调度列名与站位编号口径。"""
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["TaskID", "StationID", "Team", "Start", "End", "Duration"],
        )
        writer.writeheader()
        for task_id, station_id, workers, start_time, finish_time in sorted(
            schedule, key=lambda row: (float(row[3]), int(row[0]))
        ):
            writer.writerow(
                {
                    "TaskID": int(task_id),
                    "StationID": int(station_id) + 1,
                    "Team": str([int(worker_id) for worker_id in workers]),
                    "Start": float(start_time),
                    "End": float(finish_time),
                    "Duration": float(finish_time) - float(start_time),
                }
            )
    temporary.replace(path)


def _verified_candidate_path(job: dict[str, Any]) -> Path:
    """返回经提交时哈希核验的候选 checkpoint 路径。"""
    candidate_path = Path(str(job["candidate_path"])).resolve()
    expected_sha256 = str(job.get("candidate_sha256", "")).strip().lower()
    if expected_sha256:
        actual_sha256 = sha256_file(candidate_path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                "异步候选 checkpoint 哈希不一致，拒绝评估可能被覆写的文件: "
                f"expected={expected_sha256} actual={actual_sha256} path={candidate_path}"
            )
    return candidate_path


def load_checkpoint_agent_for_evaluation(job: dict[str, Any], device: "torch.device"):
    import torch

    from configs import configs
    from models.hb_gat_pn import HBGATPN
    from ppo_agent import PPOAgent
    from runtime.checkpoints import apply_checkpoint_model_spec, load_checkpoint, load_policy_weights

    candidate_path = _verified_candidate_path(job)
    raw_checkpoint = torch.load(candidate_path, map_location="cpu", weights_only=False)
    if isinstance(raw_checkpoint, dict) and raw_checkpoint.get("checkpoint_format") == "literature_baseline_v2":
        saved_config = raw_checkpoint.get("config")
        if not isinstance(saved_config, dict):
            raise ValueError("literature 异步 checkpoint 缺少 config")
        configs.update_from_dict(saved_config)
        from baselines.literature.common import LiteraturePolicyAdapter
        from baselines.literature.evaluate_literature_baseline import _build_model

        model = _build_model(raw_checkpoint, device)
        return raw_checkpoint, saved_config, LiteraturePolicyAdapter(model, device)
    checkpoint = load_checkpoint(candidate_path, map_location="cpu")
    saved_config = checkpoint.metadata.get("config")
    if not isinstance(saved_config, dict):
        raise ValueError("异步候选 checkpoint 缺少 apal_metadata.config")
    configs.update_from_dict(saved_config)
    apply_checkpoint_model_spec(configs, checkpoint.model_spec)
    model = HBGATPN(configs).to(device)
    load_policy_weights(model, checkpoint, strict=True)
    total_updates = math.ceil(int(configs.max_episodes) / int(configs.update_every_episodes))
    agent = PPOAgent(
        model=model,
        lr=float(configs.lr),
        gamma=float(configs.gamma),
        k_epochs=int(configs.k_epochs),
        eps_clip=float(configs.eps_clip),
        device=device,
        batch_size=int(configs.batch_size),
        total_timesteps=total_updates,
        config=configs,
    )
    payload = checkpoint.payload
    optimizer_states = payload.get("optimizer_states", []) if isinstance(payload, dict) else []
    if not optimizer_states:
        raise ValueError("异步候选 checkpoint 缺少 optimizer_states，无法恢复 ScheduleFree 评估权重")
    agent.optimizer.load_state_dict(optimizer_states[0])
    agent_state = payload.get("apal_agent_state", {}) if isinstance(payload, dict) else {}
    if isinstance(agent_state, dict):
        agent.current_step = int(agent_state.get("current_step", agent.current_step))
        scaler_state = agent_state.get("scaler")
        if isinstance(scaler_state, dict):
            agent.scaler.load_state_dict(scaler_state)
    return checkpoint, saved_config, agent


def _evaluate_initial_multiscale_job(
    job: dict[str, Any],
    *,
    saved_config: dict[str, Any],
    agent: Any,
) -> dict[str, Any]:
    from configs import configs
    from environment import AirLineEnv_Graph
    from runtime.initial_checkpoint_selection import parse_job_selection_manifest
    from runtime.initial_worker_mapping import apply_initial_worker_mapping
    from training.observation import refresh_env_observation
    from scripts.validate_initial_schedule import validate_schedule

    manifest = parse_job_selection_manifest(job.get("selection_manifest"))
    result_dir = Path(str(job["result_dir"])).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    was_training = bool(agent.policy.training)
    agent.policy.eval()
    try:
        for index, entry in enumerate(manifest.entries):
            configs.update_from_dict(saved_config)
            apply_initial_worker_mapping(configs, entry.data_path, explicit_fields=set())
            for field_name in (
                "enable_dynamic_events",
                "enable_station_breakdown",
                "enable_material_delay",
                "enable_online_duration_perturb",
                "enable_worker_fatigue",
                "randomize_durations",
            ):
                setattr(configs, field_name, False)
            reset_seed = int(manifest.seed) + index
            env = AirLineEnv_Graph(str(entry.data_path), seed=reset_seed)
            state = env.reset(randomize_duration=False, randomize_workers=False, seed=reset_seed)
            invalid_step_count = 0
            start_time = time.time()
            done = False
            total_reward = 0.0
            for _ in range(int(env.num_tasks) * 3):
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
                    state.to(agent.device),
                    mask_task=task_mask.to(agent.device),
                    mask_station_matrix=station_mask.to(agent.device),
                    mask_worker=worker_mask.to(agent.device),
                    deterministic=True,
                    temperature=0.0,
                    is_eval=True,
                    compute_value=False,
                )
                if action_ret[0] is None:
                    invalid_step_count += 1
                    break
                action, _logprob, _value, _station_mask, is_invalid = action_ret
                if is_invalid:
                    invalid_step_count += 1
                    break
                state, reward, done, info = env.step(action)
                total_reward += float(reward)
                if info.get("invalid_action", False) or info.get("error"):
                    invalid_step_count += 1
                    break

            schedule = list(env.assigned_tasks)
            complete = len(schedule) == int(env.num_tasks)
            engine_legal = False
            if complete and invalid_step_count == 0:
                engine_report = env.validate_assignments(schedule)
                engine_legal = bool(engine_report.is_legal)
                if not engine_legal:
                    invalid_step_count += 1
            makespan = (
                float(np.max(env.station_wall_clock))
                if complete and invalid_step_count == 0
                else float(env.ideal_makespan * 3.0)
            )
            schedule_path = result_dir / f"{entry.instance_id}_schedule.csv"
            audit_path = result_dir / f"{entry.instance_id}_legality_audit.json"
            _write_canonical_initial_schedule(schedule_path, schedule)
            audit = validate_schedule(
                data_path=entry.data_path,
                schedule_path=schedule_path,
                config_obj=configs,
                task_id_mode="internal",
            )
            atomic_write_json(audit_path, audit)
            violation_total = int(sum(int(value) for value in audit["violations"].values()))
            legal = bool(audit["is_legal_against_environment_duration"])
            eligible = bool(complete and invalid_step_count == 0 and engine_legal and legal)
            rows.append(
                {
                    "instance_id": entry.instance_id,
                    "data_path": str(entry.data_path),
                    "data_sha256": entry.sha256,
                    "reset_seed": reset_seed,
                    "reference_makespan": entry.reference_makespan,
                    "makespan": makespan,
                    "normalized_score": float(makespan / entry.reference_makespan),
                    "complete": bool(complete),
                    "invalid_step_count": int(invalid_step_count),
                    "engine_legal": bool(engine_legal),
                    "audit_legal": legal,
                    "hard_violation_total": violation_total,
                    "eligible": eligible,
                    "reward": total_reward,
                    "duration_sec": float(time.time() - start_time),
                    "schedule_path": str(schedule_path),
                    "audit_path": str(audit_path),
                }
            )
    finally:
        if was_training:
            agent.policy.train()

    composite_score = float(np.mean([row["normalized_score"] for row in rows]))
    eligible = bool(all(row["eligible"] for row in rows))
    output: dict[str, Any] = {
        "episode": int(job["episode"]),
        "evaluation_kind": "initial_multi_benchmark",
        "instance_id": "real_283_680_2338_3182",
        "scenario_id": "standard",
        "candidate_path": str(job["candidate_path"]),
        "candidate_sha256": str(job.get("candidate_sha256", "")),
        "selection_manifest_path": str(manifest.path),
        "selection_manifest_sha256": manifest.sha256,
        "selection_protocol_id": manifest.protocol_id,
        "selection_role": manifest.role,
        "temperature": manifest.temperature,
        "eligible": float(eligible),
        "composite_score": composite_score,
        "selection_score": composite_score if eligible else float("inf"),
        "makespan": float(np.mean([row["makespan"] for row in rows])),
        "duration_sec": float(sum(row["duration_sec"] for row in rows)),
        "hard_violation_total": int(sum(row["hard_violation_total"] for row in rows)),
        "instances": rows,
    }
    for row in rows:
        prefix = str(row["instance_id"])
        output[f"{prefix}_makespan"] = float(row["makespan"])
        output[f"{prefix}_normalized_score"] = float(row["normalized_score"])
        output[f"{prefix}_complete"] = float(row["complete"])
        output[f"{prefix}_eligible"] = float(row["eligible"])
        output[f"{prefix}_invalid_step_count"] = int(row["invalid_step_count"])
        output[f"{prefix}_hard_violation_total"] = int(row["hard_violation_total"])
    return output


def _evaluate_job(job: dict[str, Any], project_root: Path, *, device: "torch.device") -> tuple[dict[str, Any], list[Any]]:
    import torch

    from configs import configs
    from environment import AirLineEnv_Graph
    from runtime.seed import set_seed

    evaluation_kind = str(job.get("evaluation_kind", "reschedule"))
    if evaluation_kind not in {"reschedule", "initial_standard", "initial_multi_benchmark"}:
        raise ValueError(f"未知异步验证类型: {evaluation_kind}")
    checkpoint, saved_config, agent = load_checkpoint_agent_for_evaluation(job, device)
    is_reschedule = bool(configs.enable_reschedule_mode)
    if evaluation_kind == "reschedule" and not is_reschedule:
        raise ValueError("异步重调度验证候选不是重调度模型")
    if evaluation_kind != "reschedule" and is_reschedule:
        raise ValueError(f"初始调度异步验证候选错误地启用了重调度模式: {evaluation_kind}")
    if (
        str(job.get("reschedule_async_protocol", "")).strip().lower()
        == R5_RESCHEDULE_ASYNC_PROTOCOL
        and device.type != "cuda"
    ):
        raise RuntimeError("r5 重调度异步验证必须使用 CUDA worker")
    set_seed(int(configs.reschedule_eval_scenario_seed) if evaluation_kind == "reschedule" else int(configs.seed))

    if evaluation_kind == "initial_multi_benchmark":
        return _evaluate_initial_multiscale_job(job, saved_config=saved_config, agent=agent), []

    common = {
        "episode": int(job["episode"]),
        "job_id": str(job.get("job_id", "")),
        "group_id": str(job.get("group_id", "")),
        "evaluation_kind": evaluation_kind,
        "instance_id": str(job["instance_id"]),
        "scenario_id": str(job["scenario_id"]),
        "candidate_path": str(job["candidate_path"]),
        "candidate_sha256": str(job.get("candidate_sha256", "")),
    }
    if job.get("reschedule_async_protocol"):
        common["reschedule_async_protocol"] = str(job["reschedule_async_protocol"])
    if evaluation_kind == "reschedule":
        from runtime.reschedule_eval import evaluate_reschedule_model
        from runtime.reschedule_manifest import load_reschedule_manifest
        from runtime.initial_worker_mapping import apply_initial_worker_mapping

        manifest = load_reschedule_manifest(configs.reschedule_manifest_path)
        entry = manifest.get(str(job["instance_id"]))
        if entry.scenario_path is None:
            raise ValueError(f"manifest 实例缺少固定场景文件: {entry.instance_id}")
        configs.data_file_path = str(entry.data_path)
        configs.reschedule_baseline_schedule_path = str(entry.baseline_schedule_path)
        configs.reschedule_eval_scenario_path = str(entry.scenario_path)
        configs.reschedule_eval_instance_id = str(entry.instance_id)
        # 异步进程从训练 checkpoint 的配置恢复，必须在构造环境前同步真实
        # 实例的工人池规模，避免跨规模 baseline 出现 worker-range 误报。
        apply_initial_worker_mapping(configs, entry.data_path, explicit_fields=set())
        env = AirLineEnv_Graph(str(entry.data_path), seed=int(configs.reschedule_eval_scenario_seed))
        result = evaluate_reschedule_model(
            env, agent, num_runs=None, temperature=float(job["temperature"]),
            current_ep=int(job["episode"]), scenario_ids=[str(job["scenario_id"])],
            use_cached_observation=bool(job["use_cached_observation"]), skip_value_estimation=True,
        )
        makespan, balance, reward, schedule, duration, worker_util, station_util = result
        score_metrics = dict(getattr(evaluate_reschedule_model, "last_metrics", {}) or {})
        scenario_rows = list(getattr(evaluate_reschedule_model, "last_scenario_metrics", []) or [])
        scenario_metrics = dict(scenario_rows[0]) if scenario_rows else {}
        output = {
            **common, "makespan": float(makespan), "balance": float(balance), "reward": float(reward),
            "duration_sec": float(duration), "worker_utilization": float(worker_util), "station_utilization": float(station_util),
            **{str(key): float(value) for key, value in score_metrics.items() if isinstance(value, (int, float))},
            **{str(key): float(value) for key, value in scenario_metrics.items() if isinstance(value, (int, float))},
        }
        output["eligible"] = float(output.get("eligible", output.get("eligible_rate", 0.0)))
        output["selection_score"] = float(output.get("selection_score", output.get("composite_score", float("inf"))))
        return output, schedule

    from runtime.evaluation import evaluate_model
    from runtime.paths import resolve_workspace_path

    data_path = resolve_workspace_path(configs.async_eval_initial_data_path)
    env = AirLineEnv_Graph(str(data_path), seed=int(configs.seed))
    result = evaluate_model(env, agent, num_runs=1, temperature=float(job["temperature"]), current_ep=int(job["episode"]), scenario_names=("standard",))
    makespan, balance, reward, schedule, duration, worker_util, station_util = result
    complete = len(schedule) == int(env.num_tasks)
    output = {
        **common, "makespan": float(makespan), "balance": float(balance), "reward": float(reward),
        "duration_sec": float(duration), "worker_utilization": float(worker_util), "station_utilization": float(station_util),
        "complete": float(complete), "eligible": float(complete), "composite_score": float(makespan),
        "selection_score": float(makespan) if complete else float("inf"),
    }
    return output, schedule


def _atomic_copy_file(source: Path, destination: Path) -> None:
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _is_better_candidate(score: float, episode: int, best_state: dict[str, Any]) -> bool:
    previous_score = float(best_state.get("selection_score", float("inf")))
    previous_episode = int(best_state.get("episode", 2**31 - 1))
    return score < previous_score - 1e-12 or (
        math.isclose(score, previous_score, rel_tol=0.0, abs_tol=1e-12) and episode < previous_episode
    )


def _group_result_paths(paths: AsyncEvalPaths, job: dict[str, Any]) -> tuple[Path, ...]:
    group_id = str(job.get("group_id", "")).strip()
    scenario_ids = tuple(str(item) for item in job.get("group_scenario_ids", []))
    if not group_id or not scenario_ids:
        return ()
    return tuple(paths.results / f"{group_id}_{scenario_id}.json" for scenario_id in scenario_ids)


def _mean_result_field(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return float(np.mean(values)) if values else 0.0


def _aggregate_group_results(
    paths: AsyncEvalPaths,
    job: dict[str, Any],
) -> dict[str, Any] | None:
    result_paths = _group_result_paths(paths, job)
    if not result_paths or not all(path.is_file() for path in result_paths):
        return None
    rows = [json.loads(path.read_text(encoding="utf-8-sig")) for path in result_paths]
    scenario_ids = tuple(str(item) for item in job["group_scenario_ids"])
    if tuple(str(row.get("scenario_id", "")) for row in rows) != scenario_ids:
        raise RuntimeError("r5 异步验证子任务结果与预期场景顺序不一致")
    all_eligible = all(float(row.get("eligible", 0.0)) >= 1.0 - 1e-9 for row in rows)
    selection_values = [float(row.get("selection_score", float("inf"))) for row in rows]
    selection_score = (
        float(np.mean(selection_values))
        if all(math.isfinite(value) for value in selection_values)
        else float("inf")
    )
    first = rows[0]
    aggregate = {
        "episode": int(job["episode"]),
        "job_id": str(job.get("group_id", "")),
        "group_id": str(job.get("group_id", "")),
        "evaluation_kind": "reschedule",
        "reschedule_async_protocol": R5_RESCHEDULE_ASYNC_PROTOCOL,
        "instance_id": str(job["instance_id"]),
        "scenario_id": "|".join(scenario_ids),
        "scenario_ids": list(scenario_ids),
        "scenario_count": len(rows),
        "group_complete": 1.0,
        "candidate_path": str(job["candidate_path"]),
        "candidate_sha256": str(job.get("candidate_sha256", "")),
        "temperature": float(job.get("temperature", 0.0)),
        "eligible": float(all_eligible),
        "eligible_rate": float(np.mean([float(row.get("eligible", 0.0)) for row in rows])),
        "selection_score": selection_score,
        "composite_score": _mean_result_field(rows, "composite_score"),
        "makespan": _mean_result_field(rows, "makespan"),
        "balance": _mean_result_field(rows, "balance"),
        "reward": _mean_result_field(rows, "reward"),
        "duration_sec": _mean_result_field(rows, "duration_sec"),
        "worker_utilization": _mean_result_field(rows, "worker_utilization"),
        "station_utilization": _mean_result_field(rows, "station_utilization"),
        "scenario_results": rows,
    }
    if "best_path" in job:
        aggregate["best_path"] = str(job["best_path"])
    if "use_cached_observation" in first:
        aggregate["use_cached_observation"] = bool(first["use_cached_observation"])
    return aggregate


def _record_result(
    paths: AsyncEvalPaths,
    job: dict[str, Any],
    result: dict[str, Any],
    schedule: list[Any],
    writer: Any,
) -> bool:
    """写入结果；分组任务只有在全部子任务完成后才允许删除候选并选择 best。"""
    episode = int(job["episode"])
    candidate_path = _verified_candidate_path(job)
    expected_sha256 = str(job.get("candidate_sha256", "")).strip().lower()
    job_id = str(job.get("job_id", f"episode_{episode:06d}"))
    is_group = bool(job.get("group_id"))
    child_result_path = paths.results / f"{job_id}.json"
    aggregate_result: dict[str, Any] | None = None
    if is_group and schedule:
        _write_schedule(
            paths.results / str(job["group_id"]) / f"{job['scenario_id']}_schedule.csv",
            schedule,
        )

    # 多 worker 可同时完成；子结果和分组聚合必须作为一个串行发布区间。
    with _ExclusivePidLock(paths.result_lock):
        atomic_write_json(child_result_path, result)
        if is_group:
            aggregate_result = _aggregate_group_results(paths, job)
            if aggregate_result is not None:
                atomic_write_json(
                    paths.results / f"{job['group_id']}.json",
                    aggregate_result,
                )
        _write_summary_csv(paths)
    selected_result = aggregate_result if aggregate_result is not None else result
    for key, value in selected_result.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            writer.add_scalar(f"AsyncEval/{key}", float(value), episode)
    writer.flush()

    if is_group and aggregate_result is None:
        return False

    eligible = float(selected_result.get("eligible", 0.0)) >= 1.0 - 1e-9
    score = float(selected_result["selection_score"])
    if not eligible or not math.isfinite(score):
        return True
    with _ExclusivePidLock(paths.selection_lock):
        best_state_path = paths.state / "best.json"
        best_state = json.loads(best_state_path.read_text(encoding="utf-8-sig")) if best_state_path.exists() else {"selection_score": float("inf")}
        is_multiscale = str(job.get("evaluation_kind")) == "initial_multi_benchmark"
        is_better = (
            _is_better_candidate(score, episode, best_state)
            if is_multiscale
            else score < float(best_state.get("selection_score", float("inf")))
        )
        if not is_better:
            return True
        _atomic_copy_file(candidate_path, Path(job["best_path"]))
        best_sha256 = sha256_file(Path(job["best_path"]))
        if expected_sha256 and best_sha256 != expected_sha256:
            raise RuntimeError(
                "最佳 checkpoint 发布后的哈希与候选文件不一致: "
                f"expected={expected_sha256} actual={best_sha256}"
            )
        best_artifacts = paths.results / "best"
        if is_group:
            for scenario_id in job["group_scenario_ids"]:
                schedule_path = paths.results / str(job["group_id"]) / f"{scenario_id}_schedule.csv"
                if schedule_path.exists():
                    _atomic_copy_file(
                        schedule_path,
                        best_artifacts / f"{job['group_id']}_{scenario_id}_schedule.csv",
                    )
        elif is_multiscale:
            for row in result["instances"]:
                instance_id = str(row["instance_id"])
                _atomic_copy_file(Path(row["schedule_path"]), best_artifacts / f"{instance_id}_schedule.csv")
                _atomic_copy_file(Path(row["audit_path"]), best_artifacts / f"{instance_id}_legality_audit.json")
        elif schedule:
            # 保持旧单实例/重调度异步验证的产物路径不变。
            _write_schedule(paths.results / "best_schedule.csv", schedule)
        best_state = {
            "episode": episode,
            "evaluation_kind": str(job.get("evaluation_kind", "reschedule")),
            "selection_score": score,
            "composite_score": float(selected_result.get("composite_score", score)),
            "makespan": float(selected_result["makespan"]),
            "instance_id": str(job["instance_id"]),
            "scenario_id": str(selected_result.get("scenario_id", job.get("scenario_id", ""))),
            "best_path": str(job["best_path"]),
            "candidate_sha256": expected_sha256,
            "best_checkpoint_sha256": best_sha256,
            "result_path": str(
                paths.results / f"{job['group_id'] if is_group else job_id}.json"
            ),
            "updated_at": time.time(),
        }
        if is_group:
            best_state["group_id"] = str(job["group_id"])
            best_state["scenario_ids"] = list(job["group_scenario_ids"])
            best_state["scenario_count"] = len(job["group_scenario_ids"])
            best_state["reschedule_async_protocol"] = R5_RESCHEDULE_ASYNC_PROTOCOL
        elif is_multiscale:
            best_state["selection_manifest_sha256"] = str(selected_result["selection_manifest_sha256"])
            best_state["selection_protocol_id"] = str(selected_result["selection_protocol_id"])
            best_state["instances"] = selected_result["instances"]
        # best.json 是所有 checkpoint 与审计产物完成后的提交标记。
        atomic_write_json(best_state_path, best_state)
        print(
            f"[AsyncEval][Best] ep={episode} score={score:.6f} "
            f"mk={float(selected_result['makespan']):.2f} path={job['best_path']}",
            flush=True,
        )
    return True


def _evaluate_job_with_cuda_oom_fallback(
    job: dict[str, Any],
    project_root: Path,
    *,
    device: "torch.device",
) -> tuple[dict[str, Any], list[Any]]:
    """CUDA OOM 时仅将当前验证任务降级到 CPU，后续任务仍使用原设备。"""
    import gc
    import torch

    try:
        result, schedule = _evaluate_job(job, project_root, device=device)
        result = dict(result)
        result["evaluation_device"] = str(device)
        result["cuda_oom_cpu_fallback"] = 0.0
        return result, schedule
    except torch.cuda.OutOfMemoryError:
        if device.type != "cuda":
            raise
        if str(job.get("reschedule_async_protocol", "")).strip().lower() == R5_RESCHEDULE_ASYNC_PROTOCOL:
            raise
        gc.collect()
        torch.cuda.empty_cache()
        print(
            f"[AsyncEval][CUDAOOM] ep={int(job['episode'])} 当前任务降级到 CPU",
            flush=True,
        )
        result, schedule = _evaluate_job(
            job,
            project_root,
            device=torch.device("cpu"),
        )
        result = dict(result)
        result["evaluation_device"] = "cpu"
        result["cuda_oom_cpu_fallback"] = 1.0
        return result, schedule


def run_worker(
    queue_root: Path,
    project_root: Path,
    *,
    heartbeat_interval: float = 30.0,
    worker_id: str = "worker",
    device_name: str = "cpu",
) -> int:
    import torch
    from torch.utils.tensorboard import SummaryWriter

    device_name = str(device_name).strip().lower()
    if device_name not in {"cpu", "cuda", "cuda:0"}:
        raise ValueError(f"异步验证设备非法: {device_name!r}")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 异步验证，但 worker 中 CUDA 不可用")
    device = torch.device(device_name)
    paths = AsyncEvalPaths.create(queue_root)
    thread_count = max(1, int(os.environ.get("APAL_ASYNC_EVAL_CPU_THREADS", "4")))
    torch.set_num_threads(thread_count)
    torch.set_num_interop_threads(1)
    lock_path = paths.worker_locks / f"{worker_id}.lock"
    heartbeat_path = paths.heartbeats / f"{worker_id}.json"
    with _WorkerLock(lock_path):
        writer = SummaryWriter(log_dir=str(paths.root / "tensorboard"))
        try:
            with _Heartbeat(heartbeat_path, heartbeat_interval):
                while True:
                    claimed = _claim_next_job(paths)
                    if claimed is None:
                        if paths.stop_when_idle.exists():
                            return 0
                        time.sleep(0.5)
                        continue
                    running_path, job = claimed
                    episode = int(job["episode"])
                    start_time = time.time()
                    try:
                        result, schedule = _evaluate_job_with_cuda_oom_fallback(
                            job,
                            project_root,
                            device=device,
                        )
                        candidate_can_delete = _record_result(paths, job, result, schedule, writer)
                        done_payload = {**job, "completed_at": time.time(), "worker_duration_sec": time.time() - start_time, "result": result}
                        atomic_write_json(paths.done / running_path.name, done_payload)
                        running_path.unlink(missing_ok=True)
                        if candidate_can_delete:
                            Path(job["candidate_path"]).unlink(missing_ok=True)
                        print(
                            f"[AsyncEval][Done] ep={episode} score={float(result['selection_score']):.6f} "
                            f"elig={int(float(result.get('eligible', 0.0)) >= 1.0 - 1e-9)} "
                            f"mk={float(result['makespan']):.2f} time={time.time() - start_time:.1f}s",
                            flush=True,
                        )
                    except Exception as exc:
                        attempt = int(job.get("attempt", 0)) + 1
                        job["attempt"] = attempt
                        job["last_error"] = str(exc)
                        job["last_traceback"] = traceback.format_exc()
                        if attempt <= int(job.get("max_retries", 1)):
                            atomic_write_json(paths.pending / running_path.name, job)
                            running_path.unlink(missing_ok=True)
                            print(f"[AsyncEval][Retry] ep={episode} attempt={attempt} error={exc}", flush=True)
                            continue
                        failure = {**job, "failed_at": time.time(), "error": str(exc), "traceback": traceback.format_exc()}
                        atomic_write_json(paths.failed / running_path.name, failure)
                        running_path.unlink(missing_ok=True)
                        print(f"[AsyncEval][Failed] ep={episode} attempts={attempt} error={exc}", flush=True)
                        return 2
        finally:
            writer.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="APAL Lightning 异步验证 worker")
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--heartbeat-interval", type=float, default=30.0)
    parser.add_argument("--worker-id", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return run_worker(args.queue_root.resolve(), args.project_root.resolve(), heartbeat_interval=float(args.heartbeat_interval), worker_id=str(args.worker_id), device_name=str(args.device))


if __name__ == "__main__":
    raise SystemExit(main())
