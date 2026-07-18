from __future__ import annotations

import argparse
import csv
import json
import math
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from training.async_evaluation import (
    AsyncEvalPaths,
    atomic_link_or_copy,
    atomic_write_json,
    process_is_alive,
)


class _Heartbeat:
    def __init__(self, path: Path, interval_sec: float) -> None:
        self.path = path
        self.interval_sec = max(1.0, float(interval_sec))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="async-eval-heartbeat", daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_sec):
            atomic_write_json(
                self.path,
                {"pid": os.getpid(), "updated_at": time.time()},
            )

    def __enter__(self) -> "_Heartbeat":
        atomic_write_json(self.path, {"pid": os.getpid(), "updated_at": time.time()})
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback_obj) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval_sec + 1.0)
        atomic_write_json(self.path, {"pid": os.getpid(), "updated_at": time.time()})


class _WorkerLock:
    """防止同一队列被两个恢复进程同时消费。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._acquired = False

    def __enter__(self) -> "_WorkerLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    existing_pid = int(self.path.read_text(encoding="ascii").strip())
                except (OSError, ValueError):
                    existing_pid = -1
                if process_is_alive(existing_pid):
                    raise RuntimeError(
                        f"异步验证队列已有活动 worker: pid={existing_pid} lock={self.path}"
                    )
                self.path.unlink(missing_ok=True)
                continue
            with os.fdopen(descriptor, "w", encoding="ascii") as handle:
                handle.write(str(os.getpid()))
            self._acquired = True
            return self
        raise RuntimeError(f"无法获取异步验证 worker 锁: {self.path}")

    def __exit__(self, exc_type, exc_value, traceback_obj) -> None:
        if self._acquired:
            self.path.unlink(missing_ok=True)


def _restore_interrupted_jobs(paths: AsyncEvalPaths) -> None:
    for running_path in sorted(paths.running.glob("*.json")):
        pending_path = paths.pending / running_path.name
        running_path.replace(pending_path)


def _claim_next_job(paths: AsyncEvalPaths) -> tuple[Path, dict[str, Any]] | None:
    for pending_path in sorted(paths.pending.glob("*.json")):
        running_path = paths.running / pending_path.name
        try:
            pending_path.replace(running_path)
        except FileNotFoundError:
            continue
        return running_path, json.loads(running_path.read_text(encoding="utf-8-sig"))
    return None


def _write_summary_csv(paths: AsyncEvalPaths) -> None:
    rows = []
    for result_path in sorted(paths.results.glob("episode_*.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8-sig"))
        rows.append(payload)
    if not rows:
        return
    preferred = [
        "episode",
        "evaluation_kind",
        "instance_id",
        "scenario_id",
        "scenario_index",
        "scenario_reset_seed",
        "eligible",
        "selection_score",
        "composite_score",
        "makespan",
        "balance",
        "reward",
        "worker_utilization",
        "station_utilization",
        "duration_sec",
        "candidate_path",
    ]
    extra = sorted({key for row in rows for key in row}.difference(preferred))
    fieldnames = [key for key in preferred if any(key in row for row in rows)] + extra
    destination = paths.results / "async_eval_summary.csv"
    temporary = destination.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)


def _write_schedule(path: Path, schedule: list[Any]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task_id", "station_id", "worker_ids", "start_time", "finish_time"])
        for task_id, station_id, workers, start_time, finish_time in schedule:
            writer.writerow(
                [
                    int(task_id),
                    int(station_id),
                    ";".join(str(int(worker_id)) for worker_id in workers),
                    float(start_time),
                    float(finish_time),
                ]
            )
    temporary.replace(path)


def _evaluate_job(job: dict[str, Any], project_root: Path) -> tuple[dict[str, Any], list[Any]]:
    import torch

    from configs import configs
    from environment import AirLineEnv_Graph
    from models.hb_gat_pn import HBGATPN
    from ppo_agent import PPOAgent
    from runtime.checkpoints import (
        apply_checkpoint_model_spec,
        load_checkpoint,
        load_policy_weights,
    )
    from runtime.seed import set_seed

    checkpoint = load_checkpoint(job["candidate_path"], map_location="cpu")
    saved_config = checkpoint.metadata.get("config")
    if not isinstance(saved_config, dict):
        raise ValueError("异步候选 checkpoint 缺少 apal_metadata.config")
    configs.update_from_dict(saved_config)
    apply_checkpoint_model_spec(configs, checkpoint.model_spec)
    evaluation_kind = str(job.get("evaluation_kind", "reschedule"))
    is_reschedule = bool(configs.enable_reschedule_mode)
    if evaluation_kind == "reschedule" and not is_reschedule:
        raise ValueError("异步重调度验证候选不是重调度模型")
    if evaluation_kind != "reschedule" and is_reschedule:
        raise ValueError(f"异步初始调度验证候选错误地启用了重调度模式: {evaluation_kind}")
    if evaluation_kind not in {
        "reschedule",
        "initial_standard",
    }:
        raise ValueError(f"未知异步验证类型: {evaluation_kind}")

    seed = (
        int(configs.reschedule_eval_scenario_seed)
        if evaluation_kind == "reschedule"
        else int(configs.seed)
    )
    set_seed(seed)
    device = torch.device("cpu")
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

    common = {
        "episode": int(job["episode"]),
        "evaluation_kind": evaluation_kind,
        "instance_id": str(job["instance_id"]),
        "scenario_id": str(job["scenario_id"]),
        "candidate_path": str(job["candidate_path"]),
    }

    if evaluation_kind == "reschedule":
        from runtime.reschedule_eval import evaluate_reschedule_model
        from runtime.reschedule_manifest import load_reschedule_manifest

        manifest = load_reschedule_manifest(configs.reschedule_manifest_path)
        entry = manifest.get(str(job["instance_id"]))
        if entry.scenario_path is None:
            raise ValueError(f"manifest 实例缺少固定场景文件: {entry.instance_id}")
        configs.data_file_path = str(entry.data_path)
        configs.reschedule_baseline_schedule_path = str(entry.baseline_schedule_path)
        configs.reschedule_eval_scenario_path = str(entry.scenario_path)
        configs.reschedule_eval_instance_id = str(entry.instance_id)
        env = AirLineEnv_Graph(str(entry.data_path), seed=seed)
        result = evaluate_reschedule_model(
            env,
            agent,
            num_runs=None,
            temperature=float(job["temperature"]),
            current_ep=int(job["episode"]),
            scenario_ids=[str(job["scenario_id"])],
            use_cached_observation=bool(job["use_cached_observation"]),
            skip_value_estimation=True,
        )
        makespan, balance, reward, schedule, duration, worker_util, station_util = result
        score_metrics = dict(getattr(evaluate_reschedule_model, "last_metrics", {}) or {})
        scenario_rows = list(getattr(evaluate_reschedule_model, "last_scenario_metrics", []) or [])
        scenario_metrics = dict(scenario_rows[0]) if scenario_rows else {}
        output = {
            **common,
            "makespan": float(makespan),
            "balance": float(balance),
            "reward": float(reward),
            "duration_sec": float(duration),
            "worker_utilization": float(worker_util),
            "station_utilization": float(station_util),
            **{
                str(key): float(value)
                for key, value in score_metrics.items()
                if isinstance(value, (int, float))
            },
            **{
                str(key): float(value)
                for key, value in scenario_metrics.items()
                if isinstance(value, (int, float))
            },
        }
        output["eligible"] = float(output.get("eligible", output.get("eligible_rate", 0.0)))
        output["selection_score"] = float(
            output.get("selection_score", output.get("composite_score", float("inf")))
        )
        return output, schedule

    from runtime.evaluation import evaluate_model
    from runtime.paths import resolve_workspace_path

    data_path = resolve_workspace_path(configs.async_eval_initial_data_path)
    env = AirLineEnv_Graph(str(data_path), seed=seed)
    result = evaluate_model(
        env,
        agent,
        num_runs=1,
        temperature=float(job["temperature"]),
        current_ep=int(job["episode"]),
        scenario_names=("standard",),
    )
    makespan, balance, reward, schedule, duration, worker_util, station_util = result
    complete = len(schedule) == int(env.num_tasks)
    output = {
        **common,
        "makespan": float(makespan),
        "balance": float(balance),
        "reward": float(reward),
        "duration_sec": float(duration),
        "worker_utilization": float(worker_util),
        "station_utilization": float(station_util),
        "complete": float(complete),
        "eligible": float(complete),
        "composite_score": float(makespan),
        "selection_score": float(makespan) if complete else float("inf"),
    }
    return output, schedule


def _record_result(
    paths: AsyncEvalPaths,
    job: dict[str, Any],
    result: dict[str, Any],
    schedule: list[Any],
    writer: Any,
) -> None:
    episode = int(job["episode"])
    atomic_write_json(paths.results / f"episode_{episode:06d}.json", result)
    _write_summary_csv(paths)
    for key, value in result.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            writer.add_scalar(f"AsyncEval/{key}", float(value), episode)
    writer.flush()

    best_state_path = paths.state / "best.json"
    best_state = (
        json.loads(best_state_path.read_text(encoding="utf-8-sig"))
        if best_state_path.exists()
        else {"selection_score": float("inf")}
    )
    eligible = float(result.get("eligible", 0.0)) >= 1.0 - 1e-9
    score = float(result["selection_score"])
    if eligible and score < float(best_state.get("selection_score", float("inf"))):
        atomic_link_or_copy(Path(job["candidate_path"]), Path(job["best_path"]))
        if schedule:
            _write_schedule(paths.results / "best_schedule.csv", schedule)
        best_state = {
            "episode": episode,
            "evaluation_kind": str(job.get("evaluation_kind", "reschedule")),
            "selection_score": score,
            "composite_score": float(result.get("composite_score", score)),
            "makespan": float(result["makespan"]),
            "instance_id": str(job["instance_id"]),
            "scenario_id": str(job["scenario_id"]),
            "best_path": str(job["best_path"]),
            "updated_at": time.time(),
        }
        atomic_write_json(best_state_path, best_state)
        print(
            f"[AsyncEval][Best] ep={episode} score={score:.6f} "
            f"mk={float(result['makespan']):.2f} path={job['best_path']}",
            flush=True,
        )


def run_worker(queue_root: Path, project_root: Path, *, heartbeat_interval: float = 30.0) -> int:
    import torch
    from torch.utils.tensorboard import SummaryWriter

    paths = AsyncEvalPaths.create(queue_root)
    thread_count = max(1, int(os.environ.get("APAL_ASYNC_EVAL_CPU_THREADS", "4")))
    torch.set_num_threads(thread_count)
    torch.set_num_interop_threads(1)
    with _WorkerLock(paths.worker_lock):
        _restore_interrupted_jobs(paths)
        writer = SummaryWriter(log_dir=str(paths.root / "tensorboard"))
        try:
            with _Heartbeat(paths.heartbeat, heartbeat_interval):
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
                        result, schedule = _evaluate_job(job, project_root)
                        _record_result(paths, job, result, schedule, writer)
                        done_payload = {
                            **job,
                            "completed_at": time.time(),
                            "worker_duration_sec": time.time() - start_time,
                            "result": result,
                        }
                        atomic_write_json(paths.done / running_path.name, done_payload)
                        running_path.unlink(missing_ok=True)
                        Path(job["candidate_path"]).unlink(missing_ok=True)
                        print(
                            f"[AsyncEval][Done] ep={episode} "
                            f"score={float(result['selection_score']):.6f} "
                            f"elig={int(float(result.get('eligible', 0.0)) >= 1.0 - 1e-9)} "
                            f"mk={float(result['makespan']):.2f} "
                            f"time={time.time() - start_time:.1f}s",
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
                            print(
                                f"[AsyncEval][Retry] ep={episode} attempt={attempt} error={exc}",
                                flush=True,
                            )
                            continue
                        failure = {
                            **job,
                            "failed_at": time.time(),
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                        atomic_write_json(paths.failed / running_path.name, failure)
                        running_path.unlink(missing_ok=True)
                        print(
                            f"[AsyncEval][Failed] ep={episode} attempts={attempt} error={exc}",
                            flush=True,
                        )
                        return 2
        finally:
            writer.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="APAL Lightning 异步验证 worker")
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--heartbeat-interval", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return run_worker(
        args.queue_root.resolve(),
        args.project_root.resolve(),
        heartbeat_interval=float(args.heartbeat_interval),
    )


if __name__ == "__main__":
    raise SystemExit(main())
