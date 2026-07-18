from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class AsyncEvaluationError(RuntimeError):
    """异步验证进程或队列进入不可恢复状态。"""


@dataclass(frozen=True)
class AsyncEvalPaths:
    root: Path
    candidates: Path
    pending: Path
    running: Path
    done: Path
    failed: Path
    results: Path
    state: Path
    heartbeat: Path
    worker_lock: Path
    stop_when_idle: Path

    @classmethod
    def create(cls, root: Path) -> "AsyncEvalPaths":
        root = Path(root)
        paths = cls(
            root=root,
            candidates=root / "candidates",
            pending=root / "queue" / "pending",
            running=root / "queue" / "running",
            done=root / "queue" / "done",
            failed=root / "queue" / "failed",
            results=root / "results",
            state=root / "state",
            heartbeat=root / "state" / "worker_heartbeat.json",
            worker_lock=root / "state" / "worker.lock",
            stop_when_idle=root / "state" / "stop_when_idle",
        )
        for directory in (
            paths.candidates,
            paths.pending,
            paths.running,
            paths.done,
            paths.failed,
            paths.results,
            paths.state,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return paths


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """使用同目录临时文件和 replace 发布完整 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_link_or_copy(source: Path, destination: Path) -> None:
    """优先硬链接，跨设备或权限不允许时退化为原子复制。"""
    source = Path(source)
    destination = Path(destination)
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint 源文件不存在: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return int(exit_code.value) == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class AsyncEvaluationManager:
    """在训练进程中管理有界异步验证队列和独立 CPU worker。"""

    def __init__(
        self,
        *,
        config: Any,
        latest_path: Path,
        best_path: Path,
        project_root: Path,
    ) -> None:
        self.config = config
        self.latest_path = Path(latest_path).resolve()
        self.best_path = Path(best_path).resolve()
        self.project_root = Path(project_root).resolve()
        self.paths = AsyncEvalPaths.create(self.latest_path.parent / "async_eval")
        self.capacity = int(config.async_eval_queue_capacity)
        self.poll_interval = float(config.async_eval_poll_interval_sec)
        self.heartbeat_interval = float(config.async_eval_heartbeat_interval_sec)
        self.stale_timeout = float(config.async_eval_stale_timeout_sec)
        self._process: subprocess.Popen[str] | None = None
        self._started_at = 0.0
        self._last_wait_log = 0.0
        self.paths.stop_when_idle.unlink(missing_ok=True)
        if self._failed_jobs():
            self._check_health()
        if self._active_job_count() > 0:
            self._start_worker()

    def _active_job_count(self) -> int:
        return len(tuple(self.paths.pending.glob("*.json"))) + len(
            tuple(self.paths.running.glob("*.json"))
        )

    def _failed_jobs(self) -> list[Path]:
        return sorted(self.paths.failed.glob("*.json"))

    def _start_worker(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        if self.paths.worker_lock.exists():
            try:
                lock_pid = int(self.paths.worker_lock.read_text(encoding="ascii").strip())
            except (OSError, ValueError):
                lock_pid = -1
            if process_is_alive(lock_pid):
                raise AsyncEvaluationError(
                    f"异步验证队列已有活动 worker: pid={lock_pid} "
                    f"lock={self.paths.worker_lock}"
                )
            self.paths.worker_lock.unlink(missing_ok=True)
        self.paths.heartbeat.unlink(missing_ok=True)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        thread_count = str(int(self.config.async_eval_cpu_threads))
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            env[name] = thread_count
        env["APAL_ASYNC_EVAL_CPU_THREADS"] = thread_count
        command = [
            sys.executable,
            "-m",
            "training.async_eval_worker",
            "--queue-root",
            str(self.paths.root),
            "--project-root",
            str(self.project_root),
            "--heartbeat-interval",
            str(self.heartbeat_interval),
        ]
        self._process = subprocess.Popen(
            command,
            cwd=str(self.project_root),
            env=env,
            text=True,
        )
        self._started_at = time.monotonic()
        print(
            f"[AsyncEval] worker_pid={self._process.pid} device=cpu "
            f"threads={thread_count} capacity={self.capacity}",
            flush=True,
        )

    def _check_health(self) -> None:
        failed = self._failed_jobs()
        if failed:
            payload = json.loads(failed[0].read_text(encoding="utf-8-sig"))
            raise AsyncEvaluationError(
                f"异步验证失败: job={failed[0].name} error={payload.get('error', 'unknown')}"
            )
        if self._process is not None:
            return_code = self._process.poll()
            if return_code not in (None, 0):
                raise AsyncEvaluationError(f"异步验证 worker 异常退出: returncode={return_code}")
            if return_code == 0 and self._active_job_count() > 0:
                raise AsyncEvaluationError("异步验证 worker 已退出，但队列仍有未完成任务")
        if self._active_job_count() <= 0:
            return
        if self.paths.heartbeat.exists():
            age = time.time() - self.paths.heartbeat.stat().st_mtime
        else:
            age = time.monotonic() - self._started_at
        if age > self.stale_timeout:
            raise AsyncEvaluationError(
                f"异步验证 worker 心跳超时: age={age:.1f}s limit={self.stale_timeout:.1f}s"
            )

    def _wait_for_slot(self) -> None:
        self._start_worker()
        while self._active_job_count() >= self.capacity:
            self._check_health()
            now = time.monotonic()
            if now - self._last_wait_log >= self.heartbeat_interval:
                print(
                    f"[AsyncEval] 队列已满，训练等待验证释放槽位: "
                    f"active={self._active_job_count()}/{self.capacity}",
                    flush=True,
                )
                self._last_wait_log = now
            time.sleep(self.poll_interval)

    def submit(self, trainer: Any, *, episode: int) -> Path:
        """保存完整候选 checkpoint，并在文件完整后原子发布队列任务。"""
        self._wait_for_slot()
        self._check_health()
        job_name = f"episode_{int(episode):06d}.json"
        collisions = [
            directory / job_name
            for directory in (
                self.paths.pending,
                self.paths.running,
                self.paths.done,
                self.paths.failed,
            )
            if (directory / job_name).exists()
        ]
        if collisions:
            raise AsyncEvaluationError(
                f"episode={episode} 已存在异步验证记录，拒绝覆盖: {collisions[0]}"
            )
        candidate = self.paths.candidates / f"episode_{int(episode):06d}.ckpt"
        temporary = candidate.with_name(f".{candidate.name}.{uuid.uuid4().hex}.tmp")
        try:
            trainer.save_checkpoint(str(temporary))
            temporary.replace(candidate)
        finally:
            temporary.unlink(missing_ok=True)
        atomic_link_or_copy(candidate, self.latest_path)

        if bool(getattr(self.config, "enable_reschedule_mode", False)):
            evaluation_kind = "reschedule"
        else:
            evaluation_kind = "initial_standard"

        job = {
            "format_version": 1,
            "episode": int(episode),
            "candidate_path": str(candidate.resolve()),
            "best_path": str(self.best_path),
            "evaluation_kind": evaluation_kind,
            "temperature": float(self.config.eval_temperature),
            "max_retries": int(self.config.async_eval_max_retries),
            "attempt": 0,
            "use_cached_observation": bool(self.config.async_eval_use_cached_observation),
            "submitted_at": time.time(),
        }
        if evaluation_kind == "reschedule":
            job["instance_id"] = str(self.config.async_eval_instance_id)
            job["scenario_id"] = str(self.config.async_eval_scenario_id)
        else:
            job["instance_id"] = Path(str(self.config.async_eval_initial_data_path)).stem
            job["scenario_id"] = "standard"
        job_path = self.paths.pending / job_name
        atomic_write_json(job_path, job)
        print(
            f"[AsyncEval] ep={episode} 已入队 "
            f"kind={evaluation_kind} target={job['instance_id']}/{job['scenario_id']} "
            f"active={self._active_job_count()}/{self.capacity}",
            flush=True,
        )
        return candidate

    def finalize(self, *, wait: bool) -> None:
        if self._process is None:
            return
        if not wait:
            return
        self.paths.stop_when_idle.touch(exist_ok=True)
        while self._process.poll() is None:
            self._check_health()
            time.sleep(self.poll_interval)
        self._check_health()
        print("[AsyncEval] 队列已清空，worker 正常退出。", flush=True)

    def terminate_for_exception(self) -> None:
        """异常退出时保留 pending/running/candidate，供下次启动恢复。"""
        if self._process is None or self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=10.0)


__all__ = [
    "AsyncEvalPaths",
    "AsyncEvaluationError",
    "AsyncEvaluationManager",
    "atomic_link_or_copy",
    "atomic_write_json",
    "process_is_alive",
]
