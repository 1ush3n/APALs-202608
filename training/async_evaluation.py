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

from runtime.initial_checkpoint_selection import (
    InitialCheckpointSelectionManifest,
    load_initial_checkpoint_selection_manifest,
    sha256_file,
)


class AsyncEvaluationError(RuntimeError):
    """异步验证进程、队列或最优模型发布进入不可恢复状态。"""


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
    heartbeats: Path
    worker_lock: Path
    worker_locks: Path
    result_lock: Path
    selection_lock: Path
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
            # 保留旧字段路径，仅用于兼容历史目录；新 worker 使用 worker_locks。
            heartbeat=root / "state" / "worker_heartbeat.json",
            heartbeats=root / "state" / "heartbeats",
            worker_lock=root / "state" / "worker.lock",
            worker_locks=root / "state" / "workers",
            result_lock=root / "state" / "results.lock",
            selection_lock=root / "state" / "selection.lock",
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
            paths.heartbeats,
            paths.worker_locks,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return paths


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """使用同目录临时文件和 replace 发布完整 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_link_or_copy(source: Path, destination: Path) -> None:
    """优先硬链接，跨设备时退化为原子复制。"""
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


def atomic_copy_file(source: Path, destination: Path) -> None:
    """原子复制文件，确保目标与源文件绝不共享 inode。"""
    source = Path(source)
    destination = Path(destination)
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint 源文件不存在: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
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


def restore_interrupted_jobs(paths: AsyncEvalPaths) -> None:
    """仅在确认不存在存活 worker 后，将中断任务放回 pending。"""
    for running_path in sorted(paths.running.glob("*.json")):
        pending_path = paths.pending / running_path.name
        if pending_path.exists():
            raise AsyncEvaluationError(f"恢复任务冲突: {pending_path}")
        running_path.replace(pending_path)


class AsyncEvaluationManager:
    """管理可恢复、有界的异步验证队列。CUDA 模式限定单 worker。"""

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
        self.worker_count = int(getattr(config, "async_eval_worker_count", 1))
        self.device = str(getattr(config, "async_eval_device", "cpu")).strip().lower()
        if self.device not in {"cpu", "cuda", "cuda:0"}:
            raise AsyncEvaluationError(f"async_eval_device 仅允许 cpu、cuda 或 cuda:0，实际为 {self.device!r}")
        if self.device.startswith("cuda") and self.worker_count != 1:
            raise AsyncEvaluationError("CUDA 异步验证只允许 async_eval_worker_count=1，避免与训练争抢显存")
        self.poll_interval = float(config.async_eval_poll_interval_sec)
        self.heartbeat_interval = float(config.async_eval_heartbeat_interval_sec)
        self.stale_timeout = float(config.async_eval_stale_timeout_sec)
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._started_at = time.monotonic()
        self._last_wait_log = 0.0
        self.selection_manifest = self._load_selection_manifest()
        self.paths.stop_when_idle.unlink(missing_ok=True)
        self._remove_stale_worker_locks()
        active_pids = self._active_worker_pids()
        if active_pids:
            raise AsyncEvaluationError(
                "异步验证目录已有存活 worker，拒绝与另一训练进程共享队列: "
                f"pids={sorted(active_pids)}"
            )
        restore_interrupted_jobs(self.paths)
        if self._failed_jobs():
            self._check_health()
        if self._active_job_count() > 0:
            self._start_workers()

    def _load_selection_manifest(self) -> InitialCheckpointSelectionManifest | None:
        if bool(getattr(self.config, "enable_reschedule_mode", False)):
            return None
        protocol = str(
            getattr(self.config, "checkpoint_selection_protocol", "single_standard")
        ).strip().lower()
        if protocol != "multiscale_manifest":
            return None
        return load_initial_checkpoint_selection_manifest(
            self.config.checkpoint_selection_manifest_path
        )

    def _active_job_count(self) -> int:
        return len(tuple(self.paths.pending.glob("*.json"))) + len(tuple(self.paths.running.glob("*.json")))

    def _failed_jobs(self) -> list[Path]:
        return sorted(self.paths.failed.glob("*.json"))

    def _remove_stale_worker_locks(self) -> None:
        self.paths.worker_lock.unlink(missing_ok=True)
        for lock_path in self.paths.worker_locks.glob("*.lock"):
            try:
                pid = int(lock_path.read_text(encoding="ascii").strip())
            except (OSError, ValueError):
                pid = -1
            if not process_is_alive(pid):
                lock_path.unlink(missing_ok=True)

    def _active_worker_pids(self) -> set[int]:
        pids: set[int] = set()
        for lock_path in self.paths.worker_locks.glob("*.lock"):
            try:
                pid = int(lock_path.read_text(encoding="ascii").strip())
            except (OSError, ValueError):
                continue
            if process_is_alive(pid):
                pids.add(pid)
        return pids

    def _start_workers(self) -> None:
        self._remove_stale_worker_locks()
        self._processes = {
            worker_id: process
            for worker_id, process in self._processes.items()
            if process.poll() is None
        }
        external_pids = self._active_worker_pids() - {process.pid for process in self._processes.values()}
        if external_pids:
            raise AsyncEvaluationError(
                f"异步验证队列已有外部 worker: pids={sorted(external_pids)}"
            )
        for _ in range(self.worker_count - len(self._processes)):
            worker_id = f"worker_{uuid.uuid4().hex}"
            env = os.environ.copy()
            if self.device == "cpu":
                env["CUDA_VISIBLE_DEVICES"] = ""
            thread_count = str(int(self.config.async_eval_cpu_threads)) if self.device == "cpu" else "1"
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
                "--worker-id",
                worker_id,
                "--device",
                self.device,
            ]
            process = subprocess.Popen(command, cwd=str(self.project_root), env=env, text=True)
            self._processes[worker_id] = process
            print(
                f"[AsyncEval] worker_id={worker_id} pid={process.pid} device={self.device} "
                f"threads={thread_count} workers={self.worker_count} capacity={self.capacity}",
                flush=True,
            )
        self._started_at = time.monotonic()

    def _check_health(self) -> None:
        failed = self._failed_jobs()
        if failed:
            payload = json.loads(failed[0].read_text(encoding="utf-8-sig"))
            raise AsyncEvaluationError(
                f"异步验证失败: job={failed[0].name} error={payload.get('error', 'unknown')}"
            )
        for worker_id, process in self._processes.items():
            return_code = process.poll()
            if return_code not in (None, 0):
                raise AsyncEvaluationError(
                    f"异步验证 worker 异常退出: worker={worker_id} returncode={return_code}"
                )
        if self._active_job_count() <= 0:
            return
        live_processes = [process for process in self._processes.values() if process.poll() is None]
        if not live_processes:
            raise AsyncEvaluationError("异步验证队列仍有任务，但没有存活 worker")
        heartbeat_files = list(self.paths.heartbeats.glob("*.json"))
        latest_heartbeat = max((path.stat().st_mtime for path in heartbeat_files), default=0.0)
        age = time.time() - latest_heartbeat if latest_heartbeat else time.monotonic() - self._started_at
        if age > self.stale_timeout:
            raise AsyncEvaluationError(
                f"异步验证 worker 心跳超时: age={age:.1f}s limit={self.stale_timeout:.1f}s"
            )

    def _wait_for_slot(self) -> None:
        self._start_workers()
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
            for directory in (self.paths.pending, self.paths.running, self.paths.done, self.paths.failed)
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
        candidate_sha256 = sha256_file(candidate)
        # latest 是训练进程后续会持续覆写的可变 checkpoint，绝不能与候选文件共享 inode。
        atomic_copy_file(candidate, self.latest_path)

        if bool(getattr(self.config, "enable_reschedule_mode", False)):
            evaluation_kind = "reschedule"
        elif self.selection_manifest is not None:
            evaluation_kind = "initial_multi_benchmark"
        else:
            evaluation_kind = "initial_standard"

        job: dict[str, Any] = {
            "format_version": 2,
            "episode": int(episode),
            "candidate_path": str(candidate.resolve()),
            "candidate_sha256": candidate_sha256,
            "best_path": str(self.best_path),
            "result_dir": str((self.paths.results / f"episode_{int(episode):06d}").resolve()),
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
        elif evaluation_kind == "initial_multi_benchmark":
            assert self.selection_manifest is not None
            job["instance_id"] = "real_283_680_2338_3182"
            job["scenario_id"] = "standard"
            job["selection_manifest"] = self.selection_manifest.as_job_payload()
        else:
            job["instance_id"] = Path(str(self.config.async_eval_initial_data_path)).stem
            job["scenario_id"] = "standard"
        job_path = self.paths.pending / job_name
        atomic_write_json(job_path, job)
        print(
            f"[AsyncEval] ep={episode} 已入队 kind={evaluation_kind} "
            f"target={job['instance_id']}/{job['scenario_id']} "
            f"active={self._active_job_count()}/{self.capacity}",
            flush=True,
        )
        return candidate

    def finalize(self, *, wait: bool) -> None:
        if not self._processes or not wait:
            return
        self.paths.stop_when_idle.touch(exist_ok=True)
        while any(process.poll() is None for process in self._processes.values()):
            self._check_health()
            time.sleep(self.poll_interval)
        self._check_health()
        print("[AsyncEval] 队列已清空，所有 worker 正常退出。", flush=True)

    def terminate_for_exception(self) -> None:
        """异常退出时保留 pending/running/candidate，供下次启动恢复。"""
        for process in self._processes.values():
            if process.poll() is not None:
                continue
            process.terminate()
        for process in self._processes.values():
            if process.poll() is not None:
                continue
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10.0)


__all__ = [
    "AsyncEvalPaths",
    "AsyncEvaluationError",
    "AsyncEvaluationManager",
    "atomic_copy_file",
    "atomic_link_or_copy",
    "atomic_write_json",
    "process_is_alive",
    "restore_interrupted_jobs",
]
