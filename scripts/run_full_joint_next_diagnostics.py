"""运行 Full Joint 后续诊断套件；默认串行以避免单 GPU 显存竞争。"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import os
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.next_suite import (
    DiagnosticJob,
    build_train_command,
    jobs_for_suite,
)


def _launcher_root() -> Path:
    return PROJECT_ROOT / "diagnostics" / "full_joint_next" / "launcher"


def _command_text(command: list[str]) -> str:
    return " ".join(command)


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _run_job(
    job: DiagnosticJob,
    *,
    max_episodes: int,
    num_envs: int,
    batch_size: int,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> int:
    command = build_train_command(
        job,
        max_episodes=max_episodes,
        num_envs=num_envs,
        batch_size=batch_size,
    )
    log_path = _launcher_root() / f"{job.run_name}.log"
    env = dict(os.environ)
    env.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONHASHSEED": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    started = datetime.now().astimezone().isoformat()
    result = {
        "suite": job.suite,
        "variant": job.variant,
        "seed": job.seed,
        "run_name": job.run_name,
        "command": command,
        "command_text": _command_text(command),
        "log_path": str(log_path.resolve()),
        "started_at": started,
        "status": "running",
    }
    manifest.setdefault("jobs", []).append(result)
    _write_manifest(manifest_path, manifest)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[启动] {started}\n[命令] {_command_text(command)}\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    result.update(
        {
            "returncode": int(completed.returncode),
            "status": "passed" if completed.returncode == 0 else "failed",
            "finished_at": datetime.now().astimezone().isoformat(),
        }
    )
    _write_manifest(manifest_path, manifest)
    return int(completed.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 Full Joint T5/T6 诊断任务")
    parser.add_argument("--suite", choices=("t5", "t6", "all"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--max-episodes", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="只打印命令和任务清单，不启动训练",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    jobs = jobs_for_suite(args.suite, args.seeds)
    commands = [
        build_train_command(
            job,
            max_episodes=args.max_episodes,
            num_envs=args.num_envs,
            batch_size=args.batch_size,
        )
        for job in jobs
    ]
    if args.plan_only:
        print(
            json.dumps(
                {
                    "suite": args.suite,
                    "jobs": [
                        {
                            "run_name": job.run_name,
                            "variant": job.variant,
                            "seed": job.seed,
                            "command": command,
                        }
                        for job, command in zip(jobs, commands)
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    manifest_path = _launcher_root() / f"{args.suite}_manifest.json"
    manifest: dict[str, Any] = {
        "suite": args.suite,
        "created_at": datetime.now().astimezone().isoformat(),
        "project_root": str(PROJECT_ROOT.resolve()),
        "max_episodes": int(args.max_episodes),
        "num_envs": int(args.num_envs),
        "batch_size": int(args.batch_size),
        "max_parallel": 1,
        "jobs": [],
    }
    _write_manifest(manifest_path, manifest)
    for job in jobs:
        returncode = _run_job(
            job,
            max_episodes=args.max_episodes,
            num_envs=args.num_envs,
            batch_size=args.batch_size,
            manifest=manifest,
            manifest_path=manifest_path,
        )
        if returncode != 0:
            manifest["failed"] = True
            _write_manifest(manifest_path, manifest)
            return returncode
    manifest["failed"] = False
    manifest["finished_at"] = datetime.now().astimezone().isoformat()
    _write_manifest(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
