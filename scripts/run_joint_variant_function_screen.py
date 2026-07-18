from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import Config
from models.hb_gat_pn import HBGATPN
from runtime.hydra_config import ExtraArgument, initialize_hydra_runtime, initialize_keyvalue_args


VARIANTS: dict[str, tuple[str, ...]] = {
    "full_joint": (),
    "operation_station": ("policy_action_scope=operation_station",),
    "operation_only": ("policy_action_scope=operation",),
    "fixed_preallocation": (
        "workforce_binding_mode=preallocated",
        "workforce_preallocation_ratio=1.0",
    ),
    "static_topq": ("team_selection_mode=static_topq",),
    "homogeneous_graphsage": ("graph_encoder_mode=homogeneous_graphsage",),
    "no_message_passing": ("graph_encoder_mode=none",),
    "mean_max_pooling": ("actor_context_mode=mean_max",),
    "local_only": ("actor_context_mode=local_only",),
}


@dataclass(frozen=True)
class ScreenResult:
    variant: str
    status: str
    return_code: int
    elapsed_sec: float
    rollout_updates: int
    validation_complete: bool
    latest_checkpoint_exists: bool
    best_checkpoint_exists: bool
    total_parameters: int
    trainable_parameters: int
    run_id: str
    run_dir: str
    log_path: str
    command: str
    evaluation_makespan: float | None = None
    mean_rollout_sps: float | None = None


def _parse_log_metrics(text: str) -> tuple[int, bool, float | None, float | None]:
    rollout_matches = re.findall(
        r"\[Rollout\]\s+ep=\d+.*?SPS=([0-9.]+)",
        text,
    )
    eval_matches = re.findall(r"\[Eval\]\s+ep=3\s+Mk=([0-9.]+)", text)
    rollout_updates = len(rollout_matches)
    mean_sps = (
        sum(float(value) for value in rollout_matches) / rollout_updates
        if rollout_updates
        else None
    )
    eval_makespan = float(eval_matches[-1]) if eval_matches else None
    return rollout_updates, bool(eval_matches), eval_makespan, mean_sps


EXTRA_ARGUMENTS = {
    "mode": ExtraArgument(default="run", help="plan 仅生成命令，run 顺序执行筛查"),
    "variants": ExtraArgument(default=list(VARIANTS), help="要筛查的变体列表"),
    "seed": ExtraArgument(default=42, help="统一随机种子"),
    "output_dir": ExtraArgument(
        default="results/functional_screen_joint_variants",
        help="汇总与逐变体日志目录",
    ),
}


def _normalize_variants(raw: object) -> list[str]:
    if raw is None:
        values = []
    elif isinstance(raw, str):
        values = [value.strip() for value in raw.split(",") if value.strip()]
    elif isinstance(raw, (list, tuple)):
        values = [str(value) for value in raw]
    else:
        raise TypeError("variants 必须是列表或逗号分隔字符串")
    unknown = sorted(set(values) - set(VARIANTS))
    if unknown:
        raise ValueError(f"未知筛查变体: {unknown}")
    return values


def _parameter_counts(overrides: Sequence[str]) -> tuple[int, int]:
    cfg = Config()
    initialize_hydra_runtime(
        [
            "experiment=joint_variant_function_screen",
            *overrides,
        ],
        target=cfg,
        project_root=PROJECT_ROOT,
        create_run_context=False,
    )
    model = HBGATPN(cfg)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return int(total), int(trainable)


def build_command(
    *,
    variant: str,
    seed: int,
    run_id: str,
) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "train_lightning.py"),
        "experiment=joint_variant_function_screen",
        f"seed={int(seed)}",
        f"run_id={run_id}",
        *VARIANTS[variant],
    ]


def _run_one(
    *,
    variant: str,
    seed: int,
    batch_id: str,
    output_dir: Path,
) -> ScreenResult:
    run_id = f"screen_{batch_id}_{variant}_seed{seed}"
    command = build_command(
        variant=variant,
        seed=seed,
        run_id=run_id,
    )
    log_path = output_dir / "logs" / f"{variant}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    started = time.perf_counter()
    lines: list[str] = []

    def write_console(line: str) -> None:
        encoding = sys.stdout.encoding or "utf-8"
        safe_line = line.encode(encoding, errors="replace").decode(
            encoding,
            errors="replace",
        )
        print(f"[{variant}] {safe_line}", end="", flush=True)

    with log_path.open("w", encoding="utf-8", newline="") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line)
            log_file.write(line)
            log_file.flush()
            write_console(line)
        return_code = int(process.wait())
    elapsed = time.perf_counter() - started
    text = "".join(lines)
    rollout_updates, validation_ran, eval_makespan, mean_sps = _parse_log_metrics(text)
    run_dir = PROJECT_ROOT / "runs" / "joint_variant_function_screen" / run_id
    latest = run_dir / "checkpoints" / "last.ckpt"
    best = run_dir / "checkpoints" / "best.ckpt"
    # 标准验证只有在完整排程时才允许保存 best checkpoint。
    validation_complete = validation_ran and best.is_file()
    total_parameters, trainable_parameters = _parameter_counts(
        VARIANTS[variant],
    )
    passed = (
        return_code == 0
        and rollout_updates == 3
        and validation_complete
        and latest.is_file()
        and best.is_file()
    )
    return ScreenResult(
        variant=variant,
        status="passed" if passed else "failed",
        return_code=return_code,
        elapsed_sec=float(elapsed),
        rollout_updates=rollout_updates,
        validation_complete=validation_complete,
        latest_checkpoint_exists=latest.is_file(),
        best_checkpoint_exists=best.is_file(),
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        run_id=run_id,
        run_dir=str(run_dir.resolve()),
        log_path=str(log_path.resolve()),
        command=subprocess.list2cmdline(command),
        evaluation_makespan=eval_makespan,
        mean_rollout_sps=mean_sps,
    )


def _write_results(output_dir: Path, results: Sequence[ScreenResult]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    (output_dir / "screen_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "screen_results.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(ScreenResult.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = initialize_keyvalue_args(argv, extra_arguments=EXTRA_ARGUMENTS)
    variants = _normalize_variants(args.variants)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    batch_id = datetime.now().strftime("%y%m%d-%H%M%S")
    plan_rows = [
        {
            "variant": variant,
            "command": subprocess.list2cmdline(
                build_command(
                    variant=variant,
                    seed=int(args.seed),
                    run_id=f"screen_{batch_id}_{variant}_seed{int(args.seed)}",
                )
            ),
        }
        for variant in variants
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "screen_plan.json").write_text(
        json.dumps(plan_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if str(args.mode).lower() == "plan":
        return 0
    if str(args.mode).lower() != "run":
        raise ValueError("mode 仅允许 plan 或 run")

    results_path = output_dir / "screen_results.json"
    results: list[ScreenResult] = []
    if results_path.is_file():
        existing_rows = json.loads(results_path.read_text(encoding="utf-8"))
        for row in existing_rows:
            if row.get("variant") in variants:
                continue
            existing = ScreenResult(**row)
            best_exists = Path(existing.run_dir, "checkpoints", "best.ckpt").is_file()
            latest_exists = Path(existing.run_dir, "checkpoints", "last.ckpt").is_file()
            log_text = Path(existing.log_path).read_text(
                encoding="utf-8",
                errors="replace",
            )
            rollout_updates, validation_ran, eval_makespan, mean_sps = _parse_log_metrics(
                log_text
            )
            passed = (
                existing.return_code == 0
                and rollout_updates == 3
                and validation_ran
                and best_exists
                and latest_exists
            )
            results.append(
                ScreenResult(
                    **{
                        **asdict(existing),
                        "status": "passed" if passed else existing.status,
                        "validation_complete": validation_ran and best_exists,
                        "rollout_updates": rollout_updates,
                        "latest_checkpoint_exists": latest_exists,
                        "best_checkpoint_exists": best_exists,
                        "evaluation_makespan": eval_makespan,
                        "mean_rollout_sps": mean_sps,
                    }
                )
            )
    for variant in variants:
        result = _run_one(
            variant=variant,
            seed=int(args.seed),
            batch_id=batch_id,
            output_dir=output_dir,
        )
        results.append(result)
        _write_results(output_dir, results)
    _write_results(output_dir, results)
    return 0 if all(result.status == "passed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
