from __future__ import annotations

import csv
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.hydra_config import ExtraArgument, HydraCliError, initialize_keyvalue_args
from runtime.paths import resolve_workspace_path


@dataclass(frozen=True)
class RescheduleAblationVariant:
    name: str
    warm_start_argument: str
    overrides: tuple[str, ...] = ()


@dataclass(frozen=True)
class CommandRow:
    variant: str
    instance_id: str
    seed: int
    run_id: str
    status: str
    warm_start_path: str
    command: str
    overrides: str


VARIANTS: dict[str, RescheduleAblationVariant] = {
    "full": RescheduleAblationVariant("full", "full_model_path"),
    "no_gat": RescheduleAblationVariant(
        "no_gat",
        "no_gat_model_path",
        ("graph_encoder_mode=none",),
    ),
    "no_attention": RescheduleAblationVariant(
        "no_attention",
        "no_attention_model_path",
        ("actor_context_mode=mean_max",),
    ),
}

# 这些变体不能被一键套件静默纳入。保留原因可让命令入口直接给出可审计的错误。
UNAVAILABLE_VARIANT_REASONS: dict[str, str] = {
    "no_mask": "当前 APAL 自回归解码无法在无可行性掩码下运行，不能作为有效实验。",
    "no_pointer": "需要重新设计合理的无 Pointer 解码器，当前实现不代表可解释消融。",
    "no_skill_hub": "用户已暂缓该消融，不能混入当前默认训练计划。",
    "no_attention_pooling": "与 no_attention 语义重复，请改用 no_attention。",
}


EXTRA_ARGUMENTS: dict[str, ExtraArgument] = {
    "mode": ExtraArgument(default="plan", help="plan 只生成命令；run 顺序执行命令"),
    "variants": ExtraArgument(
        default=["full", "no_gat", "no_attention"],
        help="消融变体列表",
    ),
    "instance_ids": ExtraArgument(default=["real_680"], help="manifest 中的验证实例列表"),
    "seeds": ExtraArgument(default=[42], help="随机种子列表"),
    "train_data_path_or_dir": ExtraArgument(
        default="data/generated/reschedule_train_400_600",
        help="与主方法一致的重调度训练随机实例目录",
    ),
    "manifest_path": ExtraArgument(
        default="data/reschedule_manifests/reschedule_400_600_seed20260701.json",
        help="固定 baseline schedule 与 low/medium/high 扰动场景 manifest",
    ),
    "max_episodes": ExtraArgument(default=300, help="训练 episode 数"),
    "batch_size": ExtraArgument(default=64, help="PPO batch size"),
    "eval_freq": ExtraArgument(default=1, help="验证与保存频率"),
    "num_envs": ExtraArgument(default=0, help="0 表示使用平台硬件配置；否则覆盖 num_envs"),
    "log_dir": ExtraArgument(default="/root/tf-logs", help="TensorBoard 根目录"),
    "runs_root": ExtraArgument(default="runs", help="运行目录根路径"),
    "artifact_layout": ExtraArgument(default="runs", help="统一运行目录布局"),
    "full_model_path": ExtraArgument(default="checkpoints/initial_schedule/680.ckpt", help="full warm start"),
    "no_gat_model_path": ExtraArgument(default="checkpoints/initial_schedule/680_no_gat.ckpt", help="no_gat warm start"),
    "no_attention_model_path": ExtraArgument(
        default="checkpoints/initial_schedule/680_no-attention-pooling.ckpt",
        help="no_attention warm start",
    ),
    "run_id_prefix": ExtraArgument(default="resched_ablation", help="run_id 前缀"),
    "output_dir": ExtraArgument(
        default="results/04_reschedule_baselines/reschedule_ablation_suite",
        help="命令清单输出目录",
    ),
    "python_executable": ExtraArgument(default="python", help="训练命令使用的 Python 可执行文件"),
    "extra_overrides": ExtraArgument(default=[], help="追加到每条训练命令的 Hydra 覆盖列表"),
    "validate_paths": ExtraArgument(default=True, help="生成命令前检查 manifest、训练目录和 warm start 路径"),
    "continue_on_error": ExtraArgument(default=False, help="mode=run 时单条失败后是否继续"),
}


def _as_list(value: Any, *, name: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return [item.strip() for item in text.split(",") if item.strip()]
    raise TypeError(f"{name} 必须是列表或逗号分隔字符串，收到: {value!r}")


def _bool_value(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    raise TypeError(f"{name} 必须为布尔值，收到: {value!r}")


def _resolve_existing(path_like: str, *, label: str) -> Path:
    path = resolve_workspace_path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"{label} 不存在: {path}")
    return path


def _load_manifest_instance_ids(manifest_path: Path) -> set[str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    instances = payload.get("instances", [])
    if not isinstance(instances, list):
        raise ValueError(f"manifest instances 字段必须是列表: {manifest_path}")
    return {str(item.get("instance_id", "")) for item in instances if isinstance(item, Mapping)}


def _command_string(tokens: Sequence[str]) -> str:
    return shlex.join([str(token) for token in tokens])


def _normalize_variants(raw_variants: Iterable[Any]) -> list[RescheduleAblationVariant]:
    variants: list[RescheduleAblationVariant] = []
    for raw in raw_variants:
        name = str(raw).strip()
        if name in UNAVAILABLE_VARIANT_REASONS:
            raise ValueError(
                f"重调度消融变体 {name} 当前不可执行：{UNAVAILABLE_VARIANT_REASONS[name]}"
            )
        if name not in VARIANTS:
            raise KeyError(
                f"未知重调度消融变体: {name}；当前可用: {sorted(VARIANTS)}；"
                f"暂不可用: {sorted(UNAVAILABLE_VARIANT_REASONS)}"
            )
        variants.append(VARIANTS[name])
    if not variants:
        raise ValueError("至少需要一个消融变体")
    return variants


def _normalize_seeds(raw_seeds: Iterable[Any]) -> list[int]:
    seeds = [int(seed) for seed in raw_seeds]
    if not seeds:
        raise ValueError("至少需要一个 seed")
    return seeds


def _variant_warm_start_path(args: Any, variant: RescheduleAblationVariant) -> str:
    value = getattr(args, variant.warm_start_argument)
    if not str(value).strip():
        raise ValueError(f"{variant.name} 缺少 warm start 路径字段: {variant.warm_start_argument}")
    return str(value)


def build_command_rows(args: Any) -> list[CommandRow]:
    variants = _normalize_variants(_as_list(getattr(args, "variants"), name="variants"))
    instance_ids = [str(item) for item in _as_list(getattr(args, "instance_ids"), name="instance_ids")]
    seeds = _normalize_seeds(_as_list(getattr(args, "seeds"), name="seeds"))
    extra_overrides = tuple(str(item) for item in _as_list(getattr(args, "extra_overrides"), name="extra_overrides"))
    if not instance_ids:
        raise ValueError("至少需要一个 instance_id")

    validate_paths = _bool_value(getattr(args, "validate_paths"), name="validate_paths")
    if validate_paths:
        manifest = _resolve_existing(str(getattr(args, "manifest_path")), label="重调度 manifest")
        _resolve_existing(str(getattr(args, "train_data_path_or_dir")), label="重调度训练数据")
        manifest_instances = _load_manifest_instance_ids(manifest)
        missing_instances = [item for item in instance_ids if item not in manifest_instances]
        if missing_instances:
            raise ValueError(f"manifest 中缺少实例: {missing_instances}")
        for variant in variants:
            _resolve_existing(_variant_warm_start_path(args, variant), label=f"{variant.name} warm start")

    rows: list[CommandRow] = []
    for variant in variants:
        warm_start_path = _variant_warm_start_path(args, variant)
        for instance_id in instance_ids:
            for seed in seeds:
                run_id = f"{getattr(args, 'run_id_prefix')}_{variant.name}_{instance_id}_seed{int(seed)}"
                tokens = [
                    str(getattr(args, "python_executable")),
                    "train.py",
                    "experiment=reschedule_task_delay",
                    f"train_data_path_or_dir={getattr(args, 'train_data_path_or_dir')}",
                    f"reschedule_manifest_path={getattr(args, 'manifest_path')}",
                    f"reschedule_eval_instance_id={instance_id}",
                    f"reschedule_baseline_model_path={warm_start_path}",
                    f"max_episodes={int(getattr(args, 'max_episodes'))}",
                    f"batch_size={int(getattr(args, 'batch_size'))}",
                    f"eval_freq={int(getattr(args, 'eval_freq'))}",
                    f"seed={int(seed)}",
                    f"run_id={run_id}",
                    f"log_dir={getattr(args, 'log_dir')}",
                    f"runs_root={getattr(args, 'runs_root')}",
                    f"artifact_layout={getattr(args, 'artifact_layout')}",
                ]
                if int(getattr(args, "num_envs")) > 0:
                    tokens.append(f"num_envs={int(getattr(args, 'num_envs'))}")
                tokens.extend(variant.overrides)
                tokens.extend(extra_overrides)
                rows.append(
                    CommandRow(
                        variant=variant.name,
                        instance_id=instance_id,
                        seed=int(seed),
                        run_id=run_id,
                        status="planned",
                        warm_start_path=warm_start_path,
                        command=_command_string(tokens),
                        overrides=" ".join((*variant.overrides, *extra_overrides)),
                    )
                )
    return rows


def write_plan(output_dir: Path, rows: Sequence[CommandRow], args: Any) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    row_dicts = [asdict(row) for row in rows]
    csv_path = output_dir / "reschedule_ablation_command_plan.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dicts[0].keys()) if row_dicts else [])
        if row_dicts:
            writer.writeheader()
            writer.writerows(row_dicts)

    (output_dir / "reschedule_ablation_command_plan.json").write_text(
        json.dumps(row_dicts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "reschedule_ablation_suite_config.json").write_text(
        json.dumps({key: getattr(args, key) for key in EXTRA_ARGUMENTS}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    script_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# 重调度消融命令清单。默认按固定 manifest、固定实例、固定 seed 顺序执行。",
        "export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}",
        "",
    ]
    for row in rows:
        script_lines.append(f"# variant={row.variant} instance={row.instance_id} seed={row.seed}")
        script_lines.append(row.command)
        script_lines.append("")
    (output_dir / "run_reschedule_ablation_suite.sh").write_text("\n".join(script_lines), encoding="utf-8")

    readme = [
        "# 重调度消融命令清单",
        "",
        "本目录由 `scripts/reschedule_ablation_suite.py` 生成。所有命令固定使用同一个重调度 manifest、同一个训练数据目录、同一套低/中/高扰动场景，并按 `variant -> instance_id -> seed` 的稳定顺序排列。",
        "",
        "## 文件",
        "",
        "- `reschedule_ablation_command_plan.csv`：命令清单。",
        "- `reschedule_ablation_command_plan.json`：结构化命令清单。",
        "- `reschedule_ablation_suite_config.json`：生成命令时使用的脚本参数。",
        "- `run_reschedule_ablation_suite.sh`：可在 Linux 服务器逐条执行的脚本。",
        "",
        "## 可复现性约束",
        "",
        "- 每条命令显式指定 `seed`、`run_id`、`reschedule_manifest_path`、`reschedule_eval_instance_id` 和对应 warm start checkpoint。",
        "- `eval_freq=1`，每个 episode 都执行重调度验证与 best checkpoint 选择。",
        "- 不在脚本内随机打乱命令顺序。",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


def run_commands(rows: Sequence[CommandRow], *, continue_on_error: bool) -> int:
    env = os.environ.copy()
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    failures = 0
    for index, row in enumerate(rows, start=1):
        print(
            f"[RescheduleAblation] {index}/{len(rows)} "
            f"variant={row.variant} instance={row.instance_id} seed={row.seed}",
            flush=True,
        )
        result = subprocess.run(shlex.split(row.command), cwd=PROJECT_ROOT, env=env, check=False)
        if result.returncode != 0:
            failures += 1
            print(
                f"[RescheduleAblation] failed variant={row.variant} "
                f"instance={row.instance_id} seed={row.seed} code={result.returncode}",
                flush=True,
            )
            if not continue_on_error:
                return result.returncode
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    try:
        args = initialize_keyvalue_args(argv, extra_arguments=EXTRA_ARGUMENTS)
        mode = str(args.mode).strip().lower()
        if mode not in {"plan", "run"}:
            raise ValueError("mode 只能是 plan 或 run")
        rows = build_command_rows(args)
        output_dir = resolve_workspace_path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        write_plan(output_dir, rows, args)
        print(f"[RescheduleAblation] mode={mode} commands={len(rows)} output={output_dir}", flush=True)
        if mode == "run":
            return run_commands(rows, continue_on_error=_bool_value(args.continue_on_error, name="continue_on_error"))
        return 0
    except (HydraCliError, KeyError, ValueError, TypeError, FileNotFoundError) as exc:
        print(f"[RescheduleAblation][Error] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
