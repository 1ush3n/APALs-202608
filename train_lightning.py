from __future__ import annotations

import os
# 启用可扩展显存段以缓解动态图 GNN 变长 batch 的碎片化；峰值显存仍由 batch 控制。
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import copy
import json
import math
import platform
import re
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger

from configs import configs
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from runtime.configuration import initialize_training_config
from runtime.paths import (
    resolve_checkpoint_paths,
    resolve_tensorboard_log_root,
    resolve_workspace_path,
    sanitize_experiment_name,
)
from runtime.reschedule_eval import (
    ensure_reschedule_baseline_available,
    ensure_reschedule_eval_scenarios_available,
    load_warm_start_weights_with_input_expansion,
)
from runtime.reschedule_manifest import resolve_manifest_eval_entry
from runtime.reschedule_manifest import (
    load_reschedule_manifest,
    resolve_explicit_five_skill_training_paths,
)
from runtime.training_data_manifest import resolve_explicit_five_skill_initial_training_paths
from utils.reschedule import load_reschedule_scenarios
from runtime.seed import set_seed
from training.lightning_module import APALDataModule, APALLightningModule
from training.rollout_service import APALRolloutService
from training.async_evaluation import AsyncEvaluationManager
from utils.vector_env import EnvCreator, VectorEnv
from runtime.artifacts import run_context as create_run_context, uses_runs_layout, write_run_context_files, write_run_manifest
from runtime.checkpoints import apply_checkpoint_model_spec, load_checkpoint, load_policy_weights
from runtime.initial_checkpoint_selection import load_initial_checkpoint_selection_manifest
from runtime.initial_worker_mapping import apply_initial_worker_mapping
from runtime.schedulefree_checkpoint import save_checkpoint_with_schedulefree_eval_parameters


PROJECT_ROOT = Path(__file__).resolve().parent
_APCF_SCOPE = "operation_station_anchor_proposal_team"
_APCF_MANIFEST_KIND = "initial_anchor_proposal_counterfactual_v1"


def _sha256_file(path: Path) -> str:
    """计算文件 SHA-256（供 APCF 预训练 manifest 可追溯校验）。"""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_apcf_regular_file(field_name: str) -> Path:
    """解析 APCF 必填文件；空值、目录和空文件均立即拒绝。"""
    raw_path = str(getattr(configs, field_name, "") or "").strip()
    if not raw_path:
        raise ValueError(f"APCF 必须显式配置 {field_name}，不得为空")
    path = resolve_workspace_path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"APCF 配置 {field_name} 指向的文件不存在：{path}")
    if not path.is_file():
        raise ValueError(f"APCF 配置 {field_name} 必须指向普通文件，实际为：{path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"APCF 配置 {field_name} 指向零字节文件：{path}")
    return path


def _load_apcf_manifest_semantics() -> tuple[Path, str]:
    """读取并校验本次 APCF 训练唯一允许使用的反事实 manifest。"""
    manifest_path = _require_apcf_regular_file("anchor_proposal_cf_manifest_path")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"APCF 反事实 manifest 无法解析：{manifest_path}") from exc
    if not isinstance(payload, dict) or payload.get("kind") != _APCF_MANIFEST_KIND:
        raise ValueError(
            "APCF 反事实 manifest 类型不兼容："
            f"要求 kind={_APCF_MANIFEST_KIND!r}，文件={manifest_path}"
        )
    return manifest_path, _sha256_file(manifest_path)


def _require_sha256(value: object, *, label: str) -> str:
    digest = str(value or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise RuntimeError(f"APCF {label} 缺少合法 manifest_sha256")
    return digest


def _validate_apcf_model_spec(model_spec: Any, *, manifest_sha256: str, phase: str) -> None:
    """APCF 冷启动或续训均使用同一组不可降级的模型语义校验。"""
    checkpoint_scope = str(getattr(model_spec, "policy_action_scope", ""))
    if checkpoint_scope != _APCF_SCOPE:
        raise RuntimeError(
            f"APCF {phase} scope 不兼容：checkpoint={checkpoint_scope}，要求={_APCF_SCOPE}"
        )
    checkpoint_mode = str(getattr(model_spec, "anchor_proposal_mode", "") or "")
    configured_mode = str(getattr(configs, "anchor_proposal_mode", "") or "")
    if checkpoint_mode != "full_team_v1" or checkpoint_mode != configured_mode:
        raise RuntimeError(
            "APCF proposal mode 不兼容："
            f"checkpoint={checkpoint_mode!r}，config={configured_mode!r}"
        )
    checkpoint_require_difference = getattr(
        model_spec, "anchor_proposal_require_difference", None
    )
    configured_require_difference = bool(
        getattr(configs, "anchor_proposal_require_difference", True)
    )
    if (
        checkpoint_require_difference is None
        or bool(checkpoint_require_difference) != configured_require_difference
    ):
        raise RuntimeError(
            "APCF require_difference 语义不兼容："
            f"checkpoint={checkpoint_require_difference!r}，"
            f"config={configured_require_difference!r}"
        )
    checkpoint_manifest_sha256 = _require_sha256(
        getattr(model_spec, "anchor_proposal_cf_manifest_sha256", None),
        label=f"{phase} checkpoint",
    )
    if checkpoint_manifest_sha256 != manifest_sha256:
        raise RuntimeError(
            "APCF 反事实 manifest SHA-256 不一致："
            f"checkpoint={checkpoint_manifest_sha256}，config={manifest_sha256}"
        )


def _save_rollout_checkpoint(
    trainer: pl.Trainer,
    pl_module: APALLightningModule,
    path: Path,
) -> None:
    """以与训练期评估一致的参数态保存 Lightning checkpoint。"""
    def _save(target: Path) -> None:
        trainer.save_checkpoint(str(target))
        _ensure_checkpoint_metadata(Path(target), pl_module)
    agent = getattr(pl_module, "agent", None)
    if agent is None:
        _save(path)
        return
    state = save_checkpoint_with_schedulefree_eval_parameters(
        save_checkpoint=_save,
        path=Path(path),
        optimizer=agent.optimizer,
        schedulefree_enabled=bool(getattr(agent, "use_schedule_free", False)),
    )
    if state.source_mode != "disabled":
        print(
            "[Checkpoint][ScheduleFree] "
            f"source={state.source_mode} saved={state.saved_mode} restored={state.restored_mode}",
            flush=True,
        )


def _ensure_checkpoint_metadata(path: Path, pl_module: APALLightningModule) -> None:
    """?? callback ??????????? Lightning/APAL metadata?"""
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = payload.get("apal_metadata") if isinstance(payload, dict) else None
    model_spec = metadata.get("model_spec") if isinstance(metadata, dict) else None
    if isinstance(metadata, dict) and isinstance(model_spec, dict):
        return
    if not isinstance(payload, dict):
        raise TypeError(f"Lightning checkpoint payload ??? dict: {checkpoint_path}")
    pl_module.on_save_checkpoint(payload)
    repaired_metadata = payload.get("apal_metadata")
    if not isinstance(repaired_metadata, dict) or not isinstance(
        repaired_metadata.get("model_spec"), dict
    ):
        raise RuntimeError(f"???? APAL checkpoint metadata: {checkpoint_path}")
    torch.save(payload, checkpoint_path)
def _resume_start_episode(checkpoint_payload: object) -> int:
    """校验已保存 PPO episode 与 Lightning 循环进度，并返回下一轮绝对 episode。"""
    if not isinstance(checkpoint_payload, dict):
        raise TypeError("续训 checkpoint payload 必须为字典")
    metadata = checkpoint_payload.get("apal_metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("续训 checkpoint 缺少 apal_metadata")
    completed_episode = int(metadata.get("episode", 0))
    if completed_episode < 0:
        raise ValueError(f"续训 checkpoint 的 episode 非法: {completed_episode}")

    loops = checkpoint_payload.get("loops", {})
    fit_loop = loops.get("fit_loop", {}) if isinstance(loops, dict) else {}
    progress = (
        fit_loop.get("epoch_loop.batch_progress", {})
        if isinstance(fit_loop, dict)
        else {}
    )
    total_progress = progress.get("total", {}) if isinstance(progress, dict) else {}
    batch_completed = total_progress.get("completed") if isinstance(total_progress, dict) else None
    if batch_completed is not None:
        expected_episode = int(batch_completed) + 1
        if completed_episode != expected_episode:
            raise ValueError(
                "续训 checkpoint 的 APAL episode 与 Lightning batch 进度不一致；"
                f"apal_episode={completed_episode}, lightning_expected={expected_episode}。"
                "该 checkpoint 可能来自旧版错误续训，拒绝继续以避免训练、异步评估和 checkpoint 标签错位。"
            )
    return completed_episode + 1


def _apply_reschedule_eval_manifest_override() -> None:
    """当 manifest 指定自动验证实例时，将验证数据、baseline 和场景路径同步到配置。"""

    if not bool(getattr(configs, "enable_reschedule_mode", False)):
        return
    entry = resolve_manifest_eval_entry(configs)
    if entry is None:
        return
    configs.data_file_path = str(entry.data_path)
    configs.reschedule_baseline_schedule_path = str(entry.baseline_schedule_path)
    if entry.scenario_path is not None:
        configs.reschedule_eval_scenario_path = str(entry.scenario_path)
    # 重调度真实实例沿用初始调度的固定工人数；否则 2338/3182 的基准团队
    # 会在较小工人池中被错误判定为越界。
    apply_initial_worker_mapping(configs, entry.data_path, explicit_fields=set())
    print(
        f"[Reschedule] eval_instance={entry.instance_id} "
        f"data={entry.data_path} baseline={entry.baseline_schedule_path} "
        f"scenarios={entry.scenario_path}",
        flush=True,
    )


def _guard_resume_scope_against_apcf(model_spec: Any) -> None:
    """resume 时禁止旧 scope checkpoint 静默把 APCF 配置降级为旧 scope。

    配置显式要求 operation_station_anchor_proposal_team 时，续训 checkpoint 必须
    同为 APCF scope；否则立即报错（防"静默降级 + 随机双头继续训练"）。
    """
    configured_scope = str(getattr(configs, "policy_action_scope", ""))
    if configured_scope != _APCF_SCOPE:
        return
    checkpoint_scope = str(getattr(model_spec, "policy_action_scope", ""))
    if checkpoint_scope != _APCF_SCOPE:
        raise RuntimeError(
            "resume 时检测到 checkpoint scope 与当前 APCF 配置不兼容："
            f"checkpoint={checkpoint_scope}，config={configured_scope}。"
            "APCF 实验必须续训 APCF scope 的 checkpoint；请勿混用旧 scope checkpoint。"
        )
    _manifest_path, manifest_sha256 = _load_apcf_manifest_semantics()
    _validate_apcf_model_spec(
        model_spec,
        manifest_sha256=manifest_sha256,
        phase="resume",
    )


def _maybe_load_apcf_pretrain(model: torch.nn.Module, device: torch.device, *, resume: bool) -> None:
    """APCF 正式训练必须从反事实预训练 checkpoint 冷启动。

    - resume=True 时由 Lightning checkpoint 恢复，不重复加载；
    - 校验 checkpoint 的 model_spec 必须为 APCF scope；
    - 校验 apal_pretrain_metadata.manifest_sha256 与当前配置指向的
      反事实 manifest 实际 SHA-256 一致（可追溯闭环的安全约束）。
    """
    if resume:
        return
    scope = str(getattr(configs, "policy_action_scope", ""))
    if scope != _APCF_SCOPE:
        return
    manifest_path, manifest_sha256 = _load_apcf_manifest_semantics()
    pretrain_path = _require_apcf_regular_file(
        "anchor_proposal_pretrain_checkpoint_path"
    )
    checkpoint = load_checkpoint(pretrain_path)
    _validate_apcf_model_spec(
        checkpoint.model_spec,
        manifest_sha256=manifest_sha256,
        phase="预训练",
    )
    pretrain_metadata = (
        checkpoint.payload.get("apal_pretrain_metadata", {})
        if isinstance(checkpoint.payload, dict)
        else {}
    )
    if not isinstance(pretrain_metadata, dict):
        raise RuntimeError("APCF 预训练 checkpoint 缺少 apal_pretrain_metadata")
    recorded_manifest_sha256 = _require_sha256(
        pretrain_metadata.get("manifest_sha256"),
        label="预训练 checkpoint",
    )
    if recorded_manifest_sha256 != manifest_sha256:
        raise RuntimeError(
            "APCF 预训练 checkpoint 的反事实 manifest SHA-256 不一致："
            f"checkpoint={recorded_manifest_sha256}，当前配置={manifest_sha256}（{manifest_path}）"
        )
    load_policy_weights(model, checkpoint, strict=True)
    configs.apcf_pretrain_loaded_model_key_count = len(checkpoint.state_dict)
    configs.anchor_proposal_pretrain_source_sha256 = _sha256_file(pretrain_path)
    print(
        f"[APCF-Pretrain] 已加载预训练 checkpoint {pretrain_path} "
        f"(manifest_sha256={recorded_manifest_sha256[:12]}...)，"
        "正式 PPO 从预训练权重微调。",
        flush=True,
    )


def _maybe_load_reschedule_warm_start(model: torch.nn.Module, device: torch.device, *, resume: bool) -> None:
    """重调度训练默认从初始调度策略 warm start；续训时由 Lightning checkpoint 恢复。"""

    if resume:
        return
    if not bool(getattr(configs, "enable_reschedule_mode", False)):
        return
    if not bool(getattr(configs, "reschedule_warm_start", True)):
        print("[Reschedule] warm_start=false，重调度模型将随机初始化。", flush=True)
        return
    model_path = resolve_workspace_path(
        getattr(configs, "reschedule_baseline_model_path", "checkpoints/initial_schedule/bestmodel/best_model.pth")
    )
    if not model_path.exists():
        raise FileNotFoundError(f"重调度 warm start 初始调度模型不存在: {model_path}")
    stats = load_warm_start_weights_with_input_expansion(model, model_path, device)
    loaded = int(stats.get("loaded_exact", 0)) + int(stats.get("loaded_expanded", 0))
    if loaded <= 0:
        raise RuntimeError(f"重调度 warm start 未加载到任何可用权重: {model_path}")
    print(
        "[Reschedule] warm_start="
        f"{model_path} loaded_exact={int(stats.get('loaded_exact', 0))} "
        f"loaded_expanded={int(stats.get('loaded_expanded', 0))} "
        f"skipped={int(stats.get('skipped', 0))}",
        flush=True,
    )


def _validate_async_eval_target() -> None:
    """在启动训练环境前验证异步 best 选择所需的固定评估目标。"""
    if not bool(getattr(configs, "async_eval_enabled", False)):
        return
    if not bool(getattr(configs, "enable_reschedule_mode", False)):
        protocol = str(
            getattr(configs, "checkpoint_selection_protocol", "single_standard")
        ).strip().lower()
        if protocol == "multiscale_manifest":
            manifest = load_initial_checkpoint_selection_manifest(
                configs.checkpoint_selection_manifest_path
            )
            targets = ", ".join(
                f"{entry.instance_id}:{entry.data_path.name}" for entry in manifest.entries
            )
            print(
                f"[AsyncEval] target=initial_multi_benchmark "
                f"protocol={manifest.protocol_id} role={manifest.role} "
                f"instances=[{targets}] seed={manifest.seed}",
                flush=True,
            )
            return
        data_path = resolve_workspace_path(configs.async_eval_initial_data_path)
        if not data_path.is_file():
            raise FileNotFoundError(f"初始调度异步验证数据不存在: {data_path}")
        print(
            f"[AsyncEval] target=initial_standard data={data_path} seed={int(configs.seed)}",
            flush=True,
        )
        return

    manifest_path = str(getattr(configs, "reschedule_manifest_path", "") or "").strip()
    if not manifest_path:
        raise ValueError("开启异步验证时必须配置 reschedule_manifest_path")
    manifest = load_reschedule_manifest(manifest_path)
    instance_id = str(configs.async_eval_instance_id)
    scenario_id = str(configs.async_eval_scenario_id)
    entry = manifest.get(instance_id)
    if entry.scenario_path is None or not entry.scenario_path.is_file():
        raise FileNotFoundError(f"异步验证实例缺少场景文件: {instance_id}")
    scenario_ids = [str(name) for name, _scenario in load_reschedule_scenarios(entry.scenario_path)]
    if scenario_id not in scenario_ids:
        raise KeyError(f"异步验证场景不存在: {instance_id}/{scenario_id}")
    source_index = scenario_ids.index(scenario_id)
    reset_seed = int(configs.reschedule_eval_scenario_seed) + source_index
    print(
        f"[AsyncEval] target={instance_id}/{scenario_id} "
        f"scenario_index={source_index} reset_seed={reset_seed}",
        flush=True,
    )


class RolloutCheckpoint(Callback):
    """按 PPO rollout 更新保存最新模型，并按验证 Makespan 保存最佳模型。"""

    def __init__(self, latest_path: Path, best_path: Path | None = None) -> None:
        super().__init__()
        if best_path is None:
            checkpoint_dir = Path(latest_path)
            self.latest_path = checkpoint_dir / "last.ckpt"
            self.best_path = checkpoint_dir / "best" / "best.ckpt"
        else:
            self.latest_path = Path(latest_path)
            self.best_path = Path(best_path)
        self.best_score = float("inf")
        self.async_manager = (
            AsyncEvaluationManager(
                config=configs,
                latest_path=self.latest_path,
                best_path=self.best_path,
                project_root=PROJECT_ROOT,
            )
            if bool(getattr(configs, "async_eval_enabled", False))
            else None
        )

    @property
    def state_key(self) -> str:
        return "RolloutCheckpoint"

    def state_dict(self) -> dict[str, float]:
        return {"best_score": float(self.best_score)}

    def load_state_dict(self, state_dict: dict[str, float]) -> None:
        self.best_score = float(
            state_dict.get("best_score", state_dict.get("best_makespan", float("inf")))
        )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: APALLightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if not bool(getattr(pl_module, "last_update_committed", True)):
            return
        self.latest_path.parent.mkdir(parents=True, exist_ok=True)
        self.best_path.parent.mkdir(parents=True, exist_ok=True)
        episode = int(pl_module.last_completed_episode)
        eval_metrics = pl_module.last_eval_metrics

        if self.async_manager is not None:
            submit_every = max(
                1,
                int(getattr(configs, "async_eval_submit_every_episodes", 1)),
            )
            if episode % submit_every == 0:
                self.async_manager.submit(trainer, episode=episode)
            else:
                _save_rollout_checkpoint(trainer, pl_module, self.latest_path)
            return

        if eval_metrics is not None:
            makespan = float(eval_metrics["makespan"])
            is_multi_benchmark = "multi_benchmark_selection_score" in eval_metrics
            is_reschedule = "reschedule_selection_score" in eval_metrics
            if is_reschedule:
                current_score = float(eval_metrics["reschedule_selection_score"])
                eligible = bool(eval_metrics.get("reschedule_eligible_rate", 0.0) >= 1.0 - 1e-9)
                metric_name = "reschedule_selection_score"
            elif is_multi_benchmark:
                current_score = float(eval_metrics["multi_benchmark_selection_score"])
                eligible = bool(eval_metrics.get("multi_benchmark_eligible", 1.0) >= 1.0 - 1e-9)
                metric_name = "multi_benchmark_normalized_makespan"
            else:
                current_score = makespan
                eligible = bool(eval_metrics.get("completion_rate", 0.0) >= 1.0 - 1e-9)
                metric_name = "eval_makespan"
            if eligible and current_score < self.best_score:
                self.best_score = current_score
                _save_rollout_checkpoint(trainer, pl_module, self.best_path)
                
                # 多尺度评估时，打印出各个子数据集的具体完工时间明细，避免日志只显示单一数据集产生误导
                if is_multi_benchmark:
                    mk_details = ", ".join(
                        f"{k.split('_')[2]}:{v:.1f}"
                        for k, v in eval_metrics.items()
                        if k.startswith("multi_benchmark_") and k.endswith("_makespan")
                    )
                    mk_str = f"Mks=[{mk_details}]"
                elif is_reschedule:
                    mk_str = (
                        f"score={float(eval_metrics.get('reschedule_composite_score', current_score)):.6f} "
                        f"elig={float(eval_metrics.get('reschedule_eligible_rate', 0.0)):.2f} "
                        f"Mk={makespan:.2f}"
                    )
                else:
                    mk_str = f"Mk={makespan:.2f}"

                marker = "N" * 20
                print(
                    f"\n{marker} [发现新的最佳模型] {marker}\n"
                    f"[Checkpoint] ep={episode} metric={metric_name} "
                    f"score={current_score:.6f} {mk_str}\n"
                    f"[Checkpoint] best_path={self.best_path}\n"
                    f"{marker} [最佳模型已保存] {marker}\n",
                    flush=True,
                )

        _save_rollout_checkpoint(trainer, pl_module, self.latest_path)
        print(
            f"[Checkpoint] ep={episode} 保存最新模型: path={self.latest_path}",
            flush=True,
        )

    def on_fit_end(self, trainer: pl.Trainer, pl_module: APALLightningModule) -> None:
        if self.async_manager is not None:
            self.async_manager.finalize(
                wait=bool(getattr(configs, "async_eval_wait_on_finish", True))
            )

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: APALLightningModule,
        exception: BaseException,
    ) -> None:
        if self.async_manager is not None:
            self.async_manager.terminate_for_exception()


def record_apcf_pretrain_load(
    manifest_path: Path,
    *,
    source_sha256: str,
    loaded_model_key_count: int,
) -> None:
    if not str(source_sha256).strip():
        raise ValueError("source_sha256 ?????APCF ??????????")
    if int(loaded_model_key_count) <= 0:
        raise ValueError("loaded_model_key_count ??????APCF ??????????")
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"run_manifest ??? JSON ???{manifest_path}")
    payload["apcf_pretrain_loaded"] = True
    payload["apcf_pretrain_source_sha256"] = str(source_sha256)
    payload["apcf_pretrain_loaded_model_key_count"] = int(loaded_model_key_count)
    Path(manifest_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _validate_apcf_smoke_output_paths(checkpoint_paths: dict[str, Path]) -> None:
    raw_root = str(getattr(configs, "apcf_smoke_guard_root", "") or "").strip()
    if not raw_root:
        return
    root = resolve_workspace_path(raw_root).resolve()
    if root.parent.name != ".pytest_tmp" or not root.name.startswith("apcf_ppo_smoke_s42_"):
        raise ValueError(f"APCF smoke ??????? .pytest_tmp ??{root}")
    from runtime.artifacts import assert_apcf_smoke_output_isolated
    log_root = resolve_tensorboard_log_root(configs).resolve()
    output_paths = [
        Path(checkpoint_paths["model_dir"]),
        Path(checkpoint_paths["lightning_dir"]),
        Path(checkpoint_paths["lightning_latest"]),
        log_root,
    ]
    assert_apcf_smoke_output_isolated(root, output_paths)
def run(args, *, config_initialized: bool = False) -> None:
    if not config_initialized:
        initialize_training_config(args)
    set_seed(int(configs.seed))
    _apply_reschedule_eval_manifest_override()
    _validate_async_eval_target()

    checkpoint_paths = resolve_checkpoint_paths(configs)
    _validate_apcf_smoke_output_paths(checkpoint_paths)
    checkpoint_dir = checkpoint_paths["lightning_dir"]
    start_episode = 1
    if args.resume:
        resume_path = checkpoint_paths["lightning_latest"]
        if not resume_path.exists():
            raise FileNotFoundError(f"找不到可恢复的 Lightning checkpoint: {resume_path}")
        resume_checkpoint = load_checkpoint(resume_path)
        _guard_resume_scope_against_apcf(resume_checkpoint.model_spec)
        apply_checkpoint_model_spec(
            configs,
            resume_checkpoint.model_spec,
            explicit_fields=getattr(args, "explicit_config_fields", set()),
        )
        start_episode = _resume_start_episode(resume_checkpoint.payload)
        if start_episode > int(configs.max_episodes):
            raise ValueError(
                f"续训 checkpoint 已完成 episode={start_episode - 1}，"
                f"不小于 train.max_episodes={int(configs.max_episodes)}；无需继续训练。"
            )
        print(
            f"[Resume] 已验证 checkpoint 连续性，将从 absolute_episode={start_episode} 继续训练。",
            flush=True,
        )
    if uses_runs_layout(configs) and str(getattr(configs, "run_dir", "") or "").strip():
        context = create_run_context(configs, PROJECT_ROOT, create_dirs=True)
        write_run_context_files(context, configs, command="train_lightning", extra={"resume": bool(args.resume)})
    else:
        write_run_manifest(
            checkpoint_dir,
            configs,
            command="train_lightning",
            extra={"resume": bool(args.resume)},
        )

    num_envs = int(configs.num_envs)
    start_method = str(configs.vector_env_start_method)
    if start_method == "auto":
        start_method = "forkserver" if platform.system() == "Linux" else "spawn"

    train_path = resolve_workspace_path(configs.train_data_path_or_dir)
    eval_path = resolve_workspace_path(configs.data_file_path)
    if bool(getattr(configs, "enable_reschedule_mode", False)):
        manifest_path = str(getattr(configs, "reschedule_manifest_path", "") or "").strip()
        if not manifest_path:
            raise ValueError("正式重调度训练必须配置 explicit_fiveskill_v1 manifest")
        training_manifest = load_reschedule_manifest(manifest_path)
        train_paths = resolve_explicit_five_skill_training_paths(training_manifest, train_path)
        print(
            f"[Reschedule] 已通过五技能协议及训练文件精确绑定: {training_manifest.path}",
            flush=True,
        )
        baseline_path = ensure_reschedule_baseline_available(configs)
        if baseline_path is not None:
            print(f"[Reschedule] baseline={baseline_path}", flush=True)
        scenario_path = ensure_reschedule_eval_scenarios_available(configs)
        if scenario_path is not None:
            print(f"[Reschedule] eval_scenarios={scenario_path}", flush=True)
    else:
        initial_manifest_path = str(getattr(configs, "training_manifest_path", "") or "").strip()
        if not initial_manifest_path:
            raise ValueError("正式初始调度训练必须配置 explicit_fiveskill_v1 training_manifest_path")
        train_paths = resolve_explicit_five_skill_initial_training_paths(initial_manifest_path, train_path)
        print(f"[Initial] 已通过五技能协议及 manifest 权威绑定: {initial_manifest_path}", flush=True)
    vector_data_source = tuple(str(path) for path in train_paths)
    vector_env = VectorEnv(
        EnvCreator(vector_data_source, seed_offset=int(configs.seed)),
        num_envs=int(num_envs),
        start_method=start_method,
        worker_threads=configs.vector_env_worker_threads,
        init_timeout_sec=float(configs.vector_env_init_timeout_sec),
        command_timeout_sec=float(configs.vector_env_command_timeout_sec),
    )
    eval_env = AirLineEnv_Graph(eval_path, seed=int(configs.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HBGATPN(configs).to(device)
    _maybe_load_apcf_pretrain(model, device, resume=bool(args.resume))
    if (
        not bool(args.resume)
        and str(getattr(configs, "policy_action_scope", "")) == _APCF_SCOPE
    ):
        record_apcf_pretrain_load(
            checkpoint_dir / "run_manifest.json",
            source_sha256=str(configs.anchor_proposal_pretrain_source_sha256),
            loaded_model_key_count=int(configs.apcf_pretrain_loaded_model_key_count),
        )
    _maybe_load_reschedule_warm_start(model, device, resume=bool(args.resume))
    if getattr(configs, 'use_compile', False):
        try:
            if platform.system() == 'Windows':
                print("ℹ️ Windows 环境检测：跳过 torch.compile（需 Linux + Triton）。")
            else:
                model = torch.compile(model, dynamic=True)
                print("🚀 成功激活 torch.compile 图算子融合编译！")
        except Exception as e:
            print(f"⚠️ 图编译失败，回退至未编译模式。Err: {e}")
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
        teacher_model_factory=(
            (lambda: HBGATPN(copy.deepcopy(configs)))
            if bool(getattr(configs, "best_anchor_distill_enabled", False))
            else None
        ),
        teacher_checkpoint_dir=(
            checkpoint_paths["lightning_dir"]
            if bool(getattr(configs, "best_anchor_distill_enabled", False))
            else None
        ),
    )
    service = APALRolloutService(
        agent=agent,
        vector_env=vector_env,
        eval_env=eval_env,
        config=configs,
        device=device,
    )
    module = APALLightningModule(agent, service, eval_freq=int(configs.eval_freq))
    data_module = APALDataModule(
        service,
        max_episodes=total_updates,
        start_episode=start_episode,
    )
    callbacks = [
        RolloutCheckpoint(
            latest_path=checkpoint_paths["lightning_latest"],
            best_path=checkpoint_paths["lightning_best"],
        )
    ]
    log_root = resolve_tensorboard_log_root(configs)
    tensorboard_logger = TensorBoardLogger(
        save_dir=str(log_root),
        name=sanitize_experiment_name(configs.experiment_name),
    )
    print(f"TensorBoard 日志目录: {tensorboard_logger.log_dir}", flush=True)
    trainer = pl.Trainer(
        accelerator=str(configs.lightning_accelerator),
        devices=int(configs.lightning_devices),
        precision=str(configs.lightning_precision) if torch.cuda.is_available() else "32-true",
        max_steps=-1,
        max_epochs=1,
        callbacks=callbacks,
        logger=tensorboard_logger,
        default_root_dir=str(checkpoint_dir),
        log_every_n_steps=1,
        enable_model_summary=True,
    )
    trainer.fit(
        module,
        datamodule=data_module,
        ckpt_path=str(checkpoint_paths["lightning_latest"]) if args.resume else None,
    )


if __name__ == "__main__":
    from train import main

    raise SystemExit(main())
