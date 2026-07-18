from __future__ import annotations

import os
# 启用可扩展显存段以缓解动态图 GNN 变长 batch 的碎片化；峰值显存仍由 batch 控制。
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import platform
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
from runtime.reschedule_manifest import load_reschedule_manifest
from utils.reschedule import load_reschedule_scenarios
from runtime.seed import set_seed
from training.lightning_module import APALDataModule, APALLightningModule
from training.rollout_service import APALRolloutService
from training.async_evaluation import AsyncEvaluationManager
from utils.vector_env import EnvCreator, VectorEnv
from runtime.artifacts import run_context as create_run_context, uses_runs_layout, write_run_context_files, write_run_manifest
from runtime.checkpoints import apply_checkpoint_model_spec, load_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parent


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
    print(
        f"[Reschedule] eval_instance={entry.instance_id} "
        f"data={entry.data_path} baseline={entry.baseline_schedule_path} "
        f"scenarios={entry.scenario_path}",
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
            if episode % max(1, int(pl_module.eval_freq)) == 0:
                self.async_manager.submit(trainer, episode=episode)
            else:
                trainer.save_checkpoint(str(self.latest_path))
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
                trainer.save_checkpoint(str(self.best_path))
                
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

        trainer.save_checkpoint(str(self.latest_path))
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


def run(args, *, config_initialized: bool = False) -> None:
    if not config_initialized:
        initialize_training_config(args)
    set_seed(int(configs.seed))
    _apply_reschedule_eval_manifest_override()
    _validate_async_eval_target()

    checkpoint_paths = resolve_checkpoint_paths(configs)
    checkpoint_dir = checkpoint_paths["lightning_dir"]
    if args.resume:
        resume_path = checkpoint_paths["lightning_latest"]
        if not resume_path.exists():
            raise FileNotFoundError(f"找不到可恢复的 Lightning checkpoint: {resume_path}")
        resume_checkpoint = load_checkpoint(resume_path)
        apply_checkpoint_model_spec(
            configs,
            resume_checkpoint.model_spec,
            explicit_fields=getattr(args, "explicit_config_fields", set()),
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
        baseline_path = ensure_reschedule_baseline_available(configs)
        if baseline_path is not None:
            print(f"[Reschedule] baseline={baseline_path}", flush=True)
        scenario_path = ensure_reschedule_eval_scenarios_available(configs)
        if scenario_path is not None:
            print(f"[Reschedule] eval_scenarios={scenario_path}", flush=True)
    vector_env = VectorEnv(
        EnvCreator(str(train_path), seed_offset=int(configs.seed)),
        num_envs=int(num_envs),
        start_method=start_method,
        worker_threads=configs.vector_env_worker_threads,
        init_timeout_sec=float(configs.vector_env_init_timeout_sec),
        command_timeout_sec=float(configs.vector_env_command_timeout_sec),
    )
    eval_env = AirLineEnv_Graph(eval_path, seed=int(configs.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HBGATPN(configs).to(device)
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
    )
    service = APALRolloutService(
        agent=agent,
        vector_env=vector_env,
        eval_env=eval_env,
        config=configs,
        device=device,
    )
    module = APALLightningModule(agent, service, eval_freq=int(configs.eval_freq))
    data_module = APALDataModule(service, max_episodes=total_updates)
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
