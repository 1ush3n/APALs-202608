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

from args_parser import get_base_parser
from configs import configs
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from train import (
    initialize_training_config,
    resolve_checkpoint_paths,
    resolve_tensorboard_log_root,
    resolve_workspace_path,
    sanitize_experiment_name,
    set_seed,
)
from training.lightning_module import APALDataModule, APALLightningModule
from training.rollout_service import APALRolloutService
from utils.vector_env import EnvCreator, VectorEnv
from runtime.artifacts import run_context as create_run_context, uses_runs_layout, write_run_context_files, write_run_manifest
from runtime.checkpoints import apply_checkpoint_model_spec, load_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parent


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
        self.latest_path.parent.mkdir(parents=True, exist_ok=True)
        self.best_path.parent.mkdir(parents=True, exist_ok=True)
        episode = int(pl_module.last_completed_episode)
        eval_metrics = pl_module.last_eval_metrics

        if eval_metrics is not None:
            makespan = float(eval_metrics["makespan"])
            is_multi_benchmark = "multi_benchmark_selection_score" in eval_metrics
            current_score = float(
                eval_metrics.get("multi_benchmark_selection_score", makespan)
            )
            eligible = bool(
                eval_metrics.get("multi_benchmark_eligible", 1.0) >= 1.0 - 1e-9
            )
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
                else:
                    mk_str = f"Mk={makespan:.2f}"

                print(
                    f"[Checkpoint] ep={episode} 保存最佳模型: "
                    f"metric={'multi_benchmark_normalized_makespan' if is_multi_benchmark else 'eval_makespan'} "
                    f"score={current_score:.6f} {mk_str} path={self.best_path}",
                    flush=True,
                )

        trainer.save_checkpoint(str(self.latest_path))
        print(
            f"[Checkpoint] ep={episode} 保存最新模型: path={self.latest_path}",
            flush=True,
        )


def run(args, *, config_initialized: bool = False) -> None:
    if not config_initialized:
        initialize_training_config(args)
    set_seed(int(configs.seed))

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
    parser = get_base_parser()
    run(parser.parse_args())
