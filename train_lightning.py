from __future__ import annotations

import os
# 启用可扩展显存段以缓解动态图 GNN 变长 batch 的碎片化；峰值显存仍由 batch 控制。
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
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


PROJECT_ROOT = Path(__file__).resolve().parent


class RolloutCheckpoint(Callback):
    """按 PPO rollout 更新保存最新模型，并按验证 Makespan 保存最佳模型。"""

    def __init__(self, checkpoint_dir: Path) -> None:
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.best_makespan = float("inf")

    @property
    def state_key(self) -> str:
        return "RolloutCheckpoint"

    def state_dict(self) -> dict[str, float]:
        return {"best_makespan": float(self.best_makespan)}

    def load_state_dict(self, state_dict: dict[str, float]) -> None:
        self.best_makespan = float(state_dict.get("best_makespan", float("inf")))

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: APALLightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        episode = int(pl_module.last_completed_episode)
        eval_metrics = pl_module.last_eval_metrics

        if eval_metrics is not None:
            makespan = float(eval_metrics["makespan"])
            if makespan < self.best_makespan:
                self.best_makespan = makespan
                best_dir = self.checkpoint_dir / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                best_path = best_dir / "best.ckpt"
                trainer.save_checkpoint(str(best_path))
                print(
                    f"[Checkpoint] ep={episode} 保存最佳模型: "
                    f"Mk={makespan:.2f} path={best_path}",
                    flush=True,
                )

        latest_path = self.checkpoint_dir / "last.ckpt"
        trainer.save_checkpoint(str(latest_path))
        print(
            f"[Checkpoint] ep={episode} 保存最新模型: path={latest_path}",
            flush=True,
        )


def run(args, *, config_initialized: bool = False) -> None:
    if not config_initialized:
        initialize_training_config(args)
    set_seed(int(configs.seed))

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
    eval_env = AirLineEnv_Graph(eval_path, seed=int(configs.seed) + 10000)
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
    checkpoint_paths = resolve_checkpoint_paths(configs)
    checkpoint_dir = checkpoint_paths["model_dir"] / "lightning"
    callbacks = [RolloutCheckpoint(checkpoint_dir)]
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
    trainer.fit(module, datamodule=data_module, ckpt_path="last" if args.resume else None)


if __name__ == "__main__":
    parser = get_base_parser()
    run(parser.parse_args())
