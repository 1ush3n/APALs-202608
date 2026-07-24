from __future__ import annotations

"""仅用于主方法候选方向筛查的独立 Lightning 训练入口。

它支持从任意兼容 checkpoint 只加载策略权重；不会恢复优化器、随机数
或 rollout 状态，因此适合 M0/M1/M2 的公平短续训。正式 ``train.py``
及正式主方法模型不依赖本脚本。
"""

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import TensorBoardLogger

from configs import configs
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from runtime.artifacts import run_context, write_run_context_files
from runtime.checkpoints import apply_checkpoint_model_spec, load_checkpoint
from runtime.hydra_config import ExtraArgument, HydraCliError, initialize_hydra_runtime, should_show_help
from runtime.paths import resolve_checkpoint_paths, resolve_tensorboard_log_root, resolve_workspace_path, sanitize_experiment_name
from runtime.seed import set_seed
from train_lightning import RolloutCheckpoint
from training.lightning_module import APALDataModule, APALLightningModule
from training.rollout_service import APALRolloutService
from utils.vector_env import EnvCreator, VectorEnv


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ScreenLightningModule(APALLightningModule):
    """在候选模型启用时额外记录可解释的门控指标。"""

    def training_step(self, batch: Any, batch_idx: int):
        result = super().training_step(batch, batch_idx)
        gate = getattr(self.policy, "last_screen_gate", None)
        if torch.is_tensor(gate) and gate.ndim == 2 and gate.shape[1] == 3:
            labels = ("station", "task", "worker")
            for index, label in enumerate(labels):
                self.log(
                    f"Screen/GateAttention_{label}",
                    float(gate[:, index].mean().item()),
                    on_step=True,
                    on_epoch=False,
                )
        return result


def _load_initial_weights(model: torch.nn.Module, checkpoint_path: Path, *, strict: bool) -> dict[str, list[str]]:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    incompatibility = model.load_state_dict(checkpoint.state_dict, strict=False)
    missing = list(incompatibility.missing_keys)
    unexpected = list(incompatibility.unexpected_keys)
    allowed_missing = {key for key in missing if key.startswith("screen_")}
    disallowed_missing = sorted(set(missing) - allowed_missing)
    if strict and (disallowed_missing or unexpected):
        raise RuntimeError(
            "初始化 checkpoint 与筛查模型不兼容："
            f"missing={disallowed_missing}; unexpected={unexpected}"
        )
    return {
        "missing": missing,
        "unexpected": unexpected,
        "loaded_checkpoint_format": [checkpoint.format_name],
    }


def run_screen(args: Any) -> None:
    set_seed(int(configs.seed))
    checkpoint_path = resolve_workspace_path(str(args.init_checkpoint))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"初始化 checkpoint 不存在：{checkpoint_path}")
    loaded = load_checkpoint(checkpoint_path, map_location="cpu")
    apply_checkpoint_model_spec(
        configs,
        loaded.model_spec,
        explicit_fields=getattr(args, "explicit_config_fields", set()),
    )
    screen_model = str(args.screen_model).strip().lower()
    if screen_model not in {"full", "scg"}:
        raise ValueError("screen_model 仅允许 full 或 scg")
    if int(configs.max_episodes) <= 0:
        raise ValueError("max_episodes 必须为正数")
    if bool(args.resume):
        raise ValueError("筛查入口禁止 resume=true；请使用同一 init_checkpoint 重新开始短训练。")

    context = run_context(configs, PROJECT_ROOT, create_dirs=True)
    write_run_context_files(
        context,
        configs,
        command="scripts/train_main_screen.py",
        extra={
            "screening_only": True,
            "screen_model": screen_model,
            "init_checkpoint": str(checkpoint_path.resolve()),
            "init_checkpoint_sha256": _sha256(checkpoint_path),
            "optimizer_state_restored": False,
        },
    )
    checkpoint_paths = resolve_checkpoint_paths(configs)
    train_path = resolve_workspace_path(configs.train_data_path_or_dir)
    eval_path = resolve_workspace_path(configs.data_file_path)
    if not train_path.exists() or not eval_path.is_file():
        raise FileNotFoundError(f"训练或验证数据不存在：train={train_path}; eval={eval_path}")

    start_method = str(configs.vector_env_start_method)
    if start_method == "auto":
        start_method = "forkserver" if sys.platform != "win32" else "spawn"
    vector_env = VectorEnv(
        EnvCreator(str(train_path), seed_offset=int(configs.seed)),
        num_envs=int(configs.num_envs),
        start_method=start_method,
        worker_threads=configs.vector_env_worker_threads,
        init_timeout_sec=float(configs.vector_env_init_timeout_sec),
        command_timeout_sec=float(configs.vector_env_command_timeout_sec),
    )
    eval_env = AirLineEnv_Graph(eval_path, seed=int(configs.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if screen_model == "full":
        model_class = HBGATPN
    else:
        # SCG 仅属于临时筛查模块；M0 不依赖该文件，也不触及正式模型代码。
        from experiments.main_screen.screen_models import ScaleGatedContextHBGATPN

        model_class = ScaleGatedContextHBGATPN
    model = model_class(configs).to(device)
    load_report = _load_initial_weights(model, checkpoint_path, strict=bool(args.init_strict))
    (context.artifacts_dir / "screen_initialization.json").write_text(
        json.dumps(
            {
                "screening_only": True,
                "screen_model": screen_model,
                "init_checkpoint": str(checkpoint_path.resolve()),
                "init_checkpoint_sha256": _sha256(checkpoint_path),
                "optimizer_state_restored": False,
                "load_report": load_report,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[Screen] model={screen_model} init={checkpoint_path} "
        f"missing={len(load_report['missing'])} unexpected={len(load_report['unexpected'])} "
        "optimizer_state_restored=false",
        flush=True,
    )

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
    service = APALRolloutService(agent=agent, vector_env=vector_env, eval_env=eval_env, config=configs, device=device)
    module = ScreenLightningModule(agent, service, eval_freq=int(configs.eval_freq))
    data_module = APALDataModule(service, max_episodes=total_updates)
    logger = TensorBoardLogger(
        save_dir=str(resolve_tensorboard_log_root(configs)),
        name=sanitize_experiment_name(configs.experiment_name),
    )
    print(f"[Screen] TensorBoard={logger.log_dir}", flush=True)
    trainer = pl.Trainer(
        accelerator=str(configs.lightning_accelerator),
        devices=int(configs.lightning_devices),
        precision=str(configs.lightning_precision) if torch.cuda.is_available() else "32-true",
        max_steps=-1,
        max_epochs=1,
        callbacks=[RolloutCheckpoint(checkpoint_paths["lightning_latest"], checkpoint_paths["lightning_best"])],
        logger=logger,
        default_root_dir=str(checkpoint_paths["lightning_dir"]),
        log_every_n_steps=1,
        enable_model_summary=True,
    )
    completed = False
    try:
        trainer.fit(module, datamodule=data_module)
        completed = True
    finally:
        if not completed:
            service.close()


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if should_show_help(raw_args):
        print("用法：python scripts/train_main_screen.py experiment=scale_400_800_schedule init_checkpoint=路径 screen_model=full|scg")
        return 0
    try:
        args = initialize_hydra_runtime(
            raw_args,
            target=configs,
            project_root=PROJECT_ROOT,
            default_experiment="scale_400_800_schedule",
            extra_arguments={
                "init_checkpoint": ExtraArgument(required=True, help="仅加载模型权重的初始 checkpoint"),
                "screen_model": ExtraArgument(default="full", help="full 或 scg"),
                "init_strict": ExtraArgument(default=True, help="是否拒绝非筛查模块的权重不匹配"),
            },
            create_run_context=True,
        )
        run_screen(args)
    except (HydraCliError, KeyError, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"[Screen Error] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
