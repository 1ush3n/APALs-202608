# -*- coding: utf-8 -*-
"""APCF 反事实预训练入口（独立于 PPO 微调）。

用法示例（必须在 rag_env 虚拟环境中执行）：
    python -X utf8 scripts/pretrain_anchor_proposal_cf.py \
        --manifest data/initial_anchor_proposal_cf_v1/manifest.json \
        --experiment conf/experiment/initial_anchor_proposal_cf_v1.yaml \
        --output checkpoints/apcf_pretrain_v1.ckpt

输出：Lightning checkpoint（state_dict=完整模型、optimizer、apal_pretrain_metadata
含 manifest SHA-256 与损失权重），供 PPO 微调通过 load_policy_weights 加载。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightning.pytorch as pl
import torch

from configs import configs, load_config_files
from models.hb_gat_pn import HBGATPN
from runtime.configuration import validate_runtime_config
from training.cf_pretrain import CFPretrainDataModule, CFPretrainLightningModule


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="APCF 反事实预训练")
    parser.add_argument("--manifest", default="data/initial_anchor_proposal_cf_v1/manifest.json")
    parser.add_argument("--experiment", default="conf/experiment/initial_anchor_proposal_cf_v1.yaml")
    parser.add_argument("--output", default="checkpoints/apcf_pretrain_v1.ckpt")
    parser.add_argument("--lr", type=float, default=0.0, help="0=使用配置 apcf_pretrain_lr")
    parser.add_argument("--max-epochs", type=int, default=0, help="0=使用配置 apcf_pretrain_epochs")
    parser.add_argument("--limit-train-batches", type=float, default=1.0, help="限制训练 batch 数（smoke 用）")
    parser.add_argument("--limit-val-batches", type=float, default=1.0)
    parser.add_argument("--cpu", action="store_true", help="强制 CPU（smoke/诊断）")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pl.seed_everything(args.seed, workers=True)

    experiment_path = _workspace_path(args.experiment)
    load_config_files([str(experiment_path)], target=configs)
    configs.anchor_proposal_cf_manifest_path = str(_workspace_path(args.manifest))
    if args.lr > 0.0:
        configs.apcf_pretrain_lr = args.lr
    if args.max_epochs > 0:
        configs.apcf_pretrain_epochs = args.max_epochs
    validate_runtime_config(configs)

    device_choice = "cpu" if args.cpu else str(configs.apcf_pretrain_device)
    accelerator = "cpu" if device_choice == "cpu" else "auto"
    model = HBGATPN(configs)
    if device_choice == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，回退 CPU", flush=True)
        accelerator = "cpu"
    model = model.to(torch.device("cuda" if accelerator == "auto" and torch.cuda.is_available() else "cpu"))

    datamodule = CFPretrainDataModule(
        configs.anchor_proposal_cf_manifest_path,
        train_split="pretrain",
        val_split=str(configs.apcf_pretrain_val_split),
        num_workers=0,
    )
    module = CFPretrainLightningModule(
        model,
        configs,
        manifest_path=configs.anchor_proposal_cf_manifest_path,
    )

    output_path = _workspace_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        precision="bf16-mixed" if accelerator != "cpu" and torch.cuda.is_available() else "32-true",
        max_epochs=int(configs.apcf_pretrain_epochs),
        limit_train_batches=float(args.limit_train_batches),
        limit_val_batches=float(args.limit_val_batches),
        log_every_n_steps=1,
        enable_model_summary=False,
        default_root_dir=str(output_path.parent),
    )
    trainer.fit(module, datamodule=datamodule)
    trainer.save_checkpoint(str(output_path))
    print(f"[apcf-pretrain] 完成：{output_path}", flush=True)
    return 0


def _workspace_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
