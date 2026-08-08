# -*- coding: utf-8 -*-
"""APCF 反事实预训练（Lightning 入口的模块部分）。

目标（与论文实现计划一致）：
  1) 相对收益 Huber 回归：价值头 ΔÂ = A_P − A_H 拟合 y=(C(H)−C(P))/max(C(H),ε)；
  2) 排序 BCE：σ(ΔÂ) 与 (y>0) 的二分类；
  3) 门控分支 CE：候选收益为正时训练 ℓ_P−ℓ_H（label = y>0 → 选提议分支）；
  4) 正收益加权 BC：提议器自回归生成正收益候选团队的对数概率，权重 = relu(y)。

数据来源：build_anchor_proposal_cf_data.py 产出的 manifest + obs_pt/npz 样本。
训练只用 manifest 的 pretrain split（96 图）；冻结诊断 split 仅用于无梯度验证。
预训练只更新 anchor_team_head 与 anchor_proposal_gate 两个头；编码器/池化保持
初始权重（由 PPO 微调阶段统一优化），确保 checkpoint 语义与 PPO 加载一致。
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import lightning.pytorch as pl
import torch
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset

from configs import Config
from worker_feature_layout import resolve_worker_feature_layout


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=512)
def _load_obs(obs_pt: str) -> Any:
    """缓存加载完整 HeteroData 观测（同一状态多个候选共享）。"""
    return torch.load(obs_pt, map_location="cpu", weights_only=False)


@lru_cache(maxsize=2048)
def _load_worker_mask(npz_path: str) -> torch.Tensor:
    """从样本 npz 读取 worker_mask（同一状态共享）。"""
    import numpy as np

    with np.load(npz_path, allow_pickle=True) as data:
        return torch.as_tensor(np.asarray(data["worker_mask"], dtype=np.bool_)).clone()


def _group_samples(manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    """按 obs_pt 分组：同一状态的所有候选共享一份图观测。"""
    groups: dict[str, dict[str, Any]] = {}
    for entry in manifest.get("files", []):
        if str(entry.get("split")) != split:
            continue
        key = str(entry["obs_pt"])
        if key not in groups:
            groups[key] = {
                "obs_pt": key,
                "task_id": int(entry["task_id"]),
                "station_id": int(entry["station_id"]),
                "anchor_team": tuple(int(w) for w in entry["anchor_team"]),
                "npz": str(entry["npz"]),
                "candidates": [],
            }
        groups[key]["candidates"].append(
            {
                "team": tuple(int(w) for w in entry["candidate_team"]),
                "source": str(entry["source"]),
                "relative_gain": float(entry["relative_gain"]),
            }
        )
    return list(groups.values())


class CFPretrainDataset(Dataset):
    """反事实预训练数据集：每项 = 一个状态的完整观测 + 全部候选团队。"""

    def __init__(self, groups: list[dict[str, Any]]):
        self.groups = groups

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.groups[index]


def _collate_single(items: list[dict[str, Any]]) -> dict[str, Any]:
    assert len(items) == 1
    return items[0]


class CFPretrainDataModule(pl.LightningDataModule):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        train_split: str = "pretrain",
        val_split: str = "frozen_diagnostic",
        num_workers: int = 0,
    ):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.train_split = train_split
        self.val_split = val_split
        self.num_workers = int(num_workers)
        self.train_groups: list[dict[str, Any]] = []
        self.val_groups: list[dict[str, Any]] = []

    def setup(self, stage: str | None = None) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.train_groups = _group_samples(manifest, self.train_split)
        self.val_groups = _group_samples(manifest, self.val_split)
        if not self.train_groups:
            raise ValueError(f"manifest 无 {self.train_split} split 样本：{self.manifest_path}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            CFPretrainDataset(self.train_groups),
            batch_size=1,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=_collate_single,
        )

    def val_dataloader(self) -> DataLoader:
        if not self.val_groups:
            return DataLoader(CFPretrainDataset([]), batch_size=1)
        return DataLoader(
            CFPretrainDataset(self.val_groups),
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate_single,
        )


class CFPretrainLightningModule(pl.LightningModule):
    """反事实预训练：冻结编码器，仅训练提议器与门控双头。"""

    automatic_optimization = False

    def __init__(
        self,
        model: Any,
        config: Config,
        *,
        manifest_path: str | Path,
        manifest_sha256: str = "",
    ):
        super().__init__()
        self.policy = model
        self.config = config
        self.manifest_path = str(manifest_path)
        self.manifest_sha256 = manifest_sha256 or _sha256_file(Path(manifest_path))
        self._optimizer = None

        # 冻结除双头外的全部参数（预训练只学习提议器与门控）。
        for name, parameter in model.named_parameters():
            frozen = not (
                name.startswith("anchor_team_head")
                or name.startswith("anchor_proposal_gate")
            )
            parameter.requires_grad_(not frozen)

        trainable = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        if not trainable:
            raise RuntimeError("预训练未找到可训练参数（需要 anchor_team_head/anchor_proposal_gate）")
        self._trainable_params = trainable
        self._step = 0

    def configure_optimizers(self):
        self._optimizer = torch.optim.AdamW(
            self._trainable_params,
            lr=float(self.config.apcf_pretrain_lr),
            weight_decay=1.0e-5,
        )
        return self._optimizer

    def _encode_state(
        self,
        group: dict[str, Any],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """编码一组观测，返回 (task_emb, station_emb, worker_embs, worker_mask, obs)。"""
        obs = _load_obs(
            str(Path(self.config.anchor_proposal_cf_manifest_path).parent / group["obs_pt"])
        ).to(device)
        worker_mask = _load_worker_mask(
            str(Path(self.config.anchor_proposal_cf_manifest_path).parent / group["npz"])
        ).to(device)
        encoded, _context = self.policy(obs)
        task_emb = encoded["task"][int(group["task_id"])].unsqueeze(0)  # [1, H]
        station_emb = encoded["station"][int(group["station_id"])].unsqueeze(0)  # [1, H]
        worker_embs = encoded["worker"].unsqueeze(0)  # [1, N, H]
        return task_emb, station_emb, worker_embs, worker_mask, obs

    def _compute_group_losses(
        self,
        group: dict[str, Any],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """一个状态组内全部候选的四项损失（组内聚合）。"""
        task_emb, station_emb, worker_embs, worker_mask, obs = self._encode_state(group, device)
        anchor_team = group["anchor_team"]
        anchor_emb = worker_embs[:, list(anchor_team), :].mean(dim=1)  # [1, H]
        worker_feats = obs["worker"].x
        task_feats = obs["task"].x
        station_feats = obs["station"].x
        layout = resolve_worker_feature_layout(self.config)
        num_workers = int(worker_feats.size(0))

        # 合法工人掩码（与 PPO 提议分支/完成器语义一致）。
        skills = worker_feats[:, layout.skill_slice]
        task_skill_vec = task_feats[int(group["task_id"]), 5 : 5 + skills.size(1)]
        skill_idx = int(torch.argmax(task_skill_vec).item())
        has_skill = skills[:, skill_idx] > 0.5  # [N]
        locks = torch.argmax(worker_feats[:, layout.lock_slice], dim=1)
        lock_ok = (locks == 0) | (locks == (int(group["station_id"]) + 1))
        illegal = (~has_skill) | (~lock_ok) | worker_mask.bool()  # True=非法
        legal_count = int((~illegal).sum().item())
        task_duration = float(task_feats[int(group["task_id"]), 0].item())
        scale = max(task_duration, 1.0e-6)
        station_wait = torch.expm1(station_feats[:, 4]).clamp_min(0.0)
        worker_wait = torch.expm1(worker_feats[:, layout.wait_idx]).clamp_min(0.0)

        huber_total = torch.zeros((), device=device)
        bce_total = torch.zeros((), device=device)
        gate_total = torch.zeros((), device=device)
        bc_total = torch.zeros((), device=device)
        count = 0
        for candidate in group["candidates"]:
            team = candidate["team"]
            y = torch.tensor([[candidate["relative_gain"]]], device=device)
            label = torch.tensor([[1 if candidate["relative_gain"] > 0.0 else 0]], device=device)
            proposal_emb = worker_embs[:, list(team), :].mean(dim=1)  # [1, H]
            hamming = float(len(set(team) - set(anchor_team)))

            gate_features = torch.tensor(
                [
                    float(len(anchor_team)) / max(num_workers, 1),
                    float(legal_count) / max(num_workers, 1),
                    1.0,
                    float(station_wait[int(group["station_id"])]) / scale,
                    float(worker_wait.std(unbiased=False).item()) / scale,
                ],
                dtype=torch.float32,
                device=device,
            ).reshape(1, -1)
            branch_logits, delta_a, _g = self.policy.anchor_proposal_gate(
                task_emb,
                station_emb,
                anchor_emb,
                proposal_emb,
                gate_features,
                torch.tensor([[hamming]], dtype=torch.float32, device=device),
            )
            delta_a = delta_a.float()  # [1, 1]

            w_huber = float(self.config.apcf_pretrain_loss_huber_weight)
            w_bce = float(self.config.apcf_pretrain_loss_bce_weight)
            w_gate = float(self.config.apcf_pretrain_loss_gate_weight)
            w_bc = float(self.config.apcf_pretrain_loss_bc_weight)

            if w_huber > 0.0:
                huber_total = huber_total + torch.nn.functional.smooth_l1_loss(
                    delta_a, y, reduction="mean"
                )
            if w_bce > 0.0:
                bce_label = torch.tensor(
                    [[1.0 if candidate["relative_gain"] > 0.0 else 0.0]],
                    device=device,
                )
                bce_total = bce_total + torch.nn.functional.binary_cross_entropy_with_logits(
                    delta_a,
                    bce_label,
                    reduction="mean",
                )
            if w_gate > 0.0:
                gate_total = gate_total + torch.nn.functional.cross_entropy(
                    branch_logits.float(), label.reshape(1)
                )
            if w_bc > 0.0 and candidate["relative_gain"] > 0.0:
                weight = float(candidate["relative_gain"])
                team_logprob = self._proposal_team_logprob(
                    task_emb,
                    station_emb,
                    anchor_emb,
                    worker_embs,
                    team,
                    illegal,
                    anchor_team,
                )
                bc_total = bc_total + weight * (-team_logprob)
            count += 1

        count = max(count, 1)
        huber = huber_total / count
        bce = bce_total / count
        gate = gate_total / count
        bc = bc_total / count
        loss = (
            w_huber * huber
            + w_bce * bce
            + w_gate * gate
            + w_bc * bc
        )
        return {
            "loss": loss,
            "huber": huber,
            "bce": bce,
            "gate": gate,
            "bc": bc,
            "count": torch.tensor(float(count), device=device),
        }

    def _proposal_team_logprob(
        self,
        task_emb: torch.Tensor,
        station_emb: torch.Tensor,
        anchor_emb: torch.Tensor,
        worker_embs: torch.Tensor,
        team: tuple[int, ...],
        illegal: torch.Tensor,
        anchor_team: tuple[int, ...],
    ) -> torch.Tensor:
        """提议器自回归生成候选团队的对数概率（掩码 = 合法性 + 去重）。

        自回归过程与 PPO 提议分支一致：首步强制非锚点（require_difference），
        每步查询含已生成成员均值嵌入 h̄_{w<j}（current_team_emb），掩码排除
        非法与已选成员。候选团队顺序已由数据构建规范为
        "非锚点成员优先、其余按锚点顺序"，保证与运行期掩码语义一致。
        """
        require_diff = bool(
            getattr(self.config, "anchor_proposal_require_difference", True)
        )
        current_illegal = illegal.clone()
        logprob = torch.zeros((), device=worker_embs.device)
        chosen_ids: list[int] = []
        for step, chosen in enumerate(team):
            step_illegal = current_illegal.clone()
            if step == 0 and require_diff:
                for worker_id in anchor_team:
                    step_illegal[int(worker_id)] = True
            context = (
                worker_embs[0, chosen_ids, :].mean(dim=0, keepdim=True)
                if chosen_ids
                else None
            )
            scores = self.policy.anchor_team_head.forward_choice(
                task_emb,
                station_emb,
                anchor_emb,
                worker_embs,
                mask=step_illegal.clone().reshape(1, -1),
                current_team_emb=context,
            )
            scores_float = scores.float()
            dist = Categorical(logits=scores_float)
            chosen_id = int(chosen)
            logprob = logprob + dist.log_prob(torch.tensor([chosen_id], device=worker_embs.device))
            current_illegal[chosen_id] = True
            chosen_ids.append(chosen_id)
        return logprob

    def _log_scalars(self, prefix: str, metrics: dict[str, torch.Tensor]) -> None:
        for name in ("loss", "huber", "bce", "gate", "bc"):
            self.log(
                f"{prefix}{name}",
                float(metrics[name].item()),
                on_step=False,
                on_epoch=True,
                prog_bar=(prefix == "train/" and name == "loss"),
                batch_size=1,
            )

    def training_step(self, batch: dict[str, Any], batch_idx: int):
        device = self._trainable_params[0].device
        metrics = self._compute_group_losses(batch, device)
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(metrics["loss"])
        torch.nn.utils.clip_grad_norm_(self._trainable_params, max_norm=10.0)
        opt.step()
        self._step += 1
        self._log_scalars("train/", metrics)
        self.log(
            "train/group_count",
            float(metrics["count"].item()),
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        return metrics["loss"]

    def validation_step(self, batch: dict[str, Any], batch_idx: int):
        device = self._trainable_params[0].device
        with torch.no_grad():
            metrics = self._compute_group_losses(batch, device)
        self._log_scalars("val/", metrics)
        return metrics["loss"]

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        from runtime.checkpoints import build_checkpoint_metadata

        checkpoint["apal_metadata"] = build_checkpoint_metadata(
            self.config,
            episode=0,
        )
        checkpoint["apal_pretrain_metadata"] = {
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "train_split": "pretrain",
            "val_split": str(self.config.apcf_pretrain_val_split),
            "loss_weights": {
                "huber": float(self.config.apcf_pretrain_loss_huber_weight),
                "bce": float(self.config.apcf_pretrain_loss_bce_weight),
                "gate": float(self.config.apcf_pretrain_loss_gate_weight),
                "bc": float(self.config.apcf_pretrain_loss_bc_weight),
            },
            "steps": int(self._step),
        }
