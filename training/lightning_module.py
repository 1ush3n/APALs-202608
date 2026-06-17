from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, IterableDataset


@dataclass(frozen=True)
class RolloutMetrics:
    episode: int
    average_reward: float
    average_makespan: float
    completion_rate: float
    environment_steps: int
    steps_per_second: float
    total_seconds: float
    ipc_mask_ms: float
    forward_ms: float
    rebuild_ms: float
    environment_step_ms: float

    def as_log_dict(self) -> dict[str, float]:
        return {
            "Rollout/AverageReward": self.average_reward,
            "Rollout/AverageMakespan": self.average_makespan,
            "Rollout/CompletionRate": self.completion_rate,
            "Rollout/EnvironmentSteps": float(self.environment_steps),
            "Rollout/StepsPerSecond": self.steps_per_second,
            "Rollout/TotalSeconds": self.total_seconds,
            "Rollout/IPCMaskMs": self.ipc_mask_ms,
            "Rollout/ForwardMs": self.forward_ms,
            "Rollout/RebuildMs": self.rebuild_ms,
            "Rollout/EnvironmentStepMs": self.environment_step_ms,
        }


@dataclass(frozen=True)
class RolloutUpdate:
    """一次同质 PPO 更新所需的轨迹及环境重建上下文。"""

    memory: Any
    env: Any
    episode: int
    rollout_metrics: tuple[RolloutMetrics, ...] = ()


class RolloutService(Protocol):
    """Lightning 与 APAL 多进程采样实现之间的依赖倒置接口。"""

    def collect(self, episode: int) -> RolloutUpdate:
        ...

    def evaluate(self, episode: int) -> dict[str, float]:
        ...

    def close(self) -> None:
        ...


class _RolloutDataset(IterableDataset):
    def __init__(self, rollout_service: RolloutService, max_episodes: int):
        self.rollout_service = rollout_service
        self.max_episodes = int(max_episodes)

    def __iter__(self):
        for episode in range(1, self.max_episodes + 1):
            yield self.rollout_service.collect(episode)

    def __len__(self) -> int:
        return self.max_episodes


class APALDataModule(pl.LightningDataModule):
    """DataModule 产生 on-policy rollout，DataLoader 不再复制环境池。"""

    def __init__(self, rollout_service: RolloutService, max_episodes: int):
        super().__init__()
        self.rollout_service = rollout_service
        self.max_episodes = int(max_episodes)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            _RolloutDataset(self.rollout_service, self.max_episodes),
            batch_size=None,
            num_workers=0,
        )

    def __len__(self) -> int:
        return self.max_episodes


class APALLightningModule(pl.LightningModule):
    """使用 Lightning 手动优化生命周期驱动 PPOAgent。"""

    automatic_optimization = False

    def __init__(
        self,
        agent: Any,
        rollout_service: RolloutService,
        *,
        eval_freq: int,
    ):
        super().__init__()
        self.agent = agent
        self.rollout_service = rollout_service
        self.eval_freq = max(1, int(eval_freq))
        self.policy = agent.policy
        self.last_completed_episode = 0
        self.last_eval_metrics: dict[str, float] | None = None

    def configure_optimizers(self):
        return self.agent.optimizer

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        from runtime.checkpoints import build_checkpoint_metadata

        checkpoint["apal_metadata"] = build_checkpoint_metadata(
            self.rollout_service.config,
            episode=int(self.last_completed_episode),
        )

    def transfer_batch_to_device(
        self,
        batch: RolloutUpdate,
        device: torch.device,
        dataloader_idx: int,
    ) -> RolloutUpdate:
        # Rollout 内含环境与 CPU 轨迹对象，由 PPOAgent 在更新时按需迁移张量。
        assert isinstance(batch, RolloutUpdate), type(batch)
        return batch

    def training_step(self, batch: RolloutUpdate, batch_idx: int):
        assert isinstance(batch, RolloutUpdate), type(batch)
        self.last_eval_metrics = None
        for rollout_metrics in batch.rollout_metrics:
            # 记录详细指标
            for name, value in rollout_metrics.as_log_dict().items():
                self.log(name, value, on_step=True, on_epoch=False)
            # 显式推送到 Lightning 终端进度条
            self.log("Rew", float(rollout_metrics.average_reward), on_step=True, on_epoch=False, prog_bar=True)
            self.log("Mk", float(rollout_metrics.average_makespan), on_step=True, on_epoch=False, prog_bar=True)
            self.log("SPS", float(rollout_metrics.steps_per_second), on_step=True, on_epoch=False, prog_bar=True)

        self.agent.validate_snapshot_homogeneity(batch.memory.states)
        metrics = self.agent.update(batch.memory, batch.env, current_ep=batch.episode)
        if float(metrics.get("OOM/SkippedUpdate", 0.0)) > 0.0:
            print(
                f"[PPO OOM] Episode {batch.episode} 更新已安全回滚并跳过，"
                f"下轮 batch_size={int(metrics['OOM/EffectiveBatchSize'])}"
            )
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                scalar = torch.tensor(float(value))
                if torch.isfinite(scalar):
                    self.log(name, float(value), on_step=True, on_epoch=False)
                    # 将总损失也展示到进度条
                    if name == "Loss/Total":
                        self.log("loss", float(value), on_step=True, on_epoch=False, prog_bar=True)

        if batch.episode % self.eval_freq == 0:
            self.last_eval_metrics = self.rollout_service.evaluate(batch.episode)
            for name, value in self.last_eval_metrics.items():
                self.log(f"Eval/{name}", float(value), on_step=True, on_epoch=False)
        self.last_completed_episode = int(batch.episode)
        return None

    def on_fit_end(self) -> None:
        self.rollout_service.close()
