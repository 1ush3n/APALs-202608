from __future__ import annotations

from dataclasses import dataclass, field
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
    extra_metrics: dict[str, float] = field(default_factory=dict)

    def as_log_dict(self) -> dict[str, float]:
        metrics = {
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
        metrics.update(self.extra_metrics)
        return metrics


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
    def __init__(
        self,
        rollout_service: RolloutService,
        max_episodes: int,
        *,
        start_episode: int = 1,
    ):
        self.rollout_service = rollout_service
        self.max_episodes = int(max_episodes)
        self.start_episode = int(start_episode)
        if self.start_episode < 1:
            raise ValueError(f"start_episode 必须大于等于 1，当前为 {self.start_episode}")
        if self.start_episode > self.max_episodes + 1:
            raise ValueError(
                f"start_episode={self.start_episode} 超过训练上限 {self.max_episodes}"
            )

    def __iter__(self):
        for episode in range(self.start_episode, self.max_episodes + 1):
            yield self.rollout_service.collect(episode)

    def __len__(self) -> int:
        return max(0, self.max_episodes - self.start_episode + 1)


class APALDataModule(pl.LightningDataModule):
    """DataModule 产生 on-policy rollout，DataLoader 不再复制环境池。"""

    def __init__(
        self,
        rollout_service: RolloutService,
        max_episodes: int,
        *,
        start_episode: int = 1,
    ):
        super().__init__()
        self.rollout_service = rollout_service
        self.max_episodes = int(max_episodes)
        self.start_episode = int(start_episode)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            _RolloutDataset(
                self.rollout_service,
                self.max_episodes,
                start_episode=self.start_episode,
            ),
            batch_size=None,
            num_workers=0,
        )

    def __len__(self) -> int:
        return max(0, self.max_episodes - self.start_episode + 1)


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
        self.last_update_committed = False
        self.resume_checkpoint_batch_size: int | None = None
        self.resume_batch_override_applied = False

    def configure_optimizers(self):
        return self.agent.optimizer

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        from runtime.checkpoints import build_checkpoint_metadata

        checkpoint["apal_metadata"] = build_checkpoint_metadata(
            self.rollout_service.config,
            episode=int(self.last_completed_episode),
        )
        agent_state = {
            "current_step": int(getattr(self.agent, "current_step", 0)),
            "batch_size": int(getattr(self.agent, "batch_size", 0)),
        }
        if hasattr(self.agent, "scaler"):
            agent_state["scaler"] = self.agent.scaler.state_dict()
        if hasattr(self.agent, "best_anchor_checkpoint_state"):
            teacher_state = self.agent.best_anchor_checkpoint_state()
            if teacher_state is not None:
                agent_state["best_anchor_teacher"] = teacher_state
        checkpoint["apal_agent_state"] = agent_state

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        metadata = checkpoint.get("apal_metadata", {})
        if isinstance(metadata, dict):
            from runtime.checkpoints import validate_checkpoint_training_spec

            validate_checkpoint_training_spec(
                self.rollout_service.config,
                metadata,
            )
            self.last_completed_episode = int(metadata.get("episode", 0))
        agent_state = checkpoint.get("apal_agent_state", {})
        if not isinstance(agent_state, dict):
            return
        saved_batch_size = int(agent_state.get("batch_size", self.agent.batch_size))
        is_v2 = (
            str(getattr(self.rollout_service.config, "team_selection_mode", ""))
            == "autoregressive_pressure_v2"
        )
        self.resume_checkpoint_batch_size = saved_batch_size
        self.resume_batch_override_applied = bool(
            is_v2 and saved_batch_size != int(self.agent.batch_size)
        )
        self.agent.current_step = int(
            agent_state.get("current_step", self.agent.current_step)
        )
        if not is_v2:
            self.agent.batch_size = saved_batch_size
        scaler_state = agent_state.get("scaler")
        if isinstance(scaler_state, dict) and hasattr(self.agent, "scaler"):
            self.agent.scaler.load_state_dict(scaler_state)
        if "best_anchor_teacher" in agent_state and hasattr(
            self.agent, "restore_best_anchor_checkpoint_state"
        ):
            self.agent.restore_best_anchor_checkpoint_state(agent_state["best_anchor_teacher"])

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
        self.last_update_committed = False
        self.agent.validate_snapshot_homogeneity(batch.memory.states)
        metrics = self.agent.update(batch.memory, batch.env, current_ep=batch.episode)
        if float(metrics.get("OOM/SkippedUpdate", 0.0)) > 0.0:
            print(
                f"[PPO OOM] Episode {batch.episode} 更新已安全回滚并跳过；"
                "不记录本轮 rollout/loss/eval，下一轮继续使用原 batch_size。"
            )
            return None
        _scope = str(
            getattr(getattr(self.rollout_service, "config", None), "policy_action_scope", "")
        )
        if _scope == "operation_station_anchor_proposal_team":
            self.agent.advance_apcf_update()

        for rollout_metrics in batch.rollout_metrics:
            # 记录详细指标
            for name, value in rollout_metrics.as_log_dict().items():
                self.log(name, value, on_step=True, on_epoch=False)
            # 显式推送到 Lightning 终端进度条
            self.log("Rew", float(rollout_metrics.average_reward), on_step=True, on_epoch=False, prog_bar=True)
            self.log("Mk", float(rollout_metrics.average_makespan), on_step=True, on_epoch=False, prog_bar=True)
            self.log("SPS", float(rollout_metrics.steps_per_second), on_step=True, on_epoch=False, prog_bar=True)

        for name, value in metrics.items():
            if str(name).startswith("_"):
                continue
            if isinstance(value, (int, float)):
                scalar = torch.tensor(float(value))
                if torch.isfinite(scalar):
                    self.log(name, float(value), on_step=True, on_epoch=False)
                    # 将总损失也展示到进度条
                    if name == "Loss/Total":
                        self.log("loss", float(value), on_step=True, on_epoch=False, prog_bar=True)

        async_eval_enabled = bool(
            getattr(getattr(self.rollout_service, "config", None), "async_eval_enabled", False)
        )
        if not async_eval_enabled and batch.episode % self.eval_freq == 0:
            self.last_eval_metrics = self.rollout_service.evaluate(batch.episode)
            for name, value in self.last_eval_metrics.items():
                self.log(f"Eval/{name}", float(value), on_step=True, on_epoch=False)
        self.last_completed_episode = int(batch.episode)
        self.last_update_committed = True
        return None

    def on_fit_end(self) -> None:
        self.rollout_service.close()
