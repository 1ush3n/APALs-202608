from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from configs import Config
from train import Memory, evaluate_model, select_actions_batch_compat
from training.heartbeat import RolloutHeartbeat
from training.lightning_module import RolloutMetrics, RolloutUpdate


class APALRolloutService:
    """收集同质窄池轨迹，并向 Lightning 提供 PPO 更新批次。"""

    def __init__(
        self,
        *,
        agent: Any,
        vector_env: Any,
        eval_env: Any,
        config: Config,
        device: torch.device,
    ) -> None:
        self.agent = agent
        self.vector_env = vector_env
        self.eval_env = eval_env
        self.config = config
        self.device = device
        self.num_envs = int(vector_env.num_envs)
        self.episodes_per_update = max(1, int(config.update_every_episodes))
        self._last_dataset_idx: int | None = None

    def _append_action(
        self,
        memory: Memory,
        *,
        state: dict,
        action: Any,
        logprob: Any,
        value: Any,
        masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(
            logprob
            if torch.is_tensor(logprob)
            else torch.tensor(logprob, device=self.device)
        )
        memory.values.append(
            value if torch.is_tensor(value) else torch.tensor(value, device=self.device)
        )
        memory.masks.append(masks)

    def _collect_episode(
        self,
        episode: int,
    ) -> tuple[list[Memory], RolloutMetrics]:
        self.agent.policy.train()
        heartbeat = RolloutHeartbeat(
            episode,
            self.num_envs,
            float(self.config.rollout_heartbeat_interval_sec),
        )
        heartbeat.start()
        total_started = time.perf_counter()
        ipc_seconds = 0.0
        forward_seconds = 0.0
        rebuild_seconds = 0.0
        environment_step_seconds = 0.0
        loop_steps = 0
        environment_steps = 0

        apply_noise = bool(
            self.config.randomize_durations
            and episode > int(self.config.curriculum_episodes)
        )
        heartbeat.update("reset", 0, 0, 0)
        states = self.vector_env.reset_all(
            randomize_duration=apply_noise,
            randomize_workers=apply_noise,
        )
        memories = [Memory() for _ in range(self.num_envs)]
        dones = [False] * self.num_envs
        max_steps = max(int(env.num_tasks) for env in self.vector_env.envs) * 2
        if self.config.rollout_max_steps > 0:
            max_steps = min(max_steps, self.config.rollout_max_steps)

        try:
            for step in range(max_steps):
                if all(dones):
                    break
                loop_steps += 1
                active_count = sum(1 for done in dones if not done)
                heartbeat.update(
                    "ipc_mask",
                    step,
                    active_count,
                    self.num_envs - active_count,
                )
                stage_started = time.perf_counter()
                masks_list, snapshots = self.vector_env.get_masks_and_snapshots_all()
                ipc_seconds += time.perf_counter() - stage_started
                active = [idx for idx, done in enumerate(dones) if not done]

                for idx in list(active):
                    if not masks_list[idx][0].all():
                        continue
                    if self.vector_env.envs[idx].try_wait_for_resources():
                        stage_started = time.perf_counter()
                        masks_list[idx], snapshots[idx] = (
                            self.vector_env.envs[idx].get_rollout_state()
                        )
                        ipc_seconds += time.perf_counter() - stage_started
                        stage_started = time.perf_counter()
                        states[idx] = self.vector_env.envs[
                            idx
                        ].rebuild_state_from_snapshot(snapshots[idx])
                        rebuild_seconds += time.perf_counter() - stage_started
                    else:
                        dones[idx] = True
                        if memories[idx].rewards:
                            memories[idx].rewards[-1] -= (
                                self.config.deadlock_penalty_constant
                                * self.config.r_coef_makespan
                                * self.config.reward_scale
                            )
                            memories[idx].is_terminals[-1] = True

                active = [idx for idx, done in enumerate(dones) if not done]
                if not active:
                    break

                heartbeat.update(
                    "forward",
                    step,
                    len(active),
                    self.num_envs - len(active),
                )
                stage_started = time.perf_counter()
                with torch.inference_mode():
                    results = select_actions_batch_compat(
                        self.agent,
                        obs_list=[states[idx] for idx in active],
                        mask_task_list=[masks_list[idx][0] for idx in active],
                        mask_station_matrix_list=[
                            masks_list[idx][1] for idx in active
                        ],
                        mask_worker_list=[masks_list[idx][2] for idx in active],
                        deterministic=False,
                        temperature=float(self.config.sample_temperature),
                        is_eval=False,
                    )
                forward_seconds += time.perf_counter() - stage_started

                actions = [None] * self.num_envs
                for result_idx, env_idx in enumerate(active):
                    action, logprob, value, _, is_invalid = results[result_idx]
                    if is_invalid:
                        raise RuntimeError(
                            f"训练阶段产生非法动作: env={env_idx}, action={action}"
                        )
                    actions[env_idx] = action
                    self._append_action(
                        memories[env_idx],
                        state=snapshots[env_idx],
                        action=action,
                        logprob=logprob,
                        value=value,
                        masks=masks_list[env_idx],
                    )

                heartbeat.update(
                    "environment_step",
                    step,
                    len(active),
                    self.num_envs - len(active),
                )
                stage_started = time.perf_counter()
                next_snapshots, rewards, step_dones, _ = (
                    self.vector_env.step_snapshot_all(actions)
                )
                environment_step_seconds += time.perf_counter() - stage_started
                environment_steps += len(active)

                for env_idx in active:
                    memories[env_idx].rewards.append(float(rewards[env_idx]))
                    memories[env_idx].is_terminals.append(bool(step_dones[env_idx]))
                    memories[env_idx].is_truncated.append(False)
                    dones[env_idx] = bool(step_dones[env_idx])
                    if actions[env_idx] is not None:
                        stage_started = time.perf_counter()
                        states[env_idx] = self.vector_env.envs[
                            env_idx
                        ].rebuild_state_from_snapshot(next_snapshots[env_idx])
                        rebuild_seconds += time.perf_counter() - stage_started
        finally:
            heartbeat.stop()

        for memory in memories:
            if memory.is_terminals and not memory.is_terminals[-1]:
                memory.is_truncated[-1] = True

        total_seconds = time.perf_counter() - total_started
        makespans = [
            float(np.max(env.station_wall_clock)) if env.assigned_tasks else 0.0
            for env in self.vector_env.envs
        ]
        completion_rates = [
            len(env.assigned_tasks) / max(1, int(env.num_tasks))
            for env in self.vector_env.envs
        ]
        divisor = max(1, loop_steps)
        metrics = RolloutMetrics(
            episode=int(episode),
            average_reward=float(np.mean([sum(memory.rewards) for memory in memories])),
            average_makespan=float(np.mean(makespans)),
            completion_rate=float(np.mean(completion_rates)),
            environment_steps=environment_steps,
            steps_per_second=environment_steps / max(total_seconds, 1e-9),
            total_seconds=total_seconds,
            ipc_mask_ms=ipc_seconds * 1000.0 / divisor,
            forward_ms=forward_seconds * 1000.0 / divisor,
            rebuild_ms=rebuild_seconds * 1000.0 / divisor,
            environment_step_ms=environment_step_seconds * 1000.0 / divisor,
        )
        return memories, metrics

    @staticmethod
    def _merge_memories(target: Memory, sources: list[Memory]) -> None:
        for source in sources:
            target.states.extend(source.states)
            target.actions.extend(source.actions)
            target.logprobs.extend(source.logprobs)
            target.rewards.extend(source.rewards)
            target.is_terminals.extend(source.is_terminals)
            target.is_truncated.extend(source.is_truncated)
            target.masks.extend(source.masks)
            target.values.extend(source.values)

    def collect(self, update_index: int) -> RolloutUpdate:
        dataset_count = int(self.vector_env.envs[0].dataset_count)
        dataset_idx = (int(update_index) - 1) % max(1, dataset_count)
        if dataset_idx != self._last_dataset_idx:
            self.vector_env.switch_dataset_all(dataset_idx)
            self._last_dataset_idx = dataset_idx

        merged = Memory()
        metrics_list: list[RolloutMetrics] = []
        first_episode = (int(update_index) - 1) * self.episodes_per_update + 1
        for offset in range(self.episodes_per_update):
            memories, metrics = self._collect_episode(first_episode + offset)
            self._merge_memories(merged, memories)
            metrics_list.append(metrics)
            print(
                f"[Rollout] ep={metrics.episode} R={metrics.average_reward:.2f} "
                f"Mk={metrics.average_makespan:.1f} "
                f"Done={metrics.completion_rate * 100:.1f}% "
                f"SPS={metrics.steps_per_second:.1f} "
                f"ms={metrics.ipc_mask_ms:.1f}/{metrics.forward_ms:.1f}/"
                f"{metrics.rebuild_ms:.1f}/{metrics.environment_step_ms:.1f}",
                flush=True,
            )

        return RolloutUpdate(
            memory=merged,
            env=self.vector_env.envs[0],
            episode=first_episode + self.episodes_per_update - 1,
            rollout_metrics=tuple(metrics_list),
        )

    def evaluate(self, episode: int) -> dict[str, float]:
        was_training = bool(self.agent.policy.training)
        config_backups = {
            name: getattr(self.config, name)
            for name in (
                "enable_dynamic_events",
                "enable_station_breakdown",
                "enable_material_delay",
            )
        }
        print(f"[Eval] ep={episode} start scenarios=standard", flush=True)
        try:
            self.agent.policy.eval()
            result = evaluate_model(
                self.eval_env,
                self.agent,
                num_runs=1,
                temperature=float(self.config.eval_temperature),
                current_ep=episode,
                scenario_names=tuple(self.config.eval_scenarios),
            )
            makespan, balance, reward, _, duration, worker_util, station_util = result
        finally:
            for name, value in config_backups.items():
                setattr(self.config, name, value)
            self.agent.policy.train(was_training)

        metrics = {
            "makespan": float(makespan),
            "balance": float(balance),
            "reward": float(reward),
            "duration_sec": float(duration),
            "worker_utilization": float(worker_util),
            "station_utilization": float(station_util),
        }
        print(
            f"[Eval] ep={episode} Mk={metrics['makespan']:.2f} "
            f"Bal={metrics['balance']:.2f} R={metrics['reward']:.2f} "
            f"W={metrics['worker_utilization'] * 100:.1f}% "
            f"S={metrics['station_utilization'] * 100:.1f}% "
            f"T={metrics['duration_sec']:.2f}s",
            flush=True,
        )
        return metrics

    def close(self) -> None:
        self.vector_env.close()
