from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from configs import Config
from environment import AirLineEnv_Graph
from runtime.evaluation import compute_apal_rollout_diagnostics, evaluate_model
from runtime.multiscale import BenchmarkScore, parse_reference_makespans, score_multi_benchmark
from runtime.paths import resolve_workspace_path
from runtime.reschedule_eval import evaluate_reschedule_model
from training.heartbeat import RolloutHeartbeat
from training.lightning_module import RolloutMetrics, RolloutUpdate
from training.memory import Memory
from training.observation import refresh_env_observation


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
        self._last_effective_batch_size: int | None = None
        self._rng = np.random.RandomState(int(config.seed))
        self.use_ipc_fusion = bool(getattr(config, "enable_rollout_ipc_fusion", False))

    def _adaptive_batch_for_task_count(self, num_tasks: int) -> tuple[int | None, str]:
        if not bool(getattr(self.config, "adaptive_ppo_batch_by_tasks", False)):
            return None, "disabled"
        n_tasks = int(num_tasks)
        small_max = int(getattr(self.config, "adaptive_ppo_batch_small_task_max", 530))
        large_min = int(getattr(self.config, "adaptive_ppo_batch_large_task_min", 550))
        if n_tasks <= small_max:
            return int(getattr(self.config, "adaptive_ppo_batch_small", 128)), f"tasks<={small_max}"
        if n_tasks >= large_min:
            return int(getattr(self.config, "adaptive_ppo_batch_large", 64)), f"tasks>={large_min}"
        return int(getattr(self.config, "batch_size", 32)), f"{small_max}<tasks<{large_min}"

    def _apply_adaptive_ppo_batch(self, dataset_idx: int) -> None:
        target, reason = self._adaptive_batch_for_task_count(int(self.vector_env.envs[0].num_tasks))
        if target is None:
            return
        cap = max(0, int(getattr(self.config, "ppo_batch_size_cap", 0)))
        effective = min(int(target), cap) if cap > 0 else int(target)
        old = int(getattr(self.agent, "batch_size", effective))
        self.agent.batch_size = max(1, effective)
        if self._last_effective_batch_size != self.agent.batch_size:
            print(
                f"[PPO Batch] dataset={dataset_idx} tasks={int(self.vector_env.envs[0].num_tasks)} "
                f"batch_size {old}->{self.agent.batch_size} rule={reason}",
                flush=True,
            )
            self._last_effective_batch_size = int(self.agent.batch_size)

    def _try_wait_for_resources_indices(self, indices: list[int]) -> dict[int, bool]:
        """兼容旧 VectorEnv：优先用索引批量接口，缺失时退回全量接口。"""
        target_indices = [int(index) for index in indices]
        if not target_indices:
            return {}
        indexed_wait = getattr(self.vector_env, "try_wait_for_resources_indices", None)
        if callable(indexed_wait):
            return indexed_wait(target_indices)
        all_wait = self.vector_env.try_wait_for_resources_all()
        return {index: bool(all_wait[index]) for index in target_indices}

    def _get_rollout_state_indices(self, indices: list[int]):
        """兼容旧 VectorEnv：优先只刷新等待环境，缺失时全量刷新后筛选。"""
        target_indices = [int(index) for index in indices]
        if not target_indices:
            return {}
        indexed_state = getattr(self.vector_env, "get_rollout_state_indices", None)
        if callable(indexed_state):
            return indexed_state(target_indices)
        masks_list, snapshots = self.vector_env.get_masks_and_snapshots_all()
        return {index: (masks_list[index], snapshots[index]) for index in target_indices}

    def _append_action(
        self,
        memory: Memory,
        *,
        state: dict,
        action: Any,
        logprob: Any,
        value: Any,
        masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        gated_team_trace: Any = None,
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
        memory.gated_team_traces.append(gated_team_trace)

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
        detailed_profile_sums: dict[str, float] = {}
        detailed_profile_samples = 0
        loop_steps = 0
        environment_steps = 0
        objective_delta_sums = np.zeros(self.num_envs, dtype=float)
        objective_final_scores = np.zeros(self.num_envs, dtype=float)
        objective_dense_steps = np.zeros(self.num_envs, dtype=int)
        objective_clip_counts = np.zeros(self.num_envs, dtype=int)
        objective_term_keys = (
            "score_makespan",
            "score_balance",
            "score_takt_violation",
            "score_start_stability",
            "score_station_change",
            "score_team_change",
        )
        objective_term_sums = {
            key: np.zeros(self.num_envs, dtype=float)
            for key in objective_term_keys
        }

        apply_noise = bool(
            self.config.randomize_durations
            and episode > int(self.config.curriculum_episodes)
        )
        heartbeat.update("reset", 0, 0, 0)
        if self.use_ipc_fusion:
            masks_list, snapshots = self.vector_env.reset_rollout_all(
                randomize_duration=apply_noise,
                randomize_workers=apply_noise,
            )
            states: list[Any] = [
                self.vector_env.envs[index].rebuild_state_from_snapshot(snapshots[index])
                for index in range(self.num_envs)
            ]
        else:
            states = self.vector_env.reset_all(
                randomize_duration=apply_noise,
                randomize_workers=apply_noise,
            )
            masks_list = []
            snapshots = []
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
                if not self.use_ipc_fusion:
                    stage_started = time.perf_counter()
                    masks_list, snapshots = self.vector_env.get_masks_and_snapshots_all()
                    ipc_seconds += time.perf_counter() - stage_started
                active = [idx for idx, done in enumerate(dones) if not done]

                waiting_indices = [idx for idx in active if masks_list[idx][0].all()]
                stage_started = time.perf_counter()
                if self.use_ipc_fusion:
                    fused_wait = self.vector_env.wait_rollout_indices(waiting_indices)
                    wait_results = {
                        index: fused_wait[index][0]
                        for index in waiting_indices
                    }
                    refreshed_states = {
                        index: (fused_wait[index][1], fused_wait[index][2])
                        for index in waiting_indices
                        if fused_wait[index][0]
                    }
                else:
                    wait_results = self._try_wait_for_resources_indices(waiting_indices)
                    refreshed_indices = [idx for idx in waiting_indices if wait_results[idx]]
                    refreshed_states = self._get_rollout_state_indices(refreshed_indices)
                ipc_seconds += time.perf_counter() - stage_started

                for idx in waiting_indices:
                    if not masks_list[idx][0].all():
                        continue
                    if wait_results[idx]:
                        masks_list[idx], snapshots[idx] = refreshed_states[idx]
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
                profile_breakdown = bool(
                    getattr(self.config, "enable_rollout_detailed_profiler", False)
                    and episode % max(1, int(self.config.rollout_profile_interval)) == 0
                    and step % max(1, int(self.vector_env.envs[0].num_tasks) // 4) == 0
                )
                with torch.inference_mode():
                    results = self.agent.select_actions_batch(
                        obs_list=[states[idx] for idx in active],
                        mask_task_list=[masks_list[idx][0] for idx in active],
                        mask_station_matrix_list=[
                            masks_list[idx][1] for idx in active
                        ],
                        mask_worker_list=[masks_list[idx][2] for idx in active],
                        deterministic=False,
                        temperature=float(self.config.sample_temperature),
                        is_eval=False,
                        profile_breakdown=profile_breakdown,
                    )
                forward_seconds += time.perf_counter() - stage_started
                if profile_breakdown:
                    detailed_profile_samples += 1
                    for key, value in self.agent.last_action_profile.items():
                        detailed_profile_sums[key] = (
                            detailed_profile_sums.get(key, 0.0) + float(value)
                        )

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
                        gated_team_trace=self.agent.last_gated_team_traces[result_idx],
                    )

                heartbeat.update(
                    "environment_step",
                    step,
                    len(active),
                    self.num_envs - len(active),
                )
                stage_started = time.perf_counter()
                if self.use_ipc_fusion:
                    fused_steps = self.vector_env.step_rollout_indices(
                        {index: actions[index] for index in active}
                    )
                    next_snapshots = list(snapshots)
                    rewards = [0.0] * self.num_envs
                    step_dones = list(dones)
                    step_infos: list[dict | None] = [None] * self.num_envs
                    for index in active:
                        next_masks, next_snapshot, reward, step_done, step_info = (
                            fused_steps[index]
                        )
                        masks_list[index] = next_masks
                        next_snapshots[index] = next_snapshot
                        rewards[index] = reward
                        step_dones[index] = step_done
                        step_infos[index] = step_info
                else:
                    next_snapshots, rewards, step_dones, step_infos = (
                        self.vector_env.step_snapshot_all(actions)
                    )
                environment_step_seconds += time.perf_counter() - stage_started
                environment_steps += len(active)

                for env_idx in active:
                    step_info = step_infos[env_idx] or {}
                    if step_info.get("invalid_action", False):
                        action = actions[env_idx]
                        raise RuntimeError(
                            "训练环境拒绝掩码允许的动作，禁止恢复以避免污染 on-policy 轨迹: "
                            f"env={env_idx} action={action} info={step_info}"
                        )
                    if "reschedule_objective_delta" in step_info:
                        objective_dense_steps[env_idx] += 1
                        objective_delta_sums[env_idx] += float(
                            step_info["reschedule_objective_delta"]
                        )
                        objective_final_scores[env_idx] = float(
                            step_info.get("reschedule_objective_score", 0.0)
                        )
                        objective_clip_counts[env_idx] += int(
                            float(step_info.get("reschedule_objective_reward_clipped", 0.0)) > 0.0
                        )
                        for key in objective_term_keys:
                            objective_term_sums[key][env_idx] += float(
                                step_info.get(f"reschedule_delta_{key}", 0.0)
                            )
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
                if self.use_ipc_fusion:
                    snapshots = next_snapshots
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
        extra_metrics: dict[str, float] = {}
        total_dense_steps = int(np.sum(objective_dense_steps))
        if total_dense_steps > 0:
            extra_metrics = {
                "Reward/ObjectiveDelta": float(np.mean(objective_delta_sums)),
                "Reward/ObjectiveFinalScore": float(np.mean(objective_final_scores)),
                "Reward/ObjectiveClipFraction": float(
                    np.sum(objective_clip_counts) / total_dense_steps
                ),
            }
            for key in objective_term_keys:
                label = key.removeprefix("score_")
                extra_metrics[f"Reward/Delta/{label}"] = float(
                    np.mean(objective_term_sums[key])
                )
        if detailed_profile_samples > 0:
            for key, value in detailed_profile_sums.items():
                label = "".join(part.title() for part in key.removesuffix("_ms").split("_"))
                extra_metrics[f"Rollout/Profile/{label}Ms"] = float(
                    value / detailed_profile_samples
                )
            extra_metrics["Rollout/Profile/SampleCount"] = float(
                detailed_profile_samples
            )
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
            extra_metrics=extra_metrics,
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
            target.gated_team_traces.extend(source.gated_team_traces)

    def collect(self, update_index: int) -> RolloutUpdate:
        dataset_count = int(self.vector_env.envs[0].dataset_count)
        if getattr(self.config, "random_sample_dataset", True):
            dataset_idx = self._rng.randint(0, max(1, dataset_count))
        else:
            dataset_idx = (int(update_index) - 1) % max(1, dataset_count)

        if dataset_idx != self._last_dataset_idx:
            self.vector_env.switch_dataset_all(dataset_idx)
            self._last_dataset_idx = dataset_idx
        self._apply_adaptive_ppo_batch(dataset_idx)

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
            profile_metrics = {
                key.removeprefix("Rollout/Profile/"): value
                for key, value in metrics.extra_metrics.items()
                if key.startswith("Rollout/Profile/") and key != "Rollout/Profile/SampleCount"
            }
            if profile_metrics:
                detail = " ".join(
                    f"{name}={value:.2f}"
                    for name, value in sorted(profile_metrics.items())
                )
                print(
                    f"[RolloutProfile] ep={metrics.episode} "
                    f"samples={int(metrics.extra_metrics['Rollout/Profile/SampleCount'])} {detail}",
                    flush=True,
                )

        return RolloutUpdate(
            memory=merged,
            env=self.vector_env.envs[0],
            episode=first_episode + self.episodes_per_update - 1,
            rollout_metrics=tuple(metrics_list),
        )

    def evaluate(self, episode: int) -> dict[str, float]:
        if bool(getattr(self.config, "enable_reschedule_mode", False)):
            return self.evaluate_reschedule(episode)
        if bool(getattr(self.config, "enable_multi_benchmark_eval", False)):
            return self.evaluate_multi_benchmark(episode)

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
            makespan, balance, reward, schedule, duration, worker_util, station_util = result
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
            "completion_rate": float(
                len(schedule) == int(self.eval_env.num_tasks)
            ),
        }
        print(
            f"[Eval] ep={episode} Mk={metrics['makespan']:.2f} "
            f"Bal={metrics['balance']:.2f} R={metrics['reward']:.2f} "
            f"Done={metrics['completion_rate'] * 100:.1f}% "
            f"W={metrics['worker_utilization'] * 100:.1f}% "
            f"S={metrics['station_utilization'] * 100:.1f}% "
            f"T={metrics['duration_sec']:.2f}s",
            flush=True,
        )
        return metrics

    def evaluate_reschedule(self, episode: int) -> dict[str, float]:
        was_training = bool(self.agent.policy.training)
        print(f"[Eval][Resched] ep={episode} start", flush=True)
        try:
            self.agent.policy.eval()
            result = evaluate_reschedule_model(
                self.eval_env,
                self.agent,
                num_runs=int(getattr(self.config, "reschedule_eval_num_scenarios", 4)),
                temperature=float(self.config.eval_temperature),
                current_ep=episode,
            )
            makespan, balance, reward, _, duration, worker_util, station_util = result
        finally:
            self.agent.policy.train(was_training)

        score_metrics = getattr(evaluate_reschedule_model, "last_metrics", {}) or {}
        metrics: dict[str, float] = {
            "makespan": float(makespan),
            "balance": float(balance),
            "reward": float(reward),
            "duration_sec": float(duration),
            "worker_utilization": float(worker_util),
            "station_utilization": float(station_util),
        }
        for name, value in score_metrics.items():
            if isinstance(value, (int, float, np.floating)):
                metrics[str(name)] = float(value)

        composite = float(metrics.get("composite_score", 0.0))
        selection = float(metrics.get("selection_score", composite))
        eligible_rate = float(metrics.get("eligible_rate", metrics.get("eligible", 0.0)))
        metrics["reschedule_composite_score"] = composite
        metrics["reschedule_selection_score"] = selection
        metrics["reschedule_eligible_rate"] = eligible_rate

        print(
            f"[Eval][Resched] ep={episode} score={composite:.6f} "
            f"selection={selection:.6f} elig={eligible_rate:.2f} "
            f"Mk={metrics['makespan']:.2f} Bal={metrics['balance']:.2f} "
            f"R={metrics['reward']:.2f} W={metrics['worker_utilization'] * 100:.1f}% "
            f"S={metrics['station_utilization'] * 100:.1f}% T={metrics['duration_sec']:.2f}s",
            flush=True,
        )
        return metrics

    def evaluate_multi_benchmark(self, episode: int) -> dict[str, float]:
        refs = parse_reference_makespans(
            getattr(self.config, "multi_benchmark_reference_makespans", {})
        )
        paths = list(getattr(self.config, "multi_benchmark_data_paths", []))
        if not paths:
            raise ValueError("multi_benchmark_data_paths 不能为空")

        was_training = bool(self.agent.policy.training)
        config_backups = {
            name: getattr(self.config, name)
            for name in (
                "enable_dynamic_events",
                "enable_station_breakdown",
                "enable_material_delay",
                "enable_online_duration_perturb",
                "enable_worker_fatigue",
                "randomize_durations",
            )
        }
        for name in config_backups:
            setattr(self.config, name, False)

        rows: list[BenchmarkScore] = []
        print(f"[Eval] ep={episode} start multi_benchmark={len(paths)}", flush=True)
        try:
            self.agent.policy.eval()
            for raw_path in paths:
                data_path = resolve_workspace_path(raw_path)
                benchmark_name = data_path.stem
                if benchmark_name not in refs:
                    raise ValueError(f"缺少基准 {benchmark_name} 的 reference makespan")

                benchmark_seed = int(self.config.seed) + len(rows)
                env = AirLineEnv_Graph(data_path_or_dir=str(data_path), seed=benchmark_seed)
                state = env.reset(
                    randomize_duration=False,
                    randomize_workers=False,
                    seed=benchmark_seed,
                )
                done = False
                invalid_step_count = 0
                start_time = time.time()

                for _ in range(int(env.num_tasks) * 3):
                    if done:
                        break
                    task_mask, station_mask, worker_mask = env.get_masks()
                    if task_mask.all():
                        if env.try_wait_for_resources():
                            state = refresh_env_observation(env)
                            continue
                        invalid_step_count += 1
                        break

                    action_ret = self.agent.select_action(
                        state.to(self.device),
                        mask_task=task_mask.to(self.device),
                        mask_station_matrix=station_mask.to(self.device),
                        mask_worker=worker_mask.to(self.device),
                        deterministic=True,
                        temperature=0.0,
                        is_eval=True,
                    )
                    if action_ret[0] is None:
                        invalid_step_count += 1
                        break
                    action, _, _, _, is_invalid = action_ret
                    if is_invalid:
                        invalid_step_count += 1
                        break
                    state, _reward, done, info = env.step(action)
                    if info.get("invalid_action", False):
                        invalid_step_count += 1
                        break

                complete = len(env.assigned_tasks) == env.num_tasks
                if complete and invalid_step_count == 0:
                    makespan = float(np.max(env.station_wall_clock))
                else:
                    makespan = float(env.ideal_makespan * 3.0)
                reference = float(refs[benchmark_name])
                row = BenchmarkScore(
                    benchmark_name=benchmark_name,
                    data_path=str(data_path),
                    makespan=makespan,
                    reference_makespan=reference,
                    normalized_score=float(makespan / reference),
                    complete=bool(complete),
                    invalid_step_count=int(invalid_step_count),
                    inference_time=float(time.time() - start_time),
                )
                rows.append(row)
                print(
                    f"[Eval][MB] {benchmark_name} Mk={row.makespan:.2f} "
                    f"Ref={row.reference_makespan:.2f} "
                    f"Norm={row.normalized_score:.4f} "
                    f"Complete={int(row.complete)} Invalid={row.invalid_step_count}",
                    flush=True,
                )
        finally:
            for name, value in config_backups.items():
                setattr(self.config, name, value)
            self.agent.policy.train(was_training)

        result = score_multi_benchmark(rows)
        primary = result.rows[0]
        metrics: dict[str, float] = {
            "makespan": float(primary.makespan),
            "multi_benchmark_composite_score": float(result.composite_score),
            "multi_benchmark_selection_score": float(result.selection_score),
            "multi_benchmark_eligible": float(result.eligible),
        }
        for row in result.rows:
            prefix = f"multi_benchmark_{row.benchmark_name}"
            metrics[f"{prefix}_makespan"] = float(row.makespan)
            metrics[f"{prefix}_normalized_score"] = float(row.normalized_score)
            metrics[f"{prefix}_complete"] = float(row.complete)
            metrics[f"{prefix}_invalid_step_count"] = float(row.invalid_step_count)
            metrics[f"{prefix}_inference_time"] = float(row.inference_time)

        print(
            f"[Eval][MB] ep={episode} Score={result.composite_score:.6f} "
            f"Eligible={int(result.eligible)} Selection={result.selection_score:.6f}",
            flush=True,
        )
        return metrics

    def close(self) -> None:
        self.vector_env.close()
