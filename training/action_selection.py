from __future__ import annotations


def select_actions_batch_compat(
    agent,
    *,
    obs_list,
    mask_task_list,
    mask_station_matrix_list,
    mask_worker_list,
    deterministic: bool,
    temperature: float,
    is_eval: bool,
    baseline_snapshots=None,
):
    """兼容旧版 PPOAgent：缺少批量动作接口时退回逐环境采样。"""
    batch_selector = getattr(agent, "select_actions_batch", None)
    if callable(batch_selector):
        return batch_selector(
            obs_list=obs_list,
            mask_task_list=mask_task_list,
            mask_station_matrix_list=mask_station_matrix_list,
            mask_worker_list=mask_worker_list,
            deterministic=deterministic,
            temperature=temperature,
            is_eval=is_eval,
            baseline_snapshots=baseline_snapshots,
        )

    if not getattr(agent, "_warned_missing_batch_selector", False):
        print("WARNING: PPOAgent 缺少 select_actions_batch，已退回逐环境 select_action。")
        setattr(agent, "_warned_missing_batch_selector", True)

    results = []
    for index, (obs, task_mask, station_mask, worker_mask) in enumerate(zip(
        obs_list,
        mask_task_list,
        mask_station_matrix_list,
        mask_worker_list,
    )):
        results.append(
            agent.select_action(
                obs,
                mask_task=task_mask,
                mask_station_matrix=station_mask,
                mask_worker=worker_mask,
                deterministic=deterministic,
                temperature=temperature,
                is_eval=is_eval,
                baseline_snapshot=(
                    baseline_snapshots[index]
                    if baseline_snapshots is not None
                    else None
                ),
            )
        )
    return results

__all__ = ["select_actions_batch_compat"]
