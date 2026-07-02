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
        )

    if not getattr(agent, "_warned_missing_batch_selector", False):
        print("WARNING: PPOAgent 缺少 select_actions_batch，已退回逐环境 select_action。")
        setattr(agent, "_warned_missing_batch_selector", True)

    results = []
    for obs, task_mask, station_mask, worker_mask in zip(
        obs_list,
        mask_task_list,
        mask_station_matrix_list,
        mask_worker_list,
    ):
        results.append(
            agent.select_action(
                obs,
                mask_task=task_mask,
                mask_station_matrix=station_mask,
                mask_worker=worker_mask,
                deterministic=deterministic,
                temperature=temperature,
                is_eval=is_eval,
            )
        )
    return results

__all__ = ["select_actions_batch_compat"]
