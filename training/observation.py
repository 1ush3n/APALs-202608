from __future__ import annotations


def refresh_env_observation(env):
    """在离散事件等待推进后重建当前观测，兼容真实环境和 VectorEnv proxy。"""
    if hasattr(env, "_get_observation"):
        return env._get_observation()
    if hasattr(env, "rebuild_state_from_snapshot") and hasattr(env, "get_state_snapshot"):
        return env.rebuild_state_from_snapshot(env.get_state_snapshot())
    raise TypeError(f"无法从环境类型 {type(env)!r} 刷新观测。")

__all__ = ["refresh_env_observation"]
