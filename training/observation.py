from __future__ import annotations

from typing import Any


def refresh_env_observation(env):
    """在离散事件等待推进后重建当前观测，兼容真实环境和 VectorEnv proxy。"""
    if hasattr(env, "_get_observation"):
        return env._get_observation()
    if hasattr(env, "rebuild_state_from_snapshot") and hasattr(env, "get_state_snapshot"):
        return env.rebuild_state_from_snapshot(env.get_state_snapshot())
    raise TypeError(f"无法从环境类型 {type(env)!r} 刷新观测。")


class CachedEnvironmentObservation:
    """复用单个 PyG 观测对象，仅刷新动态特征和动态分配边。"""

    def __init__(self, env: Any) -> None:
        if not hasattr(env, "get_state_snapshot") or not hasattr(env, "rebuild_state_from_snapshot"):
            raise TypeError(f"环境不支持快照观测缓存: {type(env).__name__}")
        self.env = env
        self._state = None

    def clear(self) -> None:
        self._state = None

    def refresh(self):
        snapshot = self.env.get_state_snapshot()
        if self._state is None:
            self._state = self.env.rebuild_state_from_snapshot(snapshot)
        else:
            self._state = self.env.rebuild_state_from_snapshot(
                snapshot,
                reusable_state=self._state,
                reuse_resource_topology=True,
            )
        return self._state


__all__ = ["CachedEnvironmentObservation", "refresh_env_observation"]
