from __future__ import annotations

import threading
import time


class RolloutHeartbeat:
    """可选的 rollout 阶段心跳；interval_sec <= 0 时不启动线程。"""

    def __init__(self, episode: int, num_envs: int, interval_sec: float) -> None:
        self.episode = int(episode)
        self.num_envs = int(num_envs)
        self.interval_sec = float(interval_sec)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = {
            "stage": "start",
            "step": 0,
            "active": 0,
            "done": 0,
            "updated_at": time.perf_counter(),
        }

    def start(self) -> None:
        if self.interval_sec <= 0:
            return
        self._thread = threading.Thread(
            target=self._run,
            name=f"rollout-heartbeat-{self.episode}",
            daemon=True,
        )
        self._thread.start()

    def update(self, stage: str, step: int, active: int, done: int) -> None:
        self._state.update(
            stage=str(stage),
            step=int(step),
            active=int(active),
            done=int(done),
            updated_at=time.perf_counter(),
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_sec):
            elapsed = time.perf_counter() - float(self._state["updated_at"])
            print(
                f"[Heartbeat] ep={self.episode} stage={self._state['stage']} "
                f"step={self._state['step']} active={self._state['active']}/{self.num_envs} "
                f"done={self._state['done']}/{self.num_envs} idle={elapsed:.1f}s",
                flush=True,
            )
