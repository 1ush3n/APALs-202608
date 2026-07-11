from __future__ import annotations

import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class DDQNTransition:
    """DDQN 经验条目；快照保持在 CPU，按需重建 PyG Batch。"""

    transition_id: int
    dataset_idx: int
    state_snapshot: dict[str, Any]
    action: tuple[int, int, list[int]]
    reward: float
    next_snapshot: dict[str, Any]
    done: bool
    masks: tuple[Any, Any, Any]
    next_masks: tuple[Any, Any, Any]


class DatasetReplayBuffer:
    """全局 FIFO 容量、按数据集索引的可复现 replay buffer。"""

    def __init__(self, capacity: int, *, seed: int) -> None:
        if int(capacity) <= 0:
            raise ValueError("replay capacity 必须大于 0")
        self.capacity = int(capacity)
        self._rng = random.Random(int(seed))
        self._next_id = 0
        self._order: deque[int] = deque()
        self._items: dict[int, DDQNTransition] = {}
        self._dataset_ids: dict[int, deque[int]] = defaultdict(deque)

    def __len__(self) -> int:
        return len(self._order)

    @property
    def next_transition_id(self) -> int:
        return int(self._next_id)

    def count(self, dataset_idx: int) -> int:
        return len(self._dataset_ids.get(int(dataset_idx), ()))

    def append(
        self,
        *,
        dataset_idx: int,
        state_snapshot: dict[str, Any],
        action: tuple[int, int, list[int]],
        reward: float,
        next_snapshot: dict[str, Any],
        done: bool,
        masks: tuple[Any, Any, Any],
        next_masks: tuple[Any, Any, Any],
    ) -> DDQNTransition:
        transition = DDQNTransition(
            transition_id=self._next_id,
            dataset_idx=int(dataset_idx),
            state_snapshot=state_snapshot,
            action=(int(action[0]), int(action[1]), [int(worker) for worker in action[2]]),
            reward=float(reward),
            next_snapshot=next_snapshot,
            done=bool(done),
            masks=masks,
            next_masks=next_masks,
        )
        self._next_id += 1
        self._order.append(transition.transition_id)
        self._items[transition.transition_id] = transition
        self._dataset_ids[transition.dataset_idx].append(transition.transition_id)
        self._evict_if_needed()
        return transition

    def _evict_if_needed(self) -> None:
        while len(self._order) > self.capacity:
            transition_id = self._order.popleft()
            transition = self._items.pop(transition_id)
            dataset_ids = self._dataset_ids[transition.dataset_idx]
            oldest_dataset_id = dataset_ids.popleft()
            if oldest_dataset_id != transition_id:
                raise RuntimeError("replay dataset 索引与全局 FIFO 顺序不一致")
            if not dataset_ids:
                del self._dataset_ids[transition.dataset_idx]

    def sample(self, dataset_idx: int, batch_size: int) -> list[DDQNTransition]:
        dataset_idx = int(dataset_idx)
        batch_size = int(batch_size)
        ids = self._dataset_ids.get(dataset_idx)
        if ids is None or len(ids) < batch_size:
            raise ValueError(
                f"dataset={dataset_idx} replay 样本不足: "
                f"{0 if ids is None else len(ids)} < {batch_size}"
            )
        selected_ids = self._rng.sample(tuple(ids), batch_size)
        return [self._items[transition_id] for transition_id in selected_ids]

    def ordered_transitions(self) -> list[DDQNTransition]:
        return [self._items[transition_id] for transition_id in self._order]

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "next_id": self._next_id,
            "rng_state": self._rng.getstate(),
            "transitions": self.ordered_transitions(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        capacity = int(state.get("capacity", self.capacity))
        if capacity != self.capacity:
            raise ValueError(
                f"replay capacity 与 sidecar 不一致: current={self.capacity}, saved={capacity}"
            )
        self._order.clear()
        self._items.clear()
        self._dataset_ids.clear()
        transitions: Iterable[DDQNTransition] = state.get("transitions", ())
        for transition in transitions:
            if not isinstance(transition, DDQNTransition):
                raise TypeError(f"sidecar 包含非法 replay 条目: {type(transition)!r}")
            self._order.append(transition.transition_id)
            self._items[transition.transition_id] = transition
            self._dataset_ids[transition.dataset_idx].append(transition.transition_id)
        self._next_id = int(state.get("next_id", len(self._order)))
        self._rng.setstate(state["rng_state"])
        self._evict_if_needed()


class DatasetUTDScheduler:
    """按数据集累计固定 update-to-data ratio，不为预热样本追补更新。"""

    def __init__(self, updates_per_transition: float) -> None:
        ratio = float(updates_per_transition)
        if ratio < 0.0:
            raise ValueError("updates_per_transition 不能小于 0")
        self.updates_per_transition = ratio
        self._credits: dict[int, float] = defaultdict(float)
        self.transitions_after_warmup = 0
        self.scheduled_updates = 0

    def record_transition(self, dataset_idx: int, *, replay_ready: bool) -> int:
        if not replay_ready or self.updates_per_transition <= 0.0:
            return 0
        dataset_idx = int(dataset_idx)
        self.transitions_after_warmup += 1
        credit = self._credits[dataset_idx] + self.updates_per_transition
        updates = int(credit + 1e-12)
        self._credits[dataset_idx] = credit - updates
        self.scheduled_updates += updates
        return updates

    @property
    def effective_utd(self) -> float:
        if self.transitions_after_warmup <= 0:
            return 0.0
        return float(self.scheduled_updates) / float(self.transitions_after_warmup)

    def state_dict(self) -> dict[str, Any]:
        return {
            "updates_per_transition": self.updates_per_transition,
            "credits": dict(self._credits),
            "transitions_after_warmup": self.transitions_after_warmup,
            "scheduled_updates": self.scheduled_updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        saved_ratio = float(state.get("updates_per_transition", self.updates_per_transition))
        if abs(saved_ratio - self.updates_per_transition) > 1e-12:
            raise ValueError(
                "UTD 与 sidecar 不一致: "
                f"current={self.updates_per_transition}, saved={saved_ratio}"
            )
        self._credits = defaultdict(
            float,
            {int(key): float(value) for key, value in state.get("credits", {}).items()},
        )
        self.transitions_after_warmup = int(state.get("transitions_after_warmup", 0))
        self.scheduled_updates = int(state.get("scheduled_updates", 0))


__all__ = ["DDQNTransition", "DatasetReplayBuffer", "DatasetUTDScheduler"]
