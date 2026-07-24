from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.literature_dqn.train_graph_ddqn_apal import (
    _dataset_index_for_episode,
    _training_randomization_flags,
)
from baselines.literature_ppo.train_l2d_ppo_apal import (
    _load_resume_checkpoint,
    _restore_service_dataset_state,
    _save_checkpoint,
    _service_dataset_state,
)
from training.lightning_module import RolloutMetrics
from training.memory import Memory
from training.rollout_service import APALRolloutService


class _FakeEnv:
    num_tasks = 3
    dataset_count = 7


class _FakeVectorEnv:
    def __init__(self, num_envs: int) -> None:
        self.num_envs = int(num_envs)
        self.envs = [_FakeEnv() for _ in range(self.num_envs)]
        self.switches: list[int] = []

    def switch_dataset_all(self, dataset_idx: int) -> None:
        self.switches.append(int(dataset_idx))

    def close(self) -> None:
        return None


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        seed=42,
        update_every_episodes=1,
        enable_rollout_ipc_fusion=False,
        adaptive_ppo_batch_by_tasks=False,
        batch_size=4,
        ppo_batch_size_cap=0,
        random_sample_dataset=True,
    )


def _fake_collect(self: APALRolloutService, episode: int):
    memories = [Memory() for _ in range(self.num_envs)]
    metric = RolloutMetrics(
        episode=int(episode),
        average_reward=float(episode),
        average_makespan=float(episode),
        completion_rate=1.0,
        environment_steps=self.num_envs,
        steps_per_second=1.0,
        total_seconds=0.01,
        ipc_mask_ms=0.0,
        forward_ms=0.0,
        rebuild_ms=0.0,
        environment_step_ms=0.0,
    )
    return memories, metric


def test_l2d_collect_is_one_update_with_shared_dataset(monkeypatch):
    vector_env = _FakeVectorEnv(num_envs=2)
    service = APALRolloutService(
        agent=SimpleNamespace(batch_size=4),
        vector_env=vector_env,
        eval_env=None,
        config=_config(),
        device=SimpleNamespace(type="cpu"),
    )
    monkeypatch.setattr(APALRolloutService, "_collect_episode", _fake_collect)

    update = service.collect(1)

    assert service._last_dataset_idx is not None
    assert service.num_envs == 2
    assert vector_env.switches == [service._last_dataset_idx]


def test_l2d_serial_mode_is_explicit_and_rng_resume_is_exact(monkeypatch):
    monkeypatch.setattr(APALRolloutService, "_collect_episode", _fake_collect)
    first = APALRolloutService(
        agent=SimpleNamespace(batch_size=4),
        vector_env=_FakeVectorEnv(num_envs=1),
        eval_env=None,
        config=_config(),
        device=SimpleNamespace(type="cpu"),
    )
    first.collect(1)
    state = _service_dataset_state(first)
    first.collect(2)
    expected = first._last_dataset_idx

    resumed = APALRolloutService(
        agent=SimpleNamespace(batch_size=4),
        vector_env=_FakeVectorEnv(num_envs=1),
        eval_env=None,
        config=_config(),
        device=SimpleNamespace(type="cpu"),
    )
    _restore_service_dataset_state(resumed, state)
    actual = resumed.collect(2)

    assert resumed.num_envs == 1
    assert expected == resumed._last_dataset_idx


def test_ddqn_dataset_selection_and_randomization_are_deterministic(monkeypatch):
    monkeypatch.setattr("baselines.literature_dqn.train_graph_ddqn_apal.configs.curriculum_episodes", 2)
    monkeypatch.setattr("baselines.literature_dqn.train_graph_ddqn_apal.configs.randomize_durations", True)

    selected = [
        _dataset_index_for_episode(7, episode, 42)
        for episode in (3, 3, 10)
    ]
    assert selected[0] == selected[1]
    assert _training_randomization_flags(1) == (False, False)
    assert _training_randomization_flags(3) == (True, True)


def test_l2d_checkpoint_resume_starts_at_next_episode(monkeypatch):
    import baselines.literature_ppo.train_l2d_ppo_apal as l2d_module

    class _Agent:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.policy = torch.nn.Linear(1, 1)
            self.optimizer = torch.optim.SGD(self.policy.parameters(), lr=0.1)
            self.scaler = torch.amp.GradScaler("cpu", enabled=False)
            self.batch_size = 4
            self.k_epochs = 2
            self.eps_clip = 0.2

    def _fake_save(path, *, algorithm, model, best_makespan, extra, **_kwargs):
        torch.save(
            {
                "algorithm": algorithm,
                "model_state_dict": model.state_dict(),
                "best_makespan": best_makespan,
                **extra,
            },
            path,
        )

    monkeypatch.setattr(l2d_module, "save_literature_checkpoint", _fake_save)
    monkeypatch.setattr(APALRolloutService, "_collect_episode", _fake_collect)
    service = APALRolloutService(
        agent=SimpleNamespace(batch_size=4),
        vector_env=_FakeVectorEnv(num_envs=1),
        eval_env=None,
        config=_config(),
        device=SimpleNamespace(type="cpu"),
    )
    service.collect(1)
    selector_state = _service_dataset_state(service)
    path = Path(__file__).with_name(".l2d_checkpoint_smoke.pth")
    agent = _Agent()
    try:
        _save_checkpoint(
            path,
            agent,
            12.5,
            SimpleNamespace(),
            episode=3,
            dataset_selector_state=selector_state,
        )

        restored_agent = _Agent()
        start_episode, best_makespan, restored_state = _load_resume_checkpoint(
            path,
            restored_agent,
        )
        assert start_episode == 4
        assert best_makespan == 12.5
        assert restored_state["rng_keys"] == selector_state["rng_keys"]
    finally:
        for artifact in (path, path.with_suffix(".meta.json")):
            if artifact.exists():
                artifact.unlink()
