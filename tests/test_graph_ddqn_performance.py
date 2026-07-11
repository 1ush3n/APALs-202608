from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest

from baselines.literature_dqn.train_graph_ddqn_apal import (
    GraphDDQNAgent,
    _load_exact_resume_state,
    _save_exact_resume_state,
)
from baselines.literature_dqn.replay import DatasetReplayBuffer, DatasetUTDScheduler
from configs import configs
from tests.runtime_safety import temporary_config
from tests.test_graph_rl_baselines import _graph


def _append(buffer: DatasetReplayBuffer, dataset_idx: int, marker: int) -> None:
    snapshot = {"dataset_idx": dataset_idx, "marker": marker}
    buffer.append(
        dataset_idx=dataset_idx,
        state_snapshot=snapshot,
        action=(0, 0, [0]),
        reward=float(marker),
        next_snapshot={"dataset_idx": dataset_idx, "marker": marker + 1},
        done=False,
        masks=(marker, marker, marker),
        next_masks=(marker + 1, marker + 1, marker + 1),
    )


def test_dataset_replay_buffer_global_fifo_and_dataset_index() -> None:
    buffer = DatasetReplayBuffer(capacity=4, seed=42)
    _append(buffer, 0, 0)
    _append(buffer, 1, 1)
    _append(buffer, 0, 2)
    _append(buffer, 1, 3)
    _append(buffer, 0, 4)

    assert len(buffer) == 4
    assert buffer.count(0) == 2
    assert buffer.count(1) == 2
    assert [item.transition_id for item in buffer.ordered_transitions()] == [1, 2, 3, 4]
    assert all(item.dataset_idx == 0 for item in buffer.sample(0, 2))


def test_dataset_replay_buffer_state_restores_sampling_rng() -> None:
    original = DatasetReplayBuffer(capacity=8, seed=7)
    for marker in range(8):
        _append(original, marker % 2, marker)
    state = original.state_dict()

    expected = [item.transition_id for item in original.sample(0, 3)]
    restored = DatasetReplayBuffer(capacity=8, seed=999)
    restored.load_state_dict(state)
    actual = [item.transition_id for item in restored.sample(0, 3)]

    assert actual == expected


def test_utd_scheduler_has_no_warmup_debt_and_is_dataset_scoped() -> None:
    scheduler = DatasetUTDScheduler(0.125)
    assert sum(
        scheduler.record_transition(0, replay_ready=False)
        for _ in range(256)
    ) == 0
    assert sum(
        scheduler.record_transition(0, replay_ready=True)
        for _ in range(8)
    ) == 1
    assert sum(
        scheduler.record_transition(1, replay_ready=True)
        for _ in range(4)
    ) == 0
    assert sum(
        scheduler.record_transition(0, replay_ready=True)
        for _ in range(4)
    ) == 0
    assert sum(
        scheduler.record_transition(1, replay_ready=True)
        for _ in range(4)
    ) == 1
    assert sum(
        scheduler.record_transition(0, replay_ready=True)
        for _ in range(4)
    ) == 1
    assert scheduler.effective_utd == 0.125


def test_utd_scheduler_state_round_trip() -> None:
    scheduler = DatasetUTDScheduler(0.25)
    for _ in range(6):
        scheduler.record_transition(3, replay_ready=True)
    state = scheduler.state_dict()
    restored = DatasetUTDScheduler(0.25)
    restored.load_state_dict(state)

    assert restored.record_transition(3, replay_ready=True) == 0
    assert restored.record_transition(3, replay_ready=True) == 1
    assert restored.scheduled_updates == 2


class _GraphSnapshotEnv:
    def __init__(self, graphs: list) -> None:
        self.graphs = graphs

    def rebuild_state_from_snapshot(self, snapshot: dict):
        return self.graphs[int(snapshot["marker"])].clone()


def _agent_args(*, batched: bool) -> SimpleNamespace:
    return SimpleNamespace(
        epsilon=0.0,
        epsilon_min=0.0,
        epsilon_decay=1.0,
        replay_start_size=2,
        memory_size=8,
        ddqn_enable_batched_replay=batched,
        ddqn_enable_profiler=False,
        ddqn_profile_interval_updates=20,
        ddqn_enable_gpu_batch_rebuild=False,
        train_data_path_or_dir="data/283.csv",
        data_path="data/283.csv",
    )


def test_batched_replay_matches_serial_replay_on_cpu() -> None:
    overrides = {
        "seed": 42,
        "hidden_dim": 16,
        "num_gat_layers": 1,
        "num_heads": 1,
        "use_skill_hub": True,
        "skill_hub_bidirectional": True,
        "use_input_layer_norm": False,
        "use_gat_layer_norm": False,
        "use_head_layer_norm": False,
        "lr": 1e-4,
    }
    with temporary_config(configs, overrides):
        torch.manual_seed(123)
        batched_agent = GraphDDQNAgent(_agent_args(batched=True), torch.device("cpu"))
        serial_agent = GraphDDQNAgent(_agent_args(batched=False), torch.device("cpu"))
        serial_agent.model.load_state_dict(batched_agent.model.state_dict())
        serial_agent.target_model.load_state_dict(batched_agent.target_model.state_dict())

        graphs = [_graph(3, 4), _graph(3, 4), _graph(3, 4)]
        env = _GraphSnapshotEnv(graphs)
        masks = (
            torch.tensor([False, True, True]),
            torch.zeros(3, 2, dtype=torch.bool),
            torch.zeros(4, dtype=torch.bool),
        )
        for marker in range(2):
            for agent in (batched_agent, serial_agent):
                agent.remember(
                    {"dataset_idx": 0, "marker": marker},
                    (0, 0, [0]),
                    -0.1 * (marker + 1),
                    {"dataset_idx": 0, "marker": marker + 1},
                    False,
                    masks,
                    masks,
                )

        batched_loss = batched_agent.replay(env, 2, dataset_idx=0)
        serial_loss = serial_agent.replay(env, 2, dataset_idx=0)

        assert batched_loss == pytest.approx(serial_loss, rel=1e-5, abs=1e-6)
        for batched_param, serial_param in zip(
            batched_agent.model.parameters(),
            serial_agent.model.parameters(),
        ):
            assert torch.allclose(batched_param, serial_param, rtol=1e-5, atol=1e-6)


def test_exact_resume_sidecar_restores_buffer_scheduler_and_rng(tmp_path) -> None:
    overrides = {
        "seed": 51,
        "hidden_dim": 16,
        "num_gat_layers": 1,
        "num_heads": 1,
        "use_skill_hub": True,
        "skill_hub_bidirectional": True,
        "use_input_layer_norm": False,
        "use_gat_layer_norm": False,
        "use_head_layer_norm": False,
    }
    args = _agent_args(batched=True)
    with temporary_config(configs, overrides):
        source = GraphDDQNAgent(args, torch.device("cpu"))
        scheduler = DatasetUTDScheduler(0.125)
        masks = (torch.zeros(1, dtype=torch.bool),) * 3
        source.remember(
            {"dataset_idx": 0, "marker": 0},
            (0, 0, [0]),
            -0.1,
            {"dataset_idx": 0, "marker": 1},
            False,
            masks,
            masks,
        )
        for _ in range(8):
            scheduler.record_transition(0, replay_ready=True)
        _save_exact_resume_state(
            tmp_path,
            source,
            scheduler,
            123.5,
            args,
            episode=3,
        )
        expected_np = source.action_np_rng.random()
        expected_py = source.action_py_rng.random()

        restored = GraphDDQNAgent(args, torch.device("cpu"))
        restored_scheduler = DatasetUTDScheduler(0.125)
        result = _load_exact_resume_state(tmp_path, restored, restored_scheduler)

        assert result == (4, 123.5)
        assert len(restored.memory) == 1
        assert restored_scheduler.scheduled_updates == 1
        assert restored.action_np_rng.random() == expected_np
        assert restored.action_py_rng.random() == expected_py
