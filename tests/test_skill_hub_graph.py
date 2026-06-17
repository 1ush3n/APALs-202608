from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import configs
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from utils.gpu_graph_manager import GPUBatchGraphManager
from utils.resource_graph import (
    apply_batched_resource_graph,
    apply_resource_graph,
    build_skill_hub_topology,
)


DATA_PATH = PROJECT_ROOT / "data" / "283.csv"


@contextmanager
def resource_graph_mode(*, use_skill_hub: bool, bidirectional: bool):
    old_hub = configs.use_skill_hub
    old_bidirectional = configs.skill_hub_bidirectional
    try:
        configs.use_skill_hub = use_skill_hub
        configs.skill_hub_bidirectional = bidirectional
        yield
    finally:
        configs.use_skill_hub = old_hub
        configs.skill_hub_bidirectional = old_bidirectional


def _build_observation(*, use_skill_hub: bool, bidirectional: bool):
    with resource_graph_mode(
        use_skill_hub=use_skill_hub,
        bidirectional=bidirectional,
    ):
        env = AirLineEnv_Graph(DATA_PATH)
        observation = env.reset(seed=42)
        return env, observation


def test_default_config_enables_bidirectional_skill_hub() -> None:
    assert configs.use_skill_hub is True
    assert configs.skill_hub_bidirectional is True
    assert configs.num_skill_types == 10
    assert configs.skill_feat_dim == configs.num_skill_types + 6


def test_legacy_resource_graph_remains_available() -> None:
    _env, observation = _build_observation(
        use_skill_hub=False,
        bidirectional=False,
    )

    assert "skill" not in observation.node_types
    assert ("worker", "can_do", "task") in observation.edge_types
    worker_skills = observation["worker"].x[:, 1:11] > 0.5
    task_skills = torch.argmax(observation["task"].x[:, 5:15], dim=1)
    expected_direct_edges = int(worker_skills[:, task_skills].sum().item())
    assert observation["worker", "can_do", "task"].edge_index.size(1) == expected_direct_edges
    assert observation["task", "precedes", "task"].edge_index.size(1) == 753


def test_skill_hub_forward_and_bidirectional_modes() -> None:
    _env, forward = _build_observation(
        use_skill_hub=True,
        bidirectional=False,
    )
    assert forward["skill"].x.shape == (10, 16)
    expected_worker_skill_edges = int((forward["worker"].x[:, 1:11] > 0.5).sum().item())
    expected_task_skill_edges = int(forward["task"].x.size(0))
    assert forward["worker", "has_skill", "skill"].edge_index.size(1) == expected_worker_skill_edges
    assert forward["skill", "required_by", "task"].edge_index.size(1) == expected_task_skill_edges
    assert ("task", "requires", "skill") not in forward.edge_types
    assert ("skill", "provided_by", "worker") not in forward.edge_types
    assert ("worker", "can_do", "task") not in forward.edge_types

    _env, bidirectional = _build_observation(
        use_skill_hub=True,
        bidirectional=True,
    )
    assert bidirectional["task", "requires", "skill"].edge_index.size(1) == int(
        bidirectional["task"].x.size(0)
    )
    assert bidirectional["skill", "provided_by", "worker"].edge_index.size(1) == int(
        (bidirectional["worker"].x[:, 1:11] > 0.5).sum().item()
    )


def test_skill_hub_snapshot_and_batched_rebuild_match() -> None:
    with resource_graph_mode(use_skill_hub=True, bidirectional=True):
        env = AirLineEnv_Graph(DATA_PATH)
        env.reset(seed=42)
        snapshot = env.get_state_snapshot()

        cpu_graph = env.rebuild_state_from_snapshot(snapshot)
        manager = GPUBatchGraphManager(torch.device("cpu"))
        batch_graph = manager.batched_rebuild_on_gpu([snapshot], env)

        assert torch.allclose(cpu_graph["skill"].x, batch_graph["skill"].x)
        for edge_type in (
            ("task", "precedes", "task"),
            ("worker", "has_skill", "skill"),
            ("skill", "required_by", "task"),
            ("task", "requires", "skill"),
            ("skill", "provided_by", "worker"),
        ):
            assert torch.equal(
                cpu_graph[edge_type].edge_index,
                batch_graph[edge_type].edge_index,
            )
        assert batch_graph["skill"].batch.shape == (configs.num_skill_types,)


def test_cached_skill_hub_is_exactly_equal_to_reference_build() -> None:
    with resource_graph_mode(use_skill_hub=True, bidirectional=True):
        old_min_workers = configs.n_w_min
        old_workers = configs.n_w
        try:
            configs.n_w_min = 60
            configs.n_w = 80
            env = AirLineEnv_Graph(DATA_PATH)
            observation = env.reset(randomize_workers=True, seed=123)
        finally:
            configs.n_w_min = old_min_workers
            configs.n_w = old_workers

        reference = observation.clone()
        cached = observation.clone()
        apply_resource_graph(
            reference,
            observation["task"].x,
            observation["worker"].x,
            configs,
        )
        topology = build_skill_hub_topology(
            observation["task"].x,
            observation["worker"].x,
            configs.num_skill_types,
        )
        apply_resource_graph(
            cached,
            observation["task"].x,
            observation["worker"].x,
            configs,
            skill_hub_topology=topology,
        )

        assert torch.equal(reference["skill"].x, cached["skill"].x)
        for edge_type in (
            ("worker", "has_skill", "skill"),
            ("skill", "required_by", "task"),
            ("task", "requires", "skill"),
            ("skill", "provided_by", "worker"),
        ):
            assert torch.equal(
                reference[edge_type].edge_index,
                cached[edge_type].edge_index,
            )


def test_cached_batched_skill_hub_matches_reference_edge_order() -> None:
    with resource_graph_mode(use_skill_hub=True, bidirectional=True):
        env = AirLineEnv_Graph(DATA_PATH)
        first = env.reset(randomize_workers=False, seed=11)
        second = env.reset(randomize_workers=False, seed=12)
        task_x = torch.cat((first["task"].x, second["task"].x), dim=0)
        worker_x = torch.cat((first["worker"].x, second["worker"].x), dim=0)

        reference = first.clone()
        cached = first.clone()
        apply_batched_resource_graph(
            reference,
            task_x,
            worker_x,
            batch_size=2,
            num_tasks=env.num_tasks,
            num_workers=env.num_workers,
            config=configs,
        )
        first_topology = build_skill_hub_topology(
            first["task"].x,
            first["worker"].x,
            configs.num_skill_types,
        )
        second_topology = build_skill_hub_topology(
            second["task"].x,
            second["worker"].x,
            configs.num_skill_types,
        )
        apply_batched_resource_graph(
            cached,
            task_x,
            worker_x,
            batch_size=2,
            num_tasks=env.num_tasks,
            num_workers=env.num_workers,
            config=configs,
            task_skill_edges=first_topology.skill_to_task,
            worker_skill_edges=[
                first_topology.worker_to_skill,
                second_topology.worker_to_skill,
            ],
        )

        assert torch.equal(reference["skill"].x, cached["skill"].x)
        for edge_type in (
            ("worker", "has_skill", "skill"),
            ("skill", "required_by", "task"),
            ("task", "requires", "skill"),
            ("skill", "provided_by", "worker"),
        ):
            assert torch.equal(
                reference[edge_type].edge_index,
                cached[edge_type].edge_index,
            )


def test_model_forward_supports_all_resource_graph_modes() -> None:
    for use_skill_hub, bidirectional in (
        (False, False),
        (True, False),
        (True, True),
    ):
        with resource_graph_mode(
            use_skill_hub=use_skill_hub,
            bidirectional=bidirectional,
        ):
            env = AirLineEnv_Graph(DATA_PATH)
            observation = env.reset(seed=42)
            model = HBGATPN(configs).eval()
            with torch.inference_mode():
                encoded, context = model(observation)

            assert encoded["task"].shape == (env.num_tasks, configs.hidden_dim)
            assert encoded["worker"].shape == (env.num_workers, configs.hidden_dim)
            assert context.shape == (1, configs.hidden_dim * 3)
            if use_skill_hub:
                assert encoded["skill"].shape == (
                    configs.num_skill_types,
                    configs.hidden_dim,
                )
