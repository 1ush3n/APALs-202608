from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import HeteroData

from baselines.evaluate_flat_rl_baseline import _load_checkpoint
from baselines.graph_baseline import (
    GraphBaselineActorCritic,
    select_graph_action,
    select_graph_actions_batch,
)
from configs import Config


def _make_config() -> Config:
    cfg = Config()
    cfg.hidden_dim = 16
    cfg.num_gat_layers = 1
    cfg.num_heads = 1
    cfg.use_skill_hub = True
    cfg.skill_hub_bidirectional = True
    cfg.use_input_layer_norm = False
    cfg.use_gat_layer_norm = False
    cfg.use_head_layer_norm = False
    return cfg


def _edge(src: list[int], dst: list[int]) -> torch.Tensor:
    return torch.tensor([src, dst], dtype=torch.long)


def _graph(num_tasks: int, num_workers: int = 4, num_stations: int = 2) -> HeteroData:
    data = HeteroData()
    data["task"].x = torch.zeros(num_tasks, 18)
    data["task"].x[:, 5] = 1.0
    data["task"].x[:, 16] = 1.0
    data["worker"].x = torch.zeros(num_workers, 22)
    data["worker"].x[:, 1] = 1.0
    data["worker"].x[:, 13] = 1.0
    data["station"].x = torch.zeros(num_stations, 15)
    data["skill"].x = torch.zeros(10, 16)
    data["skill"].x[:, 0] = 1.0

    if num_tasks > 1:
        data["task", "precedes", "task"].edge_index = _edge(list(range(num_tasks - 1)), list(range(1, num_tasks)))
    else:
        data["task", "precedes", "task"].edge_index = torch.empty(2, 0, dtype=torch.long)
    data["task", "assigned_to", "station"].edge_index = torch.empty(2, 0, dtype=torch.long)
    data["station", "has_task", "task"].edge_index = torch.empty(2, 0, dtype=torch.long)
    data["task", "done_by", "worker"].edge_index = torch.empty(2, 0, dtype=torch.long)
    data["worker", "has_skill", "skill"].edge_index = _edge(list(range(num_workers)), [0] * num_workers)
    data["skill", "required_by", "task"].edge_index = _edge([0] * num_tasks, list(range(num_tasks)))
    data["task", "requires", "skill"].edge_index = _edge(list(range(num_tasks)), [0] * num_tasks)
    data["skill", "provided_by", "worker"].edge_index = _edge([0] * num_workers, list(range(num_workers)))
    return data


def test_graph_baseline_outputs_follow_graph_size() -> None:
    model = GraphBaselineActorCritic(_make_config())
    small = _graph(3)
    large = _graph(5)

    small_x, small_ctx = model(small)
    large_x, large_ctx = model(large)

    assert small_x["task"].shape == (3, 16)
    assert large_x["task"].shape == (5, 16)
    assert small_ctx.shape == (1, 48)
    assert large_ctx.shape == (1, 48)


def test_select_graph_action_uses_masks_and_variable_workers() -> None:
    model = GraphBaselineActorCritic(_make_config())
    graph = _graph(4, num_workers=5)
    task_mask = torch.tensor([True, False, False, False])
    station_mask = torch.zeros(4, 2, dtype=torch.bool)
    worker_mask = torch.zeros(5, dtype=torch.bool)

    result = select_graph_action(
        model,
        graph,
        masks=(task_mask, station_mask, worker_mask),
        device=torch.device("cpu"),
        deterministic=True,
    )

    assert result.action is not None
    task_idx, station_idx, team = result.action
    assert task_idx != 0
    assert 0 <= station_idx < 2
    assert len(team) == 1


def test_batched_graph_action_matches_serial_for_variable_graphs() -> None:
    torch.manual_seed(42)
    model = GraphBaselineActorCritic(_make_config())
    model.eval()
    graphs = [_graph(3, num_workers=4), _graph(5, num_workers=6)]
    masks = [
        (
            torch.tensor([True, False, False]),
            torch.zeros(3, 2, dtype=torch.bool),
            torch.zeros(4, dtype=torch.bool),
        ),
        (
            torch.tensor([False, True, False, False, False]),
            torch.zeros(5, 2, dtype=torch.bool),
            torch.zeros(6, dtype=torch.bool),
        ),
    ]

    serial = [
        select_graph_action(
            model,
            graph,
            masks=graph_masks,
            device=torch.device("cpu"),
            deterministic=True,
        )
        for graph, graph_masks in zip(graphs, masks)
    ]
    batched = select_graph_actions_batch(
        model,
        graphs,
        masks_list=masks,
        device=torch.device("cpu"),
        deterministic=True,
    )

    assert [result.action for result in batched] == [result.action for result in serial]


def test_old_flat_checkpoint_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "old_flat.pth"
    torch.save({"model_state_dict": {}, "state_dim": 10, "action_dim_list": [1, 1, 1]}, path)

    try:
        _load_checkpoint(path)
    except ValueError as exc:
        assert "flat-state" in str(exc)
    else:
        raise AssertionError("旧 flat-state checkpoint 不应被图 baseline 评估入口接受")
