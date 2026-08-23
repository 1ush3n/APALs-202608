from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import scripts.evaluate_graph_ddqn_r5_manifest as target
import baselines.literature.common as literature_common


def _formal_checkpoint(*, task_dim: int = 24) -> dict[str, object]:
    return {
        "algorithm": "Graph-DDQN-APAL",
        "literature_family": "graph_double_dqn",
        "model_type": "GraphDDQNAPAL",
        "feature_mode": "apal_hetero_graph",
        "task_feat_dim": task_dim,
        "station_feat_dim": 15,
        "worker_feat_dim": 17,
        "skill_feat_dim": 11,
        "hidden_dim": 128,
        "num_gat_layers": 5,
        "num_heads": 4,
        "worker_feature_layout_version": "five_skill_v2",
        "worker_skill_feature_slots": 5,
        "reschedule_async_protocol": "r5_task_delay_v1",
        "experiment": "reschedule_task_delay_r5",
        "formal_r5_baseline": True,
        "selection_protocol": "r5_validation_only",
        "selection_instance_ids": ["validation_0001"],
        "selection_scenario_ids": ["low_early", "medium_early", "high_early"],
        "model_state_dict": {
            "embedder.task_emb.0.weight": torch.zeros(128, task_dim),
            "embedder.station_emb.0.weight": torch.zeros(128, 15),
            "embedder.worker_emb.0.weight": torch.zeros(128, 17),
            "embedder.skill_emb.0.weight": torch.zeros(128, 11),
        },
    }


def test_old_initial_checkpoint_is_rejected_by_r5_shape_gate() -> None:
    checkpoint = _formal_checkpoint(task_dim=18)

    with pytest.raises(ValueError, match="task_feat_dim|24"):
        target.validate_graph_ddqn_r5_checkpoint(
            checkpoint,
            observation_dims={"task": 24, "station": 15, "worker": 17, "skill": 11},
        )


def test_r5_checkpoint_requires_formal_protocol_metadata() -> None:
    checkpoint = _formal_checkpoint()
    checkpoint.pop("experiment")

    with pytest.raises(ValueError, match="experiment"):
        target.validate_graph_ddqn_r5_checkpoint(
            checkpoint,
            observation_dims={"task": 24, "station": 15, "worker": 17, "skill": 11},
        )


def test_partial_schedule_is_exported_and_incomplete_result_is_retained() -> None:
    schedule = [(3, 1, [7, 9], 12.5, 18.0)]

    rows = target.serialize_scenario_schedule(
        instance_id="real_283",
        scenario_id="low_early",
        schedule=schedule,
    )
    flags = target.summarize_r5_outcomes(
        [
            {
                "instance_id": "real_283",
                "scenario_id": "low_early",
                "complete": 0.0,
                "eligible": 0.0,
                "makespan": 999.0,
                "selection_score": 999.0,
            }
        ],
        execution_complete=True,
        audit_ok=True,
    )

    assert rows == [
        {
            "instance_id": "real_283",
            "scenario_id": "low_early",
            "task_id": 3,
            "station_id": 1,
            "worker_ids": "[7, 9]",
            "start_time": 12.5,
            "finish_time": 18.0,
        }
    ]
    assert flags["execution_complete"] is True
    assert flags["all_scenarios_complete"] is False
    assert flags["all_scenarios_eligible"] is False


def test_adapter_contract_is_the_existing_literature_adapter(monkeypatch) -> None:
    called = {}

    class FakeResult:
        action = (1, 2, [3])
        logprob = None
        value = None

    def fake_select_graph_action(*args, **kwargs):
        called.update(kwargs)
        return FakeResult()

    monkeypatch.setattr(literature_common, "select_graph_action", fake_select_graph_action)
    model = SimpleNamespace()
    adapter = target.LiteraturePolicyAdapter(model, torch.device("cpu"))

    action = adapter.select_action(
        state=SimpleNamespace(),
        mask_task=torch.zeros(2, dtype=torch.bool),
        mask_station_matrix=torch.zeros(2, 2, dtype=torch.bool),
        mask_worker=torch.zeros(3, dtype=torch.bool),
        deterministic=True,
        temperature=0.0,
        is_eval=True,
    )

    assert action == ((1, 2, [3]), None, None, None, False)
    assert called["deterministic"] is True
    assert called["temperature"] == 0.0
