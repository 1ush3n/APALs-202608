from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from baselines.graph_baseline import SimpleWorkerHead
from runtime.modes import is_worker_pointer_v2_mode
from runtime.hydra_config import _parse_value

import scripts.evaluate_l2d_ppo_r5_manifest as target
from baselines.literature_ppo.train_l2d_ppo_apal import preflight_l2d_ppo_r5_config


def _legacy_r5_config() -> SimpleNamespace:
    return SimpleNamespace(
        reschedule_async_protocol="r5_task_delay_v1",
        team_selection_mode="autoregressive",
        worker_pointer_v2_dynamic_eft_features=False,
        task_feat_dim=24,
        worker_feat_dim=17,
        num_skill_types=5,
        worker_skill_feature_slots=5,
        reschedule_baseline_model_path="",
    )


def _formal_checkpoint() -> dict[str, object]:
    return {
        "checkpoint_format": "literature_baseline_v2",
        "algorithm": "L2D-PPO-APAL",
        "literature_family": "learned_dispatching_rule_ppo",
        "model_type": "SimpleHeteroGATPPO",
        "implementation_variant": "apal_heterogat_joint_action_v1",
        "feature_mode": "apal_hetero_graph",
        "training_protocol": "r5_task_delay_v1",
        "initialization": "random",
        "team_selection_mode": "autoregressive",
        "worker_pointer_v2_dynamic_eft_features": False,
        "task_feat_dim": 24,
        "station_feat_dim": 15,
        "worker_feat_dim": 17,
        "skill_feat_dim": 11,
        "hidden_dim": 128,
        "num_gat_layers": 5,
        "num_heads": 4,
        "worker_feature_layout_version": "five_skill_v2",
        "worker_skill_feature_slots": 5,
        "reschedule_async_protocol": "r5_task_delay_v1",
        "experiment": "l2d_ppo_apal_r5",
        "manifest_path": "manifest.json",
        "manifest_sha256": "a" * 64,
        "formal_r5_checkpoint": True,
        "formal_r5_baseline": True,
        "selection_protocol": "r5_validation_only",
        "selection_instance_ids": ["validation_0001"],
        "selection_scenario_ids": ["low_early", "medium_early", "high_early"],
        "model_state_dict": {
            "embedder.task_emb.0.weight": torch.zeros(128, 24),
            "embedder.station_emb.0.weight": torch.zeros(128, 15),
            "embedder.worker_emb.0.weight": torch.zeros(128, 17),
            "embedder.skill_emb.0.weight": torch.zeros(128, 11),
        },
        "optimizer_state_dict": {},
    }


def test_generic_r5_mode_reproduces_missing_v2_worker_interface() -> None:
    bad_config = SimpleNamespace(team_selection_mode="autoregressive_pressure_v2")
    assert is_worker_pointer_v2_mode(bad_config)
    with pytest.raises(AttributeError, match="initialize_v2_state"):
        SimpleWorkerHead(8).initialize_v2_state(batch_size=1, device=torch.device("cpu"))


def test_l2d_preflight_rejects_v2_and_accepts_legacy_r5() -> None:
    config = _legacy_r5_config()
    assert preflight_l2d_ppo_r5_config(config) is True

    config.team_selection_mode = "autoregressive_pressure_v2"
    with pytest.raises(ValueError, match="team_selection_mode"):
        preflight_l2d_ppo_r5_config(config)


def test_hydra_empty_path_override_remains_empty_string() -> None:
    assert _parse_value("") == ""



def test_l2d_checkpoint_is_strictly_formal_or_auxiliary() -> None:
    checkpoint = _formal_checkpoint()
    result = target.validate_l2d_ppo_r5_checkpoint(
        checkpoint,
        observation_dims={"task": 24, "station": 15, "worker": 17, "skill": 11},
        require_formal=True,
    )
    assert result["formal_r5_checkpoint"] is True

    old = dict(checkpoint)
    old["algorithm"] = "Simple-HeteroGAT-PPO"
    old.pop("training_protocol")
    old.pop("initialization")
    old.pop("formal_r5_checkpoint")
    old.pop("formal_r5_baseline")
    result = target.validate_l2d_ppo_r5_checkpoint(
        old,
        observation_dims={"task": 24, "station": 15, "worker": 17, "skill": 11},
        require_formal=False,
    )
    assert result["formal_r5_checkpoint"] is False
    assert result["comparison_role"] == "auxiliary_initial_checkpoint"

    old_shape = _formal_checkpoint()
    old_shape["task_feat_dim"] = 18
    old_shape["model_state_dict"] = dict(old_shape["model_state_dict"])
    old_shape["model_state_dict"]["embedder.task_emb.0.weight"] = torch.zeros(128, 18)
    with pytest.raises(ValueError, match="task feature dimension mismatch|task_feat_dim"):
        target.validate_l2d_ppo_r5_checkpoint(
            old_shape,
            observation_dims={"task": 24, "station": 15, "worker": 17, "skill": 11},
            require_formal=True,
        )


def test_l2d_manifest_scenario_contract_and_partial_export() -> None:
    assert target.expected_r5_scenario_ids() == (
        "low_early",
        "medium_early",
        "high_early",
        "low_middle",
        "medium_middle",
        "high_middle",
        "low_late",
        "medium_late",
        "high_late",
    )
    rows = target.serialize_scenario_schedule(
        instance_id="real_283",
        scenario_id="low_early",
        schedule=[(3, 1, [7, 9], 12.5, 18.0)],
    )
    assert rows[0]["worker_ids"] == "[7, 9]"
    flags = target.summarize_r5_outcomes(
        [{"complete": 0.0, "eligible": 0.0}],
        execution_complete=True,
        audit_ok=True,
        formal_r5_checkpoint=False,
    )
    assert flags["execution_complete"] is True
    assert flags["all_scenarios_complete"] is False
    assert flags["all_scenarios_eligible"] is False
    assert flags["strict_main_table_eligible"] is False

def test_l2d_manifest_entry_rejects_cpu_before_loading_assets(monkeypatch) -> None:
    monkeypatch.setattr(target.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="只允许 CUDA"):
        target.evaluate_l2d_ppo_r5_manifest(
            model_path=Path("missing.pth"),
            manifest_path=Path("missing.json"),
        )
