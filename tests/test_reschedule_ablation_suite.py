from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.reschedule_ablation_suite import build_command_rows, write_plan


def _base_args(tmp_path: Path) -> Namespace:
    manifest = tmp_path / "manifest.json"
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    full = tmp_path / "full.ckpt"
    no_gat = tmp_path / "no_gat.ckpt"
    no_attention = tmp_path / "no_attention.ckpt"
    for path in (full, no_gat, no_attention):
        path.write_text("checkpoint", encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "reschedule_dataset_manifest",
                "instances": [
                    {
                        "instance_id": "real_680",
                        "data_path": "data/680.csv",
                        "baseline_schedule_path": "baseline.csv",
                        "scenario_path": "scenario.csv",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return Namespace(
        mode="plan",
        variants=["full", "no_gat", "no_attention"],
        instance_ids=["real_680"],
        seeds=[42],
        train_data_path_or_dir=str(train_dir),
        manifest_path=str(manifest),
        max_episodes=300,
        batch_size=64,
        eval_freq=1,
        num_envs=0,
        log_dir="/root/tf-logs",
        runs_root="runs",
        artifact_layout="runs",
        full_model_path=str(full),
        no_gat_model_path=str(no_gat),
        no_attention_model_path=str(no_attention),
        run_id_prefix="resched_ablation",
        output_dir=str(tmp_path / "out"),
        python_executable="python",
        extra_overrides=[],
        validate_paths=True,
        continue_on_error=False,
    )


def test_reschedule_ablation_commands_fix_environment_and_seed(tmp_path: Path) -> None:
    args = _base_args(tmp_path)
    rows = build_command_rows(args)
    assert [row.variant for row in rows] == [
        "full",
        "no_gat",
        "no_attention",
    ]
    for row in rows:
        assert "experiment=reschedule_task_delay" in row.command
        assert f"train_data_path_or_dir={args.train_data_path_or_dir}" in row.command
        assert f"reschedule_manifest_path={args.manifest_path}" in row.command
        assert "reschedule_eval_instance_id=real_680" in row.command
        assert "eval_freq=1" in row.command
        assert "batch_size=64" in row.command
        assert "seed=42" in row.command
        assert "log_dir=/root/tf-logs" in row.command
        assert "artifact_layout=runs" in row.command


def test_reschedule_ablation_variant_overrides_and_warm_starts(tmp_path: Path) -> None:
    args = _base_args(tmp_path)
    rows = {row.variant: row for row in build_command_rows(args)}

    assert rows["full"].overrides == ""
    assert rows["no_gat"].warm_start_path == str(tmp_path / "no_gat.ckpt")
    assert "graph_encoder_mode=none" in rows["no_gat"].command
    assert rows["no_attention"].warm_start_path == str(tmp_path / "no_attention.ckpt")
    assert "actor_context_mode=mean_max" in rows["no_attention"].command


@pytest.mark.parametrize("variant", ["no_mask", "no_pointer", "no_skill_hub", "no_attention_pooling"])
def test_reschedule_ablation_explicitly_rejects_unavailable_variants(tmp_path: Path, variant: str) -> None:
    args = _base_args(tmp_path)
    args.variants = [variant]
    with pytest.raises(ValueError, match="当前不可执行"):
        build_command_rows(args)


def test_reschedule_ablation_rejects_missing_manifest_instance(tmp_path: Path) -> None:
    args = _base_args(tmp_path)
    args.instance_ids = ["real_3182"]
    with pytest.raises(ValueError, match="manifest 中缺少实例"):
        build_command_rows(args)


def test_reschedule_ablation_write_plan_outputs_auditable_files(tmp_path: Path) -> None:
    args = _base_args(tmp_path)
    args.variants = ["full"]
    rows = build_command_rows(args)
    output_dir = tmp_path / "plan"
    write_plan(output_dir, rows, args)
    assert (output_dir / "reschedule_ablation_command_plan.csv").exists()
    assert (output_dir / "reschedule_ablation_command_plan.json").exists()
    assert (output_dir / "reschedule_ablation_suite_config.json").exists()
    script = (output_dir / "run_reschedule_ablation_suite.sh").read_text(encoding="utf-8")
    assert "CUBLAS_WORKSPACE_CONFIG" in script


def test_strict_variants_freeze_scopes_completion_masks_and_bic(tmp_path: Path) -> None:
    args = _base_args(tmp_path)
    args.variants = [
        "operation_only_strict",
        "operation_station_strict",
        "homogeneous_graphsage_strict",
    ]

    rows = {row.variant: row for row in build_command_rows(args)}

    assert "policy_action_scope=operation" in rows["operation_only_strict"].command
    assert "policy_observation_scope=task" in rows["operation_only_strict"].command
    assert "action_completion_mode=min_wait" in rows["operation_only_strict"].command
    assert "task_feature_scope=intrinsic" in rows["operation_only_strict"].command
    assert "task_mask_mode=precedence_release_only" in rows["operation_only_strict"].command
    assert "station_mask_mode=structural_only" in rows["operation_only_strict"].command

    assert "policy_action_scope=operation_station" in rows["operation_station_strict"].command
    assert "policy_observation_scope=task_station" in rows["operation_station_strict"].command
    assert "action_completion_mode=min_wait" in rows["operation_station_strict"].command
    assert "station_mask_mode=structural_only" in rows["operation_station_strict"].command

    assert "graph_encoder_mode=homogeneous_graphsage_strict" in rows["homogeneous_graphsage_strict"].command
    assert "homogeneous_use_type_embedding=false" in rows["homogeneous_graphsage_strict"].command

    assert all(
        "reschedule_baseline_identity_conditioning=false" in row.command
        for row in rows.values()
    )
