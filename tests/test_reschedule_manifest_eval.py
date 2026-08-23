from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import scripts.evaluate_reschedule_manifest as target


def test_manifest_evaluation_preserves_manifest_and_current_instance(monkeypatch, tmp_path: Path):
    entries = {}
    for instance_id in target.REAL_INSTANCE_IDS:
        entries[instance_id] = SimpleNamespace(
            instance_id=instance_id,
            data_path=tmp_path / f"{instance_id}.csv",
            baseline_schedule_path=tmp_path / f"{instance_id}_schedule.csv",
            scenario_path=tmp_path / f"{instance_id}_scenarios.csv",
            data_sha256="data",
            baseline_sha256="baseline",
            scenario_sha256="scenario",
            scenario_metadata_path=None,
            scenario_metadata_sha256="",
            num_tasks=1,
            baseline_makespan=1.0,
        )

    manifest_path = tmp_path / "manifest.json"
    manifest = SimpleNamespace(
        path=manifest_path,
        payload={"reschedule_protocol": "r5_task_delay_v1"},
        get=entries.__getitem__,
    )

    monkeypatch.setattr(target, "load_checkpoint", lambda path: SimpleNamespace(model_spec=None, format_name="test"))
    monkeypatch.setattr(target, "apply_checkpoint_model_spec", lambda *args, **kwargs: None)
    monkeypatch.setattr(target, "load_reschedule_manifest", lambda path: manifest)
    monkeypatch.setattr(target, "validate_r5_manifest_assets", lambda value: None)
    monkeypatch.setattr(target, "apply_initial_worker_mapping", lambda *args, **kwargs: None)

    def fake_evaluate(**kwargs):
        assert target.configs.reschedule_manifest_path == str(manifest_path)
        assert target.configs.reschedule_eval_instance_id == kwargs["output_dir"].name
        return {
            "scenario_count": 9,
            "avg_makespan": 1.0,
            "avg_score": 1.0,
            "avg_selection_score": 1.0,
            "eligible_rate": 1.0,
            "avg_duration_sec": 0.1,
            "worker_util": 0.0,
            "station_util": 0.0,
        }

    monkeypatch.setattr(target, "evaluate_saved_reschedule_model", fake_evaluate)
    target.evaluate_manifest_instances(
        model_path=tmp_path / "model.ckpt",
        manifest_path=manifest_path,
        instance_ids=list(target.REAL_INSTANCE_IDS),
        num_runs=None,
        scenario_ids=None,
        temperature=0.0,
        output_dir=tmp_path / "out",
    )
