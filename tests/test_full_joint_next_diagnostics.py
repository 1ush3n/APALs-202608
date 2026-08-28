from __future__ import annotations

from diagnostics.next_suite import build_jobs, build_train_command


def test_next_diagnostic_suite_has_non_overlapping_t5_and_t6_jobs() -> None:
    jobs = build_jobs(seeds=(42, 43, 44))

    assert len(jobs) == 7
    assert len({job.run_name for job in jobs}) == len(jobs)
    assert sum(job.suite == "t5" for job in jobs) == 6
    assert sum(job.suite == "t6" for job in jobs) == 1


def test_v2_job_keeps_full_action_scope_and_disables_dynamic_eft() -> None:
    jobs = build_jobs(seeds=(42,))
    v2 = next(job for job in jobs if job.variant == "v2")

    assert "model.team_selection_mode=autoregressive_pressure_v2" in v2.overrides
    assert "model.policy_action_scope=operation_station_worker" in v2.overrides
    assert "model.policy_observation_scope=full" in v2.overrides
    assert "model.worker_pointer_v2_dynamic_eft_features=false" in v2.overrides
    assert "model.worker_pointer_v2_replay_mode=behavior_group_exact_v1" in v2.overrides


def test_t6_job_only_removes_global_attention_context() -> None:
    jobs = build_jobs(seeds=(42,))
    t6 = next(job for job in jobs if job.suite == "t6")

    assert t6.variant == "local_only"
    assert "model.team_selection_mode=autoregressive" in t6.overrides
    assert "model.policy_action_scope=operation_station_worker" in t6.overrides
    assert "model.policy_observation_scope=full" in t6.overrides
    assert "model.actor_context_mode=local_only" in t6.overrides


def test_train_command_is_hydra_key_value_only() -> None:
    job = build_jobs(seeds=(42,))[0]
    command = build_train_command(job, max_episodes=20, num_envs=4, batch_size=32)

    assert command[0].endswith("python.exe") or command[0] == "python"
    assert command[1:] == [
        "train.py",
        "experiment=full_joint_diagnostic",
        "experiment.experiment_name=full_joint_t5_legacy_seed42",
        "seed=42",
        "num_envs=4",
        "train.max_episodes=20",
        "train.batch_size=32",
        "train.accumulation_steps=2",
        "train.async_eval_enabled=false",
        "model.team_selection_mode=autoregressive",
        "model.policy_action_scope=operation_station_worker",
        "model.policy_observation_scope=full",
        "model.actor_context_mode=attention",
    ]
