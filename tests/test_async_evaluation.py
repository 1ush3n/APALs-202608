from __future__ import annotations

import os
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from configs import Config, configs
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from runtime.configuration import validate_runtime_config
from runtime.initial_checkpoint_selection import sha256_file, sha256_normalized_text_file
from runtime.reschedule_eval import evaluate_reschedule_model
from training.async_eval_worker import _record_result, _restore_interrupted_jobs, _verified_candidate_path
from training.async_evaluation import (
    AsyncEvaluationManager,
    AsyncEvalPaths,
    atomic_copy_file,
    atomic_link_or_copy,
    atomic_write_json,
    process_is_alive,
)
from training.observation import CachedEnvironmentObservation


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_selection_csv_hash_normalizes_windows_and_linux_newlines(tmp_path: Path) -> None:
    linux_csv = tmp_path / "linux.csv"
    windows_csv = tmp_path / "windows.csv"
    linux_csv.write_bytes(b"task,station\n1,2\n")
    windows_csv.write_bytes(b"task,station\r\n1,2\r\n")

    assert sha256_file(linux_csv) != sha256_file(windows_csv)
    assert sha256_normalized_text_file(linux_csv) == sha256_normalized_text_file(windows_csv)


def _assert_graph_equal(left, right) -> None:
    assert left.node_types == right.node_types
    assert left.edge_types == right.edge_types
    for node_type in left.node_types:
        assert torch.equal(left[node_type].x, right[node_type].x), node_type
    for edge_type in left.edge_types:
        assert torch.equal(left[edge_type].edge_index, right[edge_type].edge_index), edge_type


def test_async_eval_config_accepts_initial_standard_and_rejects_unsupported_options() -> None:
    config = Config(async_eval_enabled=True)
    validate_runtime_config(config)
    config.enable_multi_benchmark_eval = True
    validate_runtime_config(config)

    config.eval_scenarios = ["duration_noise"]
    with pytest.raises(ValueError, match="仅支持 eval_scenarios=\[standard\]"):
        validate_runtime_config(config)

    config.eval_scenarios = ["standard"]
    config.enable_reschedule_mode = True
    config.enable_multi_benchmark_eval = False
    config.async_eval_device = "cuda"
    validate_runtime_config(config)

    config.async_eval_worker_count = 2
    validate_runtime_config(config)

    config.async_eval_worker_count = 3
    with pytest.raises(ValueError, match="worker_count"):
        validate_runtime_config(config)

    config.async_eval_device = "cpu"
    config.async_eval_worker_count = 1
    config.eval_freq = 2
    with pytest.raises(ValueError, match="eval_freq=1"):
        validate_runtime_config(config)


def test_r5_reschedule_async_config_requires_cuda_and_allows_three_workers() -> None:
    config = Config(async_eval_enabled=True)
    config.enable_reschedule_mode = True
    config.reschedule_async_protocol = "r5_task_delay_v1"
    config.async_eval_device = "cuda"
    config.async_eval_worker_count = 3
    config.async_eval_queue_capacity = 3
    config.async_eval_submit_every_episodes = 2
    config.async_eval_allow_cpu_fallback = False
    config.async_eval_instance_id = "validation_0001"
    config.async_eval_scenario_ids = ["low_early", "medium_early", "high_early"]

    validate_runtime_config(config)

    config.eval_freq = 2
    validate_runtime_config(config)

    config.eval_freq = 3
    with pytest.raises(ValueError, match="eval_freq 仅支持 1 或 2"):
        validate_runtime_config(config)

    config.async_eval_device = "cpu"
    with pytest.raises(ValueError, match="CUDA"):
        validate_runtime_config(config)
    config.async_eval_device = "cuda"
    config.async_eval_queue_capacity = 4
    with pytest.raises(ValueError, match="队列容量必须为 3"):
        validate_runtime_config(config)


def test_r5_manager_submits_one_group_with_three_scenario_jobs(tmp_path: Path) -> None:
    config = Config(async_eval_enabled=True)
    config.enable_reschedule_mode = True
    config.reschedule_async_protocol = "r5_task_delay_v1"
    config.async_eval_device = "cuda"
    config.async_eval_worker_count = 3
    config.async_eval_queue_capacity = 3
    config.async_eval_allow_cpu_fallback = False
    config.async_eval_instance_id = "validation_0001"
    config.async_eval_scenario_ids = ["low_early", "medium_early", "high_early"]
    manager = AsyncEvaluationManager(
        config=config,
        latest_path=tmp_path / "checkpoints" / "last.ckpt",
        best_path=tmp_path / "checkpoints" / "best.ckpt",
        project_root=PROJECT_ROOT,
    )
    manager._wait_for_slot = lambda required_slots=1: None  # type: ignore[method-assign]
    manager._check_health = lambda: None  # type: ignore[method-assign]
    trainer = SimpleNamespace(save_checkpoint=lambda path: Path(path).write_bytes(b"checkpoint"))

    manager.submit(trainer, episode=2)

    pending = sorted(manager.paths.pending.glob("episode_000002_*.json"))
    jobs = [json.loads(path.read_text(encoding="utf-8-sig")) for path in pending]
    assert [job["scenario_id"] for job in jobs] == [
        "high_early",
        "low_early",
        "medium_early",
    ]
    assert {job["group_id"] for job in jobs} == {"episode_000002"}
    assert {tuple(job["group_scenario_ids"]) for job in jobs} == {
        ("low_early", "medium_early", "high_early")
    }
    assert all(job["reschedule_async_protocol"] == "r5_task_delay_v1" for job in jobs)


def test_r5_group_publishes_best_only_after_all_three_children_finish(tmp_path: Path) -> None:
    paths = AsyncEvalPaths.create(tmp_path / "async_eval")
    candidate = paths.candidates / "episode_000002.ckpt"
    candidate.write_bytes(b"candidate")
    writer = SimpleNamespace(add_scalar=lambda *args, **kwargs: None, flush=lambda: None)
    scenario_ids = ("low_early", "medium_early", "high_early")

    for index, scenario_id in enumerate(scenario_ids):
        job = {
            "episode": 2,
            "job_id": f"episode_000002_{scenario_id}",
            "group_id": "episode_000002",
            "group_scenario_ids": list(scenario_ids),
            "reschedule_async_protocol": "r5_task_delay_v1",
            "candidate_path": str(candidate),
            "candidate_sha256": sha256_file(candidate),
            "best_path": str(tmp_path / "best.ckpt"),
            "evaluation_kind": "reschedule",
            "instance_id": "validation_0001",
            "scenario_id": scenario_id,
        }
        result = {
            "episode": 2,
            "job_id": job["job_id"],
            "group_id": job["group_id"],
            "scenario_id": scenario_id,
            "eligible": 1.0,
            "selection_score": float(index + 1),
            "composite_score": float(index + 1),
            "makespan": float(100 + index),
        }
        _record_result(paths, job, result, [], writer)
        if index < 2:
            assert not Path(job["best_path"]).exists()

    assert (tmp_path / "best.ckpt").read_bytes() == b"candidate"
    aggregate = json.loads(
        (paths.results / "episode_000002.json").read_text(encoding="utf-8-sig")
    )
    assert aggregate["scenario_count"] == 3
    assert aggregate["eligible"] == 1.0
    assert aggregate["selection_score"] == pytest.approx(2.0)


def test_r5_cuda_oom_never_falls_back_to_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from training.async_eval_worker import _evaluate_job_with_cuda_oom_fallback

    devices: list[str] = []

    def _evaluate(job, project_root, *, device):
        devices.append(device.type)
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr("training.async_eval_worker._evaluate_job", _evaluate)

    with pytest.raises(torch.cuda.OutOfMemoryError):
        _evaluate_job_with_cuda_oom_fallback(
            {"episode": 2, "reschedule_async_protocol": "r5_task_delay_v1"},
            tmp_path,
            device=torch.device("cuda"),
        )

    assert devices == ["cuda"]


def test_async_manager_records_initial_evaluation_kind(tmp_path: Path) -> None:
    config = Config(async_eval_enabled=True)
    config.data_file_path = "data/283.csv"
    config.async_eval_initial_data_path = "data/680.csv"
    config.enable_multi_benchmark_eval = True
    manager = AsyncEvaluationManager(
        config=config,
        latest_path=tmp_path / "checkpoints" / "last.ckpt",
        best_path=tmp_path / "checkpoints" / "best.ckpt",
        project_root=PROJECT_ROOT,
    )
    manager._wait_for_slot = lambda: None  # type: ignore[method-assign]
    trainer = SimpleNamespace(
        save_checkpoint=lambda path: Path(path).write_bytes(b"checkpoint")
    )

    manager.submit(trainer, episode=3)

    job = json.loads(
        (manager.paths.pending / "episode_000003.json").read_text(encoding="utf-8-sig")
    )
    assert job["evaluation_kind"] == "initial_standard"
    assert job["instance_id"] == "680"
    assert job["scenario_id"] == "standard"


def test_async_candidate_remains_immutable_after_latest_checkpoint_is_overwritten(tmp_path: Path) -> None:
    config = Config(async_eval_enabled=True)
    manager = AsyncEvaluationManager(
        config=config,
        latest_path=tmp_path / "checkpoints" / "last.ckpt",
        best_path=tmp_path / "checkpoints" / "best.ckpt",
        project_root=PROJECT_ROOT,
    )
    manager._wait_for_slot = lambda: None  # type: ignore[method-assign]
    trainer = SimpleNamespace(save_checkpoint=lambda path: Path(path).write_bytes(b"episode-5"))

    candidate = manager.submit(trainer, episode=5)
    manager.latest_path.write_bytes(b"episode-6")

    assert candidate.read_bytes() == b"episode-5"
    assert manager.latest_path.read_bytes() == b"episode-6"
    job = json.loads((manager.paths.pending / "episode_000005.json").read_text(encoding="utf-8-sig"))
    assert len(job["candidate_sha256"]) == 64


def test_async_manager_records_main_only_multiscale_manifest_snapshot(tmp_path: Path) -> None:
    config = Config(async_eval_enabled=True)
    config.checkpoint_selection_protocol = "multiscale_manifest"
    config.checkpoint_selection_manifest_path = (
        "data/initial_selection_manifests/real_four_instances_temperature0_v1.json"
    )
    config.async_eval_worker_count = 2
    config.async_eval_submit_every_episodes = 5
    validate_runtime_config(config)
    manager = AsyncEvaluationManager(
        config=config,
        latest_path=tmp_path / "checkpoints" / "last.ckpt",
        best_path=tmp_path / "checkpoints" / "best.ckpt",
        project_root=PROJECT_ROOT,
    )
    manager._wait_for_slot = lambda: None  # type: ignore[method-assign]
    trainer = SimpleNamespace(save_checkpoint=lambda path: Path(path).write_bytes(b"checkpoint"))

    manager.submit(trainer, episode=5)

    job = json.loads(
        (manager.paths.pending / "episode_000005.json").read_text(encoding="utf-8-sig")
    )
    assert job["evaluation_kind"] == "initial_multi_benchmark"
    assert job["selection_manifest"]["protocol_id"].endswith("_v1")
    assert [item["instance_id"] for item in job["selection_manifest"]["instances"]] == [
        "real_283",
        "real_680",
        "real_2338",
        "real_3182",
    ]


def test_multiscale_best_publication_requires_all_instances_eligible(tmp_path: Path) -> None:
    paths = AsyncEvalPaths.create(tmp_path / "async_eval")
    candidate = paths.candidates / "episode_000005.ckpt"
    candidate.write_bytes(b"candidate")
    writer = SimpleNamespace(add_scalar=lambda *args, **kwargs: None, flush=lambda: None)
    instance_rows = []
    for instance_id, eligible in (("real_283", True), ("real_680", False), ("real_2338", True), ("real_3182", True)):
        schedule = paths.results / "episode_000005" / f"{instance_id}_schedule.csv"
        audit = paths.results / "episode_000005" / f"{instance_id}_legality_audit.json"
        schedule.parent.mkdir(parents=True, exist_ok=True)
        schedule.write_text("TaskID,StationID,Team,Start,End,Duration\n", encoding="utf-8")
        audit.write_text("{}", encoding="utf-8")
        instance_rows.append({"instance_id": instance_id, "schedule_path": str(schedule), "audit_path": str(audit)})
    result = {
        "episode": 5,
        "evaluation_kind": "initial_multi_benchmark",
        "instance_id": "real_283_680_2338_3182",
        "scenario_id": "standard",
        "candidate_path": str(candidate),
        "eligible": 0.0,
        "selection_score": float("inf"),
        "composite_score": 1.0,
        "makespan": 100.0,
        "instances": instance_rows,
    }
    job = {"episode": 5, "candidate_path": str(candidate), "best_path": str(tmp_path / "best.ckpt"), "evaluation_kind": "initial_multi_benchmark", "instance_id": result["instance_id"], "scenario_id": "standard"}
    _record_result(paths, job, result, [], writer)
    assert not Path(job["best_path"]).exists()
    assert not (paths.state / "best.json").exists()


def test_equal_multiscale_scores_keep_earlier_episode(tmp_path: Path) -> None:
    paths = AsyncEvalPaths.create(tmp_path / "async_eval")
    writer = SimpleNamespace(add_scalar=lambda *args, **kwargs: None, flush=lambda: None)
    for episode in (10, 5):
        candidate = paths.candidates / f"episode_{episode:06d}.ckpt"
        candidate.write_bytes(str(episode).encode("ascii"))
        rows = []
        for instance_id in ("real_283", "real_680", "real_2338", "real_3182"):
            schedule = paths.results / f"episode_{episode:06d}" / f"{instance_id}_schedule.csv"
            audit = paths.results / f"episode_{episode:06d}" / f"{instance_id}_legality_audit.json"
            schedule.parent.mkdir(parents=True, exist_ok=True)
            schedule.write_text("TaskID,StationID,Team,Start,End,Duration\n", encoding="utf-8")
            audit.write_text("{}", encoding="utf-8")
            rows.append({"instance_id": instance_id, "schedule_path": str(schedule), "audit_path": str(audit)})
        result = {
            "episode": episode, "evaluation_kind": "initial_multi_benchmark",
            "instance_id": "real_283_680_2338_3182", "scenario_id": "standard",
            "candidate_path": str(candidate), "eligible": 1.0, "selection_score": 1.0,
            "composite_score": 1.0, "makespan": 100.0,
            "selection_manifest_sha256": "manifest", "selection_protocol_id": "protocol",
            "instances": rows,
        }
        job = {"episode": episode, "candidate_path": str(candidate), "best_path": str(tmp_path / "best.ckpt"), "evaluation_kind": "initial_multi_benchmark", "instance_id": result["instance_id"], "scenario_id": "standard"}
        _record_result(paths, job, result, [], writer)
    state = json.loads((paths.state / "best.json").read_text(encoding="utf-8"))
    assert state["episode"] == 5
    assert (tmp_path / "best.ckpt").read_bytes() == b"5"


def test_best_checkpoint_is_immutable_after_candidate_file_changes(tmp_path: Path) -> None:
    paths = AsyncEvalPaths.create(tmp_path / "async_eval")
    candidate = paths.candidates / "episode_000005.ckpt"
    candidate.write_bytes(b"episode-5")
    writer = SimpleNamespace(add_scalar=lambda *args, **kwargs: None, flush=lambda: None)
    result = {
        "episode": 5,
        "evaluation_kind": "initial_standard",
        "instance_id": "real_680",
        "scenario_id": "standard",
        "candidate_path": str(candidate),
        "eligible": 1.0,
        "selection_score": 100.0,
        "composite_score": 100.0,
        "makespan": 100.0,
    }
    job = {
        "episode": 5,
        "candidate_path": str(candidate),
        "best_path": str(tmp_path / "best.ckpt"),
        "evaluation_kind": "initial_standard",
        "instance_id": "real_680",
        "scenario_id": "standard",
    }
    _record_result(paths, job, result, [], writer)
    candidate.write_bytes(b"later-episode")
    assert (tmp_path / "best.ckpt").read_bytes() == b"episode-5"


def test_result_publication_rejects_candidate_hash_mismatch(tmp_path: Path) -> None:
    paths = AsyncEvalPaths.create(tmp_path / "async_eval")
    candidate = paths.candidates / "episode_000005.ckpt"
    candidate.write_bytes(b"committed-candidate")
    expected_sha256 = sha256_file(candidate)
    candidate.write_bytes(b"overwritten-candidate")
    writer = SimpleNamespace(add_scalar=lambda *args, **kwargs: None, flush=lambda: None)
    result = {
        "episode": 5,
        "evaluation_kind": "initial_standard",
        "instance_id": "real_680",
        "scenario_id": "standard",
        "candidate_path": str(candidate),
        "eligible": 1.0,
        "selection_score": 100.0,
        "composite_score": 100.0,
        "makespan": 100.0,
    }
    job = {
        "episode": 5,
        "candidate_path": str(candidate),
        "candidate_sha256": expected_sha256,
        "best_path": str(tmp_path / "best.ckpt"),
        "evaluation_kind": "initial_standard",
        "instance_id": "real_680",
        "scenario_id": "standard",
    }
    with pytest.raises(RuntimeError, match="哈希不一致"):
        _record_result(paths, job, result, [], writer)
    assert not Path(job["best_path"]).exists()


def test_worker_rejects_hash_mismatch_before_loading_candidate(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.ckpt"
    candidate.write_bytes(b"committed-candidate")
    expected_sha256 = sha256_file(candidate)
    candidate.write_bytes(b"overwritten-candidate")
    with pytest.raises(RuntimeError, match="哈希不一致"):
        _verified_candidate_path({"candidate_path": str(candidate), "candidate_sha256": expected_sha256})


def test_process_liveness_probe_is_side_effect_free() -> None:
    assert process_is_alive(os.getpid())
    assert not process_is_alive(2_000_000_000)


def test_cuda_two_workers_reuse_legacy_attached_process_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """双 CUDA worker 仅替换设备，进程启动方式保持旧版实现。"""
    config = Config(async_eval_enabled=True)
    config.async_eval_device = "cuda"
    config.async_eval_worker_count = 2
    manager = AsyncEvaluationManager(
        config=config,
        latest_path=tmp_path / "checkpoints" / "last.ckpt",
        best_path=tmp_path / "checkpoints" / "best.ckpt",
        project_root=PROJECT_ROOT,
    )
    captured: list[tuple[list[str], dict[str, object]]] = []

    class _Process:
        pid = 12345

        @staticmethod
        def poll() -> None:
            return None

    def _popen(command, **kwargs):
        captured.append((command, kwargs))
        return _Process()

    monkeypatch.setattr("training.async_evaluation.subprocess.Popen", _popen)
    manager._start_workers()

    assert len(captured) == 2
    for command, kwargs in captured:
        device_index = command.index("--device")
        assert command[device_index + 1] == "cuda"
        assert kwargs["cwd"] == str(PROJECT_ROOT)
        assert kwargs["text"] is True
        assert "stdin" not in kwargs
        assert "stdout" not in kwargs
        assert "stderr" not in kwargs


def test_cuda_oom_falls_back_only_for_current_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from training.async_eval_worker import _evaluate_job_with_cuda_oom_fallback

    devices: list[str] = []

    def _evaluate(job, project_root, *, device):
        devices.append(device.type)
        if len(devices) == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return {"episode": int(job["episode"]), "eligible": 1.0}, []

    cache_clears: list[bool] = []
    monkeypatch.setattr("training.async_eval_worker._evaluate_job", _evaluate)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cache_clears.append(True))

    first_result, _ = _evaluate_job_with_cuda_oom_fallback(
        {"episode": 2}, tmp_path, device=torch.device("cuda")
    )
    second_result, _ = _evaluate_job_with_cuda_oom_fallback(
        {"episode": 4}, tmp_path, device=torch.device("cuda")
    )

    assert devices == ["cuda", "cpu", "cuda"]
    assert cache_clears == [True]
    assert first_result["evaluation_device"] == "cpu"
    assert first_result["cuda_oom_cpu_fallback"] == 1.0
    assert second_result["evaluation_device"] == "cuda"
    assert second_result["cuda_oom_cpu_fallback"] == 0.0


def test_atomic_link_or_copy_publishes_complete_file(tmp_path: Path) -> None:
    source = tmp_path / "candidate.ckpt"
    destination = tmp_path / "best" / "best.ckpt"
    source.write_bytes(b"complete-checkpoint")
    atomic_link_or_copy(source, destination)
    source.unlink()
    assert destination.read_bytes() == b"complete-checkpoint"

    copy_source = tmp_path / "copy_source.ckpt"
    copy_destination = tmp_path / "copy" / "latest.ckpt"
    copy_source.write_bytes(b"candidate")
    atomic_copy_file(copy_source, copy_destination)
    copy_destination.write_bytes(b"latest")
    assert copy_source.read_bytes() == b"candidate"

    payload_path = tmp_path / "queue" / "episode.json"
    atomic_write_json(payload_path, {"episode": 3, "state": "pending"})
    assert '"episode": 3' in payload_path.read_text(encoding="utf-8")


def test_interrupted_running_job_is_restored_to_pending(tmp_path: Path) -> None:
    paths = AsyncEvalPaths.create(tmp_path / "async_eval")
    running = paths.running / "episode_000003.json"
    atomic_write_json(running, {"episode": 3, "attempt": 0})
    _restore_interrupted_jobs(paths)
    assert not running.exists()
    assert (paths.pending / running.name).is_file()


def test_queue_reader_accepts_utf8_bom(tmp_path: Path) -> None:
    from training.async_eval_worker import _claim_next_job

    paths = AsyncEvalPaths.create(tmp_path / "async_eval")
    pending = paths.pending / "episode_000004.json"
    pending.write_text(
        "\ufeff" + json.dumps({"episode": 4, "attempt": 0}),
        encoding="utf-8",
    )
    claimed = _claim_next_job(paths)
    assert claimed is not None
    running_path, payload = claimed
    assert running_path.parent == paths.running
    assert payload["episode"] == 4


def test_two_workers_claim_different_pending_jobs_atomically(tmp_path: Path) -> None:
    from training.async_eval_worker import _claim_next_job

    paths = AsyncEvalPaths.create(tmp_path / "async_eval")
    for episode in (5, 10):
        atomic_write_json(paths.pending / f"episode_{episode:06d}.json", {"episode": episode})
    with ThreadPoolExecutor(max_workers=2) as executor:
        claimed = list(executor.map(lambda _unused: _claim_next_job(paths), range(2)))
    assert {item[1]["episode"] for item in claimed if item is not None} == {5, 10}
    assert len(list(paths.running.glob("episode_*.json"))) == 2


def test_claim_next_job_survives_permission_error_on_running_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows 文件锁竞态：任务已移入 running 但读取被拒时，任务必须移回
    pending 且不崩溃（否则 worker 异常退出导致 AsyncEvaluationError）。"""
    from training.async_eval_worker import _claim_next_job

    paths = AsyncEvalPaths.create(tmp_path / "async_eval")
    pending = paths.pending / "episode_000009.json"
    atomic_write_json(pending, {"episode": 9, "attempt": 0})

    def flaky_loads(*_args, **_kwargs) -> object:
        raise PermissionError("simulated Windows file lock on running job")

    monkeypatch.setattr("training.async_eval_worker.json.loads", flaky_loads)
    claimed = _claim_next_job(paths)

    assert claimed is None
    # 任务未丢失：被原子移回 pending，等待后续轮次重试。
    assert (paths.pending / "episode_000009.json").exists()
    assert not (paths.running / "episode_000009.json").exists()


def test_cached_observation_matches_canonical_rebuild(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(configs, "enable_reschedule_mode", False)
    monkeypatch.setattr(configs, "use_skill_hub", True)
    monkeypatch.setattr(configs, "skill_hub_bidirectional", True)
    env = AirLineEnv_Graph(PROJECT_ROOT / "data" / "283.csv", seed=42)
    env.reset(randomize_duration=False, randomize_workers=False, seed=42)
    cache = CachedEnvironmentObservation(env)
    first = cache.refresh()
    first.to(torch.device("cpu"))

    env.current_time = float(env.current_time) + 0.125
    env.station_loads[0] = float(env.station_loads[0]) + 0.25
    env.worker_free_time[0] = float(env.current_time) + 0.5
    canonical = env.rebuild_state_from_snapshot(env.get_state_snapshot())
    cached = cache.refresh()

    assert cached is first
    _assert_graph_equal(canonical, cached)


def test_pure_actor_evaluation_does_not_call_critic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(configs, "enable_reschedule_mode", False)
    monkeypatch.setattr(configs, "use_schedule_free", False)
    env = AirLineEnv_Graph(PROJECT_ROOT / "data" / "283.csv", seed=42)
    state = env.reset(randomize_duration=False, randomize_workers=False, seed=42)
    task_mask, station_mask, worker_mask = env.get_masks()
    model = HBGATPN(configs)

    def _forbidden_value(*args, **kwargs):
        raise AssertionError("纯 Actor 评估不应执行 Critic")

    monkeypatch.setattr(model, "get_value", _forbidden_value)
    agent = PPOAgent(
        model=model,
        lr=float(configs.lr),
        gamma=float(configs.gamma),
        k_epochs=1,
        eps_clip=float(configs.eps_clip),
        device=torch.device("cpu"),
        batch_size=2,
        total_timesteps=1,
        config=configs,
    )
    action, _logprob, value, _station_mask, _invalid = agent.select_action(
        state,
        mask_task=task_mask,
        mask_station_matrix=station_mask,
        mask_worker=worker_mask,
        deterministic=True,
        temperature=0.0,
        is_eval=True,
        compute_value=False,
    )
    assert action is not None
    assert value == 0.0


def test_exact_scenario_keeps_original_reset_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Policy:
        training = True

        def eval(self):
            self.training = False

        def train(self, mode=True):
            self.training = bool(mode)

    class _Agent:
        policy = _Policy()

    class _ResetProbe:
        skip_obs_building = False

        def reset(self, **kwargs):
            raise RuntimeError(f"reset_seed={kwargs['seed']}")

    scenarios = [(f"low_{idx:03d}", object()) for idx in range(20)]
    scenarios.append(("medium_000", object()))
    monkeypatch.setattr("runtime.reschedule_eval.ensure_reschedule_eval_scenarios_available", lambda _cfg: Path("fixed.csv"))
    monkeypatch.setattr("runtime.reschedule_eval.load_reschedule_scenarios", lambda _path: scenarios)
    monkeypatch.setattr(configs, "reschedule_eval_scenario_seed", 42)

    with pytest.raises(RuntimeError, match="reset_seed=62"):
        evaluate_reschedule_model(
            _ResetProbe(),
            _Agent(),
            num_runs=None,
            temperature=0.0,
            scenario_ids=["medium_000"],
        )


def test_async_checkpoint_only_enqueues_committed_update(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import train_lightning

    calls: list[tuple[str, int | bool]] = []

    class _Manager:
        def __init__(self, **kwargs):
            calls.append(("init", True))

        def submit(self, trainer, *, episode: int):
            calls.append(("submit", episode))

        def finalize(self, *, wait: bool):
            calls.append(("finalize", wait))

        def terminate_for_exception(self):
            calls.append(("terminate", True))

    monkeypatch.setattr(configs, "async_eval_enabled", True)
    monkeypatch.setattr(configs, "async_eval_wait_on_finish", True)
    monkeypatch.setattr(train_lightning, "AsyncEvaluationManager", _Manager)
    callback = train_lightning.RolloutCheckpoint(tmp_path)
    trainer = SimpleNamespace(save_checkpoint=lambda _path: None)
    module = SimpleNamespace(
        last_completed_episode=7,
        last_eval_metrics=None,
        last_update_committed=False,
        eval_freq=1,
    )

    callback.on_train_batch_end(trainer, module, None, None, 0)
    assert ("submit", 7) not in calls
    module.last_update_committed = True
    callback.on_train_batch_end(trainer, module, None, None, 0)
    callback.on_fit_end(trainer, module)
    assert ("submit", 7) in calls
    assert ("finalize", True) in calls


def test_async_checkpoint_uses_explicit_submission_cadence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import train_lightning

    submitted: list[int] = []

    class _Manager:
        def __init__(self, **kwargs):
            pass

        def submit(self, trainer, *, episode: int):
            submitted.append(episode)

        def finalize(self, *, wait: bool):
            pass

        def terminate_for_exception(self):
            pass

    monkeypatch.setattr(configs, "async_eval_enabled", True)
    monkeypatch.setattr(configs, "async_eval_submit_every_episodes", 5)
    monkeypatch.setattr(train_lightning, "AsyncEvaluationManager", _Manager)
    callback = train_lightning.RolloutCheckpoint(tmp_path)
    saved: list[str] = []
    def _save_checkpoint(path: str) -> None:
        saved.append(str(path))
        torch.save({"apal_metadata": {"model_spec": {}}}, path)

    trainer = SimpleNamespace(save_checkpoint=_save_checkpoint)
    module = SimpleNamespace(
        agent=SimpleNamespace(optimizer=None, use_schedule_free=False),
        last_completed_episode=4,
        last_eval_metrics=None,
        last_update_committed=True,
        eval_freq=1,
    )
    callback.on_train_batch_end(trainer, module, None, None, 0)
    module.last_completed_episode = 5
    callback.on_train_batch_end(trainer, module, None, None, 0)
    assert submitted == [5]
    assert saved
