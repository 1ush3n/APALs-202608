from __future__ import annotations

import os
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from configs import Config, configs
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from runtime.configuration import validate_runtime_config
from runtime.reschedule_eval import evaluate_reschedule_model
from training.async_eval_worker import _restore_interrupted_jobs
from training.async_evaluation import (
    AsyncEvalPaths,
    atomic_link_or_copy,
    atomic_write_json,
    process_is_alive,
)
from training.observation import CachedEnvironmentObservation


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _assert_graph_equal(left, right) -> None:
    assert left.node_types == right.node_types
    assert left.edge_types == right.edge_types
    for node_type in left.node_types:
        assert torch.equal(left[node_type].x, right[node_type].x), node_type
    for edge_type in left.edge_types:
        assert torch.equal(left[edge_type].edge_index, right[edge_type].edge_index), edge_type


def test_async_eval_config_rejects_non_reschedule_and_gpu() -> None:
    config = Config(async_eval_enabled=True)
    with pytest.raises(ValueError, match="仅支持重调度"):
        validate_runtime_config(config)

    config.enable_reschedule_mode = True
    config.async_eval_device = "cuda"
    with pytest.raises(ValueError, match="必须为 cpu"):
        validate_runtime_config(config)

    config.async_eval_device = "cpu"
    config.eval_freq = 2
    with pytest.raises(ValueError, match="eval_freq=1"):
        validate_runtime_config(config)


def test_process_liveness_probe_is_side_effect_free() -> None:
    assert process_is_alive(os.getpid())
    assert not process_is_alive(2_000_000_000)


def test_atomic_link_or_copy_publishes_complete_file(tmp_path: Path) -> None:
    source = tmp_path / "candidate.ckpt"
    destination = tmp_path / "best" / "best.ckpt"
    source.write_bytes(b"complete-checkpoint")
    atomic_link_or_copy(source, destination)
    source.unlink()
    assert destination.read_bytes() == b"complete-checkpoint"

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
