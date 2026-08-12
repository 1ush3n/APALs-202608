from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest


pytest.importorskip("lightning")

from training.lightning_module import (
    APALDataModule,
    APALLightningModule,
    RolloutMetrics,
)


class _Agent:
    def __init__(self):
        import torch

        self.policy = torch.nn.Linear(1, 1)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        self.updated = False

    def validate_snapshot_homogeneity(self, states):
        assert states[0]["dataset_idx"] == 0

    def update(self, memory, env, current_ep):
        self.updated = True
        return {"Loss/Total": 0.0}


class _OOMAgent(_Agent):
    def update(self, memory, env, current_ep):
        self.updated = True
        return {"OOM/SkippedUpdate": 1.0, "_skip_training_log": 1.0}


class _RolloutService:
    def __init__(self):
        self.closed = False
        self.eval_calls = 0

    def collect(self, episode):
        from training.lightning_module import RolloutUpdate

        memory = SimpleNamespace(
            states=[{"dataset_idx": 0, "worker_free_time": [0.0]}]
        )
        metrics = RolloutMetrics(
            episode=episode,
            average_reward=1.0,
            average_makespan=2.0,
            completion_rate=1.0,
            environment_steps=3,
            steps_per_second=4.0,
            total_seconds=0.75,
            ipc_mask_ms=1.0,
            forward_ms=2.0,
            rebuild_ms=3.0,
            environment_step_ms=4.0,
        )
        return RolloutUpdate(
            memory=memory,
            env=object(),
            episode=episode,
            rollout_metrics=(metrics,),
        )

    def evaluate(self, episode):
        self.eval_calls += 1
        return {"makespan": 1.0}

    def close(self):
        self.closed = True


def _checkpointing_trainer(saved_paths: list[str]) -> SimpleNamespace:
    """返回会写出可加载 checkpoint 的 trainer 替身。"""
    import torch

    def save_checkpoint(path: str) -> None:
        saved_paths.append(path)
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"apal_metadata": {"model_spec": {}}}, checkpoint_path)

    return SimpleNamespace(save_checkpoint=save_checkpoint)


def test_lightning_module_uses_manual_optimization_contract() -> None:
    agent = _Agent()
    service = _RolloutService()
    module = APALLightningModule(agent, service, eval_freq=1)
    data = APALDataModule(service, max_episodes=1)

    assert module.automatic_optimization is False
    batch = next(iter(data.train_dataloader()))
    module.training_step(batch, 0)
    assert agent.updated is True
    assert module.last_completed_episode == 1
    assert module.last_eval_metrics == {"makespan": 1.0}


def test_rollout_data_module_uses_absolute_episode_range_after_resume() -> None:
    service = _RolloutService()
    data = APALDataModule(service, max_episodes=10, start_episode=7)

    updates = list(data.train_dataloader())

    assert [update.episode for update in updates] == [7, 8, 9, 10]
    assert len(data) == 4


def test_resume_start_episode_rejects_misaligned_checkpoint() -> None:
    from train_lightning import _resume_start_episode

    valid = {
        "apal_metadata": {"episode": 33},
        "loops": {"fit_loop": {"epoch_loop.batch_progress": {"total": {"completed": 32}}}},
    }
    assert _resume_start_episode(valid) == 34

    corrupted = {
        "apal_metadata": {"episode": 4},
        "loops": {"fit_loop": {"epoch_loop.batch_progress": {"total": {"completed": 36}}}},
    }
    with pytest.raises(ValueError, match="不一致"):
        _resume_start_episode(corrupted)


def test_lightning_module_skips_logging_and_eval_on_oom_update() -> None:
    agent = _OOMAgent()
    service = _RolloutService()
    module = APALLightningModule(agent, service, eval_freq=1)
    data = APALDataModule(service, max_episodes=1)

    batch = next(iter(data.train_dataloader()))
    module.training_step(batch, 0)

    assert agent.updated is True
    assert module.last_completed_episode == 0
    assert module.last_eval_metrics is None
    assert service.eval_calls == 0


def test_lightning_checkpoint_contains_apal_metadata() -> None:
    from configs import Config

    agent = _Agent()
    service = _RolloutService()
    service.config = Config()
    module = APALLightningModule(agent, service, eval_freq=1)
    checkpoint = {}

    module.on_save_checkpoint(checkpoint)

    assert checkpoint["apal_metadata"]["model_type"] == "HB-GAT-PN"
    assert (
        checkpoint["apal_metadata"]["model_spec"]["resource_graph_mode"]
        == "skill_hub_bidirectional"
    )


def test_lightning_v2_resume_rejects_effective_batch_override() -> None:
    from configs import Config
    from runtime.checkpoints import build_checkpoint_metadata

    cfg = Config()
    cfg.team_selection_mode = "autoregressive_pressure_v2"
    cfg.policy_action_scope = "operation_station_worker"
    cfg.actor_context_mode = "attention"
    cfg.worker_pointer_v2_behavior_replay = True
    cfg.worker_pointer_v2_logical_batch_cap = 64
    cfg.worker_pointer_v2_rollout_group_upper_bound = 4
    cfg.accumulation_steps = 16
    agent = _Agent()
    agent.current_step = 0
    agent.batch_size = 64
    service = _RolloutService()
    service.config = cfg
    module = APALLightningModule(agent, service, eval_freq=1)
    checkpoint = {
        "apal_metadata": build_checkpoint_metadata(cfg),
        "apal_agent_state": {"current_step": 3, "batch_size": 256},
    }

    with pytest.raises(ValueError, match="有效逻辑 batch"):
        module.on_load_checkpoint(checkpoint)

    assert agent.current_step == 0
    assert agent.batch_size == 64


def test_rollout_checkpoint_saves_latest_and_best(tmp_path) -> None:
    from train_lightning import RolloutCheckpoint

    saved_paths = []
    trainer = _checkpointing_trainer(saved_paths)
    module = SimpleNamespace(
        last_completed_episode=2,
        last_eval_metrics={"makespan": 10.0, "completion_rate": 1.0},
    )
    callback = RolloutCheckpoint(tmp_path)

    callback.on_train_batch_end(trainer, module, None, None, 0)

    assert saved_paths == [
        str(tmp_path / "best" / "best.ckpt"),
        str(tmp_path / "last.ckpt"),
    ]
    assert callback.best_score == 10.0

    saved_paths.clear()
    module.last_completed_episode = 3
    module.last_eval_metrics = None
    callback.on_train_batch_end(trainer, module, None, None, 1)
    assert saved_paths == [str(tmp_path / "last.ckpt")]

    saved_paths.clear()
    module.last_completed_episode = 4
    module.last_eval_metrics = {"makespan": 12.0, "completion_rate": 1.0}
    callback.on_train_batch_end(trainer, module, None, None, 2)
    assert saved_paths == [str(tmp_path / "last.ckpt")]

    saved_paths.clear()
    module.last_completed_episode = 5
    module.last_eval_metrics = {
        "makespan": 8.0,
        "multi_benchmark_selection_score": float("inf"),
        "multi_benchmark_eligible": 0.0,
    }
    callback.on_train_batch_end(trainer, module, None, None, 3)
    assert saved_paths == [str(tmp_path / "last.ckpt")]

    saved_paths.clear()
    module.last_completed_episode = 6
    module.last_eval_metrics = {
        "makespan": 8.0,
        "multi_benchmark_selection_score": 0.9,
        "multi_benchmark_eligible": 1.0,
    }
    callback.on_train_batch_end(trainer, module, None, None, 4)
    assert saved_paths == [
        str(tmp_path / "best" / "best.ckpt"),
        str(tmp_path / "last.ckpt"),
    ]
    assert callback.best_score == 0.9


def test_rollout_checkpoint_uses_reschedule_selection_score(tmp_path) -> None:
    from train_lightning import RolloutCheckpoint

    saved_paths = []
    trainer = _checkpointing_trainer(saved_paths)
    module = SimpleNamespace(
        last_completed_episode=1,
        last_eval_metrics={
            "makespan": 100.0,
            "reschedule_composite_score": 0.8,
            "reschedule_selection_score": 0.8,
            "reschedule_eligible_rate": 0.0,
        },
    )
    callback = RolloutCheckpoint(tmp_path)

    callback.on_train_batch_end(trainer, module, None, None, 0)
    assert saved_paths == [str(tmp_path / "last.ckpt")]
    assert callback.best_score == float("inf")

    saved_paths.clear()
    module.last_completed_episode = 2
    module.last_eval_metrics = {
        "makespan": 120.0,
        "reschedule_composite_score": 0.7,
        "reschedule_selection_score": 0.7,
        "reschedule_eligible_rate": 1.0,
    }
    callback.on_train_batch_end(trainer, module, None, None, 1)

    assert saved_paths == [
        str(tmp_path / "best" / "best.ckpt"),
        str(tmp_path / "last.ckpt"),
    ]
    assert callback.best_score == 0.7


def test_rollout_checkpoint_rejects_incomplete_initial_schedule(tmp_path) -> None:
    from train_lightning import RolloutCheckpoint

    saved_paths = []
    trainer = _checkpointing_trainer(saved_paths)
    module = SimpleNamespace(
        last_completed_episode=1,
        last_eval_metrics={"makespan": 1.0, "completion_rate": 0.0},
    )
    callback = RolloutCheckpoint(tmp_path)

    callback.on_train_batch_end(trainer, module, None, None, 0)

    assert saved_paths == [str(tmp_path / "last.ckpt")]
    assert callback.best_score == float("inf")


def test_rollout_checkpoint_writes_model_spec_metadata_to_final_file(tmp_path) -> None:
    import torch

    from train_lightning import RolloutCheckpoint

    saved_paths: list[str] = []

    def save_checkpoint(path: str) -> None:
        saved_paths.append(path)
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({}, checkpoint_path)

    class _MetadataModule:
        last_completed_episode = 1
        last_eval_metrics = {"makespan": 1.0, "completion_rate": 1.0}

        @staticmethod
        def on_save_checkpoint(checkpoint: dict) -> None:
            checkpoint["apal_metadata"] = {
                "model_spec": {"team_selection_mode": "autoregressive_pressure_v2"}
            }

    callback = RolloutCheckpoint(tmp_path)
    callback.on_train_batch_end(
        SimpleNamespace(save_checkpoint=save_checkpoint),
        _MetadataModule(),
        None,
        None,
        0,
    )

    assert saved_paths == [
        str(tmp_path / "best" / "best.ckpt"),
        str(tmp_path / "last.ckpt"),
    ]
    final_checkpoint = torch.load(tmp_path / "last.ckpt", map_location="cpu")
    assert final_checkpoint["apal_metadata"]["model_spec"] == {
        "team_selection_mode": "autoregressive_pressure_v2"
    }


def test_reschedule_warm_start_rejects_legacy_raw_checkpoint(tmp_path) -> None:
    import torch
    from configs import configs
    from tests.runtime_safety import temporary_config
    from train_lightning import _maybe_load_reschedule_warm_start

    source = torch.nn.Linear(2, 2)
    target = torch.nn.Linear(2, 2)
    with torch.no_grad():
        source.weight.fill_(3.0)
        source.bias.fill_(1.5)
        target.weight.zero_()
        target.bias.zero_()
    checkpoint_path = tmp_path / "initial_model.pth"
    torch.save(source.state_dict(), checkpoint_path)

    overrides = {
        "enable_reschedule_mode": True,
        "reschedule_warm_start": True,
        "reschedule_baseline_model_path": str(checkpoint_path),
    }
    with temporary_config(configs, overrides):
        with pytest.raises(ValueError, match="checkpoint 格式不兼容"):
            _maybe_load_reschedule_warm_start(target, torch.device("cpu"), resume=False)


def test_rollout_metrics_expose_expected_log_keys() -> None:
    metrics = _RolloutService().collect(1).rollout_metrics[0]
    logged = metrics.as_log_dict()

    assert logged["Rollout/StepsPerSecond"] == 4.0
    assert logged["Rollout/ForwardMs"] == 2.0
    assert logged["Rollout/CompletionRate"] == 1.0


def test_standard_evaluation_restores_policy_and_config(monkeypatch, capsys) -> None:
    import torch

    from configs import Config
    from training.rollout_service import APALRolloutService

    config = Config()
    config.enable_dynamic_events = True
    config.enable_station_breakdown = True
    config.enable_material_delay = True
    config.eval_scenarios = ["standard"]
    policy = torch.nn.Linear(1, 1)
    policy.train()
    agent = SimpleNamespace(policy=policy)
    vector_env = SimpleNamespace(num_envs=1)
    seen = {}

    def fake_evaluate(*args, **kwargs):
        seen["scenario_names"] = kwargs["scenario_names"]
        config.enable_dynamic_events = False
        config.enable_station_breakdown = False
        config.enable_material_delay = False
        return 10.0, 2.0, 3.0, [(0, 0, [0], 0.0, 1.0)], 0.5, 0.6, 0.7

    monkeypatch.setattr("training.rollout_service.evaluate_model", fake_evaluate)
    service = APALRolloutService(
        agent=agent,
        vector_env=vector_env,
        eval_env=SimpleNamespace(num_tasks=1),
        config=config,
        device=torch.device("cpu"),
    )
    metrics = service.evaluate(2)

    assert seen["scenario_names"] == ("standard",)
    assert policy.training is True
    assert config.enable_dynamic_events is True
    assert config.enable_station_breakdown is True
    assert config.enable_material_delay is True
    assert metrics["makespan"] == 10.0
    assert metrics["completion_rate"] == 1.0
    output = capsys.readouterr().out
    assert "[Eval] ep=2 start" in output
    assert "Mk=10.00" in output


def test_reschedule_evaluation_returns_selection_metrics(monkeypatch) -> None:
    import torch

    from configs import Config
    from training.rollout_service import APALRolloutService

    config = Config()
    config.enable_reschedule_mode = True
    config.reschedule_eval_num_scenarios = 4
    policy = torch.nn.Linear(1, 1)
    policy.train()
    agent = SimpleNamespace(policy=policy)
    vector_env = SimpleNamespace(num_envs=1)

    def fake_reschedule_eval(*args, **kwargs):
        fake_reschedule_eval.last_metrics = {
            "eligible_rate": 1.0,
            "composite_score": 0.75,
            "selection_score": 0.75,
            "demand_violation_count": 0.0,
        }
        return 10.0, 2.0, 3.0, [], 0.5, 0.6, 0.7

    monkeypatch.setattr("training.rollout_service.evaluate_reschedule_model", fake_reschedule_eval)
    service = APALRolloutService(
        agent=agent,
        vector_env=vector_env,
        eval_env=object(),
        config=config,
        device=torch.device("cpu"),
    )

    metrics = service.evaluate(3)

    assert policy.training is True
    assert metrics["makespan"] == 10.0
    assert metrics["reschedule_composite_score"] == 0.75
    assert metrics["reschedule_selection_score"] == 0.75
    assert metrics["reschedule_eligible_rate"] == 1.0
    assert metrics["demand_violation_count"] == 0.0
