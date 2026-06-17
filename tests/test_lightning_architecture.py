from __future__ import annotations

from types import SimpleNamespace

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


class _RolloutService:
    def __init__(self):
        self.closed = False

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
        return {"makespan": 1.0}

    def close(self):
        self.closed = True


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


def test_rollout_checkpoint_saves_latest_and_best(tmp_path) -> None:
    from train_lightning import RolloutCheckpoint

    saved_paths = []
    trainer = SimpleNamespace(
        save_checkpoint=lambda path: saved_paths.append(path)
    )
    module = SimpleNamespace(
        last_completed_episode=2,
        last_eval_metrics={"makespan": 10.0},
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
    module.last_eval_metrics = {"makespan": 12.0}
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
        return 10.0, 2.0, 3.0, [], 0.5, 0.6, 0.7

    monkeypatch.setattr("training.rollout_service.evaluate_model", fake_evaluate)
    service = APALRolloutService(
        agent=agent,
        vector_env=vector_env,
        eval_env=object(),
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
    output = capsys.readouterr().out
    assert "[Eval] ep=2 start" in output
    assert "Mk=10.00" in output
