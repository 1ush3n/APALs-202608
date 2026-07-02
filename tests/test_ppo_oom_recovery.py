from __future__ import annotations

from types import MethodType, SimpleNamespace

import torch

from configs import configs
from ppo_agent import PPOAgent
from tests.runtime_safety import temporary_config
from utils.gpu_graph_manager import GPUBatchGraphManager


def _make_tiny_agent(batch_size: int = 4) -> PPOAgent:
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 1),
    )
    return PPOAgent(
        model=model,
        lr=1.0e-3,
        gamma=0.99,
        k_epochs=1,
        eps_clip=0.2,
        device=torch.device("cpu"),
        batch_size=batch_size,
        total_timesteps=2,
    )


def test_gpu_graph_manager_evicts_other_datasets() -> None:
    manager = GPUBatchGraphManager(torch.device("cpu"))
    manager.templates[(0, 4)] = object()
    manager.templates[(0, 2)] = object()
    manager.templates[(1, 4)] = object()

    removed = manager.retain_dataset(1)

    assert removed == 2
    assert set(manager.templates) == {(1, 4)}
    assert manager.clear() == 1
    assert manager.templates == {}


def test_oom_rolls_back_partial_update_and_keeps_batch_size() -> None:
    overrides = {
        "use_schedule_free": False,
        "use_ema": False,
        "ppo_batch_size_cap": 0,
        "auto_oom_retry": False,
        "skip_update_on_oom": True,
        "oom_min_batch_size": 1,
        "oom_max_retries": 0,
        "oom_transactional_updates": True,
    }
    with temporary_config(configs, overrides):
        agent = _make_tiny_agent(batch_size=4)
        initial = {
            name: value.detach().clone()
            for name, value in agent.policy.state_dict().items()
        }
        def fake_update_once(self, memory, env=None, current_ep=1):
            with torch.no_grad():
                next(self.policy.parameters()).add_(10.0)
            self.current_step = 9
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

        agent._update_once = MethodType(fake_update_once, agent)
        metrics = agent.update(SimpleNamespace(), current_ep=3)

    assert agent.batch_size == 4
    assert metrics["OOM/SkippedUpdate"] == 1.0
    assert metrics["_skip_training_log"] == 1.0
    assert agent.current_step == 0
    for name, value in agent.policy.state_dict().items():
        assert torch.equal(value, initial[name])


def test_oom_retry_exhaustion_safely_skips_update() -> None:
    overrides = {
        "use_schedule_free": False,
        "use_ema": False,
        "ppo_batch_size_cap": 0,
        "auto_oom_retry": False,
        "skip_update_on_oom": True,
        "oom_min_batch_size": 1,
        "oom_max_retries": 0,
        "oom_transactional_updates": True,
    }
    with temporary_config(configs, overrides):
        agent = _make_tiny_agent(batch_size=2)
        initial = {
            name: value.detach().clone()
            for name, value in agent.policy.state_dict().items()
        }

        def always_oom(self, memory, env=None, current_ep=1):
            with torch.no_grad():
                next(self.policy.parameters()).mul_(0.0)
            self.current_step = 7
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

        agent._update_once = MethodType(always_oom, agent)
        metrics = agent.update(SimpleNamespace(), current_ep=5)

    assert metrics["OOM/SkippedUpdate"] == 1.0
    assert metrics["_skip_training_log"] == 1.0
    assert agent.batch_size == 2
    assert agent.current_step == 0
    for name, value in agent.policy.state_dict().items():
        assert torch.equal(value, initial[name])
