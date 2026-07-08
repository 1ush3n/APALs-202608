from __future__ import annotations

from types import SimpleNamespace

from training.rollout_service import APALRolloutService


def _service_for_tasks(num_tasks: int, **overrides):
    values = {
        "adaptive_ppo_batch_by_tasks": True,
        "adaptive_ppo_batch_small_task_max": 530,
        "adaptive_ppo_batch_large_task_min": 550,
        "adaptive_ppo_batch_small": 128,
        "adaptive_ppo_batch_large": 64,
        "batch_size": 128,
        "ppo_batch_size_cap": 0,
    }
    values.update(overrides)
    config = SimpleNamespace(**values)
    service = object.__new__(APALRolloutService)
    service.config = config
    service.agent = SimpleNamespace(batch_size=int(config.batch_size))
    service.vector_env = SimpleNamespace(envs=[SimpleNamespace(num_tasks=int(num_tasks))])
    service._last_effective_batch_size = None
    return service


def test_adaptive_ppo_batch_uses_requested_thresholds() -> None:
    assert _service_for_tasks(529)._adaptive_batch_for_task_count(529) == (128, "tasks<=530")
    assert _service_for_tasks(530)._adaptive_batch_for_task_count(530) == (128, "tasks<=530")
    assert _service_for_tasks(540)._adaptive_batch_for_task_count(540) == (128, "530<tasks<550")
    assert _service_for_tasks(550)._adaptive_batch_for_task_count(550) == (64, "tasks>=550")
    assert _service_for_tasks(680)._adaptive_batch_for_task_count(680) == (64, "tasks>=550")


def test_adaptive_ppo_batch_applies_cap_without_changing_model_state() -> None:
    service = _service_for_tasks(680, ppo_batch_size_cap=32)
    service._apply_adaptive_ppo_batch(dataset_idx=0)
    assert service.agent.batch_size == 32


def test_adaptive_ppo_batch_can_be_disabled() -> None:
    service = _service_for_tasks(680, adaptive_ppo_batch_by_tasks=False)
    service._apply_adaptive_ppo_batch(dataset_idx=0)
    assert service.agent.batch_size == 128
