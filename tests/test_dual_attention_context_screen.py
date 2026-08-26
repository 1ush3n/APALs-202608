from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from configs import Config
from experiments.main_screen.screen_models import (
    DualAttentionContextHBGATPN,
    ScaleGatedContextHBGATPN,
)
from models.hb_gat_pn import HBGATPN
from models.worker_pointer_context import build_worker_pressure_context
from runtime.checkpoints import ModelSpec
from scripts.train_main_screen import _load_initial_weights, _resolve_screen_model_class
from scripts.evaluate_main_screen import (
    _extract_screen_model,
    _resolve_screen_model_class as _resolve_eval_screen_model_class,
)


class _BatchData:
    def __init__(self, *, batch_size: int = 2, hidden_dim: int = 8) -> None:
        self._stores = {
            "task": SimpleNamespace(
                batch=torch.tensor([0, 0, 1, 1], dtype=torch.long),
                x=torch.randn(4, 18),
            ),
            "station": SimpleNamespace(
                batch=torch.tensor([0, 0, 1, 1], dtype=torch.long),
                x=torch.randn(4, 15),
            ),
            "worker": SimpleNamespace(
                batch=torch.tensor([0, 0, 1, 1], dtype=torch.long),
                x=torch.randn(4, 17),
            ),
        }
        self.encoded = {
            name: torch.randn(store.x.size(0), hidden_dim)
            for name, store in self._stores.items()
        }
        assert batch_size == 2

    def __getitem__(self, name: str) -> SimpleNamespace:
        return self._stores[name]


def _config(*, v2: bool = False) -> Config:
    config = Config()
    config.hidden_dim = 8
    config.num_gat_layers = 0
    config.use_shared_trunk = True
    config.use_skill_hub = False
    config.skill_hub_bidirectional = False
    config.team_selection_mode = "autoregressive_pressure_v2" if v2 else "autoregressive"
    config.actor_context_mode = "attention"
    config.policy_action_scope = "operation_station_worker"
    config.seed = 42
    return config


def test_dual_attention_context_is_ordered_6h_and_task_logits_are_finite() -> None:
    config = _config()
    model = DualAttentionContextHBGATPN(config)
    batch = _BatchData(hidden_dim=config.hidden_dim)

    context = model._compute_global_context(
        batch.encoded,
        batch,
        mode="attention",
        station_attn=model.actor_station_attn,
        task_worker_attn=model.actor_task_worker_attn,
    )
    expected_first = HBGATPN._compute_global_context(
        model,
        batch.encoded,
        batch,
        mode="attention",
        station_attn=model.actor_station_attn,
        task_worker_attn=model.actor_task_worker_attn,
    )
    expected_second = HBGATPN._compute_global_context(
        model,
        batch.encoded,
        batch,
        mode="attention",
        station_attn=model.dual_attention_station_attn,
        task_worker_attn=model.dual_attention_task_worker_attn,
    )

    assert context.shape == (2, 6 * config.hidden_dim)
    torch.testing.assert_close(context[:, : 3 * config.hidden_dim], expected_first)
    torch.testing.assert_close(context[:, 3 * config.hidden_dim :], expected_second)

    task_logits = model.task_head(
        batch.encoded["task"],
        context,
        mask=torch.zeros((2, 4), dtype=torch.bool),
    )
    assert task_logits.shape == (2, 4)
    assert torch.isfinite(task_logits).all()


def test_dual_attention_has_independent_scorers_and_trainable_second_projection() -> None:
    config = _config()
    model = DualAttentionContextHBGATPN(config)
    first = model.actor_station_attn[0].weight
    second = model.dual_attention_station_attn[0].weight

    assert first.data_ptr() != second.data_ptr()
    assert not torch.equal(first, second)

    batch = _BatchData(hidden_dim=config.hidden_dim)
    context = model._compute_global_context(
        batch.encoded,
        batch,
        mode="attention",
        station_attn=model.actor_station_attn,
        task_worker_attn=model.actor_task_worker_attn,
    )
    model.task_head(context.new_zeros((4, config.hidden_dim)), context).sum().backward()
    grad = model.task_head.context_proj.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert torch.linalg.vector_norm(grad[:, 3 * config.hidden_dim :]) > 0


def test_dual_attention_warm_start_preserves_old_task_logits_and_zeroes_second_half(
    tmp_path: Path,
) -> None:
    config = _config()
    old_model = HBGATPN(config)
    checkpoint_path = tmp_path / "old.ckpt"
    torch.save(
        {
            "state_dict": old_model.state_dict(),
            "apal_metadata": {
                "format_version": 2,
                "model_spec": asdict(
                    ModelSpec(
                        resource_graph_mode="legacy_direct",
                        team_selection_mode=config.team_selection_mode,
                        actor_context_mode="attention",
                        hidden_dim=config.hidden_dim,
                    )
                ),
            },
        },
        checkpoint_path,
    )

    dual_model = DualAttentionContextHBGATPN(config)
    report = _load_initial_weights(dual_model, checkpoint_path, strict=True)
    assert report["unexpected"] == []
    assert report["warm_start_transformed"] is True
    assert report["missing"]
    assert all(key.startswith("dual_attention_") for key in report["missing"])

    full_report = _load_initial_weights(HBGATPN(config), checkpoint_path, strict=True)
    assert full_report["missing"] == []
    scg_report = _load_initial_weights(
        ScaleGatedContextHBGATPN(config), checkpoint_path, strict=True
    )
    assert scg_report["unexpected"] == []
    assert scg_report["missing"]
    assert all(key.startswith("screen_") for key in scg_report["missing"])

    old_weight = old_model.task_head.context_proj.weight.detach()
    new_weight = dual_model.task_head.context_proj.weight.detach()
    assert new_weight.shape == (config.hidden_dim, 6 * config.hidden_dim)
    torch.testing.assert_close(new_weight[:, : 3 * config.hidden_dim], old_weight)
    torch.testing.assert_close(
        new_weight[:, 3 * config.hidden_dim :], torch.zeros_like(new_weight[:, 3 * config.hidden_dim :])
    )
    torch.testing.assert_close(
        dual_model.task_head.context_proj.bias,
        old_model.task_head.context_proj.bias,
    )

    batch = _BatchData(hidden_dim=config.hidden_dim)
    old_context = old_model._compute_global_context(
        batch.encoded,
        batch,
        mode="attention",
        station_attn=old_model.actor_station_attn,
        task_worker_attn=old_model.actor_task_worker_attn,
    )
    dual_context = dual_model._compute_global_context(
        batch.encoded,
        batch,
        mode="attention",
        station_attn=dual_model.actor_station_attn,
        task_worker_attn=dual_model.actor_task_worker_attn,
    )
    old_logits = old_model.task_head(batch.encoded["task"], old_context)
    dual_logits = dual_model.task_head(batch.encoded["task"], dual_context)
    torch.testing.assert_close(dual_logits, old_logits, atol=1.0e-6, rtol=0.0)


class _RecordingCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def forward(self, value_input: torch.Tensor) -> torch.Tensor:
        self.inputs.append(value_input.detach())
        return value_input[:, :1]


def test_dual_attention_critic_keeps_single_3h_context() -> None:
    config = _config()
    model = DualAttentionContextHBGATPN(config)
    critic = _RecordingCritic()
    model.critic = critic
    batch = _BatchData(hidden_dim=config.hidden_dim)

    value = model.get_value(batch, actor_x_dict_encoded=batch.encoded)

    assert critic.inputs[-1].shape == (2, 3 * config.hidden_dim)
    assert value.shape == (2, 1)
    assert torch.isfinite(value).all()


def test_screen_model_selection_preserves_full_and_scg() -> None:
    assert _resolve_screen_model_class("full") is HBGATPN
    assert _resolve_screen_model_class("scg") is ScaleGatedContextHBGATPN
    assert _resolve_screen_model_class("dual_attention") is DualAttentionContextHBGATPN
    with pytest.raises(ValueError, match="full、scg 或 dual_attention"):
        _resolve_screen_model_class("unknown")
    assert _resolve_eval_screen_model_class("full") is HBGATPN
    assert _resolve_eval_screen_model_class("scg") is ScaleGatedContextHBGATPN
    assert _extract_screen_model(["screen_model=dual_attention", "no_gantt=true"]) == (
        "dual_attention",
        ["no_gantt=true"],
    )


def test_dual_attention_worker_v2_forward_smoke() -> None:
    config = _config(v2=True)
    model = DualAttentionContextHBGATPN(config)
    batch = _BatchData(batch_size=2, hidden_dim=config.hidden_dim)
    global_context = model._compute_global_context(
        batch.encoded,
        batch,
        mode="attention",
        station_attn=model.actor_station_attn,
        task_worker_attn=model.actor_task_worker_attn,
    )[:1]
    pressure_inputs = {
        "task_features": torch.zeros((1, 4, 18)),
        "worker_features": torch.zeros((1, 3, 17)),
        "task_present": torch.tensor([[True, False, False, False]]),
        "task_action_invalid": torch.tensor([[False, True, True, True]]),
        "worker_present": torch.tensor([[True, True, False]]),
        "worker_queue_invalid": torch.tensor([[False, False, True]]),
    }
    pressure = build_worker_pressure_context(
        **pressure_inputs,
        temperature=1.0,
        supply_epsilon=1.0e-6,
    )
    state = model.worker_head.initialize_v2_state(batch_size=1, device=torch.device("cpu"))
    logits = model.worker_head.forward_choice_v2(
        task_emb=batch.encoded["task"][:1],
        station_emb=batch.encoded["station"][:1],
        global_context=global_context,
        worker_embs=torch.randn((1, 3, config.hidden_dim)),
        pressure_context=pressure,
        team_state=state,
        demand=torch.tensor([1.0]),
        mask=torch.tensor([[False, False, True]]),
    )

    assert logits.shape == (1, 3)
    assert torch.isfinite(logits).all()
