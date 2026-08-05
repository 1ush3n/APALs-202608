from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from schedulefree import AdamWScheduleFree

from runtime.schedulefree_checkpoint import save_checkpoint_with_schedulefree_eval_parameters
from runtime.schedulefree_export import export_schedulefree_eval_payload
from scripts.export_schedulefree_eval_checkpoint import optional_finite_score
from training.async_eval_worker import load_checkpoint_agent_for_evaluation
from training.async_evaluation import _save_async_candidate_checkpoint


def _clone_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def test_schedulefree_save_exports_eval_parameters_and_restores_live_training_state(tmp_path: Path) -> None:
    """保存的 checkpoint 必须是 x 参数，内存中的训练模型必须恢复为 y 参数。"""
    torch.manual_seed(7)
    model = torch.nn.Linear(3, 1, bias=False)
    optimizer = AdamWScheduleFree(model.parameters(), lr=0.1, warmup_steps=1, foreach=False)
    optimizer.train()
    loss = model(torch.tensor([[1.0, -2.0, 3.0]])).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    y_state = _clone_state_dict(model)
    optimizer.eval()
    x_state = _clone_state_dict(model)
    optimizer.train()
    assert all(torch.equal(model.state_dict()[name], value) for name, value in y_state.items())

    captured: dict[str, object] = {}

    def _save(path: Path) -> None:
        captured["path"] = path
        captured["state"] = _clone_state_dict(model)
        captured["modes"] = [group["train_mode"] for group in optimizer.param_groups]

    result = save_checkpoint_with_schedulefree_eval_parameters(
        save_checkpoint=_save,
        path=tmp_path / "candidate.ckpt",
        optimizer=optimizer,
        schedulefree_enabled=True,
    )

    assert result.source_mode == "train_y"
    assert result.saved_mode == "eval_x"
    assert result.restored_mode == "train_y"
    assert captured["modes"] == [False]
    saved_state = captured["state"]
    assert isinstance(saved_state, dict)
    assert all(torch.equal(saved_state[name], value) for name, value in x_state.items())
    assert all(torch.equal(model.state_dict()[name], value) for name, value in y_state.items())
    assert [group["train_mode"] for group in optimizer.param_groups] == [True]


def test_non_schedulefree_checkpoint_save_keeps_existing_semantics(tmp_path: Path) -> None:
    """非 ScheduleFree 优化器必须不改变原有保存流程。"""
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    saved: list[Path] = []

    state = save_checkpoint_with_schedulefree_eval_parameters(
        save_checkpoint=saved.append,
        path=tmp_path / "plain.ckpt",
        optimizer=optimizer,
        schedulefree_enabled=False,
    )

    assert saved == [tmp_path / "plain.ckpt"]
    assert state.source_mode == state.saved_mode == state.restored_mode == "disabled"


def test_sync_rollout_checkpoint_saves_schedulefree_eval_parameters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """同步 best/latest 保存时必须观察到 eval_x，而训练实例仍回到 train_y。"""
    import train_lightning

    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = AdamWScheduleFree(model.parameters(), lr=0.1, warmup_steps=1, foreach=False)
    optimizer.train()
    loss = model(torch.tensor([[1.0, 2.0]])).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    monkeypatch.setattr(train_lightning.configs, "async_eval_enabled", False)
    callback = train_lightning.RolloutCheckpoint(tmp_path)
    saved_modes: list[list[bool]] = []
    trainer = SimpleNamespace(
        save_checkpoint=lambda _path: saved_modes.append(
            [group["train_mode"] for group in optimizer.param_groups]
        )
    )
    module = SimpleNamespace(
        agent=SimpleNamespace(optimizer=optimizer, use_schedule_free=True),
        last_completed_episode=1,
        last_eval_metrics={"makespan": 10.0, "completion_rate": 1.0},
        last_update_committed=True,
    )

    callback.on_train_batch_end(trainer, module, None, None, 0)

    assert saved_modes == [[False], [False]]
    assert [group["train_mode"] for group in optimizer.param_groups] == [True]


def test_async_candidate_checkpoint_saves_schedulefree_eval_parameters(tmp_path: Path) -> None:
    """异步候选 checkpoint 也必须从 eval_x 发布，随后恢复训练态。"""
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = AdamWScheduleFree(model.parameters(), lr=0.1, warmup_steps=1, foreach=False)
    optimizer.train()
    loss = model(torch.tensor([[1.0, 2.0]])).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    saved_modes: list[list[bool]] = []
    trainer = SimpleNamespace(
        lightning_module=SimpleNamespace(agent=SimpleNamespace(optimizer=optimizer, use_schedule_free=True)),
        save_checkpoint=lambda _path: saved_modes.append(
            [group["train_mode"] for group in optimizer.param_groups]
        ),
    )

    _save_async_candidate_checkpoint(trainer, tmp_path / "candidate.ckpt")

    assert saved_modes == [[False]]
    assert [group["train_mode"] for group in optimizer.param_groups] == [True]


def test_export_payload_rewrites_policy_and_optimizer_to_eval_x_without_mutating_source() -> None:
    """迁移副本必须保存 x、保留源 payload 的 y，并恢复导出进程内存状态。"""
    torch.manual_seed(11)
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = AdamWScheduleFree(model.parameters(), lr=0.1, warmup_steps=1, foreach=False)
    optimizer.train()
    loss = model(torch.tensor([[2.0, -1.0]])).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    y_state = _clone_state_dict(model)
    source_payload = {
        "state_dict": {f"policy.{name}": value.detach().clone() for name, value in y_state.items()},
        "optimizer_states": [optimizer.state_dict()],
        "apal_metadata": {"format_version": 2},
    }
    optimizer.eval()
    x_state = _clone_state_dict(model)
    optimizer.train()

    exported = export_schedulefree_eval_payload(
        payload=source_payload,
        policy=model,
        optimizer=optimizer,
    )

    assert [group["train_mode"] for group in exported["optimizer_states"][0]["param_groups"]] == [False]
    assert all(
        torch.equal(exported["state_dict"][f"policy.{name}"], value.cpu())
        for name, value in x_state.items()
    )
    assert all(
        torch.equal(source_payload["state_dict"][f"policy.{name}"], value)
        for name, value in y_state.items()
    )
    assert [group["train_mode"] for group in optimizer.param_groups] == [True]
    assert all(torch.equal(model.state_dict()[name], value) for name, value in y_state.items())


def test_checkpoint_agent_loader_is_public_for_migration_tools() -> None:
    """迁移工具必须复用异步 worker 的完整 checkpoint 恢复语义。"""
    assert callable(load_checkpoint_agent_for_evaluation)


def test_conversion_audit_rejects_non_finite_callback_score() -> None:
    """异步选择未写入 RolloutCallback 分数时，审计 JSON 不得写入 Infinity。"""
    assert optional_finite_score(0.970978) == pytest.approx(0.970978)
    assert optional_finite_score(float("inf")) is None
    assert optional_finite_score(float("nan")) is None
    assert optional_finite_score("not-a-score") is None
