from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from baselines.literature import common


def test_r5_training_source_returns_manifest_declared_paths(monkeypatch, tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    declared = (train_dir / "train_0001.csv", train_dir / "train_0002.csv")
    manifest = object()
    calls: list[tuple[object, Path]] = []

    monkeypatch.setattr(common, "load_reschedule_manifest", lambda path: manifest)

    def _resolve(current_manifest: object, configured_path: Path) -> tuple[Path, ...]:
        calls.append((current_manifest, configured_path))
        return declared

    monkeypatch.setattr(common, "resolve_r5_training_paths", _resolve)
    config = SimpleNamespace(
        enable_reschedule_mode=True,
        reschedule_async_protocol="r5_task_delay_v1",
        reschedule_manifest_path=str(tmp_path / "manifest.json"),
    )
    args = SimpleNamespace(train_data_path_or_dir=str(train_dir))

    result = common.resolve_literature_training_paths(args, config=config)

    assert result == declared
    assert calls == [(manifest, train_dir)]


def test_literature_policy_adapter_uses_deterministic_graph_action(monkeypatch) -> None:
    expected = (3, 1, [2, 4])
    observed: dict[str, object] = {}

    def _select(model, state, *, masks, device, deterministic, temperature, need_value):
        observed.update(
            {
                "model": model,
                "state": state,
                "masks": masks,
                "device": device,
                "deterministic": deterministic,
                "temperature": temperature,
                "need_value": need_value,
            }
        )
        return SimpleNamespace(action=expected, logprob=torch.tensor(0.0), value=None)

    monkeypatch.setattr(common, "select_graph_action", _select)
    model = object()
    device = torch.device("cpu")
    adapter = common.LiteraturePolicyAdapter(model, device)
    masks = (torch.zeros(5, dtype=torch.bool), torch.zeros((5, 2), dtype=torch.bool), torch.zeros(6, dtype=torch.bool))

    result = adapter.select_action(
        "state",
        mask_task=masks[0],
        mask_station_matrix=masks[1],
        mask_worker=masks[2],
        deterministic=True,
        temperature=0.0,
        is_eval=True,
    )

    assert result[0] == expected
    assert result[4] is False
    assert observed["model"] is model
    assert observed["device"] == device
    assert observed["deterministic"] is True
    assert observed["temperature"] == 0.0
    assert observed["need_value"] is False


def test_r5_training_source_validates_manifest_but_returns_train_directory(monkeypatch, tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    declared = (train_dir / "train_0001.csv", train_dir / "train_0002.csv")
    config = SimpleNamespace(
        reschedule_async_protocol="r5_task_delay_v1",
        train_data_path_or_dir=str(train_dir),
        reschedule_manifest_path=str(tmp_path / "manifest.json"),
    )
    monkeypatch.setattr(common, "configs", config)
    monkeypatch.setattr(common, "load_reschedule_manifest", lambda path: object())
    monkeypatch.setattr(common, "resolve_r5_training_paths", lambda manifest, path: declared)
    args = SimpleNamespace(train_data_path_or_dir=str(train_dir))

    assert common.training_data_source(args) == train_dir
