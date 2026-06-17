from __future__ import annotations

from pathlib import Path

import pytest
import torch

from configs import Config
from models.hb_gat_pn import HBGATPN
from runtime.checkpoints import (
    apply_checkpoint_model_spec,
    build_checkpoint_metadata,
    infer_model_spec,
    load_checkpoint,
)


def _state_for(*, use_skill_hub: bool, bidirectional: bool) -> tuple[Config, dict]:
    cfg = Config()
    cfg.use_skill_hub = use_skill_hub
    cfg.skill_hub_bidirectional = bidirectional if use_skill_hub else False
    return cfg, HBGATPN(cfg).state_dict()


@pytest.mark.parametrize(
    ("use_skill_hub", "bidirectional", "expected"),
    [
        (False, False, "legacy_direct"),
        (True, False, "skill_hub_forward"),
        (True, True, "skill_hub_bidirectional"),
    ],
)
def test_infers_resource_graph_mode(use_skill_hub, bidirectional, expected) -> None:
    _cfg, state = _state_for(
        use_skill_hub=use_skill_hub,
        bidirectional=bidirectional,
    )
    assert infer_model_spec(state).resource_graph_mode == expected


def test_loads_lightning_and_legacy_checkpoint_formats(tmp_path: Path) -> None:
    cfg, state = _state_for(use_skill_hub=False, bidirectional=False)
    metadata = build_checkpoint_metadata(cfg)
    lightning_path = tmp_path / "model.ckpt"
    torch.save({
        "state_dict": {f"policy.{key}": value for key, value in state.items()},
        "apal_metadata": metadata,
    }, lightning_path)
    loaded = load_checkpoint(lightning_path)
    assert loaded.format_name == "lightning"
    assert loaded.model_spec.resource_graph_mode == "legacy_direct"
    assert set(loaded.state_dict) == set(state)

    legacy_path = tmp_path / "model.pth"
    torch.save({"model_state_dict": state, "apal_metadata": metadata}, legacy_path)
    assert load_checkpoint(legacy_path).format_name == "legacy_full"


def test_explicit_structural_conflict_is_rejected() -> None:
    cfg = Config()
    cfg.use_skill_hub = True
    _legacy_cfg, state = _state_for(use_skill_hub=False, bidirectional=False)
    spec = infer_model_spec(state)

    with pytest.raises(ValueError, match="checkpoint 冲突"):
        apply_checkpoint_model_spec(cfg, spec, explicit_fields={"use_skill_hub"})

    apply_checkpoint_model_spec(cfg, spec, explicit_fields=set())
    assert cfg.use_skill_hub is False
