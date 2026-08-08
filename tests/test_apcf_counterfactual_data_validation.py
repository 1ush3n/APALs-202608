# -*- coding: utf-8 -*-
"""APCF 反事实数据资产独立审计器的行为测试。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_manifest(
    tmp_path: Path, *, first_filename: str = "syn_000.csv"
) -> tuple[Path, dict[str, list[str]]]:
    hashes = [hashlib.sha256(f"graph-{i}".encode()).hexdigest() for i in range(160)]
    ordered = sorted(hashes)
    split = {
        "pretrain": ordered[:96],
        "frozen_diagnostic": ordered[96:120],
        "ppo_only": ordered[120:],
    }
    payload = {
        "protocol": "explicit_fiveskill_v1",
        "files": [
            {
                "file": first_filename if graph_sha == ordered[0] else f"syn_{i:03d}.csv",
                "sha256": graph_sha,
            }
            for i, graph_sha in enumerate(hashes)
        ],
    }
    path = tmp_path / "source_manifest.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path, split


def _write_asset(
    tmp_path: Path,
    *,
    source_path: Path,
    split: dict[str, list[str]],
    manifest_gain: float = 0.0,
    tamper_sample_digest: bool = False,
    include_anchor: bool = True,
) -> Path:
    root = tmp_path / "asset"
    samples = root / "samples"
    samples.mkdir(parents=True)
    source_sha = _sha256(source_path)
    task_x = np.asarray([[1.0, 2.0]], dtype=np.float32)
    worker_x = np.asarray([[1.0, 0.0], [0.5, 1.0]], dtype=np.float32)
    station_x = np.asarray([[0.0, 1.0]], dtype=np.float32)
    task_mask = np.asarray([False], dtype=np.bool_)
    station_mask = np.asarray([[False]], dtype=np.bool_)
    worker_mask = np.asarray([False, False], dtype=np.bool_)
    arrays = {
        "task_x": task_x,
        "worker_x": worker_x,
        "station_x": station_x,
        "task_mask": task_mask,
        "station_mask": station_mask,
        "worker_mask": worker_mask,
    }
    graph_sha = split["pretrain"][0]
    meta = {
        "csv_sha256": graph_sha,
        "manifest_sha256": source_sha,
        "state_seed": "state-0",
        "task_id": 0,
        "station_id": 0,
        "anchor_team": [0],
        "candidate_team": [0],
        "source": "anchor" if include_anchor else "one_swap",
        "baseline_makespan": 10.0,
        "candidate_makespan": 10.0,
        "relative_gain": 0.0,
    }
    meta_bytes = json.dumps(meta, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(meta_bytes)
    for name in ("task_x", "worker_x", "station_x", "task_mask", "station_mask", "worker_mask"):
        digest.update(np.ascontiguousarray(arrays[name]).tobytes())
    sample_sha = digest.hexdigest()
    if tamper_sample_digest:
        sample_sha = "0" * 64
    npz = samples / "sample.npz"
    np.savez_compressed(npz, meta=meta_bytes, **arrays)
    obs = {
        "task": SimpleNamespace(x=torch.from_numpy(task_x)),
        "worker": SimpleNamespace(x=torch.from_numpy(worker_x)),
        "station": SimpleNamespace(x=torch.from_numpy(station_x)),
    }
    obs_path = samples / "obs.pt"
    torch.save(obs, obs_path)
    entry = {
        "csv_sha256": graph_sha,
        "split": "pretrain",
        "state_seed": "state-0",
        "task_id": 0,
        "station_id": 0,
        "anchor_team": [0],
        "candidate_team": [0],
        "source": meta["source"],
        "sample_sha256": sample_sha,
        "npz": "samples/sample.npz",
        "obs_pt": "samples/obs.pt",
        "relative_gain": manifest_gain,
    }
    asset = {
        "version": 1,
        "kind": "initial_anchor_proposal_counterfactual_v1",
        "source_manifest_path": str(source_path),
        "source_manifest_sha256": source_sha,
        "split": split,
        "split_counts": {key: len(value) for key, value in split.items()},
        "sample_counts": {"pretrain": 1, "frozen_diagnostic": 0, "ppo_only": 0},
        "candidate_budget": {"one_swap_top_k": 2, "two_swap_top_k": 2, "two_swap_pool": 24, "hash_two_swap_representative": 1},
        "state_fractions": [0.125, 0.375, 0.625, 0.875],
        "command_args": {},
        "files": [entry],
    }
    asset["manifest_sha256"] = hashlib.sha256(
        json.dumps(asset, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    (root / "manifest.json").write_text(
        json.dumps(asset, sort_keys=True, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return root


def test_validator_accepts_consistent_npz_meta_and_manifest(tmp_path: Path) -> None:
    from scripts.validate_anchor_proposal_cf_data import validate_counterfactual_asset

    source_path, split = _source_manifest(tmp_path)
    asset_dir = _write_asset(tmp_path, source_path=source_path, split=split)

    report = validate_counterfactual_asset(asset_dir, source_path)

    assert report["status"] == "passed"
    assert report["sample_count"] == 1
    assert report["state_count"] == 1


def test_validator_rejects_gain_disagreement_between_manifest_and_npz_meta(tmp_path: Path) -> None:
    from scripts.validate_anchor_proposal_cf_data import validate_counterfactual_asset

    source_path, split = _source_manifest(tmp_path)
    asset_dir = _write_asset(
        tmp_path, source_path=source_path, split=split, manifest_gain=0.01
    )

    with pytest.raises(ValueError, match="relative_gain"):
        validate_counterfactual_asset(asset_dir, source_path)


def test_validator_rejects_tampered_npz_sample_digest(tmp_path: Path) -> None:
    from scripts.validate_anchor_proposal_cf_data import validate_counterfactual_asset

    source_path, split = _source_manifest(tmp_path)
    asset_dir = _write_asset(
        tmp_path, source_path=source_path, split=split, tamper_sample_digest=True
    )

    with pytest.raises(ValueError, match="sample_sha256"):
        validate_counterfactual_asset(asset_dir, source_path)


def test_validator_rejects_real_four_source_filename(tmp_path: Path) -> None:
    from scripts.validate_anchor_proposal_cf_data import validate_counterfactual_asset

    source_path, split = _source_manifest(tmp_path, first_filename="680.csv")
    asset_dir = _write_asset(
        tmp_path, source_path=source_path, split=split
    )

    with pytest.raises(ValueError, match="真实四实例"):
        validate_counterfactual_asset(asset_dir, source_path)


def test_validator_rejects_state_without_anchor_row(tmp_path: Path) -> None:
    from scripts.validate_anchor_proposal_cf_data import validate_counterfactual_asset

    source_path, split = _source_manifest(tmp_path)
    asset_dir = _write_asset(
        tmp_path, source_path=source_path, split=split, include_anchor=False
    )

    with pytest.raises(ValueError, match="anchor"):
        validate_counterfactual_asset(asset_dir, source_path)
