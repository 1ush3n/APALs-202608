# -*- coding: utf-8 -*-
"""APCF gate 鏍囧畾銆佺敓浜ц矾寰勫悓鏋勪笌鎺ㄧ悊鏈?prior 鎵弿璇婃柇銆?
鏈剼鏈彧璇诲姞杞?checkpoint 鍜?APCF 璧勪骇锛屼笉鎵ц PPO銆佷笉鍙嶅悜浼犳挱锛屼篃涓嶅啓鍏ユ寮忕粨鏋滅洰褰曘€?"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import inspect
import json
import math
import re
import sys
import textwrap
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import lightning.pytorch as pl
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import Config, configs as global_configs, load_training_config
from core.action_completion import EarliestFinishActionCompleter
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from runtime.checkpoints import apply_checkpoint_model_spec, load_checkpoint, load_policy_weights
from runtime.configuration import validate_runtime_config
from runtime.initial_worker_mapping import apply_initial_worker_mapping
from scripts.audit_initial_team_opportunity_full import _configure as configure_replay
from scripts.build_anchor_proposal_cf_data import (
    _make_state_key,
    _plan_graph_jobs,
    _select_sample_states,
    _select_pair,
    _split_by_sha256,
)
from worker_feature_layout import resolve_worker_feature_layout


EXPECTED_ASSET_RAW_SHA = "6cd61afa7e3478b591570d01362d3788dd65699a8cc3afa24f35861b847b11aa"
EXPECTED_SOURCE_SHA = "becdd7a45d37fe628a6b67981ad8af00c8a075dfe1955c042d5d2b0c4d894a50"
EXPECTED_SCOPE = "operation_station_anchor_proposal_team"
EXPECTED_MODE = "full_team_v1"
EXPECTED_FROZEN_GRAPH_COUNT = 24
EXPECTED_FROZEN_STATE_COUNT = 96
EXPECTED_FROZEN_CANDIDATE_COUNT = 504
DEFAULT_PRIOR_LOGITS = (-4.0, -2.0, -1.0, 0.0)
MAX_EPISODE_STEPS = 1200
MAX_CANDIDATES = 4


@dataclass(frozen=True)
class FormulaSpec:
    prior_logit: float
    residual_scale: float
    delta_temperature: float
    anchor_proposal_mode: str
    source: str


def reconstruct_residual_term(
    spec: FormulaSpec,
    *,
    gate_value: float,
    predicted_delta_a: float,
) -> float:
    # 使用实际 gate 公式计算 proposal 的 gate 加权残差。
    if not all(math.isfinite(float(value)) for value in (gate_value, predicted_delta_a)):
        raise ValueError("gate_value 鍜?predicted_delta_A 蹇呴』鏈夐檺")
    if spec.delta_temperature <= 0.0 or spec.residual_scale <= 0.0:
        raise ValueError("鍏紡涓殑 residual_scale 鍜?delta_temperature 蹇呴』涓烘")
    return float(
        gate_value
        * spec.residual_scale
        * math.tanh(predicted_delta_a / spec.delta_temperature)
    )


def reconstruct_raw_gap(
    spec: FormulaSpec,
    *,
    gate_value: float,
    predicted_delta_a: float,
) -> float:
    return float(
        spec.prior_logit
        + reconstruct_residual_term(
            spec,
            gate_value=gate_value,
            predicted_delta_a=predicted_delta_a,
        )
    )


def sweep_gap(prior_logit: float, residual_term: float) -> float:
    return float(prior_logit + residual_term)


def expected_raw_branch(raw_gap: float) -> int:
    return int(float(raw_gap) > 0.0)


def normalize_hamming_distance(hamming_distance: int, required_team_size: int) -> float:
    if int(required_team_size) <= 0:
        raise ValueError("required_team_size 蹇呴』涓烘")
    if int(hamming_distance) < 0:
        raise ValueError("hamming_distance 涓嶈兘涓鸿礋")
    return float(hamming_distance) / float(required_team_size)


def policy_state_fields(
    *,
    proposal_available: bool,
    proposal_team: Iterable[int] | None,
    hamming_distance: int | None,
    required_team_size: int,
    gate_logit: float | None,
    gate_value: float | None,
    predicted_delta_a: float | None,
    raw_branch_logit_gap: float | None,
    production_raw_branch: int | None,
) -> dict[str, Any]:
    if not proposal_available:
        return {
            "proposal_team": None,
            "hamming_distance": None,
            "normalized_hamming_distance": None,
            "gate_logit": None,
            "gate_value": None,
            "predicted_delta_A": None,
            "raw_branch_logit_gap": None,
            "production_raw_branch": None,
        }
    if hamming_distance is None:
        raise ValueError("鍙敤 proposal 蹇呴』鎻愪緵 hamming_distance")
    return {
        "proposal_team": [int(worker) for worker in (proposal_team or ())],
        "hamming_distance": int(hamming_distance),
        "normalized_hamming_distance": normalize_hamming_distance(
            int(hamming_distance), int(required_team_size)
        ),
        "gate_logit": float(gate_logit),
        "gate_value": float(gate_value),
        "predicted_delta_A": float(predicted_delta_a),
        "raw_branch_logit_gap": float(raw_branch_logit_gap),
        "production_raw_branch": int(production_raw_branch),
    }


def validate_production_branch(
    production_raw_branch: int,
    diagnostic_reconstructed_gap: float,
    *,
    proposal_available: bool,
) -> bool:
    if not proposal_available:
        return True
    expected = expected_raw_branch(diagnostic_reconstructed_gap)
    assert int(production_raw_branch) == expected, (
        "production raw branch 涓?diagnostic reconstructed gap 涓嶄竴鑷? "
        f"production={production_raw_branch}, expected={expected}, "
        f"gap={diagnostic_reconstructed_gap}"
    )
    return True


def assert_candidate_scores_close(
    production: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-6,
) -> None:
    for field_name in ("gate_logit", "predicted_delta_A", "raw_gap"):
        if field_name not in production or field_name not in candidate:
            raise AssertionError(f"candidate scorer 缂哄皯瀛楁: {field_name}")
        actual = float(candidate[field_name])
        expected = float(production[field_name])
        if not math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol):
            raise AssertionError(
                "candidate scorer 涓?production scorer 涓嶄竴鑷? "
                f"field={field_name}, production={expected}, candidate={actual}"
            )


def binary_classification_metrics(
    labels: Iterable[bool], predictions: Iterable[bool]
) -> dict[str, float]:
    labels_list = [bool(value) for value in labels]
    predictions_list = [bool(value) for value in predictions]
    if len(labels_list) != len(predictions_list) or not labels_list:
        raise ValueError("labels and predictions must be non-empty and have equal length")
    tp = sum(label and prediction for label, prediction in zip(labels_list, predictions_list))
    fp = sum((not label) and prediction for label, prediction in zip(labels_list, predictions_list))
    fn = sum(label and (not prediction) for label, prediction in zip(labels_list, predictions_list))
    tn = sum((not label) and (not prediction) for label, prediction in zip(labels_list, predictions_list))
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": float(tp + fn),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: float, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} 蹇呴』鏈夐檺锛屽疄闄呬负 {value!r}")
    return result


def _as_tuple_team(value: Iterable[Any]) -> tuple[int, ...]:
    return tuple(int(worker) for worker in value)


def _parse_state_seed(state_seed: str, csv_sha256: str) -> int:
    parts = str(state_seed).split("|")
    if len(parts) != 4 or parts[0] != csv_sha256:
        raise ValueError(f"闈炴硶 state_seed: {state_seed!r}")
    if int(parts[1]) < 0 or int(parts[2]) < 0:
        raise ValueError(f"闈炴硶 state_seed task/station: {state_seed!r}")
    return int(parts[3])


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 椤跺眰蹇呴』涓哄璞? {path}")
    return value


def _walk_numeric_constants(node: ast.AST) -> list[float]:
    values: list[float] = []
    for item in ast.walk(node):
        if isinstance(item, ast.Constant) and isinstance(item.value, (int, float)):
            values.append(float(item.value))
    return values


def _contains_name(node: ast.AST, name: str) -> bool:
    return any(isinstance(item, ast.Name) and item.id == name for item in ast.walk(node))


def _contains_tanh_call(node: ast.AST) -> bool:
    return any(
        isinstance(item, ast.Call)
        and isinstance(item.func, ast.Attribute)
        and item.func.attr == "tanh"
        for item in ast.walk(node)
    )


def _extract_gate_formula_constants(gate_module: torch.nn.Module) -> tuple[float, float, str]:
    try:
        source = textwrap.dedent(inspect.getsource(type(gate_module).forward))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError) as exc:
        raise RuntimeError('cannot read the production AnchorProposalGate.forward formula') from exc

    proposal_expr: ast.AST | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "proposal_logit" for target in node.targets):
                proposal_expr = node.value
                break
    if proposal_expr is None:
        raise RuntimeError("AnchorProposalGate.forward 鏈壘鍒?proposal_logit 鍏紡")

    delta_temperature: float | None = None
    for node in ast.walk(proposal_expr):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
            continue
        if isinstance(node.left, ast.Name) and node.left.id == "delta_a":
            constants = _walk_numeric_constants(node.right)
            if len(constants) == 1 and constants[0] > 0.0:
                delta_temperature = constants[0]
                break

    residual_scale: float | None = None
    for node in ast.walk(proposal_expr):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
            continue
        if not _contains_name(node, "g") or not _contains_tanh_call(node):
            continue
        constants = [value for value in _walk_numeric_constants(node) if value > 0.0]
        if delta_temperature is not None:
            constants = [value for value in constants if not math.isclose(value, delta_temperature)]
        if len(constants) == 1:
            residual_scale = constants[0]
            break

    if residual_scale is None or delta_temperature is None:
        raise RuntimeError("鏃犳硶浠庣敓浜?AnchorProposalGate.forward 鎻愬彇 residual_scale/delta_temperature")
    return residual_scale, delta_temperature, source


def _resolve_formula_spec(
    checkpoint: Any,
    config: Config,
    gate_module: torch.nn.Module,
) -> tuple[FormulaSpec, dict[str, Any]]:
    model_spec = checkpoint.model_spec
    mode = str(getattr(model_spec, "anchor_proposal_mode", None) or getattr(config, "anchor_proposal_mode", ""))
    if mode != EXPECTED_MODE:
        raise RuntimeError(f"anchor_proposal_mode 涓嶇鍚?APCF 鍗忚: {mode!r}")

    prior_margin = getattr(model_spec, "anchor_proposal_prior_margin", None)
    source = "checkpoint.model_spec.anchor_proposal_prior_margin"
    if prior_margin is None:
        prior_margin = getattr(config, "anchor_proposal_prior_margin", None)
        source = "resolved_config.anchor_proposal_prior_margin"
    if prior_margin is None:
        raise RuntimeError("缂哄皯 anchor_proposal_prior_margin锛屾棤娉曠‘瀹?prior_logit")

    residual_scale = getattr(model_spec, "residual_scale", None)
    delta_temperature = getattr(model_spec, "delta_temperature", None)
    if residual_scale is None:
        residual_scale = getattr(config, "residual_scale", None)
    if delta_temperature is None:
        delta_temperature = getattr(config, "delta_temperature", None)
    formula_source = source
    source_formula = ""
    if residual_scale is None or delta_temperature is None:
        residual_scale, delta_temperature, source_formula = _extract_gate_formula_constants(gate_module)
        formula_source = "AnchorProposalGate.forward AST"

    spec = FormulaSpec(
        prior_logit=-float(prior_margin),
        residual_scale=float(residual_scale),
        delta_temperature=float(delta_temperature),
        anchor_proposal_mode=mode,
        source=formula_source,
    )
    expected = {"prior_logit": -4.0, "residual_scale": 6.0, "delta_temperature": 0.01}
    actual = asdict(spec)
    for key, expected_value in expected.items():
        if not math.isclose(float(actual[key]), expected_value, rel_tol=0.0, abs_tol=1.0e-12):
            raise RuntimeError(
                f"褰撳墠 checkpoint 鍏紡鍙傛暟涓嶇鍚堟湰杞瘖鏂崗璁? {key}={actual[key]!r}, "
                f"expected={expected_value!r}"
            )
    return spec, {
        "model_spec": asdict(model_spec),
        "formula": asdict(spec),
        "formula_source": formula_source,
        "production_formula_source": source_formula,
    }


def _load_asset_semantics(asset_dir: Path, source_manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    manifest_path = asset_dir / "manifest.json"
    integrity_path = asset_dir / "integrity_check.json"
    asset = _load_json(manifest_path)
    integrity = _load_json(integrity_path)
    asset_raw_sha = _sha256_file(manifest_path)
    source_sha = _sha256_file(source_manifest_path)
    if integrity.get("status") != "passed":
        raise RuntimeError("APCF asset integrity_check.status 涓嶆槸 passed")
    if asset_raw_sha != EXPECTED_ASSET_RAW_SHA:
        raise RuntimeError(f"APCF asset raw manifest SHA 涓嶆纭? {asset_raw_sha}")
    if source_sha != EXPECTED_SOURCE_SHA:
        raise RuntimeError(f"source manifest SHA 涓嶆纭? {source_sha}")
    if integrity.get("asset_manifest_sha256") != asset.get("manifest_sha256"):
        raise RuntimeError('asset canonical manifest SHA mismatch')
    if int(integrity.get("sample_graph_count", -1)) != 120:
        raise RuntimeError("sample-bearing 鍥炬暟涓嶆槸 120")
    counts = integrity.get("sample_graph_counts_by_split", {})
    if counts.get("pretrain") != 96 or counts.get("frozen_diagnostic") != 24:
        raise RuntimeError(f"sample-bearing split 鍥炬暟涓嶆纭? {counts}")
    sample_counts = integrity.get("sample_counts_by_split", {})
    if sample_counts.get("frozen_diagnostic") != EXPECTED_FROZEN_CANDIDATE_COUNT:
        raise RuntimeError(f"frozen candidate 鏁伴噺涓嶆纭? {sample_counts}")
    if sample_counts.get("ppo_only") != 0:
        raise RuntimeError('ppo_only must contain zero samples')
    return asset, integrity, asset_raw_sha, source_sha


def _group_frozen_rows(asset: dict[str, Any], asset_dir: Path) -> dict[tuple[str, int, int, int, tuple[int, ...]], dict[str, Any]]:
    groups: dict[tuple[str, int, int, int, tuple[int, ...]], dict[str, Any]] = {}
    for row in asset.get("files", []):
        if row.get("split") != "frozen_diagnostic":
            continue
        csv_sha = str(row["csv_sha256"])
        anchor = tuple(sorted(int(worker) for worker in row["anchor_team"]))
        decision_count = _parse_state_seed(str(row["state_seed"]), csv_sha)
        key = (csv_sha, decision_count, int(row["task_id"]), int(row["station_id"]), anchor)
        group = groups.setdefault(
            key,
            {
                "key": key,
                "csv_sha256": csv_sha,
                "task_id": int(row["task_id"]),
                "station_id": int(row["station_id"]),
                "anchor_team": anchor,
                "obs_pt": asset_dir / str(row["obs_pt"]),
                "npz": asset_dir / str(row["npz"]),
                "candidates": [],
            },
        )
        group["candidates"].append(row)
    if len(groups) != EXPECTED_FROZEN_STATE_COUNT:
        raise RuntimeError(f"frozen 鐘舵€佺粍鏁伴噺涓嶆槸 96: {len(groups)}")
    if sum(len(group["candidates"]) for group in groups.values()) != EXPECTED_FROZEN_CANDIDATE_COUNT:
        raise RuntimeError("frozen candidate 鎬绘暟涓嶆槸 504")
    for group in groups.values():
        if not group["obs_pt"].is_file() or not group["npz"].is_file():
            raise FileNotFoundError(f"frozen candidate 鏂囦欢缂哄け: {group['obs_pt']} / {group['npz']}")
    return groups


def _clone_mask(mask: Any) -> Any:
    return mask.detach().cpu().clone() if torch.is_tensor(mask) else copy.deepcopy(mask)


def _assert_observation_equal(expected: Any, actual: Any) -> None:
    """Recursively compare the serialized HeteroData observation."""
    if hasattr(expected, "to_dict") or hasattr(actual, "to_dict"):
        if not hasattr(expected, "to_dict") or not hasattr(actual, "to_dict"):
            raise RuntimeError("asset obs.pt HeteroData structure differs from replayed observation")
        _assert_observation_equal(expected.to_dict(), actual.to_dict())
        return
    if torch.is_tensor(expected) or torch.is_tensor(actual):
        if not torch.is_tensor(expected) or not torch.is_tensor(actual):
            raise RuntimeError("asset obs.pt tensor structure differs from replayed observation")
        if not torch.equal(expected.cpu(), actual.cpu()):
            raise RuntimeError("asset obs.pt tensor differs from replayed observation")
        return
    if isinstance(expected, Mapping) or isinstance(actual, Mapping):
        if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
            raise RuntimeError("asset obs.pt mapping structure differs from replayed observation")
        if set(expected) != set(actual):
            raise RuntimeError("asset obs.pt mapping keys differ from replayed observation")
        for key in expected:
            _assert_observation_equal(expected[key], actual[key])
        return
    if isinstance(expected, (list, tuple)) or isinstance(actual, (list, tuple)):
        if not isinstance(expected, (list, tuple)) or not isinstance(actual, (list, tuple)):
            raise RuntimeError("asset obs.pt sequence structure differs from replayed observation")
        if len(expected) != len(actual):
            raise RuntimeError("asset obs.pt sequence length differs from replayed observation")
        for expected_item, actual_item in zip(expected, actual):
            _assert_observation_equal(expected_item, actual_item)
        return
    if isinstance(expected, np.ndarray) or isinstance(actual, np.ndarray):
        if not isinstance(expected, np.ndarray) or not isinstance(actual, np.ndarray):
            raise RuntimeError("asset obs.pt array structure differs from replayed observation")
        if not np.array_equal(expected, actual):
            raise RuntimeError("asset obs.pt array differs from replayed observation")
        return
    if expected != actual:
        raise RuntimeError(f"asset obs.pt scalar differs from replayed observation: {expected!r} != {actual!r}")


def _replay_frozen_states(
    *,
    csv_path: Path,
    csv_sha256: str,
    expected_groups: Mapping[tuple[str, int, int, int, tuple[int, ...]], dict[str, Any]],
    replay_config: Config,
    seed: int,
) -> dict[tuple[str, int, int, int, tuple[int, ...]], dict[str, Any]]:
    completer = EarliestFinishActionCompleter(replay_config)
    env = AirLineEnv_Graph(data_path_or_dir=str(csv_path), seed=seed)
    obs = env.reset(randomize_duration=False, randomize_workers=False, seed=seed)
    found: dict[tuple[str, int, int, int, tuple[int, ...]], dict[str, Any]] = {}
    decision_count = 0
    step = 0
    done = False
    while not done and step < MAX_EPISODE_STEPS:
        selected = _select_pair(env, obs, completer, max_candidates=MAX_CANDIDATES)
        if selected is None:
            if not env.try_wait_for_resources():
                break
            obs = env._get_observation()
            continue
        decision_count += 1
        if len(selected.candidates.teams) > 1:
            key = _make_state_key(
                csv_sha256,
                {
                    "decision_count": decision_count,
                    "task_id": selected.task_id,
                    "station_id": selected.station_id,
                    "base_team": tuple(sorted(selected.candidates.teams[0])),
                },
            )
            if key in expected_groups:
                if key in found:
                    raise RuntimeError(f"鐘舵€佸洖鏀惧彂鐜伴噸澶嶇姸鎬? {key!r}")
                masks = tuple(_clone_mask(value) for value in env.get_masks())
                stored_obs = torch.load(
                    expected_groups[key]["obs_pt"], map_location="cpu", weights_only=False
                )
                _assert_observation_equal(stored_obs, obs)
                with np.load(expected_groups[key]["npz"], allow_pickle=True) as npz_data:
                    stored_worker_mask = torch.as_tensor(np.asarray(npz_data["worker_mask"], dtype=np.bool_))
                if not torch.equal(stored_worker_mask, masks[2].to(torch.bool)):
                    raise RuntimeError(f"worker_mask 涓庣姸鎬佸洖鏀句笉涓€鑷? {key!r}")
                found[key] = {
                    "obs": copy.deepcopy(obs),
                    "masks": masks,
                    "task_id": int(selected.task_id),
                    "station_id": int(selected.station_id),
                    "anchor_team": tuple(sorted(int(worker) for worker in selected.candidates.teams[0])),
                    "decision_count": decision_count,
                    "trajectory_step": step,
                }
        obs, _reward, done, info = env.step(
            (selected.task_id, selected.station_id, list(selected.candidates.teams[0]))
        )
        if "error" in info:
            raise RuntimeError(f"anchor trajectory action rejected: {info['error']}")
        step += 1
    if set(found) != set(expected_groups):
        raise RuntimeError(
            "frozen state replay did not reach the complete target set: "
            f"expected={len(expected_groups)}, actual={len(found)}, "
            f"missing={sorted(set(expected_groups) - set(found))[:5]}, "
            f"unexpected={sorted(set(found) - set(expected_groups))[:5]}"
        )
    return found


def _build_replay_config(*, data_file: Path) -> Config:
    """Use the builder global configuration used by AirLineEnv_Graph."""
    configure_replay(data_file)
    return global_configs


def _build_config(
    *,
    experiment_path: Path,
    checkpoint: Any,
    asset_manifest: Path,
    data_file: Path,
) -> Config:
    config = Config()
    load_training_config([str(experiment_path)], target=config)
    apply_checkpoint_model_spec(config, checkpoint.model_spec)
    config.anchor_proposal_cf_manifest_path = str(asset_manifest)
    config.data_file_path = str(data_file)
    config.train_data_path_or_dir = str(data_file.parent)
    config.randomize_durations = False
    config.enable_dynamic_events = False
    config.enable_station_breakdown = False
    config.enable_material_delay = False
    config.enable_online_duration_perturb = False
    config.enable_worker_fatigue = False
    config.enable_reschedule_mode = False
    apply_initial_worker_mapping(config, data_file, explicit_fields=set())
    validate_runtime_config(config)
    return config


def _build_inference_agent(config: Config, checkpoint: Any, device: torch.device) -> PPOAgent:
    model = HBGATPN(config).to(device)
    load_policy_weights(model, checkpoint, strict=True)
    model.eval()
    agent = PPOAgent(
        model,
        lr=float(getattr(config, "lr", 1.0e-4)),
        gamma=float(getattr(config, "gamma", 0.99)),
        k_epochs=1,
        eps_clip=float(getattr(config, "eps_clip", 0.2)),
        device=device,
        batch_size=1,
        total_timesteps=0,
        config=config,
    )
    agent.policy.eval()
    return agent


def _production_gate_logit(
    model: Any,
    *,
    task_emb: torch.Tensor,
    station_emb: torch.Tensor,
    gate_features: torch.Tensor,
    hamming: torch.Tensor,
) -> torch.Tensor:
    gate_module = model.anchor_proposal_gate
    if gate_module is None or not hasattr(gate_module, "gate"):
        raise RuntimeError('production AnchorProposalGate does not expose gate submodule')
    gate_input = torch.cat(
        [task_emb.float(), station_emb.float(), gate_features.float(), hamming.float()],
        dim=-1,
    )
    with torch.autocast(device_type=task_emb.device.type, enabled=False):
        result = gate_module.gate(gate_input).float()
    return result


def _score_candidate_via_production(
    agent: PPOAgent,
    *,
    task_emb: torch.Tensor,
    station_emb: torch.Tensor,
    worker_embs: torch.Tensor,
    anchor_team: tuple[int, ...],
    candidate_team: tuple[int, ...],
    gate_features: tuple[float, ...],
) -> dict[str, float]:
    if not candidate_team or len(candidate_team) != len(anchor_team):
        raise RuntimeError("candidate team 蹇呴』涓轰笌 anchor 鍚屼汉鏁扮殑闈炵┖鍥㈤槦")
    worker_embs3 = worker_embs.unsqueeze(0) if worker_embs.ndim == 2 else worker_embs
    anchor_emb = worker_embs3[:, list(anchor_team), :].mean(dim=1)
    candidate_emb = worker_embs3[:, list(candidate_team), :].mean(dim=1)
    gate_features_tensor = torch.tensor(
        list(gate_features), dtype=torch.float32, device=worker_embs3.device
    ).reshape(1, -1)
    hamming = torch.tensor(
        [[float(len(set(candidate_team) - set(anchor_team)))]],
        dtype=torch.float32,
        device=worker_embs3.device,
    )
    branch_logits, delta_a, gate_value = agent._apcf_float32_gate_logits(
        agent.policy,
        task_emb=task_emb,
        station_emb=station_emb,
        anchor_emb=anchor_emb,
        proposal_emb=candidate_emb,
        gate_features=gate_features_tensor,
        hamming=hamming,
    )
    gate_logit = _production_gate_logit(
        agent.policy,
        task_emb=task_emb,
        station_emb=station_emb,
        gate_features=gate_features_tensor,
        hamming=hamming,
    )
    return {
        "gate_logit": _finite(gate_logit.reshape(-1)[0].item(), label="gate_logit"),
        "gate_value": _finite(gate_value.reshape(-1)[0].item(), label="gate_value"),
        "predicted_delta_A": _finite(delta_a.reshape(-1)[0].item(), label="predicted_delta_A"),
        "raw_gap": _finite(
            (branch_logits[0, 1] - branch_logits[0, 0]).item(),
            label="raw_gap",
        ),
    }


def _tensorboard_summary(run_dir: Path) -> dict[str, Any]:
    event_files = sorted(run_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"鏈壘鍒?APCF 棰勮缁?TensorBoard event: {run_dir}")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tags: dict[str, list[float]] = defaultdict(list)
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file.parent), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            tags[tag].extend(float(event.value) for event in accumulator.Scalars(tag))
    if not tags:
        raise RuntimeError("TensorBoard event 涓病鏈?scalar 鎸囨爣")
    summary: dict[str, Any] = {"event_files": [str(path) for path in event_files], "scalars": {}}
    for tag, values in sorted(tags.items()):
        if not values or not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"TensorBoard 鎸囨爣闈炴湁闄? {tag}")
        summary["scalars"][tag] = {
            "count": len(values),
            "first": values[0],
            "last": values[-1],
            "min": min(values),
            "max": max(values),
        }
    return summary


def _distribution(values: Iterable[float]) -> dict[str, float | int]:
    numbers = np.asarray([float(value) for value in values], dtype=np.float64)
    if numbers.size == 0:
        return {"count": 0, "min": None, "p10": None, "p50": None, "p90": None, "max": None}
    if not np.isfinite(numbers).all():
        raise ValueError('distribution contains non-finite values')
    return {
        "count": int(numbers.size),
        "min": float(np.min(numbers)),
        "p10": float(np.percentile(numbers, 10)),
        "p50": float(np.percentile(numbers, 50)),
        "p90": float(np.percentile(numbers, 90)),
        "max": float(np.max(numbers)),
    }


def _candidate_level_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [float(row["relative_gain"]) > 0.0 for row in rows]
    delta_predictions = [float(row["predicted_delta_A"]) > 0.0 for row in rows]
    gate_predictions = [float(row["raw_gap"]) > 0.0 for row in rows]
    return {
        "count": len(rows),
        "positive_count": sum(labels),
        "positive_rate": float(sum(labels) / len(labels)),
        "delta_sign": binary_classification_metrics(labels, delta_predictions),
        "gate_branch": binary_classification_metrics(labels, gate_predictions),
    }


def _load_checkpoint_and_validate(
    checkpoint_path: Path,
    *,
    asset_raw_sha: str,
) -> tuple[Any, str]:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model_spec = checkpoint.model_spec
    if str(getattr(model_spec, "policy_action_scope", "")) != EXPECTED_SCOPE:
        raise RuntimeError("checkpoint policy_action_scope 涓嶆槸 APCF scope")
    if str(getattr(model_spec, "anchor_proposal_mode", "")) != EXPECTED_MODE:
        raise RuntimeError("checkpoint anchor_proposal_mode 涓嶆槸 full_team_v1")
    recorded_asset_sha = str(getattr(model_spec, "anchor_proposal_cf_manifest_sha256", "") or "")
    if recorded_asset_sha != asset_raw_sha:
        raise RuntimeError(
            "checkpoint APCF asset manifest SHA 涓嶄竴鑷? "
            f"checkpoint={recorded_asset_sha}, asset={asset_raw_sha}"
        )
    return checkpoint, _sha256_file(checkpoint_path)


def _prepare_state_groups(
    *,
    asset: dict[str, Any],
    asset_dir: Path,
    source_manifest: dict[str, Any],
    source_manifest_path: Path,
) -> tuple[dict[tuple[str, int, int, int, tuple[int, ...]], dict[str, Any]], dict[str, Path]]:
    groups = _group_frozen_rows(asset, asset_dir)
    split = _split_by_sha256(source_manifest.get("files", []))
    frozen_jobs = _plan_graph_jobs(split, max_graphs=0)
    frozen_items = [item for item in split["frozen_diagnostic"]]
    if len(frozen_items) != EXPECTED_FROZEN_GRAPH_COUNT:
        raise RuntimeError(f"source frozen 鍥炬暟涓嶆槸 24: {len(frozen_items)}")
    csv_by_sha = {
        str(item["sha256"]): source_manifest_path.parent / str(item["file"])
        for item in frozen_items
    }
    del frozen_jobs
    for path in csv_by_sha.values():
        if not path.is_file():
            raise FileNotFoundError(f"source CSV 涓嶅瓨鍦? {path}")
    if set(csv_by_sha) != {key[0] for key in groups}:
        raise RuntimeError('asset frozen states do not match source frozen graph set')
    return groups, csv_by_sha


def run_diagnostic(
    *,
    checkpoint_path: Path,
    asset_dir: Path,
    source_manifest_path: Path,
    data_file: Path,
    experiment_path: Path,
    pretrain_run: Path,
    output_dir: Path,
    seed: int,
    device_name: str,
    prior_logits: tuple[float, ...],
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"璇婃柇杈撳嚭鐩綍宸插瓨鍦紝鎷掔粷澶嶇敤: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    if not output_dir.resolve().is_relative_to((PROJECT_ROOT / ".pytest_tmp").resolve()):
        raise ValueError('diagnostic output must be under project .pytest_tmp')

    asset, integrity, asset_raw_sha, source_sha = _load_asset_semantics(asset_dir, source_manifest_path)
    source_manifest = _load_json(source_manifest_path)
    groups, csv_by_sha = _prepare_state_groups(
        asset=asset,
        asset_dir=asset_dir,
        source_manifest=source_manifest,
        source_manifest_path=source_manifest_path,
    )
    checkpoint, checkpoint_sha = _load_checkpoint_and_validate(
        checkpoint_path, asset_raw_sha=asset_raw_sha
    )
    config = _build_config(
        experiment_path=experiment_path,
        checkpoint=checkpoint,
        asset_manifest=asset_dir / "manifest.json",
        data_file=data_file,
    )
    device = torch.device(
        "cuda"
        if device_name == "cuda" or (device_name == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    gate_probe = HBGATPN(config).to(device)
    load_policy_weights(gate_probe, checkpoint, strict=True)
    gate_probe.eval()
    formula_spec, formula_audit = _resolve_formula_spec(
        checkpoint, config, gate_probe.anchor_proposal_gate
    )
    del gate_probe
    agent = _build_inference_agent(config, checkpoint, device)
    replay_config = _build_replay_config(data_file=data_file)

    replayed: dict[tuple[str, int, int, int, tuple[int, ...]], dict[str, Any]] = {}
    for csv_sha, csv_path in sorted(csv_by_sha.items()):
        expected = {key: group for key, group in groups.items() if key[0] == csv_sha}
        replayed.update(
            _replay_frozen_states(
                csv_path=csv_path,
                csv_sha256=csv_sha,
                expected_groups=expected,
                replay_config=replay_config,
                seed=seed,
            )
        )
    if len(replayed) != EXPECTED_FROZEN_STATE_COUNT:
        raise RuntimeError(f"瀹為檯鍥炴斁鐘舵€佹暟涓嶆槸 96: {len(replayed)}")

    state_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    production_equivalence_checks = 0
    with torch.inference_mode():
        for key in sorted(replayed):
            state = replayed[key]
            group = groups[key]
            obs = state["obs"]
            obs_device = copy.deepcopy(obs).to(device)
            encoded, _context = agent.policy(obs_device)
            task_id = int(state["task_id"])
            station_id = int(state["station_id"])
            task_emb = encoded["task"][task_id].unsqueeze(0)
            station_emb = encoded["station"][station_id].unsqueeze(0)
            worker_embs = encoded["worker"]
            worker_mask = state["masks"][2].to(device)
            result = agent._select_anchor_proposal_team(
                agent.policy,
                obs=obs_device,
                task_id=task_id,
                station_id=station_id,
                worker_mask=worker_mask,
                task_emb=task_emb,
                station_emb=station_emb,
                worker_embs=worker_embs,
                deterministic=True,
                temperature=0.0,
                branch_floor=0.0,
            )
            if result is None:
                raise RuntimeError(f"鐢熶骇 APCF branch 鏃犳硶澶勭悊鐘舵€? {key!r}")
            _selected_team, _team_logprob, trace = result
            required_team_size = int(agent.get_task_demand(obs_device["task"].x, task_id))
            if required_team_size != len(trace.anchor_team):
                raise RuntimeError(f"required_team_size 涓庣敓浜?anchor 浜烘暟涓嶄竴鑷? {key!r}")

            state_row: dict[str, Any] = {
                "csv_sha256": key[0],
                "decision_count": key[1],
                "task_id": task_id,
                "station_id": station_id,
                "anchor_team": list(trace.anchor_team),
                "required_team_size": required_team_size,
                "proposal_available": bool(trace.proposal_available),
            }
            if trace.proposal_available:
                generated_score = _score_candidate_via_production(
                    agent,
                    task_emb=task_emb,
                    station_emb=station_emb,
                    worker_embs=worker_embs,
                    anchor_team=tuple(trace.anchor_team),
                    candidate_team=tuple(trace.proposal_team),
                    gate_features=tuple(trace.gate_features),
                )
                production_score = {
                    "gate_logit": float(torch.logit(torch.tensor(trace.gate_value).clamp(1.0e-6, 1.0 - 1.0e-6)).item()),
                    "predicted_delta_A": float(trace.predicted_delta_a),
                    "raw_gap": float(trace.raw_branch_logit_gap),
                }
                assert_candidate_scores_close(production_score, generated_score)
                production_equivalence_checks += 1
                residual_term = reconstruct_residual_term(
                    formula_spec,
                    gate_value=generated_score["gate_value"],
                    predicted_delta_a=generated_score["predicted_delta_A"],
                )
                reconstructed_gap = reconstruct_raw_gap(
                    formula_spec,
                    gate_value=generated_score["gate_value"],
                    predicted_delta_a=generated_score["predicted_delta_A"],
                )
                if not math.isclose(
                    generated_score["raw_gap"],
                    reconstructed_gap,
                    rel_tol=1.0e-5,
                    abs_tol=1.0e-6,
                ):
                    raise AssertionError(
                        "production raw gap 与 checkpoint 公式不一致: "
                        f"production={generated_score['raw_gap']}, "
                        f"reconstructed={reconstructed_gap}"
                    )
                validate_production_branch(
                    int(trace.raw_argmax_branch),
                    reconstructed_gap,
                    proposal_available=True,
                )
                state_row.update(
                    policy_state_fields(
                        proposal_available=True,
                        proposal_team=trace.proposal_team,
                        hamming_distance=int(trace.hamming_distance),
                        required_team_size=required_team_size,
                        gate_logit=generated_score["gate_logit"],
                        gate_value=generated_score["gate_value"],
                        predicted_delta_a=generated_score["predicted_delta_A"],
                        raw_branch_logit_gap=generated_score["raw_gap"],
                        production_raw_branch=int(trace.raw_argmax_branch),
                    )
                )
                state_row["diagnostic_reconstructed_gap"] = reconstructed_gap
                state_row["diagnostic_expected_branch"] = expected_raw_branch(reconstructed_gap)
                state_row["branch_match"] = True
                state_row["residual_term"] = residual_term
                state_row["raw_argmax_branch"] = int(trace.raw_argmax_branch)
                base_gate_features = tuple(trace.gate_features)
                for candidate in group["candidates"]:
                    candidate_team = _as_tuple_team(candidate["candidate_team"])
                    score = _score_candidate_via_production(
                        agent,
                        task_emb=task_emb,
                        station_emb=station_emb,
                        worker_embs=worker_embs,
                        anchor_team=tuple(trace.anchor_team),
                        candidate_team=candidate_team,
                        gate_features=base_gate_features,
                    )
                    candidate_rows.append(
                        {
                            "csv_sha256": key[0],
                            "decision_count": key[1],
                            "task_id": task_id,
                            "station_id": station_id,
                            "candidate_team": list(candidate_team),
                            "relative_gain": float(candidate["relative_gain"]),
                            "predicted_delta_A": score["predicted_delta_A"],
                            "raw_gap": score["raw_gap"],
                            "gate_logit": score["gate_logit"],
                            "gate_value": score["gate_value"],
                        }
                    )
            else:
                state_row.update(
                    policy_state_fields(
                        proposal_available=False,
                        proposal_team=None,
                        hamming_distance=None,
                        required_team_size=required_team_size,
                        gate_logit=None,
                        gate_value=None,
                        predicted_delta_a=None,
                        raw_branch_logit_gap=None,
                        production_raw_branch=None,
                    )
                )
                state_row.update(
                    {
                        "diagnostic_reconstructed_gap": None,
                        "diagnostic_expected_branch": None,
                        "branch_match": None,
                        "residual_term": None,
                        "raw_argmax_branch": None,
                    }
                )
            state_rows.append(state_row)

    if len(candidate_rows) != EXPECTED_FROZEN_CANDIDATE_COUNT:
        raise RuntimeError(f"candidate-level 璇勫垎鏁伴噺涓嶆槸 504: {len(candidate_rows)}")

    available_rows = [row for row in state_rows if row["proposal_available"]]
    default_gaps = [float(row["raw_branch_logit_gap"]) for row in available_rows]
    gate_values = [float(row["gate_value"]) for row in available_rows]
    delta_values = [float(row["predicted_delta_A"]) for row in available_rows]
    hamming_values = [float(row["hamming_distance"]) for row in available_rows]
    normalized_hamming_values = [float(row["normalized_hamming_distance"]) for row in available_rows]
    sweep_rows: list[dict[str, Any]] = []
    for prior_logit in prior_logits:
        gaps = [sweep_gap(float(prior_logit), float(row["residual_term"])) for row in available_rows]
        sweep_rows.append(
            {
                "prior_logit": float(prior_logit),
                "proposal_selected_count": int(sum(gap > 0.0 for gap in gaps)),
                "proposal_selected_rate": float(sum(gap > 0.0 for gap in gaps) / len(gaps)) if gaps else 0.0,
                "gap_distribution": _distribution(gaps),
            }
        )

    default_selection_count = sum(int(row["production_raw_branch"]) for row in available_rows)
    if default_selection_count > 0:
        decision = "ready_for_online_validation"
    elif gate_values and max(gate_values) <= 2.0 / 3.0:
        decision = "gate_locked"
    elif delta_values and sum(value > 0.0 for value in delta_values) == 0:
        decision = "value_head_collapse"
    elif any(int(row["proposal_selected_count"]) > 0 for row in sweep_rows[1:]):
        decision = "margin_sensitive"
    else:
        decision = "insufficient_evidence"

    tensorboard = _tensorboard_summary(pretrain_run)
    report = {
        "status": "passed",
        "decision": decision,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "model_key_count": len(checkpoint.state_dict),
        },
        "asset": {
            "manifest_raw_sha256": asset_raw_sha,
            "manifest_canonical_sha256": asset.get("manifest_sha256"),
            "source_manifest_sha256": source_sha,
            "integrity_status": integrity.get("status"),
        },
        "state_summary": {
            "frozen_graph_count": len(csv_by_sha),
            "frozen_state_count": len(state_rows),
            "proposal_available_count": len(available_rows),
            "proposal_available_rate": float(len(available_rows) / len(state_rows)),
            "default_raw_proposal_selected_count": default_selection_count,
            "default_raw_proposal_selected_rate": float(default_selection_count / len(available_rows)) if available_rows else 0.0,
            "production_candidate_equivalence_checks": production_equivalence_checks,
        },
        "policy_level": {
            "gate_value": _distribution(gate_values),
            "predicted_delta_A": _distribution(delta_values),
            "raw_branch_logit_gap": _distribution(default_gaps),
            "hamming_distance": _distribution(hamming_values),
            "normalized_hamming_distance": _distribution(normalized_hamming_values),
            "gate_value_gt_two_thirds_count": sum(value > 2.0 / 3.0 for value in gate_values),
            "predicted_delta_A_positive_count": sum(value > 0.0 for value in delta_values),
            "raw_gap_positive_count": sum(value > 0.0 for value in default_gaps),
            "required_team_size_ge_2_count": sum(int(row["required_team_size"]) >= 2 for row in available_rows),
            "single_worker_edit_rate_ge_2": float(
                sum(int(row["hamming_distance"]) == 1 for row in available_rows if int(row["required_team_size"]) >= 2)
                / max(1, sum(int(row["required_team_size"]) >= 2 for row in available_rows))
            ),
        },
        "candidate_level": _candidate_level_metrics(candidate_rows),
        "prior_sweep": sweep_rows,
        "model_formula": formula_audit,
        "checkpoint_model_spec": asdict(checkpoint.model_spec),
        "proposal_available_masking_rule": "proposal_available=false 鏃?proposal 鍒嗘敮瀛楁鍏ㄩ儴涓?JSON null锛屼笉璁＄畻 gap",
        "tie_break_rule": "temperature=0 涓?raw gap <= 0 鏃堕€夋嫨 anchor",
        "tensorboard": tensorboard,
        "errors": [],
    }

    (output_dir / "gate_calibration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "gate_calibration_by_state.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        if state_rows:
            writer = csv.DictWriter(handle, fieldnames=list(state_rows[0]))
            writer.writeheader()
            writer.writerows(state_rows)
    with (output_dir / "prior_sweep_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prior_logit", "proposal_selected_count", "proposal_selected_rate", "gap_distribution"])
        writer.writeheader()
        for row in sweep_rows:
            writer.writerow({**row, "gap_distribution": json.dumps(row["gap_distribution"], ensure_ascii=False)})
    run_manifest = {
        "script": "diagnose_apcf_gate_calibration.py",
        "seed": seed,
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "asset_manifest_raw_sha256": asset_raw_sha,
        "asset_manifest_canonical_sha256": asset.get("manifest_sha256"),
        "source_manifest_sha256": source_sha,
        "model_formula": formula_audit,
        "checkpoint_model_spec": asdict(checkpoint.model_spec),
        "proposal_available_masking_rule": report["proposal_available_masking_rule"],
        "tie_break_rule": report["tie_break_rule"],
        "production_scorer_used": "PPOAgent._apcf_float32_gate_logits + AnchorProposalGate.gate",
        "production_branch_selector_used": "PPOAgent._select_anchor_proposal_team",
        "prior_logits": list(prior_logits),
        "state_count": len(state_rows),
        "candidate_count": len(candidate_rows),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# APCF Gate Calibration\n\n"
        "鏈洰褰曚负鍙 gate 鏍囧畾銆佺敓浜ц矾寰勫悓鏋勫拰 prior sweep 璇婃柇浜х墿锛涙湭鎵ц PPO 鎴栧弽鍚戜紶鎾€俓n",
        encoding="utf-8",
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="APCF gate 鏍囧畾涓?prior sweep 璇婃柇")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--asset-dir", default="data/initial_anchor_proposal_cf_v1")
    parser.add_argument("--source-manifest", default="data/scale_400_800_datasets/manifest_ctg_160_explicit_fiveskill_v1.json")
    parser.add_argument("--data-file", default="data/680.csv")
    parser.add_argument("--experiment", default="conf/experiment/initial_anchor_proposal_cf_v1.yaml")
    parser.add_argument("--pretrain-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--prior-logits", nargs="+", type=float, default=list(DEFAULT_PRIOR_LOGITS))
    return parser.parse_args(argv)


def _workspace_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        pl.seed_everything(int(args.seed), workers=True)
        report = run_diagnostic(
            checkpoint_path=_workspace_path(args.checkpoint),
            asset_dir=_workspace_path(args.asset_dir),
            source_manifest_path=_workspace_path(args.source_manifest),
            data_file=_workspace_path(args.data_file),
            experiment_path=_workspace_path(args.experiment),
            pretrain_run=_workspace_path(args.pretrain_run),
            output_dir=_workspace_path(args.output_dir),
            seed=int(args.seed),
            device_name=str(args.device),
            prior_logits=tuple(float(value) for value in args.prior_logits),
        )
        print(
            f"[apcf-calibration] status={report['status']} decision={report['decision']} "
            f"states={report['state_summary']['frozen_state_count']} "
            f"candidates={report['candidate_level']['count']}",
            flush=True,
        )
        return 0
    except Exception as exc:
        print(f"[apcf-calibration] failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

