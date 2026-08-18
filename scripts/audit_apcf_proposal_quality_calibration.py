# -*- coding: utf-8 -*-
"""APCF 自由 Proposal 质量与评分校准只读审计。"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import inspect
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnose_apcf_gate_calibration import (  # noqa: E402
    EXPECTED_ASSET_RAW_SHA,
    EXPECTED_FROZEN_CANDIDATE_COUNT,
    EXPECTED_FROZEN_GRAPH_COUNT,
    EXPECTED_FROZEN_STATE_COUNT,
    EXPECTED_SOURCE_SHA,
    _build_config,
    _build_inference_agent,
    _build_replay_config,
    _load_asset_semantics,
    _load_checkpoint_and_validate,
    _prepare_state_groups,
    _resolve_formula_spec,
    _sha256_file,
)
from scripts.diagnose_apcf_prior_sweep_online_gain import (  # noqa: E402
    CONTINUATION_PRIOR_LOGIT,
    CONTINUATION_TEMPERATURE,
    MAX_EPISODE_STEPS,
    _finite,
    _git_commit,
    _load_json,
    _replay_frozen_state_snapshots,
    _run_forced_policy_episode,
    _select_state_for_prior,
    _state_dict_sha256,
    expected_raw_branch,
)

EXPECTED_PRIOR_COUNTS = {-4.0: 0, -2.0: 9, -1.0: 33, 0.0: 63}
EXPECTED_ASSET_CANONICAL_SHA = "7fb6f695e73c5658ef0d5c17b96f080660da42d9bbb6168cf2e4511d7abc96a3"
EVIDENCE_GRAPH_THRESHOLD = 8


def _parse_team(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        value = json.loads(value)
    if value is None:
        raise ValueError("team 不能为空")
    return tuple(sorted(int(worker) for worker in value))


def make_full_state_key(
    *,
    csv_sha256: str,
    decision_count: int,
    task_id: int,
    station_id: int,
    anchor_team: Iterable[int],
) -> tuple[str, int, int, int, tuple[int, ...]]:
    return (
        str(csv_sha256),
        int(decision_count),
        int(task_id),
        int(station_id),
        _parse_team(anchor_team),
    )


def cache_key_anchor(
    checkpoint_sha256: str,
    state_key: tuple[str, int, int, int, tuple[int, ...]],
) -> tuple[Any, ...]:
    return (str(checkpoint_sha256), state_key, CONTINUATION_PRIOR_LOGIT, CONTINUATION_TEMPERATURE)


def cache_key_proposal(
    checkpoint_sha256: str,
    state_key: tuple[str, int, int, int, tuple[int, ...]],
    proposal_team: Iterable[int],
) -> tuple[Any, ...]:
    return (
        str(checkpoint_sha256),
        state_key,
        _parse_team(proposal_team),
        CONTINUATION_PRIOR_LOGIT,
        CONTINUATION_TEMPERATURE,
    )


def _finite_or_none(value: Any, *, name: str) -> float | None:
    if value in (None, "", "null"):
        return None
    return _finite(value, name=name)


def _approx_equal(left: Any, right: Any, *, name: str) -> None:
    if not math.isclose(float(left), float(right), rel_tol=1.0e-5, abs_tol=1.0e-5):
        raise ValueError(f"{name} 不一致: {left!r} != {right!r}")


def _state_key_from_row(row: Mapping[str, Any]) -> tuple[str, int, int, int, tuple[int, ...]]:
    return make_full_state_key(
        csv_sha256=str(row["csv_sha256"]),
        decision_count=int(row["decision_count"]),
        task_id=int(row["task_id"]),
        station_id=int(row["station_id"]),
        anchor_team=_parse_team(row["anchor_team"]),
    )


def _protocol_contract_source() -> dict[str, bool]:
    source = inspect.getsource(_run_forced_policy_episode) + inspect.getsource(_select_state_for_prior)
    return {
        "deterministic": "deterministic=True" in source,
        "temperature_zero": "temperature=CONTINUATION_TEMPERATURE" in source,
        "branch_floor_zero": "branch_floor=0.0" in source,
        "eval_mode": "is_eval=True" in source,
        "compute_value_false": "compute_value=False" in source,
    }


def _validate_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    source_contract = _protocol_contract_source()
    for key, expected in (
        ("continuation_prior_logit", CONTINUATION_PRIOR_LOGIT),
        ("temperature", CONTINUATION_TEMPERATURE),
    ):
        if key in protocol:
            _approx_equal(protocol[key], expected, name=key)
    if protocol.get("branch_floor", 0.0) != 0.0:
        raise ValueError("branch_floor 必须为 0")
    if protocol.get("deterministic", True) is not True:
        raise ValueError("deterministic 必须为 True")
    if not all(source_contract.values()):
        raise ValueError(f"生产 continuation 协议不完整: {source_contract}")
    return {
        "continuation_prior_logit": CONTINUATION_PRIOR_LOGIT,
        "temperature": CONTINUATION_TEMPERATURE,
        "deterministic": True,
        "branch_floor": 0.0,
        "is_eval": True,
        "compute_value": False,
        "source_contract": source_contract,
    }


def validate_reused_online_row(
    row: Mapping[str, Any],
    *,
    expected_state_key: tuple[str, int, int, int, tuple[int, ...]],
    expected_anchor_team: Iterable[int],
    expected_proposal_team: Iterable[int],
    expected_checkpoint_sha256: str,
    online_protocol: Mapping[str, Any],
    expected_asset_sha256: Mapping[str, str],
    report_asset_sha256: Mapping[str, str],
) -> None:
    if _state_key_from_row(row) != expected_state_key:
        raise ValueError("复用结果完整状态键不一致")
    if _parse_team(row["anchor_team"]) != _parse_team(expected_anchor_team):
        raise ValueError("复用结果 anchor_team 不一致")
    if _parse_team(row["proposal_team"]) != _parse_team(expected_proposal_team):
        raise ValueError("复用结果 proposal_team 不一致")
    if str(row.get("proposal_available")).lower() != "true":
        raise ValueError("复用结果 proposal 不可用")
    if str(row.get("selected")).lower() != "true":
        raise ValueError("复用结果必须来自已采用 proposal")
    if str(row.get("anchor_done")).lower() != "true" or str(row.get("proposal_done")).lower() != "true":
        raise ValueError("复用结果 episode 未完成")
    if not str(row.get("online_cache_id", "")):
        raise ValueError("复用结果缺少 online_cache_id")
    for field in ("anchor_makespan", "proposal_makespan", "relative_gain"):
        _finite(row[field], name=field)
    protocol = _validate_protocol(online_protocol)
    if protocol["continuation_prior_logit"] != CONTINUATION_PRIOR_LOGIT:
        raise ValueError("复用结果 continuation prior 不一致")
    if dict(report_asset_sha256) != dict(expected_asset_sha256):
        raise ValueError("复用结果资产 SHA 不一致")
    if str(expected_checkpoint_sha256) == "":
        raise ValueError("checkpoint SHA 不能为空")


def select_missing_available_states(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("proposal_available")).lower() == "true"
        and row.get("online_cache_id") in (None, "")
    ]


def validate_online_outcome(outcome: Mapping[str, Any]) -> None:
    if outcome.get("done") is not True:
        raise ValueError("online episode 必须 done=True")
    steps = int(outcome.get("steps", 0))
    if steps <= 0 or steps > MAX_EPISODE_STEPS:
        raise ValueError("online episode steps 非法")
    makespan = _finite(outcome.get("makespan"), name="makespan")
    if makespan <= 0.0:
        raise ValueError("makespan 必须为正")


def relative_gain(anchor_makespan: float, proposal_makespan: float) -> float:
    anchor = _finite(anchor_makespan, name="anchor_makespan")
    proposal = _finite(proposal_makespan, name="proposal_makespan")
    if anchor <= 0.0 or proposal <= 0.0:
        raise ValueError("makespan 必须为正")
    return _finite((anchor - proposal) / max(anchor, 1.0e-9), name="relative_gain")


def bootstrap_graph_means(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_reps: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    if bootstrap_reps < 1:
        raise ValueError("bootstrap_reps 必须为正")
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["graph_id"])].append(_finite(row["relative_gain"], name="relative_gain"))
    graph_means = {key: float(np.mean(values)) for key, values in sorted(grouped.items())}
    if not graph_means:
        return {
            "graph_count": 0,
            "state_count": 0,
            "point_estimate": None,
            "ci_low": None,
            "ci_high": None,
            "bootstrap_reps": int(bootstrap_reps),
            "bootstrap_seed": int(seed),
        }
    values = np.asarray(list(graph_means.values()), dtype=np.float64)
    sampled = np.random.default_rng(int(seed)).choice(
        values, size=(int(bootstrap_reps), values.size), replace=True
    ).mean(axis=1)
    return {
        "graph_count": int(values.size),
        "state_count": int(sum(len(items) for items in grouped.values())),
        "point_estimate": float(values.mean()),
        "ci_low": float(np.percentile(sampled, 2.5)),
        "ci_high": float(np.percentile(sampled, 97.5)),
        "bootstrap_reps": int(bootstrap_reps),
        "bootstrap_seed": int(seed),
    }


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    sorted_values = array[order]
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman_or_null(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    if len(x) != len(y):
        raise ValueError("Spearman 输入长度不一致")
    if len(x) < 2:
        return {"value": None, "reason": "insufficient_samples", "count": len(x)}
    left = np.asarray([_finite(value, name="spearman_x") for value in x], dtype=np.float64)
    right = np.asarray([_finite(value, name="spearman_y") for value in y], dtype=np.float64)
    left_rank = _rankdata(left)
    right_rank = _rankdata(right)
    if np.ptp(left_rank) == 0.0 or np.ptp(right_rank) == 0.0:
        return {"value": None, "reason": "constant_input", "count": len(x)}
    value = float(np.corrcoef(left_rank, right_rank)[0, 1])
    return {"value": value if math.isfinite(value) else None, "reason": None, "count": len(x)}


def binary_metrics(actual: Sequence[bool], predicted: Sequence[bool]) -> dict[str, Any]:
    if len(actual) != len(predicted):
        raise ValueError("分类输入长度不一致")
    tp = sum(bool(a) and bool(p) for a, p in zip(actual, predicted))
    fp = sum(not bool(a) and bool(p) for a, p in zip(actual, predicted))
    fn = sum(bool(a) and not bool(p) for a, p in zip(actual, predicted))
    support = tp + fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / support if support else None
    f1 = 2.0 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "support": support}


def validate_layer_separation(*, candidate_count: int, policy_count: int) -> bool:
    if int(candidate_count) != EXPECTED_FROZEN_CANDIDATE_COUNT:
        raise ValueError("candidate-level 必须为 504 条 frozen 候选")
    if int(policy_count) > EXPECTED_FROZEN_STATE_COUNT:
        raise ValueError("policy-level 不得超过 96 条自由 proposal")
    return True


def ensure_output_dir(output_dir: Path, *, pytest_root: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {output_dir}")
    resolved_root = pytest_root.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if not output_dir.resolve().parent.is_relative_to(resolved_root):
        raise ValueError("输出目录必须位于 .pytest_tmp")
    output_dir.mkdir(parents=False, exist_ok=False)
    return output_dir

def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _asset_sha_map(asset: Mapping[str, Any]) -> dict[str, str]:
    return {
        "raw": str(asset["manifest_raw_sha256"]),
        "canonical": str(asset["manifest_canonical_sha256"]),
        "source": str(asset["source_manifest_sha256"]),
    }


def _validate_online_inputs(
    *,
    checkpoint_sha256: str,
    asset_sha256: Mapping[str, str],
    online_report: Mapping[str, Any],
    online_integrity: Mapping[str, Any],
) -> dict[str, Any]:
    if online_report.get("status") != "passed" or online_integrity.get("status") != "passed":
        raise ValueError("既有 prior online 结果未通过完整性验收")
    if str(online_report["checkpoint"]["sha256"]) != checkpoint_sha256:
        raise ValueError("既有 online checkpoint SHA 不一致")
    report_asset = _asset_sha_map(online_report["asset"])
    if report_asset != dict(asset_sha256):
        raise ValueError("既有 online 三类资产 SHA 不一致")
    if online_integrity.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("既有 online integrity checkpoint SHA 不一致")
    run_manifest = online_report.get("run_manifest", {})
    if run_manifest.get("model_state_hash_equal") is not True:
        raise ValueError("既有 online 模型参数哈希未通过")
    protocol = _validate_protocol(online_report.get("protocol", {}))
    return {"report_asset_sha256": report_asset, "protocol": protocol}


def _validate_calibration_row(
    row: Mapping[str, Any],
    calibration_row: Mapping[str, Any],
) -> None:
    if _state_key_from_row(row) != _state_key_from_row(calibration_row):
        raise ValueError("policy row 与 calibration 状态键不一致")
    expected_proposal = _parse_team(calibration_row["proposal_team"])
    if _parse_team(row["anchor_team"]) != _parse_team(calibration_row["anchor_team"]):
        raise ValueError("policy row 与 calibration anchor_team 不一致")
    if str(row["proposal_available"]).lower() != str(calibration_row["proposal_available"]).lower():
        raise ValueError("proposal_available 不一致")
    if str(row["proposal_available"]).lower() == "true":
        if _parse_team(row["proposal_team"]) != expected_proposal:
            raise ValueError("policy row 与 calibration proposal_team 不一致")
        _approx_equal(row["predicted_delta_A"], calibration_row["predicted_delta_A"], name="predicted_delta_A")
        _approx_equal(row["gate_value"], calibration_row["gate_value"], name="gate_value")
        _approx_equal(row["raw_gap"], calibration_row["residual_term"], name="prior=0 raw_gap")
        if int(row["hamming_distance"]) != int(calibration_row["hamming_distance"]):
            raise ValueError("hamming_distance 不一致")
        if int(row["raw_branch"]) != expected_raw_branch(float(row["raw_gap"])):
            raise ValueError("raw_branch 与 raw_gap 不一致")


def _validate_online_rows(
    *,
    online_rows: Sequence[Mapping[str, Any]],
    prior_zero_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Mapping[Any, Mapping[str, Any]],
    checkpoint_sha256: str,
    asset_sha256: Mapping[str, str],
    online_protocol: Mapping[str, Any],
) -> tuple[dict[Any, Mapping[str, Any]], dict[Any, Mapping[str, Any]]]:
    by_key: dict[Any, Mapping[str, Any]] = {}
    for row in prior_zero_rows:
        key = _state_key_from_row(row)
        if key in by_key:
            raise ValueError(f"prior=0 状态键重复: {key!r}")
        if key not in calibration_rows:
            raise ValueError(f"prior=0 状态不在 calibration: {key!r}")
        _validate_calibration_row(row, calibration_rows[key])
        by_key[key] = row
    if len(by_key) != EXPECTED_FROZEN_STATE_COUNT:
        raise ValueError(f"prior=0 状态数不是 96: {len(by_key)}")

    reused_by_key: dict[Any, Mapping[str, Any]] = {}
    for row in online_rows:
        if str(row.get("selected")).lower() != "true":
            continue
        key = _state_key_from_row(row)
        if key not in calibration_rows:
            raise ValueError(f"复用 row 状态不在 calibration: {key!r}")
        _validate_calibration_row(by_key[key], calibration_rows[key])
        validate_reused_online_row(
            row,
            expected_state_key=key,
            expected_anchor_team=_parse_team(calibration_rows[key]["anchor_team"]),
            expected_proposal_team=_parse_team(calibration_rows[key]["proposal_team"]),
            expected_checkpoint_sha256=checkpoint_sha256,
            online_protocol=online_protocol,
            expected_asset_sha256=asset_sha256,
            report_asset_sha256=asset_sha256,
        )
        previous = reused_by_key.get(key)
        if previous is not None:
            if str(previous.get("online_cache_id")) != str(row.get("online_cache_id")):
                raise ValueError("同一状态的复用 cache_id 不一致")
        reused_by_key[key] = row
    return by_key, reused_by_key


def _describe(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    array = np.asarray([_finite(value, name="metric") for value in values], dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10.0)),
        "p90": float(np.percentile(array, 90.0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _policy_quality_stats(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    gains = [_finite(row["relative_gain"], name="relative_gain") for row in rows]
    graph_groups: dict[str, list[float]] = defaultdict(list)
    for row, gain in zip(rows, gains):
        graph_groups[str(row["graph_id"])].append(gain)
    bootstrap = bootstrap_graph_means(
        [{"graph_id": graph_id, "relative_gain": gain} for graph_id, values in graph_groups.items() for gain in values],
        bootstrap_reps=bootstrap_reps,
        seed=bootstrap_seed,
    )
    positive_graph_count = sum(any(value > 0.0 for value in values) for values in graph_groups.values())
    return {
        "sample_count": len(rows),
        "graph_count": len(graph_groups),
        "gain": _describe(gains),
        "positive_state_count": sum(value > 0.0 for value in gains),
        "positive_state_rate": (sum(value > 0.0 for value in gains) / len(gains)) if gains else None,
        "positive_graph_count": positive_graph_count,
        "positive_graph_rate": (positive_graph_count / len(graph_groups)) if graph_groups else None,
        "bootstrap_graph_means": bootstrap,
    }


def _five_quantile_bins(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    order = sorted(range(len(rows)), key=lambda index: float(rows[index]["predicted_delta_A"]))
    bins: list[dict[str, Any]] = []
    for bin_index, indices in enumerate(np.array_split(np.asarray(order, dtype=np.int64), 5), start=1):
        if len(indices) == 0:
            continue
        selected = [rows[int(index)] for index in indices]
        predicted = [float(row["predicted_delta_A"]) for row in selected]
        gains = [float(row["relative_gain"]) for row in selected]
        bins.append(
            {
                "bin": bin_index,
                "count": len(selected),
                "predicted_delta_A_min": min(predicted),
                "predicted_delta_A_max": max(predicted),
                "actual_mean_relative_gain": float(np.mean(gains)),
                "actual_positive_rate": float(np.mean([gain > 0.0 for gain in gains])),
            }
        )
    return bins


def _prior_selection_quality(
    *,
    all_policy_rows: Sequence[Mapping[str, Any]],
    online_rows: Sequence[Mapping[str, Any]],
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    gain_by_key = {
        _state_key_from_row(row): row["relative_gain"]
        for row in all_policy_rows
    }
    full_stats = _policy_quality_stats(
        all_policy_rows,
        bootstrap_reps=bootstrap_reps,
        bootstrap_seed=bootstrap_seed,
    )
    result: list[dict[str, Any]] = []
    for prior in (-4.0, -2.0, -1.0, 0.0):
        selected: list[dict[str, Any]] = []
        for row in online_rows:
            if float(row["prior_logit"]) != prior or str(row["selected"]).lower() != "true":
                continue
            key = _state_key_from_row(row)
            if key not in gain_by_key:
                raise ValueError("prior selection 引用了不存在的 policy gain")
            selected.append(
                {
                    "graph_id": row["graph_id"],
                    "relative_gain": gain_by_key[key],
                }
            )
        stats = _policy_quality_stats(
            selected,
            bootstrap_reps=bootstrap_reps,
            bootstrap_seed=bootstrap_seed,
        )
        selected_mean = stats["gain"]["mean"]
        selected_median = stats["gain"]["median"]
        result.append(
            {
                "prior_logit": prior,
                "selected_state_count": stats["sample_count"],
                "selected_graph_count": stats["graph_count"],
                "selected_gain_mean": selected_mean,
                "selected_gain_median": selected_median,
                "positive_state_rate": stats["positive_state_rate"],
                "bootstrap_ci_low": stats["bootstrap_graph_means"]["ci_low"],
                "bootstrap_ci_high": stats["bootstrap_graph_means"]["ci_high"],
                "bootstrap_point_estimate": stats["bootstrap_graph_means"]["point_estimate"],
                "bootstrap_reps": bootstrap_reps,
                "bootstrap_seed": bootstrap_seed,
                "difference_vs_all_mean": (selected_mean - full_stats["gain"]["mean"]) if selected_mean is not None else None,
                "difference_vs_all_median": (selected_median - full_stats["gain"]["median"]) if selected_median is not None else None,
                "admission": (
                    "not_selected" if stats["sample_count"] == 0
                    else "insufficient_evidence" if stats["graph_count"] < EVIDENCE_GRAPH_THRESHOLD
                    else "reported"
                ),
            }
        )
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
def _build_policy_row(
    *,
    row: Mapping[str, Any],
    calibration_row: Mapping[str, Any],
    outcome: Mapping[str, Any],
    source: str,
    selected_by_prior: Mapping[float, bool],
) -> dict[str, Any]:
    anchor_makespan = _finite(outcome.get("makespan_anchor"), name="anchor_makespan")
    proposal_makespan = _finite(outcome.get("makespan_proposal"), name="proposal_makespan")
    if anchor_makespan <= 0.0 or proposal_makespan <= 0.0:
        raise ValueError("双 episode makespan 必须为正")
    if int(outcome.get("anchor_steps", 0)) <= 0 or int(outcome.get("proposal_steps", 0)) <= 0:
        raise ValueError("双 episode steps 非法")
    required_team_size = int(calibration_row["required_team_size"])
    hamming_distance = int(calibration_row["hamming_distance"])
    proposal_team = _parse_team(row["proposal_team"])
    anchor_team = _parse_team(row["anchor_team"])
    if proposal_team == anchor_team or len(proposal_team) != required_team_size:
        raise ValueError("自由 proposal 团队结构非法")
    return {
        "graph_id": str(row["graph_id"]),
        "csv_sha256": str(row["csv_sha256"]),
        "decision_count": int(row["decision_count"]),
        "task_id": int(row["task_id"]),
        "station_id": int(row["station_id"]),
        "anchor_team": list(anchor_team),
        "proposal_team": list(proposal_team),
        "required_team_size": required_team_size,
        "hamming_distance": hamming_distance,
        "normalized_hamming_distance": hamming_distance / required_team_size,
        "proposal_available": True,
        "predicted_delta_A": _finite(row["predicted_delta_A"], name="predicted_delta_A"),
        "gate_value": _finite(row["gate_value"], name="gate_value"),
        "residual_term": _finite(calibration_row["residual_term"], name="residual_term"),
        "raw_gap_prior_minus4": _finite(calibration_row["raw_branch_logit_gap"], name="raw_gap_prior_minus4"),
        "raw_gap_prior_zero": _finite(row["raw_gap"], name="raw_gap_prior_zero"),
        "raw_branch_prior_zero": int(row["raw_branch"]),
        "relative_gain": relative_gain(outcome["makespan_anchor"], outcome["makespan_proposal"]),
        "anchor_makespan": _finite(outcome["makespan_anchor"], name="anchor_makespan"),
        "proposal_makespan": _finite(outcome["makespan_proposal"], name="proposal_makespan"),
        "anchor_done": True,
        "proposal_done": True,
        "anchor_steps": int(outcome["anchor_steps"]),
        "proposal_steps": int(outcome["proposal_steps"]),
        "source": source,
        "selected_by_prior_minus4": bool(selected_by_prior.get(-4.0, False)),
        "selected_by_prior_minus2": bool(selected_by_prior.get(-2.0, False)),
        "selected_by_prior_minus1": bool(selected_by_prior.get(-1.0, False)),
        "selected_by_prior_zero": bool(selected_by_prior.get(0.0, False)),
    }


def _outcome_from_online_row(row: Mapping[str, Any]) -> dict[str, Any]:
    outcome = {
        "makespan_anchor": _finite(row["anchor_makespan"], name="anchor_makespan"),
        "makespan_proposal": _finite(row["proposal_makespan"], name="proposal_makespan"),
        "anchor_done": True,
        "proposal_done": True,
        "anchor_steps": int(row["anchor_steps"]),
        "proposal_steps": int(row["proposal_steps"]),
    }
    if outcome["anchor_steps"] <= 0 or outcome["proposal_steps"] <= 0:
        raise ValueError("复用 episode steps 非法")
    return outcome


def _metric_rows_for_prior(
    *,
    online_rows: Sequence[Mapping[str, Any]],
    policy_by_key: Mapping[Any, Mapping[str, Any]],
    prior: float,
) -> list[dict[str, Any]]:
    result = []
    for row in online_rows:
        if float(row["prior_logit"]) != float(prior) or str(row["selected"]).lower() != "true":
            continue
        key = _state_key_from_row(row)
        if key not in policy_by_key:
            raise ValueError("prior 选择状态缺少全体 policy gain")
        result.append(
            {
                "graph_id": row["graph_id"],
                "relative_gain": policy_by_key[key]["relative_gain"],
            }
        )
    return result


def _candidate_policy_summary(
    candidate_level: Mapping[str, Any],
    policy_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    validate_layer_separation(
        candidate_count=int(candidate_level["count"]),
        policy_count=len(policy_rows),
    )
    policy_metrics = binary_metrics(
        [float(row["relative_gain"]) > 0.0 for row in policy_rows],
        [float(row["predicted_delta_A"]) > 0.0 for row in policy_rows],
    )
    candidate_delta = candidate_level["delta_sign"]
    candidate_gate = candidate_level["gate_branch"]
    common = {
        "positive_rate": candidate_level["positive_rate"],
        "tp": None,
        "fp": None,
        "fn": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "mean_relative_gain": None,
    }
    rows = []
    for metric_name, metric in (("delta_sign", candidate_delta), ("gate_branch", candidate_gate)):
        rows.append(
            {
                "layer": "candidate_level",
                "metric": metric_name,
                "count": candidate_level["count"],
                "positive_count": candidate_level["positive_count"],
                "positive_rate": candidate_level["positive_rate"],
                "tp": metric["tp"],
                "fp": metric["fp"],
                "fn": metric["fn"],
                "precision": metric["precision"],
                "recall": metric["recall"],
                "f1": metric["f1"],
                "mean_relative_gain": None,
            }
        )
    rows.append(
        {
            "layer": "policy_level",
            "metric": "predicted_delta_A_positive",
            "count": len(policy_rows),
            "positive_count": sum(float(row["relative_gain"]) > 0.0 for row in policy_rows),
            "positive_rate": (sum(float(row["relative_gain"]) > 0.0 for row in policy_rows) / len(policy_rows)) if policy_rows else None,
            "tp": policy_metrics["tp"],
            "fp": policy_metrics["fp"],
            "fn": policy_metrics["fn"],
            "precision": policy_metrics["precision"],
            "recall": policy_metrics["recall"],
            "f1": policy_metrics["f1"],
            "mean_relative_gain": float(np.mean([row["relative_gain"] for row in policy_rows])) if policy_rows else None,
        }
    )
    return rows


def _policy_graph_rows(policy_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in policy_rows:
        grouped[str(row["graph_id"])].append(row)
    result = []
    for graph_id, rows in sorted(grouped.items()):
        gains = [float(row["relative_gain"]) for row in rows]
        result.append(
            {
                "graph_id": graph_id,
                "state_count": len(rows),
                "mean_relative_gain": float(np.mean(gains)),
                "median_relative_gain": float(np.median(gains)),
                "positive_state_count": sum(value > 0.0 for value in gains),
                "positive_state_rate": float(np.mean([value > 0.0 for value in gains])),
                "mean_predicted_delta_A": float(np.mean([float(row["predicted_delta_A"]) for row in rows])),
                "mean_raw_gap_prior_minus4": float(np.mean([float(row["raw_gap_prior_minus4"]) for row in rows])),
            }
        )
    return result


def _build_decision(
    *,
    full_stats: Mapping[str, Any],
    calibration: Mapping[str, Any],
    prior_quality: Sequence[Mapping[str, Any]],
    missing_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    bootstrap = full_stats["bootstrap_graph_means"]
    if int(full_stats["graph_count"]) < EVIDENCE_GRAPH_THRESHOLD:
        return {
            "status": "insufficient_evidence",
            "reason": "全体可用 proposal 图数不足 8",
            "next_action": "扩大独立冻结诊断图，不运行 PPO",
        }
    predicted_spearman = calibration["spearman_predicted_delta_A_vs_gain"]["value"]
    raw_spearman = calibration["spearman_raw_gap_vs_gain"]["value"]
    missing_positive_rate = calibration["missing_prior_zero_positive_rate"]
    if bootstrap["ci_low"] is not None and bootstrap["ci_low"] <= 0.0:
        if predicted_spearman is None or predicted_spearman <= 0.0:
            return {
                "status": "diagnostic_evidence",
                "reason": "全体可用 proposal 的图级 CI 不支持稳定正收益，且评分相关性不足",
                "next_action": "改进 proposal pointer 教师信号，并重设 value/gate 监督目标",
            }
        return {
            "status": "diagnostic_evidence",
            "reason": "全体可用 proposal 的图级 CI 不支持稳定正收益",
            "next_action": "暂不启动 PPO；优先检查 proposal 质量和 gate 标定",
        }
    if missing_positive_rate is not None and missing_positive_rate > 0.0:
        return {
            "status": "positive_proposals_need_calibration",
            "reason": "未被 prior=0 选择的 proposal 中存在真实正收益样本",
            "next_action": "优先研究 value/gate 正类召回与 branch-logit 校准",
        }
    return {
        "status": "requires_review",
        "reason": "证据未归入预设模式",
        "next_action": "继续独立校准审计，不启动 PPO",
    }
def run_quality_calibration(
    *,
    checkpoint_path: Path,
    asset_dir: Path,
    source_manifest_path: Path,
    calibration_dir: Path,
    online_dir: Path,
    data_file: Path,
    experiment_path: Path,
    output_dir: Path,
    bootstrap_reps: int = 10_000,
    bootstrap_seed: int = 42,
    max_episode_steps: int = MAX_EPISODE_STEPS,
    device_name: str = "auto",
) -> dict[str, Any]:
    pytest_root = (PROJECT_ROOT / ".pytest_tmp").resolve()
    ensure_output_dir(output_dir, pytest_root=pytest_root)

    asset, integrity, asset_raw_sha, source_sha = _load_asset_semantics(
        asset_dir,
        source_manifest_path,
    )
    checkpoint, checkpoint_sha = _load_checkpoint_and_validate(
        checkpoint_path,
        asset_raw_sha=asset_raw_sha,
    )
    asset_sha256 = {
        "raw": asset_raw_sha,
        "canonical": str(asset.get("manifest_sha256")),
        "source": source_sha,
    }
    if asset_sha256["canonical"] != EXPECTED_ASSET_CANONICAL_SHA:
        raise ValueError("asset canonical SHA 不一致")
    if source_sha != EXPECTED_SOURCE_SHA or asset_raw_sha != EXPECTED_ASSET_RAW_SHA:
        raise ValueError("资产或 source SHA 不符合固定输入")

    online_report = _load_json(online_dir / "prior_sweep_online_gain_report.json")
    online_integrity = _load_json(online_dir / "integrity_check.json")
    online_contract = _validate_online_inputs(
        checkpoint_sha256=checkpoint_sha,
        asset_sha256=asset_sha256,
        online_report=online_report,
        online_integrity=online_integrity,
    )
    online_rows = _load_csv_rows(online_dir / "prior_sweep_online_gain_by_state.csv")
    prior_zero_rows = [row for row in online_rows if float(row["prior_logit"]) == 0.0]
    if len(prior_zero_rows) != EXPECTED_FROZEN_STATE_COUNT:
        raise ValueError("既有 prior=0 状态数不是 96")
    observed_counts = {
        prior: sum(float(row["prior_logit"]) == prior and str(row["selected"]).lower() == "true" for row in online_rows)
        for prior in (-4.0, -2.0, -1.0, 0.0)
    }
    if observed_counts != EXPECTED_PRIOR_COUNTS:
        raise ValueError(f"既有 prior 采用数量不一致: {observed_counts}")

    from scripts.diagnose_apcf_prior_sweep_online_gain import _validate_calibration

    calibration_report, calibration_rows = _validate_calibration(
        calibration_dir,
        checkpoint_sha256=checkpoint_sha,
    )
    candidate_level = calibration_report.get("candidate_level", {})
    validate_layer_separation(
        candidate_count=int(candidate_level.get("count", -1)),
        policy_count=EXPECTED_FROZEN_STATE_COUNT,
    )
    if calibration_report.get("asset") != {
        "manifest_raw_sha256": asset_sha256["raw"],
        "manifest_canonical_sha256": asset_sha256["canonical"],
        "source_manifest_sha256": asset_sha256["source"],
        "integrity_status": "passed",
    }:
        raise ValueError("calibration 报告资产 SHA 不一致")

    prior_zero_by_key, reused_by_key = _validate_online_rows(
        online_rows=online_rows,
        prior_zero_rows=prior_zero_rows,
        calibration_rows=calibration_rows,
        checkpoint_sha256=checkpoint_sha,
        asset_sha256=asset_sha256,
        online_protocol=online_report.get("protocol", {}),
    )
    missing_rows = select_missing_available_states(prior_zero_rows)
    if any(str(row["proposal_available"]).lower() != "true" for row in missing_rows):
        raise ValueError("不可用 proposal 不得进入补齐集合")

    config = _build_config(
        experiment_path=experiment_path,
        checkpoint=checkpoint,
        asset_manifest=asset_dir / "manifest.json",
        data_file=data_file,
    )
    device = torch.device(
        "cuda" if device_name == "cuda" or (device_name == "auto" and torch.cuda.is_available()) else "cpu"
    )
    agent = _build_inference_agent(config, checkpoint, device)
    agent.policy.eval()
    model_hash_before = _state_dict_sha256(agent.policy)
    formula_spec, formula_audit = _resolve_formula_spec(
        checkpoint,
        config,
        agent.policy.anchor_proposal_gate,
    )

    supplemented: dict[Any, dict[str, Any]] = {}
    if missing_rows:
        source_manifest = _load_json(source_manifest_path)
        groups, csv_by_sha = _prepare_state_groups(
            asset=asset,
            asset_dir=asset_dir,
            source_manifest=source_manifest,
            source_manifest_path=source_manifest_path,
        )
        missing_keys = {_state_key_from_row(row) for row in missing_rows}
        replay_config = _build_replay_config(data_file=data_file)
        snapshots: dict[Any, dict[str, Any]] = {}
        for csv_sha, csv_path in sorted(csv_by_sha.items()):
            expected = {
                key: group
                for key, group in groups.items()
                if key in missing_keys and key[0] == csv_sha
            }
            if expected:
                snapshots.update(
                    _replay_frozen_state_snapshots(
                        csv_path=csv_path,
                        csv_sha256=csv_sha,
                        expected_groups=expected,
                        replay_config=replay_config,
                        seed=42,
                        max_episode_steps=max_episode_steps,
                    )
                )
        if set(snapshots) != missing_keys:
            raise ValueError(
                f"补齐状态回放不完整: expected={len(missing_keys)} actual={len(snapshots)}"
            )
        anchor_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        proposal_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        for row in missing_rows:
            key = _state_key_from_row(row)
            state = snapshots[key]
            calibration_row = calibration_rows[key]
            payload = _select_state_for_prior(
                agent=agent,
                state=state,
                prior_logit=0.0,
                residual_scale=formula_spec.residual_scale,
                delta_temperature=formula_spec.delta_temperature,
                calibration_row=calibration_row,
            )
            if not payload["proposal_available"]:
                raise ValueError("补齐状态 proposal unexpectedly unavailable")
            if tuple(payload["proposal_team"]) != _parse_team(row["proposal_team"]):
                raise ValueError("补齐 proposal 与 production/calibration 不一致")
            _approx_equal(payload["predicted_delta_A"], row["predicted_delta_A"], name="补齐 predicted_delta_A")
            _approx_equal(payload["gate_value"], row["gate_value"], name="补齐 gate_value")
            _approx_equal(payload["raw_gap"], row["raw_gap"], name="补齐 raw_gap")
            anchor_team = _parse_team(row["anchor_team"])
            proposal_team = _parse_team(row["proposal_team"])
            anchor_key = cache_key_anchor(checkpoint_sha, key)
            proposal_key = cache_key_proposal(checkpoint_sha, key, proposal_team)
            if anchor_key not in anchor_cache:
                anchor_cache[anchor_key] = _run_forced_policy_episode(
                    state_env=state["env"],
                    agent=agent,
                    task_id=int(row["task_id"]),
                    station_id=int(row["station_id"]),
                    forced_team=anchor_team,
                    max_episode_steps=max_episode_steps,
                )
            if proposal_key not in proposal_cache:
                proposal_cache[proposal_key] = _run_forced_policy_episode(
                    state_env=state["env"],
                    agent=agent,
                    task_id=int(row["task_id"]),
                    station_id=int(row["station_id"]),
                    forced_team=proposal_team,
                    max_episode_steps=max_episode_steps,
                )
            anchor_outcome = anchor_cache[anchor_key]
            proposal_outcome = proposal_cache[proposal_key]
            validate_online_outcome(anchor_outcome)
            validate_online_outcome(proposal_outcome)
            supplemented[key] = {
                "makespan_anchor": anchor_outcome["makespan"],
                "makespan_proposal": proposal_outcome["makespan"],
                "anchor_steps": anchor_outcome["steps"],
                "proposal_steps": proposal_outcome["steps"],
            }
        if len(supplemented) != len(missing_rows):
            raise ValueError("补齐结果数量不一致")

    selected_by_key_prior: dict[Any, dict[float, bool]] = defaultdict(dict)
    for row in online_rows:
        selected_by_key_prior[_state_key_from_row(row)][float(row["prior_logit"])] = str(row["selected"]).lower() == "true"

    policy_rows: list[dict[str, Any]] = []
    for key, row in sorted(prior_zero_by_key.items()):
        if str(row["proposal_available"]).lower() != "true":
            continue
        calibration_row = calibration_rows[key]
        _validate_calibration_row(row, calibration_row)
        if str(row.get("online_cache_id", "")):
            outcome = _outcome_from_online_row(row)
            source = "reused_online"
        else:
            if key not in supplemented:
                raise ValueError("可用但未覆盖的 proposal 未补齐")
            outcome = supplemented[key]
            source = "supplemented_online"
        policy_rows.append(
            _build_policy_row(
                row=row,
                calibration_row=calibration_row,
                outcome=outcome,
                source=source,
                selected_by_prior=selected_by_key_prior[key],
            )
        )
    if len(policy_rows) != EXPECTED_FROZEN_STATE_COUNT:
        raise ValueError(f"全体可用 proposal 数不是 96: {len(policy_rows)}")

    model_hash_after = _state_dict_sha256(agent.policy)
    if model_hash_after != model_hash_before:
        raise RuntimeError("补齐诊断改变了模型参数")
    gate = agent.policy.anchor_proposal_gate
    if gate is None or float(gate.prior_margin) != 4.0:
        raise RuntimeError("补齐诊断结束时 prior_margin 不是 4.0")

    full_stats = _policy_quality_stats(
        policy_rows,
        bootstrap_reps=bootstrap_reps,
        bootstrap_seed=bootstrap_seed,
    )
    team_ge2 = [row for row in policy_rows if int(row["required_team_size"]) >= 2]
    full_stats["team_edit"] = {
        "required_team_size": _describe([float(row["required_team_size"]) for row in policy_rows]),
        "hamming_distance": _describe([float(row["hamming_distance"]) for row in policy_rows]),
        "normalized_hamming_distance": _describe([float(row["normalized_hamming_distance"]) for row in policy_rows]),
        "required_team_size_ge_2_count": len(team_ge2),
        "single_worker_edit_count_ge_2": sum(int(row["hamming_distance"]) == 1 for row in team_ge2),
        "single_worker_edit_rate_ge_2": (
            sum(int(row["hamming_distance"]) == 1 for row in team_ge2) / len(team_ge2)
            if team_ge2 else None
        ),
    }
    policy_by_key = {_state_key_from_row(row): row for row in policy_rows}
    predicted_values = [float(row["predicted_delta_A"]) for row in policy_rows]
    raw_values = [float(row["raw_gap_prior_minus4"]) for row in policy_rows]
    gains = [float(row["relative_gain"]) for row in policy_rows]
    predicted_metrics = binary_metrics(
        [gain > 0.0 for gain in gains],
        [value > 0.0 for value in predicted_values],
    )
    value_gate_calibration = {
        "spearman_predicted_delta_A_vs_gain": spearman_or_null(predicted_values, gains),
        "spearman_raw_gap_vs_gain": spearman_or_null(raw_values, gains),
        "predicted_positive_metrics": predicted_metrics,
        "actual_positive_support": sum(gain > 0.0 for gain in gains),
        "five_quantile_predicted_delta_A_bins": _five_quantile_bins(policy_rows),
        "predicted_delta_A": _describe(predicted_values),
        "gate_value": _describe([float(row["gate_value"]) for row in policy_rows]),
        "residual_term": _describe([float(row["residual_term"]) for row in policy_rows]),
        "raw_gap_prior_minus4": _describe(raw_values),
        "missing_prior_zero_state_count": len(missing_rows),
        "missing_prior_zero_positive_count": sum(
            float(policy_by_key[_state_key_from_row(row)]["relative_gain"]) > 0.0
            for row in missing_rows
        ),
        "missing_prior_zero_positive_rate": (
            sum(float(policy_by_key[_state_key_from_row(row)]["relative_gain"]) > 0.0 for row in missing_rows) / len(missing_rows)
            if missing_rows else None
        ),
    }
    prior_quality = _prior_selection_quality(
        all_policy_rows=policy_rows,
        online_rows=online_rows,
        bootstrap_reps=bootstrap_reps,
        bootstrap_seed=bootstrap_seed,
    )
    decision = _build_decision(
        full_stats=full_stats,
        calibration=value_gate_calibration,
        prior_quality=prior_quality,
        missing_rows=missing_rows,
    )
    candidate_policy_rows = _candidate_policy_summary(candidate_level, policy_rows)
    policy_graph_rows = _policy_graph_rows(policy_rows)
    source_counts = {
        "reused_online": sum(row["source"] == "reused_online" for row in policy_rows),
        "supplemented_online": sum(row["source"] == "supplemented_online" for row in policy_rows),
    }
    report = {
        "status": "passed",
        "decision": decision,
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha},
        "asset": {
            "manifest_raw_sha256": asset_sha256["raw"],
            "manifest_canonical_sha256": asset_sha256["canonical"],
            "source_manifest_sha256": asset_sha256["source"],
            "integrity_status": integrity.get("status"),
        },
        "inputs": {
            "calibration_dir": str(calibration_dir),
            "online_dir": str(online_dir),
            "data_file": str(data_file),
            "real_680_used_as_diagnostic_sample": False,
        },
        "protocol": online_contract["protocol"],
        "state_summary": {
            "frozen_graph_count": EXPECTED_FROZEN_GRAPH_COUNT,
            "frozen_state_count": EXPECTED_FROZEN_STATE_COUNT,
            "frozen_candidate_count": EXPECTED_FROZEN_CANDIDATE_COUNT,
            "available_policy_count": len(policy_rows),
            "reused_online_count": source_counts["reused_online"],
            "supplemented_online_count": source_counts["supplemented_online"],
        },
        "proposal_quality": full_stats,
        "value_gate_calibration": value_gate_calibration,
        "prior_selection_quality": prior_quality,
        "candidate_level": candidate_level,
        "layer_separation": {
            "candidate_level_count": EXPECTED_FROZEN_CANDIDATE_COUNT,
            "policy_level_count": len(policy_rows),
            "candidate_labels_used_for_policy_gain": False,
        },
        "formula": formula_audit,
        "errors": [],
    }
    run_manifest = {
        "script": "audit_apcf_proposal_quality_calibration.py",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "asset_manifest_raw_sha256": asset_sha256["raw"],
        "asset_manifest_canonical_sha256": asset_sha256["canonical"],
        "source_manifest_sha256": asset_sha256["source"],
        "calibration_dir": str(calibration_dir),
        "online_dir": str(online_dir),
        "target_prior_scope": "reused_selection_and_missing_prior_zero",
        "continuation_prior_logit": CONTINUATION_PRIOR_LOGIT,
        "temperature": CONTINUATION_TEMPERATURE,
        "deterministic": True,
        "branch_floor": 0.0,
        "bootstrap_reps": bootstrap_reps,
        "bootstrap_seed": bootstrap_seed,
        "device": str(device),
        "seed": 42,
        "frozen_graph_count": EXPECTED_FROZEN_GRAPH_COUNT,
        "frozen_state_count": EXPECTED_FROZEN_STATE_COUNT,
        "frozen_candidate_count": EXPECTED_FROZEN_CANDIDATE_COUNT,
        "reused_online_count": source_counts["reused_online"],
        "supplemented_online_count": source_counts["supplemented_online"],
        "model_state_hash_before": model_hash_before,
        "model_state_hash_after": model_hash_after,
        "model_state_hash_equal": model_hash_before == model_hash_after,
        "git_commit": _git_commit(),
        "online_protocol_verification": online_contract["protocol"],
    }
    report["run_manifest"] = run_manifest

    state_fields = (
        "graph_id", "csv_sha256", "decision_count", "task_id", "station_id",
        "anchor_team", "proposal_team", "required_team_size", "hamming_distance",
        "normalized_hamming_distance", "proposal_available", "predicted_delta_A",
        "gate_value", "residual_term", "raw_gap_prior_minus4", "raw_gap_prior_zero",
        "raw_branch_prior_zero", "anchor_makespan", "proposal_makespan", "relative_gain",
        "anchor_done", "proposal_done", "anchor_steps", "proposal_steps", "source",
        "selected_by_prior_minus4", "selected_by_prior_minus2", "selected_by_prior_minus1",
        "selected_by_prior_zero",
    )
    _write_csv(output_dir / "policy_level_by_state.csv", policy_rows, state_fields)
    _write_csv(
        output_dir / "policy_level_by_graph.csv",
        policy_graph_rows,
        ("graph_id", "state_count", "mean_relative_gain", "median_relative_gain", "positive_state_count", "positive_state_rate", "mean_predicted_delta_A", "mean_raw_gap_prior_minus4"),
    )
    _write_csv(
        output_dir / "prior_selection_quality.csv",
        prior_quality,
        ("prior_logit", "selected_state_count", "selected_graph_count", "selected_gain_mean", "selected_gain_median", "positive_state_rate", "bootstrap_ci_low", "bootstrap_ci_high", "bootstrap_point_estimate", "bootstrap_reps", "bootstrap_seed", "difference_vs_all_mean", "difference_vs_all_median", "admission"),
    )
    _write_csv(
        output_dir / "candidate_vs_policy_summary.csv",
        candidate_policy_rows,
        ("layer", "metric", "count", "positive_count", "positive_rate", "tp", "fp", "fn", "precision", "recall", "f1", "mean_relative_gain"),
    )
    (output_dir / "proposal_quality_calibration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "integrity_check.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "checkpoint_sha256": checkpoint_sha,
                "asset_sha256": asset_sha256,
                "model_state_hash_equal": model_hash_before == model_hash_after,
                "reused_online_count": source_counts["reused_online"],
                "supplemented_online_count": source_counts["supplemented_online"],
                "policy_level_count": len(policy_rows),
                "candidate_level_count": EXPECTED_FROZEN_CANDIDATE_COUNT,
                "errors": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "APCF 自由 Proposal 质量与评分校准只读审计。\n"
        "本审计复用既有 prior online 结果，并只补齐 prior=0 未选择但可用的 proposal。\n"
        "data/680.csv 仅用于复用初始调度工人映射配置；frozen 状态来自 APCF asset/source manifest。\n"
        "real_680 不参与本次诊断样本。\n"
        "candidate-level 504 条资产候选与 policy-level 96 条自由 proposal 严格分开。\n"
        "本轮不训练、不修改 checkpoint、gate loss、PPO、配置或正式结果。\n",
        encoding="utf-8",
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="APCF Proposal 质量与评分校准只读审计")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--asset-dir", default="data/initial_anchor_proposal_cf_v1")
    parser.add_argument("--source-manifest", default="data/scale_400_800_datasets/manifest_ctg_160_explicit_fiveskill_v1.json")
    parser.add_argument("--calibration-dir", required=True)
    parser.add_argument("--online-dir", required=True)
    parser.add_argument("--data-file", default="data/680.csv")
    parser.add_argument("--experiment", default="conf/experiment/initial_anchor_proposal_cf_v1.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--max-episode-steps", type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_quality_calibration(
            checkpoint_path=Path(args.checkpoint).resolve(),
            asset_dir=Path(args.asset_dir).resolve(),
            source_manifest_path=Path(args.source_manifest).resolve(),
            calibration_dir=Path(args.calibration_dir).resolve(),
            online_dir=Path(args.online_dir).resolve(),
            data_file=Path(args.data_file).resolve(),
            experiment_path=Path(args.experiment).resolve(),
            output_dir=Path(args.output_dir).resolve(),
            bootstrap_reps=int(args.bootstrap_reps),
            bootstrap_seed=int(args.bootstrap_seed),
            max_episode_steps=int(args.max_episode_steps),
            device_name=str(args.device),
        )
        print(
            "[apcf-quality] status=passed "
            f"policy={report['state_summary']['available_policy_count']} "
            f"reused={report['state_summary']['reused_online_count']} "
            f"supplemented={report['state_summary']['supplemented_online_count']}"
        )
        return 0
    except Exception as error:
        print(f"[apcf-quality] failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())