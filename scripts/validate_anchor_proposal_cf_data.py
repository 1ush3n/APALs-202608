# -*- coding: utf-8 -*-
"""独立核验 APCF 反事实数据资产的来源、样本摘要与训练可读性。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


REAL_FOUR_FILENAMES = {"283.csv", "680.csv", "2338.csv", "3182.csv"}
SPLIT_NAMES = ("pretrain", "frozen_diagnostic", "ppo_only")
SAMPLE_SPLITS = {"pretrain", "frozen_diagnostic"}
ARRAY_NAMES = (
    "task_x",
    "worker_x",
    "station_x",
    "task_mask",
    "station_mask",
    "worker_mask",
)
META_FIELDS = {
    "csv_sha256",
    "manifest_sha256",
    "state_seed",
    "task_id",
    "station_id",
    "anchor_team",
    "candidate_team",
    "source",
    "baseline_makespan",
    "candidate_makespan",
    "relative_gain",
    "episode_steps",
    "terminal_done",
}
ALLOWED_SOURCES = {"anchor", "one_swap", "two_swap", "two_swap_hash"}
EXPECTED_SPLIT_COUNTS = {"pretrain": 96, "frozen_diagnostic": 24, "ppo_only": 40}
GAIN_TOLERANCE = 1.0e-12


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 JSON：{path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return value


def _resolve_inside(root: Path, raw_path: object, *, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} 必须是非空相对路径")
    candidate = (root / raw_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} 越出数据目录：{raw_path}") from exc
    if not candidate.is_file():
        raise ValueError(f"{label} 文件缺失：{raw_path}")
    return candidate


def _decode_meta(value: Any) -> tuple[bytes, dict[str, Any]]:
    scalar = value.item() if isinstance(value, np.ndarray) and value.shape == () else value
    if isinstance(scalar, np.bytes_):
        scalar = bytes(scalar)
    if not isinstance(scalar, bytes):
        raise ValueError("NPZ meta 必须是 UTF-8 bytes，拒绝非规范格式")
    try:
        decoded = json.loads(scalar.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"NPZ meta 不是有效 UTF-8 JSON：{exc}") from exc
    if not isinstance(decoded, dict):
        raise ValueError("NPZ meta JSON 顶层必须是对象")
    return scalar, decoded


def _as_finite_float(value: object, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 必须是数值") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} 必须是有限数")
    return result


def _canonical_manifest_digest(manifest: dict[str, Any]) -> str:
    copy = dict(manifest)
    copy.pop("manifest_sha256", None)
    return _sha256_bytes(
        json.dumps(copy, sort_keys=True, ensure_ascii=False).encode("utf-8")
    )


def _expected_split(source_files: list[dict[str, Any]]) -> dict[str, list[str]]:
    ordered = sorted(str(entry["sha256"]) for entry in source_files)
    return {
        "pretrain": ordered[:96],
        "frozen_diagnostic": ordered[96:120],
        "ppo_only": ordered[120:],
    }


def _validate_source_manifest(source: dict[str, Any]) -> tuple[dict[str, str], dict[str, list[str]]]:
    if source.get("protocol") != "explicit_fiveskill_v1":
        raise ValueError("源 manifest protocol 必须是 explicit_fiveskill_v1")
    files = source.get("files")
    if not isinstance(files, list) or len(files) != 160:
        raise ValueError("源 manifest 必须恰好包含 160 个训练图")
    source_names: dict[str, str] = {}
    normalised: list[dict[str, Any]] = []
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("源 manifest files 条目必须是对象")
        graph_sha = str(entry.get("sha256", ""))
        filename = str(entry.get("file", ""))
        if len(graph_sha) != 64:
            raise ValueError(f"源 manifest 包含非法 CSV SHA-256：{graph_sha!r}")
        if not filename:
            raise ValueError("源 manifest 包含空文件名")
        if Path(filename).name in REAL_FOUR_FILENAMES:
            raise ValueError(f"源 manifest 不得包含真实四实例：{filename}")
        if graph_sha in source_names:
            raise ValueError(f"源 manifest CSV SHA-256 重复：{graph_sha}")
        source_names[graph_sha] = filename
        normalised.append({"sha256": graph_sha})
    return source_names, _expected_split(normalised)


def _validate_arrays(data: Any) -> dict[str, np.ndarray]:
    missing = set(ARRAY_NAMES) - set(data.files)
    if missing:
        raise ValueError(f"NPZ 缺少数组：{sorted(missing)}")
    arrays = {name: np.asarray(data[name]) for name in ARRAY_NAMES}
    for name in ("task_x", "worker_x", "station_x"):
        if arrays[name].ndim != 2 or not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} 必须是二维有限特征数组")
    if arrays["task_mask"].shape != (arrays["task_x"].shape[0],):
        raise ValueError("task_mask 与 task_x 行数不兼容")
    if arrays["station_mask"].shape != (arrays["task_x"].shape[0], arrays["station_x"].shape[0]):
        raise ValueError("station_mask 与 task/station 特征不兼容")
    if arrays["worker_mask"].shape != (arrays["worker_x"].shape[0],):
        raise ValueError("worker_mask 与 worker_x 行数不兼容")
    for name in ("task_mask", "station_mask", "worker_mask"):
        if arrays[name].dtype != np.bool_:
            raise ValueError(f"{name} 必须是 bool 数组")
    return arrays


def _validate_observation_shapes(obs_path: Path, arrays: dict[str, np.ndarray]) -> None:
    try:
        obs = torch.load(obs_path, map_location="cpu", weights_only=False)
        shapes = {
            "task_x": tuple(obs["task"].x.shape),
            "worker_x": tuple(obs["worker"].x.shape),
            "station_x": tuple(obs["station"].x.shape),
        }
    except (OSError, KeyError, AttributeError, RuntimeError) as exc:
        raise ValueError(f"无法读取或解析观测文件：{obs_path}: {exc}") from exc
    for name, observed_shape in shapes.items():
        if observed_shape != tuple(arrays[name].shape):
            raise ValueError(
                f"观测 {name} shape={observed_shape} 与 NPZ={tuple(arrays[name].shape)} 不一致"
            )


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(max(values)),
    }


def validate_counterfactual_asset(
    dataset_dir: Path,
    source_manifest_path: Path,
) -> dict[str, object]:
    """验证 APCF 资产；任何来源、摘要或样本语义不一致均直接抛出 ValueError。"""
    root = Path(dataset_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"数据目录不存在：{root}")
    source_path = Path(source_manifest_path).resolve()
    if not source_path.is_file():
        raise ValueError(f"源 manifest 不存在：{source_path}")
    source = _load_json(source_path)
    source_names, expected_split = _validate_source_manifest(source)
    source_sha = _sha256_file(source_path)

    asset_path = root / "manifest.json"
    asset = _load_json(asset_path)
    if asset.get("kind") != "initial_anchor_proposal_counterfactual_v1":
        raise ValueError("资产 manifest kind 不匹配")
    if asset.get("source_manifest_sha256") != source_sha:
        raise ValueError("资产 source_manifest_sha256 与当前源 manifest 不一致")
    if asset.get("manifest_sha256") != _canonical_manifest_digest(asset):
        raise ValueError("资产 manifest_sha256 不匹配")
    if asset.get("split_counts") != EXPECTED_SPLIT_COUNTS:
        raise ValueError("资产 split_counts 必须是 96/24/40")
    if asset.get("split") != expected_split:
        raise ValueError("asset split does not match source split")
    command_args = asset.get("command_args")
    if not isinstance(command_args, dict):
        command_args = {}
    max_episode_steps = int(command_args.get("max_episode_steps", 1200))
    formal_asset = int(command_args.get("max_graphs", 0)) == 0

    entries = asset.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("资产 files 必须是非空列表")
    seen_sample_hashes: set[str] = set()
    state_rows: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    source_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    positive_gains: list[float] = []
    differing_candidates = 0

    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("资产 files 条目必须是对象")
        split_name = str(entry.get("split", ""))
        if split_name not in SAMPLE_SPLITS:
            raise ValueError(f"sample is not in a supported split: {split_name!r}")
        source_name = str(entry.get("source", ""))
        if source_name not in ALLOWED_SOURCES:
            raise ValueError(f"source is not allowed: {source_name!r}")
        entry_steps = entry.get("episode_steps")
        if isinstance(entry_steps, bool) or not isinstance(entry_steps, int) or entry_steps <= 0:
            raise ValueError("episode_steps must be a positive integer")
        if entry_steps > max_episode_steps:
            raise ValueError("episode_steps exceeds max_episode_steps")
        if entry.get("terminal_done") is not True:
            raise ValueError("terminal_done must be true")
        graph_sha = str(entry.get("csv_sha256", ""))
        if graph_sha not in expected_split[split_name]:
            raise ValueError("样本 CSV SHA-256 与所属 split 不一致")
        filename = source_names.get(graph_sha)
        if filename is None:
            raise ValueError("样本 CSV SHA-256 不属于源 manifest")
        if Path(filename).name in REAL_FOUR_FILENAMES:
            raise ValueError(f"样本引用真实四实例：{filename}")
        sample_sha = str(entry.get("sample_sha256", ""))
        if len(sample_sha) != 64:
            raise ValueError("样本 sample_sha256 非法")
        if sample_sha in seen_sample_hashes:
            raise ValueError(f"样本 sample_sha256 重复：{sample_sha}")
        seen_sample_hashes.add(sample_sha)
        npz_path = _resolve_inside(root, entry.get("npz"), label="npz")
        obs_path = _resolve_inside(root, entry.get("obs_pt"), label="obs_pt")
        expected_npz_sha = str(entry.get("npz_sha256", ""))
        expected_obs_sha = str(entry.get("obs_pt_sha256", ""))
        if expected_npz_sha != _sha256_file(npz_path):
            raise ValueError("npz_sha256 does not match file bytes")
        if expected_obs_sha != _sha256_file(obs_path):
            raise ValueError("obs_pt_sha256 does not match file bytes")

        with np.load(npz_path, allow_pickle=False) as data:
            if "meta" not in data.files:
                raise ValueError("NPZ 缺少 meta")
            meta_bytes, meta = _decode_meta(data["meta"])
            arrays = _validate_arrays(data)
        missing_fields = META_FIELDS - set(meta)
        if missing_fields:
            raise ValueError(f"NPZ meta 缺少字段：{sorted(missing_fields)}")
        for name in ("csv_sha256", "state_seed", "task_id", "station_id", "anchor_team", "candidate_team", "source", "episode_steps", "terminal_done"):
            if meta[name] != entry.get(name):
                raise ValueError(f"NPZ meta.{name} does not match manifest entry")
        if meta["terminal_done"] is not True:
            raise ValueError("terminal_done must be true in NPZ meta")
        if isinstance(meta["episode_steps"], bool) or not isinstance(meta["episode_steps"], int):
            raise ValueError("episode_steps must be an integer in NPZ meta")
        if meta["manifest_sha256"] != source_sha:
            raise ValueError("NPZ meta.manifest_sha256 与源 manifest 不一致")
        baseline = _as_finite_float(meta["baseline_makespan"], label="baseline_makespan")
        candidate = _as_finite_float(meta["candidate_makespan"], label="candidate_makespan")
        if baseline <= 0.0 or candidate <= 0.0:
            raise ValueError("baseline_makespan 与 candidate_makespan 必须为正")
        expected_gain = (baseline - candidate) / baseline
        meta_gain = _as_finite_float(meta["relative_gain"], label="NPZ meta.relative_gain")
        entry_gain = _as_finite_float(entry.get("relative_gain"), label="manifest relative_gain")
        if abs(meta_gain - expected_gain) > GAIN_TOLERANCE or abs(entry_gain - expected_gain) > GAIN_TOLERANCE:
            raise ValueError("relative_gain 与 NPZ meta makespan 计算结果不一致")
        digest = hashlib.sha256()
        digest.update(meta_bytes)
        for name in ARRAY_NAMES:
            digest.update(np.ascontiguousarray(arrays[name]).tobytes())
        if digest.hexdigest() != sample_sha:
            raise ValueError("sample_sha256 与 NPZ meta/数组摘要不一致")
        _validate_observation_shapes(obs_path, arrays)

        anchor = tuple(int(worker) for worker in entry["anchor_team"])
        team = tuple(int(worker) for worker in entry["candidate_team"])
        if not anchor or len(anchor) != len(team):
            raise ValueError("candidate_team must have the same nonzero size as anchor_team")
        if len(set(anchor)) != len(anchor) or len(set(team)) != len(team):
            raise ValueError("team contains duplicate workers")
        worker_count = int(arrays["worker_x"].shape[0])
        for label, members in (("anchor_team", anchor), ("candidate_team", team)):
            for worker in members:
                if worker < 0 or worker >= worker_count:
                    raise ValueError(f"{label} worker id is out of range")
                if bool(arrays["worker_mask"][worker]):
                    raise ValueError(f"{label} worker violates worker_mask")
        if source_name == "anchor":
            if team != anchor:
                raise ValueError("anchor candidate_team must equal anchor_team")
            if abs(expected_gain) > GAIN_TOLERANCE:
                raise ValueError("anchor relative_gain must be zero")
        elif team == anchor:
            raise ValueError("non-anchor candidate_team must differ from anchor_team")
        state_key = (graph_sha, str(entry["state_seed"]), int(entry["task_id"]), int(entry["station_id"]))
        state_rows[state_key].append(
            {
                "source": source_name,
                "gain": expected_gain,
                "anchor": anchor,
                "team": team,
                "episode_steps": entry_steps,
                "terminal_done": True,
            }
        )
        source_counts[source_name] += 1
        split_counts[split_name] += 1
        if expected_gain > 0.0:
            positive_gains.append(expected_gain)
        if set(team) != set(anchor):
            differing_candidates += 1

    positive_states = 0
    candidate_counts: list[int] = []
    episode_steps_values: list[int] = []
    graph_state_counts: Counter[str] = Counter()
    for state_key, rows in state_rows.items():
        anchors = [row for row in rows if row["source"] == "anchor"]
        if len(anchors) != 1:
            raise ValueError(f"state {state_key} must contain exactly one anchor row")
        if abs(float(anchors[0]["gain"])) > GAIN_TOLERANCE:
            raise ValueError(f"state {state_key} anchor relative_gain must be zero")
        candidate_teams = [tuple(row["team"]) for row in rows]
        if len(set(candidate_teams)) != len(candidate_teams):
            raise ValueError(f"state {state_key} contains duplicate candidate_team")
        if formal_asset and not 2 <= len(rows) <= 6:
            raise ValueError(f"state {state_key} candidate count must be in [2, 6]")
        candidate_counts.append(len(rows))
        episode_steps_values.extend(int(row["episode_steps"]) for row in rows)
        graph_state_counts[state_key[0]] += 1
        if any(float(row["gain"]) > 0.0 for row in rows):
            positive_states += 1

    if formal_asset:
        expected_graphs = set(expected_split["pretrain"] + expected_split["frozen_diagnostic"])
        observed_graphs = set(graph_state_counts)
        if observed_graphs != expected_graphs:
            raise ValueError("all 120 sample-bearing graphs must appear in samples")
        bad_graphs = {
            graph_sha for graph_sha in expected_graphs if graph_state_counts[graph_sha] != 4
        }
        if bad_graphs:
            raise ValueError("each sample-bearing graph must contain exactly four states")
    actual_sample_counts = {name: int(split_counts[name]) for name in SPLIT_NAMES}
    if actual_sample_counts["ppo_only"] != 0:
        raise ValueError("ppo_only must contain zero samples")
    if asset.get("sample_counts") != actual_sample_counts:
        raise ValueError("资产 sample_counts 与 files 实际计数不一致")
    return {
        "status": "passed",
        "sample_count": len(entries),
        "state_count": len(state_rows),
        "sample_counts_by_split": actual_sample_counts,
        "sample_counts_by_source": dict(sorted(source_counts.items())),
        "positive_gain_fraction": float(len(positive_gains) / len(entries)),
        "positive_gain_quantiles": _quantiles(positive_gains),
        "states_with_positive_candidate": positive_states,
        "candidate_differs_from_anchor_fraction": float(differing_candidates / len(entries)),
        "sample_graph_count": len(graph_state_counts),
        "sample_graph_counts_by_split": {
            name: len(
                set(graph_sha for graph_sha in graph_state_counts if graph_sha in expected_split[name])
            )
            for name in SAMPLE_SPLITS
        },
        "candidate_count_summary": {
            "min": int(min(candidate_counts)),
            "max": int(max(candidate_counts)),
            "total": int(sum(candidate_counts)),
        },
        "episode_steps_summary": _quantiles([float(value) for value in episode_steps_values]),
        "source_manifest_sha256": source_sha,
        "asset_manifest_sha256": str(asset["manifest_sha256"]),
        "worker_mapping_reference": "data/680.csv configures the 100-worker initial scheduling universe only; samples come from source manifest CSVs.",
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--write-report", required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = validate_counterfactual_asset(
            Path(args.dataset_dir), Path(args.source_manifest)
        )
    except ValueError as exc:
        print(f"[apcf-validate] FAILED: {exc}", file=sys.stderr, flush=True)
        return 2
    report_path = Path(args.write_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
