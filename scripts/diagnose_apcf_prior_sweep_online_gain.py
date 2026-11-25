# -*- coding: utf-8 -*-
"""APCF prior sweep 的在线真实收益只读诊断。"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import Config
from core.action_completion import EarliestFinishActionCompleter
from environment import AirLineEnv_Graph
from ppo_agent import PPOAgent
from scripts.audit_initial_team_opportunity_full import (
    _SHARED_ENV_ATTRIBUTES,
    _select_pair,
)
from scripts.diagnose_apcf_gate_calibration import (
    EXPECTED_ASSET_RAW_SHA,
    EXPECTED_FROZEN_CANDIDATE_COUNT,
    EXPECTED_FROZEN_GRAPH_COUNT,
    EXPECTED_FROZEN_STATE_COUNT,
    EXPECTED_SOURCE_SHA,
    MAX_EPISODE_STEPS,
    _assert_observation_equal,
    _build_config,
    _build_inference_agent,
    _build_replay_config,
    _load_asset_semantics,
    _load_checkpoint_and_validate,
    _load_json,
    _prepare_state_groups,
    _resolve_formula_spec,
    _sha256_file,
    _workspace_path,
    expected_raw_branch,
)

EXPECTED_PRIOR_COUNTS: dict[float, int] = {-4.0: 0, -2.0: 9, -1.0: 33, 0.0: 63}
CONTINUATION_PRIOR_LOGIT = -4.0
CONTINUATION_PRIOR_MARGIN = 4.0
CONTINUATION_TEMPERATURE = 0.0
EVIDENCE_GRAPH_THRESHOLD = 8


def _finite(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} 必须为有限数值")
    return result


def relative_gain(
    anchor_makespan: float,
    proposal_makespan: float,
    *,
    epsilon: float = 1.0e-9,
) -> float:
    anchor = _finite(anchor_makespan, name="anchor_makespan")
    proposal = _finite(proposal_makespan, name="proposal_makespan")
    if anchor <= 0.0 or proposal <= 0.0:
        raise ValueError("makespan 必须为正")
    if epsilon <= 0.0 or not math.isfinite(float(epsilon)):
        raise ValueError("epsilon 必须为正且有限")
    return _finite((anchor - proposal) / max(anchor, float(epsilon)), name="relative_gain")


def bootstrap_graph_means(
    rows: list[dict[str, object]],
    *,
    bootstrap_reps: int = 10_000,
    seed: int = 42,
) -> dict[str, object]:
    if bootstrap_reps < 1:
        raise ValueError("bootstrap_reps 必须为正")
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["graph_id"])].append(
            _finite(row["relative_gain"], name="relative_gain")
        )
    graph_means = {
        graph_id: float(np.mean(values))
        for graph_id, values in sorted(grouped.items())
    }
    if not graph_means:
        return {
            "graph_count": 0,
            "state_count": 0,
            "point_estimate": None,
            "ci_low": None,
            "ci_high": None,
            "bootstrap_reps": int(bootstrap_reps),
            "bootstrap_seed": int(seed),
            "graph_means": {},
        }
    values = np.asarray(list(graph_means.values()), dtype=np.float64)
    sampled = np.random.default_rng(int(seed)).choice(
        values,
        size=(int(bootstrap_reps), values.size),
        replace=True,
    ).mean(axis=1)
    return {
        "graph_count": int(values.size),
        "state_count": int(sum(len(items) for items in grouped.values())),
        "point_estimate": float(values.mean()),
        "ci_low": float(np.percentile(sampled, 2.5)),
        "ci_high": float(np.percentile(sampled, 97.5)),
        "bootstrap_reps": int(bootstrap_reps),
        "bootstrap_seed": int(seed),
        "graph_means": graph_means,
    }


@contextmanager
def prior_margin_override(gate: Any, *, prior_logit: float) -> Iterator[None]:
    original_margin = float(gate.prior_margin)
    try:
        gate.prior_margin = -float(prior_logit)
        yield
    finally:
        gate.prior_margin = original_margin


def make_state_key(
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
        tuple(sorted(int(worker) for worker in anchor_team)),
    )


def make_anchor_cache_key(
    *,
    checkpoint_sha256: str,
    state_key: tuple[str, int, int, int, tuple[int, ...]],
    continuation_prior_logit: float,
    temperature: float,
) -> tuple[Any, ...]:
    return (
        str(checkpoint_sha256),
        state_key,
        float(continuation_prior_logit),
        float(temperature),
    )


def make_proposal_cache_key(
    *,
    checkpoint_sha256: str,
    state_key: tuple[str, int, int, int, tuple[int, ...]],
    proposal_team: Iterable[int],
    continuation_prior_logit: float,
    temperature: float,
) -> tuple[Any, ...]:
    return (
        str(checkpoint_sha256),
        state_key,
        tuple(sorted(int(worker) for worker in proposal_team)),
        float(continuation_prior_logit),
        float(temperature),
    )


def validate_prior_sweep_counts(expected: Mapping[float, int], actual: Mapping[float, int]) -> bool:
    expected_norm = {float(key): int(value) for key, value in expected.items()}
    actual_norm = {float(key): int(value) for key, value in actual.items()}
    if expected_norm != actual_norm:
        raise ValueError(f"prior sweep 采用数量不一致: expected={expected_norm}, actual={actual_norm}")
    return True


def classify_admission(*, selected_graph_count: int, ci_low: float | None) -> str:
    if int(selected_graph_count) <= 0:
        return "not_selected"
    if int(selected_graph_count) < EVIDENCE_GRAPH_THRESHOLD:
        return "insufficient_evidence"
    if ci_low is None or float(ci_low) <= 0.0:
        return "rejected"
    return "positive_evidence"


def make_state_prior_record(
    *,
    state_key: tuple[str, int, int, int, tuple[int, ...]],
    prior_logit: float,
    proposal_available: bool,
    selected: bool,
    anchor_team: Iterable[int],
    proposal_team: Iterable[int] | None,
    raw_gap: float | None,
    graph_id: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "graph_id": graph_id,
        "csv_sha256": state_key[0],
        "decision_count": state_key[1],
        "task_id": state_key[2],
        "station_id": state_key[3],
        "anchor_team": list(state_key[4]),
        "prior_logit": float(prior_logit),
        "proposal_available": bool(proposal_available),
        "selected": bool(selected),
        "proposal_team": None,
        "raw_gap": None,
        "anchor_makespan": None,
        "proposal_makespan": None,
        "relative_gain": None,
        "anchor_done": None,
        "proposal_done": None,
        "anchor_steps": None,
        "proposal_steps": None,
        "online_cache_id": None,
    }
    if not proposal_available:
        if proposal_team is not None or raw_gap is not None:
            raise ValueError("proposal 不可用时 proposal 字段必须为 null")
    else:
        if proposal_team is None or raw_gap is None:
            raise ValueError("proposal 可用时必须提供 proposal 字段")
        record["proposal_team"] = list(sorted(int(worker) for worker in proposal_team))
        record["raw_gap"] = _finite(raw_gap, name="raw_gap")
    return record


def _state_dict_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def _read_prior_sweep_counts(path: Path) -> dict[float, int]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return {
            float(row["prior_logit"]): int(row["proposal_selected_count"])
            for row in csv.DictReader(handle)
        }


def _read_calibration_state_rows(path: Path) -> dict[tuple[str, int, int, int, tuple[int, ...]], dict[str, Any]]:
    result: dict[tuple[str, int, int, int, tuple[int, ...]], dict[str, Any]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = make_state_key(
                csv_sha256=row["csv_sha256"],
                decision_count=int(row["decision_count"]),
                task_id=int(row["task_id"]),
                station_id=int(row["station_id"]),
                anchor_team=json.loads(row["anchor_team"]),
            )
            if key in result:
                raise ValueError(f"calibration state key 重复: {key!r}")
            result[key] = row
    if len(result) != EXPECTED_FROZEN_STATE_COUNT:
        raise ValueError(f"calibration state 数量不是 96: {len(result)}")
    return result


def _validate_calibration(calibration_dir: Path, *, checkpoint_sha256: str) -> tuple[dict[str, Any], dict[Any, dict[str, Any]]]:
    report = _load_json(calibration_dir / "gate_calibration_report.json")
    if report.get("status") != "passed":
        raise ValueError("calibration report 未通过")
    if str(report["checkpoint"]["sha256"]) != checkpoint_sha256:
        raise ValueError("calibration checkpoint SHA 与当前 checkpoint 不一致")
    validate_prior_sweep_counts(
        EXPECTED_PRIOR_COUNTS,
        _read_prior_sweep_counts(calibration_dir / "prior_sweep_summary.csv"),
    )
    return report, _read_calibration_state_rows(calibration_dir / "gate_calibration_by_state.csv")


def _clone_env(env: AirLineEnv_Graph) -> AirLineEnv_Graph:
    memo = {
        id(getattr(env, name)): getattr(env, name)
        for name in _SHARED_ENV_ATTRIBUTES
        if hasattr(env, name)
    }
    clone = copy.deepcopy(env, memo=memo)
    clone.skip_obs_building = False
    return clone

def _replay_frozen_state_snapshots(
    *,
    csv_path: Path,
    csv_sha256: str,
    expected_groups: Mapping[Any, dict[str, Any]],
    replay_config: Config,
    seed: int,
    max_episode_steps: int,
) -> dict[Any, dict[str, Any]]:
    completer = EarliestFinishActionCompleter(replay_config)
    env = AirLineEnv_Graph(data_path_or_dir=str(csv_path), seed=seed)
    obs = env.reset(randomize_duration=False, randomize_workers=False, seed=seed)
    found: dict[Any, dict[str, Any]] = {}
    decision_count = 0
    step = 0
    done = False
    while not done and step < int(max_episode_steps):
        selected = _select_pair(env, obs, completer, max_candidates=4)
        if selected is None:
            if not env.try_wait_for_resources():
                break
            obs = env._get_observation()
            continue
        decision_count += 1
        if len(selected.candidates.teams) > 1:
            key = make_state_key(
                csv_sha256=csv_sha256,
                decision_count=decision_count,
                task_id=selected.task_id,
                station_id=selected.station_id,
                anchor_team=selected.candidates.teams[0],
            )
            if key in expected_groups:
                if key in found:
                    raise ValueError(f"状态回放重复: {key!r}")
                group = expected_groups[key]
                obs_path = Path(group["obs_pt"])
                expected_hashes = {
                    str(row.get("obs_pt_sha256"))
                    for row in group["candidates"]
                    if row.get("obs_pt_sha256")
                }
                observed_obs_sha256 = _sha256_file(obs_path)
                if expected_hashes and observed_obs_sha256 not in expected_hashes:
                    raise ValueError(f"观测 payload SHA 不一致: {key!r}")
                stored_obs = torch.load(obs_path, map_location="cpu", weights_only=False)
                _assert_observation_equal(stored_obs, obs)
                masks = tuple(
                    value.detach().cpu().clone()
                    if torch.is_tensor(value)
                    else copy.deepcopy(value)
                    for value in env.get_masks()
                )
                with np.load(group["npz"], allow_pickle=True) as npz_data:
                    stored_worker_mask = torch.as_tensor(
                        np.asarray(npz_data["worker_mask"], dtype=np.bool_)
                    )
                if not torch.equal(stored_worker_mask, masks[2].to(torch.bool)):
                    raise ValueError(f"worker_mask 不一致: {key!r}")
                found[key] = {
                    "state_key": key,
                    "graph_id": csv_path.name,
                    "obs": copy.deepcopy(obs),
                    "masks": masks,
                    "observation_verification": {
                        "state_key_verified": True,
                        "obs_pt_path": str(obs_path),
                        "obs_pt_sha256": observed_obs_sha256,
                        "persisted_obs_pt_sha256_checked": bool(expected_hashes),
                        "recursive_payload_equal": True,
                        "worker_mask_equal": True,
                    },
                    "env": _clone_env(env),
                    "task_id": int(selected.task_id),
                    "station_id": int(selected.station_id),
                    "anchor_team": tuple(sorted(int(worker) for worker in selected.candidates.teams[0])),
                    "decision_count": int(decision_count),
                }
        obs, _reward, done, info = env.step(
            (selected.task_id, selected.station_id, list(selected.candidates.teams[0]))
        )
        if "error" in info:
            raise ValueError(f"anchor trajectory action 被拒绝: {info['error']}")
        step += 1
    if set(found) != set(expected_groups):
        raise ValueError(
            "frozen 状态回放目标集合不完整: "
            f"expected={len(expected_groups)} actual={len(found)}"
        )
    del env
    gc.collect()
    return found


def observation_verification_summary(
    snapshots: Mapping[Any, Mapping[str, Any]],
) -> dict[str, object]:
    if not snapshots:
        raise ValueError("observation snapshots 不能为空")
    checks = [snapshot.get("observation_verification") for snapshot in snapshots.values()]
    if any(not isinstance(check, Mapping) for check in checks):
        raise ValueError("缺少 observation verification 记录")
    required = ("state_key_verified", "recursive_payload_equal", "worker_mask_equal")
    for check in checks:
        assert isinstance(check, Mapping)
        if any(check.get(field) is not True for field in required):
            raise ValueError("观测或状态核验未通过")
    return {
        "status": "passed",
        "state_count": len(checks),
        "state_keys_verified": len(checks),
        "recursive_payload_equal_count": sum(bool(check["recursive_payload_equal"]) for check in checks),
        "worker_mask_equal_count": sum(bool(check["worker_mask_equal"]) for check in checks),
        "persisted_obs_pt_sha256_checked_count": sum(bool(check["persisted_obs_pt_sha256_checked"]) for check in checks),
    }

def _final_makespan(env: AirLineEnv_Graph) -> float:
    values = np.asarray(getattr(env, "station_wall_clock", []), dtype=np.float64)
    if values.size == 0:
        raise ValueError("环境没有 station_wall_clock")
    result = float(np.max(values))
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("makespan 必须为正且有限")
    return result


def _run_forced_policy_episode(
    *,
    state_env: AirLineEnv_Graph,
    agent: PPOAgent,
    task_id: int,
    station_id: int,
    forced_team: tuple[int, ...],
    max_episode_steps: int,
) -> dict[str, Any]:
    gate = agent.policy.anchor_proposal_gate
    if gate is None or float(gate.prior_margin) != CONTINUATION_PRIOR_MARGIN:
        raise RuntimeError("进入 continuation 前 prior_margin 不是 4.0")
    clone = _clone_env(state_env)
    try:
        obs, _reward, done, info = clone.step(
            (int(task_id), int(station_id), list(forced_team))
        )
        if "error" in info:
            raise ValueError(f"强制团队动作被环境拒绝: {info['error']}")
        steps = 1
        while not done and steps < int(max_episode_steps):
            masks = clone.get_masks()
            # 环境返回的 HeteroData 与 mask 默认位于 CPU；推理模型可能位于 CUDA。
            # 每一步显式迁移，确保生产 select_action 的输入设备一致。
            obs_device = obs.to(agent.device)
            action, _logprob, _value, _specific_mask, invalid = agent.select_action(
                obs_device,
                mask_task=masks[0].to(agent.device),
                mask_station_matrix=masks[1].to(agent.device),
                mask_worker=masks[2].to(agent.device),
                deterministic=True,
                temperature=CONTINUATION_TEMPERATURE,
                is_eval=True,
                compute_value=False,
                manage_optimizer_mode=False,
            )
            if invalid or action is None:
                raise ValueError(f"continuation policy 返回非法动作: {action!r}")
            obs, _reward, done, info = clone.step(action)
            if "error" in info:
                raise ValueError(f"continuation 环境拒绝动作: {info['error']}")
            steps += 1
        if not done:
            raise ValueError(
                f"continuation episode 未完成: steps={steps}, limit={max_episode_steps}"
            )
        return {"makespan": _final_makespan(clone), "steps": steps, "done": True}
    finally:
        del clone
        gc.collect()


def _select_state_for_prior(
    *,
    agent: PPOAgent,
    state: Mapping[str, Any],
    prior_logit: float,
    residual_scale: float,
    delta_temperature: float,
    calibration_row: Mapping[str, Any],
) -> dict[str, Any]:
    obs = copy.deepcopy(state["obs"]).to(agent.device)
    with torch.inference_mode():
        encoded, _context = agent.policy(obs)
        task_id = int(state["task_id"])
        station_id = int(state["station_id"])
        gate = agent.policy.anchor_proposal_gate
        if gate is None:
            raise RuntimeError("APCF gate 不存在")
        with prior_margin_override(gate, prior_logit=prior_logit):
            result = agent._select_anchor_proposal_team(
                agent.policy,
                obs=obs,
                task_id=task_id,
                station_id=station_id,
                worker_mask=state["masks"][2].to(agent.device),
                task_emb=encoded["task"][task_id].unsqueeze(0),
                station_emb=encoded["station"][station_id].unsqueeze(0),
                worker_embs=encoded["worker"],
                deterministic=True,
                temperature=0.0,
                branch_floor=0.0,
            )
    if gate.prior_margin != CONTINUATION_PRIOR_MARGIN:
        raise RuntimeError("prior override 未恢复为 4.0")
    if result is None:
        raise RuntimeError("生产 APCF branch 无法处理 frozen 状态")
    _selected_team, _team_logprob, trace = result
    if not trace.proposal_available:
        return {
            "proposal_available": False,
            "selected": False,
            "anchor_team": tuple(trace.anchor_team),
            "proposal_team": None,
            "raw_gap": None,
            "predicted_delta_A": None,
            "gate_value": None,
            "hamming_distance": None,
            "raw_branch": None,
        }
    proposal_team = tuple(sorted(int(worker) for worker in trace.proposal_team))
    anchor_team = tuple(sorted(int(worker) for worker in trace.anchor_team))
    if proposal_team == anchor_team or len(proposal_team) != len(set(proposal_team)):
        raise ValueError("proposal team 不合法")
    raw_gap = _finite(trace.raw_branch_logit_gap, name="raw_gap")
    predicted_delta = _finite(trace.predicted_delta_a, name="predicted_delta_A")
    gate_value = _finite(trace.gate_value, name="gate_value")
    reconstructed = float(prior_logit) + gate_value * float(residual_scale) * math.tanh(
        predicted_delta / float(delta_temperature)
    )
    if not math.isclose(raw_gap, reconstructed, rel_tol=1.0e-5, abs_tol=1.0e-6):
        raise ValueError("production gap 与公式不一致")
    raw_branch = int(trace.raw_argmax_branch)
    if raw_branch != expected_raw_branch(raw_gap):
        raise ValueError("production branch 与 raw gap 不一致")
    if float(prior_logit) == -4.0:
        if not math.isclose(
            raw_gap,
            float(calibration_row["raw_branch_logit_gap"]),
            rel_tol=1.0e-5,
            abs_tol=1.0e-6,
        ):
            raise ValueError("当前 production gap 与 calibration gap 不一致")
        if raw_branch != int(calibration_row["production_raw_branch"]):
            raise ValueError("当前 production branch 与 calibration 不一致")
    return {
        "proposal_available": True,
        "selected": raw_branch == 1,
        "anchor_team": anchor_team,
        "proposal_team": proposal_team,
        "raw_gap": raw_gap,
        "predicted_delta_A": predicted_delta,
        "gate_value": gate_value,
        "hamming_distance": int(trace.hamming_distance),
        "raw_branch": raw_branch,
    }
def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _run_manifest_base(
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    asset_raw_sha256: str,
    asset_canonical_sha256: str,
    source_sha256: str,
    calibration_dir: Path,
    prior_logits: Sequence[float],
    bootstrap_reps: int,
    bootstrap_seed: int,
    device: torch.device,
    model_state_hash_before: str,
) -> dict[str, Any]:
    return {
        "script": "diagnose_apcf_prior_sweep_online_gain.py",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "asset_manifest_raw_sha256": asset_raw_sha256,
        "asset_manifest_canonical_sha256": asset_canonical_sha256,
        "source_manifest_sha256": source_sha256,
        "calibration_dir": str(calibration_dir),
        "prior_logits": [float(value) for value in prior_logits],
        "target_prior_scope": "target_branch_only",
        "continuation_prior_logit": CONTINUATION_PRIOR_LOGIT,
        "continuation_temperature": CONTINUATION_TEMPERATURE,
        "bootstrap_reps": int(bootstrap_reps),
        "bootstrap_seed": int(bootstrap_seed),
        "device": str(device),
        "seed": 42,
        "frozen_graph_count": EXPECTED_FROZEN_GRAPH_COUNT,
        "frozen_state_count": EXPECTED_FROZEN_STATE_COUNT,
        "frozen_candidate_count": EXPECTED_FROZEN_CANDIDATE_COUNT,
        "git_commit": _git_commit(),
        "model_state_hash_before": model_state_hash_before,
    }


def run_online_diagnostic(
    *,
    checkpoint_path: Path,
    calibration_dir: Path,
    asset_dir: Path,
    source_manifest_path: Path,
    data_file: Path,
    experiment_path: Path,
    output_dir: Path,
    prior_logits: tuple[float, ...],
    bootstrap_reps: int = 10_000,
    bootstrap_seed: int = 42,
    max_episode_steps: int = MAX_EPISODE_STEPS,
    device_name: str = "auto",
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在: {output_dir}")
    pytest_root = (PROJECT_ROOT / ".pytest_tmp").resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    if not output_dir.resolve().is_relative_to(pytest_root):
        raise ValueError("在线诊断输出必须位于项目 .pytest_tmp 下")

    asset, integrity, asset_raw_sha, source_sha = _load_asset_semantics(
        asset_dir,
        source_manifest_path,
    )
    checkpoint, checkpoint_sha = _load_checkpoint_and_validate(
        checkpoint_path,
        asset_raw_sha=asset_raw_sha,
    )
    calibration_report, calibration_rows = _validate_calibration(
        calibration_dir,
        checkpoint_sha256=checkpoint_sha,
    )
    source_manifest = _load_json(source_manifest_path)
    groups, csv_by_sha = _prepare_state_groups(
        asset=asset,
        asset_dir=asset_dir,
        source_manifest=source_manifest,
        source_manifest_path=source_manifest_path,
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
    agent = _build_inference_agent(config, checkpoint, device)
    agent.policy.eval()
    model_hash_before = _state_dict_sha256(agent.policy)
    formula_spec, formula_audit = _resolve_formula_spec(
        checkpoint,
        config,
        agent.policy.anchor_proposal_gate,
    )
    replay_config = _build_replay_config(data_file=data_file)
    snapshots: dict[Any, dict[str, Any]] = {}
    for csv_sha, csv_path in sorted(csv_by_sha.items()):
        expected = {key: group for key, group in groups.items() if key[0] == csv_sha}
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
    observation_verification = observation_verification_summary(snapshots)
    if len(snapshots) != EXPECTED_FROZEN_STATE_COUNT:
        raise ValueError(f"回放状态不是 96: {len(snapshots)}")
    if sorted(float(value) for value in prior_logits) != sorted(EXPECTED_PRIOR_COUNTS):
        raise ValueError(f"prior 集合必须为 {sorted(EXPECTED_PRIOR_COUNTS)}")

    prior_state_rows: dict[float, list[dict[str, Any]]] = {
        float(prior): [] for prior in prior_logits
    }
    online_cache_anchor: dict[tuple[Any, ...], dict[str, Any]] = {}
    online_cache_proposal: dict[tuple[Any, ...], dict[str, Any]] = {}
    online_evaluation_count = 0
    gate = agent.policy.anchor_proposal_gate
    if gate is None:
        raise RuntimeError("APCF gate 不存在")

    for state_key in sorted(snapshots):
        state = snapshots[state_key]
        calibration_row = calibration_rows[state_key]
        selected_payloads: dict[float, dict[str, Any]] = {}
        selected_records: dict[float, dict[str, Any]] = {}
        for prior in prior_logits:
            payload = _select_state_for_prior(
                agent=agent,
                state=state,
                prior_logit=float(prior),
                residual_scale=formula_spec.residual_scale,
                delta_temperature=formula_spec.delta_temperature,
                calibration_row=calibration_row,
            )
            record = make_state_prior_record(
                state_key=state_key,
                prior_logit=float(prior),
                proposal_available=bool(payload["proposal_available"]),
                selected=bool(payload["selected"]),
                anchor_team=payload["anchor_team"],
                proposal_team=payload["proposal_team"],
                raw_gap=payload["raw_gap"],
                graph_id=str(state["graph_id"]),
            )
            record.update(
                {
                    "predicted_delta_A": payload["predicted_delta_A"],
                    "gate_value": payload["gate_value"],
                    "hamming_distance": payload["hamming_distance"],
                    "raw_branch": payload["raw_branch"],
                }
            )
            prior_state_rows[float(prior)].append(record)
            if payload["selected"]:
                selected_payloads[float(prior)] = payload
                selected_records[float(prior)] = record

        if gate.prior_margin != CONTINUATION_PRIOR_MARGIN:
            raise RuntimeError("进入在线排程前 prior_margin 不是 4.0")
        if selected_payloads:
            first_payload = next(iter(selected_payloads.values()))
            proposal_team = tuple(first_payload["proposal_team"])
            anchor_team = tuple(first_payload["anchor_team"])
            anchor_key = make_anchor_cache_key(
                checkpoint_sha256=checkpoint_sha,
                state_key=state_key,
                continuation_prior_logit=CONTINUATION_PRIOR_LOGIT,
                temperature=CONTINUATION_TEMPERATURE,
            )
            proposal_key = make_proposal_cache_key(
                checkpoint_sha256=checkpoint_sha,
                state_key=state_key,
                proposal_team=proposal_team,
                continuation_prior_logit=CONTINUATION_PRIOR_LOGIT,
                temperature=CONTINUATION_TEMPERATURE,
            )
            if anchor_key not in online_cache_anchor:
                online_cache_anchor[anchor_key] = _run_forced_policy_episode(
                    state_env=state["env"],
                    agent=agent,
                    task_id=int(state["task_id"]),
                    station_id=int(state["station_id"]),
                    forced_team=anchor_team,
                    max_episode_steps=max_episode_steps,
                )
            if proposal_key not in online_cache_proposal:
                online_cache_proposal[proposal_key] = _run_forced_policy_episode(
                    state_env=state["env"],
                    agent=agent,
                    task_id=int(state["task_id"]),
                    station_id=int(state["station_id"]),
                    forced_team=proposal_team,
                    max_episode_steps=max_episode_steps,
                )
            anchor_outcome = online_cache_anchor[anchor_key]
            proposal_outcome = online_cache_proposal[proposal_key]
            gain = relative_gain(anchor_outcome["makespan"], proposal_outcome["makespan"])
            online_id = hashlib.sha256(
                repr((anchor_key, proposal_key)).encode("utf-8")
            ).hexdigest()[:16]
            for record in selected_records.values():
                record.update(
                    {
                        "anchor_makespan": anchor_outcome["makespan"],
                        "proposal_makespan": proposal_outcome["makespan"],
                        "relative_gain": gain,
                        "anchor_done": anchor_outcome["done"],
                        "proposal_done": proposal_outcome["done"],
                        "anchor_steps": anchor_outcome["steps"],
                        "proposal_steps": proposal_outcome["steps"],
                        "online_cache_id": online_id,
                    }
                )
            online_evaluation_count += 1

    prior_summary: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    for prior in sorted(prior_state_rows):
        rows = prior_state_rows[prior]
        state_rows.extend(rows)
        selected_rows = [row for row in rows if row["selected"]]
        selected_by_graph: dict[str, list[float]] = defaultdict(list)
        for row in selected_rows:
            selected_by_graph[str(row["graph_id"])].append(
                _finite(row["relative_gain"], name="relative_gain")
            )
        graph_rows.extend(
            {
                "prior_logit": float(prior),
                "graph_id": graph_id,
                "selected_state_count": len(values),
                "mean_relative_gain": float(np.mean(values)),
                "positive_state_count": int(sum(value > 0.0 for value in values)),
            }
            for graph_id, values in sorted(selected_by_graph.items())
        )
        bootstrap = bootstrap_graph_means(
            [
                {"graph_id": graph_id, "relative_gain": gain}
                for graph_id, values in selected_by_graph.items()
                for gain in values
            ],
            bootstrap_reps=bootstrap_reps,
            seed=bootstrap_seed,
        )
        prior_summary.append(
            {
                "prior_logit": float(prior),
                "selected_state_count": len(selected_rows),
                "selected_graph_count": int(bootstrap["graph_count"]),
                "online_evaluation_count": int(
                    len({row["online_cache_id"] for row in selected_rows})
                ),
                "positive_state_count": int(
                    sum(float(row["relative_gain"]) > 0.0 for row in selected_rows)
                ),
                "point_estimate": bootstrap["point_estimate"],
                "ci_low": bootstrap["ci_low"],
                "ci_high": bootstrap["ci_high"],
                "bootstrap_reps": bootstrap["bootstrap_reps"],
                "bootstrap_seed": bootstrap["bootstrap_seed"],
                "admission": classify_admission(
                    selected_graph_count=int(bootstrap["graph_count"]),
                    ci_low=bootstrap["ci_low"],
                ),
            }
        )

    model_hash_after = _state_dict_sha256(agent.policy)
    if model_hash_after != model_hash_before:
        raise RuntimeError("模型 state_dict 在只读诊断前后发生变化")
    if gate.prior_margin != CONTINUATION_PRIOR_MARGIN:
        raise RuntimeError("诊断结束时 prior_margin 不是 4.0")
    actual_counts = {
        float(prior): int(sum(bool(row["selected"]) for row in rows))
        for prior, rows in prior_state_rows.items()
    }
    validate_prior_sweep_counts(EXPECTED_PRIOR_COUNTS, actual_counts)

    report = {
        "status": "passed",
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha},
        "asset": {
            "manifest_raw_sha256": asset_raw_sha,
            "manifest_canonical_sha256": asset.get("manifest_sha256"),
            "source_manifest_sha256": source_sha,
            "integrity_status": integrity.get("status"),
        },
        "calibration": {
            "path": str(calibration_dir),
            "checkpoint_sha256": calibration_report["checkpoint"]["sha256"],
        },
        "protocol": {
            "target_prior_scope": "target_branch_only",
            "continuation_prior_logit": CONTINUATION_PRIOR_LOGIT,
            "temperature": CONTINUATION_TEMPERATURE,
            "branch_floor": 0.0,
            "seed": 42,
            "max_episode_steps": int(max_episode_steps),
        },
        "state_summary": {
            "frozen_graph_count": EXPECTED_FROZEN_GRAPH_COUNT,
            "frozen_state_count": EXPECTED_FROZEN_STATE_COUNT,
            "frozen_candidate_count": EXPECTED_FROZEN_CANDIDATE_COUNT,
            "online_unique_state_evaluations": online_evaluation_count,
            "prior_selected_counts": actual_counts,
        },
        "formula": formula_audit,
        "observation_verification": observation_verification,
        "prior_summary": prior_summary,
        "errors": [],
    }
    model_manifest = _run_manifest_base(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        asset_raw_sha256=asset_raw_sha,
        asset_canonical_sha256=str(asset.get("manifest_sha256")),
        source_sha256=source_sha,
        calibration_dir=calibration_dir,
        prior_logits=prior_logits,
        bootstrap_reps=bootstrap_reps,
        bootstrap_seed=bootstrap_seed,
        device=device,
        model_state_hash_before=model_hash_before,
    )
    model_manifest["model_state_hash_after"] = model_hash_after
    model_manifest["model_state_hash_equal"] = model_hash_before == model_hash_after
    model_manifest["prior_selected_counts"] = actual_counts
    model_manifest["observation_verification"] = observation_verification
    report["run_manifest"] = model_manifest

    _write_csv(
        output_dir / "prior_sweep_online_gain_by_state.csv",
        state_rows,
        (
            "graph_id", "csv_sha256", "decision_count", "task_id", "station_id",
            "anchor_team", "prior_logit", "proposal_available", "selected",
            "proposal_team", "raw_gap", "predicted_delta_A", "gate_value",
            "hamming_distance", "raw_branch", "anchor_makespan",
            "proposal_makespan", "relative_gain", "anchor_done", "proposal_done",
            "anchor_steps", "proposal_steps", "online_cache_id",
        ),
    )
    _write_csv(
        output_dir / "prior_sweep_online_gain_by_graph.csv",
        graph_rows,
        ("prior_logit", "graph_id", "selected_state_count", "mean_relative_gain", "positive_state_count"),
    )
    _write_csv(
        output_dir / "prior_sweep_online_gain_summary.csv",
        prior_summary,
        (
            "prior_logit", "selected_state_count", "selected_graph_count",
            "online_evaluation_count", "positive_state_count", "point_estimate",
            "ci_low", "ci_high", "bootstrap_reps", "bootstrap_seed", "admission",
        ),
    )
    (output_dir / "prior_sweep_online_gain_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(model_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "integrity_check.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "checkpoint_sha256": checkpoint_sha,
                "model_state_hash_equal": model_hash_before == model_hash_after,
                "prior_selected_counts": actual_counts,
                "online_unique_state_evaluations": online_evaluation_count,
                "observation_verification": observation_verification,
                "errors": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "APCF prior sweep online gain 只读诊断。\n"
        "data/680.csv 仅用于复用初始调度工人映射配置；\n"
        "frozen 状态来自 APCF asset/source manifest；\n"
        "real_680 不参与本次诊断样本。\n"
        "只改变目标状态 prior，后续统一使用 prior=-4、temperature=0；"
        "不训练、不修改 checkpoint、不写入正式结果目录。\n",
        encoding="utf-8",
    )
    return report
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="APCF prior sweep 在线真实收益只读诊断")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration-dir", required=True)
    parser.add_argument("--asset-dir", default="data/initial_anchor_proposal_cf_v1")
    parser.add_argument(
        "--source-manifest",
        default="data/scale_400_800_datasets/manifest_ctg_160_explicit_fiveskill_v1.json",
    )
    parser.add_argument("--data-file", default="data/680.csv")
    parser.add_argument(
        "--experiment",
        default="conf/experiment/initial_anchor_proposal_cf_v1.yaml",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prior-logits", nargs="+", type=float, default=[-4.0, -2.0, -1.0, 0.0])
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--max-episode-steps", type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_online_diagnostic(
            checkpoint_path=_workspace_path(args.checkpoint),
            calibration_dir=_workspace_path(args.calibration_dir),
            asset_dir=_workspace_path(args.asset_dir),
            source_manifest_path=_workspace_path(args.source_manifest),
            data_file=_workspace_path(args.data_file),
            experiment_path=_workspace_path(args.experiment),
            output_dir=_workspace_path(args.output_dir),
            prior_logits=tuple(float(value) for value in args.prior_logits),
            bootstrap_reps=int(args.bootstrap_reps),
            bootstrap_seed=int(args.bootstrap_seed),
            max_episode_steps=int(args.max_episode_steps),
            device_name=str(args.device),
        )
        print(
            "[apcf-online-gain] status=passed "
            f"counts={report['state_summary']['prior_selected_counts']} "
            f"online={report['state_summary']['online_unique_state_evaluations']}",
            flush=True,
        )
        return 0
    except Exception as exc:
        print(f"[apcf-online-gain] failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())