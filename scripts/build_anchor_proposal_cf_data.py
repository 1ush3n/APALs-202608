# -*- coding: utf-8 -*-
"""构建 APCF 反事实预训练数据（data/initial_anchor_proposal_cf_v1/）。

协议（与论文实现计划一致）：
  - 输入：CTG-FV1 160 图 manifest（manifest_ctg_160_explicit_fiveskill_v1.json）；
  - split：文件按 CSV SHA-256 确定性排序 → 前 96 图=反事实预训练，
    96..120 图=冻结诊断，余 40 图=仅 PPO（不生成样本）；
  - 每图：沿确定性锚点轨迹行走，在"存在合法替代团队"的决策点中按
    12.5%/37.5%/62.5%/87.5% 分位取 4 个状态；
  - 候选预算：锚点 + 2 个局部启发式最优单替换 + 2 个有界双替换
    + 1 个由状态哈希确定的双替换池代表；
  - 每个候选执行完整续排反事实（强制该团队一步、固定工序—工位、之后
    候选 0 策略跑完），记录相对收益 y=(C(H)-C(P))/max(C(H),eps)；
  - 样本保存 CPU 图观测、三类动作掩码、锚点/候选、收益、来源与哈希链；
  - 四真实实例路径、非五技能 CSV、源 manifest 不匹配或重复样本一律拒绝写入。

本脚本只生成离线数据；绝不用于 real_283/680/2338/3182 的监督训练。
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import configs
from core.action_completion import EarliestFinishActionCompleter
from environment import AirLineEnv_Graph
from scripts.audit_initial_team_opportunity_full import (
    _bounded_two_swaps,
    _configure,
    _enumerate_one_swaps,
    _forced_team_full_episode,
    _heuristic_features,
    _heuristic_finish_with,
    _select_pair,
    _workspace_path,
)

REAL_FOUR_INSTANCES = {"283.csv", "680.csv", "2338.csv", "3182.csv"}
SPLIT_PRETRAIN = "pretrain"
SPLIT_FROZEN_DIAGNOSTIC = "frozen_diagnostic"
SPLIT_PPO_ONLY = "ppo_only"
SPLIT_RATIO_PRETRAIN = 0.6  # 96/160
SPLIT_RATIO_FROZEN = 0.15   # 24/160
STATE_FRACTIONS = (0.125, 0.375, 0.625, 0.875)
ONE_SWAP_TOP_K = 2
TWO_SWAP_TOP_K = 2
TWO_SWAP_POOL = 24


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _split_by_sha256(files: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """按文件 CSV SHA-256 确定性排序后切分（96 预训练 / 24 冻结诊断 / 40 仅 PPO）。"""
    ordered = sorted(files, key=lambda item: str(item["sha256"]))
    total = len(ordered)
    n_pretrain = int(round(total * SPLIT_RATIO_PRETRAIN))
    n_frozen = int(round(total * SPLIT_RATIO_FROZEN))
    return {
        SPLIT_PRETRAIN: ordered[:n_pretrain],
        SPLIT_FROZEN_DIAGNOSTIC: ordered[n_pretrain : n_pretrain + n_frozen],
        SPLIT_PPO_ONLY: ordered[n_pretrain + n_frozen :],
    }


def _state_key(task_id: int, station_id: int, decision_count: int) -> str:
    return f"t{int(task_id)}_s{int(station_id)}_d{int(decision_count)}"


def _select_sample_states(
    decision_states: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """在"存在合法替代"的决策点中按分位取 4 个状态。"""
    if not decision_states:
        return []
    picked: list[dict[str, Any]] = []
    for fraction in STATE_FRACTIONS:
        target = fraction * (len(decision_states) - 1)
        index = min(int(round(target)), len(decision_states) - 1)
        state = decision_states[index]
        if all(state["task_id"] != p["task_id"] or state["station_id"] != p["station_id"] for p in picked):
            picked.append(state)
    return picked


def _hash_two_swap_representative(
    base_team: tuple[int, ...],
    legal_workers: list[int],
    state_seed: str,
    completer: EarliestFinishActionCompleter,
    obs: Any,
    task_id: int,
    station_id: int,
    demand: int,
    task_duration: float,
    features: dict[str, torch.Tensor],
) -> list[tuple[int, ...]]:
    """由状态哈希确定性选出的双替换池代表（非启发式排序，避免被局部排序绑架）。"""
    candidates = _bounded_two_swaps(
        base_team,
        legal_workers,
        completer=completer,
        obs=obs,
        task_id=task_id,
        station_id=station_id,
        demand=demand,
        task_duration=task_duration,
        pool_size=max(8, TWO_SWAP_POOL),
        features=features,
    )
    if not candidates:
        return []
    digest = int(_sha256_text(state_seed), 16)
    return [candidates[digest % len(candidates)]]


def _candidate_sources(
    base_team: tuple[int, ...],
    legal_workers: list[int],
    *,
    completer: EarliestFinishActionCompleter,
    obs: Any,
    task_id: int,
    station_id: int,
    demand: int,
    task_duration: float,
    state_seed: str,
) -> list[tuple[tuple[int, ...], str]]:
    """候选预算：锚点 + 2 单换 + 2 双换 + 1 哈希双换代表（去重保序）。"""
    features = _heuristic_features(completer, obs)
    results: list[tuple[tuple[int, ...], str]] = [(tuple(base_team), "anchor")]
    one_swaps = _enumerate_one_swaps(base_team, legal_workers)
    ranked_one = sorted(
        one_swaps,
        key=lambda team: _heuristic_finish_with(
            completer, station_id=station_id, team=team,
            demand=demand, task_duration=task_duration, features=features,
        ),
    )
    for team in ranked_one[:ONE_SWAP_TOP_K]:
        results.append((team, "one_swap"))
    two_swaps = _bounded_two_swaps(
        base_team, legal_workers, completer=completer, obs=obs,
        task_id=task_id, station_id=station_id, demand=demand,
        task_duration=task_duration, pool_size=TWO_SWAP_POOL, features=features,
    )
    for team in two_swaps[:TWO_SWAP_TOP_K]:
        results.append((team, "two_swap"))
    representative = _hash_two_swap_representative(
        base_team, legal_workers, state_seed, completer, obs,
        task_id, station_id, demand, task_duration, features,
    )
    for team in representative:
        results.append((team, "two_swap_hash"))
    deduped: list[tuple[tuple[int, ...], str]] = []
    seen: set[tuple[int, ...]] = set()
    for team, source in results:
        if team in seen:
            continue
        seen.add(team)
        deduped.append((team, source))
    return deduped


def _build_graph_samples(
    csv_path: Path,
    csv_sha256: str,
    manifest_sha256: str,
    output_dir: Path,
    completer: EarliestFinishActionCompleter,
    *,
    max_episode_steps: int,
    max_candidates: int,
) -> list[dict[str, Any]]:
    """单图：两遍确定性轨迹。第一遍收集"有合法替代"的决策点并取分位目标；
    第二遍在目标状态（env 正处于该决策点）构建候选反事实样本。"""
    decisions: list[dict[str, Any]] = []

    def _walk(collect_only: bool) -> list[dict[str, Any]]:
        env = AirLineEnv_Graph(data_path_or_dir=str(csv_path), seed=42)
        obs = env.reset(randomize_duration=False, randomize_workers=False, seed=42)
        collected: list[dict[str, Any]] = []
        decision_count = 0
        step = 0
        done = False
        while not done and step < max_episode_steps:
            selected = _select_pair(env, obs, completer, max_candidates=max_candidates)
            if selected is None:
                if not env.try_wait_for_resources():
                    break
                obs = env._get_observation()
                continue
            decision_count += 1
            if len(selected.candidates.teams) > 1:  # 存在合法替代团队
                record = {
                    "task_id": int(selected.task_id),
                    "station_id": int(selected.station_id),
                    "base_team": tuple(sorted(selected.candidates.teams[0])),
                    "decision_count": decision_count,
                }
                if collect_only:
                    collected.append(record)
                else:
                    record["obs"] = copy.deepcopy(obs)
                    record["masks"] = env.get_masks()
                    rows = _build_state_samples(
                        env,
                        record,
                        csv_path=csv_path,
                        csv_sha256=csv_sha256,
                        manifest_sha256=manifest_sha256,
                        output_dir=output_dir,
                        completer=completer,
                        max_episode_steps=max_episode_steps,
                        max_candidates=max_candidates,
                    )
                    collected.extend(rows)
            obs, _reward, done, info = env.step(
                (selected.task_id, selected.station_id, list(selected.candidates.teams[0]))
            )
            if "error" in info:
                raise RuntimeError(f"锚点轨迹动作被拒绝：{info['error']}")
            step += 1
            if done:
                break
        del env
        gc.collect()
        return collected

    decision_states = _walk(collect_only=True)
    targets = _select_sample_states(decision_states)
    if not targets:
        return []
    target_counts = {int(target["decision_count"]) for target in targets}
    # 第二遍：环境确定性重放，决策点顺序与第一遍完全一致
    sample_rows: list[dict[str, Any]] = []
    decision_count = 0
    env = AirLineEnv_Graph(data_path_or_dir=str(csv_path), seed=42)
    obs = env.reset(randomize_duration=False, randomize_workers=False, seed=42)
    step = 0
    done = False
    while not done and step < max_episode_steps:
        selected = _select_pair(env, obs, completer, max_candidates=max_candidates)
        if selected is None:
            if not env.try_wait_for_resources():
                break
            obs = env._get_observation()
            continue
        decision_count += 1
        if (
            decision_count in target_counts
            and len(selected.candidates.teams) > 1
        ):
            record = {
                "task_id": int(selected.task_id),
                "station_id": int(selected.station_id),
                "base_team": tuple(sorted(selected.candidates.teams[0])),
                "decision_count": decision_count,
                "obs": copy.deepcopy(obs),
                "masks": env.get_masks(),
            }
            rows = _build_state_samples(
                env,
                record,
                csv_path=csv_path,
                csv_sha256=csv_sha256,
                manifest_sha256=manifest_sha256,
                output_dir=output_dir,
                completer=completer,
                max_episode_steps=max_episode_steps,
                max_candidates=max_candidates,
            )
            sample_rows.extend(rows)
        obs, _reward, done, info = env.step(
            (selected.task_id, selected.station_id, list(selected.candidates.teams[0]))
        )
        if "error" in info:
            raise RuntimeError(f"锚点轨迹动作被拒绝：{info['error']}")
        step += 1
        if done:
            break
    del env
    gc.collect()
    return sample_rows


def _build_state_samples(
    env: AirLineEnv_Graph,
    state: dict[str, Any],
    *,
    csv_path: Path,
    csv_sha256: str,
    manifest_sha256: str,
    output_dir: Path,
    completer: EarliestFinishActionCompleter,
    max_episode_steps: int,
    max_candidates: int,
) -> list[dict[str, Any]]:
    """单状态（env 正处决策点）：候选预算 → 每候选一次完整续排 → 保存 npz。"""
    obs = state["obs"]
    task_id, station_id = state["task_id"], state["station_id"]
    base_team = state["base_team"]
    state_seed = (
        f"{csv_sha256}|{task_id}|{station_id}|{state['decision_count']}"
    )
    _task_mask, _station_mask, worker_mask = state["masks"]
    requirements = completer._extract_task_requirements(obs["task"].x, task_id)
    if requirements is None:
        return []
    required_skill, demand, task_duration = requirements
    legal_workers = [
        int(worker_id)
        for worker_id in completer._legal_worker_ids(
            obs["worker"].x,
            required_skill=required_skill,
            station_id=station_id,
            worker_mask=worker_mask,
        )
        if int(worker_id) not in set(base_team)
    ]
    if not legal_workers:
        return []
    candidates = _candidate_sources(
        base_team,
        legal_workers,
        completer=completer,
        obs=obs,
        task_id=task_id,
        station_id=station_id,
        demand=demand,
        task_duration=task_duration,
        state_seed=state_seed,
    )
    if len(candidates) < 2:
        return []
    results: list[tuple[tuple[int, ...], str, float]] = []
    baseline_makespan: float | None = None
    safe_seed = state_seed.replace("|", "-").replace("_", "-")
    obs_path = output_dir / "samples" / f"obs_{csv_sha256[:12]}_{safe_seed}.pt"
    torch.save(obs, obs_path)
    obs_rel = str(obs_path.relative_to(output_dir))
    for team, source in candidates:
        outcome = _forced_team_full_episode(
            env,
            task_id=task_id,
            station_id=station_id,
            team=team,
            completer=completer,
            max_candidates=max_candidates,
            max_episode_steps=max_episode_steps,
        )
        if not outcome["done"]:
            raise RuntimeError(
                f"反事实续排未完成：{csv_path.name} 状态 {state_seed} 候选 {team}"
            )
        makespan = float(outcome["makespan"])
        results.append((team, source, makespan))
        if source == "anchor":
            baseline_makespan = makespan
    if baseline_makespan is None or not math.isfinite(baseline_makespan):
        return []
    anchor_team = candidates[0][0]
    rows: list[dict[str, Any]] = []
    for team, source, makespan in results:
        relative_gain = (
            (baseline_makespan - makespan) / max(baseline_makespan, 1.0e-6)
            if baseline_makespan > 0.0
            else 0.0
        )
        row = {
            "task_id": int(task_id),
            "station_id": int(station_id),
            "state_seed": state_seed,
            "anchor_team": list(anchor_team),
            "candidate_team": list(team),
            "source": source,
            "baseline_makespan": float(baseline_makespan),
            "candidate_makespan": makespan,
            "relative_gain": float(relative_gain),
            "obs_pt": obs_rel,
            "sample_sha256": "",
            "npz_path": "",
        }
        row = _save_sample_npz(
            row,
            obs,
            masks=state["masks"],
            output_dir=output_dir,
            csv_sha256=csv_sha256,
            manifest_sha256=manifest_sha256,
            state_seed=state_seed,
        )
        rows.append(row)
    return rows


def _save_sample_npz(
    row: dict[str, Any],
    obs: Any,
    *,
    masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    output_dir: Path,
    csv_sha256: str,
    manifest_sha256: str,
    state_seed: str,
) -> dict[str, Any]:
    """保存 CPU 图观测特征 + 三类动作掩码 + 元数据为 npz，计算样本 SHA-256。"""
    task_mask, station_mask, worker_mask = masks
    arrays = {
        "task_x": np.asarray(obs["task"].x.detach().cpu().numpy(), dtype=np.float32),
        "worker_x": np.asarray(obs["worker"].x.detach().cpu().numpy(), dtype=np.float32),
        "station_x": np.asarray(obs["station"].x.detach().cpu().numpy(), dtype=np.float32),
        "task_mask": np.asarray(task_mask.detach().cpu().numpy(), dtype=np.bool_),
        "station_mask": np.asarray(station_mask.detach().cpu().numpy(), dtype=np.bool_),
        "worker_mask": np.asarray(worker_mask.detach().cpu().numpy(), dtype=np.bool_),
    }
    payload = json.dumps(
        {
            "csv_sha256": csv_sha256,
            "manifest_sha256": manifest_sha256,
            "state_seed": state_seed,
            "task_id": row["task_id"],
            "station_id": row["station_id"],
            "anchor_team": row["anchor_team"],
            "candidate_team": row["candidate_team"],
            "source": row["source"],
            "baseline_makespan": row["baseline_makespan"],
            "candidate_makespan": row["candidate_makespan"],
            "relative_gain": row["relative_gain"],
        },
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(payload)
    for name in ("task_x", "worker_x", "station_x", "task_mask", "station_mask", "worker_mask"):
        digest.update(np.ascontiguousarray(arrays[name]).tobytes())
    sample_sha = digest.hexdigest()
    row["sample_sha256"] = sample_sha
    safe_seed = state_seed.replace("|", "-").replace("_", "-")
    key = f"{csv_sha256[:12]}_{safe_seed}_{row['source']}_{sample_sha[:8]}"
    npz_path = output_dir / "samples" / f"{key}.npz"
    np.savez_compressed(npz_path, meta=payload, **arrays)
    row["npz_path"] = str(npz_path.relative_to(output_dir))
    return row


def _write_manifest(
    output_dir: Path,
    *,
    manifest_sha256: str,
    manifest_path: str,
    split: dict[str, list[dict[str, Any]]],
    sample_rows: list[dict[str, Any]],
    command_args: dict[str, Any],
) -> Path:
    """固化 split、图哈希、样本计数、候选预算与文件哈希。"""
    counts = {key: len(value) for key, value in split.items()}
    manifest = {
        "version": 1,
        "kind": "initial_anchor_proposal_counterfactual_v1",
        "source_manifest_path": manifest_path,
        "source_manifest_sha256": manifest_sha256,
        "split": {key: [str(item["sha256"]) for item in values] for key, values in split.items()},
        "split_counts": counts,
        "sample_counts": {
            key: sum(1 for row in sample_rows if row["split"] == key)
            for key in counts
        },
        "candidate_budget": {
            "one_swap_top_k": ONE_SWAP_TOP_K,
            "two_swap_top_k": TWO_SWAP_TOP_K,
            "two_swap_pool": TWO_SWAP_POOL,
            "hash_two_swap_representative": 1,
        },
        "state_fractions": list(STATE_FRACTIONS),
        "command_args": command_args,
        "files": [],
    }
    seen_samples: set[str] = set()
    for row in sample_rows:
        if row["sample_sha256"] in seen_samples:
            raise RuntimeError(f"重复样本拒绝写入：{row['sample_sha256']}")
        seen_samples.add(row["sample_sha256"])
        manifest["files"].append(
            {
                "csv_sha256": row["csv_sha256"],
                "split": row["split"],
                "state_seed": row["state_seed"],
                "source": row["source"],
                "sample_sha256": row["sample_sha256"],
                "npz": row["npz_path"],
                "obs_pt": row["obs_pt"],
                "relative_gain": row["relative_gain"],
            }
        )
    manifest_bytes = json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
    manifest["manifest_sha256"] = _sha256_bytes(manifest_bytes)
    manifest_path_out = output_dir / "manifest.json"
    manifest_path_out.write_text(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path_out


def _configure_global(data_path: Path) -> None:
    _configure(data_path)


def run_build(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = _workspace_path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest 不存在：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = _sha256_bytes(manifest_path.read_bytes())
    if str(manifest.get("protocol")) != "explicit_fiveskill_v1":
        raise ValueError("仅支持 explicit_fiveskill_v1（五技能）manifest")
    files = manifest.get("files", [])
    if not files:
        raise ValueError("manifest 无文件条目")
    split = _split_by_sha256(files)
    _configure_global(_workspace_path(args.data_file))
    completer = EarliestFinishActionCompleter(configs)

    output_dir = _workspace_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "samples").mkdir(exist_ok=False)

    command_args = {
        "manifest": args.manifest,
        "data_file": args.data_file,
        "output_dir": args.output_dir,
        "max_graphs": args.max_graphs,
        "seed": args.seed,
    }
    sample_rows: list[dict[str, Any]] = []
    built_graphs = 0
    for split_name, group in split.items():
        if split_name == SPLIT_PPO_ONLY:
            continue  # 仅 PPO 的图不生成反事实样本
        for item in group:
            if args.max_graphs > 0 and built_graphs >= args.max_graphs:
                break
            csv_sha256 = str(item["sha256"])
            if csv_sha256 in REAL_FOUR_INSTANCES or Path(item["file"]).name in REAL_FOUR_INSTANCES:
                raise ValueError(f"四真实实例禁止进入反事实集：{item['file']}")
            csv_path = _workspace_path(
                f"data/scale_400_800_datasets/{item['file']}"
            )
            if not csv_path.exists():
                raise FileNotFoundError(f"训练 CSV 缺失：{csv_path}")
            actual_sha = _sha256_bytes(csv_path.read_bytes())
            if actual_sha != csv_sha256:
                raise RuntimeError(
                    f"训练 CSV 哈希不匹配：{item['file']} manifest={csv_sha256} 实际={actual_sha}"
                )
            rows = _build_graph_samples(
                csv_path,
                csv_sha256=csv_sha256,
                manifest_sha256=manifest_sha,
                output_dir=output_dir,
                completer=completer,
                max_episode_steps=int(args.max_episode_steps),
                max_candidates=int(args.max_candidates),
            )
            for row in rows:
                row["csv_sha256"] = csv_sha256
                row["split"] = split_name
            sample_rows.extend(rows)
            built_graphs += 1
            print(
                f"[cf] {split_name} {item['file']} 样本 {len(rows)} "
                f"(累计 {len(sample_rows)})", flush=True
            )
            if args.max_graphs > 0 and built_graphs >= args.max_graphs:
                break
    manifest_out = _write_manifest(
        output_dir,
        manifest_sha256=manifest_sha,
        manifest_path=str(manifest_path),
        split=split,
        sample_rows=sample_rows,
        command_args=command_args,
    )
    print(f"[cf] 完成：{len(sample_rows)} 个样本，manifest {manifest_out}", flush=True)
    return {"samples": len(sample_rows), "manifest": str(manifest_out)}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="APCF 反事实预训练数据构建")
    parser.add_argument("--manifest", default="data/scale_400_800_datasets/manifest_ctg_160_explicit_fiveskill_v1.json")
    parser.add_argument("--data-file", default="data/680.csv")
    parser.add_argument("--output-dir", default="data/initial_anchor_proposal_cf_v1")
    parser.add_argument("--max-graphs", type=int, default=0, help="0=全部；>0 限制图数（smoke 用）")
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_build(parse_args())
