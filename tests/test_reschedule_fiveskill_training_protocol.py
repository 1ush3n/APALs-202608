from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from runtime.reschedule_manifest import (
    load_reschedule_manifest,
    resolve_explicit_five_skill_training_paths,
    validate_explicit_five_skill_training_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_graph(path: Path, skills: tuple[int, ...] = (0, 1, 2, 3, 4)) -> None:
    fields = ["序号", "AO号", "类型", "专业编码", "工种", "紧前工序AO号", "需求人数", "加工时间/h", "限定站位", "部位容量"]
    codes = ("A", "D", "B", "N", "C")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"序号": 1, "AO号": "A", "类型": 1, "专业编码": "", "工种": -1, "紧前工序AO号": "", "需求人数": 0, "加工时间/h": 0, "限定站位": "", "部位容量": ""})
        for index, skill in enumerate(skills, start=2):
            code = codes[skill]
            writer.writerow({"序号": index, "AO号": f"R{code}{index:04d}", "类型": 2, "专业编码": code, "工种": skill, "紧前工序AO号": "A", "需求人数": 1, "加工时间/h": 1, "限定站位": "", "部位容量": ""})


def _entry(instance_id: str, split: str, graph: Path, baseline: Path, scenario: Path | None = None) -> dict[str, str]:
    row = {
        "instance_id": instance_id, "split": split, "source": "generated_fiveskill", "data_path": str(graph),
        "baseline_schedule_path": str(baseline), "data_sha256": _sha256(graph),
        "baseline_sha256": _sha256(baseline), "status": "ready",
    }
    if scenario is not None:
        row["scenario_path"] = str(scenario)
        row["scenario_sha256"] = _sha256(scenario)
    return row


def _write_manifest(root: Path, *, protocol: str = "explicit_fiveskill_v1", skills: tuple[int, ...] = (0, 1, 2, 3, 4)) -> Path:
    train = root / "train"
    train.mkdir()
    rows: list[dict[str, str]] = []
    for index in range(30):
        graph, baseline = train / f"train_{index:02d}.csv", root / f"baseline_{index:02d}.csv"
        _write_graph(graph, skills)
        baseline.write_text("TaskID,StationID,Team,Start,End\n", encoding="utf-8")
        rows.append(_entry(f"train_{index:04d}", "train", graph, baseline))
    for name in ("real_283", "real_680", "real_2338", "real_3182"):
        graph, baseline, scenario = root / f"{name}.csv", root / f"{name}_baseline.csv", root / f"{name}_scenario.csv"
        _write_graph(graph, skills)
        baseline.write_text("TaskID,StationID,Team,Start,End\n", encoding="utf-8")
        scenario.write_text("scenario_id,level,TaskID,release_time\nlow_000,low,RA0002,0\n", encoding="utf-8")
        rows.append(_entry(name, "eval", graph, baseline, scenario))
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps({"protocol": protocol, "instances": rows, "skipped": []}, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def test_protocol_accepts_complete_explicit_fiveskill_assets(tmp_path: Path) -> None:
    manifest = load_reschedule_manifest(_write_manifest(tmp_path))
    validate_explicit_five_skill_training_manifest(manifest)
    assert len(resolve_explicit_five_skill_training_paths(manifest, tmp_path / "train")) == 30


def test_protocol_rejects_missing_skill_coverage(tmp_path: Path) -> None:
    manifest = load_reschedule_manifest(_write_manifest(tmp_path, skills=(0, 1, 2, 3)))
    with pytest.raises(ValueError, match="未完整覆盖五技能"):
        validate_explicit_five_skill_training_manifest(manifest)


def test_protocol_rejects_legacy_protocol(tmp_path: Path) -> None:
    manifest = load_reschedule_manifest(_write_manifest(tmp_path, protocol="r4_fiveskill_explicit_training_v1"))
    with pytest.raises(ValueError, match="explicit_fiveskill_v1"):
        validate_explicit_five_skill_training_manifest(manifest)


def test_training_directory_rejects_extra_csv(tmp_path: Path) -> None:
    manifest = load_reschedule_manifest(_write_manifest(tmp_path))
    _write_graph(tmp_path / "train" / "extra.csv")
    with pytest.raises(ValueError, match="精确文件列表不一致"):
        resolve_explicit_five_skill_training_paths(manifest, tmp_path / "train")
