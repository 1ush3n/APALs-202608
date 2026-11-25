from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from runtime.paths import PROJECT_ROOT, resolve_workspace_path
from runtime.five_skill_schema import (
    ALLOWED_PRODUCTION_PROTOCOLS,
    EXPLICIT_FIVE_SKILL_PROTOCOL,
    validate_explicit_five_skill_csv,
)
from utils.reschedule import load_baseline_schedule, load_reschedule_scenarios
from utils.reschedule_r5 import validate_r5_scenario_library


REAL_INSTANCE_IDS = ("real_283", "real_680", "real_2338", "real_3182")
R5_RESCHEDULE_PROTOCOL = "r5_task_delay_v1"


@dataclass(frozen=True)
class RescheduleManifestEntry:
    """重调度数据清单中的一个可用实例。"""

    instance_id: str
    split: str
    data_path: Path
    baseline_schedule_path: Path
    scenario_path: Path | None = None
    num_tasks: int | None = None
    baseline_makespan: float | None = None
    status: str = "ready"
    source: str = ""
    data_sha256: str = ""
    baseline_sha256: str = ""
    scenario_sha256: str = ""
    scenario_metadata_path: Path | None = None
    scenario_metadata_sha256: str = ""


@dataclass(frozen=True)
class RescheduleManifest:
    """重调度训练/评估使用的数据、baseline 和场景映射。"""

    path: Path
    payload: dict[str, Any]
    entries: tuple[RescheduleManifestEntry, ...]

    def ready_entries(self) -> tuple[RescheduleManifestEntry, ...]:
        return tuple(entry for entry in self.entries if entry.status == "ready")

    def get(self, instance_id: str) -> RescheduleManifestEntry:
        for entry in self.ready_entries():
            if entry.instance_id == instance_id:
                return entry
        raise KeyError(f"manifest 中未找到可用实例: {instance_id}")

    def find_by_data_path(self, data_path: str | Path) -> RescheduleManifestEntry:
        target = resolve_workspace_path(data_path).resolve()
        matches = [entry for entry in self.ready_entries() if entry.data_path.resolve() == target]
        if not matches:
            raise KeyError(f"manifest 中没有匹配数据集的 baseline: {target}")
        if len(matches) > 1:
            ids = [entry.instance_id for entry in matches]
            raise ValueError(f"manifest 中数据集路径重复，无法唯一匹配: {target} -> {ids}")
        return matches[0]

    def filter(self, *, split: str | None = None, source: str | None = None) -> tuple[RescheduleManifestEntry, ...]:
        entries: Iterable[RescheduleManifestEntry] = self.ready_entries()
        if split is not None:
            entries = (entry for entry in entries if entry.split == split)
        if source is not None:
            entries = (entry for entry in entries if entry.source == source)
        return tuple(entries)


def _resolve_manifest_path(value: str | Path | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return resolve_workspace_path(value)


def _optional_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return resolve_workspace_path(value)


@lru_cache(maxsize=8)
def _load_reschedule_manifest_cached(path_key: str) -> RescheduleManifest:
    manifest_path = Path(path_key)
    if not manifest_path.exists():
        raise FileNotFoundError(f"重调度 manifest 不存在: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_entries = payload.get("instances", [])
    if not isinstance(raw_entries, list):
        raise ValueError("重调度 manifest 的 instances 必须是列表")

    entries: list[RescheduleManifestEntry] = []
    for row in raw_entries:
        if not isinstance(row, dict):
            raise ValueError(f"manifest 实例必须是对象: {row!r}")
        status = str(row.get("status", "ready"))
        entry = RescheduleManifestEntry(
            instance_id=str(row["instance_id"]),
            split=str(row.get("split", "")),
            data_path=resolve_workspace_path(row["data_path"]),
            baseline_schedule_path=resolve_workspace_path(row["baseline_schedule_path"]),
            scenario_path=_optional_path(row.get("scenario_path")),
            num_tasks=None if row.get("num_tasks") is None else int(row["num_tasks"]),
            baseline_makespan=None if row.get("baseline_makespan") is None else float(row["baseline_makespan"]),
            status=status,
            source=str(row.get("source", "")),
            data_sha256=str(row.get("data_sha256", "")).strip().lower(),
            baseline_sha256=str(row.get("baseline_sha256", "")).strip().lower(),
            scenario_sha256=str(row.get("scenario_sha256", "")).strip().lower(),
            scenario_metadata_path=_optional_path(row.get("scenario_metadata_path")),
            scenario_metadata_sha256=str(row.get("scenario_metadata_sha256", "")).strip().lower(),
        )
        entries.append(entry)
    return RescheduleManifest(path=manifest_path, payload=payload, entries=tuple(entries))


def load_reschedule_manifest(path: str | Path) -> RescheduleManifest:
    manifest_path = resolve_workspace_path(path).resolve()
    return _load_reschedule_manifest_cached(str(manifest_path))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_asset_hash(entry: RescheduleManifestEntry, *, field: str, path: Path) -> None:
    expected = str(getattr(entry, field)).strip().lower()
    if not expected:
        raise ValueError(f"{entry.instance_id} 缺少正式协议哈希字段 {field}")
    actual = _sha256(path)
    if actual != expected:
        # 兼容跨平台 (Windows CRLF <-> Linux LF) 的换行符归一化 SHA-256 校验
        content = path.read_bytes()
        crlf_hash = hashlib.sha256(content.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")).hexdigest()
        lf_hash = hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()
        if expected in (crlf_hash, lf_hash):
            return
        raise ValueError(f"{entry.instance_id} 的 {field} 与文件 SHA-256 不一致 (actual={actual}, expected={expected})")


def validate_explicit_five_skill_training_manifest(manifest: RescheduleManifest) -> None:
    """拒绝将旧单技能资产用于正式五技能重调度训练。

    这是训练入口的最终防线。r4 构建阶段已完成更完整的 DAG、映射和基准排程
    审计；这里以低开销方式复核 protocol、30 个训练条目以及每张训练图的显式
    语义字段和五工种覆盖，防止历史 DataLoader 兼容回退再次静默生效。
    """
    protocol = str(manifest.payload.get("protocol", "")).strip()
    if protocol not in ALLOWED_PRODUCTION_PROTOCOLS:
        raise ValueError(
            f"正式重调度训练只接受 {sorted(ALLOWED_PRODUCTION_PROTOCOLS)} manifest；"
            f"当前 protocol={protocol or '<missing>'!r}。历史 r3/无 protocol 资产禁止用于训练。"
        )
    train_entries = tuple(entry for entry in manifest.ready_entries() if entry.split == "train")
    if len(train_entries) != 30:
        raise ValueError(f"正式协议要求 30 个 ready 训练图，实际为 {len(train_entries)}")
    ids = [entry.instance_id for entry in train_entries]
    if len(set(ids)) != len(ids):
        raise ValueError("训练 manifest 存在重复 instance_id")

    real_entries = tuple(entry for entry in manifest.ready_entries() if entry.instance_id in REAL_INSTANCE_IDS)
    if tuple(entry.instance_id for entry in real_entries) != REAL_INSTANCE_IDS:
        raise ValueError(f"正式协议必须精确包含四个真实实例 {REAL_INSTANCE_IDS}")
    if len(manifest.ready_entries()) != 34 or manifest.payload.get("skipped", []) not in ([], None):
        raise ValueError("正式协议必须恰有 34 个 ready 条目且零 skipped")

    for entry in (*train_entries, *real_entries):
        if not entry.data_path.is_file():
            raise FileNotFoundError(f"manifest 数据图不存在: {entry.data_path}")
        if not entry.baseline_schedule_path.is_file():
            raise FileNotFoundError(f"manifest 基准排程不存在: {entry.baseline_schedule_path}")
        _validate_asset_hash(entry, field="data_sha256", path=entry.data_path)
        _validate_asset_hash(entry, field="baseline_sha256", path=entry.baseline_schedule_path)
        validate_explicit_five_skill_csv(entry.data_path, require_all_skills=True)
        if entry.split == "eval":
            if entry.scenario_path is None or not entry.scenario_path.is_file():
                raise FileNotFoundError(f"{entry.instance_id} 缺少真实实例场景文件")
            _validate_asset_hash(entry, field="scenario_sha256", path=entry.scenario_path)


def resolve_explicit_five_skill_training_paths(
    manifest: RescheduleManifest,
    configured_data_path: str | Path,
) -> tuple[Path, ...]:
    """验证目录与 manifest 完全一致，并返回唯一允许运行的有序训练文件。"""
    validate_explicit_five_skill_training_manifest(manifest)
    directory = resolve_workspace_path(configured_data_path).resolve()
    if not directory.is_dir():
        raise ValueError("正式重调度训练的 train_data_path_or_dir 必须是 manifest 训练图所在目录")
    declared = tuple(entry.data_path.resolve() for entry in manifest.filter(split="train"))
    discovered = tuple(sorted(path.resolve() for path in directory.iterdir() if path.suffix.lower() == ".csv"))
    if len(set(declared)) != len(declared):
        raise ValueError("manifest 训练数据路径重复")
    if discovered != tuple(sorted(declared)):
        extra = sorted(str(path) for path in set(discovered) - set(declared))
        missing = sorted(str(path) for path in set(declared) - set(discovered))
        raise ValueError(f"训练目录与 manifest 精确文件列表不一致: extra={extra}; missing={missing}")
    return declared


def get_configured_reschedule_manifest(config_obj: Any) -> RescheduleManifest | None:
    path = _resolve_manifest_path(getattr(config_obj, "reschedule_manifest_path", ""))
    if path is None:
        return None
    return load_reschedule_manifest(path)


def resolve_manifest_entry_for_data(config_obj: Any, data_path: str | Path) -> RescheduleManifestEntry | None:
    manifest = get_configured_reschedule_manifest(config_obj)
    if manifest is None:
        return None
    return manifest.find_by_data_path(data_path)


def resolve_manifest_eval_entry(config_obj: Any) -> RescheduleManifestEntry | None:
    manifest = get_configured_reschedule_manifest(config_obj)
    if manifest is None:
        return None
    instance_id = str(getattr(config_obj, "reschedule_eval_instance_id", "") or "").strip()
    if instance_id:
        return manifest.get(instance_id)
    real_entries = manifest.filter(split="eval", source="real")
    if not real_entries:
        real_entries = manifest.filter(source="real")
    if not real_entries:
        return None
    return real_entries[0]


def validate_r5_manifest_shape(manifest: RescheduleManifest) -> None:
    """校验 r5 的 24/6/4 数据划分，不读取具体文件内容。"""
    protocol = str(manifest.payload.get("reschedule_protocol", "")).strip()
    if protocol != R5_RESCHEDULE_PROTOCOL:
        raise ValueError("manifest 的 reschedule_protocol 必须是 r5_task_delay_v1")
    entries = manifest.ready_entries()
    train_entries = tuple(entry for entry in entries if entry.split == "train")
    validation_entries = tuple(entry for entry in entries if entry.split == "validation")
    eval_entries = tuple(entry for entry in entries if entry.split == "eval")
    if (len(train_entries), len(validation_entries), len(eval_entries)) != (24, 6, 4):
        raise ValueError("r5 manifest 必须包含 24 train、6 validation、4 eval 条目")
    if tuple(entry.instance_id for entry in eval_entries) != REAL_INSTANCE_IDS:
        raise ValueError(f"r5 manifest 的 eval 条目必须是 {REAL_INSTANCE_IDS}")
    if any(entry.instance_id in REAL_INSTANCE_IDS for entry in (*train_entries, *validation_entries)):
        raise ValueError("真实实例不得进入 train 或 validation split")
    ids = [entry.instance_id for entry in entries]
    if len(set(ids)) != len(ids):
        raise ValueError("r5 manifest 存在重复 instance_id")


def validate_r5_manifest_assets(manifest: RescheduleManifest) -> None:
    """校验 r5 训练、验证和真实测试资产及其哈希。"""
    validate_r5_manifest_shape(manifest)
    for entry in manifest.ready_entries():
        if not entry.data_path.is_file():
            raise FileNotFoundError(f"r5 实例数据不存在: {entry.data_path}")
        if not entry.baseline_schedule_path.is_file():
            raise FileNotFoundError(f"r5 baseline 不存在: {entry.baseline_schedule_path}")
        _validate_asset_hash(entry, field="data_sha256", path=entry.data_path)
        _validate_asset_hash(entry, field="baseline_sha256", path=entry.baseline_schedule_path)
        validate_explicit_five_skill_csv(entry.data_path, require_all_skills=True)
        if entry.split in {"validation", "eval"}:
            if entry.scenario_path is None or not entry.scenario_path.is_file():
                raise FileNotFoundError(f"{entry.instance_id} 缺少 r5 场景文件")
            _validate_asset_hash(entry, field="scenario_sha256", path=entry.scenario_path)
            if entry.scenario_metadata_path is None or not entry.scenario_metadata_path.is_file():
                raise FileNotFoundError(f"{entry.instance_id} 缺少 r5 场景元数据")
            _validate_asset_hash(
                entry,
                field="scenario_metadata_sha256",
                path=entry.scenario_metadata_path,
            )
            metadata = json.loads(
                entry.scenario_metadata_path.read_text(encoding="utf-8-sig")
            )
            if str(metadata.get("baseline_sha256", "")).strip().lower() != entry.baseline_sha256:
                raise ValueError(f"{entry.instance_id} 的场景元数据 baseline 哈希不匹配")
            validate_r5_scenario_library(
                load_baseline_schedule(entry.baseline_schedule_path),
                metadata,
                instance_id=entry.instance_id,
            )
            scenario_rows = load_reschedule_scenarios(entry.scenario_path)
            if len(scenario_rows) != 9:
                raise ValueError(f"{entry.instance_id} 的 r5 场景数量必须为 9")


def resolve_r5_training_paths(
    manifest: RescheduleManifest,
    configured_data_path: str | Path,
) -> tuple[Path, ...]:
    """验证 r5 train 目录只包含 24 个训练图，并返回其有序路径。"""
    validate_r5_manifest_assets(manifest)
    directory = resolve_workspace_path(configured_data_path).resolve()
    if not directory.is_dir():
        raise ValueError("r5 train_data_path_or_dir 必须是 24 个训练图所在目录")
    declared = tuple(entry.data_path.resolve() for entry in manifest.filter(split="train"))
    discovered = tuple(sorted(path.resolve() for path in directory.iterdir() if path.suffix.lower() == ".csv"))
    if discovered != tuple(sorted(declared)):
        extra = sorted(str(path) for path in set(discovered) - set(declared))
        missing = sorted(str(path) for path in set(declared) - set(discovered))
        raise ValueError(f"r5 训练目录与 manifest 不一致: extra={extra}; missing={missing}")
    return declared


def to_manifest_path(path: str | Path) -> str:
    resolved = resolve_workspace_path(path)
    try:
        return resolved.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved.resolve())


__all__ = [
    "RescheduleManifest",
    "RescheduleManifestEntry",
    "get_configured_reschedule_manifest",
    "load_reschedule_manifest",
    "resolve_explicit_five_skill_training_paths",
    "resolve_r5_training_paths",
    "validate_explicit_five_skill_training_manifest",
    "validate_r5_manifest_assets",
    "resolve_manifest_entry_for_data",
    "resolve_manifest_eval_entry",
    "validate_r5_manifest_shape",
    "to_manifest_path",
]
