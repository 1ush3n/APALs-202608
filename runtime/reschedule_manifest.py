from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from runtime.paths import PROJECT_ROOT, resolve_workspace_path


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
        )
        entries.append(entry)
    return RescheduleManifest(path=manifest_path, payload=payload, entries=tuple(entries))


def load_reschedule_manifest(path: str | Path) -> RescheduleManifest:
    manifest_path = resolve_workspace_path(path).resolve()
    return _load_reschedule_manifest_cached(str(manifest_path))


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
    "resolve_manifest_entry_for_data",
    "resolve_manifest_eval_entry",
    "to_manifest_path",
]
