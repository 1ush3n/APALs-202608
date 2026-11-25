"""主方法初始调度 checkpoint 多实例选择协议。

该模块只服务于 ``checkpoint_selection_protocol=multiscale_manifest``。
默认的单实例异步验证和所有基线/消融实验均不经过这里。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runtime.paths import resolve_workspace_path


@dataclass(frozen=True)
class InitialSelectionEntry:
    """一个固定初始调度选择实例。"""

    instance_id: str
    data_path: Path
    sha256: str
    reference_makespan: float

    def as_job_payload(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "data_path": str(self.data_path),
            "sha256": self.sha256,
            "reference_makespan": self.reference_makespan,
        }


@dataclass(frozen=True)
class InitialCheckpointSelectionManifest:
    """训练期多实例选择的不可变输入快照。"""

    path: Path
    sha256: str
    protocol_id: str
    role: str
    temperature: float
    seed: int
    entries: tuple[InitialSelectionEntry, ...]

    def as_job_payload(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "manifest_path": str(self.path),
            "manifest_sha256": self.sha256,
            "protocol_id": self.protocol_id,
            "role": self.role,
            "temperature": self.temperature,
            "seed": self.seed,
            "instances": [entry.as_job_payload() for entry in self.entries],
        }


def sha256_file(path: Path) -> str:
    """返回文件 SHA-256；使用流式读取以支持大数据文件。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_normalized_text_file(path: Path) -> str:
    """返回以 LF 规范化换行后的文本文件 SHA-256。"""
    content = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def _require_mapping(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} 必须是对象")
    return value


def load_initial_checkpoint_selection_manifest(
    path_like: str | Path,
) -> InitialCheckpointSelectionManifest:
    """加载并严格校验主方法四 real 实例选择 manifest。"""
    path = resolve_workspace_path(path_like).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"初始调度 checkpoint 选择 manifest 不存在: {path}")
    raw = _require_mapping(json.loads(path.read_text(encoding="utf-8")), "manifest")
    if int(raw.get("format_version", 0)) != 1:
        raise ValueError("初始调度 checkpoint 选择 manifest format_version 必须为 1")
    protocol_id = str(raw.get("protocol_id", "")).strip()
    role = str(raw.get("role", "")).strip()
    evaluation = _require_mapping(raw.get("evaluation"), "evaluation")
    temperature = float(evaluation.get("temperature", float("nan")))
    if temperature != 0.0:
        raise ValueError("checkpoint 选择 manifest 必须使用确定性 temperature=0.0")
    seed = int(evaluation.get("seed", -1))
    if seed < 0:
        raise ValueError("checkpoint 选择 manifest 缺少非负 evaluation.seed")
    raw_entries = raw.get("instances")
    if not isinstance(raw_entries, list) or len(raw_entries) != 4:
        raise ValueError("第一阶段 checkpoint 选择 manifest 必须恰好包含 4 个实例")

    entries: list[InitialSelectionEntry] = []
    seen_ids: set[str] = set()
    for index, raw_entry in enumerate(raw_entries):
        entry = _require_mapping(raw_entry, f"instances[{index}]")
        instance_id = str(entry.get("instance_id", "")).strip()
        relative_path = str(entry.get("data_path", "")).strip()
        expected_hash = str(entry.get("sha256", "")).strip().lower()
        reference = float(entry.get("reference_makespan", float("nan")))
        if not instance_id or instance_id in seen_ids:
            raise ValueError(f"instances[{index}] 的 instance_id 缺失或重复: {instance_id!r}")
        if not relative_path:
            raise ValueError(f"instances[{index}] 缺少 data_path")
        if len(expected_hash) != 64 or any(char not in "0123456789abcdef" for char in expected_hash):
            raise ValueError(f"instances[{index}] 的 sha256 非法")
        if not reference > 0.0:
            raise ValueError(f"instances[{index}] 的 reference_makespan 必须大于 0")
        data_path = resolve_workspace_path(relative_path).resolve()
        if not data_path.is_file():
            raise FileNotFoundError(f"选择实例数据不存在: {data_path}")
        actual_hash = sha256_normalized_text_file(data_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"选择实例数据哈希不一致: {instance_id}; "
                f"expected={expected_hash}, actual={actual_hash}"
            )
        seen_ids.add(instance_id)
        entries.append(
            InitialSelectionEntry(
                instance_id=instance_id,
                data_path=data_path,
                sha256=actual_hash,
                reference_makespan=reference,
            )
        )

    required = {"real_283", "real_680", "real_2338", "real_3182"}
    if seen_ids != required:
        raise ValueError(f"第一阶段选择实例必须为 {sorted(required)}，实际为 {sorted(seen_ids)}")
    if not protocol_id or not role:
        raise ValueError("checkpoint 选择 manifest 缺少 protocol_id 或 role")
    return InitialCheckpointSelectionManifest(
        path=path,
        sha256=sha256_file(path),
        protocol_id=protocol_id,
        role=role,
        temperature=temperature,
        seed=seed,
        entries=tuple(entries),
    )


def parse_job_selection_manifest(payload: object) -> InitialCheckpointSelectionManifest:
    """从已入队任务快照重建 manifest，不重新读取可变源文件。"""
    raw = _require_mapping(payload, "selection_manifest")
    raw_entries = raw.get("instances")
    if not isinstance(raw_entries, list) or len(raw_entries) != 4:
        raise ValueError("异步任务中的 selection_manifest 必须包含 4 个实例")
    entries: list[InitialSelectionEntry] = []
    for index, raw_entry in enumerate(raw_entries):
        entry = _require_mapping(raw_entry, f"selection_manifest.instances[{index}]")
        data_path = Path(str(entry["data_path"])).resolve()
        if not data_path.is_file():
            raise FileNotFoundError(f"异步任务选择数据不存在: {data_path}")
        expected_hash = str(entry["sha256"]).lower()
        actual_hash = sha256_normalized_text_file(data_path)
        if actual_hash != expected_hash:
            raise ValueError(f"异步任务选择数据哈希不一致: {data_path}")
        entries.append(
            InitialSelectionEntry(
                instance_id=str(entry["instance_id"]),
                data_path=data_path,
                sha256=actual_hash,
                reference_makespan=float(entry["reference_makespan"]),
            )
        )
    return InitialCheckpointSelectionManifest(
        path=Path(str(raw["manifest_path"])).resolve(),
        sha256=str(raw["manifest_sha256"]),
        protocol_id=str(raw["protocol_id"]),
        role=str(raw["role"]),
        temperature=float(raw["temperature"]),
        seed=int(raw["seed"]),
        entries=tuple(entries),
    )


__all__ = [
    "InitialCheckpointSelectionManifest",
    "InitialSelectionEntry",
    "load_initial_checkpoint_selection_manifest",
    "parse_job_selection_manifest",
    "sha256_file",
    "sha256_normalized_text_file",
]
