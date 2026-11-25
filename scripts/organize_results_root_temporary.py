"""归档 results 根目录中的已解析临时下载文件，并在核验后可选清理源文件。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STOCHASTIC_SOURCE = RESULTS / "01_initial_main_stochastic_20260723"
STOCHASTIC_AGGREGATE = RESULTS / "01_initial_main" / "stochastic_supplement_20260723"
LEGACY_FAILURE_TARGET = RESULTS / "90_legacy_and_smoke" / "local_only_repair_failure_evidence_20260721"
M2_FAILED_SOURCE = RESULTS / "screen_m2_scg_context_warmstart_seed42_20260724_111016"
M2_SOURCE = RESULTS / "screen_m2_scg_context_warmstart_seed42_20260724_111651"
M2_TARGET = RESULTS / "90_legacy_and_smoke" / "initial_main_screen_m2_scg_context_warmstart_seed42_20260724_111651"

METHODS = (
    "joint100_full_joint_seed42_20260719",
    "ablation_joint100_mean_max_pooling_seed42_20260720",
    "ablation_joint100_operation_only_seed42_20260720",
    "ablation_joint100_operation_station_seed42_20260720",
    "ablation_joint100_fixed_preallocation_seed42_20260720",
)


@dataclass(frozen=True)
class FileCheck:
    source: str
    target: str
    sha256: str
    bytes: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_under(path: Path) -> Iterable[Path]:
    return sorted(item for item in path.rglob("*") if item.is_file())


def copy_file_verified(source: Path, target: Path) -> list[FileCheck]:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)
    if not target.is_file() or sha256(source) != sha256(target):
        raise RuntimeError(f"文件哈希不一致：{source} -> {target}")
    return [FileCheck(str(source.relative_to(ROOT)), str(target.relative_to(ROOT)), sha256(source), source.stat().st_size)]


def copy_tree_verified(source: Path, target: Path, *, allow_target_extras: bool = False) -> list[FileCheck]:
    if not source.is_dir():
        raise FileNotFoundError(source)
    checks: list[FileCheck] = []
    for item in files_under(source):
        checks.extend(copy_file_verified(item, target / item.relative_to(source)))
    source_relatives = {item.relative_to(source) for item in files_under(source)}
    target_relatives = {item.relative_to(target) for item in files_under(target)} if target.exists() else set()
    extras = sorted(str(item) for item in target_relatives - source_relatives)
    if extras and not allow_target_extras:
        raise RuntimeError(f"目标目录包含不应混入的额外文件：{target} -> {extras[:5]}")
    return checks


def update_master_paths() -> int:
    master = RESULTS / "experiment_master_results.csv"
    with master.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if not fieldnames:
        raise RuntimeError("主表缺少表头")
    changed = 0
    old_prefix = "results/01_initial_main_stochastic_20260723/"
    for row in rows:
        current = row.get("source_file", "")
        if not current.startswith(old_prefix):
            continue
        for method in METHODS:
            old = f"{old_prefix}{method}/eval/initial_sixrun_20260722_stochastic"
            new = f"results/01_initial_main/{method}/eval/initial_sixrun_20260722_stochastic"
            if current.startswith(old):
                row["source_file"] = current.replace(old, new, 1)
                changed += 1
                break
    with master.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remove-sources", action="store_true", help="仅在全部哈希核验通过后删除根目录临时源文件")
    args = parser.parse_args()

    all_checks: list[FileCheck] = []
    for method in METHODS:
        source = STOCHASTIC_SOURCE / method / "eval" / "initial_sixrun_20260722_stochastic"
        target = RESULTS / "01_initial_main" / method / "eval" / "initial_sixrun_20260722_stochastic"
        all_checks.extend(copy_tree_verified(source, target))

    aggregate_files = {
        "file_manifest_all.json": "source_file_manifest_all.json",
        "initial_stochastic_supplement_runs_detail.csv": "runs_detail.csv",
        "initial_stochastic_supplement_summary.csv": "summary.csv",
        "initial_stochastic_supplement_summary.json": "summary.json",
        "README.md": "README_source_download.md",
        "README_initial_stochastic_supplement_20260723.md": "README_root_index_source_download.md",
    }
    for source_name, target_name in aggregate_files.items():
        source = RESULTS / source_name if source_name.startswith("README_initial") else STOCHASTIC_SOURCE / source_name
        all_checks.extend(copy_file_verified(source, STOCHASTIC_AGGREGATE / target_name))

    all_checks.extend(
        copy_tree_verified(
            STOCHASTIC_SOURCE / "local_only_repair_failure_evidence_20260721",
            LEGACY_FAILURE_TARGET,
        )
    )

    # M2 主训练下载目录此前已归档；这里对源文件逐一反向复核，不重复复制 checkpoint。
    all_checks.extend(copy_tree_verified(M2_SOURCE, M2_TARGET, allow_target_extras=True))
    all_checks.extend(
        copy_tree_verified(
            M2_FAILED_SOURCE,
            M2_TARGET / "provenance" / "failed_launch_20260724_111016",
            allow_target_extras=True,
        )
    )

    changed_rows = update_master_paths()
    audit = {
        "purpose": "results 根目录临时下载文件归档与 SHA-256 核验",
        "status": "passed",
        "verified_file_count": len(all_checks),
        "master_source_file_rows_relinked": changed_rows,
        "checks": [asdict(check) for check in all_checks],
        "sources_pending_removal": [
            str(STOCHASTIC_SOURCE.relative_to(ROOT)),
            str(M2_SOURCE.relative_to(ROOT)),
            str(M2_FAILED_SOURCE.relative_to(ROOT)),
            str((RESULTS / "README_initial_stochastic_supplement_20260723.md").relative_to(ROOT)),
        ],
    }
    STOCHASTIC_AGGREGATE.mkdir(parents=True, exist_ok=True)
    (STOCHASTIC_AGGREGATE / "temporary_relocation_audit_20260724.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (STOCHASTIC_AGGREGATE / "README.md").write_text(
        "# 初始调度随机补充验证（规范归档）\n\n"
        "本目录保存 2026-07-23 下载到 results 根目录的五种方法随机补充验证汇总。"
        "逐方法的原始逐场景输出位于各方法目录的 `eval/initial_sixrun_20260722_stochastic/`。\n\n"
        "- 方案：temperature=0.01，seed=42–46，四个真实实例；共 100 个 schedule。\n"
        "- 原始审计结论：100/100 完整且合法，硬约束违规为 0。\n"
        "- 证据等级：`conditional`，原因是原始逐次 run manifest 未独立保存 CLI temperature。\n"
        "- 完整迁移核验见 `temporary_relocation_audit_20260724.json`。\n",
        encoding="utf-8",
    )

    if args.remove_sources:
        for path in (STOCHASTIC_SOURCE, M2_SOURCE, M2_FAILED_SOURCE):
            if not path.is_dir() or RESULTS not in path.parents:
                raise RuntimeError(f"拒绝删除非 results 根目录临时目录：{path}")
            shutil.rmtree(path)
        root_readme = RESULTS / "README_initial_stochastic_supplement_20260723.md"
        if not root_readme.is_file():
            raise RuntimeError(f"未找到待清理的临时索引：{root_readme}")
        root_readme.unlink()

    print(json.dumps({"status": "passed", "verified_file_count": len(all_checks), "master_rows_relinked": changed_rows, "removed": args.remove_sources}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
