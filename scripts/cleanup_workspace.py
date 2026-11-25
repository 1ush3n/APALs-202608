"""
工作区临时文件、缓存及 Smoke 测试产物自动化归档与清理脚本
遵循原则：
1. 纯临时缓存（pytest, pycache, tmp）直接删除
2. 具有调试与回溯价值的日志、smoke结果、runs产物先打包至 archive/*.zip 并校验完整性，再安全删除原文件
3. 记录完整的操作审计账本至 archive/cleanup_record_20260912.json
"""

import argparse
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT_DIR = Path(r"d:\APAL-Dynamic-v4")
ARCHIVE_DIR = ROOT_DIR / "archive"


def fmt_size(num):
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num) < 1024.0:
            return f"{num:3.2f} {unit}"
        num /= 1024.0
    return f"{num:.2f} TB"


def get_dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for r, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(r, f)
            try:
                total += os.path.getsize(fp)
            except Exception:
                pass
    return total


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return 1
    total = 0
    for _, _, files in os.walk(path):
        total += len(files)
    return total


def create_zip_archive(zip_path: Path, items_to_archive: list[tuple[Path, str]]) -> bool:
    """
    items_to_archive: list of (source_path, arcname_in_zip)
    Returns True if archive created and passed testzip()
    """
    print(f"  [ARCHIVE] 创建压缩包: {zip_path.relative_to(ROOT_DIR)} ...")
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for src, arcname in items_to_archive:
            if not src.exists():
                continue
            if src.is_file():
                zf.write(src, arcname)
            else:
                for root, _, files in os.walk(src):
                    for file in files:
                        full_p = Path(root) / file
                        rel_to_src = full_p.relative_to(src)
                        inner_arcname = f"{arcname}/{rel_to_src}".replace("\\", "/")
                        zf.write(full_p, inner_arcname)

    # 校验完整性
    with zipfile.ZipFile(zip_path, "r") as zf:
        corrupt = zf.testzip()
        if corrupt is not None:
            raise RuntimeError(f"压缩包校验损坏: {zip_path}, 损坏文件: {corrupt}")
    
    zip_size = zip_path.stat().st_size
    print(f"  [ARCHIVE] 完整性校验通过: {zip_path.name} (大小: {fmt_size(zip_size)})")
    return True


def _remove_readonly(func, path, excinfo):
    import stat
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def safe_remove_path(path: Path, dry_run: bool = False):
    import stat
    if not path.exists():
        return
    if dry_run:
        print(f"    [DRY-RUN] 将删除: {path.relative_to(ROOT_DIR)}")
        return
    if path.is_file():
        try:
            os.chmod(path, stat.S_IWRITE)
        except Exception:
            pass
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path, onerror=_remove_readonly)


def run_cleanup(dry_run: bool = False, include_legacy_smoke: bool = True):
    print("=" * 70)
    print(f"开始工作区清理与归档任务 | DRY_RUN = {dry_run}")
    print(f"工作区根目录: {ROOT_DIR}")
    print("=" * 70)

    start_time = time.time()
    audit_record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dry_run": dry_run,
        "actions": [],
        "freed_bytes_estimated": 0,
        "archives_created": [],
    }

    # -------------------------------------------------------------
    # 梯队 1：归档并清理根目录运行与冒烟日志 (*.log)
    # -------------------------------------------------------------
    print("\n>>> [梯队 1] 处理根目录执行与训练日志 (*.log) ...")
    root_logs = sorted(list(ROOT_DIR.glob("*.log")))
    if root_logs:
        log_archive = ARCHIVE_DIR / "root_logs_backup_20260912.zip"
        items = [(p, p.name) for p in root_logs]
        orig_size = sum(p.stat().st_size for p in root_logs)
        if not dry_run:
            create_zip_archive(log_archive, items)
            audit_record["archives_created"].append(str(log_archive.relative_to(ROOT_DIR)))
            for p in root_logs:
                safe_remove_path(p, dry_run)
        print(f"  已处理 {len(root_logs)} 个日志文件 (原始大小: {fmt_size(orig_size)})")
        audit_record["actions"].append({
            "group": "root_logs",
            "type": "archive_and_delete",
            "count": len(root_logs),
            "orig_size": orig_size,
            "archive": str(log_archive.name),
        })
        audit_record["freed_bytes_estimated"] += orig_size

    # -------------------------------------------------------------
    # 梯队 2：归档并清理 results/ 评估中的 smoke 子目录
    # -------------------------------------------------------------
    print("\n>>> [梯队 2] 处理 results/ 评估中的 smoke 子目录 ...")
    eval_smoke_dirs = [
        ROOT_DIR / "results" / "local_eval_20260825" / "smoke_m1_real283_low",
        ROOT_DIR / "results" / "local_eval_20260825" / "smoke_m1_real283_low_corrected",
        ROOT_DIR / "results" / "local_eval_20260825" / "smoke_m1_real680_low_corrected",
        ROOT_DIR / "results" / "revalidation_20260829" / "smoke_operation_only_eft",
    ]
    valid_eval_smoke = [p for p in eval_smoke_dirs if p.exists()]
    if valid_eval_smoke:
        smoke_archive = ARCHIVE_DIR / "smoke_eval_results_backup_20260912.zip"
        items = [(p, p.name) for p in valid_eval_smoke]
        orig_size = sum(get_dir_size(p) for p in valid_eval_smoke)
        if not dry_run:
            create_zip_archive(smoke_archive, items)
            audit_record["archives_created"].append(str(smoke_archive.relative_to(ROOT_DIR)))
            for p in valid_eval_smoke:
                safe_remove_path(p, dry_run)
        print(f"  已处理 {len(valid_eval_smoke)} 个评估 smoke 目录 (原始大小: {fmt_size(orig_size)})")
        audit_record["actions"].append({
            "group": "eval_smoke_dirs",
            "type": "archive_and_delete",
            "count": len(valid_eval_smoke),
            "orig_size": orig_size,
            "archive": str(smoke_archive.name),
        })
        audit_record["freed_bytes_estimated"] += orig_size

    # -------------------------------------------------------------
    # 梯队 3：归档并清理 runs/ 中的 smoke 运行记录
    # -------------------------------------------------------------
    print("\n>>> [梯队 3] 处理 runs/ 中的 smoke 运行产物 ...")
    runs_smoke_dirs = [
        ROOT_DIR / "runs" / "gate_tracefix_local_smoke",
        ROOT_DIR / "runs" / "reschedule_fiveskill_v3_smoke",
        ROOT_DIR / "runs" / "worker_pointer_v2_fast_exact_pilot",
    ]
    valid_runs_smoke = [p for p in runs_smoke_dirs if p.exists()]
    if valid_runs_smoke:
        runs_archive = ARCHIVE_DIR / "runs_smoke_backup_20260912.zip"
        items = [(p, p.name) for p in valid_runs_smoke]
        orig_size = sum(get_dir_size(p) for p in valid_runs_smoke)
        if not dry_run:
            create_zip_archive(runs_archive, items)
            audit_record["archives_created"].append(str(runs_archive.relative_to(ROOT_DIR)))
            for p in valid_runs_smoke:
                safe_remove_path(p, dry_run)
        print(f"  已处理 {len(valid_runs_smoke)} 个 runs smoke 目录 (原始大小: {fmt_size(orig_size)})")
        audit_record["actions"].append({
            "group": "runs_smoke",
            "type": "archive_and_delete",
            "count": len(valid_runs_smoke),
            "orig_size": orig_size,
            "archive": str(runs_archive.name),
        })
        audit_record["freed_bytes_estimated"] += orig_size

    # -------------------------------------------------------------
    # 梯队 4：归档并清理 _docx_work 目录
    # -------------------------------------------------------------
    print("\n>>> [梯队 4] 处理 _docx_work 历史工作目录 ...")
    docx_work_dir = ROOT_DIR / "_docx_work"
    if docx_work_dir.exists():
        docx_archive = ARCHIVE_DIR / "docx_work_backup_20260912.zip"
        orig_size = get_dir_size(docx_work_dir)
        if not dry_run:
            create_zip_archive(docx_archive, [(docx_work_dir, "_docx_work")])
            audit_record["archives_created"].append(str(docx_archive.relative_to(ROOT_DIR)))
            safe_remove_path(docx_work_dir, dry_run)
        print(f"  已归档并删除 _docx_work (原始大小: {fmt_size(orig_size)})")
        audit_record["actions"].append({
            "group": "_docx_work",
            "type": "archive_and_delete",
            "orig_size": orig_size,
            "archive": str(docx_archive.name),
        })
        audit_record["freed_bytes_estimated"] += orig_size

    # -------------------------------------------------------------
    # 梯队 5：处理 results/90_legacy_and_smoke (若启用)
    # -------------------------------------------------------------
    if include_legacy_smoke:
        legacy_dir = ROOT_DIR / "results" / "90_legacy_and_smoke"
        if legacy_dir.exists():
            print("\n>>> [梯队 5] 处理 results/90_legacy_and_smoke (4.63 GB 历史遗留库) ...")
            orig_size = get_dir_size(legacy_dir)

            # 1. 如果内部已有 7z 文件，直接移动到 archive/，避免二次解压或重压缩
            pre_7z = legacy_dir / "root_artifacts_20260717.7z"
            if pre_7z.exists():
                target_7z = ARCHIVE_DIR / "root_artifacts_20260717.7z"
                print(f"  [MOVE] 移动已有压缩包: {pre_7z.name} -> archive/ ...")
                if not dry_run:
                    if target_7z.exists():
                        target_7z.unlink()
                    shutil.move(str(pre_7z), str(target_7z))
                    audit_record["archives_created"].append(str(target_7z.relative_to(ROOT_DIR)))

            # 2. 对其余未压缩的子目录进行 zip 归档
            remaining_items = []
            for item in legacy_dir.iterdir():
                if item.is_dir():
                    remaining_items.append((item, item.name))
                elif item.is_file():
                    remaining_items.append((item, item.name))

            if remaining_items:
                legacy_archive = ARCHIVE_DIR / "legacy_smoke_results_pre20260815.zip"
                if not dry_run:
                    create_zip_archive(legacy_archive, remaining_items)
                    audit_record["archives_created"].append(str(legacy_archive.relative_to(ROOT_DIR)))
                    safe_remove_path(legacy_dir, dry_run)
                print(f"  已归档未压缩历史目录至 {legacy_archive.name}，并清理原目录 (原大小: {fmt_size(orig_size)})")

            audit_record["actions"].append({
                "group": "90_legacy_and_smoke",
                "type": "archive_and_delete",
                "orig_size": orig_size,
                "archive": "root_artifacts_20260717.7z + legacy_smoke_results_pre20260815.zip",
            })
            audit_record["freed_bytes_estimated"] += orig_size

    # -------------------------------------------------------------
    # 梯队 6：纯临时缓存直接删除 (pytest, pycache, tmp, 空目录)
    # -------------------------------------------------------------
    print("\n>>> [梯队 6] 清理纯临时缓存与空目录 (无需备份) ...")
    direct_delete_dirs = [
        ROOT_DIR / ".pytest_cache",
        ROOT_DIR / ".pytest_tmp_v2",
        ROOT_DIR / "tmp",
        ROOT_DIR / "._tools",
        ROOT_DIR / "_r5_ea_work",
    ]
    # diagnostics 下的空 smoke 目录
    for dname in [
        "smoke_identity", "smoke_identity_check", "smoke_identity_final",
        "smoke_protocol", "smoke_protocol_final", "smoke_protocol_final2",
        "smoke_tflogs", "smoke_worker_ab"
    ]:
        direct_delete_dirs.append(ROOT_DIR / "diagnostics" / dname)

    deleted_cache_count = 0
    deleted_cache_size = 0
    for p in direct_delete_dirs:
        if p.exists():
            sz = get_dir_size(p)
            cnt = count_files(p)
            deleted_cache_count += cnt
            deleted_cache_size += sz
            safe_remove_path(p, dry_run)
            print(f"  已删除目录: {p.relative_to(ROOT_DIR)} ({cnt} files, {fmt_size(sz)})")

    # data/.apcf_cf_smoke_*.log
    data_smoke_logs = list(ROOT_DIR.glob("data/.apcf_cf_smoke_*.log"))
    for p in data_smoke_logs:
        sz = p.stat().st_size
        deleted_cache_count += 1
        deleted_cache_size += sz
        safe_remove_path(p, dry_run)
        print(f"  已删除临时数据日志: {p.relative_to(ROOT_DIR)} ({fmt_size(sz)})")

    # 全项目 __pycache__
    pycache_count = 0
    pycache_size = 0
    for r, dirs, files in os.walk(ROOT_DIR):
        if ".git" in r or "archive" in r:
            continue
        if os.path.basename(r) == "__pycache__":
            dp = Path(r)
            sz = get_dir_size(dp)
            cnt = len(files)
            pycache_count += cnt
            pycache_size += sz
            safe_remove_path(dp, dry_run)

    print(f"  已清理全部 __pycache__ 缓存: {pycache_count} 个编译文件 ({fmt_size(pycache_size)})")
    audit_record["actions"].append({
        "group": "transient_caches",
        "type": "direct_delete",
        "cache_files_count": deleted_cache_count + pycache_count,
        "orig_size": deleted_cache_size + pycache_size,
    })
    audit_record["freed_bytes_estimated"] += (deleted_cache_size + pycache_size)

    # -------------------------------------------------------------
    # 写入清理审计账本
    # -------------------------------------------------------------
    elapsed = time.time() - start_time
    audit_record["elapsed_seconds"] = round(elapsed, 2)
    audit_record["freed_human"] = fmt_size(audit_record["freed_bytes_estimated"])

    audit_file = ARCHIVE_DIR / "cleanup_record_20260912.json"
    if not dry_run:
        with open(audit_file, "w", encoding="utf-8") as f:
            json.dump(audit_record, f, ensure_ascii=False, indent=2)
        print(f"\n[AUDIT] 审计日志已保存至: {audit_file.relative_to(ROOT_DIR)}")

    print("\n" + "=" * 70)
    print(f"清理任务完成！耗时: {elapsed:.2f} 秒")
    print(f"预计累计释放磁盘空间: {audit_record['freed_human']}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean workspace temporary files and smoke test artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Perform a trial run with no changes made")
    parser.add_argument("--skip-legacy-smoke", action="store_true", help="Skip 90_legacy_and_smoke directory")
    args = parser.parse_args()

    run_cleanup(dry_run=args.dry_run, include_legacy_smoke=not args.skip_legacy_smoke)
