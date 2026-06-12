# -*- coding: utf-8 -*-
"""
路径兼容性护栏。

这些测试只做静态编译、轻量文件写入和路径 API 检查，不启动正式训练，
用于防止 Windows/Linux 路径处理在后续重构中回退到 os.path 写法。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PATHLIB_TARGETS = [
    PROJECT_ROOT / "scripts" / "generate_schedule.py",
    PROJECT_ROOT / "scripts" / "evaluate_model.py",
    PROJECT_ROOT / "scripts" / "sensitivity_analysis.py",
    PROJECT_ROOT / "scripts" / "generate_synthetic_dataset.py",
    PROJECT_ROOT / "utils" / "generate_random_dataset.py",
    PROJECT_ROOT / "utils" / "generate_worker_pool.py",
    PROJECT_ROOT / "utils" / "logger.py",
    PROJECT_ROOT / "utils" / "report_generator.py",
    PROJECT_ROOT / "utils" / "verify_schedule.py",
]


def test_pathlib_migrated_scripts_compile() -> None:
    """被迁移的脚本必须能直接通过 Python 编译，避免路径改动引入语法回归。"""

    for path in PATHLIB_TARGETS:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")


def test_no_legacy_path_operations_in_migrated_scripts() -> None:
    """迁移范围内不允许继续使用 os.path/os.makedirs/sys.path.append。"""

    forbidden_tokens = ("import os", "os.path", "os.makedirs", "os.remove", "sys.path.append")
    for path in PATHLIB_TARGETS:
        source = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in source, f"{path} 仍包含旧路径写法: {token}"


def test_logger_accepts_pathlib_output_dir(tmp_path: Path) -> None:
    """logger 应接受 pathlib.Path 输出目录，并创建实验日志文件。"""

    from utils.logger import init_logger

    args = SimpleNamespace(data_path=PROJECT_ROOT / "data" / "283.csv", result_dir=tmp_path)
    logger, exp_dir = init_logger(args, "pathlib_guard")

    exp_path = Path(exp_dir)
    assert exp_path.exists()
    assert exp_path.parent == tmp_path
    assert (exp_path / "pathlib_guard.log").exists()

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def test_training_reporter_accepts_pathlib_output_dir(tmp_path: Path) -> None:
    """自动报告器应使用 pathlib.Path 写入 Markdown 报告。"""

    from utils.report_generator import TrainingReporter

    reporter = TrainingReporter(log_dir=tmp_path)
    reporter.add_record(
        ep=1,
        makespan=1.0,
        balance=0.1,
        w_util=0.2,
        s_util=0.3,
        best_sch=[],
        eval_reward=0.4,
    )
    reporter.generate_report(current_ep=1)

    reports = list(tmp_path.glob("report_ep1_*.md"))
    assert len(reports) == 1
    assert reports[0].read_text(encoding="utf-8").startswith("#")
