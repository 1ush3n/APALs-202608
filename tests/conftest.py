from __future__ import annotations

import os
from pathlib import Path
import sys


def pytest_configure() -> None:
    """统一测试进程输出编码，避免 Windows fd 捕获读取中文日志时解码失败。"""
    (Path(__file__).resolve().parents[1] / ".pytest_tmp_v2").mkdir(
        parents=True,
        exist_ok=True,
    )
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
