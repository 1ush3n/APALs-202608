from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _default_rag_env() -> Path:
    return Path.home() / "miniconda3" / "envs" / "rag_env"


def _env_path(conda_prefix: Path) -> str:
    path_parts = [
        conda_prefix,
        conda_prefix / "Library" / "bin",
        conda_prefix / "Scripts",
        conda_prefix / "Library" / "miktex" / "texmfs" / "install" / "miktex" / "bin" / "x64",
    ]
    return os.pathsep.join(str(path) for path in path_parts if path.exists())


def _resolve_conda_prefix() -> Path:
    configured_prefix = os.environ.get("CONDA_PREFIX")
    if configured_prefix:
        return Path(configured_prefix)
    return _default_rag_env()


def _build_command(args: argparse.Namespace, latexmk_path: str) -> list[str]:
    command = [
        latexmk_path,
        "-pdf",
        "-synctex=1",
        "-interaction=nonstopmode",
        "-halt-on-error",
    ]
    if args.clean:
        command.append("-C")
    if args.watch:
        command.append("-pvc")
    command.append(str(args.tex_file))
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 rag_env 中的 latexmk 编译 LaTeX 论文。")
    parser.add_argument("tex_file", type=Path, help="主 .tex 文件路径，例如 paper/main.tex。")
    parser.add_argument("--watch", action="store_true", help="启用 latexmk 持续监听模式。")
    parser.add_argument("--clean", action="store_true", help="清理 LaTeX 中间产物。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assert args.tex_file.suffix == ".tex", "输入文件必须是 .tex 文件。"
    assert args.tex_file.exists(), f"找不到 LaTeX 主文件：{args.tex_file}"
    args.tex_file = args.tex_file.resolve()

    conda_prefix = _resolve_conda_prefix()
    assert conda_prefix.exists(), f"找不到 Conda 环境目录：{conda_prefix}"

    env = os.environ.copy()
    env["PATH"] = _env_path(conda_prefix) + os.pathsep + env.get("PATH", "")
    latexmk_path = shutil.which("latexmk", path=env["PATH"])
    assert latexmk_path is not None, "当前环境找不到 latexmk，请确认已安装 miktex 和 latexmk。"

    command = _build_command(args, latexmk_path)
    result = subprocess.run(command, cwd=args.tex_file.parent, env=env, check=False)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
