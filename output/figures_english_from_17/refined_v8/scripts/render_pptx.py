"""Render a PowerPoint (.pptx) file to per-slide PNGs using PowerPoint COM.

Cross-process lockfile-safe so concurrent swarm workers rendering their own
scratch PPTX files serialize cleanly through PowerPoint COM without crashing.
"""
from __future__ import annotations

import errno
import os
import sys
import time
from contextlib import contextmanager

import comtypes
import comtypes.client

BASE_DIR = r"D:\APAL-Dynamic-v4\output\figures_english_from_17\refined_v8"
DEFAULT_LOCK_PATH = os.path.join(BASE_DIR, "work", "ppt_com.lock")
STALE_LOCK_SECONDS = 120.0
ACQUIRE_TIMEOUT_SECONDS = 300.0


@contextmanager
def com_file_lock(lock_path: str = DEFAULT_LOCK_PATH,
                  stale_seconds: float = STALE_LOCK_SECONDS,
                  timeout_seconds: float = ACQUIRE_TIMEOUT_SECONDS):
    os.makedirs(os.path.dirname(os.path.abspath(lock_path)), exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    fd = None
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, f"pid={os.getpid()} time={time.time():.3f}\n".encode("utf-8"))
            os.close(fd)
            fd = None
            break
        except FileExistsError:
            # Check for stale lock (> stale_seconds old)
            try:
                age = time.time() - os.path.getmtime(lock_path)
                if age > stale_seconds:
                    try:
                        os.unlink(lock_path)
                        continue
                    except OSError:
                        pass
            except OSError:
                # Lock disappeared between checks; retry immediately
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Timed out waiting {timeout_seconds:.0f}s for PowerPoint COM lock: {lock_path}"
                )
            time.sleep(0.35)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                time.sleep(0.35)
                continue
            raise
    try:
        yield
    finally:
        try:
            if os.path.exists(lock_path):
                os.unlink(lock_path)
        except OSError:
            pass


def render_pptx(
    pptx_path: str,
    out_dir: str,
    width: int = 1920,
    height: int = 1080,
    slide_indices=None,
    lock_path: str = DEFAULT_LOCK_PATH,
):
    pptx_path = os.path.abspath(pptx_path)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with com_file_lock(lock_path=lock_path):
        comtypes.CoInitialize()
        app = None
        prs = None
        try:
            app = comtypes.client.CreateObject("PowerPoint.Application")
            # PowerPoint requires WithWindow=False for silent open
            prs = app.Presentations.Open(
                pptx_path, ReadOnly=True, Untitled=False, WithWindow=False
            )
            total = prs.Slides.Count
            indices = slide_indices if slide_indices else list(range(1, total + 1))
            for idx in indices:
                slide = prs.Slides.Item(idx)
                out_png = os.path.join(out_dir, f"slide_{idx:02d}.png")
                slide.Export(out_png, "PNG", width, height)
                print(f"Exported {out_png}", flush=True)
        finally:
            if prs is not None:
                try:
                    prs.Close()
                except Exception:
                    pass
            if app is not None:
                try:
                    if app.Presentations.Count == 0:
                        app.Quit()
                except Exception:
                    pass
            comtypes.CoUninitialize()
            # Brief cooldown so PowerPoint COM tears down cleanly before next process acquires lock
            time.sleep(0.4)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python render_pptx.py <pptx_path> <out_dir> [slide_idx1,slide_idx2,...]")
        sys.exit(1)
    slides = None
    if len(sys.argv) >= 4 and sys.argv[3].strip():
        slides = [int(x) for x in sys.argv[3].split(",") if x.strip()]
    render_pptx(sys.argv[1], sys.argv[2], slide_indices=slides)

