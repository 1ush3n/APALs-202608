"""Export APAL_Paper_Integrated_Figures_v1.pptx -> Vector PDFs (tight-cropped),
600 DPI tight-cropped PNGs, web previews, and run full automated QA.

Pipeline:
1. PowerPoint COM exports:
   - Full native lossless vector PDF (`work/all_slides_raw.pdf`)
   - Ultra-high-resolution per-slide PNGs (`work/raw_png/slide_XX.png`, 4800x2700)
2. Pixel-exact ink bounding box computed on each PNG (+ 0.06 in uniform margin).
3. `pypdf` extracts each slide page and applies the exact MediaBox/CropBox ->
   100% lossless vector PDF in `pdf/<slug>.pdf` ready for IEEE `\includegraphics`.
4. Exact tight-cropped 600 DPI PNG saved to `png_600dpi/<slug>.png` (and compact
   review PNG in `previews/<slug>.png`).
5. Automated text/terminology/font/geometry QA prints full report and saves JSON.
"""
from __future__ import annotations

import errno
import json
import os
import re
import sys
import time
from contextlib import contextmanager

import comtypes
import comtypes.client
import numpy as np
from PIL import Image
from pptx import Presentation
from pypdf import PdfReader, PdfWriter
from pypdf.generic import RectangleObject

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
PPTX_PATH = os.path.join(BASE_DIR, "APAL_Paper_Integrated_Figures_v1.pptx")
WORK_DIR = os.path.join(BASE_DIR, "work")
RAW_PDF_PATH = os.path.join(WORK_DIR, "all_slides_raw.pdf")
RAW_PNG_DIR = os.path.join(WORK_DIR, "raw_png")
PDF_DIR = os.path.join(BASE_DIR, "pdf")
PNG600_DIR = os.path.join(BASE_DIR, "png_600dpi")
PREV_DIR = os.path.join(BASE_DIR, "previews")
LOCK_PATH = os.path.join(WORK_DIR, "ppt_com.lock")

SLIDE_SLUGS = [
    "fig1_apal_unfolding_delay",
    "fig2_compact_b1_b5",
    "fig2_standalone_b1",
    "fig3_standalone_b5",
    "fig_appA_b8_worker_pointer",
    "extra_b3_hetero_gatv2",
    "extra_b6_op_pointer",
    "extra_b10_critic_head",
    "extra_b11_ppo_gae",
]

FORBIDDEN_STRINGS = [
    "Trade Skill Hub",
    "GATv2Conv",
    "concat=",
    "concat =",
    "aggr=",
    "aggr =",
    "ckpt:",
    "Table VII",
    "HB-GAT-PPO",
    "O2O Mask",
    "Hidden Unit j",
    "illustrative",
    ".csv",
    "Module B",
    "Module A",
]
CJK_RE = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")


@contextmanager
def com_file_lock(lock_path: str = LOCK_PATH, stale_seconds: float = 120.0, timeout_seconds: float = 300.0):
    os.makedirs(os.path.dirname(os.path.abspath(lock_path)), exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, f"pid={os.getpid()} time={time.time():.3f}\n".encode("utf-8"))
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock_path) > stale_seconds:
                    os.unlink(lock_path)
                    continue
            except OSError:
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out waiting for lock {lock_path}")
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


def export_via_powerpoint(pptx_path: str, pdf_path: str, raw_png_dir: str, png_w: int = 4800, png_h: int = 2700):
    os.makedirs(os.path.dirname(os.path.abspath(pdf_path)), exist_ok=True)
    os.makedirs(raw_png_dir, exist_ok=True)
    pptx_abs = os.path.abspath(pptx_path)
    pdf_abs = os.path.abspath(pdf_path)

    with com_file_lock():
        comtypes.CoInitialize()
        app = None
        prs = None
        try:
            app = comtypes.client.CreateObject("PowerPoint.Application")
            prs = app.Presentations.Open(pptx_abs, ReadOnly=True, Untitled=False, WithWindow=False)
            # 1. Native lossless vector PDF (ppSaveAsPDF = 32)
            prs.SaveAs(pdf_abs, 32)
            print(f"Exported native vector PDF -> {pdf_abs}", flush=True)

            # 2. Ultra-high-resolution PNGs
            total = prs.Slides.Count
            for idx in range(1, total + 1):
                slide = prs.Slides.Item(idx)
                out_png = os.path.join(raw_png_dir, f"slide_{idx:02d}.png")
                try:
                    slide.Export(out_png, "PNG", png_w, png_h)
                except Exception:
                    slide.Export(out_png, "PNG", 3840, 2160)
                print(f"Exported raw PNG slide {idx}/{total} -> {out_png}", flush=True)
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
            time.sleep(0.4)


def compute_ink_bbox_frac(png_path: str, pad_in: float = 0.06, slide_w_in: float = 13.333333, slide_h_in: float = 7.5):
    """Return pixel-exact normalized (fx0, fy0, fx1, fy1) top-left bounding box with uniform inch margin."""
    im = Image.open(png_path).convert("RGB")
    arr = np.asarray(im)
    h, w, _ = arr.shape
    mask = np.any(arr < 250, axis=2)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0.0, 0.0, 1.0, 1.0), im
    pad_x = int(round((pad_in / slide_w_in) * w))
    pad_y = int(round((pad_in / slide_h_in) * h))
    x0 = max(0, int(xs.min()) - pad_x)
    x1 = min(w, int(xs.max()) + 1 + pad_x)
    y0 = max(0, int(ys.min()) - pad_y)
    y1 = min(h, int(ys.max()) + 1 + pad_y)
    return (x0 / w, y0 / h, x1 / w, y1 / h), im


def crop_all_outputs():
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(PNG600_DIR, exist_ok=True)
    os.makedirs(PREV_DIR, exist_ok=True)

    reader = PdfReader(RAW_PDF_PATH)
    assert len(reader.pages) == len(SLIDE_SLUGS), f"PDF pages {len(reader.pages)} != {len(SLIDE_SLUGS)}"

    metrics = []
    for idx, slug in enumerate(SLIDE_SLUGS):
        raw_png = os.path.join(RAW_PNG_DIR, f"slide_{idx + 1:02d}.png")
        (fx0, fy0, fx1, fy1), im = compute_ink_bbox_frac(raw_png, pad_in=0.06)
        w_px, h_px = im.size

        # 1. Tight-cropped native vector PDF
        page = reader.pages[idx]
        pw_pt = float(page.mediabox.width)
        ph_pt = float(page.mediabox.height)
        x0_pt = fx0 * pw_pt
        x1_pt = fx1 * pw_pt
        y0_pt = (1.0 - fy1) * ph_pt
        y1_pt = (1.0 - fy0) * ph_pt

        writer = PdfWriter()
        writer.add_page(page)
        out_page = writer.pages[0]
        rect = RectangleObject([x0_pt, y0_pt, x1_pt, y1_pt])
        out_page.mediabox = rect
        out_page.cropbox = rect
        out_page.trimbox = rect
        out_page.artbox = rect
        pdf_out = os.path.join(PDF_DIR, f"{slug}.pdf")
        with open(pdf_out, "wb") as f:
            writer.write(f)

        # 2. Tight-cropped 600 DPI PNG (scaled cleanly if needed so 7.16 in print width == 600 DPI -> >= 4296 px)
        crop_box = (int(round(fx0 * w_px)), int(round(fy0 * h_px)), int(round(fx1 * w_px)), int(round(fy1 * h_px)))
        cropped = im.crop(crop_box)
        cw, ch = cropped.size
        # Target exact 600 DPI at IEEE double-column width 7.16 in (4296 px wide)
        target_w = 4296
        if cw < target_w:
            scale = target_w / cw
            png600 = cropped.resize((target_w, int(round(ch * scale))), Image.Resampling.LANCZOS)
        else:
            png600 = cropped
        png600_out = os.path.join(PNG600_DIR, f"{slug}.png")
        png600.save(png600_out, dpi=(600, 600), optimize=True)

        # 3. Web-review preview PNG (~1800 px wide)
        prev_w = 1800
        prev_h = int(round(ch * (prev_w / cw)))
        prev_im = cropped.resize((prev_w, prev_h), Image.Resampling.LANCZOS)
        prev_out = os.path.join(PREV_DIR, f"{slug}.png")
        prev_im.save(prev_out, optimize=True)

        crop_w_in = (x1_pt - x0_pt) / 72.0
        crop_h_in = (y1_pt - y0_pt) / 72.0
        aspect = crop_w_in / crop_h_in
        ieee_h_in = 7.16 / aspect
        ieee_textheight_pct = (ieee_h_in / 9.25) * 100.0  # IEEEtran 10pt \textheight ~ 9.25 in

        metrics.append({
            "slide": idx + 1,
            "slug": slug,
            "raw_png_size": [w_px, h_px],
            "cropped_png_600_size": list(png600.size),
            "pdf_size_in": [round(crop_w_in, 2), round(crop_h_in, 2)],
            "aspect_ratio": round(aspect, 2),
            "ieee_716in_print_h_in": round(ieee_h_in, 2),
            "ieee_textheight_pct": round(ieee_textheight_pct, 1),
        })
        print(
            f"  [{idx+1}/9] {slug:28s} | PDF {crop_w_in:5.2f}x{crop_h_in:4.2f} in "
            f"(AR {aspect:4.2f}:1) | @7.16in print h={ieee_h_in:4.2f} in ({ieee_textheight_pct:4.1f}% \\textheight) | 600DPI={png600.size}",
            flush=True,
        )
    return metrics


def run_qa(metrics: list[dict]):
    prs = Presentation(PPTX_PATH)
    print("\n=== AUTOMATED PUBLICATION QA ===")
    all_ok = True
    qa_records = []
    for idx, slide in enumerate(prs.slides):
        slug = SLIDE_SLUGS[idx]
        texts = []
        fonts = []
        for sh in slide.shapes:
            if sh.has_text_frame:
                for p in sh.text_frame.paragraphs:
                    for r in p.runs:
                        if r.text.strip():
                            texts.append(r.text)
                            if r.font.size is not None:
                                fonts.append(round(r.font.size.pt, 1))
        notes = slide.notes_slide.notes_text_frame.text if slide.has_notes_slide else ""
        joined = " ".join(texts)
        cjk_hits = CJK_RE.findall(joined + " " + notes)
        forb_hits = [s for s in FORBIDDEN_STRINGS if s.lower() in joined.lower()]
        min_f = min(fonts) if fonts else 0.0
        max_f = max(fonts) if fonts else 0.0
        scale_716 = 7.16 / metrics[idx]["pdf_size_in"][0]
        eff_min_print_pt = round(min_f * scale_716, 1)

        ok = (len(cjk_hits) == 0) and (len(forb_hits) == 0)
        if not ok:
            all_ok = False
        rec = {
            **metrics[idx],
            "min_ppt_pt": min_f,
            "max_ppt_pt": max_f,
            "min_print_pt_at_716in": eff_min_print_pt,
            "cjk_count": len(cjk_hits),
            "forbidden_hits": forb_hits,
            "status": "PASS" if ok else "FAIL",
        }
        qa_records.append(rec)
        print(
            f"  [{idx+1}] {slug:28s} | PPT font {min_f:4.1f}-{max_f:4.1f}pt "
            f"(~{eff_min_print_pt:3.1f}pt @7.16in) | CJK={len(cjk_hits)} | Forb={forb_hits} | {rec['status']}"
        )

    qa_path = os.path.join(BASE_DIR, "qa_report_v1.json")
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_records, f, indent=2)
    print(f"\nSaved QA report -> {qa_path}")
    if not all_ok:
        sys.exit(2)


def main():
    export_via_powerpoint(PPTX_PATH, RAW_PDF_PATH, RAW_PNG_DIR, png_w=4800, png_h=2700)
    metrics = crop_all_outputs()
    run_qa(metrics)


if __name__ == "__main__":
    main()
