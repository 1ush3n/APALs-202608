"""Automated QA verification & 4x4 overview contact sheet builder for refined_v8."""
from __future__ import annotations

import os
import re
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation

ROOT = r"D:\APAL-Dynamic-v4\output\figures_english_from_17"
V8_DIR = os.path.join(ROOT, "refined_v8")
PPTX_PATH = os.path.join(V8_DIR, "APAL_Modules_English_Refined_v8.pptx")
PREV_DIR = os.path.join(V8_DIR, "previews_refined_v8")
COMP_DIR = os.path.join(V8_DIR, "comparisons")

EXPECTED_MODULES = [
    (1, "A1"), (2, "A2"), (3, "A4"), (4, "A5"), (5, "A6"),
    (6, "B1"), (7, "B2"), (8, "B3"), (9, "B4"), (10, "B5"),
    (11, "B6"), (12, "B7"), (13, "B8"), (14, "B9"), (15, "B10"), (16, "B11"),
]
V8_UPGRADED = {6, 8, 10, 11, 13, 15, 16}

# Strict academic-publication audit tokens banned from all 7 upgraded paper figures!
FORBIDDEN_VISIBLE_TOKENS = [
    "(B4)", "(B7)", "(B11)", "B11",
    "self.critic", "station_attn", "task_worker_attn",
    "concat=False", "concat = False", "aggr = sum", "GATv2Conv", "WorkerPointer",
    "Trade Skill Hub", "Skill hubs", "Hidden Unit", "illustrative negative",
    "O2O Mask", "-1e4", "\u22121e4",
    "Table VII", "ckpt:", "per-head clip", "Filled = traced",
]
CJK_RE = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")


def _iter_shape_texts_and_sizes(shapes):
    for sh in shapes:
        if sh.shape_type == 6:
            yield from _iter_shape_texts_and_sizes(sh.shapes)
        if getattr(sh, "has_text_frame", False):
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    txt = r.text or ""
                    sz = r.font.size.pt if r.font.size is not None else None
                    yield txt, sz
        if getattr(sh, "has_table", False):
            for row in sh.table.rows:
                for cell in row.cells:
                    for p in cell.text_frame.paragraphs:
                        for r in p.runs:
                            txt = r.text or ""
                            sz = r.font.size.pt if r.font.size is not None else None
                            yield txt, sz


def run_qa():
    print("=" * 74)
    print("APAL refined_v8 Automated Paper-Terminology & Quality Assurance Audit")
    print("=" * 74)
    prs = Presentation(PPTX_PATH)
    n_slides = len(prs.slides)
    assert n_slides == 16, f"Expected 16 slides, got {n_slides}"
    print(f"[PASS] Slide count: {n_slides} (A1, A2, A4-A6, B1-B11; A3 absent)")

    cjk_hits = []
    forbidden_hits = []
    font_stats = {}

    for slide_idx, mod_id in EXPECTED_MODULES:
        slide = prs.slides[slide_idx - 1]
        vis_texts = []
        sizes = []
        for txt, sz in _iter_shape_texts_and_sizes(slide.shapes):
            if txt.strip():
                vis_texts.append(txt)
                if sz is not None:
                    sizes.append(sz)
            if CJK_RE.search(txt):
                cjk_hits.append((slide_idx, mod_id, "slide", txt[:60]))

        notes_txt = ""
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            notes_txt = slide.notes_slide.notes_text_frame.text or ""
            if CJK_RE.search(notes_txt):
                cjk_hits.append((slide_idx, mod_id, "notes", notes_txt[:60]))

        combined_vis = " ".join(vis_texts)
        assert (mod_id in combined_vis) or (mod_id in notes_txt), (
            f"Module ID {mod_id} missing from both slide {slide_idx} and its notes!"
        )
        assert "Module A3" not in notes_txt and "A3" not in combined_vis.split(), (
            f"Unexpected A3 reference on slide {slide_idx}"
        )

        if slide_idx in V8_UPGRADED:
            for tok in FORBIDDEN_VISIBLE_TOKENS:
                if tok in combined_vis:
                    forbidden_hits.append((slide_idx, mod_id, tok))
            min_pt = min(sizes) if sizes else 0.0
            max_pt = max(sizes) if sizes else 0.0
            font_stats[slide_idx] = (mod_id, len(slide.shapes), min_pt, max_pt)

    assert not cjk_hits, f"CJK characters found: {cjk_hits}"
    print("[PASS] Zero CJK characters across all 16 slides and all 16 speaker notes")

    assert not forbidden_hits, f"Forbidden non-paper/code/debug tokens in v8 slides: {forbidden_hits}"
    print("[PASS] Zero non-paper terms, Python kwargs, debug leaks, or wrong table/slide IDs across all 7 upgraded figures")

    print("\nUpgraded v8 Slide Typography & Shape Summary:")
    for s_idx in sorted(V8_UPGRADED):
        mod_id, n_sh, mn, mx = font_stats[s_idx]
        print(f"  Slide {s_idx:02d} ({mod_id:3s}): shapes={n_sh:3d}, min_font={mn:4.1f} pt, max_font={mx:4.1f} pt")

    missing_prev = [i for i in range(1, 17) if not os.path.exists(os.path.join(PREV_DIR, f"slide_{i:02d}.png"))]
    assert not missing_prev, f"Missing preview PNGs: {missing_prev}"
    print(f"[PASS] All 16 full-res preview PNGs exist in {PREV_DIR}")

    missing_comp = [
        f"compare_{i:02d}_{m}.png"
        for i, m in EXPECTED_MODULES
        if not os.path.exists(os.path.join(COMP_DIR, f"compare_{i:02d}_{m}.png"))
    ]
    assert not missing_comp, f"Missing comparison PNGs: {missing_comp}"
    print(f"[PASS] All 16 2x2 comparison plates exist in {COMP_DIR}")

    prior_files = [
        r"D:\APAL-Dynamic-v4\output\figures_reference_revision\APAL_17_Modules.pptx",
        os.path.join(ROOT, "APAL_Modules_English_Review.pptx"),
        os.path.join(ROOT, "refined_v2", "APAL_Modules_English_Refined_v2.pptx"),
        os.path.join(ROOT, "refined_v3", "APAL_Modules_English_Refined_v3.pptx"),
        os.path.join(ROOT, "refined_v4", "APAL_Modules_English_Refined_v4.pptx"),
        os.path.join(ROOT, "refined_v5", "APAL_Modules_English_Refined_v5.pptx"),
        os.path.join(ROOT, "refined_v6", "APAL_Modules_English_Refined_v6.pptx"),
        os.path.join(ROOT, "refined_v7", "APAL_Modules_English_Refined_v7.pptx"),
    ]
    print("\nProtected Milestone Integrity Check:")
    for pf in prior_files:
        assert os.path.exists(pf), f"Protected file missing: {pf}"
        st = os.stat(pf)
        mt = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  [OK] {os.path.basename(pf):42s} size={st.st_size:>10,d} B  mtime={mt}")

    return True


def build_contact_sheet():
    cols, rows = 4, 4
    tw, th = 720, 405
    pad = 16
    hdr_h = 34
    title_h = 54
    canvas_w = cols * tw + (cols + 1) * pad
    canvas_h = title_h + rows * (th + hdr_h) + (rows + 1) * pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), (244, 247, 250))
    draw = ImageDraw.Draw(canvas)
    try:
        font_title = ImageFont.truetype("arialbd.ttf", 23)
        font_hdr = ImageFont.truetype("arialbd.ttf", 17)
    except Exception:
        font_title = ImageFont.load_default()
        font_hdr = ImageFont.load_default()

    draw.rectangle([0, 0, canvas_w, title_h], fill=(24, 42, 60))
    draw.text(
        (20, 14),
        "APAL 16-Module English Deck (refined_v8)  |  Green = v8 Paper-Audit Polished (B1, B3, B5, B6, B8, B10, B11)  |  Slate = Carried Forward",
        fill=(255, 255, 255),
        font=font_title,
    )

    for idx, (s_idx, mod_id) in enumerate(EXPECTED_MODULES):
        r, c = divmod(idx, cols)
        x = pad + c * (tw + pad)
        y = title_h + pad + r * (th + hdr_h + pad)
        is_up = s_idx in V8_UPGRADED
        bg_hdr = (26, 112, 72) if is_up else (52, 74, 96)
        tag = "[v8 PAPER-AUDITED]" if is_up else "(saved review)"
        draw.rectangle([x, y, x + tw, y + hdr_h], fill=bg_hdr)
        draw.text((x + 12, y + 7), f"Slide {s_idx:02d}  ({mod_id})   {tag}", fill=(255, 255, 255), font=font_hdr)

        im = Image.open(os.path.join(PREV_DIR, f"slide_{s_idx:02d}.png")).convert("RGB")
        im = im.resize((tw, th), Image.Resampling.LANCZOS)
        canvas.paste(im, (x, y + hdr_h))
        bdr = (26, 112, 72) if is_up else (180, 192, 204)
        draw.rectangle([x, y, x + tw, y + hdr_h + th], outline=bdr, width=3 if is_up else 2)

    out_path = os.path.join(COMP_DIR, "00_all_16_overview_contact_sheet.png")
    canvas.save(out_path, quality=95)
    print(f"\n[PASS] Saved 4x4 overview contact sheet: {out_path}")


if __name__ == "__main__":
    run_qa()
    build_contact_sheet()
