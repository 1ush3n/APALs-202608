"""Build 2x2 comparison plates (v6 | v7 | v8 NEW | Ref) for refined_v8."""
from __future__ import annotations

import os
import sys
from PIL import Image, ImageDraw, ImageFont

ROOT = r"D:\APAL-Dynamic-v4\output\figures_english_from_17"
V6_PREV = os.path.join(ROOT, "refined_v6", "previews_refined_v6")
V7_PREV = os.path.join(ROOT, "refined_v7", "previews_refined_v7")
V8_PREV = os.path.join(ROOT, "refined_v8", "previews_refined_v8")
REF_DIR = r"D:\APAL-Dynamic-v4\output\figure_study_15\assets"
OUT_DIR = os.path.join(ROOT, "refined_v8", "comparisons")

MODULES = [
    (1, "A1", "a1.png", "Ref: Zhang et al., NeurIPS 2020 Fig. 3"),
    (2, "A2", "b4.png", "Ref (7.5): PipeDream, SOSP 2019 Fig. 5"),
    (3, "A4", "b4.png", "Ref (7.5): PipeDream, SOSP 2019 Fig. 5"),
    (4, "A5", "b4.png", "Ref (7.5): PipeDream, SOSP 2019 Fig. 5"),
    (5, "A6", "b4.png", "Ref (7.5): PipeDream, SOSP 2019 Fig. 5"),
    (6, "B1", "a3.png", "Ref (8.5): Wang & Gombolay, RSS 2020 Fig. 1"),
    (7, "B2", "a3.png", "Ref (8.5): Wang & Gombolay, RSS 2020 Fig. 1"),
    (8, "B3", "a5.png", "Ref (7.5): Smit et al., AAAI 2025 Fig. 1"),
    (9, "B4", "a2.png", "Ref (8.5): ReSched, ICLR 2026 Fig. 2"),
    (10, "B5", "a4.png", "Ref (8.0): Dual-Attention Net, TNNLS 2024 Fig. 3"),
    (11, "B6", "a2.png", "Ref (8.5): ReSched, ICLR 2026 Fig. 2"),
    (12, "B7", "b2.png", "Ref: Switch Transformers, JMLR 2022 Fig. 2"),
    (13, "B8", "a4.png", "Ref (8.0): Dual-Attention Net, TNNLS 2024 Fig. 3"),
    (14, "B9", "b4.png", "Ref (7.5): PipeDream, SOSP 2019 Fig. 5"),
    (15, "B10", "a5.png", "Ref (7.5): Smit et al., AAAI 2025 Fig. 1 + ReSched Fig. 2"),
    (16, "B11", "c3.png", "Ref: TD-MPC, ICML 2022 Fig. 2"),
]
V8_UPGRADED = {6, 8, 10, 11, 13, 15, 16}


def _fit(im: Image.Image, tw: int, th: int) -> Image.Image:
    im = im.convert("RGB")
    bg = Image.new("RGB", (tw, th), (255, 255, 255))
    r = min(tw / im.width, th / im.height)
    nw, nh = max(1, int(im.width * r)), max(1, int(im.height * r))
    resized = im.resize((nw, nh), Image.Resampling.LANCZOS)
    bg.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
    return bg


def make_plate(slide_idx: int, mod_id: str, ref_file: str | None, ref_lbl: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    tw, th = 1120, 630
    hdr = 42
    pad = 24
    cw = 2 * tw + 3 * pad
    ch = 2 * (th + hdr) + 3 * pad
    canvas = Image.new("RGB", (cw, ch), (245, 247, 250))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arialbd.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    fn = f"slide_{slide_idx:02d}.png"
    p_v6 = os.path.join(V6_PREV, fn)
    p_v7 = os.path.join(V7_PREV, fn)
    p_v8 = os.path.join(V8_PREV, fn)
    p_ref = os.path.join(REF_DIR, ref_file) if ref_file else None
    is_up = slide_idx in V8_UPGRADED

    cells = [
        (0, 0, p_v6, f"[Refined v6 | Slide {slide_idx}] Milestone v6", (70, 90, 110)),
        (0, 1, p_v7, f"[Refined v7 | Slide {slide_idx}] Saved Milestone v7", (44, 86, 128)),
        (1, 0, p_v8, f"[Refined v8 | Slide {slide_idx}] {mod_id} {'NEW AUDIT POLISH' if is_up else '(Carried Forward)'}", (24, 116, 72) if is_up else (76, 98, 120)),
        (1, 1, p_ref if (p_ref and os.path.exists(p_ref)) else p_v8, f"[Reference Style Target] {ref_lbl}", (142, 74, 34)),
    ]

    for r, c, img_p, title, col in cells:
        x = pad + c * (tw + pad)
        y = pad + r * (th + hdr + pad)
        draw.rectangle([x, y, x + tw, y + hdr], fill=col)
        draw.text((x + 14, y + 9), title, fill=(255, 255, 255), font=font)
        if img_p and os.path.exists(img_p):
            tile = _fit(Image.open(img_p), tw, th)
        else:
            tile = Image.new("RGB", (tw, th), (250, 251, 252))
        canvas.paste(tile, (x, y + hdr))
        draw.rectangle([x, y, x + tw, y + hdr + th], outline=col, width=3)

    out_p = os.path.join(OUT_DIR, f"compare_{slide_idx:02d}_{mod_id}.png")
    canvas.save(out_p, quality=94)
    print(f"Saved {out_p}")


if __name__ == "__main__":
    targets = None
    if len(sys.argv) >= 2 and sys.argv[1].strip():
        targets = {int(x) for x in sys.argv[1].split(",") if x.strip()}
    for s_idx, mod_id, rf, rl in MODULES:
        if targets is None or s_idx in targets:
            make_plate(s_idx, mod_id, rf, rl)
