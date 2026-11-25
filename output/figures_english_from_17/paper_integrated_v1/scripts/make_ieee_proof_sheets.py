"""Generate IEEE Double-Column (7.16 in) Real-Width Print Proof & Comparison Sheets.

Outputs inside `paper_integrated_v1/review_sheets/`:
  1. `01_old_vs_new_fig1_and_fig2.png`
     Direct before/after comparison against existing manuscript `fig1.png` and `fig2.png`.
  2. `02_compact_vs_standalone_ieee_page_sim.png`
     True-to-scale IEEE TSMC 2-column 10pt journal page simulation (8.5 x 11 in US Letter,
     7.16 in double-column width, authentic 2-column body text) comparing:
       Page A: New Fig. 1 (A2+A5) + Compact Fig. 2(a)-(b) (B1+B5)
       Page B: High-Clarity Standalone Fig. 2 (B1) + Standalone Fig. 3 (B5)
       Page C: New Standalone Appendix A Fig. (B8) at true 7.16 in width
  3. `03_all_9_figures_gallery.png`
     3x3 overview gallery of all 9 tight-cropped publication figures with exact metrics.
"""
from __future__ import annotations

import json
import os
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
PREV_DIR = os.path.join(BASE_DIR, "previews")
PNG600_DIR = os.path.join(BASE_DIR, "png_600dpi")
OUT_DIR = os.path.join(BASE_DIR, "review_sheets")
QA_JSON = os.path.join(BASE_DIR, "qa_report_v1.json")

OLD_FIG1 = r"D:\APAL-Dynamic-v4\docs\APAL_HGP_PPO_Journal_Manuscript\04_Figures_and_Assets\fig1.png"
OLD_FIG2 = r"D:\APAL-Dynamic-v4\docs\APAL_HGP_PPO_Journal_Manuscript\04_Figures_and_Assets\fig2.png"


def _font(size: int, bold: bool = False, serif: bool = False):
    candidates = []
    if serif:
        candidates = [
            r"C:\Windows\Fonts\timesbd.ttf" if bold else r"C:\Windows\Fonts\times.ttf",
            r"C:\Windows\Fonts\cambriab.ttf" if bold else r"C:\Windows\Fonts\cambria.ttc",
        ]
    else:
        candidates = [
            r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
        ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _fit_image(im: Image.Image, max_w: int, max_h: int) -> Image.Image:
    w, h = im.size
    scale = min(max_w / w, max_h / h)
    return im.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.Resampling.LANCZOS)


def build_old_vs_new():
    os.makedirs(OUT_DIR, exist_ok=True)
    W, H = 3000, 1980
    canvas = Image.new("RGB", (W, H), (247, 249, 252))
    draw = ImageDraw.Draw(canvas)

    f_title = _font(36, bold=True)
    f_sub   = _font(24, bold=True)
    f_meta  = _font(20, bold=False)

    draw.rectangle([0, 0, W, 86], fill=(27, 45, 66))
    draw.text((40, 22), "Before vs. After: Existing Manuscript Figures (fig1/fig2) vs. New Paper-Integrated Figures v1", fill=(255, 255, 255), font=f_title)

    with open(QA_JSON, "r", encoding="utf-8") as f:
        qa_map = {rec["slug"]: rec for rec in json.load(f)}
    ar1 = qa_map["fig1_apal_unfolding_delay"]["aspect_ratio"]
    ar2 = qa_map["fig2_compact_b1_b5"]["aspect_ratio"]

    cards = [
        (40,   114, 1430, 880, "EXISTING fig1.png  (4 typos, wrong t_r, conflicting specs)", OLD_FIG1, (166, 38, 39)),
        (1530, 114, 1430, 880, f"NEW Fig. 1 (A2 + A5 Integrated, AR {ar1:.2f}:1, 0 errors)", os.path.join(PREV_DIR, "fig1_apal_unfolding_delay.png"), (29, 111, 99)),
        (40,  1040, 1430, 900, "EXISTING fig2.png  (HB-GAT-PPO, mislabeled Critic, GATConv)", OLD_FIG2, (166, 38, 39)),
        (1530,1040, 1430, 900, f"NEW Compact Fig. 2(a)-(b) (B1 + B5 Seamless Fusion, AR {ar2:.2f}:1)", os.path.join(PREV_DIR, "fig2_compact_b1_b5.png"), (29, 111, 99)),
    ]

    for x, y, cw, ch, label, img_p, hdr_col in cards:
        draw.rounded_rectangle([x, y, x + cw, y + ch], radius=12, fill=(255, 255, 255), outline=(190, 200, 212), width=2)
        draw.rounded_rectangle([x, y, x + cw, y + 54], radius=12, fill=hdr_col)
        draw.rectangle([x, y + 36, x + cw, y + 54], fill=hdr_col)
        draw.text((x + 20, y + 13), label, fill=(255, 255, 255), font=f_sub)
        if os.path.exists(img_p):
            im = Image.open(img_p).convert("RGB")
            fitted = _fit_image(im, cw - 36, ch - 82)
            px = x + (cw - fitted.size[0]) // 2
            py = y + 66 + ((ch - 82) - fitted.size[1]) // 2
            canvas.paste(fitted, (px, py))
        else:
            draw.text((x + 30, y + 100), f"Missing: {img_p}", fill=(180, 0, 0), font=f_meta)

    out_p = os.path.join(OUT_DIR, "01_old_vs_new_fig1_and_fig2.png")
    canvas.save(out_p, optimize=True)
    print(f"Saved {out_p}", flush=True)


def _draw_mock_ieee_columns(draw, px0, py0, page_w, y_start, y_end, margin_x, col_gap, dpi):
    """Draw authentic 2-column 10pt IEEE body text lines to give true-to-scale visual weight."""
    col_w = (page_w - 2 * margin_x - col_gap) // 2
    f_body = _font(max(12, int(round(10.0 * dpi / 72.0))), serif=True)
    line_h = int(round(12.2 * dpi / 72.0))
    sample_lines = [
        "In the aircraft pulsating assembly line (APAL),",
        "multiple aircraft advance synchronously across",
        "five physical stations M = {1,...,5} under a",
        "cyclic template horizon H. At each decision step",
        "t, HGP-PPO encodes the four-type heterogeneous",
        "state graph G_t = (O U M U W U K, E_t) via five",
        "relation-specific 4-head GATv2 layers with",
        "disjoint actor (theta) and critic (phi) encoders.",
        "The autoregressive three-tier pointer sequentially",
        "selects operation i_t, station m_t, and worker",
        "team W_t under exact feasibility masking (-10^4).",
        "Under material release disruptions at trigger",
        "tau^r, committed prefix operations retain their",
        "baseline schedules while remaining tasks are",
        "repaired by stability-aware HGP-PPO decoding.",
    ]
    for c_idx in range(2):
        cx = px0 + margin_x + c_idx * (col_w + col_gap)
        yy = y_start
        li = c_idx * 4
        while yy + line_h < y_end:
            txt = sample_lines[li % len(sample_lines)]
            draw.text((cx, yy), txt, fill=(80, 88, 96), font=f_body)
            yy += line_h
            li += 1


def build_ieee_page_sim():
    """Render 3 side-by-side true-to-scale IEEE US Letter pages at 180 DPI (1530 x 1980 px each)."""
    dpi = 180
    pg_w = int(round(8.5 * dpi))     # 1530 px
    pg_h = int(round(11.0 * dpi))    # 1980 px
    text_w = int(round(7.16 * dpi))  # 1289 px (exact IEEEtran double-column \textwidth!)
    margin_x = (pg_w - text_w) // 2  # 120 px
    col_gap = int(round(0.24 * dpi)) # 43 px

    gap_pg = 56
    hdr_h = 110
    total_w = 3 * pg_w + 4 * gap_pg
    total_h = pg_h + hdr_h + 64

    canvas = Image.new("RGB", (total_w, total_h), (235, 239, 245))
    draw = ImageDraw.Draw(canvas)

    f_banner = _font(44, bold=True)
    f_pg_lbl = _font(28, bold=True)
    f_cap    = _font(int(round(8.5 * dpi / 72.0)), serif=True)
    f_cap_b  = _font(int(round(8.5 * dpi / 72.0)), bold=True, serif=True)
    f_run    = _font(int(round(8.0 * dpi / 72.0)), serif=True)

    draw.rectangle([0, 0, total_w, 96], fill=(27, 45, 66))
    draw.text(
        (gap_pg, 24),
        "IEEE TSMC True-to-Scale Double-Column Print Simulation (8.5x11 in Letter, \\textwidth = 7.16 in, 10pt body)",
        fill=(255, 255, 255), font=f_banner,
    )

    pages_spec = [
        (
            "OPTION A (COMPACT): New Fig. 1 + Compact Fig. 2(a)-(b)",
            [
                ("fig1_apal_unfolding_delay.png", 1.00, "Fig. 1.  (a) Five-station pulsating cyclic unfolding (Prop. 1). (b) Real-data 283-operation material delay disruption (tau^r = 65.83 h) and stability-aware repair."),
                ("fig2_compact_b1_b5.png",        1.00, "Fig. 2.  Compact HGP-PPO overview: (a) APAL workcell to heterogeneous state graph G_t (photo adapted from E. Russell, CC BY 2.0); (b) Dual-tower Actor-Critic."),
            ],
        ),
        (
            "OPTION B (HIGH-CLARITY): Standalone Fig. 2 (B1) + Fig. 3 (B5)",
            [
                ("fig2_standalone_b1.png",        1.00, "Fig. 2.  From physical APAL workcell to four-type heterogeneous state graph G_t (photo adapted from E. Russell, Wikimedia Commons, CC BY 2.0)."),
                ("fig3_standalone_b5.png",        0.82, "Fig. 3.  Independent dual-tower HGP-PPO architecture with disjoint 5-layer 4-head GATv2 encoders, 3-tier pointer policy, value head, and PPO coupling."),
            ],
        ),
        (
            "NEW METHOD FIGURE: Appendix A / Sec. IV-C Standalone B8",
            [
                ("fig_appA_b8_worker_pointer.png", 0.96, "Fig. A1.  Autoregressive worker pointer unrolling across substeps r = 1 -> 2, illustrating W_2 self-masking, partial-team pooling, and dynamic EFT/scarcity."),
            ],
        ),
    ]

    for p_idx, (pg_title, figs) in enumerate(pages_spec):
        px0 = gap_pg + p_idx * (pg_w + gap_pg)
        py0 = hdr_h + 18
        # Drop shadow + crisp white IEEE page
        draw.rectangle([px0 + 8, py0 + 8, px0 + pg_w + 8, py0 + pg_h + 8], fill=(195, 202, 212))
        draw.rectangle([px0, py0, px0 + pg_w, py0 + pg_h], fill=(255, 255, 255), outline=(140, 152, 168), width=2)
        # Top label badge above page
        draw.rectangle([px0, py0 - 44, px0 + pg_w, py0], fill=(40, 68, 94))
        draw.text((px0 + 20, py0 - 38), pg_title, fill=(255, 255, 255), font=f_pg_lbl)

        # Running IEEE header
        draw.text((px0 + margin_x, py0 + 46), "IEEE TRANSACTIONS ON SYSTEMS, MAN, AND CYBERNETICS: SYSTEMS", fill=(110, 118, 128), font=f_run)
        draw.line([px0 + margin_x, py0 + 72, px0 + pg_w - margin_x, py0 + 72], fill=(200, 206, 214), width=1)

        cur_y = py0 + 96
        for f_idx, (fname, frac_w, caption) in enumerate(figs):
            im = Image.open(os.path.join(PNG600_DIR, fname)).convert("RGB")
            target_w = int(round(text_w * frac_w))
            target_h = int(round(im.size[1] * (target_w / im.size[0])))
            im_rs = im.resize((target_w, target_h), Image.Resampling.LANCZOS)
            fx = px0 + (pg_w - target_w) // 2
            canvas.paste(im_rs, (fx, cur_y))
            # Subtle dashed IEEE figure width box guide
            draw.rectangle([fx, cur_y, fx + target_w, cur_y + target_h], outline=(220, 226, 234), width=1)
            cur_y += target_h + 14

            # Wrap caption neatly into 2 lines inside text_w
            words = caption.split()
            lines = []
            cur_line = ""
            for w_tok in words:
                test = (cur_line + " " + w_tok).strip()
                if len(test) > 106:
                    lines.append(cur_line)
                    cur_line = w_tok
                else:
                    cur_line = test
            if cur_line:
                lines.append(cur_line)
            for l_i, line_txt in enumerate(lines):
                draw.text((px0 + margin_x, cur_y + l_i * 25), line_txt, fill=(25, 32, 40), font=f_cap_b if l_i == 0 else f_cap)
            cur_y += len(lines) * 25 + 24

            if f_idx == 0 and len(figs) == 2:
                # Brief body text block between top and bottom figures
                band_end = min(cur_y + (135 if p_idx == 1 else 210), py0 + pg_h - 680)
                if band_end > cur_y + 40:
                    _draw_mock_ieee_columns(draw, px0, py0, pg_w, cur_y, band_end, margin_x, col_gap, dpi)
                    cur_y = band_end + 26

        # Fill remaining bottom of IEEE page with authentic 2-column 10pt body text!
        _draw_mock_ieee_columns(draw, px0, py0, pg_w, cur_y + 10, py0 + pg_h - 84, margin_x, col_gap, dpi)

    out_p = os.path.join(OUT_DIR, "02_compact_vs_standalone_ieee_page_sim.png")
    canvas.save(out_p, optimize=True)
    print(f"Saved {out_p}", flush=True)


def build_gallery():
    with open(QA_JSON, "r", encoding="utf-8") as f:
        qa = json.load(f)

    cols, rows = 3, 3
    cw, ch = 1040, 680
    pad = 28
    hdr = 86
    W = cols * cw + (cols + 1) * pad
    H = hdr + rows * ch + (rows + 1) * pad

    canvas = Image.new("RGB", (W, H), (245, 248, 251))
    draw = ImageDraw.Draw(canvas)
    f_hdr  = _font(34, bold=True)
    f_card = _font(20, bold=True)
    f_meta = _font(17, bold=False)

    draw.rectangle([0, 0, W, hdr], fill=(27, 45, 66))
    draw.text((pad, 24), "APAL Paper-Integrated Figures v1 — All 9 Tight-Cropped Publication Exports (Vector PDF + 600 DPI PNG)", fill=(255, 255, 255), font=f_hdr)

    for idx, rec in enumerate(qa):
        r = idx // cols
        c = idx % cols
        x = pad + c * (cw + pad)
        y = hdr + pad + r * (ch + pad)
        is_core = idx < 5
        bar_col = (29, 95, 130) if is_core else (82, 96, 112)
        draw.rounded_rectangle([x, y, x + cw, y + ch], radius=10, fill=(255, 255, 255), outline=(192, 202, 214), width=2)
        draw.rounded_rectangle([x, y, x + cw, y + 48], radius=10, fill=bar_col)
        draw.rectangle([x, y + 32, x + cw, y + 48], fill=bar_col)
        tag = "CORE" if is_core else "EXTRA"
        draw.text((x + 14, y + 12), f"[{idx+1}] {tag}  {rec['slug']}", fill=(255, 255, 255), font=f_card)

        im = Image.open(os.path.join(PREV_DIR, f"{rec['slug']}.png")).convert("RGB")
        fitted = _fit_image(im, cw - 28, ch - 104)
        px = x + (cw - fitted.size[0]) // 2
        py = y + 56 + ((ch - 104) - fitted.size[1]) // 2
        canvas.paste(fitted, (px, py))

        meta_str = (
            f"PDF {rec['pdf_size_in'][0]:.2f}x{rec['pdf_size_in'][1]:.2f} in (AR {rec['aspect_ratio']:.2f}:1)  |  "
            f"@7.16in h={rec['ieee_716in_print_h_in']:.2f} in ({rec['ieee_textheight_pct']:.0f}% \\textheight)  |  "
            f"min font {rec['min_ppt_pt']:.1f}pt (~{rec['min_print_pt_at_716in']:.1f}pt print)"
        )
        draw.text((x + 16, y + ch - 34), meta_str, fill=(55, 68, 82), font=f_meta)

    out_p = os.path.join(OUT_DIR, "03_all_9_figures_gallery.png")
    canvas.save(out_p, optimize=True)
    print(f"Saved {out_p}", flush=True)


def main():
    build_old_vs_new()
    build_ieee_page_sim()
    build_gallery()


if __name__ == "__main__":
    main()
