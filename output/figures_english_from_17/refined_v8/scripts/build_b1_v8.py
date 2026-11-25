"""Build Slide 6 (B1) in refined_v8: 100% paper-faithful terminology aligned with main_tsmc.tex Tables I-III.

v8 terminology & publication polish:
1. Replaced non-paper 'Trade Skill Hub / 5 trades' with exact main_tsmc.tex Table II term:
   'K_2  Skill Node' / 'x_K: skill ID, count, cap., mean eta'.
2. Aligned all four node callouts verbatim with main_tsmc.tex Tables I & II:
   - S_3  Physical Station / x_M: load L_m, ratios, wait, c_m slots
   - O_3  Operation (wing join, q=2) / x_O: p_tilde_i, status, skill, q_i, r_i
   - K_2  Skill Node / x_K: skill ID, count, cap., mean eta
   - W_5  Multi-Skilled Worker / x_W: eta_w, 5-d skills, wait, idle, l_w
3. Fixed graph relation label 'assigned' -> 'assigned-to' (exact main_tsmc.tex Table III relation name).
"""
from __future__ import annotations

import argparse
import os
import sys
from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REFINED_V8_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from pptx_helpers import (
    BLACK,
    FONT_SANS,
    FONT_SERIF,
    PALETTE,
    add_box,
    add_circle,
    add_line,
    add_swatch_strip,
    add_textbox,
    clear_slide,
    rgb,
    set_notes,
)

DEFAULT_PPTX_PATH = os.path.join(REFINED_V8_DIR, "APAL_Modules_English_Refined_v8.pptx")
PHOTO_PATH = os.path.join(REFINED_V8_DIR, "assets", "b1_apal_workcell_v4_ccby2.png")


def _set_fill_alpha(shape, alpha_pct: int = 94):
    spPr = shape._element.find(qn("p:spPr"))
    if spPr is None:
        return
    srgb = spPr.find(".//" + qn("a:srgbClr"))
    if srgb is not None:
        alpha = OxmlElement("a:alpha")
        alpha.set("val", str(alpha_pct * 1000))
        srgb.append(alpha)


def _add_pointer_call(slide, x1, y1, tx, ty, color):
    add_line(slide, x1, y1, tx, ty, color=PALETTE["white"], width_pt=4.4)
    add_line(slide, x1, y1, tx, ty, color=color, width_pt=2.5, arrow_end="triangle", arrow_size="med")
    add_circle(slide, tx, ty, 0.095, fill=color, border=PALETTE["white"], border_pt=1.7)


def build_b1_v8(slide):
    """Slide 6 · B1 publication-ready paper figure (100% aligned with main_tsmc.tex Tables I-III)."""
    clear_slide(slide)

    im = Image.open(PHOTO_PATH)
    pw_px, ph_px = im.size
    photo_w = 8.46
    photo_h = photo_w * (ph_px / pw_px)   # 5.693 in
    gap_l = 0.03
    arr_w = 0.26
    gap_r = 0.03
    gw = 3.44
    total_w = photo_w + gap_l + arr_w + gap_r + gw   # 12.22 in

    photo_x = (13.333 - total_w) / 2.0    # 0.5565 in
    photo_y = (7.50 - photo_h) / 2.0      # 0.9035 in

    slide.shapes.add_picture(
        PHOTO_PATH,
        Inches(photo_x),
        Inches(photo_y),
        width=Inches(photo_w),
        height=Inches(photo_h),
    )
    add_box(slide, photo_x, photo_y, photo_w, photo_h, fill=None, border=rgb("#7F8C99"), border_pt=0.95)

    def pxy(u: float, v: float):
        return (photo_x + u * photo_w, photo_y + v * photo_h)

    # ---- FOUR PAPER-FAITHFUL, SNUG IN-PHOTO CALLOUTS (main_tsmc.tex Tables I & II) ----
    # 1. Station S3 (STAGE 2 tower at u=0.205, v=0.335)
    tx_s3, ty_s3 = pxy(0.205, 0.335)
    _add_pointer_call(slide, tx_s3, photo_y + 0.84, tx_s3, ty_s3, PALETTE["st_dark"])
    sh_s3 = add_box(
        slide, photo_x + 0.14, photo_y + 0.14, 2.98, 0.72,
        [
            [("S", "b", PALETTE["st_dark"], 16.5), ("3", "b_", PALETTE["st_dark"], 16.5), ("  Physical Station", "bsans", BLACK, 15.0)],
            [("x", "bi", PALETTE["st_dark"], 13.4), ("M", "b_", PALETTE["st_dark"], 13.4), (": load L", "bsans", rgb("#1F2933"), 12.2), ("m", "b_", rgb("#1F2933"), 12.2), (", ratios, wait, c", "bsans", rgb("#1F2933"), 12.2), ("m", "b_", rgb("#1F2933"), 12.2), (" slots", "bsans", rgb("#1F2933"), 12.2)],
        ],
        fill=rgb("#DFF3EE"), border=PALETTE["st_dark"], border_pt=1.9, rounded=True, round_radius=0.20,
        line_spacing=1.02, margin_lr=0.04, margin_tb=0.01,
    )
    _set_fill_alpha(sh_s3, 94)

    # 2. Operation O3 (wing-body join at u=0.675, v=0.375)
    tx_o3, ty_o3 = pxy(0.675, 0.375)
    _add_pointer_call(slide, tx_o3, photo_y + 0.84, tx_o3, ty_o3, PALETTE["op_red_dark"])
    sh_o3 = add_box(
        slide, photo_x + 3.92, photo_y + 0.14, 3.60, 0.72,
        [
            [("O", "b", PALETTE["op_red_dark"], 16.5), ("3", "b_", PALETTE["op_red_dark"], 16.5), ("  Operation (wing join, q=2)", "bsans", BLACK, 14.6)],
            [("x", "bi", PALETTE["op_red_dark"], 13.4), ("O", "b_", PALETTE["op_red_dark"], 13.4), (": 18d/24d (p\u0303", "bsans", rgb("#1F2933"), 12.2), ("i", "b_", rgb("#1F2933"), 12.2), (", status, k", "bsans", rgb("#1F2933"), 12.2), ("i", "b_", rgb("#1F2933"), 12.2), (", r", "bsans", rgb("#1F2933"), 12.2), ("i", "b_", rgb("#1F2933"), 12.2), (")", "bsans", rgb("#1F2933"), 12.2)],
        ],
        fill=rgb("#FCE3E0"), border=PALETTE["op_red_dark"], border_pt=1.9, rounded=True, round_radius=0.20,
        line_spacing=1.02, margin_lr=0.04, margin_tb=0.01,
    )
    _set_fill_alpha(sh_o3, 94)

    # 3. Skill Node K2 (structural skill group at u=0.268, v=0.615) — EXACT Table II paper term!
    tx_k2, ty_k2 = pxy(0.268, 0.615)
    _add_pointer_call(slide, photo_x + 1.56, photo_y + photo_h - 0.84, tx_k2, ty_k2, PALETTE["sk_dark"])
    sh_k2 = add_box(
        slide, photo_x + 0.14, photo_y + photo_h - 0.86, 2.86, 0.72,
        [
            [("K", "b", PALETTE["sk_dark"], 16.5), ("2", "b_", PALETTE["sk_dark"], 16.5), ("  Skill Node  (|K|=5)", "bsans", BLACK, 14.6)],
            [("x", "bi", PALETTE["sk_dark"], 13.4), ("K", "b_", PALETTE["sk_dark"], 13.4), (": skill ID, count, cap., mean \u03b7", "bsans", rgb("#1F2933"), 12.0)],
        ],
        fill=rgb("#ECE2F7"), border=PALETTE["sk_dark"], border_pt=1.9, rounded=True, round_radius=0.20,
        line_spacing=1.02, margin_lr=0.04, margin_tb=0.01,
    )
    _set_fill_alpha(sh_k2, 94)

    # 4. Multi-Skilled Worker W5 (black-shirt technician at u=0.803, v=0.805) — EXACT Table II paper term!
    tx_w5, ty_w5 = pxy(0.803, 0.805)
    _add_pointer_call(slide, photo_x + 6.32, photo_y + photo_h - 0.50, tx_w5, ty_w5, PALETTE["wk_dark"])
    sh_w5 = add_box(
        slide, photo_x + 3.20, photo_y + photo_h - 0.86, 3.16, 0.72,
        [
            [("W", "b", PALETTE["wk_dark"], 16.5), ("5", "b_", PALETTE["wk_dark"], 16.5), ("  Multi-Skilled Worker", "bsans", BLACK, 14.6)],
            [("x", "bi", PALETTE["wk_dark"], 13.4), ("W", "b_", PALETTE["wk_dark"], 13.4), (": \u03b7", "bsans", rgb("#1F2933"), 12.0), ("w", "b_", rgb("#1F2933"), 12.0), (", 5-d skills, wait, idle, \u2113", "bsans", rgb("#1F2933"), 12.0), ("w", "b_", rgb("#1F2933"), 12.0)],
        ],
        fill=rgb("#FBEBD4"), border=PALETTE["wk_dark"], border_pt=1.9, rounded=True, round_radius=0.20,
        line_spacing=1.02, margin_lr=0.04, margin_tb=0.01,
    )
    _set_fill_alpha(sh_w5, 94)

    # =========================================================================
    # TIGHT SLEEK WANG & GOMBOLAY TRANSITION ARROW (total gap = 0.32 in)
    # =========================================================================
    arr_x = photo_x + photo_w + gap_l
    arr_h = 0.42
    arr = slide.shapes.add_shape(
        MSO_SHAPE.RIGHT_ARROW,
        Inches(arr_x),
        Inches(photo_y + photo_h / 2.0 - arr_h / 2.0),
        Inches(arr_w),
        Inches(arr_h),
    )
    arr.adjustments[0] = 0.46
    arr.adjustments[1] = 0.52
    arr.fill.solid()
    arr.fill.fore_color.rgb = rgb("#DED3EA")
    arr.line.color.rgb = rgb("#5B4A73")
    arr.line.width = Pt(1.45)
    arr.shadow.inherit = False

    # =========================================================================
    # RIGHT PANEL: HETEROGENEOUS STATE GRAPH G_t [w = 3.44 in]
    # =========================================================================
    gx0 = arr_x + arr_w + gap_r
    gy0 = photo_y
    gh = photo_h

    add_box(slide, gx0, gy0, gw, gh, fill=rgb("#EBF2FA"), border=rgb("#3F638E"), border_pt=1.65, rounded=True, round_radius=0.04)

    add_textbox(
        slide, gx0 + 0.04, gy0 + 0.08, gw - 0.08, 0.34,
        [[("Heterogeneous State Graph ", "bsans", BLACK, 14.0), ("G", "bi", BLACK, 15.2), ("t", "b_", BLACK, 15.2)]],
        font_name=FONT_SERIF, size_pt=14.0, align=PP_ALIGN.CENTER, margin_lr=0.0, word_wrap=False,
    )
    add_box(
        slide, gx0 + 0.08, gy0 + 0.44, gw - 0.16, 0.42,
        [[("h", "bi"), ("v", "b_"), ("(0)", "b^"), (" = f", "b"), ("X", "b_"), ("(x", "b"), ("v", "b_"), (") \u2208 \u211d", "b"), ("128", "b^"), (" (LN\u2192LReLU)", "bsans", rgb("#334150"), 10.6)]],
        fill=rgb("#FDF5DC"), border=BLACK, border_pt=1.25, rounded=True, round_radius=0.18,
        color=BLACK, size_pt=12.8, margin_lr=0.01, margin_tb=0.0,
    )

    n_s2 = (gx0 + 0.42, gy0 + 1.40)
    n_s3 = (gx0 + 1.60, gy0 + 1.40)
    n_o1 = (gx0 + 0.42, gy0 + 2.40)
    n_o2 = (gx0 + 0.42, gy0 + 3.24)
    n_o3 = (gx0 + 1.40, gy0 + 2.75)
    n_k2 = (gx0 + 1.40, gy0 + 3.78)
    n_w2 = (gx0 + 0.62, gy0 + 4.54)
    n_w5 = (gx0 + 1.76, gy0 + 4.54)
    rg = 0.245

    # Crisp thick typed edges matching Table III relations
    add_line(slide, n_o1[0], n_o1[1] - rg, n_s2[0], n_s2[1] + rg + 0.015, color=PALETTE["st_dark"], width_pt=1.85, dash="dash", arrow_end="triangle", arrow_size="sm")
    add_line(slide, n_o1[0] + rg - 0.02, n_o1[1] + 0.06, n_o3[0] - rg - 0.02, n_o3[1] - 0.06, color=PALETTE["op_red_dark"], width_pt=1.9, arrow_end="triangle", arrow_size="sm")
    add_line(slide, n_o2[0] + rg - 0.02, n_o2[1] - 0.08, n_o3[0] - rg - 0.02, n_o3[1] + 0.08, color=PALETTE["op_red_dark"], width_pt=1.9, arrow_end="triangle", arrow_size="sm")
    add_line(slide, n_o3[0], n_o3[1] + rg + 0.02, n_k2[0], n_k2[1] - rg - 0.02, color=PALETTE["sk_dark"], width_pt=1.9, arrow_start="triangle", arrow_end="triangle", arrow_size="sm")
    add_line(slide, n_w2[0] + 0.18, n_w2[1] - 0.17, n_k2[0] - 0.17, n_k2[1] + 0.17, color=PALETTE["wk_dark"], width_pt=1.75, arrow_start="triangle", arrow_end="triangle", arrow_size="sm")
    add_line(slide, n_w5[0] - 0.10, n_w5[1] - 0.21, n_k2[0] + 0.10, n_k2[1] + 0.21, color=PALETTE["wk_dark"], width_pt=1.75, arrow_start="triangle", arrow_end="triangle", arrow_size="sm")

    add_circle(slide, n_s2[0], n_s2[1], rg, [[("S", "b"), ("2", "b_")]], fill=rgb("#DFF3EE"), border=BLACK, border_pt=1.35, color=PALETTE["st_dark"], size_pt=13.4)
    add_circle(slide, n_s3[0], n_s3[1], rg, [[("S", "b"), ("3", "b_")]], fill=PALETTE["st_dark"], border=BLACK, border_pt=1.85, color=PALETTE["white"], size_pt=13.4)
    add_circle(slide, n_o1[0], n_o1[1], rg, [[("O", "b"), ("1", "b_")]], fill=rgb("#FCE3E0"), border=BLACK, border_pt=1.35, color=PALETTE["op_red_dark"], size_pt=13.4)
    add_circle(slide, n_o2[0], n_o2[1], rg, [[("O", "b"), ("2", "b_")]], fill=rgb("#FCE3E0"), border=BLACK, border_pt=1.35, color=PALETTE["op_red_dark"], size_pt=13.4)
    add_circle(slide, n_o3[0], n_o3[1], rg, [[("O", "b"), ("3", "b_")]], fill=PALETTE["op_red_dark"], border=BLACK, border_pt=1.85, color=PALETTE["white"], size_pt=13.4)
    add_circle(slide, n_k2[0], n_k2[1], rg, [[("K", "b"), ("2", "b_")]], fill=PALETTE["sk_dark"], border=BLACK, border_pt=1.85, color=PALETTE["white"], size_pt=13.4)
    add_circle(slide, n_w2[0], n_w2[1], rg, [[("W", "b"), ("2", "b_")]], fill=rgb("#FBEBD4"), border=BLACK, border_pt=1.35, color=PALETTE["wk_dark"], size_pt=13.4)
    add_circle(slide, n_w5[0], n_w5[1], rg, [[("W", "b"), ("5", "b_")]], fill=PALETTE["wk_dark"], border=BLACK, border_pt=1.85, color=PALETTE["white"], size_pt=13.4)

    # Exact Table III relation names ('assigned-to', 'precedes', 'requires', 'has-skill')
    add_textbox(slide, n_s2[0] + 0.04, gy0 + 1.85, 0.90, 0.24, [[("assigned-to", "bsans")]], font_name=FONT_SANS, size_pt=9.6, color=PALETTE["st_dark"], align=PP_ALIGN.LEFT, margin_lr=0.0, word_wrap=False)
    add_textbox(slide, n_s3[0] - 0.50, n_s3[1] + rg + 0.02, 1.04, 0.22, [[("open station", "isans")]], font_name=FONT_SANS, size_pt=9.8, color=PALETTE["st_dark"], align=PP_ALIGN.CENTER, margin_lr=0.0, word_wrap=False)
    add_textbox(slide, gx0 + 0.66, gy0 + 2.21, 0.80, 0.24, [[("precedes", "bsans")]], font_name=FONT_SANS, size_pt=10.2, color=PALETTE["op_red_dark"], align=PP_ALIGN.CENTER, margin_lr=0.0, word_wrap=False)
    add_textbox(slide, gx0 + 0.58, gy0 + 3.16, 0.76, 0.24, [[("requires", "bsans")]], font_name=FONT_SANS, size_pt=10.2, color=PALETTE["sk_dark"], align=PP_ALIGN.RIGHT, margin_lr=0.0, word_wrap=False)
    add_textbox(slide, gx0 + 0.82, gy0 + 4.24, 0.78, 0.24, [[("has-skill", "bsans")]], font_name=FONT_SANS, size_pt=10.2, color=PALETTE["wk_dark"], align=PP_ALIGN.CENTER, margin_lr=0.0, word_wrap=False)

    # Close-attached 4-cell latent vectors h^(0)(.)
    vx = gx0 + 2.38
    add_textbox(slide, vx - 0.08, gy0 + 0.96, 1.08, 0.26, [[("Latent ", "bsans"), ("h", "bi"), ("v", "b_"), ("(0)", "b^")]], font_name=FONT_SERIF, size_pt=11.4, color=BLACK, align=PP_ALIGN.CENTER, margin_lr=0.0, word_wrap=False)
    vec_specs = [
        (n_s3, "S", "3", PALETTE["st_dark"], [rgb("#84BEB4"), rgb("#1D6F63"), rgb("#C5E2DD"), rgb("#4B998C")]),
        (n_o3, "O", "3", PALETTE["op_red_dark"], [rgb("#D88E87"), rgb("#A63D33"), rgb("#EBC4BF"), rgb("#C4665D")]),
        (n_k2, "K", "2", PALETTE["sk_dark"], [rgb("#B69AC9"), rgb("#6D458E"), rgb("#DCCEE8"), rgb("#916BB0")]),
        (n_w5, "W", "5", PALETTE["wk_dark"], [rgb("#DFB182"), rgb("#A85C1E"), rgb("#F0D7BC"), rgb("#C98443")]),
    ]
    for (nx, ny), sym, sub, col_dk, sw_cols in vec_specs:
        add_line(slide, nx + rg + 0.02, ny, vx - 0.03, ny, color=col_dk, width_pt=1.35, dash="dot")
        add_swatch_strip(slide, vx, ny - 0.145, sw_cols, cell_w=0.22, cell_h=0.29, gap=0.015, border=BLACK, border_pt=0.7)
        add_textbox(
            slide, vx - 0.08, ny + 0.15, 1.08, 0.25,
            [[("h", "bi"), ("(0)", "b^"), (f"({sym}", "b"), (sub, "b_"), (")", "b")]],
            font_name=FONT_SERIF, size_pt=11.0, color=col_dk, align=PP_ALIGN.CENTER, margin_lr=0.0, word_wrap=False,
        )

    # Self-explanatory visual icon legend
    leg_y = gy0 + gh - 0.66
    leg_h = 0.56
    add_box(
        slide, gx0 + 0.08, leg_y, gw - 0.16, leg_h,
        fill=PALETTE["white"], border=rgb("#9BB2CC"), border_pt=1.05, rounded=True, round_radius=0.18,
    )
    r1y = leg_y + 0.15
    add_circle(slide, gx0 + 0.22, r1y, 0.075, fill=PALETTE["st_dark"], border=BLACK, border_pt=1.15)
    add_textbox(slide, gx0 + 0.33, r1y - 0.11, 1.40, 0.22, [[("Highlighted in photo", "bsans")]], font_name=FONT_SANS, size_pt=9.6, color=BLACK, align=PP_ALIGN.LEFT, margin_lr=0.0, word_wrap=False)
    add_circle(slide, gx0 + 1.82, r1y, 0.075, fill=rgb("#FCE3E0"), border=BLACK, border_pt=1.0)
    add_textbox(slide, gx0 + 1.93, r1y - 0.11, 1.32, 0.22, [[("Other graph node", "bsans")]], font_name=FONT_SANS, size_pt=9.6, color=rgb("#334150"), align=PP_ALIGN.LEFT, margin_lr=0.0, word_wrap=False)

    r2y = leg_y + 0.41
    add_line(slide, gx0 + 0.14, r2y, gx0 + 0.39, r2y, color=PALETTE["op_red_dark"], width_pt=1.7, arrow_end="triangle", arrow_size="sm")
    add_textbox(slide, gx0 + 0.43, r2y - 0.11, 1.24, 0.22, [[("Static/fixed edge", "bsans")]], font_name=FONT_SANS, size_pt=9.6, color=BLACK, align=PP_ALIGN.LEFT, margin_lr=0.0, word_wrap=False)
    add_line(slide, gx0 + 1.74, r2y, gx0 + 1.99, r2y, color=PALETTE["st_dark"], width_pt=1.7, dash="dash", arrow_end="triangle", arrow_size="sm")
    add_textbox(slide, gx0 + 2.03, r2y - 0.11, 1.24, 0.22, [[("Dynamic assign.", "bsans")]], font_name=FONT_SANS, size_pt=9.6, color=PALETTE["st_dark"], align=PP_ALIGN.LEFT, margin_lr=0.0, word_wrap=False)

    set_notes(
        slide,
        "Module B1. From the physical APAL workcell to a four-type heterogeneous state graph G_t = (O U M U W U K, E_t) (main_tsmc.tex Tables I-III; FeatureEmbedder in models/hb_gat_pn.py; rescheduling expands operation features 18d -> 24d). Refined v8 aligns every entity name, state feature symbol (L_m, c_m, p_tilde_i, k_i, r_i, eta_w, l_w), and relation label ('assigned-to', 'precedes', 'requires', 'has-skill') verbatim with the IEEE TSMC manuscript, replacing non-paper 'Trade Skill Hub' with 'Skill Node (|K|=5)'.\n"
        "Legal & Academic Photo Attribution (place in paper LaTeX figure caption): 'Assembly floor photograph adapted from Edward Russell (Wikimedia Commons, CC BY 2.0).' Source: https://commons.wikimedia.org/wiki/File:Final_assembly_of_the_2nd_787-10_(32910009961).jpg\n"
        "Layout reference: Zheyuan Wang and Matthew Gombolay, Heterogeneous Graph Attention Networks for Scalable Multi-Robot Scheduling with Temporospatial Constraints, RSS 2020, Fig. 1.",
    )


def main():
    parser = argparse.ArgumentParser(description="Build Slide 6 (B1) for refined_v8.")
    parser.add_argument("--pptx", default=DEFAULT_PPTX_PATH, help="Target PPTX path.")
    args = parser.parse_args()
    prs = Presentation(args.pptx)
    build_b1_v8(prs.slides[5])  # Slide 6: B1
    prs.save(args.pptx)
    print("Updated Slide 6 (B1) in", args.pptx)


if __name__ == "__main__":
    main()
