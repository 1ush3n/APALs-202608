"""Build Slide 15 (B10) in refined_v8: 100% paper-faithful Critic Value Head (main_tsmc.tex Sections IV-B & IV-D).

v8 terminology & module-naming polish addressing User Comment 2 + full paper audit:
1. Replaced awkward 'Hidden Unit j (illustrative negative-return trace)' with clean formal IEEE module name:
   'Cross-Type Feature Fusion & Activation  (Channel j)' (no parenthetical disclaimer inside the figure!).
2. Replaced non-paper 'Skill hubs K (5) / folded in H^O, H^W' with exact paper wording:
   'Skill nodes K (5)  --- 5x GATv2 --->  encoded in H^O, H^W'.
3. Aligned Panel (a) title & row notes with main_tsmc.tex:
   - '(a) Attention Pooling  H_phi^(5) -> g_phi in R^384'
   - Replaced invented '.12-.31 (5 st.)' with exact '5 physical stations'; 'skill-aware' -> 'multi-skilled pool'.
4. Fixed mathematical term on Panel (c): 'GAE delta_t' -> 'TD Error delta_t' (exact main_tsmc.tex line 361-363!).
5. Preserved all v7 gains: uniform `in R^{...}` notation, zero code/slide IDs, 100% rounded rectangles in (b)&(c),
   zero bottom void inside zoom box, min_font >= 9.8pt, and wide 0.44 in 3-lane (a)->(b) colored routing bus.
"""
from __future__ import annotations

import argparse
import os
import sys
from pptx import Presentation
from pptx.enum.text import PP_ALIGN

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REFINED_V8_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from pptx_helpers import (
    BLACK,
    FONT_SANS,
    FONT_SERIF,
    PALETTE,
    add_activation_glyph,
    add_box,
    add_circle,
    add_line,
    add_polyline,
    add_textbox,
    clear_slide,
    rgb,
    set_notes,
)

DEFAULT_PPTX_PATH = os.path.join(REFINED_V8_DIR, "APAL_Modules_English_Refined_v8.pptx")

C_ST_BG  = rgb("#E0F2EE")
C_ST_DK  = rgb("#1A6B5E")
C_OP_BG  = rgb("#E4EFFA")
C_OP_DK  = rgb("#215384")
C_WK_BG  = rgb("#FBEEDB")
C_WK_DK  = rgb("#A15518")
C_SK_BG  = rgb("#EFE8F7")
C_SK_DK  = rgb("#683D8E")
C_BLUE   = rgb("#CFE3F6")
C_MINT   = rgb("#DBF0D5")
C_PEACH  = rgb("#FAD2B4")
C_SAND   = rgb("#F5E7BA")


def build_b10_v8(slide):
    """Slide 15 · B10 publication-ready Critic Value Head (refined_v8, 100% paper-faithful)."""
    clear_slide(slide)

    y_top, h_pan = 1.42, 4.66

    ax, aw = 0.517, 3.74
    gutter_ab = 0.44
    bx, bw = ax + aw + gutter_ab, 4.64
    gutter_bc = 0.22
    cx, cw = bx + bw + gutter_bc, 3.26

    # =========================================================================
    # PANEL (a): INDEPENDENT CRITIC ATTENTION POOLING [w = 3.74 in]
    # =========================================================================
    add_box(slide, ax, y_top, aw, h_pan, fill=rgb("#FCFDFF"), border=BLACK, border_pt=1.7, rounded=True, round_radius=0.04)
    add_textbox(
        slide, ax + 0.04, y_top + 0.06, aw - 0.08, 0.28,
        [[("(a) Attention Pooling  ", "bsans"), ("H", "b"), ("\u03c6", "b_"), ("(5)", "b^"), (" \u2192 ", "b"), ("g", "bi"), ("\u03c6", "b_"), (" \u2208 \u211d", "b"), ("384", "b^")]],
        font_name=FONT_SERIF, size_pt=11.8, color=BLACK, align=PP_ALIGN.CENTER, margin_lr=0.0, margin_tb=0.0,
    )

    add_box(slide, ax + 0.08, 1.79, aw - 0.16, 0.65, fill=rgb("#F3F7FB"), border=BLACK, border_pt=1.25, rounded=True, round_radius=0.14)
    y_sc = 1.99
    x_p = ax + 0.14
    add_box(slide, x_p, y_sc - 0.135, 0.52, 0.27, [[("128d", "bsans")]], fill=PALETTE["white"], border=BLACK, border_pt=1.1, rounded=True, round_radius=0.18, color=BLACK, size_pt=9.8, margin_lr=0.0, margin_tb=0.0)
    add_line(slide, x_p + 0.52, y_sc, x_p + 0.63, y_sc, color=BLACK, width_pt=1.2, arrow_end="triangle", arrow_size="sm")
    add_box(slide, x_p + 0.64, y_sc - 0.135, 1.24, 0.27, [[("32d LN+LReLU", "bsans")]], fill=C_BLUE, border=BLACK, border_pt=1.1, rounded=True, round_radius=0.18, color=BLACK, size_pt=9.8, margin_lr=0.0, margin_tb=0.0)
    add_line(slide, x_p + 1.88, y_sc, x_p + 1.99, y_sc, color=BLACK, width_pt=1.2, arrow_end="triangle", arrow_size="sm")
    add_box(slide, x_p + 2.00, y_sc - 0.135, 0.40, 0.27, [[("1d", "bsans")]], fill=C_PEACH, border=BLACK, border_pt=1.1, rounded=True, round_radius=0.18, color=BLACK, size_pt=9.8, margin_lr=0.0, margin_tb=0.0)
    add_line(slide, x_p + 2.40, y_sc, x_p + 2.51, y_sc, color=BLACK, width_pt=1.2, arrow_end="triangle", arrow_size="sm")
    add_box(slide, x_p + 2.52, y_sc - 0.135, 0.94, 0.27, [[("softmax \u03b2", "bsans")]], fill=C_SAND, border=BLACK, border_pt=1.1, rounded=True, round_radius=0.18, color=BLACK, size_pt=9.8, margin_lr=0.0, margin_tb=0.0)
    add_textbox(
        slide, ax + 0.08, 2.18, aw - 0.16, 0.23,
        [[
            ("Station Attn. (M)", "bsans", C_ST_DK, 9.8),
            (" | ", "sans", PALETTE["line_gray"], 9.8),
            ("Shared Op/Worker Attn. (O, W)", "bsans", C_OP_DK, 9.8),
        ]],
        align=PP_ALIGN.CENTER, margin_lr=0.0, margin_tb=0.0,
    )

    # Paper-faithful descriptors (ZERO invented numbers!)
    pool_rows = [
        ("Station  M", "|M| = 5 \u00d7 128", "\u03b2", "5 physical stations", "g", "M", "[0:128]",   C_ST_BG, C_ST_DK, 2.95),
        ("Operation  O", "|O| = N \u00d7 128", "\u03b2", "N = 283\u20133,182", "g", "O", "[128:256]", C_OP_BG, C_OP_DK, 3.90),
        ("Worker  W", "|W| \u00d7 128", "\u03b2", "multi-skilled pool",  "g", "W", "[256:384]", C_WK_BG, C_WK_DK, 4.85),
    ]
    g_box_right = ax + 3.64
    for title_t, dim_t, b_sym, w_note, g_sym, g_sup, sl_txt, c_bg, c_dk, yc in pool_rows:
        add_box(
            slide, ax + 0.08, yc - 0.28, 1.20, 0.56,
            [[(title_t, "bsans", c_dk, 10.8)], [(dim_t, "sans", BLACK, 9.8)]],
            fill=c_bg, border=BLACK, border_pt=1.35, rounded=True, round_radius=0.14, line_spacing=1.04, margin_lr=0.01, margin_tb=0.01,
        )
        add_line(slide, ax + 1.28, yc, ax + 1.46, yc, color=c_dk, width_pt=1.4, arrow_end="triangle", arrow_size="sm")
        add_circle(slide, ax + 1.67, yc, 0.195, [[("\u03a3", "b"), (b_sym, "b_")]], fill=PALETTE["white"], border=c_dk, border_pt=1.55, color=c_dk, size_pt=12.4)
        add_textbox(slide, ax + 1.26, yc - 0.425, 0.82, 0.20, [[("softmax", "isans")]], font_name=FONT_SANS, size_pt=9.8, color=c_dk, align=PP_ALIGN.CENTER, margin_lr=0.0, margin_tb=0.0)
        add_textbox(slide, ax + 0.96, yc + 0.270, 1.42, 0.20, [[(w_note, "sans")]], font_name=FONT_SANS, size_pt=9.8, color=PALETTE["ink_soft"], align=PP_ALIGN.CENTER, margin_lr=0.0, margin_tb=0.0)
        add_line(slide, ax + 1.87, yc, ax + 2.06, yc, color=c_dk, width_pt=1.4, arrow_end="triangle", arrow_size="sm")
        add_box(
            slide, ax + 2.08, yc - 0.28, 1.56, 0.56,
            [[(g_sym, "bi", c_dk, 13.0), (g_sup, "b^", c_dk, 13.0), (" \u2208 \u211d", "b", BLACK, 11.4), ("128", "b^", BLACK, 11.4)], [("dims " + sl_txt, "sans", PALETTE["ink_soft"], 9.8)]],
            fill=c_bg, border=BLACK, border_pt=1.4, rounded=True, round_radius=0.14, line_spacing=1.02, margin_lr=0.01, margin_tb=0.01,
        )

    # Exact paper term: 'Skill nodes K (5)' -> 'encoded in H^O, H^W'
    y_sk = 5.74
    add_box(slide, ax + 0.08, y_sk - 0.19, 1.28, 0.38, [[("Skill nodes K (5)", "bsans")]], fill=C_SK_BG, border=C_SK_DK, border_pt=1.25, rounded=True, round_radius=0.18, color=C_SK_DK, size_pt=9.8, margin_lr=0.0, margin_tb=0.0)
    add_line(slide, ax + 1.36, y_sk, ax + 2.24, y_sk, color=C_SK_DK, width_pt=1.4, dash="dash", arrow_end="triangle", arrow_size="sm")
    add_textbox(slide, ax + 1.32, y_sk - 0.27, 0.96, 0.21, [[("5\u00d7 GATv2", "bsans")]], font_name=FONT_SANS, size_pt=9.8, color=C_SK_DK, align=PP_ALIGN.CENTER, margin_lr=0.0, margin_tb=0.0)
    add_box(slide, ax + 2.26, y_sk - 0.19, 1.38, 0.38, [[("encoded in H", "bsans"), ("O", "b^"), (", H", "bsans"), ("W", "b^")]], fill=rgb("#F5F2FA"), border=BLACK, border_pt=1.2, rounded=True, round_radius=0.18, color=BLACK, size_pt=9.8, margin_lr=0.0, margin_tb=0.0)

    # =========================================================================
    # PANEL (b): TYPE-BLOCKED LINEAR PROJECTION + CHANNEL-j CROSS-TYPE FUSION
    # =========================================================================
    add_box(slide, bx, y_top, bw, h_pan, fill=PALETTE["white"], border=BLACK, border_pt=1.7, rounded=True, round_radius=0.04)
    add_textbox(
        slide, bx + 0.05, y_top + 0.06, bw - 0.10, 0.28,
        [[
            ("(b) Type-Blocked  ", "bsans"), ("z = W", "bi"), ("1", "b_"), (" g", "bi"), ("\u03c6", "b_"), (" + b", "b"), ("1", "b_"),
            ("  (", "sans", PALETTE["ink_soft"], 10.6), ("W", "b", PALETTE["ink_soft"], 10.6), ("1", "b_", PALETTE["ink_soft"], 10.6),
            (" \u2208 \u211d", "b", PALETTE["ink_soft"], 10.6), ("64\u00d7384", "b^", PALETTE["ink_soft"], 10.6), (")", "sans", PALETTE["ink_soft"], 10.6),
        ]],
        font_name=FONT_SERIF, size_pt=12.0, color=BLACK, align=PP_ALIGN.CENTER, margin_lr=0.0, margin_tb=0.0,
    )

    wy0 = 2.08
    row_h = 0.340
    row_gap = 0.030
    tot_h = 5 * row_h - row_gap
    y_mid_w = wy0 + 2 * row_h + (row_h - row_gap) / 2.0

    gx = bx + 0.13
    gw_vec = 0.48
    g_gap = 0.030
    cell_h_g = (tot_h - 2 * g_gap) / 3.0
    add_textbox(
        slide, gx - 0.14, 1.81, gw_vec + 0.30, 0.24,
        [[("g", "bi"), ("\u03c6", "b_"), (" \u2208 \u211d", "b"), ("384", "b^")]],
        font_name=FONT_SERIF, size_pt=10.0, color=BLACK, align=PP_ALIGN.CENTER, margin_lr=0.0, margin_tb=0.0,
    )
    g_cells = [
        ([("g", "bi", C_ST_DK), ("M", "b^", C_ST_DK)], C_ST_BG),
        ([("g", "bi", C_OP_DK), ("O", "b^", C_OP_DK)], C_OP_BG),
        ([("g", "bi", C_WK_DK), ("W", "b^", C_WK_DK)], C_WK_BG),
    ]
    g_centers = []
    for g_idx, (g_segs, g_fill) in enumerate(g_cells):
        gy = wy0 + g_idx * (cell_h_g + g_gap)
        g_centers.append(gy + cell_h_g / 2.0)
        add_box(
            slide, gx, gy, gw_vec, cell_h_g, [g_segs],
            fill=g_fill, border=BLACK, border_pt=1.35, rounded=True, round_radius=0.16,
            size_pt=12.4, margin_lr=0.0, margin_tb=0.0,
        )

    # Wide 0.44 in 3-lane colored orthogonal routing bus
    x_lane_m = ax + aw + 0.09
    x_lane_o = ax + aw + 0.19
    x_lane_w = ax + aw + 0.29
    add_polyline(slide, [(g_box_right, 2.95), (x_lane_m, 2.95), (x_lane_m, g_centers[0]), (gx - 0.02, g_centers[0])], color=C_ST_DK, width_pt=1.9, arrow_end="triangle", arrow_size="med")
    add_polyline(slide, [(g_box_right, 3.90), (x_lane_o, 3.90), (x_lane_o, g_centers[1]), (gx - 0.02, g_centers[1])], color=C_OP_DK, width_pt=1.9, arrow_end="triangle", arrow_size="med")
    add_polyline(slide, [(g_box_right, 4.85), (x_lane_w, 4.85), (x_lane_w, g_centers[2]), (gx - 0.02, g_centers[2])], color=C_WK_DK, width_pt=1.9, arrow_end="triangle", arrow_size="med")

    wx0 = gx + gw_vec + 0.20
    add_line(slide, gx + gw_vec, y_mid_w, wx0 - 0.03, y_mid_w, color=BLACK, width_pt=1.45, arrow_end="triangle", arrow_size="sm")

    blk_w, blk_gap = 0.98, 0.045
    blk_defs = [
        ("M", C_ST_BG, C_ST_DK, rgb("#2B8A7A")),
        ("O", C_OP_BG, C_OP_DK, rgb("#3671A8")),
        ("W", C_WK_BG, C_WK_DK, rgb("#C26E24")),
    ]
    for b_idx, (t_sup, b_bg, b_dk, b_hi) in enumerate(blk_defs):
        bx_blk = wx0 + b_idx * (blk_w + blk_gap)
        add_textbox(
            slide, bx_blk - 0.04, 1.81, blk_w + 0.08, 0.24,
            [[("W", "b"), ("1", "b_"), (t_sup, "b^"), (" \u2208 \u211d", "b"), ("64\u00d7128", "b^")]],
            font_name=FONT_SERIF, size_pt=9.8, color=b_dk, align=PP_ALIGN.CENTER, margin_lr=0.0, margin_tb=0.0,
        )
        for r_idx in range(5):
            ry = wy0 + r_idx * row_h
            is_j = (r_idx == 2)
            rf = b_hi if is_j else b_bg
            rc = PALETTE["white"] if is_j else b_dk
            if is_j:
                r_segs = [("W", "b"), ("j", "b_"), (t_sup, "b^"), (" \u2208 \u211d", "b"), ("128", "b^")]
            elif r_idx in (1, 3):
                r_segs = [("\u22ee", "sans")]
            elif r_idx == 0:
                r_segs = [("row 1", "sans")]
            else:
                r_segs = [("row 64", "sans")]
            add_box(
                slide, bx_blk, ry, blk_w, row_h - row_gap, [r_segs],
                fill=rf, border=BLACK if is_j else rgb("#9AA4AE"), border_pt=1.35 if is_j else 0.85,
                rounded=True, round_radius=0.22,
                color=rc, size_pt=10.2 if is_j else 9.8, margin_lr=0.0, margin_tb=0.0,
            )

    w_tot_w = 3 * blk_w + 2 * blk_gap
    yj_top = wy0 + 2 * row_h - 0.024
    yj_h   = (row_h - row_gap) + 0.048
    add_box(
        slide, wx0 - 0.032, yj_top, w_tot_w + 0.064, yj_h,
        fill=None, border=BLACK, border_pt=1.95, rounded=True, round_radius=0.22,
    )

    zx_v = wx0 + w_tot_w + 0.20
    zw_vec = 0.46
    add_line(slide, wx0 + w_tot_w + 0.04, y_mid_w, zx_v - 0.02, y_mid_w, color=BLACK, width_pt=1.45, arrow_end="triangle", arrow_size="sm")
    add_textbox(
        slide, zx_v - 0.14, 1.81, zw_vec + 0.28, 0.24,
        [[("z", "bi"), (" \u2208 \u211d", "b"), ("64", "b^")]],
        font_name=FONT_SERIF, size_pt=10.0, color=C_SK_DK, align=PP_ALIGN.CENTER, margin_lr=0.0, margin_tb=0.0,
    )
    for r_idx in range(5):
        ry = wy0 + r_idx * row_h
        is_j = (r_idx == 2)
        z_lbl = [("z", "bi", PALETTE["white"]), ("j", "b_", PALETTE["white"])] if is_j else ([("z", "i"), ("1", "_")] if r_idx == 0 else ([("z", "i"), ("64", "_")] if r_idx == 4 else [("\u22ee", "")]))
        add_box(
            slide, zx_v, ry, zw_vec, row_h - row_gap, [z_lbl],
            fill=C_SK_DK if is_j else C_SK_BG, border=BLACK if is_j else rgb("#9AA4AE"), border_pt=1.55 if is_j else 0.85,
            rounded=True, round_radius=0.22,
            color=PALETTE["white"] if is_j else C_SK_DK, size_pt=10.6 if is_j else 9.8, margin_lr=0.0, margin_tb=0.0,
        )

    # --- LOWER HALF: FORMAL IEEE MODULE ZOOM BOX [y=4.14..5.93, h=1.79 in] ---
    # Replaced awkward 'Hidden Unit j (illustrative negative-return trace)' with clean architectural title!
    zm_x, zm_y, zm_w, zm_h = bx + 0.10, 4.14, bw - 0.20, 1.79
    add_line(slide, wx0 - 0.03, yj_top + yj_h, zm_x + 0.44, zm_y, color=BLACK, width_pt=1.25, dash="dot")
    add_line(slide, wx0 + w_tot_w + 0.03, yj_top + yj_h, zm_x + zm_w - 0.44, zm_y, color=BLACK, width_pt=1.25, dash="dot")

    add_box(slide, zm_x, zm_y, zm_w, zm_h, fill=rgb("#FCFDFF"), border=BLACK, border_pt=1.5, dash="dot", rounded=True, round_radius=0.06)

    # Formal module name (centered, authoritative, zero awkward disclaimers!)
    add_textbox(
        slide, zm_x + 0.08, zm_y + 0.05, zm_w - 0.16, 0.24,
        [[("Cross-Type Feature Fusion & Activation  (Channel ", "bsans"), ("j", "bi"), (")", "bsans")]],
        font_name=FONT_SERIF, size_pt=10.8, color=BLACK, align=PP_ALIGN.CENTER, margin_lr=0.0, margin_tb=0.0,
    )

    add_textbox(
        slide, zm_x + 0.04, zm_y + 0.32, zm_w - 0.08, 0.30,
        [[
            ("z", "bi", C_SK_DK), ("j", "b_", C_SK_DK), (" = ", "b", BLACK),
            ("W", "b", C_ST_DK), ("j", "b_", C_ST_DK), ("M", "b^", C_ST_DK), ("\u00b7g", "b", C_ST_DK), ("M", "b^", C_ST_DK),
            (" + ", "b", BLACK),
            ("W", "b", C_OP_DK), ("j", "b_", C_OP_DK), ("O", "b^", C_OP_DK), ("\u00b7g", "b", C_OP_DK), ("O", "b^", C_OP_DK),
            (" + ", "b", BLACK),
            ("W", "b", C_WK_DK), ("j", "b_", C_WK_DK), ("W", "b^", C_WK_DK), ("\u00b7g", "b", C_WK_DK), ("W", "b^", C_WK_DK),
            (" + b", "b", BLACK), ("j", "b_", BLACK),
        ]],
        font_name=FONT_SERIF, size_pt=12.4, align=PP_ALIGN.CENTER, margin_lr=0.0, margin_tb=0.0,
    )

    y_num = zm_y + 0.69
    num_pills = [
        (zm_x + 0.09, 0.78, [("M: \u22120.42", "bsans")], C_ST_BG, C_ST_DK),
        (zm_x + 1.01, 0.78, [("O: +0.18", "bsans")], C_OP_BG, C_OP_DK),
        (zm_x + 1.93, 0.78, [("W: \u22120.31", "bsans")], C_WK_BG, C_WK_DK),
        (zm_x + 2.85, 0.68, [("b", "bsans"), ("j", "b_"), (": +.05", "bsans")], rgb("#EEF1F5"), BLACK),
    ]
    for idx_p, (px, pw_p, psegs, pbg, pdk) in enumerate(num_pills):
        add_box(slide, px, y_num, pw_p, 0.38, [psegs], fill=pbg, border=BLACK, border_pt=1.2, rounded=True, round_radius=0.18, color=pdk, size_pt=10.0, margin_lr=0.0, margin_tb=0.0)
        if idx_p < 3:
            add_textbox(slide, px + pw_p, y_num, 0.14, 0.38, [[("+", "b")]], font_name=FONT_SANS, size_pt=10.8, color=BLACK, align=PP_ALIGN.CENTER, margin_lr=0.0, margin_tb=0.0)
    add_textbox(slide, zm_x + 3.53, y_num, 0.16, 0.38, [[("=", "b")]], font_name=FONT_SANS, size_pt=11.0, color=BLACK, align=PP_ALIGN.CENTER, margin_lr=0.0, margin_tb=0.0)
    add_box(slide, zm_x + 3.69, y_num, 0.66, 0.38, [[("\u22120.50", "bsans")]], fill=C_SK_BG, border=C_SK_DK, border_pt=1.45, rounded=True, round_radius=0.18, color=C_SK_DK, size_pt=10.4, margin_lr=0.0, margin_tb=0.0)

    y_act = zm_y + 1.19
    h_act = 0.50
    y_act_mid = y_act + h_act / 2.0
    add_box(slide, zm_x + 0.09, y_act, 0.92, h_act, [[("z", "bi"), ("j", "b_"), (" = \u22120.50", "b")]], fill=C_SK_BG, border=BLACK, border_pt=1.25, rounded=True, round_radius=0.16, color=C_SK_DK, size_pt=10.6, margin_lr=0.0, margin_tb=0.0)
    add_line(slide, zm_x + 1.01, y_act_mid, zm_x + 1.17, y_act_mid, color=BLACK, width_pt=1.35, arrow_end="triangle", arrow_size="sm")
    add_box(slide, zm_x + 1.18, y_act, 0.98, h_act, [[("LN: \u22120.62", "bsans")]], fill=C_BLUE, border=BLACK, border_pt=1.3, rounded=True, round_radius=0.16, color=BLACK, size_pt=10.4, margin_lr=0.0, margin_tb=0.0)
    add_line(slide, zm_x + 2.16, y_act_mid, zm_x + 2.32, y_act_mid, color=BLACK, width_pt=1.35, arrow_end="triangle", arrow_size="sm")
    add_activation_glyph(slide, zm_x + 2.52, y_act_mid, r=0.19, kind="lrelu", fill=C_MINT, border=BLACK, border_pt=1.4, curve_pt=1.45)
    add_line(slide, zm_x + 2.71, y_act_mid, zm_x + 2.87, y_act_mid, color=BLACK, width_pt=1.35, arrow_end="triangle", arrow_size="sm")
    add_box(
        slide, zm_x + 2.88, y_act, 1.47, h_act,
        [[("h", "bi"), ("j", "b_"), (" = \u22120.062", "b")], [("slope 0.1  (\u2207 \u2260 0)", "bsans", C_ST_DK, 9.8)]],
        fill=C_MINT, border=BLACK, border_pt=1.45, rounded=True, round_radius=0.16, color=C_ST_DK, size_pt=10.8, line_spacing=1.02, margin_lr=0.01, margin_tb=0.01,
    )

    # =========================================================================
    # PANEL (c): SCALAR VALUE HEAD V_phi(s_t) + EXACT TD ERROR & MSE BRANCHES
    # =========================================================================
    ccx = cx + cw / 2.0
    add_box(slide, cx, y_top, cw, h_pan, fill=rgb("#FCFDFF"), border=BLACK, border_pt=1.7, rounded=True, round_radius=0.04)
    add_textbox(
        slide, cx + 0.06, y_top + 0.06, cw - 0.12, 0.28,
        [[("(c) Scalar Value Head  ", "bsans"), ("V", "bi"), ("\u03c6", "b_"), ("(s", "b"), ("t", "b_"), (")", "b")]],
        font_name=FONT_SERIF, size_pt=12.0, color=BLACK, align=PP_ALIGN.CENTER, margin_lr=0.0, margin_tb=0.0,
    )

    zc_w, zc_gap = 0.50, 0.025
    zc_tot = 5 * zc_w + 4 * zc_gap
    zc_x0 = ccx - zc_tot / 2.0
    zc_y0, zc_h = 1.82, 0.36

    x_bus_bc = bx + bw + gutter_bc / 2.0
    add_polyline(slide, [(zx_v + zw_vec, y_mid_w), (x_bus_bc, y_mid_w), (x_bus_bc, zc_y0 + zc_h / 2.0), (zc_x0 - 0.02, zc_y0 + zc_h / 2.0)], color=C_SK_DK, width_pt=1.7, arrow_end="triangle", arrow_size="sm")

    zc_specs = [
        ([("z", "i"), ("1", "_")],                                       C_SK_BG, C_SK_DK, 1.3),
        ([("\u22ef", "")],                                                C_SK_BG, C_SK_DK, 1.3),
        ([("z", "bi", PALETTE["white"]), ("j", "b_", PALETTE["white"])], C_SK_DK, PALETTE["white"], 1.55),
        ([("\u22ef", "")],                                                C_SK_BG, C_SK_DK, 1.3),
        ([("z", "i"), ("64", "_")],                                      C_SK_BG, C_SK_DK, 1.3),
    ]
    for z_i, (z_segs, z_bg, z_tc, z_bpt) in enumerate(zc_specs):
        add_box(
            slide, zc_x0 + z_i * (zc_w + zc_gap), zc_y0, zc_w, zc_h, [z_segs],
            fill=z_bg, border=BLACK, border_pt=z_bpt, rounded=True, round_radius=0.20,
            color=z_tc, size_pt=11.4, margin_lr=0.0, margin_tb=0.0,
        )

    add_line(slide, ccx, 2.18, ccx, 2.34, color=BLACK, width_pt=1.45, arrow_end="triangle", arrow_size="sm")
    add_box(slide, ccx - 1.36, 2.36, 2.72, 0.35, [[("LayerNorm  (64)", "bsans")]], fill=C_BLUE, border=BLACK, border_pt=1.4, rounded=True, round_radius=0.18, color=BLACK, size_pt=11.4, margin_lr=0.02, margin_tb=0.0)

    add_line(slide, ccx, 2.71, ccx, 2.87, color=BLACK, width_pt=1.45, arrow_end="triangle", arrow_size="sm")
    add_box(slide, ccx - 1.36, 2.89, 2.72, 0.35, [[("     LeakyReLU (0.1) \u2192 h \u2208 \u211d", "bsans"), ("64", "b^")]], fill=C_MINT, border=BLACK, border_pt=1.4, rounded=True, round_radius=0.18, color=BLACK, size_pt=10.8, margin_lr=0.02, margin_tb=0.0)
    add_activation_glyph(slide, ccx - 1.15, 3.065, r=0.13, kind="lrelu", fill=PALETTE["white"], border=BLACK, border_pt=1.3, curve_pt=1.25)

    add_line(slide, ccx, 3.24, ccx, 3.40, color=BLACK, width_pt=1.45, arrow_end="triangle", arrow_size="sm")
    add_box(slide, ccx - 1.36, 3.42, 2.72, 0.35, [[("Linear (64\u21921):  ", "bsans"), ("w", "b"), ("2", "b_"), ("T", "b^"), (" h + b", "b"), ("2", "b_")]], fill=C_PEACH, border=BLACK, border_pt=1.4, rounded=True, round_radius=0.18, color=BLACK, size_pt=11.0, margin_lr=0.02, margin_tb=0.0)

    add_line(slide, ccx, 3.77, ccx, 3.93, color=C_SK_DK, width_pt=1.6, arrow_end="triangle", arrow_size="sm")
    add_box(
        slide, ccx - 1.38, 3.95, 2.76, 0.43,
        [[("V", "b", C_SK_DK), ("\u03c6", "b_", C_SK_DK), ("(s", "b", C_SK_DK), ("t", "b_", C_SK_DK), (")  =  \u22120.48  \u2208 \u211d", "b", C_SK_DK)]],
        fill=C_SK_BG, border=C_SK_DK, border_pt=1.9, rounded=True, round_radius=0.22, size_pt=13.0, margin_lr=0.02, margin_tb=0.0,
    )

    y_fork = 4.54
    bw_p = 1.50
    x_boxL = cx + 0.09
    x_boxR = cx + cw - 0.09 - bw_p
    x_gL = x_boxL + bw_p / 2.0
    x_gR = x_boxR + bw_p / 2.0
    add_line(slide, ccx, 4.38, ccx, y_fork, color=C_SK_DK, width_pt=1.45)
    add_polyline(slide, [(ccx, y_fork), (x_gL, y_fork), (x_gL, 4.69)], color=C_OP_DK, width_pt=1.45, arrow_end="triangle", arrow_size="sm")
    add_polyline(slide, [(ccx, y_fork), (x_gR, y_fork), (x_gR, 4.69)], color=C_SK_DK, width_pt=1.45, arrow_end="triangle", arrow_size="sm")

    # Left branch: Exact paper term 'TD Error delta_t' (main_tsmc.tex line 361-363!)
    add_box(
        slide, x_boxL, 4.71, bw_p, 0.62,
        [
            [("TD Error  \u03b4", "bsans", C_OP_DK, 10.8), ("t", "b_", C_OP_DK, 10.8)],
            [("r", "bserif", BLACK, 10.6), ("t", "b_", BLACK, 10.6), (" + \u03b3b", "bserif", BLACK, 10.6), ("t", "b_", BLACK, 10.6), (" V", "bserif", BLACK, 10.6), ("t+1", "b_", BLACK, 10.6), (" \u2212 V", "bserif", BLACK, 10.6), ("t", "b_", BLACK, 10.6)],
        ],
        fill=C_OP_BG, border=BLACK, border_pt=1.35, rounded=True, round_radius=0.16, line_spacing=1.06, margin_lr=0.01, margin_tb=0.01,
    )
    add_line(slide, x_gL, 5.33, x_gL, 5.53, color=C_OP_DK, width_pt=1.4, arrow_end="triangle", arrow_size="sm")
    add_box(
        slide, x_boxL, 5.55, bw_p, 0.38,
        [[("Policy Update (\u2207\u03b8)", "bsans")]],
        fill=rgb("#D9E9F7"), border=BLACK, border_pt=1.25, rounded=True, round_radius=0.20,
        color=C_OP_DK, size_pt=9.8, margin_lr=0.0, margin_tb=0.0,
    )

    # Right branch: Value Loss MSE L_V(phi) -> Critic grad_phi (clip 0.05)
    add_box(
        slide, x_boxR, 4.71, bw_p, 0.62,
        [
            [("MSE  L", "bsans", C_SK_DK, 10.8), ("V", "b_", C_SK_DK, 10.8), ("(\u03c6)", "bsans", C_SK_DK, 10.8)],
            [("c", "bserif", BLACK, 10.6), ("V", "b_", BLACK, 10.6), (" (V", "bserif", BLACK, 10.6), ("\u03c6", "b_", BLACK, 10.6), (" \u2212 R", "bserif", BLACK, 10.6), ("t", "b_", BLACK, 10.6), (")", "bserif", BLACK, 10.6), ("2", "b^", BLACK, 10.6)],
        ],
        fill=C_SK_BG, border=BLACK, border_pt=1.35, rounded=True, round_radius=0.16, line_spacing=1.06, margin_lr=0.01, margin_tb=0.01,
    )
    add_line(slide, x_gR, 5.33, x_gR, 5.53, color=C_SK_DK, width_pt=1.4, arrow_end="triangle", arrow_size="sm")
    add_box(
        slide, x_boxR, 5.55, bw_p, 0.38,
        [[("Critic \u2207\u03c6 (clip 0.05)", "bsans")]],
        fill=C_SK_DK, border=BLACK, border_pt=1.25, rounded=True, round_radius=0.20,
        color=PALETTE["white"], size_pt=9.8, margin_lr=0.0, margin_tb=0.0,
    )

    set_notes(
        slide,
        "Module B10. Critic value head (main_tsmc.tex Sections IV-B & IV-D; models/hb_gat_pn.py: critic_station_attn, critic_task_worker_attn, and self.critic; grad clip 0.05 in Appendix B). Refined v8 replaces the awkward 'Hidden Unit j (illustrative negative-return trace)' label with the formal module title 'Cross-Type Feature Fusion & Activation (Channel j)' (note in LaTeX caption that sample scalars -0.42, +0.18, -0.31 illustrate negative-regime LeakyReLU gradient retention), replaces 'Skill hubs' with 'Skill nodes K (5) -> encoded in H^O, H^W', replaces arbitrary '.12-.31' with '5 physical stations', and fixes 'GAE delta_t' -> 'TD Error delta_t' matching main_tsmc.tex Eq. (17).\n"
        "Layout reference: Igor G. Smit et al., AAAI 2025 Fig. 1; ReSched, ICLR 2026 Fig. 2.",
    )


def main():
    parser = argparse.ArgumentParser(description="Build Slide 15 (B10) for refined_v8.")
    parser.add_argument("--pptx", default=DEFAULT_PPTX_PATH, help="Target PPTX path.")
    args = parser.parse_args()
    prs = Presentation(args.pptx)
    build_b10_v8(prs.slides[14])  # Slide 15: B10
    prs.save(args.pptx)
    print("Updated Slide 15 (B10) in", args.pptx)


if __name__ == "__main__":
    main()
