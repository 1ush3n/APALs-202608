"""Build Slide 11 (B6) in refined_v8: 100% paper-faithful Operation Pointer (main_tsmc.tex Section IV-C).

v8 terminology & notation polish:
1. Feasible set symbol aligned verbatim with Section IV-C line 328: 'F_t^O = {O_3, O_4}' (replaced 'A_O(s_t)').
2. Masked logit value upgraded from programming notation '-1e4' to math notation '-10^4' (exact Appendix A line 690).
3. Replaced ReSched-borrowed 'O2O Mask (5x5)' with APAL term 'Precedence (5x5)'.
4. Upgraded bottom conditions to exact paper symbols:
   - 'm_i = 1 iff P(i) done ^ t >= r_i ^ (M, W) feasible'
   - 'Mask Effect: O_2 scores highest (u_2=1.5 > u_3=1.2) yet waits on O_3'
   - 'h(O_3), k_3, q_3 -> Station Ptr'
"""
from __future__ import annotations

import argparse
import os
import sys
from pptx import Presentation
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REFINED_V8_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from pptx_helpers import (
    BLACK,
    FONT_SANS,
    FONT_SERIF,
    PALETTE,
    add_activation_glyph,
    add_angled_label,
    add_box,
    add_circle,
    add_line,
    add_polyline,
    add_textbox,
    clear_slide,
    rgb,
    set_notes,
)

DEFAULT_PPTX = os.path.join(REFINED_V8_DIR, "APAL_Modules_English_Refined_v8.pptx")

RS_LEARN    = rgb("#55C7B8")
RS_NONLRN   = rgb("#F5E7BA")
RS_PINK_BG  = rgb("#FBE4E2")
RS_DARK     = rgb("#46444B")
RS_MAT_ON   = rgb("#3B100E")
RS_MAT_PEND = rgb("#885B58")
RS_GREEN    = rgb("#1F7A4C")
RS_BLUE_HI  = rgb("#1D5B8F")
RS_RED      = rgb("#B8292B")
RS_TEAL_DK  = rgb("#16665C")
RS_MUTED    = rgb("#667380")


def _draw_rich_stage_bracket(
    slide, x1: float, x2: float, y: float, rich_lines, *, color=BLACK, tick_h: float = 0.08, label_y_offset: float = 0.10, size_pt: float = 13.8,
):
    xm = (x1 + x2) / 2.0
    notch = 0.075
    pts = [(x1, y - tick_h), (x1, y), (xm - notch, y), (xm, y + notch), (xm + notch, y), (x2, y), (x2, y - tick_h)]
    add_polyline(slide, pts, color=color, width_pt=1.35)
    add_textbox(
        slide, x1, y + label_y_offset, x2 - x1, 0.34, rich_lines,
        font_name=FONT_SERIF, size_pt=size_pt, color=color, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.TOP, word_wrap=False,
    )


def _draw_contiguous_strip(
    slide, x: float, y: float, colors: list, *, cell_w: float, cell_h: float, inner_border, inner_pt: float = 0.65, outer_border, outer_pt: float = 1.10,
):
    for idx, col in enumerate(colors):
        add_box(slide, x + idx * cell_w, y, cell_w, cell_h, fill=col, border=inner_border, border_pt=inner_pt)
    add_box(slide, x, y, cell_w * len(colors), cell_h, fill=None, border=outer_border, border_pt=outer_pt)


def _draw_mini_dag(slide, x0: float, y0: float, w: float, h: float, mode: str = "state"):
    cy_mid = y0 + h * 0.56
    cy_top = y0 + h * 0.32
    cy_bot = y0 + h * 0.80
    xs = [x0 + w * 0.16, x0 + w * 0.50, x0 + w * 0.84]
    pts = {"O1": (xs[0], cy_mid), "O3": (xs[1], cy_top), "O4": (xs[1], cy_bot), "O2": (xs[2], cy_top), "O5": (xs[2], cy_bot)}
    r = 0.145 if w >= 1.9 else 0.135
    sz = 9.8 if w >= 1.9 else 9.4

    for u, v in [("O1", "O3"), ("O1", "O4"), ("O3", "O2"), ("O4", "O5")]:
        ux, uy = pts[u]
        vx, vy = pts[v]
        dx, dy = vx - ux, vy - uy
        dist = (dx * dx + dy * dy) ** 0.5
        sx, sy = ux + dx / dist * (r + 0.01), uy + dy / dist * (r + 0.01)
        ex, ey = vx - dx / dist * (r + 0.02), vy - dy / dist * (r + 0.02)
        if mode == "state":
            if (u, v) == ("O3", "O2"):
                ecol, ew, edash = RS_RED, 1.50, "dash"
            elif (u, v) == ("O4", "O5"):
                ecol, ew, edash = RS_MAT_PEND, 1.15, "dash"
            else:
                ecol, ew, edash = RS_MAT_ON, 1.45, None
        elif mode == "cand":
            ecol = RS_GREEN if u == "O1" else rgb("#CFA9A7")
            ew = 1.75 if u == "O1" else 0.95
            edash = None if u == "O1" else "dash"
        else:
            ecol = RS_BLUE_HI if (u, v) == ("O1", "O3") else rgb("#D6B5B3")
            ew = 1.95 if (u, v) == ("O1", "O3") else 0.90
            edash = None if (u, v) == ("O1", "O3") else "dash"
        add_line(slide, sx, sy, ex, ey, color=ecol, width_pt=ew, dash=edash, arrow_end="triangle", arrow_size="sm")

    for nid, (nx, ny) in pts.items():
        num = nid[1]
        if mode == "state":
            if nid == "O1":
                nfill, nbdr, npt, ncol = rgb("#DCE2E8"), rgb("#707C88"), 1.15, rgb("#4F5B67")
            elif nid in ("O3", "O4"):
                nfill, nbdr, npt, ncol = PALETTE["white"], BLACK, 1.85, BLACK
            elif nid == "O2":
                nfill, nbdr, npt, ncol = rgb("#FCE4E2"), RS_RED, 1.70, RS_RED
            else:
                nfill, nbdr, npt, ncol = rgb("#EFE3E2"), RS_MAT_PEND, 1.10, rgb("#735452")
        elif mode == "cand":
            if nid in ("O3", "O4"):
                nfill, nbdr, npt, ncol = PALETTE["white"], BLACK, 1.95, BLACK
            else:
                nfill, nbdr, npt, ncol = rgb("#F4D4D2"), rgb("#CCA8A6"), 1.0, rgb("#AA8886")
        else:
            if nid == "O3":
                nfill, nbdr, npt, ncol = RS_BLUE_HI, BLACK, 2.1, PALETTE["white"]
            else:
                nfill, nbdr, npt, ncol = rgb("#F4D4D2"), rgb("#D2B2B0"), 0.95, rgb("#B29492")
        add_circle(slide, nx, ny, r, [[("O", "b"), (num, "b_")]], fill=nfill, border=nbdr, border_pt=npt, color=ncol, size_pt=sz)

    if mode == "state":
        add_textbox(slide, xs[0] - 0.28, cy_mid - r - 0.18, 0.56, 0.17, [[("done", "bsans")]], font_name=FONT_SANS, size_pt=9.4, color=rgb("#53606D"), align=PP_ALIGN.CENTER, margin_lr=0.0, margin_tb=0.0, word_wrap=False)
        add_textbox(slide, xs[1] - 0.30, cy_top - r - 0.18, 0.60, 0.17, [[("ready", "bsans")]], font_name=FONT_SANS, size_pt=9.4, color=RS_GREEN, align=PP_ALIGN.CENTER, margin_lr=0.0, margin_tb=0.0, word_wrap=False)
        add_textbox(slide, xs[2] - 0.34, cy_top - r - 0.18, 0.68, 0.17, [[("wait O\u2083", "bsans")]], font_name=FONT_SANS, size_pt=9.4, color=RS_RED, align=PP_ALIGN.CENTER, margin_lr=0.0, margin_tb=0.0, word_wrap=False)


def _draw_precedence_matrix(slide, x0: float, y0: float, cell: float = 0.146):
    lbl_w, lbl_h = 0.19, 0.15
    gx, gy = x0 + lbl_w, y0 + lbl_h
    grid_sz = 5 * cell
    add_textbox(slide, x0, y0 - 0.01, lbl_w - 0.01, lbl_h + 0.01, [[("\u2192", "bsans")]], font_name=FONT_SANS, size_pt=9.2, color=rgb("#6A5D5D"), align=PP_ALIGN.CENTER, margin_lr=0.0, margin_tb=0.0, word_wrap=False)
    for i in range(5):
        idx_s = str(i + 1)
        add_textbox(slide, gx + i * cell, y0 - 0.01, cell, lbl_h + 0.01, [[(idx_s, "bsans")]], font_name=FONT_SANS, size_pt=9.2, color=RS_RED if i == 1 else BLACK, align=PP_ALIGN.CENTER, margin_lr=0.0, margin_tb=0.0, word_wrap=False)
        add_textbox(slide, x0, gy + i * cell, lbl_w - 0.02, cell, [[(idx_s, "bsans")]], font_name=FONT_SANS, size_pt=9.2, color=RS_BLUE_HI if i == 2 else BLACK, align=PP_ALIGN.RIGHT, margin_lr=0.0, margin_tb=0.0, word_wrap=False)

    edge_fills = {(0, 2): RS_MAT_ON, (0, 3): RS_MAT_ON, (2, 1): RS_RED, (3, 4): RS_MAT_PEND}
    for r_i in range(5):
        for c_i in range(5):
            add_box(slide, gx + c_i * cell, gy + r_i * cell, cell, cell, fill=edge_fills.get((r_i, c_i), PALETTE["white"]), border=BLACK, border_pt=0.65)
    add_box(slide, gx, gy, grid_sz, grid_sz, fill=None, border=BLACK, border_pt=1.35)


def build_b6_v8(slide):
    """Slide 11 · B6 refined_v8: 100% paper-faithful Operation Pointer."""
    clear_slide(slide)

    ax, aw = 0.48, 3.74
    bx, bw = 4.38, 4.44
    cx, cw = 8.98, 3.68

    q_y0, q_h   = 0.40, 1.20
    y_stage_top = 1.94
    y_stage_bot = 6.26
    h_stage = y_stage_bot - y_stage_top

    RY = [2.60, 3.20, 3.80, 4.40, 5.00]

    # TOP-LEFT LEGEND
    add_box(slide, ax, 0.36, 0.17, 0.17, fill=RS_LEARN, border=BLACK, border_pt=1.25)
    add_textbox(slide, ax + 0.21, 0.31, 1.94, 0.25, [[("Learnable (W", "bsans"), ("g", "b_"), (", W", "bsans"), ("O", "b_"), (", a", "bsans"), ("O", "b_"), (")", "bsans")]], font_name=FONT_SANS, size_pt=9.8, color=BLACK, align=PP_ALIGN.LEFT, margin_lr=0.0, word_wrap=False)
    add_box(slide, ax + 2.20, 0.36, 0.17, 0.17, fill=RS_NONLRN, border=BLACK, border_pt=1.25)
    add_textbox(slide, ax + 2.41, 0.31, 1.36, 0.25, [[("Non-Learnable", "bsans")]], font_name=FONT_SANS, size_pt=9.8, color=BLACK, align=PP_ALIGN.LEFT, margin_lr=0.0, word_wrap=False)

    add_circle(slide, ax + 0.085, 0.69, 0.085, fill=PALETTE["white"], border=BLACK, border_pt=1.7)
    add_textbox(slide, ax + 0.21, 0.57, 1.14, 0.24, [[("Ready (m=1)", "bsans")]], font_name=FONT_SANS, size_pt=9.6, color=BLACK, align=PP_ALIGN.LEFT, margin_lr=0.0, word_wrap=False)
    add_circle(slide, ax + 1.44, 0.69, 0.085, fill=rgb("#FCE4E2"), border=RS_RED, border_pt=1.55)
    add_textbox(slide, ax + 1.56, 0.57, 1.24, 0.24, [[("Masked (m=0)", "bsans")]], font_name=FONT_SANS, size_pt=9.6, color=RS_RED, align=PP_ALIGN.LEFT, margin_lr=0.0, word_wrap=False)
    add_line(slide, ax + 2.86, 0.69, ax + 3.22, 0.69, color=RS_TEAL_DK, width_pt=1.5, dash="dash", arrow_end="triangle", arrow_size="sm")
    add_textbox(slide, ax + 3.26, 0.56, 0.60, 0.25, [[("q", "bi"), ("t", "b_"), (" bus", "bsans", RS_TEAL_DK, 9.4)]], font_name=FONT_SERIF, size_pt=10.0, color=RS_TEAL_DK, align=PP_ALIGN.LEFT, margin_lr=0.0, word_wrap=False)

    # TOP-CENTER: SHARED CONTEXT QUERY BRANCH
    add_box(slide, bx, q_y0, bw, q_h, fill=PALETTE["white"], border=BLACK, border_pt=1.85, rounded=True, round_radius=0.09)
    add_textbox(slide, bx + 0.12, q_y0 + 0.05, 2.30, 0.26, [[("Shared Context Query Branch", "bi")]], font_name=FONT_SERIF, size_pt=11.6, color=BLACK, align=PP_ALIGN.LEFT, margin_lr=0.0, word_wrap=False)
    add_textbox(
        slide, bx + 2.28, q_y0 + 0.05, 2.04, 0.26,
        [[("u", "bi"), ("i", "b_"), (" = a", "b"), ("O", "b_"), ("T", "b^"), (" tanh(q", "b"), ("t", "b_"), (" + W", "b"), ("O", "b_"), ("h", "bi"), ("i", "b_"), (")", "b")]],
        font_name=FONT_SERIF, size_pt=11.2, color=BLACK, align=PP_ALIGN.RIGHT, margin_lr=0.0, word_wrap=False,
    )

    y_qrow = q_y0 + 0.37
    add_box(slide, bx + 0.11, y_qrow, 1.36, 0.45, fill=RS_DARK, border=BLACK, border_pt=1.3, rounded=True, round_radius=0.22)
    for k, (gsym, gcol) in enumerate([("M", rgb("#4EA394")), ("O", rgb("#D2655B")), ("W", rgb("#D48F4C"))]):
        add_box(slide, bx + 0.18 + k * 0.415, y_qrow + 0.06, 0.365, 0.33, [[("g", "b"), (gsym, "b^")]], fill=gcol, border=PALETTE["white"], border_pt=0.8, color=PALETTE["white"], size_pt=11.0, margin_lr=0.0, margin_tb=0.0)
    add_textbox(slide, bx + 0.08, y_qrow + 0.47, 1.42, 0.28, [[("g", "bi"), ("t", "b_"), (" \u2208 \u211d", "b"), ("3d", "b^")]], font_name=FONT_SERIF, size_pt=11.8, color=BLACK, align=PP_ALIGN.CENTER, margin_lr=0.0, word_wrap=False)

    add_line(slide, bx + 1.47, y_qrow + 0.225, bx + 1.70, y_qrow + 0.225, color=BLACK, width_pt=1.5, arrow_end="triangle", arrow_size="sm")
    add_box(slide, bx + 1.72, y_qrow, 0.68, 0.45, [[("W", "b"), ("g", "b_")]], fill=RS_LEARN, border=BLACK, border_pt=1.55, rounded=True, round_radius=0.18, color=BLACK, size_pt=13.8)
    add_textbox(slide, bx + 1.56, y_qrow + 0.48, 1.00, 0.28, [[("proj.", "bsans")]], font_name=FONT_SANS, size_pt=11.2, color=rgb("#364554"), align=PP_ALIGN.CENTER, margin_lr=0.0, word_wrap=False)

    add_line(slide, bx + 2.40, y_qrow + 0.225, bx + 2.63, y_qrow + 0.225, color=BLACK, width_pt=1.5, arrow_end="triangle", arrow_size="sm")
    sw_qx = bx + 2.65
    q_cell_w = 0.305
    q_strip_w = 5 * q_cell_w
    _draw_contiguous_strip(
        slide, sw_qx, y_qrow + 0.045,
        [rgb("#2A8C7F"), rgb("#55C7B8"), rgb("#1C635A"), rgb("#7FD6C9"), rgb("#3CA89B")],
        cell_w=q_cell_w, cell_h=0.36, inner_border=BLACK, inner_pt=0.75, outer_border=BLACK, outer_pt=1.25,
    )
    add_textbox(
        slide, sw_qx - 0.14, y_qrow + 0.48, q_strip_w + 0.28, 0.28,
        [[("q", "bi"), ("t", "b_"), (" = W", "b"), ("g", "b_"), ("g", "bi"), ("t", "b_"), (" \u2208 \u211d", "b"), ("128", "b^")]],
        font_name=FONT_SERIF, size_pt=11.6, color=RS_TEAL_DK, align=PP_ALIGN.CENTER, margin_lr=0.0, word_wrap=False,
    )
    q_exit_x = bx + 4.32
    q_exit_start_x = sw_qx + q_strip_w
    q_exit_y = y_qrow + 0.225

    # TOP-RIGHT: DECISION DAG PANELS
    pw_s = (cw - 0.14) / 2.0
    dag_h_top = q_h
    add_box(slide, cx, q_y0, pw_s, dag_h_top, fill=RS_PINK_BG, border=rgb("#D9A6A3"), border_pt=1.3, rounded=True, round_radius=0.05)
    add_textbox(slide, cx + 0.04, q_y0 + 0.04, pw_s - 0.08, 0.24, [[("Feasible: {O", "bsans"), ("3", "b_"), (", O", "bsans"), ("4", "b_"), ("}", "bsans")]], font_name=FONT_SANS, size_pt=10.0, color=BLACK, align=PP_ALIGN.CENTER, margin_lr=0.0, word_wrap=False)
    _draw_mini_dag(slide, cx + 0.04, q_y0 + 0.22, pw_s - 0.08, dag_h_top - 0.25, mode="cand")

    add_line(slide, cx + pw_s + 0.01, q_y0 + dag_h_top / 2, cx + pw_s + 0.13, q_y0 + dag_h_top / 2, color=BLACK, width_pt=1.55, arrow_end="triangle", arrow_size="sm")

    px2 = cx + pw_s + 0.14
    add_box(slide, px2, q_y0, pw_s, dag_h_top, fill=RS_PINK_BG, border=BLACK, border_pt=1.7, rounded=True, round_radius=0.05)
    add_textbox(slide, px2 + 0.04, q_y0 + 0.04, pw_s - 0.08, 0.24, [[("Select i", "bsans"), ("t", "b_"), (" = O", "bsans"), ("3", "b_")]], font_name=FONT_SANS, size_pt=10.2, color=RS_BLUE_HI, align=PP_ALIGN.CENTER, margin_lr=0.0, word_wrap=False)
    _draw_mini_dag(slide, px2 + 0.04, q_y0 + 0.22, pw_s - 0.08, dag_h_top - 0.25, mode="sel")

    # STAGE (a): STATE s_t & PRECEDENCE MATRIX (APAL terminology — NO borrowed 'O2O Mask'!)
    add_box(slide, ax, 0.88, aw, y_stage_bot - 0.88, fill=RS_PINK_BG, border=BLACK, border_pt=1.85, rounded=True, round_radius=0.04)

    add_textbox(slide, ax + 0.08, 0.91, 2.26, 0.21, [[("Precedence DAG", "bsans")]], font_name=FONT_SANS, size_pt=10.2, color=BLACK, align=PP_ALIGN.CENTER, margin_lr=0.0, word_wrap=False)
    add_textbox(slide, ax + 2.42, 0.91, 1.24, 0.21, [[("Precedence (5\u00d75)", "bsans")]], font_name=FONT_SANS, size_pt=10.0, color=BLACK, align=PP_ALIGN.CENTER, margin_lr=0.0, word_wrap=False)

    _draw_mini_dag(slide, ax + 0.08, 1.12, 2.26, 0.80, mode="state")
    add_line(slide, ax + 2.39, 0.96, ax + 2.39, 1.88, color=rgb("#D9B2B0"), width_pt=0.9, dash="dot")
    _draw_precedence_matrix(slide, ax + 2.52, 1.07, cell=0.144)

    add_line(slide, ax + 0.10, y_stage_top, ax + aw - 0.10, y_stage_top, color=BLACK, width_pt=1.35, dash="dash")

    add_box(slide, bx, y_stage_top, bw, h_stage, fill=PALETTE["white"], border=BLACK, border_pt=1.85, rounded=True, round_radius=0.05)
    add_box(slide, cx, y_stage_top, cw, h_stage, fill=PALETTE["white"], border=BLACK, border_pt=1.85, rounded=True, round_radius=0.05)

    # UNIFIED COLUMN HEADERS
    y_hdr = y_stage_top + 0.05

    a_hdrs = [
        (ax + 0.12, 0.36, [[("Op", "bsans")]], FONT_SANS, 10.2),
        (ax + 0.52, 0.42, [[("p\u0303", "bi")]], FONT_SERIF, 11.0),
        (ax + 0.94, 0.38, [[("k", "bi")]], FONT_SERIF, 10.8),
        (ax + 1.32, 0.30, [[("q", "bi")]], FONT_SERIF, 10.8),
        (ax + 1.69, 0.88, [[("h", "bi"), ("i", "b_"), ("(5)", "b^")]], FONT_SERIF, 10.8),
        (ax + 2.66, 0.94, [[("Status", "bsans")]], FONT_SANS, 10.2),
    ]
    for hx, hw, hlines, hfn, hsz in a_hdrs:
        add_textbox(slide, hx, y_hdr, hw, 0.25, hlines, font_name=hfn, size_pt=hsz, color=BLACK, align=PP_ALIGN.CENTER, margin_lr=0.0, margin_tb=0.0, word_wrap=False)

    bus_x = bx + 1.66
    cx_plus = bx + 2.04
    x_tanh, w_tanh = bx + 2.34, 0.44
    x_ao,   w_ao   = bx + 2.93, 0.46

    b_hdrs = [
        (bx + 0.06, 0.72, [[("Key proj.", "bsans")]], FONT_SANS, 10.0),
        (bx + 0.81, 0.72, [[("Key ", "bsans"), ("k", "bi"), ("i", "b_")]], FONT_SERIF, 10.2),
        (cx_plus - 0.24, 0.48, [[("q", "bi"), ("t", "b_"), ("+k", "bi"), ("i", "b_")]], FONT_SERIF, 10.2),
        (x_tanh - 0.06, w_tanh + 0.12, [[("Nonlin.", "bsans")]], FONT_SANS, 10.0),
        (x_ao - 0.06, w_ao + 0.12, [[("Scorer", "bsans")]], FONT_SANS, 10.0),
        (bx + 3.50, 0.82, [[("Logit ", "bsans"), ("u", "bi"), ("i", "b_")]], FONT_SERIF, 10.2),
    ]
    for hx, hw, hlines, hfn, hsz in b_hdrs:
        add_textbox(slide, hx, y_hdr, hw, 0.25, hlines, font_name=hfn, size_pt=hsz, color=BLACK, align=PP_ALIGN.CENTER, margin_lr=0.0, margin_tb=0.0, word_wrap=False)

    x_sm, w_sm = cx + 1.32, 0.41
    rail_x, rail_w = cx + 1.88, 0.98
    c_hdrs = [
        (cx + 0.07, 0.42, [[("Mask", "bsans")]], FONT_SANS, 9.8),
        (cx + 0.54, 0.68, [[("Masked ", "bsans"), ("u\u0303", "bi"), ("i", "b_")]], FONT_SERIF, 9.6),
        (x_sm - 0.06, w_sm + 0.12, [[("Norm.", "bsans")]], FONT_SANS, 9.8),
        (cx + 1.84, 1.76, [[("Policy ", "bsans"), ("\u03c0", "b"), ("O", "b_"), ("(", "b"), ("i", "bi"), ("|", "b"), ("s", "bi"), ("t", "b_"), (")", "b")]], FONT_SERIF, 10.2),
    ]
    for hx, hw, hlines, hfn, hsz in c_hdrs:
        add_textbox(slide, hx, y_hdr, hw, 0.25, hlines, font_name=hfn, size_pt=hsz, color=BLACK, align=PP_ALIGN.CENTER, margin_lr=0.0, margin_tb=0.0, word_wrap=False)

    add_line(slide, bx + 0.10, 2.27, bx + bw - 0.10, 2.27, color=rgb("#DCE2E8"), width_pt=0.85)
    add_line(slide, cx + 0.10, 2.27, cx + cw - 0.10, 2.27, color=rgb("#DCE2E8"), width_pt=0.85)

    for x_band, w_band in [(ax + 0.07, aw - 0.14), (bx + 0.07, bw - 0.14), (cx + 0.07, cw - 0.14)]:
        add_box(slide, x_band, RY[1] - 0.245, w_band, 0.49, fill=rgb("#FDF0EE"), border=RS_RED, border_pt=1.15, dash="dash", rounded=True, round_radius=0.14)
        add_box(slide, x_band, RY[2] - 0.255, w_band, 0.51, fill=rgb("#E4F2FC"), border=RS_BLUE_HI, border_pt=1.45, rounded=True, round_radius=0.14)

    add_polyline(
        slide,
        [(q_exit_start_x, q_exit_y), (q_exit_x, q_exit_y), (q_exit_x, 1.77), (bus_x, 1.77), (bus_x, RY[4] - 0.21)],
        color=RS_TEAL_DK, width_pt=1.45, dash="dash",
    )
    for ry in RY:
        add_polyline(
            slide,
            [(bus_x, ry - 0.21), (cx_plus, ry - 0.21), (cx_plus, ry - 0.15)],
            color=RS_TEAL_DK, width_pt=1.25, dash="dash", arrow_end="triangle", arrow_size="sm",
        )

    rows_data = [
        ("1", "0.8", "1", "1", "done", RS_MUTED, rgb("#EAECEF"),
         [rgb("#EBEEF2"), rgb("#E6E8ED"), rgb("#EBEEF2")],
         [rgb("#D4DAE0"), rgb("#C2CAD2"), rgb("#DFE4E9"), rgb("#B5BFC9")],
         [rgb("#C6D9D5"), rgb("#B3CEC8"), rgb("#D4E4E1"), rgb("#A6C5BE")],
         "0.4", "0", None, 0.0, "0.00", False, False, False),
        ("2", "1.1", "2", "1", "wait O\u2083", RS_RED, rgb("#FADBD8"),
         [rgb("#FCE4E2"), rgb("#F8D3D0"), rgb("#FCE4E2")],
         [rgb("#E3B4B0"), rgb("#D48F8A"), rgb("#EBC9C6"), rgb("#C46D67")],
         [rgb("#D9C5C4"), rgb("#CAA9A7"), rgb("#E2D3D2"), rgb("#BE9390")],
         "1.5", "0", None, 0.0, "0.00", False, True, False),
        ("3", "1.4", "2", "2", "ready", RS_GREEN, rgb("#D3EEDC"),
         [rgb("#CDE5F8"), rgb("#E7DCF5"), rgb("#FBE4CB")],
         [rgb("#4D84B3"), rgb("#1D5B8F"), rgb("#86B1D4"), rgb("#296A9E")],
         [rgb("#2E9689"), rgb("#176158"), rgb("#55C7B8"), rgb("#217C70")],
         "1.2", "1", "1.2", 0.690, "0.69 \u2605", True, False, True),
        ("4", "0.9", "0", "1", "ready", RS_GREEN, rgb("#D3EEDC"),
         [rgb("#E6F1FA"), rgb("#F1EBF9"), rgb("#FDF2E4")],
         [rgb("#78A3C7"), rgb("#4579A3"), rgb("#9EC0DB"), rgb("#598DB6")],
         [rgb("#4BA99D"), rgb("#2A7F74"), rgb("#77CFC3"), rgb("#3B9488")],
         "0.4", "1", "0.4", 0.310, "0.31", False, False, True),
        ("5", "0.7", "1", "2", "wait O\u2084", RS_MUTED, rgb("#EAECEF"),
         [rgb("#EBEEF2"), rgb("#E6E8ED"), rgb("#EBEEF2")],
         [rgb("#D4DAE0"), rgb("#C2CAD2"), rgb("#DFE4E9"), rgb("#B5BFC9")],
         [rgb("#C6D9D5"), rgb("#B3CEC8"), rgb("#D4E4E1"), rgb("#A6C5BE")],
         "0.8", "0", None, 0.0, "0.00", False, False, False),
    ]

    add_line(slide, rail_x + rail_w * 0.5, RY[0] - 0.21, rail_x + rail_w * 0.5, RY[4] + 0.16, color=rgb("#CFD6DF"), width_pt=0.8, dash="dot")

    for idx, (num, p_dur, sk_i, q_req, st_txt, st_col, st_fill, feat_fills, h_cols, k_cols, logit, m_val, m_str, p_val, p_str, is_sel, is_bad, is_ready) in enumerate(rows_data):
        ry = RY[idx]
        active = is_sel or is_ready
        b_col = BLACK if active else (RS_RED if is_bad else rgb("#96A0AA"))
        b_pt = 1.70 if is_sel else (1.35 if (is_ready or is_bad) else 1.0)
        arr_col = RS_BLUE_HI if is_sel else (RS_RED if is_bad else (BLACK if is_ready else rgb("#96A0AA")))
        arr_pt = 1.50 if is_sel else (1.20 if (is_ready or is_bad) else 0.95)
        f_txt_col = BLACK if active else (RS_RED if is_bad else rgb("#63707D"))
        f_bdr = BLACK if active else (RS_RED if is_bad else rgb("#8C96A0"))
        f_bpt = 0.80 if active else (0.75 if is_bad else 0.60)

        # STAGE (a)
        nfill = RS_BLUE_HI if is_sel else (PALETTE["white"] if is_ready else (rgb("#FCE4E2") if is_bad else rgb("#ECEFF2")))
        ncol = PALETTE["white"] if is_sel else (BLACK if is_ready else (RS_RED if is_bad else rgb("#6B7783")))
        add_circle(slide, ax + 0.30, ry, 0.175, [[("O", "b"), (num, "b_")]], fill=nfill, border=b_col, border_pt=b_pt, color=ncol, size_pt=11.0)

        cell_h = 0.29
        fy = ry - cell_h / 2.0
        add_box(slide, ax + 0.52, fy, 0.42, cell_h, [[(p_dur, "b" if (active or is_bad) else "")]], fill=feat_fills[0], border=f_bdr, border_pt=f_bpt, font_name=FONT_SANS, color=f_txt_col, size_pt=9.8, margin_lr=0.0, margin_tb=0.0)
        add_box(slide, ax + 0.94, fy, 0.38, cell_h, [[("K", "b" if (active or is_bad) else ""), (sk_i, "b_" if (active or is_bad) else "_")]], fill=feat_fills[1], border=f_bdr, border_pt=f_bpt, font_name=FONT_SANS, color=f_txt_col, size_pt=9.8, margin_lr=0.0, margin_tb=0.0)
        add_box(slide, ax + 1.32, fy, 0.30, cell_h, [[(q_req, "b" if (active or is_bad) else "")]], fill=feat_fills[2], border=f_bdr, border_pt=f_bpt, font_name=FONT_SANS, color=f_txt_col, size_pt=9.8, margin_lr=0.0, margin_tb=0.0)
        if active or is_bad:
            add_box(slide, ax + 0.52, fy, 1.10, cell_h, fill=None, border=f_bdr, border_pt=1.20 if is_sel else 1.05)

        _draw_contiguous_strip(
            slide, ax + 1.69, fy, h_cols, cell_w=0.22, cell_h=cell_h,
            inner_border=BLACK if active else (RS_RED if is_bad else rgb("#8C96A0")), inner_pt=0.65,
            outer_border=BLACK if active else (RS_RED if is_bad else rgb("#8C96A0")), outer_pt=1.15 if is_sel else (0.95 if (is_ready or is_bad) else 0.70),
        )
        add_box(slide, ax + 2.66, fy, 0.94, cell_h, [[(st_txt, "bsans" if (is_ready or is_bad) else "sans")]], fill=st_fill, border=st_col, border_pt=1.15 if (is_ready or is_bad) else 0.75, rounded=True, round_radius=0.22, color=st_col, size_pt=9.8, margin_lr=0.0, margin_tb=0.0)
        add_line(slide, ax + aw - 0.12, ry, bx + 0.10, ry, color=arr_col, width_pt=arr_pt, arrow_end="triangle", arrow_size="sm")

        # STAGE (b)
        wo_fill = RS_LEARN if active else rgb("#BCE8E1")
        add_box(slide, bx + 0.10, ry - 0.155, 0.65, 0.31, [[("W", "b"), ("O", "b_"), (" h", "bi"), (num, "b_")]], fill=wo_fill, border=BLACK if active else rgb("#6B8B86"), border_pt=1.40 if is_sel else 1.05, rounded=True, round_radius=0.14, color=BLACK if active else rgb("#4A5D5A"), size_pt=10.2, margin_lr=0.0, margin_tb=0.0)
        add_line(slide, bx + 0.75, ry, bx + 0.87, ry, color=arr_col, width_pt=arr_pt, arrow_end="triangle", arrow_size="sm")
        _draw_contiguous_strip(
            slide, bx + 0.88, ry - 0.12, k_cols, cell_w=0.148, cell_h=0.24,
            inner_border=BLACK if active else rgb("#8C96A0"), inner_pt=0.60,
            outer_border=BLACK if active else rgb("#8C96A0"), outer_pt=1.10 if is_sel else (0.90 if is_ready else 0.70),
        )
        add_line(slide, bx + 1.472, ry, cx_plus - 0.15, ry, color=rgb("#E4F2FC") if is_sel else (rgb("#FDF0EE") if is_bad else PALETTE["white"]), width_pt=3.2)
        add_line(slide, bx + 1.472, ry, cx_plus - 0.15, ry, color=arr_col, width_pt=arr_pt, arrow_end="triangle", arrow_size="sm")
        add_circle(slide, cx_plus, ry, 0.145, [[("+", "b")]], fill=RS_NONLRN if active else rgb("#FAF2DA"), border=BLACK if active else rgb("#96A0AA"), border_pt=1.5 if is_sel else 1.15, color=BLACK if active else rgb("#727E8A"), size_pt=12.6)

        add_line(slide, cx_plus + 0.15, ry, x_tanh - 0.01, ry, color=arr_col, width_pt=arr_pt, arrow_end="triangle", arrow_size="sm")
        add_line(slide, x_tanh + w_tanh, ry, x_ao - 0.01, ry, color=arr_col, width_pt=arr_pt, arrow_end="triangle", arrow_size="sm")
        add_line(slide, x_ao + w_ao, ry, bx + 3.58, ry, color=arr_col, width_pt=arr_pt, arrow_end="triangle", arrow_size="sm")

        lcol = RS_BLUE_HI if is_sel else (RS_RED if is_bad else (BLACK if is_ready else rgb("#7E8A96")))
        add_textbox(slide, bx + 3.58, ry - 0.16, 0.66, 0.32, [[(logit, "b" if (is_sel or is_bad or is_ready) else "")]], font_name=FONT_SERIF, size_pt=13.4 if (is_sel or is_bad) else 12.2, color=lcol, align=PP_ALIGN.CENTER, margin_lr=0.0, word_wrap=False)
        add_line(slide, bx + 4.26, ry, cx + 0.11, ry, color=arr_col, width_pt=arr_pt, arrow_end="triangle", arrow_size="sm")

        # STAGE (c) — Exact Appendix A mathematical masking `-10^4` (NO `-1e4`!)
        if is_ready:
            m_bg, m_bd, m_tc = rgb("#D3EEDC"), RS_GREEN, RS_GREEN
        elif is_bad:
            m_bg, m_bd, m_tc = rgb("#FADBD8"), RS_RED, RS_RED
        else:
            m_bg, m_bd, m_tc = rgb("#ECEFF3"), rgb("#AAB4BE"), rgb("#7A8694")
        add_box(slide, cx + 0.12, ry - 0.15, 0.32, 0.30, [[(m_val, "b")]], fill=m_bg, border=m_bd, border_pt=1.45 if (is_ready or is_bad) else 1.0, color=m_tc, size_pt=11.6, margin_lr=0.0, margin_tb=0.0)
        add_line(slide, cx + 0.44, ry, cx + 0.56, ry, color=arr_col, width_pt=arr_pt, arrow_end="triangle", arrow_size="sm")

        ml_bg = RS_NONLRN if is_ready else (rgb("#FADBD8") if is_bad else rgb("#ECEFF3"))
        ml_bd = BLACK if is_sel else (rgb("#9C7A2A") if is_ready else (RS_RED if is_bad else rgb("#AAB4BE")))
        ml_tc = BLACK if is_ready else (RS_RED if is_bad else rgb("#7A8694"))
        if m_str is None:
            ml_segs = [("\u221210", "b" if is_bad else ""), ("4", "b^" if is_bad else "^")]
        else:
            ml_segs = [(m_str, "b")]
        add_box(slide, cx + 0.58, ry - 0.15, 0.60, 0.30, [ml_segs], fill=ml_bg, border=ml_bd, border_pt=1.45 if (is_sel or is_bad) else 1.05, rounded=True, round_radius=0.10, font_name=FONT_SERIF, color=ml_tc, size_pt=11.2 if not is_ready else 11.6, margin_lr=0.0, margin_tb=0.0)
        add_line(slide, cx + 1.18, ry, x_sm - 0.01, ry, color=arr_col, width_pt=arr_pt, arrow_end="triangle", arrow_size="sm")
        add_line(slide, x_sm + w_sm, ry, rail_x - 0.02, ry, color=arr_col, width_pt=arr_pt, arrow_end="triangle", arrow_size="sm")

        add_box(
            slide, rail_x, ry - 0.125, rail_w, 0.25,
            [[("masked (m\u2082=0)", "bsans", RS_RED, 9.0)]] if is_bad else None,
            fill=rgb("#FCE8E6") if is_bad else rgb("#EEF1F5"),
            border=RS_RED if is_bad else rgb("#BFC7D1"), border_pt=1.05 if is_bad else 0.75,
            rounded=True, round_radius=0.16, margin_lr=0.0, margin_tb=0.0,
        )
        if p_val > 0:
            add_box(slide, rail_x, ry - 0.125, rail_w * p_val, 0.25, fill=RS_BLUE_HI if is_sel else rgb("#72A0C7"), border=BLACK, border_pt=0.95, rounded=True, round_radius=0.16)
        p_col = RS_BLUE_HI if is_sel else (RS_RED if is_bad else (BLACK if is_ready else rgb("#7E8A96")))
        add_textbox(
            slide, rail_x + rail_w + 0.05, ry - 0.16, 0.74, 0.32,
            [[(p_str, "b" if (is_sel or is_bad or is_ready) else "")]],
            font_name=FONT_SERIF, size_pt=12.0 if is_sel else 11.4,
            color=p_col, align=PP_ALIGN.LEFT, margin_lr=0.0, word_wrap=False,
        )

    y_ax = RY[4] + 0.155
    add_line(slide, rail_x, y_ax, rail_x + rail_w, y_ax, color=rgb("#9AA5B1"), width_pt=0.75)
    for frac, tlbl in [(0.0, "0"), (0.5, "0.5"), (1.0, "1.0")]:
        tx = rail_x + rail_w * frac
        add_line(slide, tx, y_ax - 0.025, tx, y_ax + 0.03, color=rgb("#6C7885"), width_pt=0.8)
        add_textbox(slide, tx - 0.18, y_ax + 0.02, 0.36, 0.18, [[(tlbl, "bsans")]], font_name=FONT_SANS, size_pt=9.2, color=PALETTE["ink_soft"], align=PP_ALIGN.CENTER, margin_lr=0.0, margin_tb=0.0, word_wrap=False)

    y_bar0 = RY[0] - 0.25
    h_bar  = (RY[4] + 0.25) - y_bar0

    add_box(slide, x_tanh, y_bar0, w_tanh, h_bar, fill=RS_NONLRN, border=BLACK, border_pt=1.55, rounded=True, round_radius=0.14)
    add_activation_glyph(slide, x_tanh + w_tanh / 2, y_bar0 + 0.32, r=0.14, kind="tanh", fill=PALETTE["white"], border=BLACK, border_pt=1.25, curve_pt=1.25)
    add_angled_label(
        slide, x_tanh + w_tanh / 2 - 1.06, y_bar0 + 1.53, 2.12, 0.30,
        [[("tanh (", "b"), ("q", "bi"), ("t", "b_"), (" + ", "b"), ("k", "bi"), ("i", "b_"), (")", "b")]],
        angle_deg=-90.0, font_name=FONT_SERIF, size_pt=12.6, color=BLACK,
    )

    add_box(slide, x_ao, y_bar0, w_ao, h_bar, fill=RS_LEARN, border=BLACK, border_pt=1.55, rounded=True, round_radius=0.14)
    add_angled_label(
        slide, x_ao + w_ao / 2 - 1.20, y_bar0 + h_bar / 2 - 0.15, 2.40, 0.30,
        [[("Scorer   ", "bsans"), ("a", "bi"), ("O", "b_"), ("T", "b^"), (" \u00b7 (\u00b7)", "b")]],
        angle_deg=-90.0, font_name=FONT_SERIF, size_pt=12.8, color=BLACK,
    )

    add_box(slide, x_sm, y_bar0, w_sm, h_bar, fill=RS_NONLRN, border=BLACK, border_pt=1.55, rounded=True, round_radius=0.14)
    add_angled_label(slide, x_sm + w_sm / 2 - 1.20, y_bar0 + h_bar / 2 - 0.15, 2.40, 0.30, "Masked Softmax", angle_deg=-90.0, size_pt=12.0, color=BLACK, bold=True)

    # BOTTOM PAPER-FAITHFUL SUMMARY CARDS [y=5.46..6.16, h=0.70 in]
    y_bot_card, h_bot_card = 5.46, 0.70

    # Stage (a): exact Section IV-C feasible set symbol F_t^O!
    add_box(slide, ax + 0.08, y_bot_card, aw - 0.16, h_bot_card, fill=PALETTE["white"], border=BLACK, border_pt=1.25, rounded=True, round_radius=0.12)
    add_textbox(
        slide, ax + 0.10, y_bot_card + 0.05, aw - 0.20, 0.28,
        [[("Feasible Set: ", "bsans", BLACK, 11.4), ("F", "bi", RS_GREEN, 12.4), ("t", "b_", RS_GREEN, 12.4), ("O", "b^", RS_GREEN, 12.4), (" = {O", "b", RS_GREEN, 12.2), ("3", "b_", RS_GREEN, 12.2), (", O", "b", RS_GREEN, 12.2), ("4", "b_", RS_GREEN, 12.2), ("}", "b", RS_GREEN, 12.2)]],
        font_name=FONT_SERIF, size_pt=11.6, color=BLACK, align=PP_ALIGN.CENTER, margin_lr=0.0, word_wrap=False,
    )
    add_textbox(
        slide, ax + 0.10, y_bot_card + 0.37, aw - 0.20, 0.28,
        [[("m", "bi", BLACK, 11.0), ("i", "b_", BLACK, 11.0), (" = 1", "b", BLACK, 10.8), (" iff ", "bsans", rgb("#384654"), 10.0), ("P(i)", "bi", rgb("#384654"), 10.4), (" done \u2227 released \u2227 (M, W) feasible", "bsans", rgb("#384654"), 9.8)]],
        font_name=FONT_SERIF, size_pt=10.4, color=BLACK, align=PP_ALIGN.CENTER, margin_lr=0.0, word_wrap=False,
    )

    # Stage (b): 'Mask Effect' replacing informal 'Counter-example'
    add_box(slide, bx + 0.08, y_bot_card, bw - 0.16, h_bot_card, fill=rgb("#F7F9FB"), border=BLACK, border_pt=1.25, rounded=True, round_radius=0.12)
    add_textbox(
        slide, bx + 0.10, y_bot_card + 0.05, bw - 0.20, 0.28,
        [[("Shared Params: ", "bsans", BLACK, 10.8), ("W", "b"), ("g", "b_"), (" \u2208 \u211d", "b"), ("d\u00d73d", "b^"), (", W", "b"), ("O", "b_"), (" \u2208 \u211d", "b"), ("d\u00d7d", "b^"), (", a", "b"), ("O", "b_"), (" \u2208 \u211d", "b"), ("d", "b^"), (" (d=128)", "bsans", rgb("#384654"), 10.0)]],
        font_name=FONT_SERIF, size_pt=10.8, color=BLACK, align=PP_ALIGN.CENTER, margin_lr=0.0, word_wrap=False,
    )
    add_textbox(
        slide, bx + 0.10, y_bot_card + 0.37, bw - 0.20, 0.28,
        [[("Mask Effect: ", "bsans", RS_RED, 10.4), ("O", "b", RS_RED, 11.0), ("2", "b_", RS_RED, 11.0), (" scores highest (", "bsans", BLACK, 10.0), ("u", "bi", RS_RED, 11.0), ("2", "b_", RS_RED, 11.0), ("=1.5", "b", RS_RED, 10.4), (" > ", "b", BLACK, 10.4), ("u", "bi", RS_BLUE_HI, 11.0), ("3", "b_", RS_BLUE_HI, 11.0), ("=1.2", "b", RS_BLUE_HI, 10.4), (") yet waits on O", "bsans", BLACK, 10.0), ("3", "b_", BLACK, 10.0)]],
        font_name=FONT_SERIF, size_pt=10.6, color=BLACK, align=PP_ALIGN.CENTER, margin_lr=0.0, word_wrap=False,
    )

    # Stage (c): math card with exact `-10^4` + sample card with exact Table I symbols
    c_math_w = 1.76
    add_box(slide, cx + 0.08, y_bot_card, c_math_w, h_bot_card, fill=rgb("#FDF8E8"), border=BLACK, border_pt=1.25, rounded=True, round_radius=0.12)
    add_textbox(
        slide, cx + 0.09, y_bot_card + 0.05, c_math_w - 0.02, 0.28,
        [[("u\u0303", "bi"), ("i", "b_"), (" = u", "b"), ("i", "b_"), (" (m", "bsans", RS_GREEN, 9.6), ("i", "b_", RS_GREEN, 9.6), ("=1), ", "bsans", RS_GREEN, 9.6), ("\u221210", "b", RS_RED, 10.2), ("4", "b^", RS_RED, 10.2)]],
        font_name=FONT_SERIF, size_pt=10.4, color=BLACK, align=PP_ALIGN.CENTER, margin_lr=0.0, word_wrap=False,
    )
    add_textbox(
        slide, cx + 0.09, y_bot_card + 0.37, c_math_w - 0.02, 0.28,
        [[("\u03c0", "b"), ("O", "b_"), (" = exp(u\u0303", "b"), ("i", "b_"), (") / \u03a3", "b"), ("j", "b_"), ("exp(u\u0303", "b"), ("j", "b_"), (")", "b")]],
        font_name=FONT_SERIF, size_pt=10.0, color=BLACK, align=PP_ALIGN.CENTER, margin_lr=0.0, word_wrap=False,
    )

    gold_ink = rgb("#8C6A1C")
    add_line(slide, x_sm + w_sm / 2, y_bot_card, x_sm + w_sm / 2, y_bar0 + h_bar + 0.01, color=gold_ink, width_pt=1.15, dash="dot")
    add_circle(slide, x_sm + w_sm / 2, y_bot_card, 0.032, fill=gold_ink, border=gold_ink, border_pt=0.4)

    c_out_x = cx + 0.08 + c_math_w + 0.06
    c_out_w = (cx + cw - 0.08) - c_out_x
    add_box(slide, c_out_x, y_bot_card, c_out_w, h_bot_card, fill=rgb("#E4F2FC"), border=RS_BLUE_HI, border_pt=1.6, rounded=True, round_radius=0.12)
    add_textbox(
        slide, c_out_x + 0.03, y_bot_card + 0.04, c_out_w - 0.06, 0.28,
        [[("Sample: i", "bsans"), ("t", "b_"), (" = O", "bsans"), ("3", "b_"), (" (0.69)", "bsans")]],
        font_name=FONT_SANS, size_pt=10.6, color=RS_BLUE_HI, align=PP_ALIGN.CENTER, margin_lr=0.0, word_wrap=False,
    )
    st_pill = add_box(
        slide, c_out_x + 0.04, y_bot_card + 0.36, c_out_w - 0.08, 0.28,
        [[("h(O", "b"), ("3", "b_"), ("), k", "bi"), ("3", "b_"), (", q", "bi"), ("3", "b_"), (" \u2192 St. Ptr", "bsans", PALETTE["white"], 9.6)]],
        fill=RS_BLUE_HI, border=BLACK, border_pt=1.05, rounded=True, round_radius=0.22,
        font_name=FONT_SERIF, size_pt=9.8, color=PALETTE["white"], margin_lr=0.0, margin_tb=0.0,
    )
    st_pill.text_frame.word_wrap = False

    x_loop = cx + cw + 0.18
    add_polyline(
        slide,
        [(cx + cw - 0.08, RY[2]), (x_loop, RY[2]), (x_loop, 1.77), (px2 + pw_s / 2, 1.77), (px2 + pw_s / 2, q_y0 + dag_h_top + 0.01)],
        color=RS_BLUE_HI, width_pt=1.80, arrow_end="triangle", arrow_size="sm",
    )
    add_polyline(
        slide,
        [(x_loop, RY[2]), (x_loop, y_bot_card + h_bot_card / 2), (cx + cw - 0.08, y_bot_card + h_bot_card / 2)],
        color=RS_BLUE_HI, width_pt=1.65, arrow_end="triangle", arrow_size="sm",
    )
    add_circle(slide, x_loop, RY[2], 0.046, fill=RS_BLUE_HI, border=RS_BLUE_HI, border_pt=0.5)

    y_br = 6.42
    _draw_rich_stage_bracket(slide, ax, ax + aw, y_br, [[("(a) State ", "b"), ("s", "bi"), ("t", "b_"), (" & Precedence Mask", "b")]], size_pt=13.8)
    _draw_rich_stage_bracket(slide, bx, bx + bw, y_br, [[("(b) Additive Pointer Scoring", "b")]], size_pt=13.8)
    _draw_rich_stage_bracket(slide, cx, cx + cw, y_br, [[("(c) Masking & Decision-Making", "b")]], size_pt=13.8)

    set_notes(
        slide,
        "Module B6. Operation pointer head (main_tsmc.tex Section IV-C; TaskPointer in models/hb_gat_pn.py). Refined v8 aligns every symbol with the IEEE TSMC manuscript: feasible set F_t^O = {O_3, O_4}, mathematical mask value -10^4 (Appendix A), APAL precedence matrix label 'Precedence (5x5)' (replacing borrowed 'O2O Mask'), formal 'Mask Effect' note, and Table I symbols P(i), k_i, q_i.\n"
        "Layout reference: ReSched, ICLR 2026, Fig. 2.",
    )


def main():
    parser = argparse.ArgumentParser(description="Build Slide 11 (B6) in refined_v8.")
    parser.add_argument("--pptx", default=DEFAULT_PPTX, help="Target PPTX path.")
    args = parser.parse_args()
    prs = Presentation(args.pptx)
    build_b6_v8(prs.slides[10])  # Slide 11: B6
    prs.save(args.pptx)
    print("Updated Slide 11 (B6) in", args.pptx)


if __name__ == "__main__":
    main()
