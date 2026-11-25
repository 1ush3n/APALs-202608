"""Build Slide 13 (B8) in refined_v8: Redesigned zero-void top layout & bottom substep-unrolling row (main_tsmc.tex Section IV-C & Appendix A).

v8 layout & terminology redesign addressing BOTH pinned comments on Slide 13 (Images 0 & 1):
1. Relocated the Substep 1 / Substep 2 titles + inline 'Shared Worker Pointer (r = 1 -> 2)' connector
   to the BOTTOM below the dotted `x q` frame (matching IEEE / ReSched bottom-caption grammar!).
2. Eliminated ALL sparse floating upper headroom above the dotted `x q` frame:
   - Right above the dotted border sits ONLY ONE tight horizontal row:
     [W_2..W_6 (Substep r=1)]  --  [w_t^(1) = W_2 amber pill filling the center pocket!]  --  [W_2..W_6 (Substep r=2)]
   - A sleek amber feedback wire hugs just 0.14 in above Draw 1's probabilities and drops straight
     into the center `w_t^(1) = W_2` badge, then down into the center causal state-update bridge.
3. Expanded the interior cards cleanly (`ch: 3.36 -> 3.68 in`) with balanced 4-card causal bridge
   (`State Update (r=1->2)` -> `1. Mask Update` -> `2. Partial-Team Pool` -> `3. Dynamic Features`),
   zero duplicate wording, and 100% paper-faithful terminology ('Substep r = 1 / 2', 'Masked Softmax',
   'Partial-Team Pool', 'Dynamic EFT & Scarcity').
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
    add_angled_label,
    add_box,
    add_circle,
    add_line,
    add_polyline,
    add_swatch_strip,
    add_textbox,
    clear_slide,
    rgb,
    set_notes,
)

DEFAULT_PPTX_PATH = os.path.join(REFINED_V8_DIR, "APAL_Modules_English_Refined_v8.pptx")

DAN_CARD1_BG = rgb("#FAF3F0")
DAN_CARD2_BG = rgb("#EFF6F8")
DAN_LAV      = rgb("#E5DCEF")
DAN_CREAM    = rgb("#FDF1C8")
DAN_PEACH    = rgb("#FDE3D1")
DAN_SAGE     = rgb("#D7E7CB")
DAN_SKY      = rgb("#E5F3FA")
DAN_RED_BG   = rgb("#FADBD8")
DAN_RED      = rgb("#A62627")
DAN_OK_BG    = rgb("#D8F1E1")
DAN_OK       = rgb("#1D7344")
DAN_WK_SEL   = rgb("#A65618")
DAN_MUTED    = rgb("#AEB7C2")
UNROLL_NAVY  = rgb("#1D3B56")


def _draw_worker_panel(
    slide,
    x0: float,
    panel_w: float,
    draw_idx: int,
    *,
    y_prob: float,
    y_circ: float,
    cy_top: float,
    ch: float,
):
    """Draw one dense, DAN Fig. 3 autoregressive substep card (Substep r=1 or r=2)."""
    is_d1 = (draw_idx == 1)
    bg_tint = DAN_CARD1_BG if is_d1 else DAN_CARD2_BG
    side_lbl = "Substep r = 1 \u2014 First Worker" if is_d1 else "Substep r = 2 \u2014 Second Worker"

    col_margin = 0.42
    span = panel_w - 2 * col_margin
    col_xs = [x0 + col_margin + k * (span / 4.0) for k in range(5)]

    add_box(slide, x0 - 0.075, cy_top + 0.075, panel_w, ch, fill=bg_tint, border=rgb("#98A2AC"), border_pt=1.05, rounded=True, round_radius=0.05)
    add_box(slide, x0, cy_top, panel_w, ch, fill=bg_tint, border=BLACK, border_pt=1.75, rounded=True, round_radius=0.05)

    if is_d1:
        add_angled_label(slide, x0 - 1.505, cy_top + ch / 2 - 0.13, 2.54, 0.26, side_lbl, angle_deg=-90.0, size_pt=11.4, color=BLACK, bold=True)
    else:
        add_angled_label(slide, x0 + panel_w - 1.00, cy_top + ch / 2 - 0.02, 2.54, 0.26, side_lbl, angle_deg=90.0, size_pt=11.4, color=BLACK, bold=True)

    if is_d1:
        cands = [
            ("2", "0.52 \u2605", "sel",  "eligible", [rgb("#C4722E"), rgb("#8F4812"), rgb("#E0A772")]),
            ("3", "0",          "off",  "no K\u2082", [rgb("#CFD6DE"), rgb("#AAB4C0"), rgb("#DFE4EA")]),
            ("4", "0",          "off",  "bound S\u2082", [rgb("#CFD6DE"), rgb("#AAB4C0"), rgb("#DFE4EA")]),
            ("5", "0.31",       "elig", "eligible", [rgb("#C4722E"), rgb("#A65C1E"), rgb("#E5B689")]),
            ("6", "0.17",       "elig", "eligible", [rgb("#D4945A"), rgb("#B8763C"), rgb("#EDD0B0")]),
        ]
    else:
        cands = [
            ("2", "0",          "used", "used",     [rgb("#E5B8B6"), rgb("#CE9492"), rgb("#F0D2D0")]),
            ("3", "0",          "off",  "no K\u2082", [rgb("#CFD6DE"), rgb("#AAB4C0"), rgb("#DFE4EA")]),
            ("4", "0",          "off",  "bound S\u2082", [rgb("#CFD6DE"), rgb("#AAB4C0"), rgb("#DFE4EA")]),
            ("5", "0.64 \u2605", "sel",  "eligible", [rgb("#A65618"), rgb("#7A3A0A"), rgb("#D48A49")]),
            ("6", "0.36",       "elig", "eligible", [rgb("#C98344"), rgb("#9E5E24"), rgb("#E6B98B")]),
        ]

    r_circ = 0.20
    # Generous balanced vertical rhythm inside ch = 3.68 in
    y_sm_top, h_sm = cy_top + 0.14, 0.33    # 0.14 .. 0.47
    y_mk_top, h_mk = cy_top + 0.67, 0.30    # 0.67 .. 0.97 (mid = cy_top + 0.82)
    y_vw_top, h_vw = cy_top + 1.18, 0.32    # 1.18 .. 1.50
    y_th_c         = cy_top + 1.71          # 1.71 (r = 0.115)
    y_cb_top, h_cb = cy_top + 1.92, 0.32    # 1.92 .. 2.24
    y_sw_top       = cy_top + 2.43          # 2.43 .. 2.64 (cell_h = 0.21, mid = cy_top + 2.535)
    y_lbl_top      = cy_top + 2.66          # 2.66 .. 2.88
    y_q_top, h_q   = cy_top + 2.98, 0.56    # 2.98 .. 3.54 (mid = cy_top + 3.26)

    in_x = x0 + 0.16
    in_w = panel_w - 0.32
    cx_p = x0 + panel_w / 2.0

    for idx, (num, p_str, st, m_lbl, sw_cols) in enumerate(cands):
        cx = col_xs[idx]
        p_col = DAN_WK_SEL if st == "sel" else (DAN_RED if st == "used" else (BLACK if st == "elig" else rgb("#8C96A2")))
        if is_d1 and idx == 0:
            add_textbox(slide, cx + 0.05, y_prob, 0.78, 0.22, [[(p_str, "b")]], font_name=FONT_SERIF, size_pt=11.8, color=p_col, align=PP_ALIGN.LEFT, margin_lr=0.0, margin_tb=0.0)
        else:
            add_textbox(slide, cx - 0.36, y_prob, 0.72, 0.22, [[(p_str, "b" if st in ("sel", "used") else "")]], font_name=FONT_SERIF, size_pt=11.8 if st == "sel" else 11.0, color=p_col, align=PP_ALIGN.CENTER, margin_lr=0.0, margin_tb=0.0)

        if st == "sel":
            cfill, cbdr, cpt, ctc = DAN_WK_SEL, BLACK, 1.95, PALETTE["white"]
        elif st == "used":
            cfill, cbdr, cpt, ctc = DAN_RED_BG, DAN_RED, 1.85, DAN_RED
        elif st == "elig":
            cfill, cbdr, cpt, ctc = PALETTE["white"], BLACK, 1.5, BLACK
        else:
            cfill, cbdr, cpt, ctc = rgb("#F2F4F7"), DAN_MUTED, 1.1, rgb("#85909C")
        add_circle(slide, cx, y_circ, r_circ, [[("W", "b"), (num, "b_")]], fill=cfill, border=cbdr, border_pt=cpt, color=ctc, size_pt=12.2)

        a_col = DAN_WK_SEL if st == "sel" else (DAN_RED if st == "used" else (BLACK if st == "elig" else DAN_MUTED))
        a_pt = 1.95 if st == "sel" else (1.35 if st in ("elig", "used") else 0.95)
        a_dash = "dash" if st in ("off", "used") else None
        add_line(slide, cx, y_sm_top, cx, y_circ + r_circ + 0.015, color=a_col, width_pt=a_pt, dash=a_dash, arrow_end="triangle", arrow_size="sm")

        ma_col = DAN_OK if st in ("sel", "elig") else DAN_MUTED
        add_line(slide, cx, y_mk_top, cx, y_sm_top + h_sm + 0.01, color=ma_col, width_pt=1.35 if st in ("sel", "elig") else 0.95, dash=None if st in ("sel", "elig") else "dash", arrow_end="triangle", arrow_size="sm")

        if st in ("sel", "elig"):
            mk_f, mk_b, mk_pt, mk_tc = DAN_OK_BG, BLACK, 1.3, DAN_OK
        elif st == "used":
            mk_f, mk_b, mk_pt, mk_tc = DAN_RED_BG, DAN_RED, 1.65, DAN_RED
        else:
            mk_f, mk_b, mk_pt, mk_tc = rgb("#EAECEF"), rgb("#AEB6BF"), 1.0, rgb("#7B8692")
        add_box(
            slide, cx - 0.34, y_mk_top, 0.68, h_mk,
            [[(m_lbl, "bsans" if st in ("sel", "elig", "used") else "sans")]],
            fill=mk_f, border=mk_b, border_pt=mk_pt, rounded=True, round_radius=0.24,
            color=mk_tc, size_pt=9.6 if st != "off" else 9.2, margin_lr=0.0, margin_tb=0.0,
        )

        sa_col = DAN_WK_SEL if st in ("sel", "elig") else DAN_MUTED
        add_line(slide, cx, y_vw_top, cx, y_mk_top + h_mk + 0.01, color=sa_col, width_pt=1.25 if st in ("sel", "elig") else 0.95, arrow_end="triangle", arrow_size="sm")
        add_line(slide, cx, y_sw_top, cx, y_cb_top + h_cb + 0.01, color=sa_col, width_pt=1.25 if st in ("sel", "elig") else 0.95, arrow_end="triangle", arrow_size="sm")

        add_swatch_strip(
            slide, cx - 0.25, y_sw_top, sw_cols,
            cell_w=0.155, cell_h=0.21, gap=0.015,
            border=BLACK if st in ("sel", "elig") else (DAN_RED if st == "used" else rgb("#9AA4AE")), border_pt=0.7,
        )
        w_tc = DAN_WK_SEL if st in ("sel", "elig") else (DAN_RED if st == "used" else rgb("#7E8A96"))
        add_textbox(slide, cx - 0.25, y_lbl_top, 0.50, 0.22, [[("W", "b"), (num, "b_")]], font_name=FONT_SERIF, size_pt=11.4, color=w_tc, align=PP_ALIGN.CENTER, margin_lr=0.0, margin_tb=0.0)

    # Consistent 'Masked Softmax' capitalization
    add_box(
        slide, in_x, y_sm_top, in_w, h_sm,
        [[("Masked Softmax  over eligible workers", "bsans")]],
        fill=DAN_SAGE, border=BLACK, border_pt=1.45, rounded=True, round_radius=0.20,
        color=BLACK, size_pt=11.2, margin_lr=0.01, margin_tb=0.0,
    )
    add_box(
        slide, in_x, y_vw_top, in_w, h_vw,
        [[("Score Projection   v", "bsans"), ("W", "b_"), ("T", "b^"), (" \u00b7 (\u00b7)", "bsans")]],
        fill=DAN_PEACH, border=BLACK, border_pt=1.4, rounded=True, round_radius=0.20,
        color=BLACK, size_pt=11.2, margin_lr=0.01, margin_tb=0.0,
    )

    add_line(slide, cx_p, y_cb_top, cx_p, y_th_c + 0.115, color=BLACK, width_pt=1.35)
    add_line(slide, cx_p, y_th_c - 0.115, cx_p, y_vw_top + h_vw + 0.01, color=BLACK, width_pt=1.35, arrow_end="triangle", arrow_size="sm")
    add_activation_glyph(slide, cx_p, y_th_c, r=0.115, kind="tanh", fill=PALETTE["white"], border=BLACK, border_pt=1.35, curve_pt=1.25)
    add_textbox(slide, cx_p + 0.15, y_th_c - 0.11, 0.86, 0.22, [[("tanh(\u00b7)", "isans")]], font_name=FONT_SANS, size_pt=10.2, color=BLACK, align=PP_ALIGN.LEFT, margin_lr=0.0, margin_tb=0.0)

    r_sup = "(1)" if is_d1 else "(2)"
    add_box(
        slide, in_x, y_cb_top, in_w, h_cb,
        [[("Combine  q", "bsans"), (r_sup, "b^"), (" + k", "bsans"), ("w", "b_"), (r_sup, "b^"), ("  (EFT & scarcity)", "sans", PALETTE["ink_soft"], 9.8)]],
        fill=DAN_CREAM, border=BLACK, border_pt=1.4, rounded=True, round_radius=0.20,
        color=BLACK, size_pt=11.0, margin_lr=0.01, margin_tb=0.0,
    )

    q_fill = DAN_LAV if is_d1 else rgb("#FCE8D5")
    if is_d1:
        q_lines = [
            [("Query ", "bsans"), ("q", "bi"), ("(1)", "b^"), (" (O", "bsans"), ("3", "b_"), (", S", "bsans"), ("3", "b_"), (", g", "bsans"), ("t", "b_"), (", team=\u2205)", "bsans")],
            [("Partial team w", "sans", PALETTE["ink_soft"], 9.8), ("(<1)", "^", PALETTE["ink_soft"], 9.8), ("=\u2205  \u00b7  progress 0/2", "sans", PALETTE["ink_soft"], 9.8)],
        ]
    else:
        q_lines = [
            [("Query ", "bsans"), ("q", "bi"), ("(2)", "b^"), (" (O", "bsans"), ("3", "b_"), (", S", "bsans"), ("3", "b_"), (", g", "bsans"), ("t", "b_"), (", ", "bsans"), ("{W\u2082}", "bsans", DAN_WK_SEL), (")", "bsans")],
            [("Partial team (W\u2082)  \u00b7  1/2  \u00b7  updated h", "sans", PALETTE["ink_soft"], 9.8), ("team", "_", PALETTE["ink_soft"], 9.8)],
        ]
    add_box(
        slide, in_x, y_q_top, in_w, h_q, q_lines,
        fill=q_fill, border=BLACK, border_pt=1.5, rounded=True, round_radius=0.18,
        color=BLACK, size_pt=11.0, line_spacing=1.04, margin_lr=0.02, margin_tb=0.01,
    )

    if is_d1:
        qx_w = x0 + 0.08
        add_polyline(slide, [(in_x, y_q_top + h_q / 2), (qx_w, y_q_top + h_q / 2), (qx_w, y_cb_top + h_cb / 2), (in_x - 0.01, y_cb_top + h_cb / 2)], color=BLACK, width_pt=1.45, arrow_end="triangle", arrow_size="sm")
    else:
        qx_w = x0 + panel_w - 0.08
        add_polyline(slide, [(in_x + in_w, y_q_top + h_q / 2), (qx_w, y_q_top + h_q / 2), (qx_w, y_cb_top + h_cb / 2), (in_x + in_w + 0.01, y_cb_top + h_cb / 2)], color=DAN_WK_SEL, width_pt=1.5, arrow_end="triangle", arrow_size="sm")

    return {
        "col_xs": col_xs,
        "y_circ": y_circ,
        "r_circ": r_circ,
        "cy_top": cy_top,
        "ch": ch,
        "y_mk": y_mk_top,
        "h_mk": h_mk,
        "y_sw": y_sw_top,
        "y_q": y_q_top,
        "h_q": h_q,
        "in_x": in_x,
        "in_w": in_w,
    }


def build_b8_v8(slide):
    """Slide 13 · B8 redesigned zero-void top & bottom-anchored unrolling caption row (refined_v8)."""
    clear_slide(slide)

    # =========================================================================
    # COMPACT OPTICALLY CENTERED ENVELOPE [x = 1.50 .. 11.84, w = 10.34 in]
    # Vertical span: amber wire [0.96] -> circles + center pill [1.46] -> dotted frame [1.76..5.86] -> bottom unroll row [6.04..6.36]
    # =========================================================================
    enc_x0 = 1.50
    enc_w  = 10.34

    pw   = 3.68               # Draw 1 / Draw 2 card width
    bw_c = 1.48               # Tight center causal bridge
    xL   = 1.92               # Substep 1: [1.92 .. 5.60]
    bx0  = 5.87               # Bridge: [5.87 .. 7.35] (0.27 in left gutter)
    bx_c = bx0 + bw_c / 2.0   # 6.61
    xR   = 7.66               # Substep 2: [7.66 .. 11.34] (0.31 in right gutter)

    # ULTRA-TIGHT TOP BAND (NO floating title tier above the circles!)
    y_hw    = 0.96            # Sleek amber feedback wire hugging just 0.10 in above probabilities!
    y_prob  = 1.05            # 1.05 .. 1.26
    y_circ  = 1.46            # 1.26 .. 1.66 (r = 0.20)
    enc_y0  = 1.76            # Dotted enclosure snugly 0.10 in below circles!
    cy_top  = 1.92            # Card top inside dotted frame
    ch      = 3.68            # Card bottom = 5.60
    enc_h   = 4.02            # Dotted frame: [1.76 .. 5.78]

    add_box(
        slide, enc_x0, enc_y0, enc_w, enc_h,
        fill=PALETTE["white"], border=BLACK, border_pt=1.55, dash="dot", rounded=True, round_radius=0.04,
    )

    mL = _draw_worker_panel(
        slide, xL, pw, draw_idx=1,
        y_prob=y_prob, y_circ=y_circ, cy_top=cy_top, ch=ch,
    )
    mR = _draw_worker_panel(
        slide, xR, pw, draw_idx=2,
        y_prob=y_prob, y_circ=y_circ, cy_top=cy_top, ch=ch,
    )

    # `x q` dead-centered in upper-right pocket inside dotted enclosure
    add_textbox(
        slide, xR + pw + 0.02, enc_y0 + 0.04, 0.46, 0.34,
        [[("\u00d7 q", "bi")]], font_name=FONT_SERIF, size_pt=16.5, color=BLACK, align=PP_ALIGN.CENTER,
        margin_lr=0.0, margin_tb=0.0,
    )

    # =========================================================================
    # INTEGRATED TOP-CENTER FEEDBACK BADGE (FILLS THE MIDDLE GAP AT CIRCLE HEIGHT!)
    # Sleek amber wire: W_2 (Substep 1) -> y_hw -> drops into center `w_t^(1) = W_2` badge -> Bridge!
    # =========================================================================
    sel_pill_y0 = 1.27
    sel_pill_h  = 0.38        # [1.27 .. 1.65] -> exact vertical match to W_2..W_6 circles [1.26 .. 1.66]!
    add_polyline(
        slide,
        [(mL["col_xs"][0], mL["y_circ"] - mL["r_circ"]), (mL["col_xs"][0], y_hw), (bx_c, y_hw), (bx_c, sel_pill_y0 - 0.015)],
        color=DAN_WK_SEL, width_pt=2.35, arrow_end="triangle", arrow_size="med",
    )
    add_box(
        slide, bx0, sel_pill_y0, bw_c, sel_pill_h,
        [[("w", "bi", DAN_WK_SEL, 11.4), ("t", "b_", DAN_WK_SEL, 11.4), ("(1)", "b^", DAN_WK_SEL, 11.4), (" = W", "bsans", DAN_WK_SEL, 11.4), ("2", "b_", DAN_WK_SEL, 11.4)]],
        fill=DAN_PEACH, border=DAN_WK_SEL, border_pt=1.65, rounded=True, round_radius=0.28,
        margin_lr=0.0, margin_tb=0.0,
    )

    # =========================================================================
    # BALANCED 4-CARD CENTER CAUSAL STATE-UPDATE BRIDGE (bw_c = 1.48 in, ZERO DUPLICATE TEXT!)
    # =========================================================================
    # Card 0: State Update (r = 1 -> 2) aligned flush with Masked Softmax [2.04 .. 2.40]
    c0_top, h_c0 = 1.98, 0.44
    add_line(slide, bx_c, sel_pill_y0 + sel_pill_h, bx_c, c0_top - 0.02, color=DAN_WK_SEL, width_pt=2.35, arrow_end="triangle", arrow_size="med")
    add_box(
        slide, bx0, c0_top, bw_c, h_c0,
        [[("State Transition", "bsans", BLACK, 10.8)], [("substep r = 1 \u2192 2", "sans", PALETTE["ink_soft"], 10.0)]],
        fill=rgb("#FFF5EB"), border=BLACK, border_pt=1.45, rounded=True, round_radius=0.18,
        line_spacing=1.02, margin_lr=0.01, margin_tb=0.01,
    )

    # Card 1: 1. Mask Update (vertically centered on Draw 2 `used` pill at y = 2.74!) [2.51 .. 2.97]
    c1_h   = 0.48
    c1_top = mR["y_mk"] + mR["h_mk"] / 2.0 - c1_h / 2.0    # 2.74 - 0.24 = 2.50
    add_line(slide, bx_c, c0_top + h_c0, bx_c, c1_top - 0.01, color=BLACK, width_pt=1.85, arrow_end="triangle", arrow_size="sm")
    add_box(
        slide, bx0, c1_top, bw_c, c1_h,
        [[("1. Mask Update", "bsans", DAN_RED, 11.2)], [("W\u2082 \u2192 used", "bsans", DAN_RED, 11.2)]],
        fill=DAN_RED_BG, border=DAN_RED, border_pt=1.65, rounded=True, round_radius=0.18,
        line_spacing=1.02, margin_lr=0.01, margin_tb=0.01,
    )
    add_line(
        slide, bx0 + bw_c, mR["y_mk"] + mR["h_mk"] / 2.0, mR["col_xs"][0] - 0.355, mR["y_mk"] + mR["h_mk"] / 2.0,
        color=DAN_RED, width_pt=2.3, arrow_end="triangle", arrow_size="med",
    )

    # Card 2: 2. Partial-Team Pool [3.16 .. 3.84]
    c2_top = c1_top + c1_h + 0.18
    c2_h   = 0.68
    add_line(slide, bx_c, c1_top + c1_h, bx_c, c2_top - 0.01, color=BLACK, width_pt=1.85, arrow_end="triangle", arrow_size="sm")
    add_box(
        slide, bx0, c2_top, bw_c, c2_h,
        [
            [("2. Team Pooling", "bsans", BLACK, 11.2)],
            [("h", "bi", BLACK, 11.2), ("team", "_", BLACK, 11.2), (" \u2190 {W\u2082}", "bsans", BLACK, 11.0)],
            [("Sum \u00b7 Mean \u00b7 Max", "sans", PALETTE["ink_soft"], 10.0)],
        ],
        fill=DAN_LAV, border=BLACK, border_pt=1.45, rounded=True, round_radius=0.18,
        line_spacing=1.05, margin_lr=0.01, margin_tb=0.01,
    )

    # Card 3: 3. Dynamic State (spans Draw 2 swatches mid y = 4.455 cleanly!) [4.02 .. 4.72]
    c3_top = c2_top + c2_h + 0.18
    c3_h   = 0.70
    add_line(slide, bx_c, c2_top + c2_h, bx_c, c3_top - 0.01, color=BLACK, width_pt=1.85, arrow_end="triangle", arrow_size="sm")
    add_box(
        slide, bx0, c3_top, bw_c, c3_h,
        [
            [("3. Dynamic State", "bsans", BLACK, 11.0)],
            [("Recompute EFT", "sans", PALETTE["ink_soft"], 10.2)],
            [("& skill scarcity", "sans", PALETTE["ink_soft"], 10.2)],
        ],
        fill=DAN_SKY, border=BLACK, border_pt=1.5, rounded=True, round_radius=0.18,
        line_spacing=1.05, margin_lr=0.01, margin_tb=0.01,
    )

    y_sw_mid = mR["y_sw"] + 0.105
    add_line(
        slide, bx0 + bw_c, y_sw_mid, mR["col_xs"][0] - 0.265, y_sw_mid,
        color=BLACK, width_pt=2.15, dash="dash", arrow_end="triangle", arrow_size="med",
    )

    y_q_mid = mR["y_q"] + mR["h_q"] / 2.0
    add_polyline(
        slide,
        [(bx_c, c3_top + c3_h), (bx_c, y_q_mid), (mR["in_x"] - 0.02, y_q_mid)],
        color=DAN_WK_SEL, width_pt=2.3, arrow_end="triangle", arrow_size="med",
    )

    add_line(
        slide, mL["in_x"] + mL["in_w"], y_q_mid, bx_c - 0.03, y_q_mid,
        color=UNROLL_NAVY, width_pt=2.0, dash="dash", arrow_end="triangle", arrow_size="sm",
    )
    add_textbox(
        slide, xL + pw + 0.02, y_q_mid - 0.25, (bx_c - (xL + pw)) - 0.02, 0.23,
        [[("context", "bisans", UNROLL_NAVY, 10.4)]],
        font_name=FONT_SANS, size_pt=10.4, align=PP_ALIGN.CENTER, margin_lr=0.0, margin_tb=0.0,
    )

    # =========================================================================
    # BOTTOM INLINE SUBSTEP CAPTION & SHARED-POINTER UNROLLING ROW [y = 5.96 .. 6.32]
    # Generous symmetric dashed navy stems flanking Shared Worker Pointer pill!
    # =========================================================================
    y_bot_row = 5.98
    y_bot_mid = y_bot_row + 0.165
    add_textbox(
        slide, xL + 0.06, y_bot_row, 2.86, 0.32,
        [[("(a) Substep r = 1:  Select W\u2082", "b")]],
        font_name=FONT_SERIF, size_pt=13.8, color=BLACK, align=PP_ALIGN.CENTER,
        margin_lr=0.0, margin_tb=0.0, word_wrap=False,
    )
    add_textbox(
        slide, xR + 0.76, y_bot_row, 2.86, 0.32,
        [[("(b) Substep r = 2:  Select W\u2085", "b")]],
        font_name=FONT_SERIF, size_pt=13.8, color=DAN_WK_SEL, align=PP_ALIGN.CENTER,
        margin_lr=0.0, margin_tb=0.0, word_wrap=False,
    )

    pill_w = 2.54
    pill_x0 = bx_c - pill_w / 2.0   # 5.34 .. 7.88
    add_line(slide, 4.86, y_bot_mid, pill_x0 - 0.02, y_bot_mid, color=UNROLL_NAVY, width_pt=2.15, dash="dash")
    add_line(slide, pill_x0 + pill_w + 0.02, y_bot_mid, 8.38, y_bot_mid, color=UNROLL_NAVY, width_pt=2.15, dash="dash", arrow_end="triangle", arrow_size="med")
    add_box(
        slide, pill_x0, y_bot_row + 0.015, pill_w, 0.30,
        [[("Shared Worker Pointer  (r = 1 \u2192 2)", "bsans", UNROLL_NAVY, 10.4)]],
        fill=rgb("#EEF4FA"), border=UNROLL_NAVY, border_pt=1.5, rounded=True, round_radius=0.30,
        margin_lr=0.01, margin_tb=0.0,
    )

    set_notes(
        slide,
        "Module B8. Autoregressive Worker Pointer (main_tsmc.tex Section IV-C & Appendix A: Sum+Mean+Max partial-team pooling, dynamic EFT, and marginal skill scarcity). Refined v8 eliminates empty upper headroom by relocating the Substep r=1 / Shared Worker Pointer / Substep r=2 caption row cleanly beneath the dotted x q frame, placing the amber selected-worker badge w_t^(1) = W_2 directly in the center pocket at circle level between Substeps 1 and 2, and aligning every term ('Substep r', 'Masked Softmax', 'Partial-Team Pool', 'Dynamic EFT & scarcity') with main_tsmc.tex.\n"
        "Layout reference: Runqing Wang et al., Flexible Job Shop Scheduling via Dual Attention Network-Based Reinforcement Learning, IEEE TNNLS 2024, Fig. 3.",
    )


def main():
    parser = argparse.ArgumentParser(description="Build Slide 13 (B8) in refined_v8.")
    parser.add_argument("--pptx", default=DEFAULT_PPTX_PATH, help="Target PPTX path.")
    args = parser.parse_args()
    prs = Presentation(args.pptx)
    build_b8_v8(prs.slides[12])  # Slide 13: B8
    prs.save(args.pptx)
    print("Updated Slide 13 (B8) in", args.pptx)


if __name__ == "__main__":
    main()
