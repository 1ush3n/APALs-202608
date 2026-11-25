"""Build Slide 1: New Fig. 1 (A2 + A5 integrated) for IEEE TSMC double-column slot.

Left (a): Streamlined 5-Pulse Pulsating Cyclic Unfolding (Stations S1..S5, Prop. 1,
          verified Pulse 5 concurrency snapshot at t = 0.30 h).
Right (b): Real-data 283-operation material delay & stability-aware repair
           (all 22 delayed operations, frozen prefix t <= tau^r = 65.83 h,
            direct delay Delta_i vs propagation resource wait r_i -> B_i^r).

Zero review chrome, zero .csv filenames, 100% paper-faithful IEEE TSMC notation.
"""
from __future__ import annotations

import os
import sys
import pandas as pd
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
sys.path.insert(0, SCRIPT_DIR)

from pptx_helpers import (
    BLACK,
    FONT_SANS,
    FONT_SERIF,
    PALETTE,
    add_box,
    add_circle,
    add_line,
    add_polyline,
    add_textbox,
    clear_slide,
    rgb,
    set_notes,
)

BASE_CSV = r"D:\APAL-Dynamic-v4\data\r5_task_delay_v1\baselines\real\real_283_schedule.csv"
SCEN_CSV = r"D:\APAL-Dynamic-v4\data\r5_task_delay_v1\scenarios\real\real_283_scenarios.csv"
REP_CSV  = r"D:\APAL-Dynamic-v4\data\r5_task_delay_v1\real_283_medium_early_repaired_schedule.csv"

PANEL_BDR = rgb("#28445E")
PANEL_BG  = rgb("#FCFDFE")


def build_fig1_a2_a5(slide):
    """Build integrated Fig. 1 (A2 + A5), aspect ratio ~2.28:1."""
    clear_slide(slide)

    y_top, pan_h = 0.86, 5.72   # [0.86 .. 6.58]
    y_bot = y_top + pan_h

    # =========================================================================
    # PANEL (a): FIVE-STATION PULSATING CYCLIC UNFOLDING (x = 0.22 .. 6.32)
    # =========================================================================
    ax0, aw = 0.22, 6.10
    add_box(slide, ax0, y_top, aw, pan_h, fill=PANEL_BG, border=PANEL_BDR, border_pt=1.55, rounded=True, round_radius=0.025)
    add_textbox(
        slide, ax0 + 0.08, y_top + 0.06, aw - 0.16, 0.30,
        [[("(a) Five-Station Pulsating Cyclic Unfolding (Prop. 1)", "bsans")]],
        font_name=FONT_SANS, size_pt=12.6, color=PALETTE["accent_navy"], align=PP_ALIGN.CENTER, margin_lr=0.0, word_wrap=False,
    )
    add_line(slide, ax0 + 0.10, y_top + 0.39, ax0 + aw - 0.10, y_top + 0.39, color=rgb("#D5DFE8"), width_pt=0.85)

    # 5 Pulse columns: Pulses 1..4 = 0.85 in each, Pulse 5 = 1.24 in (total 4.64 in)
    col_x0 = ax0 + 1.36    # 1.58
    col_ws = [0.85, 0.85, 0.85, 0.85, 1.24]
    col_xs = []
    cur_x = col_x0
    for cw_i in col_ws:
        col_xs.append(cur_x)
        cur_x += cw_i
    col_x_end = cur_x      # 6.22

    # Pulse 5 highlight frame (drawn BEFORE bars; bottom aligns cleanly with diagonal pill at 5.74)
    p5_x = col_xs[4]
    p5_w = col_ws[4]
    add_box(
        slide, p5_x - 0.02, 1.32, p5_w + 0.04, 4.42,
        fill=rgb("#EEF7F5"), border=PALETTE["st_dark"], border_pt=1.4, dash="dash", rounded=True, round_radius=0.04,
    )
    add_box(
        slide, p5_x + 0.03, 1.35, p5_w - 0.06, 0.22,
        [[("t = 0.30 h snapshot", "bsans")]],
        fill=PALETTE["st_dark"], border=None, rounded=True, round_radius=0.24,
        color=PALETTE["white"], size_pt=8.8, margin_lr=0.0, margin_tb=0.0,
    )

    # Vertical dashed grid dividers for Pulses 1..5 (end cleanly at y = 5.30, BEFORE Cycle span H bracket!)
    add_line(slide, col_x0, 1.60, col_x0, 5.30, color=PALETTE["accent_navy"], width_pt=1.25, dash="dash")
    for p_idx in range(5):
        px = col_xs[p_idx]
        cw_i = col_ws[p_idx]
        if p_idx > 0:
            add_line(slide, px, 1.60, px, 5.30, color=rgb("#D7E0E8"), width_pt=0.75, dash="dash")
        add_textbox(
            slide, px, 1.59, cw_i, 0.23,
            [[(f"Pulse {p_idx + 1}", "bsans")]],
            font_name=FONT_SANS, size_pt=10.2,
            color=PALETTE["st_dark"] if p_idx == 4 else PALETTE["accent_navy"],
            align=PP_ALIGN.CENTER, margin_lr=0.0, margin_tb=0.0, word_wrap=False,
        )
    add_line(slide, col_x_end, 1.60, col_x_end, 5.30, color=rgb("#D7E0E8"), width_pt=0.75, dash="dash")

    ac_colors = {
        0:  (PALETTE["accent_navy"], PALETTE["white"], True),
        -1: (rgb("#ECD3BC"), PALETTE["ink_soft"], False),
        1:  (rgb("#C7DFD7"), PALETTE["ink_soft"], False),
        2:  (rgb("#E7D6E8"), PALETTE["ink_soft"], False),
    }
    default_ac = (rgb("#E8ECEF"), PALETTE["muted"], False)

    p5_ops = [
        "O15 [W72, W12]",
        "O18 [W18, W65]",
        "O12, O29 (2 ops)",
        "O20 [W55, W6]",
        "O24, O30 (2 ops)",
    ]

    row_y0 = 1.88
    row_pitch = 0.74
    bar_h = 0.38

    for s_idx in range(5):
        ry = row_y0 + s_idx * row_pitch
        # Native aircraft assembly station illustration (1448 x 1086 -> exact 4:3)
        pic_p = os.path.join(ASSETS_DIR, f"a2_station_{s_idx + 1}.png")
        pic_w, pic_h = 0.78, 0.585
        slide.shapes.add_picture(
            pic_p,
            Inches(ax0 + 0.08),
            Inches(ry + bar_h / 2.0 - pic_h / 2.0),
            width=Inches(pic_w),
            height=Inches(pic_h),
        )
        # Station badge S1..S5
        add_textbox(
            slide, ax0 + 0.89, ry + 0.04, 0.44, 0.30,
            [[("S", "b"), (str(s_idx + 1), "b_")]],
            font_name=FONT_SERIF, size_pt=12.4, color=PALETTE["ink"], align=PP_ALIGN.CENTER, margin_lr=0.0, word_wrap=False,
        )

        for p_idx in range(5):
            delta = p_idx - s_idx
            bg, fg, is_k = ac_colors.get(delta, default_ac)
            if delta == 0:
                lbl = "Aircraft k"
            elif delta > 0:
                lbl = f"k + {delta}"
            else:
                lbl = f"k \u2212 {abs(delta)}"

            cw_i = col_ws[p_idx]
            pad = 0.04 if p_idx < 4 else 0.06
            bw_i = cw_i - 2 * pad
            bx = col_xs[p_idx] + pad
            box_sh = add_box(
                slide, bx, ry, bw_i, bar_h,
                [[(lbl, "bsans" if is_k else "sans")]],
                fill=bg, border=PALETTE["accent_navy"] if is_k else None,
                border_pt=1.0 if is_k else 0.0, rounded=True, round_radius=0.09,
                color=fg, size_pt=9.6 if is_k else 10.0, margin_lr=0.01, margin_tb=0.0,
            )
            box_sh.text_frame.word_wrap = False

            if p_idx == 4:
                add_textbox(
                    slide, col_xs[4] + 0.02, ry + bar_h + 0.02, p5_w - 0.04, 0.22,
                    [[(p5_ops[s_idx], "bsans")]],
                    font_name=FONT_SANS, size_pt=8.8, color=PALETTE["st_dark"], align=PP_ALIGN.CENTER,
                    margin_lr=0.0, margin_tb=0.0, word_wrap=False,
                )

    # Diagonal orange arrows linking Aircraft k across Stations 1 -> 5
    for s_idx in range(4):
        pad1 = 0.04
        pad2 = 0.04 if (s_idx + 1) < 4 else 0.06
        bw1 = col_ws[s_idx] - 2 * pad1
        bw2 = col_ws[s_idx + 1] - 2 * pad2
        x1 = col_xs[s_idx] + pad1 + bw1 * 0.78
        y1 = row_y0 + s_idx * row_pitch + bar_h + 0.02
        x2 = col_xs[s_idx + 1] + pad2 + bw2 * 0.16
        y2 = row_y0 + (s_idx + 1) * row_pitch - 0.02
        add_line(slide, x1, y1, x2, y2, color=PALETTE["accent_orange"], width_pt=1.8, arrow_end="triangle", arrow_size="sm")

    # Pulse 1 cycle span bracket + Diagonal tag (clean row at y = 5.36 .. 5.76)
    add_polyline(
        slide,
        [(col_xs[0], 5.40), (col_xs[0], 5.52), (col_xs[0] + col_ws[0], 5.52), (col_xs[0] + col_ws[0], 5.40)],
        color=PALETTE["accent_navy"], width_pt=1.25,
    )
    add_textbox(
        slide, col_xs[0] - 0.06, 5.54, col_ws[0] + 0.12, 0.24,
        [[("Cycle span H", "bsans")]],
        font_name=FONT_SANS, size_pt=9.4, color=PALETTE["accent_navy"], align=PP_ALIGN.CENTER, margin_lr=0.0, word_wrap=False,
    )
    add_box(
        slide, col_xs[1] + 0.08, 5.44, 2.36, 0.30,
        [[("Diagonal: Aircraft k  (S1 \u2192 S5)", "bsans")]],
        fill=PALETTE["wk_fill"], border=PALETTE["accent_orange"], border_pt=1.05,
        rounded=True, round_radius=0.22, color=PALETTE["accent_orange"], size_pt=9.2, margin_lr=0.02, margin_tb=0.0,
    )

    # Bottom Proposition 1 formula pill + Parallel column pill [y = 5.98 .. 6.42]
    add_box(
        slide, ax0 + 0.10, 5.98, 3.86, 0.44,
        [[("Window(", "bsans"), ("k, m", "bi"), (") = [(", "b"), ("k + m \u2212 1", "bi"), (")H,  (", "b"), ("k + m", "bi"), (")H)", "b")]],
        fill=rgb("#F0F4F8"), border=rgb("#9FB0C0"), border_pt=1.05, rounded=True, round_radius=0.18,
        font_name=FONT_SERIF, color=PALETTE["ink"], size_pt=11.0, margin_lr=0.02, margin_tb=0.0,
    )
    add_box(
        slide, ax0 + 4.06, 5.98, 1.94, 0.44,
        [[("Pulse col.: 5 aircraft parallel", "bsans")]],
        fill=PALETTE["st_fill"], border=PALETTE["st_dark"], border_pt=1.1, rounded=True, round_radius=0.18,
        color=PALETTE["st_dark"], size_pt=9.4, margin_lr=0.02, margin_tb=0.0,
    )

    # =========================================================================
    # PANEL (b): REAL-DATA MATERIAL DELAY & STABILITY REPAIR (x = 6.46 .. 13.11)
    # =========================================================================
    base = pd.read_csv(BASE_CSV)
    scen = pd.read_csv(SCEN_CSV)
    rep  = pd.read_csv(REP_CSV)
    h0 = float(base[base["Duration"] > 0]["End"].max())   # 292.57
    hr = float(rep[rep["Duration"] > 0]["End"].max())     # 305.85
    me = scen[scen["scenario_id"] == "medium_early"].merge(
        rep[["TaskID", "Start", "End"]], on="TaskID", suffixes=("", "_rep")
    ).sort_values("baseline_start").reset_index(drop=True)
    tau_r = float(me["reschedule_start_time"].iloc[0])    # 65.83

    bx0, bw = 6.46, 6.65
    add_box(slide, bx0, y_top, bw, pan_h, fill=PANEL_BG, border=PANEL_BDR, border_pt=1.55, rounded=True, round_radius=0.025)
    add_textbox(
        slide, bx0 + 0.08, y_top + 0.06, bw - 0.16, 0.30,
        [[("(b) Material Delay & Stability-Aware Repair (283-Op Line, 22 Delayed Ops)", "bsans")]],
        font_name=FONT_SANS, size_pt=12.4, color=PALETTE["accent_navy"], align=PP_ALIGN.CENTER, margin_lr=0.0, word_wrap=False,
    )
    add_line(slide, bx0 + 0.10, y_top + 0.39, bx0 + bw - 0.10, y_top + 0.39, color=rgb("#D5DFE8"), width_pt=0.85)

    # Compact uniformly-spaced legend strip [y = 1.30 .. 1.54]
    ly = 1.42
    add_circle(slide, bx0 + 0.22, ly, 0.058, fill=PALETTE["white"], border=PALETTE["accent_navy"], border_pt=1.5)
    add_textbox(slide, bx0 + 0.32, ly - 0.12, 1.06, 0.24, [[("Baseline B", "bsans"), ("i", "b_"), ("0", "b^")]], font_name=FONT_SANS, size_pt=9.6, color=PALETTE["accent_navy"], align=PP_ALIGN.LEFT, margin_lr=0.0, word_wrap=False)

    add_line(slide, bx0 + 1.46, ly, bx0 + 1.76, ly, color=PALETTE["accent_orange"], width_pt=2.0)
    add_textbox(slide, bx0 + 1.81, ly - 0.12, 0.86, 0.24, [[("delay \u0394", "bsans"), ("i", "b_")]], font_name=FONT_SANS, size_pt=9.6, color=PALETTE["accent_orange"], align=PP_ALIGN.LEFT, margin_lr=0.0, word_wrap=False)

    dm_leg = slide.shapes.add_shape(MSO_SHAPE.DIAMOND, Inches(bx0 + 2.74), Inches(ly - 0.065), Inches(0.13), Inches(0.13))
    dm_leg.fill.solid(); dm_leg.fill.fore_color.rgb = PALETTE["accent_orange"]; dm_leg.line.fill.background(); dm_leg.shadow.inherit = False
    add_textbox(slide, bx0 + 2.91, ly - 0.12, 0.98, 0.24, [[("Release r", "bsans"), ("i", "b_")]], font_name=FONT_SANS, size_pt=9.6, color=PALETTE["accent_orange"], align=PP_ALIGN.LEFT, margin_lr=0.0, word_wrap=False)

    add_line(slide, bx0 + 3.96, ly, bx0 + 4.26, ly, color=PALETTE["ok_dark"], width_pt=2.0)
    add_textbox(slide, bx0 + 4.31, ly - 0.12, 0.92, 0.24, [[("prop. wait", "bsans")]], font_name=FONT_SANS, size_pt=9.6, color=PALETTE["ok_dark"], align=PP_ALIGN.LEFT, margin_lr=0.0, word_wrap=False)

    add_box(slide, bx0 + 5.28, ly - 0.055, 0.11, 0.11, fill=PALETTE["ok_dark"], border=None)
    add_textbox(slide, bx0 + 5.44, ly - 0.12, 1.14, 0.24, [[("Repaired B", "bsans"), ("i", "b_"), ("r", "b^")]], font_name=FONT_SANS, size_pt=9.6, color=PALETTE["ok_dark"], align=PP_ALIGN.LEFT, margin_lr=0.0, word_wrap=False)

    # Chart geometry [cx0 .. cx1] = [7.18 .. 12.96] (5.78 in for 0..320 h)
    # Raised cy_bot to 5.50 so x-axis ticks and 'Schedule time (h)' have clean breathing space!
    cx0, cx1 = bx0 + 0.72, bx0 + bw - 0.15   # 7.18 .. 12.96
    cw = cx1 - cx0                           # 5.78 in
    t_max = 320.0
    cy_top, cy_bot = 1.82, 5.50
    x_tau = cx0 + (tau_r / t_max) * cw
    x_h0  = cx0 + (h0 / t_max) * cw

    # Shaded frozen prefix [0, tau_r]
    add_box(slide, cx0, cy_top, x_tau - cx0, cy_bot - cy_top, fill=rgb("#EDF2F6"), border=None)
    add_textbox(
        slide, cx0 + 0.03, 2.80, (x_tau - cx0) - 0.06, 1.48,
        [
            [("Frozen prefix", "bsans", PALETTE["accent_navy"], 10.2)],
            [("t \u2264 \u03c4\u02b3 = 65.83 h", "bsans", PALETTE["ink"], 9.2)],
            [("67 committed ops", "sans", PALETTE["ink_soft"], 8.8)],
            [("fixed times &", "sans", PALETTE["ink_soft"], 8.8)],
            [("worker bindings", "sans", PALETTE["ink_soft"], 8.8)],
        ],
        font_name=FONT_SANS, align=PP_ALIGN.CENTER, line_spacing=1.18, margin_lr=0.01,
    )

    # Vertical gridlines & x-axis ticks (clean separation from Schedule time label!)
    for tick in range(0, 321, 40):
        tx = cx0 + (tick / t_max) * cw
        add_line(slide, tx, cy_top, tx, cy_bot, color=rgb("#DFE6EC"), width_pt=0.75, dash="dash" if tick > 0 else None)
        add_textbox(
            slide, tx - 0.22, cy_bot + 0.03, 0.44, 0.19,
            [[(str(tick), "sans")]], font_name=FONT_SANS, size_pt=9.2, color=PALETTE["ink_soft"],
            align=PP_ALIGN.CENTER, margin_lr=0.0, margin_tb=0.0, word_wrap=False,
        )
    add_textbox(
        slide, cx0, cy_bot + 0.23, cw, 0.20,
        [[("Schedule time (h)", "bsans")]],
        font_name=FONT_SANS, size_pt=9.6, color=PALETTE["ink_soft"], align=PP_ALIGN.CENTER, margin_lr=0.0, margin_tb=0.0,
    )

    # Trigger tau^r and Baseline H^0 vertical markers
    add_line(slide, x_tau, cy_top - 0.03, x_tau, cy_bot + 0.02, color=PALETTE["accent_navy"], width_pt=1.4, dash="dash")
    add_textbox(
        slide, x_tau - 0.04, 1.56, 2.38, 0.24,
        [[("Trigger \u03c4", "bsans"), ("r", "b^"), (f" = {tau_r:.2f} h  (\u2260 release r", "bsans"), ("i", "b_"), (")", "bsans")]],
        font_name=FONT_SANS, size_pt=9.6, color=PALETTE["accent_navy"], align=PP_ALIGN.LEFT, margin_lr=0.0, word_wrap=False,
    )
    add_line(slide, x_h0, cy_top, x_h0, cy_bot + 0.02, color=PALETTE["accent_orange"], width_pt=1.15, dash="dot")
    add_textbox(
        slide, x_h0 - 1.56, 1.56, 1.54, 0.24,
        [[("Baseline H", "bsans"), ("0", "b^"), (f" = {h0:.2f} h", "bsans")]],
        font_name=FONT_SANS, size_pt=9.4, color=PALETTE["accent_orange"], align=PP_ALIGN.RIGHT, margin_lr=0.0, word_wrap=False,
    )

    # Y-axis column label
    add_textbox(
        slide, bx0 + 0.08, 1.57, 0.62, 0.22,
        [[("Op ID", "bsans")]],
        font_name=FONT_SANS, size_pt=9.2, color=PALETTE["accent_navy"], align=PP_ALIGN.RIGHT, margin_lr=0.0, word_wrap=False,
    )

    # Draw all 22 delayed operations
    n_rows = len(me)
    r_pitch = (cy_bot - cy_top - 0.14) / n_rows   # ~0.161 in
    for idx, r in me.iterrows():
        yc = cy_top + 0.10 + idx * r_pitch
        tid = int(r["TaskID"])
        t_b0  = float(r["baseline_start"])
        t_rel = float(r["release_time"])
        t_rep = float(r["Start"])

        if idx % 2 == 0:
            add_box(slide, bx0 + 0.10, yc - r_pitch * 0.46, 0.58, r_pitch * 0.92, fill=rgb("#F0F4F8"), border=None)
        add_textbox(
            slide, bx0 + 0.08, yc - 0.085, 0.60, 0.17,
            [[(f"O{tid}", "bsans")]],
            font_name=FONT_SANS, size_pt=8.8, color=PALETTE["accent_navy"], align=PP_ALIGN.RIGHT,
            margin_lr=0.02, margin_tb=0.0, word_wrap=False,
        )

        x_b0  = cx0 + (t_b0 / t_max) * cw
        x_rel = cx0 + (t_rel / t_max) * cw
        x_rep = cx0 + (t_rep / t_max) * cw

        if x_rel > x_b0 + 0.008:
            add_line(slide, x_b0, yc, x_rel, yc, color=PALETTE["accent_orange"], width_pt=1.55)
        if x_rep > x_rel + 0.012:
            add_line(slide, x_rel, yc, x_rep, yc, color=PALETTE["ok_dark"], width_pt=1.75)

        add_circle(slide, x_b0, yc, 0.043, fill=PALETTE["white"], border=PALETTE["accent_navy"], border_pt=1.35)
        sq = 0.080
        add_box(slide, x_rep - sq / 2, yc - sq / 2, sq, sq, fill=PALETTE["ok_dark"], border=None)
        dm_s = 0.096
        dsh = slide.shapes.add_shape(MSO_SHAPE.DIAMOND, Inches(x_rel - dm_s / 2), Inches(yc - dm_s / 2), Inches(dm_s), Inches(dm_s))
        dsh.fill.solid(); dsh.fill.fore_color.rgb = PALETTE["accent_orange"]; dsh.line.color.rgb = PALETTE["white"]; dsh.line.width = Inches(0.006); dsh.shadow.inherit = False

    # Callout on O111 (row idx=1) ending cleanly LEFT of vertical Baseline H^0 dotted line (x_h0 = 12.46)!
    y_o111 = cy_top + 0.10 + 1 * r_pitch
    x_o111_rep = cx0 + (float(me.loc[1, "Start"]) / t_max) * cw
    cb_x = 9.70
    cb_w = 2.66   # ends at 12.36 -> 0.10 in clear left of H^0 line!
    add_line(slide, x_o111_rep + 0.07, y_o111, cb_x - 0.02, y_o111, color=PALETTE["ok_dark"], width_pt=1.2, dash="dash")
    add_box(
        slide, cb_x, 1.90, cb_w, 0.78,
        [
            [("Propagation beyond r", "bsans", PALETTE["ink"], 9.5), ("i", "b_", PALETTE["ink"], 9.5), (" (O111):", "bsans", PALETTE["ink"], 9.5)],
            [("\u2022 Direct delay \u0394", "bsans", PALETTE["accent_orange"], 9.0), ("i", "b_", PALETTE["accent_orange"], 9.0), (" = +18.3 h (orange)", "sans", PALETTE["ink_soft"], 9.0)],
            [("\u2022 Resource wait = +26.0 h \u2192 B", "bsans", PALETTE["ok_dark"], 9.0), ("i", "b_", PALETTE["ok_dark"], 9.0), ("r", "b^", PALETTE["ok_dark"], 9.0), (" > r", "bsans", PALETTE["ok_dark"], 9.0), ("i", "b_", PALETTE["ok_dark"], 9.0)],
        ],
        fill=rgb("#F5FBF8"), border=PALETTE["ok_dark"], border_pt=1.05, rounded=True, round_radius=0.10,
        align=PP_ALIGN.LEFT, margin_lr=0.06, margin_tb=0.02, line_spacing=1.12,
    )

    # Bottom summary pill (exact vertical alignment with Panel (a) bottom pills!)
    add_box(
        slide, bx0 + 0.12, 5.98, bw - 0.24, 0.44,
        [[
            ("67 frozen ops  \u00b7  216 reschedulable (22 delayed)  \u00b7  ", "bsans", PALETTE["accent_navy"], 9.8),
            ("H", "b", PALETTE["accent_orange"], 10.6), ("0", "b^", PALETTE["accent_orange"], 10.6),
            (f" = {h0:.2f} h \u2192 ", "bsans", PALETTE["ink"], 9.8),
            ("H", "b", PALETTE["ok_dark"], 10.6), ("r", "b^", PALETTE["ok_dark"], 10.6),
            (f" = {hr:.2f} h", "bsans", PALETTE["ok_dark"], 9.8),
        ]],
        fill=rgb("#EFF4F9"), border=rgb("#9FB0C0"), border_pt=1.05, rounded=True, round_radius=0.18,
        font_name=FONT_SERIF, margin_lr=0.02, margin_tb=0.0,
    )

    set_notes(
        slide,
        "Integrated Fig. 1 (A2 + A5). Panel (a) illustrates five-station pulsating cyclic unfolding (Proposition 1): tracking Aircraft k along the navy diagonal through Stations S1-S5 over Pulses 1-5 in unfolded window [(k+m-1)H, (k+m)H), while reading down Pulse 5 reveals five distinct aircraft processed concurrently with verified operations and multi-skilled worker teams active at offset t = 0.30 h. Panel (b) depicts a real-data material delay disruption on the 283-operation factory line triggered at tau^r = 65.83 h: 67 operations in the frozen prefix (t <= tau^r) retain baseline schedules and worker bindings, while 216 remaining operations are reschedulable (22 directly delayed). For each delayed operation, markers trace baseline start B_i^0 (open circle), direct material delay Delta_i (orange bar to diamond r_i), and downstream resource propagation wait (green bar to square B_i^r), with cycle span shifting from H^0 = 292.57 h to H^r = 305.85 h.",
    )
