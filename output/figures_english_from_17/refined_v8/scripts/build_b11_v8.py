"""Build Slide 16 (B11) in refined_v8: 100% paper-faithful PPO Learning & GAE (main_tsmc.tex Section IV-D & Appendix B).

v8 terminology, notation & scientific rigor fixes:
1. Removed internal debug leak '(ckpt: per-head clip)' from Clipped PPO Objective box!
2. Resolved symbol collision with Eq. (1) team coordination discount factor rho=0.95:
   - Replaced ratio symbol 'rho_t' with exact paper symbol 'r_t^PPO' (main_tsmc.tex line 365-368!).
   - Replaced 'theta_old' with exact paper symbol 'theta^-' (main_tsmc.tex line 365!).
   - Replaced 'L_pi(theta)' with exact paper symbol 'L_policy(theta)' (main_tsmc.tex line 368!).
   - Replaced 'TD Residual' with exact paper term 'TD Error' (main_tsmc.tex line 361!).
3. Fixed wrong hardcoded table number: 'Table VII Config:' (actually Table XII in main_tsmc.tex!)
   -> table-number-independent 'PPO Hyperparameters:'.
4. Elevated legend & minor labels (>= 9.2pt everywhere) while preserving zero line crossings.
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

TD_BLUE      = rgb("#3A7EC2")
TD_BLUE_SOFT = rgb("#E7F1FA")
TD_BLUE_DK   = rgb("#1D4E7B")
TD_AMBER_BG  = rgb("#FCE9D4")
TD_AMBER_DK  = rgb("#A85516")
TD_SAND      = rgb("#FDF1C8")
TD_GREEN_BG  = rgb("#DFF2E7")
TD_GREEN_DK  = rgb("#1B7344")
TD_PURP_BG   = rgb("#EFE7F7")
TD_PURP_DK   = rgb("#633789")
TD_MUTED_BDR = rgb("#93A4B5")
TD_MUTED_TXT = rgb("#5F7182")


def build_b11_v8(slide):
    """Slide 16 · B11 publication-ready PPO Learning & GAE figure (100% aligned with main_tsmc.tex Section IV-D)."""
    clear_slide(slide)

    reg_y, reg_h = 0.62, 5.48

    ax, aw = 0.40, 4.24
    bx, bw = 5.20, 3.52
    cx, cw = 9.28, 3.66

    add_box(slide, ax, reg_y, aw, reg_h, fill=rgb("#FCFDFF"), border=BLACK, border_pt=1.75, rounded=True, round_radius=0.04)
    add_box(slide, bx, reg_y, bw, reg_h, fill=PALETTE["white"], border=BLACK, border_pt=1.75, rounded=True, round_radius=0.04)
    add_box(slide, cx, reg_y, cw, reg_h, fill=rgb("#FCFDFF"), border=BLACK, border_pt=1.75, rounded=True, round_radius=0.04)

    add_textbox(slide, ax + 0.10, reg_y + 0.08, aw - 0.20, 0.30, [[("(a) Rollout Buffer Trajectory", "b")]], font_name=FONT_SERIF, size_pt=12.8, color=BLACK, align=PP_ALIGN.CENTER)
    add_textbox(slide, bx + 0.10, reg_y + 0.08, bw - 0.20, 0.30, [[("(b) Action Replay & GAE", "b")]], font_name=FONT_SERIF, size_pt=12.8, color=BLACK, align=PP_ALIGN.CENTER)
    add_textbox(slide, cx + 0.10, reg_y + 0.08, cw - 0.20, 0.30, [[("(c) PPO Objectives & Updates", "b")]], font_name=FONT_SERIF, size_pt=12.8, color=BLACK, align=PP_ALIGN.CENTER)

    # =========================================================================
    # REGION (a): STORED ROLLOUT TRAJECTORY + EMBEDDED SYMBOL LEGEND
    # =========================================================================
    xs_t = [1.14, 2.52, 3.90]
    y_a  = 2.12
    y_s  = 3.04
    r_s  = 0.255
    y_r  = 3.96
    y_v  = 4.46
    r_v  = 0.195

    add_box(slide, 1.88, 1.50, 1.28, 3.24, fill=rgb("#EFF6FC"), border=rgb("#A4C2DE"), border_pt=1.1, rounded=True, round_radius=0.06)

    add_line(slide, xs_t[0] + r_s + 0.02, y_s, xs_t[1] - r_s - 0.02, y_s, color=TD_BLUE_DK, width_pt=1.55, arrow_end="triangle", arrow_size="sm")
    add_line(slide, xs_t[1] + r_s + 0.02, y_s, xs_t[2] - r_s - 0.02, y_s, color=TD_BLUE_DK, width_pt=1.55, arrow_end="triangle", arrow_size="sm")
    add_textbox(slide, 1.54, y_s - 0.26, 0.58, 0.22, [[("env", "isans")]], font_name=FONT_SANS, size_pt=9.4, color=TD_MUTED_TXT, align=PP_ALIGN.CENTER)
    add_textbox(slide, 2.92, y_s - 0.26, 0.58, 0.22, [[("env", "isans")]], font_name=FONT_SANS, size_pt=9.4, color=TD_BLUE_DK, align=PP_ALIGN.CENTER)

    aw_c, ah_c = 0.58, 0.32
    for idx, top_lbl, sub_lbl, is_act in [(0, "t \u2212 1", "t\u22121", False), (1, "step t", "t", True), (2, "t + 1", "t+1", False)]:
        x_c = xs_t[idx]
        add_textbox(slide, x_c - 0.44, 1.54, 0.88, 0.24, [[(top_lbl, "b" if is_act else "")]], font_name=FONT_SERIF, size_pt=12.0 if is_act else 11.2, color=TD_BLUE_DK if is_act else TD_MUTED_TXT, align=PP_ALIGN.CENTER)

        add_box(
            slide, x_c - aw_c / 2, y_a - ah_c / 2, aw_c, ah_c,
            [[("\u00e3", "bi"), (sub_lbl, "b_")]],
            fill=TD_AMBER_BG if is_act else rgb("#FDF7EF"),
            border=BLACK if is_act else rgb("#C9A682"), border_pt=1.5 if is_act else 1.0,
            rounded=True, round_radius=0.22,
            color=TD_AMBER_DK if is_act else rgb("#9B724E"), size_pt=13.0 if is_act else 11.6,
            margin_lr=0.0, margin_tb=0.0,
        )
        add_line(slide, x_c, y_s - r_s - 0.01, x_c, y_a + ah_c / 2 + 0.02, color=TD_AMBER_DK if is_act else rgb("#C9A682"), width_pt=1.55 if is_act else 1.05, arrow_end="triangle", arrow_size="sm")
        if is_act:
            # Exact paper symbol pi_{theta^-}!
            add_textbox(slide, x_c + 0.06, 2.42, 0.62, 0.24, [[("\u03c0", "bi"), ("\u03b8\u207b", "b_")]], font_name=FONT_SERIF, size_pt=10.6, color=TD_AMBER_DK, align=PP_ALIGN.LEFT)

        add_circle(
            slide, x_c, y_s, r_s,
            [[("s", "bi"), (sub_lbl, "b_")]],
            fill=TD_BLUE if is_act else TD_BLUE_SOFT,
            border=BLACK if is_act else TD_MUTED_BDR, border_pt=1.75 if is_act else 1.15,
            color=PALETTE["white"] if is_act else TD_BLUE_DK, size_pt=15.0 if is_act else 13.0,
        )

        vx, rx_c = x_c - 0.34, x_c + 0.34
        v_active = is_act or (idx == 2)
        add_polyline(slide, [(x_c - 0.10, y_s + r_s - 0.02), (vx, y_s + r_s + 0.22), (vx, y_v - r_v - 0.02)], color=TD_PURP_DK if v_active else rgb("#B59EC9"), width_pt=1.45 if v_active else 1.0, arrow_end="triangle", arrow_size="sm")
        add_polyline(slide, [(x_c + 0.10, y_s + r_s - 0.02), (rx_c, y_s + r_s + 0.22), (rx_c, y_r - 0.155)], color=TD_GREEN_DK if is_act else rgb("#8DB8A0"), width_pt=1.45 if is_act else 1.0, arrow_end="triangle", arrow_size="sm")

        add_box(slide, rx_c - 0.25, y_r - 0.145, 0.50, 0.29, [[("r", "bi"), (sub_lbl, "b_")]], fill=TD_GREEN_BG if is_act else rgb("#F2FAF5"), border=BLACK if is_act else rgb("#8DB8A0"), border_pt=1.45 if is_act else 1.0, rounded=True, round_radius=0.22, color=TD_GREEN_DK if is_act else rgb("#6B977E"), size_pt=12.6 if is_act else 11.0, margin_lr=0.0, margin_tb=0.0)
        add_circle(slide, vx, y_v, r_v, [[("V", "b"), (sub_lbl, "b_")]], fill=TD_PURP_BG if v_active else rgb("#F7F3FB"), border=BLACK if v_active else rgb("#B59EC9"), border_pt=1.5 if v_active else 1.0, color=TD_PURP_DK if v_active else rgb("#8D73A5"), size_pt=12.0 if is_act else (10.8 if idx == 2 else 10.4))

    # Embedded Symbol Legend (enlarged >= 9.2pt!)
    add_line(slide, ax + 0.14, 5.28, ax + aw - 0.14, 5.28, color=PALETTE["line_gray"], width_pt=1.0, dash="dash")
    y_lg1 = 5.48
    add_circle(slide, ax + 0.24, y_lg1, 0.11, [[("s", "bi")]], fill=TD_BLUE, border=BLACK, border_pt=1.05, color=PALETTE["white"], size_pt=9.2)
    add_textbox(slide, ax + 0.38, y_lg1 - 0.12, 0.74, 0.24, [[("State G", "bsans"), ("t", "b_")]], font_name=FONT_SANS, size_pt=9.2, color=BLACK, align=PP_ALIGN.LEFT, margin_lr=0.0)

    add_box(slide, ax + 1.15, y_lg1 - 0.11, 0.28, 0.22, [[("\u00e3", "bi")]], fill=TD_AMBER_BG, border=BLACK, border_pt=1.05, rounded=True, round_radius=0.22, color=TD_AMBER_DK, size_pt=9.2, margin_lr=0.0, margin_tb=0.0)
    add_textbox(slide, ax + 1.46, y_lg1 - 0.12, 0.66, 0.24, [[("Action", "bsans")]], font_name=FONT_SANS, size_pt=9.2, color=BLACK, align=PP_ALIGN.LEFT, margin_lr=0.0)

    add_box(slide, ax + 2.13, y_lg1 - 0.11, 0.26, 0.22, [[("r", "bi")]], fill=TD_GREEN_BG, border=BLACK, border_pt=1.05, rounded=True, round_radius=0.22, color=TD_GREEN_DK, size_pt=9.2, margin_lr=0.0, margin_tb=0.0)
    add_textbox(slide, ax + 2.42, y_lg1 - 0.12, 0.70, 0.24, [[("Reward", "bsans")]], font_name=FONT_SANS, size_pt=9.2, color=BLACK, align=PP_ALIGN.LEFT, margin_lr=0.0)

    add_circle(slide, ax + 3.24, y_lg1, 0.11, [[("V", "b")]], fill=TD_PURP_BG, border=BLACK, border_pt=1.05, color=TD_PURP_DK, size_pt=9.2)
    add_textbox(slide, ax + 3.38, y_lg1 - 0.12, 0.80, 0.24, [[("Value V", "bsans"), ("\u03c6", "b_")]], font_name=FONT_SANS, size_pt=9.2, color=BLACK, align=PP_ALIGN.LEFT, margin_lr=0.0)

    add_textbox(
        slide, ax + 0.08, 5.72, aw - 0.16, 0.28,
        [[("Continuation: ", "bsans", BLACK, 9.2), ("b", "bi", BLACK, 10.0), ("t", "b_", BLACK, 10.0), (" = (1\u2212d", "serif", BLACK, 9.8), ("t", "_", BLACK, 9.8), (")(1\u2212u", "serif", BLACK, 9.8), ("t", "_", BLACK, 9.8), (")", "serif", BLACK, 9.8), ("  (0 at end)", "sans", PALETTE["ink_soft"], 9.2)]],
        align=PP_ALIGN.CENTER, margin_lr=0.0,
    )

    # =========================================================================
    # REGION (b): ACTION REPLAY & GAE ESTIMATION (exact Section IV-D symbols!)
    # =========================================================================
    c1x, c1w = bx + 0.18, bw - 0.36
    c1cx = c1x + c1w / 2.0

    y_rep, h_rep = 1.14, 0.66
    add_box(
        slide, c1x, y_rep, c1w, h_rep,
        [
            [("Replay ordered ", "bsans", BLACK, 10.6), ("\u00e3", "bi", TD_AMBER_DK, 11.4), ("t", "b_", TD_AMBER_DK, 11.4), (" under ", "bsans", BLACK, 10.6), ("\u03c0", "bi", BLACK, 11.4), ("\u03b8", "b_", BLACK, 11.4)],
            [("eval (i", "sans", PALETTE["ink_soft"], 9.4), ("t", "_", PALETTE["ink_soft"], 9.4), (", m", "sans", PALETTE["ink_soft"], 9.4), ("t", "_", PALETTE["ink_soft"], 9.4), (", w", "sans", PALETTE["ink_soft"], 9.4), ("t", "_", PALETTE["ink_soft"], 9.4), (") with stored masks (K=2)", "sans", PALETTE["ink_soft"], 9.4)],
        ],
        fill=TD_SAND, border=BLACK, border_pt=1.45, rounded=True, round_radius=0.16, line_spacing=1.08, margin_lr=0.04, margin_tb=0.02,
    )

    # Exact Section IV-D ratio symbol: r_t^PPO(theta) = exp(l_{t,theta} - l_{t,theta^-})!
    y_ll, h_ll = 2.04, 0.70
    add_line(slide, c1cx, y_rep + h_rep, c1cx, y_ll - 0.01, color=TD_AMBER_DK, width_pt=1.5, arrow_end="triangle", arrow_size="sm")
    add_box(
        slide, c1x, y_ll, c1w, h_ll,
        [
            [("\u2113", "bi", TD_AMBER_DK, 11.2), ("t,\u03b8", "b_", TD_AMBER_DK, 11.2), (" = \u2113", "b", TD_AMBER_DK, 11.2), ("O", "b^", TD_AMBER_DK, 11.2), (" + \u2113", "b", TD_AMBER_DK, 11.2), ("M", "b^", TD_AMBER_DK, 11.2), (" + \u03a3", "b", TD_AMBER_DK, 11.2), ("r=1", "b_", TD_AMBER_DK, 11.2), ("q", "b^", TD_AMBER_DK, 11.2), (" \u2113", "b", TD_AMBER_DK, 11.2), ("W,r", "b^", TD_AMBER_DK, 11.2)],
            [("r", "bi", BLACK, 10.6), ("t", "b_", BLACK, 10.6), ("PPO", "b^", BLACK, 10.4), ("(\u03b8) = exp(\u2113", "serif", BLACK, 10.2), ("t,\u03b8", "_", BLACK, 10.2), (" \u2212 \u2113", "serif", BLACK, 10.2), ("t,\u03b8\u207b", "_", BLACK, 10.2), (")", "serif", BLACK, 10.2)],
        ],
        fill=TD_AMBER_BG, border=BLACK, border_pt=1.45, rounded=True, round_radius=0.16, line_spacing=1.08, margin_lr=0.04, margin_tb=0.02,
    )

    add_line(slide, bx + 0.14, 2.94, bx + bw - 0.14, 2.94, color=PALETTE["line_gray"], width_pt=1.05, dash="dash")

    # Exact Section IV-D term: TD Error delta_t
    y_td, h_td = 3.14, 0.68
    add_box(
        slide, c1x, y_td, c1w, h_td,
        [
            [("TD Error", "bsans", TD_PURP_DK, 10.8), ("   (\u03b3 = 0.999)", "sans", PALETTE["ink_soft"], 9.4)],
            [("\u03b4", "bi", BLACK, 11.8), ("t", "b_", BLACK, 11.8), (" = r", "b", BLACK, 11.6), ("t", "b_", BLACK, 11.6), (" + \u03b3 b", "b", BLACK, 11.6), ("t", "b_", BLACK, 11.6), (" V", "b", BLACK, 11.6), ("t+1", "b_", BLACK, 11.6), (" \u2212 V", "b", BLACK, 11.6), ("t", "b_", BLACK, 11.6)],
        ],
        fill=TD_PURP_BG, border=BLACK, border_pt=1.45, rounded=True, round_radius=0.16, line_spacing=1.08, margin_lr=0.04, margin_tb=0.02,
    )

    y_gae, h_gae = 4.08, 0.68
    add_line(slide, c1cx, y_td + h_td, c1cx, y_gae - 0.01, color=TD_PURP_DK, width_pt=1.5, arrow_end="triangle", arrow_size="sm")
    add_box(
        slide, c1x, y_gae, c1w, h_gae,
        [
            [("GAE Advantage", "bsans", TD_PURP_DK, 10.8), ("   (\u03bb = 0.9)", "sans", PALETTE["ink_soft"], 9.4)],
            [("\u00c2", "b", BLACK, 11.8), ("t", "b_", BLACK, 11.8), (" = \u03b4", "b", BLACK, 11.6), ("t", "b_", BLACK, 11.6), (" + \u03b3\u03bb b", "b", BLACK, 11.6), ("t", "b_", BLACK, 11.6), (" \u00c2", "b", BLACK, 11.6), ("t+1", "b_", BLACK, 11.6)],
        ],
        fill=TD_PURP_BG, border=BLACK, border_pt=1.45, rounded=True, round_radius=0.16, line_spacing=1.08, margin_lr=0.04, margin_tb=0.02,
    )

    y_rt, h_rt = 5.14, 0.68
    add_line(slide, c1cx, y_gae + h_gae, c1cx, y_rt - 0.01, color=TD_PURP_DK, width_pt=1.5, arrow_end="triangle", arrow_size="sm")
    add_box(
        slide, c1x, y_rt, c1w, h_rt,
        [
            [("Return Target", "bsans", TD_PURP_DK, 10.8), ("   (GAE + V", "sans", PALETTE["ink_soft"], 9.4), ("t", "_", PALETTE["ink_soft"], 9.4), (")", "sans", PALETTE["ink_soft"], 9.4)],
            [("R", "bi", BLACK, 12.0), ("t", "b_", BLACK, 12.0), (" = \u00c2", "b", BLACK, 11.8), ("t", "b_", BLACK, 11.8), (" + V", "b", BLACK, 11.8), ("t", "b_", BLACK, 11.8)],
        ],
        fill=TD_PURP_BG, border=BLACK, border_pt=1.45, rounded=True, round_radius=0.16, line_spacing=1.08, margin_lr=0.04, margin_tb=0.02,
    )

    # =========================================================================
    # REGION (c): ACTOR PPO LOSS & CRITIC MSE LOSS (ZERO internal 'ckpt' leaks!)
    # =========================================================================
    c2x, c2w = cx + 0.18, cw - 0.36
    c2cx = c2x + c2w / 2.0

    y_ppo, h_ppo = 1.22, 1.32
    add_box(
        slide, c2x, y_ppo, c2w, h_ppo,
        [
            [("Clipped PPO Loss  L", "bsans", TD_BLUE_DK, 11.0), ("policy", "b_", TD_BLUE_DK, 10.4), ("(\u03b8)", "bsans", TD_BLUE_DK, 11.0)],
            [("\u2212E", "b", BLACK, 10.0), ("t", "b_", BLACK, 10.0), ("[min(r", "b", BLACK, 10.0), ("t", "b_", BLACK, 10.0), ("PPO", "b^", BLACK, 9.4), ("\u00c3", "b", BLACK, 10.0), ("t", "b_", BLACK, 10.0), (", clip(r", "b", BLACK, 10.0), ("t", "b_", BLACK, 10.0), ("PPO", "b^", BLACK, 9.4), (", 1\u00b1\u03b5)\u00c3", "b", BLACK, 10.0), ("t", "b_", BLACK, 10.0), (")]", "b", BLACK, 10.0)],
            [("\u03b5: 0.20 \u2192 0.10   \u00b7   \u00c3", "sans", PALETTE["ink_soft"], 9.4), ("t", "_", PALETTE["ink_soft"], 9.4), (" = norm(\u00c2", "sans", PALETTE["ink_soft"], 9.4), ("t", "_", PALETTE["ink_soft"], 9.4), (")", "sans", PALETTE["ink_soft"], 9.4)],
            [("with entropy bonus  \u2212c", "sans", TD_MUTED_TXT, 9.4), ("H", "_", TD_MUTED_TXT, 9.4), (" H(\u03c0", "bi", TD_MUTED_TXT, 9.6), ("\u03b8", "b_", TD_MUTED_TXT, 9.6), (")", "b", TD_MUTED_TXT, 9.6)],
        ],
        fill=rgb("#E5F1FA"), border=BLACK, border_pt=1.6, rounded=True, round_radius=0.14, line_spacing=1.12, margin_lr=0.04, margin_tb=0.02,
    )

    y_act, h_act = 2.82, 0.60
    add_line(slide, c2cx, y_ppo + h_ppo, c2cx, y_act - 0.01, color=TD_BLUE_DK, width_pt=1.6, arrow_end="triangle", arrow_size="sm")
    add_box(
        slide, c2x + 0.14, y_act, c2w - 0.28, h_act,
        [[("Actor Update  \u03b8", "bsans", PALETTE["white"], 11.8)], [("ScheduleFree AdamW  \u00b7  clip 0.5", "sans", rgb("#D2E5F6"), 9.4)]],
        fill=TD_BLUE_DK, border=BLACK, border_pt=1.4, rounded=True, round_radius=0.22, line_spacing=1.06, margin_lr=0.04, margin_tb=0.01,
    )

    add_line(slide, cx + 0.14, 3.68, cx + cw - 0.14, 3.68, color=PALETTE["line_gray"], width_pt=1.05, dash="dash")

    y_vl, h_vl = 3.96, 0.94
    add_box(
        slide, c2x, y_vl, c2w, h_vl,
        [
            [("Value Loss (MSE)  L", "bsans", TD_PURP_DK, 11.0), ("V", "b_", TD_PURP_DK, 11.0), ("(\u03c6)", "bsans", TD_PURP_DK, 11.0)],
            [("c", "bi", BLACK, 11.8), ("V", "b_", BLACK, 11.8), (" \u00b7 E", "b", BLACK, 11.8), ("t", "b_", BLACK, 11.8), ("[(V", "b", BLACK, 11.8), ("\u03c6", "b_", BLACK, 11.8), ("(s", "b", BLACK, 11.8), ("t", "b_", BLACK, 11.8), (") \u2212 R", "b", BLACK, 11.8), ("t", "b_", BLACK, 11.8), (")", "b", BLACK, 11.8), ("2", "b^", BLACK, 11.8), ("]", "b", BLACK, 11.8)],
            [("independent critic \u03c6  (\u03b8 \u2229 \u03c6 = \u2205)", "sans", PALETTE["ink_soft"], 9.4)],
        ],
        fill=TD_PURP_BG, border=BLACK, border_pt=1.6, rounded=True, round_radius=0.14, line_spacing=1.10, margin_lr=0.04, margin_tb=0.02,
    )

    y_cri, h_cri = 5.20, 0.60
    add_line(slide, c2cx, y_vl + h_vl, c2cx, y_cri - 0.01, color=TD_PURP_DK, width_pt=1.6, arrow_end="triangle", arrow_size="sm")
    add_box(
        slide, c2x + 0.14, y_cri, c2w - 0.28, h_cri,
        [[("Critic Update  \u03c6", "bsans", PALETTE["white"], 11.8)], [("ScheduleFree AdamW  \u00b7  clip 0.05", "sans", rgb("#E5D8F2"), 9.4)]],
        fill=TD_PURP_DK, border=BLACK, border_pt=1.4, rounded=True, round_radius=0.22, line_spacing=1.06, margin_lr=0.04, margin_tb=0.01,
    )

    # TABLE-NUMBER-INDEPENDENT HYPERPARAMETERS STRIP (FIXED wrong 'Table VII'!)
    add_box(
        slide, ax, 6.32, (cx + cw) - ax, 0.56,
        [[
            ("PPO Hyperparameters:   ", "bsans", BLACK, 10.6),
            ("K = 2", "bsans", TD_AMBER_DK, 10.6), (" epochs   |   ", "sans", PALETTE["ink_soft"], 10.2),
            ("\u03b3 = 0.999", "bsans", TD_PURP_DK, 10.6), ("   |   ", "sans", PALETTE["ink_soft"], 10.2),
            ("\u03bb = 0.9", "bsans", TD_PURP_DK, 10.6), ("   |   ", "sans", PALETTE["ink_soft"], 10.2),
            ("\u03b5: 0.20 \u2192 0.10", "bsans", TD_BLUE_DK, 10.6), ("   |   ", "sans", PALETTE["ink_soft"], 10.2),
            ("ScheduleFree AdamW  (clip: \u03b8 0.5, \u03c6 0.05)", "bsans", BLACK, 10.2),
        ]],
        fill=rgb("#F4F6FA"), border=BLACK, border_pt=1.3, rounded=True, round_radius=0.24,
    )

    # =========================================================================
    # ZERO-CROSSING SIGNAL ROUTING ACROSS ALL THREE REGIONS
    # =========================================================================
    add_polyline(
        slide,
        [(xs_t[1] + aw_c / 2, y_a), (3.24, y_a), (3.24, 1.30), (5.02, 1.30), (5.02, y_rep + h_rep / 2), (c1x - 0.02, y_rep + h_rep / 2)],
        color=TD_AMBER_DK, width_pt=1.65, arrow_end="triangle", arrow_size="sm",
    )
    add_box(
        slide, 3.36, 1.15, 1.56, 0.30,
        [[("stored ", "bsans", TD_AMBER_DK, 9.4), ("\u00e3", "bi", TD_AMBER_DK, 10.4), ("t", "b_", TD_AMBER_DK, 10.4), (" = (i,m,w)", "b", TD_AMBER_DK, 9.6)]],
        fill=PALETTE["white"], border=TD_AMBER_DK, border_pt=1.15, rounded=True, round_radius=0.24, margin_lr=0.0, margin_tb=0.0,
    )

    add_polyline(slide, [(xs_t[2] - 0.34, y_v + r_v), (xs_t[2] - 0.34, 4.78), (4.78, 4.78), (4.78, y_td + 0.16), (c1x - 0.02, y_td + 0.16)], color=TD_PURP_DK, width_pt=1.5, arrow_end="triangle", arrow_size="sm")
    add_polyline(slide, [(xs_t[1] + 0.34, y_r + 0.145), (xs_t[1] + 0.34, 4.94), (4.92, 4.94), (4.92, y_td + 0.34), (c1x - 0.02, y_td + 0.34)], color=TD_GREEN_DK, width_pt=1.5, arrow_end="triangle", arrow_size="sm")
    add_polyline(slide, [(xs_t[1] - 0.34, y_v + r_v), (xs_t[1] - 0.34, 5.10), (5.06, 5.10), (5.06, y_td + 0.52), (c1x - 0.02, y_td + 0.52)], color=TD_PURP_DK, width_pt=1.5, arrow_end="triangle", arrow_size="sm")
    add_circle(slide, 5.06, 5.10, 0.045, fill=TD_PURP_DK, border=None)
    add_polyline(slide, [(5.06, 5.10), (5.06, y_rt + h_rt / 2), (c1x - 0.02, y_rt + h_rt / 2)], color=TD_PURP_DK, width_pt=1.5, arrow_end="triangle", arrow_size="sm")

    # Gutter (b) -> (c): r_t^PPO, A_tilde_t, R_t
    add_line(slide, c1x + c1w, 2.22, c2x - 0.02, 2.22, color=TD_AMBER_DK, width_pt=1.65, arrow_end="triangle", arrow_size="sm")
    add_box(slide, 8.70, 1.90, 0.62, 0.28, [[("r", "bi"), ("t", "b_"), ("PPO", "b^")]], fill=PALETTE["white"], border=None, font_name=FONT_SERIF, size_pt=10.2, color=TD_AMBER_DK, margin_lr=0.0, margin_tb=0.0)
    add_polyline(slide, [(c1x + c1w, y_gae + h_gae / 2), (9.00, y_gae + h_gae / 2), (9.00, 2.42), (c2x - 0.02, 2.42)], color=TD_PURP_DK, width_pt=1.55, arrow_end="triangle", arrow_size="sm")
    add_box(slide, 8.77, 3.24, 0.46, 0.30, [[("\u00c3", "b"), ("t", "b_")]], fill=PALETTE["white"], border=TD_PURP_DK, border_pt=1.35, rounded=True, round_radius=0.24, color=TD_PURP_DK, size_pt=10.8, margin_lr=0.0, margin_tb=0.0)
    add_polyline(slide, [(c1x + c1w, y_rt + h_rt / 2), (9.00, y_rt + h_rt / 2), (9.00, 4.68), (c2x - 0.02, 4.68)], color=TD_PURP_DK, width_pt=1.55, arrow_end="triangle", arrow_size="sm")
    add_box(slide, 8.77, 4.94, 0.46, 0.30, [[("R", "bi"), ("t", "b_")]], fill=PALETTE["white"], border=TD_PURP_DK, border_pt=1.35, rounded=True, round_radius=0.24, color=TD_PURP_DK, size_pt=10.8, margin_lr=0.0, margin_tb=0.0)

    set_notes(
        slide,
        "Module B11. PPO learning and GAE advantage estimation (main_tsmc.tex Section IV-D & Appendix B reproducibility table). Refined v8 resolves the ratio symbol collision with team coordination discount rho=0.95 by adopting exact paper notation r_t^PPO(theta), theta^-, L_policy(theta), and TD Error delta_t, removes internal debug note '(ckpt: per-head clip)', and replaces outdated 'Table VII Config' (now Table XII in main_tsmc.tex) with table-number-independent 'PPO Hyperparameters'.\n"
        "Layout reference: Nicklas Hansen et al., Temporal Difference Learning for Model Predictive Control (TD-MPC), ICML 2022, Fig. 2.",
    )


def main():
    parser = argparse.ArgumentParser(description="Build Slide 16 (B11) in refined_v8.")
    parser.add_argument("--pptx", default=DEFAULT_PPTX_PATH, help="Target PPTX path.")
    args = parser.parse_args()
    prs = Presentation(args.pptx)
    build_b11_v8(prs.slides[15])  # Slide 16: B11
    prs.save(args.pptx)
    print("Updated Slide 16 (B11) in", args.pptx)


if __name__ == "__main__":
    main()
