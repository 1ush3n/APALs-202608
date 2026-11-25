"""Build Slide 10 (B5) in refined_v8: 100% aligned with main_tsmc.tex Sections IV-B & IV-C notation.

v8 terminology & publication polish:
1. Policy header updated to 'Policy pi_theta(a_tilde_t | s_t)' matching Section IV-C ordered trajectory notation.
2. Pooling bars updated to exact paper name: 'Node-Type Attention Pooling  [M | O | W]'.
3. Projection bars updated to clean 'Node Type Projection  f_X  (d = 128)' (removed informal 'once').
4. State graph base aligned with Section IV-B line 247: 'Shared Heterogeneous State Graph  G_t = (O U M U W U K, E_t)'.
5. Standardized capitalization ('Op Ptr', 'St Ptr', 'Wk Ptr', 'Per-Relation Softmax').
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
    add_line,
    add_partitioned_vector,
    add_polyline,
    add_textbox,
    clear_slide,
    rgb,
    set_notes,
)

DEFAULT_PPTX_PATH = os.path.join(REFINED_V8_DIR, "APAL_Modules_English_Refined_v8.pptx")

DAN_ROSE_BG   = rgb("#FAF0F1")
DAN_TEAL_BG   = rgb("#EDF5F7")
DAN_LAV       = rgb("#E5DCEF")
DAN_CREAM     = rgb("#FDF1C8")
DAN_SAGE      = rgb("#D7E7CB")
DAN_PEACH     = rgb("#FDE3D1")
DAN_SKY       = rgb("#E5F3FA")
DAN_OP_OUT    = rgb("#B8332E")
DAN_ST_OUT    = rgb("#217A91")
DAN_WK_OUT    = rgb("#B05F1A")
DAN_SK_OUT    = rgb("#683B91")
PPO_BRIDGE_BG = rgb("#F6EFFB")


def build_b5_v8(slide):
    """Slide 10 · B5 publication-ready DAN Fig. 3 dual-tower figure (refined_v8, 100% paper-faithful)."""
    clear_slide(slide)

    env_x0 = 2.18
    env_w  = 8.98
    env_x1 = env_x0 + env_w

    card_w  = 3.32
    inner_w = 2.94
    bx_L    = 2.75
    cx_L    = bx_L + card_w / 2.0    # 4.41
    bx_R    = 7.12
    cx_R    = bx_R + card_w / 2.0    # 8.78
    x_div   = 6.64

    # =========================================================================
    # TOP SYMMETRIC POLICY & VALUE HEADS + PPO COUPLING BRIDGE
    # =========================================================================
    y_ptr = 0.92
    h_ptr = 0.40
    pw_p  = 1.08
    pgap  = 0.12
    p_tot = 3 * pw_p + 2 * pgap      # 3.48 in
    p_x0  = cx_L - p_tot / 2.0       # 2.67 .. 6.15

    grp_L_x0 = p_x0 - 0.08
    grp_y0   = y_ptr - 0.07
    grp_L_w  = p_tot + 0.16
    grp_h    = h_ptr + 0.14
    add_box(
        slide, grp_L_x0, grp_y0, grp_L_w, grp_h,
        fill=rgb("#FDF7F7"), border=DAN_SK_OUT, border_pt=1.35, dash="dash",
        rounded=True, round_radius=0.18,
    )

    ptr_defs = [
        ([("Op Ptr ", "bsans"), ("i", "bi"), ("t", "b_")], rgb("#FADBD8")),
        ([("St Ptr ", "bsans"), ("m", "bi"), ("t", "b_")], rgb("#D6EEF3")),
        ([("Wk Ptr ", "bsans"), ("W", "b"), ("t", "b_")], rgb("#FCE7CF")),
    ]
    for idx, (psegs, pfill) in enumerate(ptr_defs):
        px = p_x0 + idx * (pw_p + pgap)
        add_box(
            slide, px, y_ptr, pw_p, h_ptr,
            [psegs], fill=pfill, border=BLACK, border_pt=1.45,
            rounded=True, round_radius=0.20, font_name=FONT_SERIF, color=BLACK, size_pt=11.6, margin_lr=0.0, margin_tb=0.0,
        )
        if idx < 2:
            add_line(
                slide, px + pw_p + 0.01, y_ptr + h_ptr / 2, px + pw_p + pgap - 0.02, y_ptr + h_ptr / 2,
                color=BLACK, width_pt=1.4, arrow_end="triangle", arrow_size="sm",
            )

    mlp_x0, mlp_w = cx_R - 1.56, 1.54
    v_x0, v_w     = cx_R + 0.22, 1.36
    grp_R_x0      = mlp_x0 - 0.08
    grp_R_w       = (v_x0 + v_w + 0.08) - grp_R_x0
    add_box(
        slide, grp_R_x0, grp_y0, grp_R_w, grp_h,
        fill=rgb("#F8F4FC"), border=DAN_SK_OUT, border_pt=1.35, dash="dash",
        rounded=True, round_radius=0.18,
    )
    add_box(
        slide, mlp_x0, y_ptr, mlp_w, h_ptr,
        [[("MLP (3d\u219264\u21921)", "bsans")]], fill=DAN_LAV, border=BLACK, border_pt=1.45,
        rounded=True, round_radius=0.16, color=BLACK, size_pt=11.4, margin_lr=0.0, margin_tb=0.0,
    )
    add_line(
        slide, mlp_x0 + mlp_w, y_ptr + h_ptr / 2, v_x0 - 0.02, y_ptr + h_ptr / 2,
        color=BLACK, width_pt=1.45, arrow_end="triangle", arrow_size="sm",
    )
    add_box(
        slide, v_x0, y_ptr, v_w, h_ptr,
        [[("V", "b"), ("\u03c6", "b_"), ("(s", "b"), ("t", "b_"), (") \u2208 \u211d", "b")]],
        fill=rgb("#E9DCF4"), border=BLACK, border_pt=1.55,
        rounded=True, round_radius=0.20, color=DAN_SK_OUT, size_pt=13.0, margin_lr=0.0, margin_tb=0.0,
    )

    y_bridge = 0.40
    x_arr_L = cx_L
    x_arr_R = cx_R
    add_polyline(
        slide,
        [(x_arr_R, grp_y0), (x_arr_R, y_bridge), (x_arr_L, y_bridge), (x_arr_L, grp_y0 - 0.01)],
        color=DAN_SK_OUT, width_pt=2.4, dash="dash", arrow_end="triangle", arrow_size="med",
    )
    pill_w  = 3.74
    pill_cx = (x_arr_L + x_arr_R) / 2.0
    add_box(
        slide, pill_cx - pill_w / 2.0, y_bridge - 0.19, pill_w, 0.38,
        [[
            ("PPO Coupling:  ", "bsans", DAN_SK_OUT, 11.8),
            ("GAE Baseline ", "bsans", BLACK, 11.4),
            ("\u00c2", "bi", DAN_SK_OUT, 12.6), ("t", "b_", DAN_SK_OUT, 12.6),
            (" \u2192 \u2207", "b", DAN_SK_OUT, 12.2), ("\u03b8", "b_", DAN_SK_OUT, 12.2),
        ]],
        fill=PPO_BRIDGE_BG, border=DAN_SK_OUT, border_pt=1.65,
        rounded=True, round_radius=0.30, font_name=FONT_SERIF, margin_lr=0.02, margin_tb=0.0,
    )
    # Exact Section IV-C trajectory policy notation: pi_theta(a_tilde_t | s_t)
    add_textbox(
        slide, grp_L_x0, 0.54, 1.76, 0.28,
        [[("Policy  ", "bsans", DAN_SK_OUT, 12.0), ("\u03c0", "bi", DAN_SK_OUT, 13.0), ("\u03b8", "b_", DAN_SK_OUT, 13.0), ("(\u00e3", "b", DAN_SK_OUT, 12.2), ("t", "b_", DAN_SK_OUT, 12.2), ("|s", "b", DAN_SK_OUT, 12.2), ("t", "b_", DAN_SK_OUT, 12.2), (")", "b", DAN_SK_OUT, 12.2)]],
        font_name=FONT_SERIF, size_pt=12.2, color=DAN_SK_OUT, align=PP_ALIGN.LEFT, margin_lr=0.0, word_wrap=False,
    )
    add_textbox(
        slide, x_arr_R + 0.12, 0.54, 1.56, 0.28,
        [[("Value  ", "bsans", DAN_SK_OUT, 12.0), ("V", "b", DAN_SK_OUT, 12.8), ("\u03c6", "b_", DAN_SK_OUT, 12.8), ("(s", "b", DAN_SK_OUT, 12.2), ("t", "b_", DAN_SK_OUT, 12.2), (")", "b", DAN_SK_OUT, 12.2)]],
        font_name=FONT_SERIF, size_pt=12.2, color=DAN_SK_OUT, align=PP_ALIGN.RIGHT, margin_lr=0.0, word_wrap=False,
    )

    # Partitioned g_theta / g_phi & Node-Type Attention Pooling (exact Section IV-B name!)
    y_g    = 1.60
    y_pool = 2.02
    for cx_t, sym in [(cx_L, "\u03b8"), (cx_R, "\u03c6")]:
        add_box(
            slide, cx_t - 1.62, y_pool, 3.24, 0.31,
            [[("Node-Type Attention Pooling  [M | O | W]", "bsans")]],
            fill=rgb("#F1F8FC"), border=BLACK, border_pt=1.3, rounded=True, round_radius=0.22,
            color=BLACK, size_pt=10.8, margin_lr=0.01, margin_tb=0.0,
        )
        add_line(slide, cx_t, y_pool, cx_t, y_g + 0.29, color=BLACK, width_pt=1.4, arrow_end="triangle", arrow_size="sm")
        add_partitioned_vector(
            slide, cx_t - 0.99, y_g,
            [[("g", "b"), ("M", "b^")], [("g", "b"), ("O", "b^")], [("g", "b"), ("W", "b^")]],
            cell_w=0.66, cell_h=0.29,
            fills=[rgb("#D8EFF4"), rgb("#FADBD8"), rgb("#FBE5CC")],
            border=BLACK, border_pt=1.35, size_pt=12.2,
        )
        add_textbox(
            slide, cx_t + 1.07, y_g, 0.96, 0.29,
            [[("g", "bi"), (sym, "b_"), (" \u2208 \u211d", "b"), ("3d", "b^")]],
            font_name=FONT_SERIF, size_pt=12.0, color=BLACK, align=PP_ALIGN.LEFT, margin_lr=0.0, word_wrap=False,
        )

    add_polyline(
        slide,
        [(cx_L, y_g), (cx_L, 1.46), (p_x0 + pw_p / 2, 1.46), (p_x0 + pw_p / 2, y_ptr + h_ptr + 0.01)],
        color=BLACK, width_pt=1.45, arrow_end="triangle", arrow_size="sm",
    )
    add_polyline(
        slide,
        [(cx_R, y_g), (cx_R, 1.46), (mlp_x0 + mlp_w / 2, 1.46), (mlp_x0 + mlp_w / 2, y_ptr + h_ptr + 0.01)],
        color=BLACK, width_pt=1.45, arrow_end="triangle", arrow_size="sm",
    )

    # =========================================================================
    # OUTER DOTTED ROUNDED ENCLOSURE (x5 layers) [y = 2.45 .. 6.04]
    # =========================================================================
    enc_y0, enc_h = 2.45, 3.59
    add_box(
        slide, env_x0, enc_y0, env_w, enc_h,
        fill=PALETTE["white"], border=BLACK, border_pt=1.55, dash="dot", rounded=True, round_radius=0.05,
    )
    add_textbox(
        slide, env_x1 - 0.58, enc_y0 + 0.04, 0.54, 0.34,
        [[("\u00d7 5", "bi")]], font_name=FONT_SERIF, size_pt=18.0, color=BLACK, align=PP_ALIGN.CENTER,
    )

    add_line(slide, x_div, enc_y0 + 0.14, x_div, enc_y0 + enc_h - 0.14, color=PALETTE["line_gray"], width_pt=1.1, dash="dash")
    add_box(
        slide, x_div - 0.18, 2.84, 0.36, 2.78,
        fill=PALETTE["white"], border=BLACK, border_pt=1.35, rounded=True, round_radius=0.38,
    )
    add_angled_label(
        slide, x_div - 1.38, 4.10, 2.76, 0.28,
        "Disjoint weights  (\u03b8 \u2229 \u03c6 = \u2205)", angle_deg=-90.0, size_pt=11.6, color=BLACK, bold=True,
    )

    tower_defs = [
        (bx_L, cx_L, DAN_ROSE_BG, "Actor Encoder Block (\u03b8)", True),
        (bx_R, cx_R, DAN_TEAL_BG, "Critic Encoder Block (\u03c6)", False),
    ]

    y_res_bar  = 2.58
    y_hm_bar   = 3.08
    y_card_top = 3.59
    card_h     = 2.04

    for bx, cx_t, bg_tint, side_lbl, is_actor in tower_defs:
        for k in (2, 1):
            add_box(
                slide, bx - k * 0.085, y_card_top + k * 0.08, card_w, card_h,
                fill=bg_tint, border=rgb("#98A2AC"), border_pt=1.05, rounded=True, round_radius=0.06,
            )
            stem_x = bx - k * 0.085 + 0.10
            add_line(slide, stem_x, y_card_top + k * 0.08, stem_x, y_hm_bar + 0.34, color=BLACK, width_pt=1.15)

        add_box(
            slide, bx, y_card_top, card_w, card_h,
            fill=bg_tint, border=BLACK, border_pt=1.55, rounded=True, round_radius=0.06,
        )

        add_textbox(
            slide, bx - 0.18, y_card_top + card_h + 0.17, 1.00, 0.24,
            [[("4 heads", "bsans")]], font_name=FONT_SANS, size_pt=11.0, color=BLACK, align=PP_ALIGN.LEFT, margin_lr=0.0, word_wrap=False,
        )

        if is_actor:
            add_angled_label(
                slide, 2.38 - 1.54, 4.12, 3.08, 0.32,
                side_lbl, angle_deg=-90.0, size_pt=13.4, color=BLACK, bold=True,
            )
        else:
            add_angled_label(
                slide, 10.94 - 1.54, 4.26, 3.08, 0.32,
                side_lbl, angle_deg=90.0, size_pt=13.4, color=BLACK, bold=True,
            )

        px_in = cx_t - inner_w / 2.0
        ph = 0.32
        y_l1 = 5.20
        y_gl = 4.92
        y_l2 = 4.44
        y_l3 = 4.05
        y_l4 = 3.66

        add_box(
            slide, px_in, y_l1, inner_w, ph,
            [[("Linear  W", "bsans"), ("s", "b_"), (" h", "bi"), ("u", "b_"), (" + W", "bsans"), ("t", "b_"), (" h", "bi"), ("v", "b_")]],
            fill=DAN_LAV, border=BLACK, border_pt=1.25, rounded=True, round_radius=0.20, color=BLACK, size_pt=11.4, margin_lr=0.01, margin_tb=0.0,
        )
        add_line(slide, cx_t, y_l1, cx_t, y_gl + 0.125, color=BLACK, width_pt=1.25)
        add_activation_glyph(slide, cx_t, y_gl, r=0.125, kind="lrelu", fill=PALETTE["white"], border=BLACK, border_pt=1.3, curve_pt=1.25)
        add_textbox(
            slide, cx_t + 0.16, y_gl - 0.13, 1.38, 0.26,
            [[("LeakyReLU (0.1)", "bisans")]], font_name=FONT_SANS, size_pt=10.8, color=rgb("#1F2933"), align=PP_ALIGN.LEFT, margin_lr=0.0, word_wrap=False,
        )
        add_line(slide, cx_t, y_gl - 0.125, cx_t, y_l2 + ph, color=BLACK, width_pt=1.25)
        add_box(
            slide, px_in, y_l2, inner_w, ph,
            [[("GATv2 Attention  a", "bsans"), ("r,h", "b_"), ("T", "b^")]],
            fill=DAN_CREAM, border=BLACK, border_pt=1.25, rounded=True, round_radius=0.20, color=BLACK, size_pt=11.4, margin_lr=0.01, margin_tb=0.0,
        )
        add_line(slide, cx_t, y_l2, cx_t, y_l3 + ph, color=BLACK, width_pt=1.25)
        add_box(
            slide, px_in, y_l3, inner_w, ph,
            [[("Per-Relation Softmax", "bsans")]],
            fill=DAN_SAGE, border=BLACK, border_pt=1.25, rounded=True, round_radius=0.20, color=BLACK, size_pt=11.4, margin_lr=0.01, margin_tb=0.0,
        )
        add_line(slide, cx_t, y_l3, cx_t, y_l4 + ph, color=BLACK, width_pt=1.25)
        add_box(
            slide, px_in, y_l4, inner_w, ph,
            [[("Weighted Value  \u03a3", "bsans"), ("u", "b_"), (" \u03b1 W", "bsans"), ("s", "b_"), (" h", "bi"), ("u", "b_")]],
            fill=DAN_PEACH, border=BLACK, border_pt=1.25, rounded=True, round_radius=0.20, color=BLACK, size_pt=11.3, margin_lr=0.01, margin_tb=0.0,
        )

        add_line(slide, cx_t, y_l4, cx_t, y_hm_bar + 0.34, color=BLACK, width_pt=1.35)

        bar_x = bx - 0.14
        bar_w = card_w + 0.14
        add_box(
            slide, bar_x, y_hm_bar, bar_w, 0.34,
            [[("Multi-Head Mean (4 Heads) + Relation Sum", "bsans")]],
            fill=DAN_SKY, border=BLACK, border_pt=1.4, rounded=True, round_radius=0.18,
            color=BLACK, size_pt=10.8, margin_lr=0.01, margin_tb=0.0,
        )

        add_line(slide, cx_t, y_hm_bar, cx_t, y_res_bar + 0.35 + 0.01, color=BLACK, width_pt=1.35, arrow_end="triangle", arrow_size="sm")
        add_box(
            slide, bar_x, y_res_bar, bar_w, 0.35,
            [[("Post-Norm LN  \u2192  LReLU (0.1)  \u2192  Add (+)", "bsans")]],
            fill=DAN_PEACH, border=BLACK, border_pt=1.4, rounded=True, round_radius=0.18,
            color=BLACK, size_pt=11.0, margin_lr=0.01, margin_tb=0.0,
        )

        rx_sk = bx + card_w + 0.10
        add_polyline(
            slide,
            [(cx_t, y_card_top + card_h + 0.14), (rx_sk, y_card_top + card_h + 0.14), (rx_sk, y_res_bar + 0.175), (bar_x + bar_w + 0.02, y_res_bar + 0.175)],
            color=BLACK, width_pt=1.35, arrow_end="triangle", arrow_size="sm",
        )
        add_angled_label(
            slide, rx_sk + 0.10 - 0.36, y_card_top + 0.90, 0.72, 0.24,
            "skip", angle_deg=90.0, size_pt=10.8, color=rgb("#2B3945"), bold=True,
        )

        add_line(slide, cx_t, y_res_bar, cx_t, y_pool + 0.32, color=BLACK, width_pt=1.45, arrow_end="triangle", arrow_size="sm")

    y_tap = 2.39
    x_byp = 2.48
    add_polyline(
        slide,
        [(cx_L, y_tap), (x_byp, y_tap), (x_byp, y_ptr + h_ptr / 2), (p_x0 - 0.03, y_ptr + h_ptr / 2)],
        color=BLACK, width_pt=1.4, dash="dash", arrow_end="triangle", arrow_size="sm",
    )
    add_box(
        slide, env_x0, 1.58, 0.62, 0.32,
        [[("H", "b"), ("\u03b8", "b_"), ("(5)", "b^")]],
        fill=PALETTE["white"], border=BLACK, border_pt=1.3, rounded=True, round_radius=0.22,
        color=BLACK, size_pt=12.0, margin_lr=0.0, margin_tb=0.0,
    )

    # =========================================================================
    # BOTTOM: NODE TYPE PROJECTION f_X + PAPER-FAITHFUL STATE GRAPH BASE
    # =========================================================================
    y_fx = 6.21
    h_fx = 0.36
    for cx_t in (cx_L, cx_R):
        add_box(
            slide, cx_t - 1.58, y_fx, 3.16, h_fx,
            [[("Node Type Projection  ", "bsans"), ("f", "bi"), ("X", "b_"), ("  (", "bsans"), ("d", "bi"), (" = 128)", "bsans")]],
            fill=DAN_LAV, border=BLACK, border_pt=1.35, rounded=True, round_radius=0.18,
            font_name=FONT_SERIF, color=BLACK, size_pt=11.4, margin_lr=0.01, margin_tb=0.0,
        )
        add_line(slide, cx_t, y_fx, cx_t, y_card_top + card_h + 0.01, color=BLACK, width_pt=1.4, arrow_end="triangle", arrow_size="sm")
        for k in (1, 2):
            tx = cx_t - k * 0.24
            add_polyline(
                slide,
                [(cx_t, y_fx - 0.09), (tx, y_fx - 0.09), (tx, y_card_top + card_h + k * 0.08 + 0.01)],
                color=BLACK, width_pt=1.15, arrow_end="triangle", arrow_size="sm",
            )

    y_base = 6.78
    h_base = 0.48
    add_box(
        slide, env_x0, y_base, env_w, h_base,
        [[
            ("Shared Heterogeneous State Graph  ", "bsans", BLACK, 14.6),
            ("G", "bi", BLACK, 15.8), ("t", "b_", BLACK, 15.8),
            (" = (", "b", BLACK, 15.2),
            ("O", "bi", DAN_OP_OUT, 15.4),
            (" \u222a ", "b", PALETTE["ink_soft"], 14.4),
            ("M", "bi", DAN_ST_OUT, 15.4),
            (" \u222a ", "b", PALETTE["ink_soft"], 14.4),
            ("W", "bi", DAN_WK_OUT, 15.4),
            (" \u222a ", "b", PALETTE["ink_soft"], 14.4),
            ("K", "bi", DAN_SK_OUT, 15.4),
            (", ", "b", BLACK, 15.2),
            ("E", "bi", BLACK, 15.4), ("t", "b_", BLACK, 15.4),
            (")", "b", BLACK, 15.2),
        ]],
        fill=rgb("#F5F8FC"), border=BLACK, border_pt=1.6, rounded=True, round_radius=0.24,
        font_name=FONT_SERIF, margin_lr=0.04, margin_tb=0.0,
    )

    add_line(slide, cx_L, y_base, cx_L, y_fx + h_fx + 0.02, color=BLACK, width_pt=1.65, arrow_end="triangle", arrow_size="med")
    add_line(slide, cx_R, y_base, cx_R, y_fx + h_fx + 0.02, color=BLACK, width_pt=1.65, arrow_end="triangle", arrow_size="med")

    set_notes(
        slide,
        "Module B5. Independent Actor and Critic heterogeneous graph encoders (HGP-PPO in main_tsmc.tex Sections IV-B & IV-C; models/hb_gat_pn.py with use_shared_trunk=False). Refined v8 aligns every symbol and module title verbatim with main_tsmc.tex: Policy pi_theta(a_tilde_t | s_t), Node-Type Attention Pooling [M | O | W], Node Type Projection f_X (d = 128), and Shared Heterogeneous State Graph G_t = (O U M U W U K, E_t), with globally elevated typography (min_font = 10.8pt).\n"
        "Layout reference: Runqing Wang et al., Flexible Job Shop Scheduling via Dual Attention Network-Based Reinforcement Learning, IEEE TNNLS 2024, Fig. 3.",
    )


def main():
    parser = argparse.ArgumentParser(description="Build Slide 10 (B5) in refined_v8.")
    parser.add_argument("--pptx", default=DEFAULT_PPTX_PATH, help="Target PPTX path.")
    args = parser.parse_args()
    prs = Presentation(args.pptx)
    build_b5_v8(prs.slides[9])  # Slide 10: B5
    prs.save(args.pptx)
    print("Updated Slide 10 (B5) in", args.pptx)


if __name__ == "__main__":
    main()
