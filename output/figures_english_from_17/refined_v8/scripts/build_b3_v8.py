"""Build Slide 8 (B3) in refined_v8: 100% academic paper terminology aligned with main_tsmc.tex Section IV-B.

v8 terminology & publication polish:
1. Stripped internal slide reference '(B4)' -> 'Node-Type Attention Pooling' (exact Section IV-B term).
2. Stripped all PyTorch Geometric code arguments/class names ('concat = False', 'aggr = sum', 'GATv2Conv'):
   - Top bar: 'Inside Layer l for Target Operation O_3  (H = 4 heads, head-averaged, relation-summed)'
   - Head 1: 'Head 1  (expanded GATv2 attention)'
   - Head Mean bar: 'Multi-Head Mean (H = 4): m_prec = 1/4 (m_{r,1} + m_{r,2} + m_{r,3} + m_{r,4})'
   - Relation Sum: 'Relation Sum / Sum_{r in R_in} m_3^(r,l)' (exact paper symbol R_in(v), removed redundant '(sum)').
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
    add_partitioned_vector,
    add_polyline,
    add_textbox,
    clear_slide,
    rgb,
    set_notes,
)

DEFAULT_PPTX_PATH = os.path.join(REFINED_V8_DIR, "APAL_Modules_English_Refined_v8.pptx")

SMIT_PEACH   = rgb("#FAD2B4")
SMIT_MINT    = rgb("#DBF0D5")
SMIT_BLUE    = rgb("#CFE3F6")
SMIT_BLUE_HI = rgb("#9CC3E8")
SMIT_GRAY    = rgb("#C2C8CE")
SMIT_LAV     = rgb("#E7DBF2")
SMIT_CREAM   = rgb("#FDF2CC")


def build_b3_v8(slide):
    """Slide 8 · B3 pure publication-ready paper figure (zero code kwargs or internal slide IDs)."""
    clear_slide(slide)

    # =========================================================================
    # LEFT: 5-LAYER ENCODER STACK [y = 0.56 .. 6.94]
    # =========================================================================
    sx_c = 1.68
    sw   = 2.60

    add_partitioned_vector(
        slide, sx_c - 1.16, 0.66,
        [[("x", "bi"), ("O", "b_")], [("x", "bi"), ("M", "b_")], [("x", "bi"), ("W", "b_")], [("x", "bi"), ("K", "b_")]],
        cell_w=0.58, cell_h=0.38,
        fills=[rgb("#FCE8E6"), rgb("#E2F2EE"), rgb("#FBF0DF"), rgb("#EFE8F7")],
        border=BLACK, border_pt=1.35, size_pt=12.6,
    )
    add_line(slide, sx_c, 1.04, sx_c, 1.30, color=BLACK, width_pt=1.35, arrow_end="triangle", arrow_size="sm")

    add_box(
        slide, sx_c - sw / 2, 1.32, sw, 0.44,
        [[("Node Type Projection ", "b"), ("f", "bi"), ("X", "b_"), (" (d=128)", "")]],
        fill=SMIT_PEACH, border=BLACK, border_pt=1.45, rounded=True, round_radius=0.14,
        color=BLACK, size_pt=11.8, margin_lr=0.02, margin_tb=0.0,
    )
    add_line(slide, sx_c, 1.76, sx_c, 2.00, color=BLACK, width_pt=1.35, arrow_end="triangle", arrow_size="sm")

    add_partitioned_vector(
        slide, sx_c - 1.16, 2.02,
        [[("h", "bi"), ("O", "b_"), ("(0)", "b^")], [("h", "bi"), ("M", "b_"), ("(0)", "b^")], [("h", "bi"), ("W", "b_"), ("(0)", "b^")], [("h", "bi"), ("K", "b_"), ("(0)", "b^")]],
        cell_w=0.58, cell_h=0.38, fill=SMIT_MINT, border=BLACK, border_pt=1.35, size_pt=11.6,
    )
    add_line(slide, sx_c, 2.40, sx_c, 2.66, color=BLACK, width_pt=1.35, arrow_end="triangle", arrow_size="sm")

    add_box(
        slide, sx_c - sw / 2, 2.68, sw, 0.44,
        [[("Hetero-GATv2 Layer 1", "b")]],
        fill=SMIT_BLUE, border=BLACK, border_pt=1.4, rounded=True, round_radius=0.12, color=BLACK, size_pt=12.4,
    )
    add_line(slide, sx_c, 3.12, sx_c, 3.38, color=BLACK, width_pt=1.35, arrow_end="triangle", arrow_size="sm")

    ly_f, lh_f = 3.40, 0.48
    add_box(
        slide, sx_c - sw / 2, ly_f, sw, lh_f,
        [[("Hetero-GATv2 Layer \u2113", "b")]],
        fill=SMIT_BLUE_HI, border=BLACK, border_pt=2.0, rounded=True, round_radius=0.12, color=BLACK, size_pt=13.0,
    )
    add_line(slide, sx_c, ly_f + lh_f, sx_c, 4.10, color=BLACK, width_pt=1.3, dash="dot")
    add_textbox(slide, sx_c - 0.30, 4.08, 0.60, 0.24, [[("\u22ee", "b")]], font_name=FONT_SERIF, size_pt=13.5, color=BLACK, align=PP_ALIGN.CENTER)
    add_line(slide, sx_c, 4.32, sx_c, 4.58, color=BLACK, width_pt=1.35, arrow_end="triangle", arrow_size="sm")

    add_box(
        slide, sx_c - sw / 2, 4.60, sw, 0.44,
        [[("Hetero-GATv2 Layer 5", "b")]],
        fill=SMIT_BLUE, border=BLACK, border_pt=1.4, rounded=True, round_radius=0.12, color=BLACK, size_pt=12.4,
    )
    add_polyline(slide, [(0.35, 2.72), (0.28, 2.72), (0.28, 5.00), (0.35, 5.00)], color=BLACK, width_pt=1.2)
    add_textbox(slide, 0.00, 3.70, 0.26, 0.32, [[("5\u00d7", "bsans")]], font_name=FONT_SANS, size_pt=10.2, color=BLACK, align=PP_ALIGN.RIGHT, margin_lr=0.0)
    add_line(slide, sx_c, 5.04, sx_c, 5.30, color=BLACK, width_pt=1.35, arrow_end="triangle", arrow_size="sm")

    add_partitioned_vector(
        slide, sx_c - 1.16, 5.32,
        [[("H", "b"), ("M", "b^")], [("H", "b"), ("O", "b^")], [("H", "b"), ("W", "b^")], [("H", "b"), ("K", "b^")]],
        cell_w=0.58, cell_h=0.38, fill=SMIT_MINT, border=BLACK, border_pt=1.35, size_pt=12.2,
    )
    add_line(slide, sx_c, 5.70, sx_c, 5.94, color=BLACK, width_pt=1.35, arrow_end="triangle", arrow_size="sm")

    # Clean exact paper term — NO '(B4)'!
    add_box(
        slide, sx_c - sw / 2, 5.96, sw, 0.40,
        [[("Node-Type Attention Pooling", "b")]],
        fill=SMIT_GRAY, border=BLACK, border_pt=1.4, rounded=True, round_radius=0.12, color=BLACK, size_pt=11.6, margin_lr=0.02, margin_tb=0.0,
    )
    add_line(slide, sx_c, 6.36, sx_c, 6.54, color=BLACK, width_pt=1.35, arrow_end="triangle", arrow_size="sm")
    add_box(
        slide, sx_c - 0.64, 6.56, 1.28, 0.36,
        [[("g", "bi"), ("t", "b_"), (" \u2208 \u211d", "b"), ("3d", "b^")]],
        fill=PALETTE["white"], border=BLACK, border_pt=1.35, color=BLACK, size_pt=12.4,
    )

    # =========================================================================
    # RIGHT: DOTTED ROUNDED ZOOM BOX [x = 3.48 .. 12.98, y = 0.56 .. 6.94]
    # =========================================================================
    zx, zy, zw, zh = 3.48, 0.56, 9.50, 6.38
    add_box(slide, zx, zy, zw, zh, fill=rgb("#FCFDFF"), border=BLACK, border_pt=1.7, dash="dot", rounded=True, round_radius=0.04)
    add_line(slide, sx_c + sw / 2, ly_f, zx + 0.06, zy + 0.08, color=BLACK, width_pt=1.3, dash="dot")
    add_line(slide, sx_c + sw / 2, ly_f + lh_f, zx + 0.06, zy + zh - 0.08, color=BLACK, width_pt=1.3, dash="dot")

    # Top title bar — pure academic wording (NO concat=False, NO aggr=sum!)
    add_textbox(
        slide, zx + 0.16, zy + 0.06, zw - 0.32, 0.32,
        [[("Inside Layer \u2113 for Target Operation ", "b"), ("O", "b", PALETTE["op_red_dark"]), ("3", "b_", PALETTE["op_red_dark"]), ("   (H = 4 heads, head-averaged, relation-summed)", "sans", PALETTE["ink_soft"], 11.0)]],
        font_name=FONT_SERIF, size_pt=14.4, color=BLACK, align=PP_ALIGN.LEFT,
    )

    # -------------------------------------------------------------------------
    # SUB-BLOCK 1: Relation 1 `precedes` (O1, O2 -> O3)
    # -------------------------------------------------------------------------
    r1_x, r1_y, r1_w, r1_h = 3.66, 1.10, 6.26, 3.88
    add_box(slide, r1_x, r1_y, r1_w, r1_h, fill=rgb("#FFFDFB"), border=PALETTE["op_red_dark"], border_pt=1.4, rounded=True, round_radius=0.04)
    add_box(
        slide, r1_x + 0.12, r1_y - 0.14, 3.04, 0.28,
        [[("Relation 1: precedes  (O\u2081, O\u2082 \u2192 O\u2083)", "bsans")]],
        fill=rgb("#FCE8E6"), border=PALETTE["op_red_dark"], border_pt=1.15, rounded=True, round_radius=0.22,
        color=PALETTE["op_red_dark"], size_pt=10.4, margin_lr=0.02, margin_tb=0.0,
    )

    h1_x, h1_y, h1_w, h1_h = r1_x + 0.12, 1.32, 5.70, 1.84
    add_box(slide, h1_x, h1_y, h1_w, h1_h, fill=rgb("#F7FAF7"), border=BLACK, border_pt=1.2, dash="dash", rounded=True, round_radius=0.05)
    add_textbox(slide, h1_x + 0.10, h1_y + 0.03, 2.94, 0.24, [[("Head 1  (expanded GATv2 attention)", "bsans")]], font_name=FONT_SANS, size_pt=10.2, color=BLACK, align=PP_ALIGN.LEFT)
    add_textbox(slide, h1_x + 3.06, h1_y + 0.03, 2.42, 0.24, [[("value path: W", "bsans"), ("s,1", "b_"), (" h", "bi"), ("u", "b_")]], font_name=FONT_SANS, size_pt=10.0, color=PALETTE["op_red_dark"], align=PP_ALIGN.CENTER)

    y_rowA = 1.96
    add_partitioned_vector(slide, h1_x + 0.10, y_rowA - 0.17, [[("h", "bi"), ("1", "b_")], [("h", "bi"), ("2", "b_")]], cell_w=0.40, cell_h=0.34, fill=rgb("#FCE8E6"), border=BLACK, border_pt=1.25, size_pt=11.6)
    add_line(slide, h1_x + 0.90, y_rowA, h1_x + 1.06, y_rowA, color=BLACK, width_pt=1.3, arrow_end="triangle", arrow_size="sm")
    add_box(slide, h1_x + 1.08, y_rowA - 0.19, 0.94, 0.38, [[("W", "b"), ("s,1", "b_"), (" h", "bi"), ("u", "b_")]], fill=SMIT_MINT, border=BLACK, border_pt=1.3, rounded=True, round_radius=0.10, color=BLACK, size_pt=11.6)

    y_rowB = 2.80
    add_box(slide, h1_x + 0.10, y_rowB - 0.18, 0.80, 0.36, [[("h", "bi"), ("3", "b_"), ("(\u2113\u22121)", "b^")]], fill=rgb("#FADBD8"), border=BLACK, border_pt=1.35, color=PALETTE["op_red_dark"], size_pt=11.4)
    add_line(slide, h1_x + 0.90, y_rowB, h1_x + 1.06, y_rowB, color=BLACK, width_pt=1.3, arrow_end="triangle", arrow_size="sm")
    add_box(slide, h1_x + 1.08, y_rowB - 0.19, 0.94, 0.38, [[("W", "b"), ("t,1", "b_"), (" h", "bi"), ("3", "b_")]], fill=SMIT_MINT, border=BLACK, border_pt=1.3, rounded=True, round_radius=0.10, color=BLACK, size_pt=11.6)

    px_add, py_add = h1_x + 2.34, 2.38
    add_polyline(slide, [(h1_x + 2.02, y_rowA), (px_add, y_rowA), (px_add, py_add - 0.16)], color=BLACK, width_pt=1.3, arrow_end="triangle", arrow_size="sm")
    add_polyline(slide, [(h1_x + 2.02, y_rowB), (px_add, y_rowB), (px_add, py_add + 0.16)], color=BLACK, width_pt=1.3, arrow_end="triangle", arrow_size="sm")
    add_circle(slide, px_add, py_add, 0.155, [[("+", "b")]], fill=SMIT_CREAM, border=BLACK, border_pt=1.35, color=BLACK, size_pt=13.5)

    px_act = h1_x + 2.82
    add_line(slide, px_add + 0.16, py_add, px_act - 0.16, py_add, color=BLACK, width_pt=1.3, arrow_end="triangle", arrow_size="sm")
    add_activation_glyph(slide, px_act, py_add, r=0.155, kind="lrelu", fill=PALETTE["white"], border=BLACK, border_pt=1.35)
    add_textbox(slide, px_act - 0.34, py_add + 0.18, 0.68, 0.22, [[("LReLU", "bsans")]], font_name=FONT_SANS, size_pt=9.8, color=BLACK, align=PP_ALIGN.CENTER)

    px_a = h1_x + 3.20
    add_line(slide, px_act + 0.16, py_add, px_a - 0.02, py_add, color=BLACK, width_pt=1.3, arrow_end="triangle", arrow_size="sm")
    add_box(slide, px_a, py_add - 0.19, 0.58, 0.38, [[("a", "bi"), ("1", "b_"), ("T", "b^")]], fill=SMIT_PEACH, border=BLACK, border_pt=1.3, rounded=True, round_radius=0.12, color=BLACK, size_pt=12.4)
    add_textbox(slide, px_a - 0.08, py_add + 0.21, 0.74, 0.22, [[("e", "bi"), ("u3", "b_")]], font_name=FONT_SERIF, size_pt=10.8, color=BLACK, align=PP_ALIGN.CENTER)

    px_sm = h1_x + 3.98
    add_line(slide, px_a + 0.58, py_add, px_sm - 0.02, py_add, color=BLACK, width_pt=1.3, arrow_end="triangle", arrow_size="sm")
    add_box(
        slide, px_sm, py_add - 0.28, 0.94, 0.56,
        [[("softmax", "b"), ("r", "b_")], [(".65, .35", "bsans", PALETTE["op_red_dark"], 9.8)]],
        fill=SMIT_CREAM, border=BLACK, border_pt=1.3, rounded=True, round_radius=0.12, color=BLACK, size_pt=10.8, line_spacing=1.04,
    )

    px_ws = h1_x + 5.32
    add_polyline(
        slide,
        [(h1_x + 1.55, y_rowA - 0.19), (h1_x + 1.55, 1.64), (px_ws, 1.64), (px_ws, py_add - 0.20)],
        color=PALETTE["op_red_dark"], width_pt=1.3, dash="dash", arrow_end="triangle", arrow_size="sm",
    )
    add_line(slide, px_sm + 0.94, py_add, px_ws - 0.20, py_add, color=BLACK, width_pt=1.3, arrow_end="triangle", arrow_size="sm")
    add_circle(slide, px_ws, py_add, 0.19, [[("\u03a3", "b"), ("u", "b_")]], fill=SMIT_BLUE, border=BLACK, border_pt=1.35, color=BLACK, size_pt=12.4)
    add_textbox(slide, px_ws - 0.34, py_add + 0.21, 0.68, 0.22, [[("m", "b"), ("r,1", "b_")]], font_name=FONT_SERIF, size_pt=11.0, color=PALETTE["op_red_dark"], align=PP_ALIGN.CENTER)

    # Parallel Heads 2, 3, 4
    head_ys = [3.26, 3.63, 4.00]
    head_h  = 0.31
    head_fills = [rgb("#EEF5FA"), rgb("#EBF3F9"), rgb("#E8F1F8")]
    for h_num, hy, hf in zip((2, 3, 4), head_ys, head_fills):
        add_box(
            slide, h1_x, hy, h1_w, head_h,
            [[(f"Head {h_num} (h={h_num}):  ", "bsans", BLACK, 10.6), ("W", "b"), (f"s,{h_num}", "b_"), (" h", "bi"), ("u", "b_"), (" + W", "b"), (f"t,{h_num}", "b_"), (" h", "bi"), ("3", "b_"), (" \u2192 LReLU \u2192 a", ""), (f"{h_num}", "_"), ("T", "^"), (" \u2192 softmax", ""), ("r", "_"), (" \u2192 \u03a3", ""), ("u", "_"), (" \u2192 ", ""), ("m", "b", PALETTE["op_red_dark"]), (f"r,{h_num}", "b_", PALETTE["op_red_dark"])]],
            fill=hf, border=BLACK, border_pt=1.2, rounded=True, round_radius=0.14, color=BLACK, size_pt=10.6,
            align=PP_ALIGN.CENTER, margin_lr=0.04, margin_tb=0.0,
        )

    # Horizontal Full-Width Multi-Head Mean Bar (NO concat = False!)
    y_hm, h_hm = 4.46, 0.42
    hm_w = r1_w - 0.24
    add_box(
        slide, h1_x, y_hm, hm_w, h_hm,
        [[("Multi-Head Mean (H = 4):   ", "bsans", BLACK, 11.0), ("m", "b", PALETTE["op_red_dark"], 11.8), ("prec", "b_", PALETTE["op_red_dark"], 11.8), (" = \u00bc (m", "b", BLACK, 11.4), ("r,1", "b_", BLACK, 11.4), (" + m", "b", BLACK, 11.4), ("r,2", "b_", BLACK, 11.4), (" + m", "b", BLACK, 11.4), ("r,3", "b_", BLACK, 11.4), (" + m", "b", BLACK, 11.4), ("r,4", "b_", BLACK, 11.4), (")", "b", BLACK, 11.4)]],
        fill=SMIT_PEACH, border=BLACK, border_pt=1.45, rounded=True, round_radius=0.14,
    )
    x_bus = h1_x + h1_w + 0.16
    add_line(slide, px_ws + 0.19, py_add, x_bus, py_add, color=PALETTE["op_red_dark"], width_pt=1.35)
    for hy in head_ys:
        add_line(slide, h1_x + h1_w, hy + head_h / 2, x_bus, hy + head_h / 2, color=PALETTE["op_red_dark"], width_pt=1.25)
    add_line(slide, x_bus, py_add, x_bus, y_hm - 0.02, color=PALETTE["op_red_dark"], width_pt=1.45, arrow_end="triangle", arrow_size="sm")

    # -------------------------------------------------------------------------
    # SUB-BLOCK 2: Relation 2 `required-by` (K2 -> O3)
    # -------------------------------------------------------------------------
    r2_x, r2_y, r2_w, r2_h = 3.66, 5.20, 6.26, 0.96
    add_box(slide, r2_x, r2_y, r2_w, r2_h, fill=rgb("#FBF9FD"), border=PALETTE["sk_dark"], border_pt=1.4, rounded=True, round_radius=0.05)
    add_box(
        slide, r2_x + 0.12, r2_y - 0.14, 3.16, 0.28,
        [[("Relation 2: required-by  (K\u2082 \u2192 O\u2083)", "bsans")]],
        fill=SMIT_LAV, border=PALETTE["sk_dark"], border_pt=1.15, rounded=True, round_radius=0.22,
        color=PALETTE["sk_dark"], size_pt=10.4, margin_lr=0.02, margin_tb=0.0,
    )
    y_r2c = r2_y + 0.54
    add_box(slide, r2_x + 0.12, y_r2c - 0.20, 0.62, 0.40, [[("h", "bi"), ("K2", "b_")]], fill=SMIT_LAV, border=BLACK, border_pt=1.25, color=PALETTE["sk_dark"], size_pt=11.6)
    add_line(slide, r2_x + 0.74, y_r2c, r2_x + 0.90, y_r2c, color=BLACK, width_pt=1.25, arrow_end="triangle", arrow_size="sm")
    add_box(slide, r2_x + 0.92, y_r2c - 0.20, 1.54, 0.40, [[("W", "b"), ("s", "b_"), (" h", "bi"), ("K2", "b_"), (" + W", "b"), ("t", "b_"), (" h", "bi"), ("3", "b_")]], fill=SMIT_MINT, border=BLACK, border_pt=1.25, rounded=True, round_radius=0.10, color=BLACK, size_pt=10.8)
    add_line(slide, r2_x + 2.46, y_r2c, r2_x + 2.62, y_r2c, color=BLACK, width_pt=1.25, arrow_end="triangle", arrow_size="sm")
    add_box(slide, r2_x + 2.64, y_r2c - 0.24, 1.54, 0.48, [[("GATv2 (4 heads)", "bsans", BLACK, 10.2)], [("softmax", ""), ("r", "_"), (": \u03b1 = 1.0", "")]], fill=SMIT_LAV, border=BLACK, border_pt=1.25, rounded=True, round_radius=0.10, color=BLACK, size_pt=10.0, line_spacing=1.02)
    add_line(slide, r2_x + 4.18, y_r2c, r2_x + 4.34, y_r2c, color=BLACK, width_pt=1.25, arrow_end="triangle", arrow_size="sm")
    add_box(slide, r2_x + 4.36, y_r2c - 0.20, 0.96, 0.40, [[("Head Mean", "bsans")]], fill=SMIT_PEACH, border=BLACK, border_pt=1.25, rounded=True, round_radius=0.10, color=BLACK, size_pt=10.2)
    add_line(slide, r2_x + 5.32, y_r2c, r2_x + 5.48, y_r2c, color=PALETTE["sk_dark"], width_pt=1.25, arrow_end="triangle", arrow_size="sm")
    add_box(slide, r2_x + 5.50, y_r2c - 0.20, 0.64, 0.40, [[("m", "b"), ("req", "b_")]], fill=SMIT_LAV, border=PALETTE["sk_dark"], border_pt=1.35, color=PALETTE["sk_dark"], size_pt=11.4)

    # -------------------------------------------------------------------------
    # SUB-BLOCK 3: Right Vertical Column
    # -------------------------------------------------------------------------
    rx = 10.42
    rw = 2.08
    rcx = rx + rw / 2.0

    add_box(
        slide, rx + 0.16, 1.10, 1.76, 0.44,
        [[("Target ", "bsans", PALETTE["op_red_dark"], 11.4), ("h", "bi", PALETTE["op_red_dark"], 12.6), ("3", "b_", PALETTE["op_red_dark"], 12.6), ("(\u2113\u22121)", "b^", PALETTE["op_red_dark"], 12.6)]],
        fill=rgb("#FADBD8"), border=BLACK, border_pt=1.45, rounded=True, round_radius=0.14,
    )

    y_rsum = 1.88
    add_box(
        slide, rx, y_rsum, rw, 0.62,
        [[("Relation Sum", "bsans", BLACK, 11.8)], [("\u03a3", "b", BLACK, 12.0), ("r\u2208\u211b", "b_", BLACK, 11.8), (" m", "b", BLACK, 12.0), ("3", "b_", BLACK, 12.0), ("(r,\u2113)", "b^", BLACK, 12.0)]],
        fill=SMIT_PEACH, border=BLACK, border_pt=1.5, rounded=True, round_radius=0.12, line_spacing=1.04,
    )
    add_polyline(slide, [(h1_x + hm_w, y_hm + h_hm / 2), (10.12, y_hm + h_hm / 2), (10.12, y_rsum + 0.20), (rx - 0.02, y_rsum + 0.20)], color=PALETTE["op_red_dark"], width_pt=1.55, arrow_end="triangle", arrow_size="sm")
    add_box(slide, 9.93, 1.74, 0.44, 0.24, [[("m", "b"), ("prec", "b_")]], fill=rgb("#FCFDFF"), border=None, font_name=FONT_SERIF, size_pt=10.4, color=PALETTE["op_red_dark"], margin_lr=0.0, margin_tb=0.0)
    add_polyline(slide, [(r2_x + r2_w, y_r2c), (10.26, y_r2c), (10.26, y_rsum + 0.44), (rx - 0.02, y_rsum + 0.44)], color=PALETTE["sk_dark"], width_pt=1.55, arrow_end="triangle", arrow_size="sm")
    add_box(slide, 10.04, 4.90, 0.42, 0.24, [[("m", "b"), ("req", "b_")]], fill=rgb("#FCFDFF"), border=None, font_name=FONT_SERIF, size_pt=10.4, color=PALETTE["sk_dark"], margin_lr=0.0, margin_tb=0.0)

    y_ln = 2.84
    add_line(slide, rcx, y_rsum + 0.62, rcx, y_ln - 0.02, color=BLACK, width_pt=1.4, arrow_end="triangle", arrow_size="sm")
    add_box(slide, rx, y_ln, rw, 0.48, [[("Post-Norm LayerNorm", "bsans")]], fill=SMIT_BLUE, border=BLACK, border_pt=1.45, rounded=True, round_radius=0.12, color=BLACK, size_pt=11.4)

    y_lr = 3.66
    add_line(slide, rcx, y_ln + 0.48, rcx, y_lr - 0.02, color=BLACK, width_pt=1.4, arrow_end="triangle", arrow_size="sm")
    add_box(slide, rx, y_lr, rw, 0.48, [[("LeakyReLU (0.1)", "bsans")]], fill=SMIT_BLUE, border=BLACK, border_pt=1.45, rounded=True, round_radius=0.12, color=BLACK, size_pt=11.4)

    y_res = 4.48
    add_line(slide, rcx, y_lr + 0.48, rcx, y_res - 0.02, color=BLACK, width_pt=1.4, arrow_end="triangle", arrow_size="sm")
    add_box(slide, rx, y_res, rw, 0.54, [[("Add Residual (+ h", "bsans"), ("3", "b_"), ("(\u2113\u22121)", "b^"), (")", "bsans")]], fill=SMIT_BLUE_HI, border=BLACK, border_pt=1.6, rounded=True, round_radius=0.12, color=BLACK, size_pt=11.2)

    x_sk = 12.68
    add_polyline(slide, [(rx + 1.92, 1.32), (x_sk, 1.32), (x_sk, y_res + 0.27), (rx + rw + 0.02, y_res + 0.27)], color=BLACK, width_pt=1.55, arrow_end="triangle", arrow_size="sm")
    add_angled_label(
        slide, x_sk - 0.90, 2.94, 2.04, 0.24,
        [[("Residual skip  ", "isans"), ("h", "bi"), ("3", "b_"), ("(\u2113\u22121)", "b^")]],
        angle_deg=90.0, font_name=FONT_SERIF, size_pt=10.6, color=BLACK,
    )

    add_line(slide, rcx, y_res + 0.54, rcx, 5.40, color=BLACK, width_pt=1.5, arrow_end="triangle", arrow_size="sm")
    add_box(slide, rx + 0.14, 5.42, 1.80, 0.50, [[("h", "bi"), ("3", "b_"), ("(\u2113)", "b^"), (" \u2208 \u211d", "b"), ("128", "b^")]], fill=SMIT_MINT, border=BLACK, border_pt=1.6, rounded=True, round_radius=0.12, color=BLACK, size_pt=13.6)

    # Bottom Formula Bar
    add_line(slide, zx + 0.16, 6.32, zx + zw - 0.16, 6.32, color=PALETTE["line_gray"], width_pt=0.95)
    add_textbox(
        slide, zx + 0.12, 6.36, zw - 0.24, 0.50,
        [[
            ("e", "bi"), ("u3", "b_"), ("(r,h)", "b^"), (" = a", "b"), ("r,h", "b_"), ("T", "b^"), (" LReLU(W", ""), ("s,r,h", "_"), (" h", "i"), ("u", "_"), (" + W", ""), ("t,r,h", "_"), (" h", "i"), ("3", "_"), (")", ""),
            ("  |  ", "", PALETTE["line_gray"]),
            ("m", "b"), ("3", "b_"), ("(r)", "b^"), (" = \u00bc \u03a3", ""), ("h", "_"), (" \u03a3", ""), ("u", "_"), (" \u03b1", ""), ("u3", "_"), ("(r,h)", "^"), (" W", ""), ("s,r,h", "_"), (" h", "i"), ("u", "_"),
            ("  |  ", "", PALETTE["line_gray"]),
            ("h", "bi"), ("3", "b_"), ("(\u2113)", "b^"), (" = h", ""), ("3", "_"), ("(\u2113\u22121)", "^"), (" + LReLU(LN(\u03a3", ""), ("r", "_"), (" m", "b"), ("3", "_"), ("(r)", "^"), ("))", ""),
        ]],
        font_name=FONT_SERIF, size_pt=13.2, color=BLACK, align=PP_ALIGN.CENTER,
    )

    set_notes(
        slide,
        "Module B3. Relation-specific GATv2 heterogeneous graph encoder (main_tsmc.tex Section IV-B; HeteroGATEncoder in models/hb_gat_pn.py: L = 5 layers, H = 4 heads, head-averaged multi-head aggregation, relation sum over R_in(v), Post-Norm LayerNorm + LeakyReLU(0.1) + residual). Refined v8 removes internal slide cross-reference '(B4)' and all PyTorch Geometric code arguments ('concat=False', 'aggr=sum', 'GATv2Conv'), aligning every term with the IEEE TSMC manuscript.\n"
        "Layout reference: Igor G. Smit et al., Neural Combinatorial Optimization for Stochastic Flexible Job Shop Scheduling Problems, AAAI 2025, Fig. 1.",
    )


def main():
    parser = argparse.ArgumentParser(description="Build Slide 8 (B3) for refined_v8.")
    parser.add_argument("--pptx", default=DEFAULT_PPTX_PATH, help="Target PPTX path.")
    args = parser.parse_args()
    prs = Presentation(args.pptx)
    build_b3_v8(prs.slides[7])  # Slide 8: B3
    prs.save(args.pptx)
    print("Updated Slide 8 (B3) in", args.pptx)


if __name__ == "__main__":
    main()
