"""Build Slide 2: Compact Fig. 2(a)-(b) (B1 + B5 Seamless Horizontal Fusion).

Left (a): Physical APAL Workcell photograph (gently cropped panoramic 1.75:1) ->
          Heterogeneous State Graph G_t = (O U M U W U K, E_t) with raw node
          feature vectors x_v, feeding directly rightward into Panel (b).
Right (b): Height-matched Dual-Tower HGP-PPO Architecture (Disjoint Actor theta
           and Critic phi 5x GATv2 encoders, Node-Type Attention Pooling,
           3-Tier Autoregressive Pointer, Scalar Value Head, and PPO Coupling),
           eliminating the redundant bottom G_t bar.

Aspect ratio ~2.24:1 matching IEEE TSMC double-column banner slot.
"""
from __future__ import annotations

import os
import sys
from PIL import Image
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
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
    add_swatch_strip,
    add_textbox,
    clear_slide,
    rgb,
    set_notes,
)

SRC_PHOTO = os.path.join(ASSETS_DIR, "b1_apal_workcell_v4_ccby2.png")
CROP_PHOTO = os.path.join(ASSETS_DIR, "b1_apal_workcell_compact.png")

PANEL_BDR     = rgb("#28445E")
DAN_ROSE_BG   = rgb("#FAF0F1")
DAN_TEAL_BG   = rgb("#EDF5F7")
DAN_LAV       = rgb("#E5DCEF")
DAN_CREAM     = rgb("#FDF1C8")
DAN_SAGE      = rgb("#D7E7CB")
DAN_PEACH     = rgb("#FDE3D1")
DAN_SKY       = rgb("#E5F3FA")
DAN_SK_OUT    = rgb("#683B91")
PPO_BRIDGE_BG = rgb("#F6EFFB")

TOP_CROP = 0.09
BOT_CROP = 0.06


def _ensure_panoramic_photo() -> tuple[str, int, int]:
    """Create panoramic cropped version of workcell photo (keeps all 4 targets intact)."""
    im = Image.open(SRC_PHOTO)
    w, h = im.size
    y0 = int(round(h * TOP_CROP))
    y1 = int(round(h * (1.0 - BOT_CROP)))
    cropped = im.crop((0, y0, w, y1))
    cropped.save(CROP_PHOTO, quality=98)
    return CROP_PHOTO, cropped.size[0], cropped.size[1]


def _set_fill_alpha(shape, alpha_pct: int = 93):
    spPr = shape._element.find(qn("p:spPr"))
    if spPr is None:
        return
    srgb = spPr.find(".//" + qn("a:srgbClr"))
    if srgb is not None:
        alpha = OxmlElement("a:alpha")
        alpha.set("val", str(alpha_pct * 1000))
        srgb.append(alpha)


def _add_pointer_call(slide, x1, y1, tx, ty, color):
    add_line(slide, x1, y1, tx, ty, color=PALETTE["white"], width_pt=3.6)
    add_line(slide, x1, y1, tx, ty, color=color, width_pt=2.0, arrow_end="triangle", arrow_size="sm")
    add_circle(slide, tx, ty, 0.072, fill=color, border=PALETTE["white"], border_pt=1.35)


def build_fig2_compact_b1_b5(slide):
    """Build Compact Integrated Fig. 2(a)-(b) (B1 + B5) with seamless G_t -> f_X fusion."""
    clear_slide(slide)

    photo_path, pw_px, ph_px = _ensure_panoramic_photo()

    y_top, pan_h = 0.78, 5.92   # [0.78 .. 6.70]
    y_bot = y_top + pan_h

    # =========================================================================
    # PANEL (a): PHYSICAL WORKCELL -> HETEROGENEOUS STATE GRAPH G_t [w = 5.94 in]
    # =========================================================================
    ax0, aw = 0.22, 5.94        # [0.22 .. 6.16]
    add_box(slide, ax0, y_top, aw, pan_h, fill=rgb("#FAFBFD"), border=PANEL_BDR, border_pt=1.55, rounded=True, round_radius=0.025)
    add_textbox(
        slide, ax0 + 0.06, y_top + 0.05, aw - 0.12, 0.28,
        [[("(a) Physical APAL Workcell to Heterogeneous State Graph ", "bsans", PALETTE["accent_navy"], 11.8), ("G", "bi", PALETTE["accent_navy"], 12.6), ("t", "b_", PALETTE["accent_navy"], 12.6)]],
        font_name=FONT_SERIF, size_pt=12.0, align=PP_ALIGN.CENTER, margin_lr=0.0, word_wrap=False,
    )
    add_line(slide, ax0 + 0.08, y_top + 0.36, ax0 + aw - 0.08, y_top + 0.36, color=rgb("#D5DFE8"), width_pt=0.85)

    # Top inside (a): Panoramic 787 assembly photo [w = 5.74, h = 3.28]
    photo_x = ax0 + 0.10
    photo_w = aw - 0.20          # 5.74 in
    photo_h = photo_w * (ph_px / pw_px)  # ~3.285 in
    photo_y = y_top + 0.42       # 1.20 .. 4.485

    slide.shapes.add_picture(
        photo_path, Inches(photo_x), Inches(photo_y), width=Inches(photo_w), height=Inches(photo_h),
    )
    add_box(slide, photo_x, photo_y, photo_w, photo_h, fill=None, border=rgb("#7B8896"), border_pt=0.95)

    def pxy(u: float, v_orig: float):
        v_crop = (v_orig - TOP_CROP) / (1.0 - TOP_CROP - BOT_CROP)
        return (photo_x + u * photo_w, photo_y + v_crop * photo_h)

    # Four snug paper-faithful callouts inside photo
    tx_s3, ty_s3 = pxy(0.205, 0.335)
    _add_pointer_call(slide, tx_s3, photo_y + 0.56, tx_s3, ty_s3, PALETTE["st_dark"])
    sh_s3 = add_box(
        slide, photo_x + 0.08, photo_y + 0.08, 2.28, 0.50,
        [
            [("S", "b", PALETTE["st_dark"], 11.8), ("3", "b_", PALETTE["st_dark"], 11.8), (" Physical Station", "bsans", BLACK, 10.6)],
            [("x", "bi", PALETTE["st_dark"], 9.8), ("M", "b_", PALETTE["st_dark"], 9.8), (": L", "bsans", rgb("#1F2933"), 9.2), ("m", "b_", rgb("#1F2933"), 9.2), (", ratios, wait, c", "bsans", rgb("#1F2933"), 9.2), ("m", "b_", rgb("#1F2933"), 9.2)],
        ],
        fill=rgb("#DFF3EE"), border=PALETTE["st_dark"], border_pt=1.55, rounded=True, round_radius=0.20,
        line_spacing=1.00, margin_lr=0.03, margin_tb=0.01,
    )
    _set_fill_alpha(sh_s3, 94)

    tx_o3, ty_o3 = pxy(0.675, 0.375)
    _add_pointer_call(slide, tx_o3, photo_y + 0.56, tx_o3, ty_o3, PALETTE["op_red_dark"])
    sh_o3 = add_box(
        slide, photo_x + 2.84, photo_y + 0.08, 2.80, 0.50,
        [
            [("O", "b", PALETTE["op_red_dark"], 11.8), ("3", "b_", PALETTE["op_red_dark"], 11.8), (" Operation (wing join, q=2)", "bsans", BLACK, 10.4)],
            [("x", "bi", PALETTE["op_red_dark"], 9.8), ("O", "b_", PALETTE["op_red_dark"], 9.8), (": 18d/24d (p\u0303", "bsans", rgb("#1F2933"), 9.2), ("i", "b_", rgb("#1F2933"), 9.2), (", status, k", "bsans", rgb("#1F2933"), 9.2), ("i", "b_", rgb("#1F2933"), 9.2), (", r", "bsans", rgb("#1F2933"), 9.2), ("i", "b_", rgb("#1F2933"), 9.2), (")", "bsans", rgb("#1F2933"), 9.2)],
        ],
        fill=rgb("#FCE3E0"), border=PALETTE["op_red_dark"], border_pt=1.55, rounded=True, round_radius=0.20,
        line_spacing=1.00, margin_lr=0.03, margin_tb=0.01,
    )
    _set_fill_alpha(sh_o3, 94)

    tx_k2, ty_k2 = pxy(0.268, 0.615)
    _add_pointer_call(slide, photo_x + 1.18, photo_y + photo_h - 0.56, tx_k2, ty_k2, PALETTE["sk_dark"])
    sh_k2 = add_box(
        slide, photo_x + 0.08, photo_y + photo_h - 0.58, 2.26, 0.50,
        [
            [("K", "b", PALETTE["sk_dark"], 11.8), ("2", "b_", PALETTE["sk_dark"], 11.8), (" Skill Node (|K|=5)", "bsans", BLACK, 10.6)],
            [("x", "bi", PALETTE["sk_dark"], 9.8), ("K", "b_", PALETTE["sk_dark"], 9.8), (": ID, count, cap., mean \u03b7", "bsans", rgb("#1F2933"), 9.0)],
        ],
        fill=rgb("#ECE2F7"), border=PALETTE["sk_dark"], border_pt=1.55, rounded=True, round_radius=0.20,
        line_spacing=1.00, margin_lr=0.03, margin_tb=0.01,
    )
    _set_fill_alpha(sh_k2, 94)

    tx_w5, ty_w5 = pxy(0.803, 0.805)
    _add_pointer_call(slide, photo_x + 4.44, photo_y + photo_h - 0.34, tx_w5, ty_w5, PALETTE["wk_dark"])
    sh_w5 = add_box(
        slide, photo_x + 2.40, photo_y + photo_h - 0.58, 2.08, 0.50,
        [
            [("W", "b", PALETTE["wk_dark"], 11.8), ("5", "b_", PALETTE["wk_dark"], 11.8), (" Multi-Skilled Worker", "bsans", BLACK, 10.2)],
            [("x", "bi", PALETTE["wk_dark"], 9.8), ("W", "b_", PALETTE["wk_dark"], 9.8), (": \u03b7", "bsans", rgb("#1F2933"), 9.0), ("w", "b_", rgb("#1F2933"), 9.0), (", 5-d skills, wait, \u2113", "bsans", rgb("#1F2933"), 9.0), ("w", "b_", rgb("#1F2933"), 9.0)],
        ],
        fill=rgb("#FBEBD4"), border=PALETTE["wk_dark"], border_pt=1.55, rounded=True, round_radius=0.20,
        line_spacing=1.00, margin_lr=0.02, margin_tb=0.01,
    )
    _set_fill_alpha(sh_w5, 94)

    # Short downward transition arrow Photo -> State Graph G_t
    arr_dn = slide.shapes.add_shape(
        MSO_SHAPE.DOWN_ARROW,
        Inches(ax0 + aw / 2.0 - 0.22),
        Inches(photo_y + photo_h + 0.03),
        Inches(0.44),
        Inches(0.22),
    )
    arr_dn.adjustments[0] = 0.46
    arr_dn.adjustments[1] = 0.52
    arr_dn.fill.solid(); arr_dn.fill.fore_color.rgb = rgb("#DED3EA")
    arr_dn.line.color.rgb = rgb("#5B4A73"); arr_dn.line.width = Pt(1.25); arr_dn.shadow.inherit = False

    # Bottom inside (a): Horizontal Heterogeneous State Graph G_t [y = 4.76 .. 6.60, h = 1.84 in]
    gx0 = photo_x
    gy0 = photo_y + photo_h + 0.28   # ~4.765
    gw  = photo_w                    # 5.74 in
    gh  = (y_bot - 0.10) - gy0       # ~1.835 in

    add_box(slide, gx0, gy0, gw, gh, fill=rgb("#EBF2FA"), border=rgb("#3F638E"), border_pt=1.55, rounded=True, round_radius=0.06)
    add_textbox(
        slide, gx0 + 0.08, gy0 + 0.04, 3.88, 0.24,
        [[
            ("State Graph  ", "bsans", BLACK, 10.6),
            ("G", "bi", BLACK, 11.4), ("t", "b_", BLACK, 11.4),
            (" = (", "b", BLACK, 11.0),
            ("O", "bi", PALETTE["op_red_dark"], 11.2), (" \u222a ", "b", PALETTE["ink_soft"], 10.2),
            ("M", "bi", PALETTE["st_dark"], 11.2), (" \u222a ", "b", PALETTE["ink_soft"], 10.2),
            ("W", "bi", PALETTE["wk_dark"], 11.2), (" \u222a ", "b", PALETTE["ink_soft"], 10.2),
            ("K", "bi", PALETTE["sk_dark"], 11.2), (", E", "bi", BLACK, 11.0), ("t", "b_", BLACK, 11.0), (")", "b", BLACK, 11.0),
        ]],
        font_name=FONT_SERIF, size_pt=10.8, align=PP_ALIGN.LEFT, margin_lr=0.0, word_wrap=False,
    )
    add_textbox(
        slide, gx0 + 4.18, gy0 + 0.04, 1.48, 0.24,
        [[("Raw features ", "bsans", BLACK, 9.6), ("x", "bi", BLACK, 10.6), ("v", "b_", BLACK, 10.6)]],
        font_name=FONT_SERIF, size_pt=10.0, align=PP_ALIGN.CENTER, margin_lr=0.0, word_wrap=False,
    )

    # Clean horizontal 8-node heterogeneous graph on left-center of G_t card (zero overlap with bottom legend!)
    rg = 0.175
    n_s2 = (gx0 + 0.34, gy0 + 0.50)
    n_s3 = (gx0 + 1.56, gy0 + 0.50)
    n_o1 = (gx0 + 0.34, gy0 + 1.08)
    n_o2 = (gx0 + 0.96, gy0 + 1.32)
    n_o3 = (gx0 + 1.56, gy0 + 1.08)
    n_k2 = (gx0 + 2.66, gy0 + 1.08)
    n_w2 = (gx0 + 2.66, gy0 + 0.50)
    n_w5 = (gx0 + 3.56, gy0 + 1.08)

    # Typed edges (exact Table III relation labels)
    add_line(slide, n_o1[0], n_o1[1] - rg, n_s2[0], n_s2[1] + rg + 0.015, color=PALETTE["st_dark"], width_pt=1.65, dash="dash", arrow_end="triangle", arrow_size="sm")
    add_line(slide, n_o1[0] + rg, n_o1[1], n_o3[0] - rg - 0.015, n_o3[1], color=PALETTE["op_red_dark"], width_pt=1.65, arrow_end="triangle", arrow_size="sm")
    add_line(slide, n_o2[0] + 0.16, n_o2[1] - 0.07, n_o3[0] - 0.16, n_o3[1] + 0.07, color=PALETTE["op_red_dark"], width_pt=1.65, arrow_end="triangle", arrow_size="sm")
    add_line(slide, n_o3[0] + rg + 0.015, n_o3[1], n_k2[0] - rg - 0.015, n_k2[1], color=PALETTE["sk_dark"], width_pt=1.7, arrow_start="triangle", arrow_end="triangle", arrow_size="sm")
    add_line(slide, n_k2[0], n_k2[1] - rg - 0.015, n_w2[0], n_w2[1] + rg + 0.015, color=PALETTE["wk_dark"], width_pt=1.6, arrow_start="triangle", arrow_end="triangle", arrow_size="sm")
    add_line(slide, n_k2[0] + rg + 0.015, n_k2[1], n_w5[0] - rg - 0.015, n_w5[1], color=PALETTE["wk_dark"], width_pt=1.6, arrow_start="triangle", arrow_end="triangle", arrow_size="sm")

    add_circle(slide, n_s2[0], n_s2[1], rg, [[("S", "b"), ("2", "b_")]], fill=rgb("#DFF3EE"), border=BLACK, border_pt=1.25, color=PALETTE["st_dark"], size_pt=10.6)
    add_circle(slide, n_s3[0], n_s3[1], rg, [[("S", "b"), ("3", "b_")]], fill=PALETTE["st_dark"], border=BLACK, border_pt=1.75, color=PALETTE["white"], size_pt=10.6)
    add_circle(slide, n_o1[0], n_o1[1], rg, [[("O", "b"), ("1", "b_")]], fill=rgb("#FCE3E0"), border=BLACK, border_pt=1.25, color=PALETTE["op_red_dark"], size_pt=10.6)
    add_circle(slide, n_o2[0], n_o2[1], rg, [[("O", "b"), ("2", "b_")]], fill=rgb("#FCE3E0"), border=BLACK, border_pt=1.25, color=PALETTE["op_red_dark"], size_pt=10.6)
    add_circle(slide, n_o3[0], n_o3[1], rg, [[("O", "b"), ("3", "b_")]], fill=PALETTE["op_red_dark"], border=BLACK, border_pt=1.75, color=PALETTE["white"], size_pt=10.6)
    add_circle(slide, n_k2[0], n_k2[1], rg, [[("K", "b"), ("2", "b_")]], fill=PALETTE["sk_dark"], border=BLACK, border_pt=1.75, color=PALETTE["white"], size_pt=10.6)
    add_circle(slide, n_w2[0], n_w2[1], rg, [[("W", "b"), ("2", "b_")]], fill=rgb("#FBEBD4"), border=BLACK, border_pt=1.25, color=PALETTE["wk_dark"], size_pt=10.6)
    add_circle(slide, n_w5[0], n_w5[1], rg, [[("W", "b"), ("5", "b_")]], fill=PALETTE["wk_dark"], border=BLACK, border_pt=1.75, color=PALETTE["white"], size_pt=10.6)

    # Crisp Table III edge labels
    add_textbox(slide, n_s2[0] + 0.04, gy0 + 0.69, 0.84, 0.20, [[("assigned-to", "bsans")]], font_name=FONT_SANS, size_pt=8.8, color=PALETTE["st_dark"], align=PP_ALIGN.LEFT, margin_lr=0.0, word_wrap=False)
    add_textbox(slide, n_s3[0] - 0.44, n_s3[1] + rg + 0.005, 0.88, 0.18, [[("open station", "isans")]], font_name=FONT_SANS, size_pt=8.6, color=PALETTE["st_dark"], align=PP_ALIGN.CENTER, margin_lr=0.0, word_wrap=False)
    add_textbox(slide, gx0 + 0.56, n_o1[1] - 0.21, 0.78, 0.20, [[("precedes", "bsans")]], font_name=FONT_SANS, size_pt=9.0, color=PALETTE["op_red_dark"], align=PP_ALIGN.CENTER, margin_lr=0.0, word_wrap=False)
    add_textbox(slide, gx0 + 1.76, n_o3[1] - 0.21, 0.70, 0.20, [[("requires", "bsans")]], font_name=FONT_SANS, size_pt=9.0, color=PALETTE["sk_dark"], align=PP_ALIGN.CENTER, margin_lr=0.0, word_wrap=False)
    add_textbox(slide, gx0 + 2.74, gy0 + 0.69, 0.74, 0.20, [[("has-skill", "bsans")]], font_name=FONT_SANS, size_pt=9.0, color=PALETTE["wk_dark"], align=PP_ALIGN.LEFT, margin_lr=0.0, word_wrap=False)

    # Compact visual legend along very bottom-left of G_t (0.12 in clear below O_2!)
    add_textbox(
        slide, gx0 + 0.10, gy0 + gh - 0.22, 3.86, 0.20,
        [[("Filled = traced in photo   |   Solid: static   |   Dashed: dynamic assign.", "bsans")]],
        font_name=FONT_SANS, size_pt=8.6, color=rgb("#334150"), align=PP_ALIGN.LEFT, margin_lr=0.0, word_wrap=False,
    )

    # Right column inside G_t card: 4 compact feature vector swatches x_M, x_O, x_K, x_W
    add_line(slide, gx0 + 4.02, gy0 + 0.30, gx0 + 4.02, gy0 + gh - 0.10, color=rgb("#B9CBE0"), width_pt=0.95, dash="dot")
    vx = gx0 + 4.14
    vec_rows = [
        ("x", "M", PALETTE["st_dark"],     [rgb("#84BEB4"), rgb("#1D6F63"), rgb("#C5E2DD"), rgb("#4B998C")]),
        ("x", "O", PALETTE["op_red_dark"], [rgb("#D88E87"), rgb("#A63D33"), rgb("#EBC4BF"), rgb("#C4665D")]),
        ("x", "K", PALETTE["sk_dark"],     [rgb("#B69AC9"), rgb("#6D458E"), rgb("#DCCEE8"), rgb("#916BB0")]),
        ("x", "W", PALETTE["wk_dark"],     [rgb("#DFB182"), rgb("#A85C1E"), rgb("#F0D7BC"), rgb("#C98443")]),
    ]
    for r_idx, (vs, vsub, vcol, sws) in enumerate(vec_rows):
        vy = gy0 + 0.36 + r_idx * 0.34
        add_textbox(
            slide, vx, vy, 0.42, 0.24,
            [[(vs, "bi"), (vsub, "b_")]],
            font_name=FONT_SERIF, size_pt=10.6, color=vcol, align=PP_ALIGN.RIGHT, margin_lr=0.0, word_wrap=False,
        )
        add_swatch_strip(slide, vx + 0.48, vy + 0.02, sws, cell_w=0.22, cell_h=0.20, gap=0.015, border=BLACK, border_pt=0.65)

    # =========================================================================
    # SEAMLESS HORIZONTAL TRANSITION G_t (Panel a) -> f_X Bus (Panel b)
    # =========================================================================
    bx0, bw = 6.54, 6.58        # [6.54 .. 13.12]
    add_box(slide, bx0, y_top, bw, pan_h, fill=PALETTE["white"], border=PANEL_BDR, border_pt=1.55, rounded=True, round_radius=0.025)
    add_textbox(
        slide, bx0 + 0.06, y_top + 0.05, bw - 0.12, 0.28,
        [[("(b) Independent Dual-Tower HGP-PPO Architecture (Actor ", "bsans", PALETTE["accent_navy"], 11.8), ("\u03b8", "bi", PALETTE["accent_navy"], 12.6), (" & Critic ", "bsans", PALETTE["accent_navy"], 11.8), ("\u03c6", "bi", PALETTE["accent_navy"], 12.6), (")", "bsans", PALETTE["accent_navy"], 11.8)]],
        font_name=FONT_SERIF, size_pt=12.0, align=PP_ALIGN.CENTER, margin_lr=0.0, word_wrap=False,
    )
    add_line(slide, bx0 + 0.08, y_top + 0.36, bx0 + bw - 0.08, y_top + 0.36, color=rgb("#D5DFE8"), width_pt=0.85)

    # Rightward block transition arrow from Panel (a) G_t right edge -> Panel (b) bottom f_X bus!
    y_bus = 6.32
    arr_r = slide.shapes.add_shape(
        MSO_SHAPE.RIGHT_ARROW,
        Inches(ax0 + aw + 0.04),
        Inches(y_bus - 0.19),
        Inches(0.30),
        Inches(0.38),
    )
    arr_r.adjustments[0] = 0.48
    arr_r.adjustments[1] = 0.50
    arr_r.fill.solid(); arr_r.fill.fore_color.rgb = rgb("#DED3EA")
    arr_r.line.color.rgb = rgb("#5B4A73"); arr_r.line.width = Pt(1.4); arr_r.shadow.inherit = False
    add_textbox(
        slide, ax0 + aw - 0.02, y_bus - 0.44, 0.44, 0.24,
        [[("G", "bi"), ("t", "b_")]],
        font_name=FONT_SERIF, size_pt=11.6, color=rgb("#5B4A73"), align=PP_ALIGN.CENTER, margin_lr=0.0, word_wrap=False,
    )

    # =========================================================================
    # PANEL (b): HEIGHT-MATCHED DUAL-TOWER HGP-PPO (NO DUPLICATE G_t BAR!)
    # Generous side pockets so `x 5`, `skip`, `Actor/Critic Encoder`, and `H_theta^(5)` have zero crowding!
    # =========================================================================
    env_x0 = bx0 + 0.10         # 6.64
    env_w  = bw - 0.20          # 6.38  -> [6.64 .. 13.02]
    env_x1 = env_x0 + env_w

    card_w  = 2.18
    inner_w = 1.98
    card_L  = env_x0 + 0.50     # 7.14 .. 9.32
    cx_L    = card_L + card_w / 2.0   # 8.23
    x_div   = env_x0 + env_w / 2.0    # 9.83
    card_R  = x_div + 0.42      # 10.25 .. 12.43
    cx_R    = card_R + card_w / 2.0   # 11.34

    # --- TOP POLICY & VALUE HEADS + PPO COUPLING BRIDGE [y = 1.22 .. 2.18] ---
    y_ptr, h_ptr = 1.71, 0.33
    pw_p, pgap   = 0.78, 0.08
    p_tot = 3 * pw_p + 2 * pgap       # 2.50 in
    p_x0  = cx_L - p_tot / 2.0        # 6.98 .. 9.48

    grp_L_x0 = p_x0 - 0.06
    grp_y0   = y_ptr - 0.06           # 1.65
    grp_L_w  = p_tot + 0.12
    grp_h    = h_ptr + 0.12
    add_box(slide, grp_L_x0, grp_y0, grp_L_w, grp_h, fill=rgb("#FDF7F7"), border=DAN_SK_OUT, border_pt=1.25, dash="dash", rounded=True, round_radius=0.18)

    ptr_defs = [
        ([("Op Ptr ", "bsans"), ("i", "bi"), ("t", "b_")], rgb("#FADBD8")),
        ([("St Ptr ", "bsans"), ("m", "bi"), ("t", "b_")], rgb("#D6EEF3")),
        ([("Wk Ptr ", "bsans"), ("W", "b"), ("t", "b_")], rgb("#FCE7CF")),
    ]
    for idx, (psegs, pfill) in enumerate(ptr_defs):
        px = p_x0 + idx * (pw_p + pgap)
        sh_p = add_box(
            slide, px, y_ptr, pw_p, h_ptr, [psegs],
            fill=pfill, border=BLACK, border_pt=1.35, rounded=True, round_radius=0.20,
            font_name=FONT_SERIF, color=BLACK, size_pt=9.6, margin_lr=0.0, margin_tb=0.0,
        )
        sh_p.text_frame.word_wrap = False
        if idx < 2:
            add_line(slide, px + pw_p + 0.01, y_ptr + h_ptr / 2, px + pw_p + pgap - 0.015, y_ptr + h_ptr / 2, color=BLACK, width_pt=1.3, arrow_end="triangle", arrow_size="sm")

    mlp_x0, mlp_w = cx_R - 1.24, 1.26
    v_x0, v_w     = cx_R + 0.16, 1.08
    grp_R_x0      = mlp_x0 - 0.06
    grp_R_w       = (v_x0 + v_w + 0.06) - grp_R_x0
    add_box(slide, grp_R_x0, grp_y0, grp_R_w, grp_h, fill=rgb("#F8F4FC"), border=DAN_SK_OUT, border_pt=1.25, dash="dash", rounded=True, round_radius=0.18)
    add_box(slide, mlp_x0, y_ptr, mlp_w, h_ptr, [[("MLP (3d\u219264\u21921)", "bsans")]], fill=DAN_LAV, border=BLACK, border_pt=1.35, rounded=True, round_radius=0.16, color=BLACK, size_pt=9.6, margin_lr=0.0, margin_tb=0.0)
    add_line(slide, mlp_x0 + mlp_w, y_ptr + h_ptr / 2, v_x0 - 0.015, y_ptr + h_ptr / 2, color=BLACK, width_pt=1.3, arrow_end="triangle", arrow_size="sm")
    add_box(slide, v_x0, y_ptr, v_w, h_ptr, [[("V", "b"), ("\u03c6", "b_"), ("(s", "b"), ("t", "b_"), (") \u2208 \u211d", "b")]], fill=rgb("#E9DCF4"), border=BLACK, border_pt=1.45, rounded=True, round_radius=0.20, color=DAN_SK_OUT, size_pt=10.6, margin_lr=0.0, margin_tb=0.0)

    # Top PPO Coupling Bridge (clean clearance above Policy & Value labels!)
    y_bridge = 1.32
    add_polyline(
        slide,
        [(cx_R, grp_y0), (cx_R, y_bridge), (cx_L, y_bridge), (cx_L, grp_y0 - 0.01)],
        color=DAN_SK_OUT, width_pt=2.1, dash="dash", arrow_end="triangle", arrow_size="sm",
    )
    pill_w = 2.74
    add_box(
        slide, x_div - pill_w / 2.0, y_bridge - 0.145, pill_w, 0.29,
        [[("PPO Coupling: ", "bsans", DAN_SK_OUT, 9.8), ("GAE ", "bsans", BLACK, 9.6), ("\u00c2", "bi", DAN_SK_OUT, 10.6), ("t", "b_", DAN_SK_OUT, 10.6), (" \u2192 \u2207", "b", DAN_SK_OUT, 10.2), ("\u03b8", "b_", DAN_SK_OUT, 10.2)]],
        fill=PPO_BRIDGE_BG, border=DAN_SK_OUT, border_pt=1.45, rounded=True, round_radius=0.30, font_name=FONT_SERIF, margin_lr=0.01, margin_tb=0.0,
    )
    add_textbox(
        slide, grp_L_x0, 1.39, (cx_L - grp_L_x0) - 0.06, 0.22,
        [[("Policy ", "bsans", DAN_SK_OUT, 9.6), ("\u03c0", "bi", DAN_SK_OUT, 10.4), ("\u03b8", "b_", DAN_SK_OUT, 10.4), ("(\u00e3", "b", DAN_SK_OUT, 9.8), ("t", "b_", DAN_SK_OUT, 9.8), ("|s", "b", DAN_SK_OUT, 9.8), ("t", "b_", DAN_SK_OUT, 9.8), (")", "b", DAN_SK_OUT, 9.8)]],
        font_name=FONT_SERIF, size_pt=9.8, align=PP_ALIGN.LEFT, margin_lr=0.0, word_wrap=False,
    )
    add_textbox(
        slide, cx_R + 0.08, 1.39, (grp_R_x0 + grp_R_w) - (cx_R + 0.08), 0.22,
        [[("Value ", "bsans", DAN_SK_OUT, 9.6), ("V", "b", DAN_SK_OUT, 10.4), ("\u03c6", "b_", DAN_SK_OUT, 10.4), ("(s", "b", DAN_SK_OUT, 9.8), ("t", "b_", DAN_SK_OUT, 9.8), (")", "b", DAN_SK_OUT, 9.8)]],
        font_name=FONT_SERIF, size_pt=9.8, align=PP_ALIGN.RIGHT, margin_lr=0.0, word_wrap=False,
    )

    # --- POOLING & PARTITIONED CONTEXT g_theta / g_phi [y = 2.25 .. 2.86] ---
    y_g    = 2.25
    y_pool = 2.59
    for cx_t, sym in [(cx_L, "\u03b8"), (cx_R, "\u03c6")]:
        add_box(
            slide, cx_t - 1.15, y_pool, 2.30, 0.25,
            [[("Attn. Pooling  [M | O | W]", "bsans")]],
            fill=rgb("#F1F8FC"), border=BLACK, border_pt=1.2, rounded=True, round_radius=0.22,
            color=BLACK, size_pt=9.6, margin_lr=0.01, margin_tb=0.0,
        )
        add_line(slide, cx_t, y_pool, cx_t, y_g + 0.24, color=BLACK, width_pt=1.25, arrow_end="triangle", arrow_size="sm")
        add_partitioned_vector(
            slide, cx_t - 0.88, y_g,
            [[("g", "b"), ("M", "b^")], [("g", "b"), ("O", "b^")], [("g", "b"), ("W", "b^")]],
            cell_w=0.45, cell_h=0.24,
            fills=[rgb("#D8EFF4"), rgb("#FADBD8"), rgb("#FBE5CC")],
            border=BLACK, border_pt=1.2, size_pt=10.2,
        )
        add_textbox(
            slide, cx_t + 0.51, y_g, 0.78, 0.24,
            [[("g", "bi"), (sym, "b_"), (" \u2208 \u211d", "b"), ("3d", "b^")]],
            font_name=FONT_SERIF, size_pt=10.2, color=BLACK, align=PP_ALIGN.LEFT, margin_lr=0.0, word_wrap=False,
        )

    add_polyline(slide, [(cx_L, y_g), (cx_L, 2.14), (p_x0 + pw_p / 2, 2.14), (p_x0 + pw_p / 2, y_ptr + h_ptr + 0.01)], color=BLACK, width_pt=1.3, arrow_end="triangle", arrow_size="sm")
    add_polyline(slide, [(cx_R, y_g), (cx_R, 2.14), (mlp_x0 + mlp_w / 2, 2.14), (mlp_x0 + mlp_w / 2, y_ptr + h_ptr + 0.01)], color=BLACK, width_pt=1.3, arrow_end="triangle", arrow_size="sm")

    # --- OUTER DOTTED ENCLOSURE (x 5 LAYERS) [y = 2.95 .. 5.90, h = 2.95 in] ---
    enc_y0, enc_h = 2.95, 2.95
    add_box(slide, env_x0, enc_y0, env_w, enc_h, fill=PALETTE["white"], border=BLACK, border_pt=1.45, dash="dot", rounded=True, round_radius=0.05)
    # `x 5` positioned 100% INSIDE upper-right pocket with generous clearance!
    add_textbox(slide, env_x1 - 0.52, enc_y0 + 0.04, 0.46, 0.28, [[("\u00d7 5", "bi")]], font_name=FONT_SERIF, size_pt=14.4, color=BLACK, align=PP_ALIGN.CENTER, margin_lr=0.0)

    # Disjoint weights center divider (pill y=3.28..5.54, center y=4.41)
    add_line(slide, x_div, enc_y0 + 0.10, x_div, enc_y0 + enc_h - 0.10, color=PALETTE["line_gray"], width_pt=1.0, dash="dash")
    add_box(slide, x_div - 0.15, 3.28, 0.30, 2.26, fill=PALETTE["white"], border=BLACK, border_pt=1.25, rounded=True, round_radius=0.38)
    add_angled_label(slide, x_div - 1.12, 4.29, 2.24, 0.24, "Disjoint  (\u03b8 \u2229 \u03c6 = \u2205)", angle_deg=-90.0, size_pt=9.6, color=BLACK, bold=True)

    y_res_bar  = 3.05
    y_hm_bar   = 3.45
    y_card_top = 3.85
    card_h     = 1.71

    for bx_c, cx_t, bg_tint, side_lbl, is_actor in [
        (card_L, cx_L, DAN_ROSE_BG, "Actor Encoder (\u03b8)", True),
        (card_R, cx_R, DAN_TEAL_BG, "Critic Encoder (\u03c6)", False),
    ]:
        for k in (2, 1):
            add_box(slide, bx_c - k * 0.065, y_card_top + k * 0.06, card_w, card_h, fill=bg_tint, border=rgb("#98A2AC"), border_pt=0.95, rounded=True, round_radius=0.06)
            stem_x = bx_c - k * 0.065 + 0.08
            add_line(slide, stem_x, y_card_top + k * 0.06, stem_x, y_hm_bar + 0.28, color=BLACK, width_pt=1.0)

        add_box(slide, bx_c, y_card_top, card_w, card_h, fill=bg_tint, border=BLACK, border_pt=1.45, rounded=True, round_radius=0.06)
        add_textbox(slide, bx_c - 0.14, y_card_top + card_h + 0.12, 0.86, 0.20, [[("4 heads", "bsans")]], font_name=FONT_SANS, size_pt=9.4, color=BLACK, align=PP_ALIGN.LEFT, margin_lr=0.0, word_wrap=False)

        if is_actor:
            add_angled_label(slide, env_x0 + 0.18 - 1.12, 4.38, 2.24, 0.26, side_lbl, angle_deg=-90.0, size_pt=11.0, color=BLACK, bold=True)
        else:
            add_angled_label(slide, env_x1 - 0.20 - 1.12, 4.50, 2.24, 0.26, side_lbl, angle_deg=90.0, size_pt=11.0, color=BLACK, bold=True)

        px_in = cx_t - inner_w / 2.0
        ph = 0.265
        y_l1 = 5.22
        y_l2 = 4.56
        y_gl = 5.0225
        y_l3 = 4.24
        y_l4 = 3.92

        add_box(slide, px_in, y_l1, inner_w, ph, [[("Linear  W", "bsans"), ("s", "b_"), (" h", "bi"), ("u", "b_"), (" + W", "bsans"), ("t", "b_"), (" h", "bi"), ("v", "b_")]], fill=DAN_LAV, border=BLACK, border_pt=1.15, rounded=True, round_radius=0.20, color=BLACK, size_pt=9.4, margin_lr=0.0, margin_tb=0.0)
        add_line(slide, cx_t, y_l1, cx_t, y_gl + 0.10, color=BLACK, width_pt=1.15)
        add_activation_glyph(slide, cx_t, y_gl, r=0.10, kind="lrelu", fill=PALETTE["white"], border=BLACK, border_pt=1.15, curve_pt=1.1)
        add_textbox(slide, cx_t + 0.13, y_gl - 0.10, 0.96, 0.20, [[("LReLU (0.1)", "bisans")]], font_name=FONT_SANS, size_pt=9.2, color=rgb("#1F2933"), align=PP_ALIGN.LEFT, margin_lr=0.0, margin_tb=0.0, word_wrap=False)
        add_line(slide, cx_t, y_gl - 0.10, cx_t, y_l2 + ph, color=BLACK, width_pt=1.15)
        add_box(slide, px_in, y_l2, inner_w, ph, [[("GATv2 Attn.  a", "bsans"), ("r,h", "b_"), ("T", "b^")]], fill=DAN_CREAM, border=BLACK, border_pt=1.15, rounded=True, round_radius=0.20, color=BLACK, size_pt=9.5, margin_lr=0.0, margin_tb=0.0)
        add_line(slide, cx_t, y_l2, cx_t, y_l3 + ph, color=BLACK, width_pt=1.15)
        add_box(slide, px_in, y_l3, inner_w, ph, [[("Per-Relation Softmax", "bsans")]], fill=DAN_SAGE, border=BLACK, border_pt=1.15, rounded=True, round_radius=0.20, color=BLACK, size_pt=9.5, margin_lr=0.0, margin_tb=0.0)
        add_line(slide, cx_t, y_l3, cx_t, y_l4 + ph, color=BLACK, width_pt=1.15)
        add_box(slide, px_in, y_l4, inner_w, ph, [[("Weighted  \u03a3", "bsans"), ("u", "b_"), (" \u03b1 W", "bsans"), ("s", "b_"), (" h", "bi"), ("u", "b_")]], fill=DAN_PEACH, border=BLACK, border_pt=1.15, rounded=True, round_radius=0.20, color=BLACK, size_pt=9.4, margin_lr=0.0, margin_tb=0.0)

        add_line(slide, cx_t, y_l4, cx_t, y_hm_bar + 0.28, color=BLACK, width_pt=1.2)
        bar_x, bar_w = bx_c - 0.10, card_w + 0.10
        add_box(slide, bar_x, y_hm_bar, bar_w, 0.28, [[("4-Head Mean + Relation Sum", "bsans")]], fill=DAN_SKY, border=BLACK, border_pt=1.25, rounded=True, round_radius=0.18, color=BLACK, size_pt=9.4, margin_lr=0.0, margin_tb=0.0)
        add_line(slide, cx_t, y_hm_bar, cx_t, y_res_bar + 0.29, color=BLACK, width_pt=1.2, arrow_end="triangle", arrow_size="sm")
        add_box(slide, bar_x, y_res_bar, bar_w, 0.29, [[("Post-Norm LN \u2192 LReLU \u2192 Add (+)", "bsans")]], fill=DAN_PEACH, border=BLACK, border_pt=1.25, rounded=True, round_radius=0.18, color=BLACK, size_pt=9.3, margin_lr=0.0, margin_tb=0.0)

        rx_sk = bx_c + card_w + 0.08
        add_polyline(
            slide,
            [(cx_t, y_card_top + card_h + 0.10), (rx_sk, y_card_top + card_h + 0.10), (rx_sk, y_res_bar + 0.145), (bar_x + bar_w + 0.015, y_res_bar + 0.145)],
            color=BLACK, width_pt=1.2, arrow_end="triangle", arrow_size="sm",
        )
        add_angled_label(slide, rx_sk + 0.08 - 0.28, y_card_top + 0.72, 0.56, 0.20, "skip", angle_deg=90.0, size_pt=9.2, color=rgb("#2B3945"), bold=True)
        add_line(slide, cx_t, y_res_bar, cx_t, y_pool + 0.26, color=BLACK, width_pt=1.3, arrow_end="triangle", arrow_size="sm")

    # H_theta^(5) dashed bypass cleanly aligned inside env_x0!
    y_tap = 2.90
    x_byp = env_x0 + 0.24
    add_polyline(
        slide,
        [(cx_L, y_tap), (x_byp, y_tap), (x_byp, y_ptr + h_ptr / 2), (p_x0 - 0.02, y_ptr + h_ptr / 2)],
        color=BLACK, width_pt=1.25, dash="dash", arrow_end="triangle", arrow_size="sm",
    )
    add_box(slide, env_x0, 2.24, 0.48, 0.26, [[("H", "b"), ("\u03b8", "b_"), ("(5)", "b^")]], fill=PALETTE["white"], border=BLACK, border_pt=1.2, rounded=True, round_radius=0.22, color=BLACK, size_pt=10.0, margin_lr=0.0, margin_tb=0.0)

    # --- BOTTOM NODE TYPE PROJECTION f_X FED DIRECTLY BY G_t FROM PANEL (a) ---
    y_fx, h_fx = 6.06, 0.30
    fx_w = 2.26
    for cx_t in (cx_L, cx_R):
        add_box(
            slide, cx_t - fx_w / 2.0, y_fx, fx_w, h_fx,
            [[("Node Projection ", "bsans"), ("f", "bi"), ("X", "b_"), (" (d=128)", "bsans")]],
            fill=DAN_LAV, border=BLACK, border_pt=1.25, rounded=True, round_radius=0.18,
            font_name=FONT_SERIF, color=BLACK, size_pt=9.6, margin_lr=0.0, margin_tb=0.0,
        )
        add_line(slide, cx_t, y_fx, cx_t, y_card_top + card_h + 0.01, color=BLACK, width_pt=1.25, arrow_end="triangle", arrow_size="sm")
        for k in (1, 2):
            tx = cx_t - k * 0.18
            add_polyline(slide, [(cx_t, y_fx - 0.07), (tx, y_fx - 0.07), (tx, y_card_top + card_h + k * 0.06 + 0.01)], color=BLACK, width_pt=1.05, arrow_end="triangle", arrow_size="sm")

    # Seamless horizontal distribution rail at y = 6.50 receiving G_t from Panel (a) into BOTH f_X bars!
    y_rail = 6.52
    add_polyline(slide, [(bx0, y_bus), (bx0 + 0.16, y_bus), (bx0 + 0.16, y_rail), (cx_R, y_rail), (cx_R, y_fx + h_fx + 0.015)], color=rgb("#5B4A73"), width_pt=1.75, arrow_end="triangle", arrow_size="sm")
    add_line(slide, cx_L, y_rail, cx_L, y_fx + h_fx + 0.015, color=rgb("#5B4A73"), width_pt=1.75, arrow_end="triangle", arrow_size="sm")
    add_circle(slide, cx_L, y_rail, 0.042, fill=rgb("#5B4A73"), border=None)

    set_notes(
        slide,
        "Compact Integrated Fig. 2(a)-(b) (B1 + B5 seamless horizontal fusion). Panel (a) maps the physical APAL workcell directly to the four-type heterogeneous state graph G_t = (O U M U W U K, E_t) with raw node features x_v, feeding seamlessly into Panel (b)'s independent Actor (theta) and Critic (phi) 5-layer 4-head GATv2 encoders (disjoint weights, theta cap phi = empty), Node-Type Attention Pooling, 3-tier autoregressive pointer policy pi_theta(a_tilde_t | s_t), scalar value head V_phi(s_t), and PPO GAE coupling. Eliminates duplicate G_t base bar for compact IEEE double-column banner placement (~2.24:1).\n"
        "Legal & Academic Photo Attribution (place in paper LaTeX figure caption): 'Assembly floor photograph adapted from Edward Russell (Wikimedia Commons, CC BY 2.0).' Source: https://commons.wikimedia.org/wiki/File:Final_assembly_of_the_2nd_787-10_(32910009961).jpg",
    )
