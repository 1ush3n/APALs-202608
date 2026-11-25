"""Native PowerPoint vector drawing helpers for APAL figure refinement.

All units are in inches (13.333 x 7.5 in widescreen slide).
All objects created are 100% native editable PowerPoint vector shapes and rich text.
"""
from __future__ import annotations

from typing import Iterable, Sequence
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt


# Cohesive publication palette (colorblind-conscious, crisp on white & grayscale)
PALETTE = {
    "ink": RGBColor(0x1B, 0x2A, 0x38),         # Primary dark slate-navy text
    "ink_soft": RGBColor(0x4A, 0x5A, 0x6A),    # Secondary slate text
    "muted": RGBColor(0x7B, 0x8A, 0x97),       # Muted gray labels / inactive
    "line_gray": RGBColor(0xB8, 0xC4, 0xCE),   # Subtle frame / divider gray
    "bg_tint": RGBColor(0xF6, 0xF8, 0xFA),     # Very soft card fill
    "white": RGBColor(0xFF, 0xFF, 0xFF),

    # Entity semantic colors (consistent across B1-B11)
    "op_dark": RGBColor(0x28, 0x56, 0x80),     # Operation steel-blue (dark)
    "op_mid": RGBColor(0x4E, 0x7E, 0xA8),
    "op_fill": RGBColor(0xE8, 0xF1, 0xF8),
    "op_red_dark": RGBColor(0xA6, 0x3D, 0x33), # Operation terracotta (graph nodes in B1-B4)
    "op_red_fill": RGBColor(0xFA, 0xEB, 0xE9),

    "st_dark": RGBColor(0x1D, 0x6F, 0x63),     # Station teal/emerald
    "st_mid": RGBColor(0x45, 0x94, 0x87),
    "st_fill": RGBColor(0xE4, 0xF3, 0xF0),

    "wk_dark": RGBColor(0xA8, 0x5C, 0x1E),     # Worker amber/ochre
    "wk_mid": RGBColor(0xC9, 0x84, 0x43),
    "wk_fill": RGBColor(0xFB, 0xF0, 0xE2),

    "sk_dark": RGBColor(0x6D, 0x45, 0x8E),     # Skill / Critic purple
    "sk_mid": RGBColor(0x95, 0x70, 0xB4),
    "sk_fill": RGBColor(0xF1, 0xEB, 0xF7),

    "calc_fill": RGBColor(0xFD, 0xF4, 0xD7),   # Non-learnable / scoring warm cream
    "calc_border": RGBColor(0x9C, 0x7A, 0x2A),

    "ok_dark": RGBColor(0x1F, 0x7A, 0x4C),     # Eligible / positive green
    "ok_fill": RGBColor(0xE2, 0xF4, 0xEA),

    "bad_dark": RGBColor(0x9E, 0x2A, 0x2B),    # Blocked / counter-example crimson
    "bad_fill": RGBColor(0xFC, 0xEA, 0xEA),
    "mask_fill": RGBColor(0xEE, 0xF1, 0xF4),   # Masked / inactive gray fill
    "mask_border": RGBColor(0xB4, 0xBE, 0xC8),

    "accent_orange": RGBColor(0xB8, 0x53, 0x18),
    "accent_navy": RGBColor(0x1D, 0x3B, 0x56),
}

FONT_SANS = "Arial"
FONT_SERIF = "Cambria"
FONT_MONO = "Consolas"


def rgb(hex_str: str) -> RGBColor:
    h = hex_str.lstrip("#")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def clear_slide(slide, keep_names: set[str] | None = None):
    """Remove all shapes from slide except those whose name is in keep_names."""
    keep_names = keep_names or set()
    to_remove = []
    for sh in slide.shapes:
        if sh.name not in keep_names:
            to_remove.append(sh._element)
    for el in to_remove:
        el.getparent().remove(el)


def set_notes(slide, text: str):
    notes_slide = slide.notes_slide
    tf = notes_slide.notes_text_frame
    tf.clear()
    tf.text = text


def _apply_run_style(
    run,
    font_name: str,
    size_pt: float,
    color: RGBColor,
    bold: bool = False,
    italic: bool = False,
    sub: bool = False,
    sup: bool = False,
):
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.italic = italic
    rPr = run._r.get_or_add_rPr()
    rPr.set("noProof", "1")
    if sub or sup:
        rPr.set("baseline", "-25000" if sub else "35000")


def populate_rich_paragraph(
    p,
    segments: Sequence[str | tuple],
    *,
    font_name: str = FONT_SERIF,
    size_pt: float = 13.0,
    color: RGBColor = PALETTE["ink"],
    bold: bool = False,
    italic: bool = False,
    align: PP_ALIGN = PP_ALIGN.CENTER,
):
    """Populate a paragraph with rich text segments.

    Each segment can be:
      - str: plain text using defaults
      - (text, mode_str): mode_str can contain 'b' (bold), 'i' (italic), '_' (sub), '^' (sup),
                          'sans', 'serif', 'mono'
      - (text, mode_str, custom_color_or_None)
      - (text, mode_str, custom_color_or_None, custom_size_pt_or_None)
    """
    p.alignment = align
    for seg in segments:
        if isinstance(seg, str):
            text, mode, c, sz = seg, "", color, size_pt
        else:
            text = seg[0]
            mode = seg[1] if len(seg) > 1 else ""
            c = seg[2] if len(seg) > 2 and seg[2] is not None else color
            sz = seg[3] if len(seg) > 3 and seg[3] is not None else size_pt
        r = p.add_run()
        r.text = text
        fn = font_name
        if "sans" in mode:
            fn = FONT_SANS
        elif "serif" in mode:
            fn = FONT_SERIF
        elif "mono" in mode:
            fn = FONT_MONO
        # Filter single-char flags after removing font tags
        flags = mode.replace("sans", "").replace("serif", "").replace("mono", "")
        is_b = bold or ("b" in flags)
        is_i = italic or ("i" in flags)
        is_sub = "_" in flags
        is_sup = "^" in flags
        if "n" in flags:  # explicit normal (not bold)
            is_b = False
        _apply_run_style(r, fn, sz, c, bold=is_b, italic=is_i, sub=is_sub, sup=is_sup)


def set_shape_rich_text(
    shape,
    lines: Sequence[Sequence[str | tuple]],
    *,
    font_name: str = FONT_SERIF,
    size_pt: float = 13.0,
    color: RGBColor = PALETTE["ink"],
    bold: bool = False,
    italic: bool = False,
    align: PP_ALIGN = PP_ALIGN.CENTER,
    valign: MSO_ANCHOR = MSO_ANCHOR.MIDDLE,
    margin_lr: float = 0.04,
    margin_tb: float = 0.02,
    word_wrap: bool = True,
    line_spacing: float | None = None,
):
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = word_wrap
    tf.vertical_anchor = valign
    tf.margin_left = Inches(margin_lr)
    tf.margin_right = Inches(margin_lr)
    tf.margin_top = Inches(margin_tb)
    tf.margin_bottom = Inches(margin_tb)
    for idx, segs in enumerate(lines):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        if line_spacing is not None:
            p.line_spacing = line_spacing
        if isinstance(segs, str):
            segs = [segs]
        populate_rich_paragraph(
            p,
            segs,
            font_name=font_name,
            size_pt=size_pt,
            color=color,
            bold=bold,
            italic=italic,
            align=align,
        )


def add_textbox(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    lines: Sequence[Sequence[str | tuple]] | str,
    *,
    font_name: str = FONT_SERIF,
    size_pt: float = 13.0,
    color: RGBColor = PALETTE["ink"],
    bold: bool = False,
    italic: bool = False,
    align: PP_ALIGN = PP_ALIGN.LEFT,
    valign: MSO_ANCHOR = MSO_ANCHOR.MIDDLE,
    margin_lr: float = 0.02,
    margin_tb: float = 0.01,
    word_wrap: bool = True,
    line_spacing: float | None = None,
):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if isinstance(lines, str):
        lines = [[lines]]
    set_shape_rich_text(
        tb,
        lines,
        font_name=font_name,
        size_pt=size_pt,
        color=color,
        bold=bold,
        italic=italic,
        align=align,
        valign=valign,
        margin_lr=margin_lr,
        margin_tb=margin_tb,
        word_wrap=word_wrap,
        line_spacing=line_spacing,
    )
    return tb


def _set_line_style(line_fmt, color: RGBColor | None, width_pt: float, dash: str | None = None):
    if color is None or width_pt <= 0:
        line_fmt.fill.background()
        return
    line_fmt.color.rgb = color
    line_fmt.width = Pt(width_pt)
    if dash == "dash":
        line_fmt.dash_style = MSO_LINE_DASH_STYLE.DASH
    elif dash == "dot":
        line_fmt.dash_style = MSO_LINE_DASH_STYLE.ROUND_DOT
    elif dash == "dashdot":
        line_fmt.dash_style = MSO_LINE_DASH_STYLE.DASH_DOT
    elif dash == "longdash":
        line_fmt.dash_style = MSO_LINE_DASH_STYLE.LONG_DASH


def add_box(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    lines: Sequence[Sequence[str | tuple]] | str | None = None,
    *,
    fill: RGBColor | None = PALETTE["white"],
    border: RGBColor | None = PALETTE["ink"],
    border_pt: float = 1.0,
    dash: str | None = None,
    rounded: bool = False,
    round_radius: float = 0.12,
    font_name: str = FONT_SERIF,
    size_pt: float = 13.0,
    color: RGBColor = PALETTE["ink"],
    bold: bool = False,
    italic: bool = False,
    align: PP_ALIGN = PP_ALIGN.CENTER,
    valign: MSO_ANCHOR = MSO_ANCHOR.MIDDLE,
    margin_lr: float = 0.04,
    margin_tb: float = 0.02,
    line_spacing: float | None = None,
):
    st = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    sh = slide.shapes.add_shape(st, Inches(x), Inches(y), Inches(w), Inches(h))
    if rounded:
        sh.adjustments[0] = round_radius
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    _set_line_style(sh.line, border, border_pt, dash)
    sh.shadow.inherit = False
    if lines is not None:
        if isinstance(lines, str):
            lines = [[lines]]
        set_shape_rich_text(
            sh,
            lines,
            font_name=font_name,
            size_pt=size_pt,
            color=color,
            bold=bold,
            italic=italic,
            align=align,
            valign=valign,
            margin_lr=margin_lr,
            margin_tb=margin_tb,
            line_spacing=line_spacing,
        )
    return sh


def add_circle(
    slide,
    cx: float,
    cy: float,
    r: float,
    lines: Sequence[Sequence[str | tuple]] | str | None = None,
    *,
    fill: RGBColor | None = PALETTE["white"],
    border: RGBColor | None = PALETTE["ink"],
    border_pt: float = 1.25,
    dash: str | None = None,
    font_name: str = FONT_SERIF,
    size_pt: float = 13.0,
    color: RGBColor = PALETTE["ink"],
    bold: bool = True,
    italic: bool = False,
):
    sh = slide.shapes.add_shape(
        MSO_SHAPE.OVAL, Inches(cx - r), Inches(cy - r), Inches(2 * r), Inches(2 * r)
    )
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    _set_line_style(sh.line, border, border_pt, dash)
    sh.shadow.inherit = False
    if lines is not None:
        if isinstance(lines, str):
            lines = [[lines]]
        set_shape_rich_text(
            sh,
            lines,
            font_name=font_name,
            size_pt=size_pt,
            color=color,
            bold=bold,
            italic=italic,
            align=PP_ALIGN.CENTER,
            valign=MSO_ANCHOR.MIDDLE,
            margin_lr=0.01,
            margin_tb=0.01,
            word_wrap=False,
        )
    return sh


def _add_arrowheads(shape_el, arrow_end: str | None, arrow_start: str | None, size: str = "med"):
    ln = shape_el.find(".//" + qn("a:ln"))
    if ln is None:
        spPr = shape_el.find(qn("p:spPr"))
        ln = OxmlElement("a:ln")
        spPr.append(ln)
    if arrow_start:
        head = OxmlElement("a:headEnd")
        head.set("type", arrow_start)
        head.set("w", size)
        head.set("len", size)
        ln.append(head)
    if arrow_end:
        tail = OxmlElement("a:tailEnd")
        tail.set("type", arrow_end)
        tail.set("w", size)
        tail.set("len", size)
        ln.append(tail)


def add_line(
    slide,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: RGBColor = PALETTE["ink"],
    width_pt: float = 1.25,
    dash: str | None = None,
    arrow_end: str | None = None,   # "triangle", "stealth", "oval"
    arrow_start: str | None = None,
    arrow_size: str = "med",        # "sm", "med", "lg"
):
    """Add an exact straight connector from (x1, y1) to (x2, y2) with proper flipH/flipV."""
    conn = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    _set_line_style(conn.line, color, width_pt, dash)
    conn.shadow.inherit = False
    if arrow_end or arrow_start:
        _add_arrowheads(conn._element, arrow_end, arrow_start, size=arrow_size)
    return conn


def add_polyline(
    slide,
    pts: Sequence[tuple[float, float]],
    *,
    color: RGBColor = PALETTE["ink"],
    width_pt: float = 1.25,
    dash: str | None = None,
    arrow_end: str | None = None,
    arrow_start: str | None = None,
    arrow_size: str = "med",
):
    """Add a single native editable open polyline connecting pts [(x, y), ...] in inches."""
    assert len(pts) >= 2
    fb = slide.shapes.build_freeform(Inches(pts[0][0]), Inches(pts[0][1]))
    fb.add_line_segments([(Inches(x), Inches(y)) for x, y in pts[1:]], close=False)
    sh = fb.convert_to_shape()
    sh.fill.background()
    _set_line_style(sh.line, color, width_pt, dash)
    sh.shadow.inherit = False
    if arrow_end or arrow_start:
        _add_arrowheads(sh._element, arrow_end, arrow_start, size=arrow_size)
    return sh


def add_stage_bracket(
    slide,
    x1: float,
    x2: float,
    y: float,
    label: str,
    *,
    color: RGBColor = PALETTE["ink"],
    tick_h: float = 0.09,
    label_y_offset: float = 0.10,
    size_pt: float = 14.5,
):
    """Add a clean ReSched-style horizontal stage bracket with a small center V-notch."""
    xm = (x1 + x2) / 2.0
    notch = 0.07
    pts = [
        (x1, y - tick_h),
        (x1, y),
        (xm - notch, y),
        (xm, y + notch),
        (xm + notch, y),
        (x2, y),
        (x2, y - tick_h),
    ]
    add_polyline(slide, pts, color=color, width_pt=1.2)
    add_textbox(
        slide,
        x1,
        y + label_y_offset,
        x2 - x1,
        0.34,
        [[(label, "b")]],
        font_name=FONT_SERIF,
        size_pt=size_pt,
        color=color,
        align=PP_ALIGN.CENTER,
        valign=MSO_ANCHOR.TOP,
    )


def add_swatch_strip(
    slide,
    x: float,
    y: float,
    colors: Sequence[RGBColor],
    *,
    cell_w: float = 0.22,
    cell_h: float = 0.28,
    gap: float = 0.015,
    border: RGBColor | None = PALETTE["white"],
    border_pt: float = 0.5,
    vertical: bool = False,
):
    """Draw a vector strip of colored feature cells (horizontal or vertical)."""
    shapes = []
    for idx, c in enumerate(colors):
        cx = x if vertical else x + idx * (cell_w + gap)
        cy = y + idx * (cell_h + gap) if vertical else y
        sh = add_box(
            slide,
            cx,
            cy,
            cell_w,
            cell_h,
            fill=c,
            border=border,
            border_pt=border_pt,
        )
        shapes.append(sh)
    return shapes


def add_hatched_bar(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: RGBColor = RGBColor(0xE8, 0xEC, 0xF0),
    hatch_color: RGBColor = RGBColor(0xB4, 0xBF, 0xCA),
    border: RGBColor = RGBColor(0x98, 0xA6, 0xB4),
    border_pt: float = 0.85,
    spacing: float = 0.16,
    label: str | None = None,
    label_color: RGBColor = PALETTE["ink_soft"],
    label_pt: float = 10.0,
):
    """Draw a PipeDream-style diagonal-hatched bar using crisp native vector strokes."""
    box = add_box(slide, x, y, w, h, fill=fill, border=border, border_pt=border_pt)
    # Draw crisp diagonal hatch lines clipped mathematically to [x, x+w] x [y, y+h]
    # Diagonal slope: dy = -h over dx = h * 0.65 (about 57 deg)
    dx_full = h * 0.60
    t = spacing * 0.6
    while t < w + dx_full:
        # Line from (x + t - dx_full, y + h) to (x + t, y)
        px0, py0 = x + t - dx_full, y + h
        px1, py1 = x + t, y
        # Clip x to [x, x + w]
        if px0 < x:
            frac = (x - px0) / (px1 - px0)
            px0 = x
            py0 = py0 + frac * (py1 - py0)
        if px1 > x + w:
            frac = (x + w - px0) / (px1 - px0)
            px1 = x + w
            py1 = py0 + frac * (py1 - py0)
        if px1 > px0 + 0.01 and abs(py0 - py1) > 0.01:
            add_line(slide, px0, py0, px1, py1, color=hatch_color, width_pt=0.65)
        t += spacing
    if label:
        add_textbox(
            slide,
            x + 0.08,
            y,
            w - 0.14,
            h,
            [[label]],
            font_name=FONT_SANS,
            size_pt=label_pt,
            color=label_color,
            align=PP_ALIGN.LEFT,
            valign=MSO_ANCHOR.MIDDLE,
            margin_lr=0.02,
            margin_tb=0.0,
        )
    return box


def add_header(
    slide,
    badge: str,
    title: str,
    subtitle: str | None = None,
    right_tag: str | None = None,
    *,
    rule: bool = True,
):
    """Standardized clean header matching the review deck."""
    # Module ID pill
    add_box(
        slide,
        0.34,
        0.24,
        0.58,
        0.36,
        [[(badge, "bsans")]],
        fill=PALETTE["accent_navy"],
        border=None,
        rounded=True,
        round_radius=0.2,
        color=PALETTE["white"],
        size_pt=13.5,
    )
    add_textbox(
        slide,
        1.02,
        0.20,
        9.9 if right_tag else 11.9,
        0.42,
        [[(title, "bsans")]],
        font_name=FONT_SANS,
        size_pt=20.0,
        color=PALETTE["ink"],
        align=PP_ALIGN.LEFT,
        valign=MSO_ANCHOR.MIDDLE,
        margin_lr=0.0,
    )
    if right_tag:
        add_textbox(
            slide,
            10.75,
            0.24,
            2.24,
            0.34,
            [[(right_tag, "bsans")]],
            font_name=FONT_SANS,
            size_pt=10.5,
            color=PALETTE["ink_soft"],
            align=PP_ALIGN.RIGHT,
            valign=MSO_ANCHOR.MIDDLE,
            margin_lr=0.0,
        )
    if subtitle:
        sub_lines = [[subtitle]] if isinstance(subtitle, str) else subtitle
        add_textbox(
            slide,
            0.34,
            0.64,
            12.65,
            0.30,
            sub_lines,
            font_name=FONT_SANS,
            size_pt=11.0,
            color=PALETTE["ink_soft"],
            align=PP_ALIGN.LEFT,
            valign=MSO_ANCHOR.MIDDLE,
            margin_lr=0.0,
        )
    if rule:
        y_rule = 0.96 if subtitle else 0.70
        add_line(slide, 0.34, y_rule, 12.99, y_rule, color=PALETTE["line_gray"], width_pt=0.8)


def add_footer(slide, text: str, y: float = 7.12):
    add_textbox(
        slide,
        0.34,
        y,
        12.65,
        0.30,
        [[text]],
        font_name=FONT_SANS,
        size_pt=9.2,
        color=PALETTE["ink_soft"],
        align=PP_ALIGN.LEFT,
        valign=MSO_ANCHOR.MIDDLE,
        margin_lr=0.0,
    )


# --- Top-venue publication vector primitives (ReSched / DAN / Smit styles) ---
BLACK = RGBColor(0x16, 0x18, 0x1C)


def add_activation_glyph(
    slide,
    cx: float,
    cy: float,
    r: float = 0.145,
    kind: str = "lrelu",
    *,
    fill: RGBColor = PALETTE["white"],
    border: RGBColor = BLACK,
    border_pt: float = 1.3,
    curve_color: RGBColor = BLACK,
    curve_pt: float = 1.25,
):
    """Draw a DAN Fig. 3 style circular activation icon (LeakyReLU kink or tanh S-curve)."""
    add_circle(slide, cx, cy, r, fill=fill, border=border, border_pt=border_pt)
    if kind == "lrelu":
        # Crisp LeakyReLU curve: slight negative slope left, steep positive slope right
        pts = [
            (cx - 0.62 * r, cy + 0.34 * r),
            (cx - 0.05 * r, cy + 0.22 * r),
            (cx + 0.56 * r, cy - 0.52 * r),
        ]
    elif kind == "tanh":
        # Smooth 7-point tanh S-curve
        import math
        pts = []
        for k in range(-3, 4):
            u = k / 3.0
            px = cx + u * 0.62 * r
            py = cy - math.tanh(2.1 * u) * 0.50 * r
            pts.append((px, py))
    else:
        # ReLU
        pts = [
            (cx - 0.60 * r, cy + 0.28 * r),
            (cx, cy + 0.28 * r),
            (cx + 0.56 * r, cy - 0.50 * r),
        ]
    add_polyline(slide, pts, color=curve_color, width_pt=curve_pt)


def add_partitioned_vector(
    slide,
    x: float,
    y: float,
    cell_lines: Sequence[Sequence[str | tuple]],
    *,
    cell_w: float = 0.44,
    cell_h: float = 0.28,
    fill: RGBColor = PALETTE["white"],
    fills: Sequence[RGBColor] | None = None,
    border: RGBColor = BLACK,
    border_pt: float = 1.15,
    font_name: str = FONT_SERIF,
    size_pt: float = 10.5,
    color: RGBColor = BLACK,
    vertical: bool = False,
):
    """Draw a Smit et al. Fig. 1 partitioned math vector [h^(1) | h^(2) | ...]."""
    shapes = []
    for i, segs in enumerate(cell_lines):
        cx = x if vertical else x + i * cell_w
        cy = y + i * cell_h if vertical else y
        f = fills[i] if fills is not None else fill
        sh = add_box(
            slide,
            cx,
            cy,
            cell_w,
            cell_h,
            [segs] if segs else None,
            fill=f,
            border=border,
            border_pt=border_pt,
            font_name=font_name,
            size_pt=size_pt,
            color=color,
            margin_lr=0.01,
            margin_tb=0.0,
        )
        shapes.append(sh)
    return shapes


def add_collector_bar(
    slide,
    xs: Sequence[float],
    y_nodes: float,
    y_bar: float,
    y_target: float,
    *,
    color: RGBColor = BLACK,
    width_pt: float = 1.2,
    arrow_end: str | None = "triangle",
    arrow_size: str = "sm",
):
    """Draw a crisp DAN Fig. 3 style collector bar merging multiple x columns into one central stem."""
    assert len(xs) >= 2
    x_min, x_max = min(xs), max(xs)
    xm = (x_min + x_max) / 2.0
    add_line(slide, x_min, y_bar, x_max, y_bar, color=color, width_pt=width_pt)
    for xv in xs:
        add_line(slide, xv, y_nodes, xv, y_bar, color=color, width_pt=width_pt)
    add_line(slide, xm, y_bar, xm, y_target, color=color, width_pt=width_pt, arrow_end=arrow_end, arrow_size=arrow_size)


def add_angled_label(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str | Sequence,
    angle_deg: float = -45.0,
    *,
    font_name: str = FONT_SANS,
    size_pt: float = 9.0,
    color: RGBColor = BLACK,
    bold: bool = False,
):
    """Add a rotated text label (supports plain string or rich-text segment lines)."""
    lines = [[(text, "b" if bold else "")]] if isinstance(text, str) else text
    tb = add_textbox(
        slide,
        x,
        y,
        w,
        h,
        lines,
        font_name=font_name,
        size_pt=size_pt,
        color=color,
        align=PP_ALIGN.CENTER,
        valign=MSO_ANCHOR.MIDDLE,
        margin_lr=0.0,
        margin_tb=0.0,
    )
    tb.rotation = angle_deg
    return tb

