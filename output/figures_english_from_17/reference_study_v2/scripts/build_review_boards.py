import json
import os
from PIL import Image, ImageDraw, ImageFont

BASE = r"D:\APAL-Dynamic-v4\output\figures_english_from_17\reference_study_v2"
ASSETS_DIR = os.path.join(BASE, "assets")
BOARDS_DIR = os.path.join(BASE, "boards")
MANIFEST = os.path.join(BASE, "crop_manifest_v2.json")

os.makedirs(BOARDS_DIR, exist_ok=True)

with open(MANIFEST, "r", encoding="utf-8") as f:
    items = json.load(f)

CAT_META = {
    "cat1_fig1_problem": {
        "prefix": "C1_Board",
        "title": "Category 1: Fig. 1 Problem Overview, Cyclic Unfolding & Dynamic Rescheduling",
        "header_rgb": (20, 48, 82),
        "badge_rgb": (22, 92, 148),
        "venue_rgb": (228, 240, 250),
        "venue_txt": (16, 66, 114),
    },
    "cat2_fig2_fusion": {
        "prefix": "C2_Board",
        "title": "Category 2: Fig. 2 Physical Workcell / Factory -> Heterogeneous Graph -> GNN-RL",
        "header_rgb": (18, 72, 62),
        "badge_rgb": (16, 110, 92),
        "venue_rgb": (224, 245, 239),
        "venue_txt": (14, 78, 66),
    },
    "cat3_fig3_encdec": {
        "prefix": "C3_Board",
        "title": "Category 3: Fig. 3 Unified Encoder + Autoregressive Pointer Decoder Architectures",
        "header_rgb": (64, 38, 92),
        "badge_rgb": (102, 58, 150),
        "venue_rgb": (240, 232, 250),
        "venue_txt": (74, 40, 114),
    },
}

def load_font(size, bold=False):
    candidates = (
        [r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\arialbd.ttf"]
        if bold
        else [r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf"]
    )
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

F_HDR = load_font(42, bold=True)
F_SUB = load_font(24, bold=False)
F_ID = load_font(25, bold=True)
F_VENUE = load_font(21, bold=True)
F_CARD_TITLE = load_font(28, bold=True)
F_CARD_SUB = load_font(20, bold=False)
F_TAKEAWAY = load_font(22, bold=False)

def wrap_lines(draw, text, font, max_w, max_lines=2):
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        cand = w if not cur else cur + " " + w
        bbox = draw.textbbox((0, 0), cand, font=font)
        if bbox[2] - bbox[0] <= max_w:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while len(last) > 4:
            bbox = draw.textbbox((0, 0), last + "...", font=font)
            if bbox[2] - bbox[0] <= max_w:
                lines[-1] = last + "..."
                break
            last = last[:-1]
    return lines

W, H = 3200, 2160
HDR_H = 122
PAD = 36
GAP = 32
CARD_W = (W - 2 * PAD - GAP) // 2
CARD_H = (H - HDR_H - 2 * PAD - GAP) // 2

by_cat = {}
for it in items:
    by_cat.setdefault(it["category"], []).append(it)

for cat, cat_items in by_cat.items():
    meta = CAT_META[cat]
    total_boards = (len(cat_items) + 3) // 4
    for b_idx in range(total_boards):
        chunk = cat_items[b_idx * 4:(b_idx + 1) * 4]
        board = Image.new("RGB", (W, H), (242, 245, 249))
        draw = ImageDraw.Draw(board)

        # Header band
        draw.rectangle([0, 0, W, HDR_H], fill=meta["header_rgb"])
        draw.text((PAD, 22), f"{meta['title']}", fill=(255, 255, 255), font=F_HDR)
        draw.text(
            (PAD, 76),
            f"Board {b_idx + 1} of {total_boards}  |  Top-Venue Reference Atlas (300 DPI Curated Crops, Strictly Zero Overlap with figure_study_15)",
            fill=(204, 222, 238),
            font=F_SUB,
        )

        for k, it in enumerate(chunk):
            r, c = k // 2, k % 2
            x0 = PAD + c * (CARD_W + GAP)
            y0 = HDR_H + PAD + r * (CARD_H + GAP)
            x1 = x0 + CARD_W
            y1 = y0 + CARD_H

            # Card container
            draw.rounded_rectangle([x0, y0, x1, y1], radius=18, fill=(255, 255, 255), outline=(188, 198, 212), width=3)

            # Card top strip
            top_h = 104
            cx = x0 + 24
            # ID pill
            id_txt = it["id"]
            id_bb = draw.textbbox((0, 0), id_txt, font=F_ID)
            id_w = (id_bb[2] - id_bb[0]) + 28
            draw.rounded_rectangle([cx, y0 + 18, cx + id_w, y0 + 56], radius=10, fill=meta["badge_rgb"])
            draw.text((cx + 14, y0 + 22), id_txt, fill=(255, 255, 255), font=F_ID)

            # Venue pill
            vx = cx + id_w + 14
            v_txt = f"{it['venue']}  |  {it['fig']}"
            v_bb = draw.textbbox((0, 0), v_txt, font=F_VENUE)
            v_w = (v_bb[2] - v_bb[0]) + 26
            draw.rounded_rectangle([vx, y0 + 19, vx + v_w, y0 + 55], radius=10, fill=meta["venue_rgb"], outline=(185, 200, 215), width=2)
            draw.text((vx + 13, y0 + 24), v_txt, fill=meta["venue_txt"], font=F_VENUE)

            # Title line right beside or below
            tx = vx + v_w + 18
            avail_tw = (x1 - 24) - tx
            t_txt = it["short_title"]
            t_bb = draw.textbbox((0, 0), t_txt, font=F_CARD_TITLE)
            if (t_bb[2] - t_bb[0]) <= avail_tw:
                draw.text((tx, y0 + 19), t_txt, fill=(20, 28, 38), font=F_CARD_TITLE)
                draw.text((cx, y0 + 66), it["paper_title"], fill=(90, 102, 118), font=F_CARD_SUB)
            else:
                draw.text((cx, y0 + 62), f"{t_txt} — {it['paper_title'][:72]}", fill=(20, 28, 38), font=F_CARD_SUB)

            # Footer band
            ftr_h = 84
            fy0 = y1 - ftr_h - 14
            draw.line([x0 + 22, fy0 - 6, x1 - 22, fy0 - 6], fill=(224, 230, 238), width=2)
            lines = wrap_lines(draw, f"APAL Takeaway: {it['takeaway']}", F_TAKEAWAY, CARD_W - 52, max_lines=2)
            for li, ln in enumerate(lines):
                draw.text((x0 + 26, fy0 + li * 30), ln, fill=(40, 52, 68), font=F_TAKEAWAY)

            # Inner figure box
            img_box_x0 = x0 + 24
            img_box_y0 = y0 + top_h
            img_box_x1 = x1 - 24
            img_box_y1 = fy0 - 16
            draw.rounded_rectangle(
                [img_box_x0, img_box_y0, img_box_x1, img_box_y1],
                radius=10,
                fill=(250, 251, 253),
                outline=(216, 224, 234),
                width=2,
            )

            fig_path = os.path.join(ASSETS_DIR, cat, f"{it['id']}.png")
            fig_im = Image.open(fig_path).convert("RGB")
            avail_w = (img_box_x1 - img_box_x0) - 24
            avail_h = (img_box_y1 - img_box_y0) - 20
            scale = min(avail_w / fig_im.width, avail_h / fig_im.height)
            # Allow modest crisp upscale for small inline figures capped at 2.4x so small column figs fill card nicely
            scale = min(scale, 2.4)
            nw = max(1, int(round(fig_im.width * scale)))
            nh = max(1, int(round(fig_im.height * scale)))
            rs = fig_im.resize((nw, nh), Image.Resampling.LANCZOS)
            px = img_box_x0 + ((img_box_x1 - img_box_x0) - nw) // 2
            py = img_box_y0 + ((img_box_y1 - img_box_y0) - nh) // 2
            board.paste(rs, (px, py))

        out_path = os.path.join(BOARDS_DIR, f"{meta['prefix']}_{b_idx + 1:02d}.png")
        board.save(out_path, optimize=True)
        print(f"Saved Board -> {out_path}")
