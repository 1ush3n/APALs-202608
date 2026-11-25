import os
import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont

BASE = r"D:\APAL-Dynamic-v4\output\figures_english_from_17\reference_study_v2"
RAW_DIR = os.path.join(BASE, "raw_papers")
PAGE_DIR = os.path.join(BASE, "rendered_pages")
TRIAGE_DIR = os.path.join(BASE, "triage_sheets")

targets = [
    ("2409.13571v1.pdf", [1, 11, 13]),
    ("2502.12617v2.pdf", [1, 8, 11]),
    ("2502.15791v1.pdf", [10]),
    ("NeurIPS-2025-heterogeneous-graph-transformers-for-simultaneous-mobile-multi-robot-task-allocation-and-scheduling-under-tempo.pdf", [10]),
    ("2201.00548v1.pdf", [8, 9, 10, 11]),
]

def get_font(sz):
    for p in [r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

items = []
for fn, pnums in targets:
    pdf_p = os.path.join(RAW_DIR, fn)
    stem = os.path.splitext(fn)[0]
    doc = pdfium.PdfDocument(pdf_p)
    for pn in pnums:
        if pn <= len(doc):
            out_p = os.path.join(PAGE_DIR, f"{stem}_p{pn:02d}.png")
            if not os.path.exists(out_p):
                im = doc[pn - 1].render(scale=165 / 72.0).to_pil()
                im.save(out_p)
            items.append((stem, pn, out_p))
    doc.close()

f_hdr = get_font(22)
per_sheet = 6
cw, ch, pad = 850, 1080, 18
for idx, s in enumerate(range(0, len(items), per_sheet), 1):
    chunk = items[s:s + per_sheet]
    canvas = Image.new("RGB", (3 * cw + 4 * pad, 2 * ch + 3 * pad), (240, 244, 248))
    draw = ImageDraw.Draw(canvas)
    for k, (stem, pnum, pg_path) in enumerate(chunk):
        r, c = k // 3, k % 3
        x0, y0 = pad + c * (cw + pad), pad + r * (ch + pad)
        draw.rectangle([x0, y0, x0 + cw, y0 + ch], fill=(255, 255, 255), outline=(140, 155, 170), width=2)
        draw.rectangle([x0, y0, x0 + cw, y0 + 38], fill=(32, 54, 78))
        draw.text((x0 + 12, y0 + 7), f"{stem[:28]} | p{pnum}", fill=(255, 255, 255), font=f_hdr)
        im = Image.open(pg_path).convert("RGB")
        scale = min((cw - 12) / im.size[0], (ch - 48) / im.size[1])
        rs = im.resize((int(im.size[0] * scale), int(im.size[1] * scale)), Image.Resampling.LANCZOS)
        canvas.paste(rs, (x0 + (cw - rs.size[0]) // 2, y0 + 42))
    out_sheet = os.path.join(TRIAGE_DIR, f"triage_late_{idx:02d}.png")
    canvas.save(out_sheet, optimize=True)
    print(f"Saved -> {out_sheet} ({len(chunk)} pages)")
