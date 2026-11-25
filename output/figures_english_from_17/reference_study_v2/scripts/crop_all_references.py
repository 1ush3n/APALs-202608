import json
import os
import pypdfium2 as pdfium
from PIL import Image, ImageChops

BASE = r"D:\APAL-Dynamic-v4\output\figures_english_from_17\reference_study_v2"
RAW_DIR = os.path.join(BASE, "raw_papers")
ASSETS_DIR = os.path.join(BASE, "assets")
MANIFEST = os.path.join(BASE, "crop_manifest_v2.json")

with open(MANIFEST, "r", encoding="utf-8") as f:
    items = json.load(f)

doc_cache = {}
scale = 300.0 / 72.0

def auto_trim_white(pil_img, pad=14, thr=248):
    rgb = pil_img.convert("RGB")
    bg = Image.new("RGB", rgb.size, (255, 255, 255))
    diff = ImageChops.difference(rgb, bg)
    # Any channel differing by > (255 - thr) becomes non-zero
    cutoff = 255 - thr
    mask = diff.convert("L").point(lambda v: 255 if v > cutoff else 0)
    bb = mask.getbbox()
    if not bb:
        return rgb
    x0 = max(0, bb[0] - pad)
    y0 = max(0, bb[1] - pad)
    x1 = min(rgb.width, bb[2] + pad)
    y1 = min(rgb.height, bb[3] + pad)
    return rgb.crop((x0, y0, x1, y1))

for item in items:
    pdf_path = os.path.join(RAW_DIR, item["pdf"])
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Missing PDF: {pdf_path}")
    if pdf_path not in doc_cache:
        doc_cache[pdf_path] = pdfium.PdfDocument(pdf_path)
    doc = doc_cache[pdf_path]
    page = doc[item["page"] - 1]
    im = page.render(scale=scale).to_pil()
    w, h = im.size
    x0, y0, x1, y1 = item["bbox"]
    crop_box = (
        max(0, int(round(x0 * w))),
        max(0, int(round(y0 * h))),
        min(w, int(round(x1 * w))),
        min(h, int(round(y1 * h))),
    )
    cropped = im.crop(crop_box)
    trimmed = auto_trim_white(cropped, pad=14)
    out_dir = os.path.join(ASSETS_DIR, item["category"])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{item['id']}.png")
    trimmed.save(out_path, optimize=True)
    print(f"OK {item['id']} -> {trimmed.size[0]}x{trimmed.size[1]} | {item['short_title']}")

for d in doc_cache.values():
    d.close()
print(f"All {len(items)} crops saved successfully at 300 DPI.")
