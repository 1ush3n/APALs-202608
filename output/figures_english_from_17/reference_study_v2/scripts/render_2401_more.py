import os, pypdfium2 as pdfium
pdf_path = r"D:\APAL-Dynamic-v4\output\figures_english_from_17\reference_study_v2\raw_papers\arxiv_2401_10241_zero_bubble.pdf"
out_dir = r"D:\APAL-Dynamic-v4\output\figures_english_from_17\reference_study_v2\rendered_pages"
doc = pdfium.PdfDocument(pdf_path)
print("Total pages:", len(doc))
for p in [6, 8, 10, 12]:
    if p <= len(doc):
        im = doc[p-1].render(scale=200/72.0).to_pil()
        im.save(os.path.join(out_dir, f"arxiv_2401_10241_p{p}.png"))
        print(f"Rendered p{p} -> {im.size}")
