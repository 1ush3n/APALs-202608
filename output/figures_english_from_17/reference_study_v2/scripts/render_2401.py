import os, pypdfium2 as pdfium
pdf_path = r"D:\APAL-Dynamic-v4\output\figures_english_from_17\reference_study_v2\raw_papers\arxiv_2401_10241_zero_bubble.pdf"
out_dir = r"D:\APAL-Dynamic-v4\output\figures_english_from_17\reference_study_v2\rendered_pages"
doc = pdfium.PdfDocument(pdf_path)
n = min(7, len(doc))
for i in range(n):
    page = doc[i]
    bmp = page.render(scale=200/72.0)
    im = bmp.to_pil()
    out_p = os.path.join(out_dir, f"arxiv_2401_10241_p{i+1}.png")
    im.save(out_p)
    print(f"Rendered p{i+1} -> {im.size}")
