import fitz
import os

pdf_path = "paper.pdf"
out_dir = "pages"
os.makedirs(out_dir, exist_ok=True)

doc = fitz.open(pdf_path)
print(f"Total pages: {len(doc)}")
for i, page in enumerate(doc):
    mat = fitz.Matrix(150/72, 150/72)
    pix = page.get_pixmap(matrix=mat)
    out_path = f"{out_dir}/page_{i+1:02d}.png"
    pix.save(out_path)
    print(f"Saved {out_path}")
doc.close()
print("Done.")
