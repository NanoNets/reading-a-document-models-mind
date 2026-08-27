"""Render the E1 fit-corpus pages from a local DocILE download.

DocILE (https://docile.rossum.ai/) requires (free) registration; its license
does not permit us to redistribute the PDFs or rendered images, so this repo
ships only the 900 document IDs (data/e1_docs_900.json) and this renderer.

Usage:
    DOCILE_PDF_DIR=/path/to/docile/pdfs python code/render_pages.py

Renderer pin (must match to reproduce the fit corpus byte-for-byte):
    poppler pdftoppm 26.02.0, flags: -png -r 120 -f 1 -l 1 -singlefile
(first page only, 120 dpi PNG).
"""
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.environ.get("DOCILE_PDF_DIR") or sys.exit("set DOCILE_PDF_DIR to your DocILE pdfs/ directory")
OUT_DIR = os.environ.get("E1_PAGES_DIR", os.path.join(ROOT, "data", "e1_pages"))
os.makedirs(OUT_DIR, exist_ok=True)

docs = json.load(open(os.path.join(ROOT, "data", "e1_docs_900.json")))
done = missing = 0
for f in docs:
    doc_id = f[:-4]                      # entries are "<id>.pdf"
    src = os.path.join(PDF_DIR, f)
    dst = os.path.join(OUT_DIR, doc_id)  # pdftoppm appends .png
    if os.path.exists(dst + ".png"):
        done += 1
        continue
    if not os.path.exists(src):
        print(f"[render] MISSING {src}")
        missing += 1
        continue
    subprocess.run(["pdftoppm", "-png", "-r", "120", "-f", "1", "-l", "1",
                    "-singlefile", src, dst], check=True)
    done += 1
print(f"[render] {done} rendered/present, {missing} missing -> {OUT_DIR}")
if missing:
    sys.exit(1)
