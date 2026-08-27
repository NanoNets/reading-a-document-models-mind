"""Build the E1 fit-corpus manifest: 900 DocILE page prompts + 100 text prompts.

Paths written into the manifest point at $E1_PAGES_DIR (default: <repo>/data/e1_pages),
so build the manifest ON the machine that will run the fit, after rendering the
DocILE pages there (see code/render_pages.py). Instruction variants rotate
deterministically so the corpus-averaged Jacobian sees varied OCR contexts.
"""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES_DIR = os.environ.get("E1_PAGES_DIR", os.path.join(ROOT, "data", "e1_pages"))
INSTRUCTIONS = [
    "Transcribe this document as plain text.",
    "Convert this document to markdown, preserving tables.",
    "Extract all line items from this document as a table.",
    "Read this document and reproduce its full text content.",
]
docs = json.load(open(os.path.join(ROOT, "data", "e1_docs_900.json")))
prompts = []
for i, f in enumerate(docs):
    doc_id = f[:-4]
    png = os.path.join(PAGES_DIR, f"{doc_id}.png")
    prompts.append(f"IMG::{png}::{INSTRUCTIONS[i % len(INSTRUCTIONS)]}")
text = json.load(open(os.path.join(ROOT, "data", "prompts_wikitext_1000.json")))[:100]
prompts += text
json.dump(prompts, open(os.path.join(ROOT, "data", "e1_prompts_manifest.json"), "w"))
print(f"manifest: {len(prompts)} prompts ({len(docs)} doc + {len(text)} text) -> data/e1_prompts_manifest.json")
