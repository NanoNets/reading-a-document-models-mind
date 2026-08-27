"""Merge the 8 shard lenses into the canonical E1 lens + provenance meta."""
import json, os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jlens, torch, transformers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLUG = os.environ.get("MERGE_SLUG", "qwen2.5-vl-3b-instruct")
NSH  = int(os.environ.get("MERGE_NSHARDS", "8"))
paths = sorted(glob.glob(os.path.join(ROOT, f"data/lenses/{SLUG}.e1.shard*.pt")))
paths = [p for p in paths if "ckpt" not in p]
assert len(paths) == NSH, f"need {NSH} shards, found {len(paths)}"
lenses = [jlens.JacobianLens.load(p) for p in paths]
merged = jlens.JacobianLens.merge(lenses)   # API takes the full list in one call
OUT = os.path.join(ROOT, f"data/lenses/{SLUG}.e1.pt")
merged.save(OUT)
json.dump({"model": os.environ.get("MERGE_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"),
           "revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
           "corpus": "e1_prompts_manifest.json (900 DocILE pages + 100 wikitext), 8 shards x 125",
           "shards": [os.path.basename(p) for p in paths],
           "torch": torch.__version__, "transformers": transformers.__version__},
          open(OUT + ".meta.json", "w"), indent=1)
print(f"[merge] -> {OUT} (+meta)", flush=True)
