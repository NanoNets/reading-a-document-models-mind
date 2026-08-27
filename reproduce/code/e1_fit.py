"""E1: fit the Jacobian lens on a document VLM over the E1 manifest, SHARDED.
Env: E1_MODEL (default Qwen/Qwen2.5-VL-3B-Instruct), E1_SHARD, E1_NSHARDS,
     E1_DIM_BATCH (default 32), E1_MAX_SEQ (default 512).
Each shard fits on its prompt slice and saves data/lenses/<slug>.e1.shard<k>.pt;
merge happens in a separate step via JacobianLens.merge.
"""
import json, logging, os, sys, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
import torch, transformers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jlens
from jlens_vlm import QwenVLLensModel

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL  = os.environ.get("E1_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
SHARD  = int(os.environ.get("E1_SHARD", "0"))
NSH    = int(os.environ.get("E1_NSHARDS", "10"))
_mem_gb = torch.cuda.get_device_properties(0).total_memory/1e9 if torch.cuda.is_available() else 0
_auto = 32 if _mem_gb >= 100 else (12 if _mem_gb >= 70 else (4 if _mem_gb >= 38 else 2))
DIMB   = int(os.environ.get("E1_DIM_BATCH", str(_auto)))
MAXSEQ = int(os.environ.get("E1_MAX_SEQ", "512"))
dev = "cuda" if torch.cuda.is_available() else sys.exit("[e1] cuda required (local-compute policy)")

slug = MODEL.split("/")[-1].lower()
OUT  = os.path.join(ROOT, "data", "lenses", f"{slug}.e1.shard{SHARD}.pt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
if os.path.exists(OUT):
    print(f"[e1] shard {SHARD} already done — idempotent exit"); sys.exit(0)

prompts = json.load(open(os.path.join(ROOT, "data", "e1_prompts_manifest.json")))[SHARD::NSH]
print(f"[e1] model={MODEL} shard={SHARD}/{NSH} n={len(prompts)} dim_batch={DIMB}", flush=True)

REV = os.environ.get("E_REV", "main")
hf = transformers.AutoModelForImageTextToText.from_pretrained(MODEL, revision=REV, dtype=torch.float32).to(dev)
proc = transformers.AutoProcessor.from_pretrained(MODEL, min_pixels=256*28*28, max_pixels=256*28*28)
model = QwenVLLensModel(hf, proc)

t0 = time.time()
lens = jlens.fit(model, prompts=prompts, max_seq_len=MAXSEQ, dim_batch=DIMB,
                 checkpoint_path=OUT + ".ckpt")
lens.save(OUT)
meta = {"model": MODEL, "revision": REV, "torch": torch.__version__,
        "transformers": transformers.__version__, "shard": SHARD, "nshards": NSH,
        "dim_batch": DIMB, "max_seq": MAXSEQ, "n_prompts": len(prompts),
        "fit_seconds": round(time.time()-t0)}
json.dump(meta, open(OUT + ".meta.json", "w"), indent=1)
print(f"[e1] shard {SHARD} done in {time.time()-t0:.0f}s -> {OUT} (+meta)", flush=True)
