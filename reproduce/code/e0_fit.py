"""E0: fit a Jacobian lens with the official anthropics/jacobian-lens code.

Runs locally (MPS) or on Explorer (CUDA). Config via env vars:
  E0_MODEL         (default Qwen/Qwen2.5-0.5B-Instruct)
  E0_PROMPTS_FILE  (default data/prompts_wikitext_1000.json)
  E0_N_PROMPTS     (default 1000)
  E0_DIM_BATCH     (default 32 on cuda, 8 otherwise)
  E0_OUT_TAG       (default "e0")
Outputs: data/lenses/<model>.<tag>.pt + artifacts/evidence/tables/<tag>_readout.json
"""
import json, os, time
import torch, transformers, jlens

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL   = os.environ.get("E0_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PFILE   = os.environ.get("E0_PROMPTS_FILE", os.path.join(ROOT, "data", "prompts_wikitext_1000.json"))
N       = int(os.environ.get("E0_N_PROMPTS", "1000"))
TAG     = os.environ.get("E0_OUT_TAG", "e0")
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DIMB    = int(os.environ.get("E0_DIM_BATCH", "32" if dev == "cuda" else "8"))

mslug = MODEL.split("/")[-1].lower()
LENS_OUT = os.path.join(ROOT, "data", "lenses", f"{mslug}.{TAG}.pt")
EVID     = os.path.join(ROOT, "artifacts", "evidence", "tables", f"{TAG}_readout.json")
os.makedirs(os.path.dirname(LENS_OUT), exist_ok=True)
os.makedirs(os.path.dirname(EVID), exist_ok=True)

print(f"[e0] device={dev} torch={torch.__version__} model={MODEL} n={N} dim_batch={DIMB}", flush=True)
prompts = json.load(open(PFILE))[:N]
print(f"[e0] {len(prompts)} prompts from {PFILE}", flush=True)

REV = os.environ.get("E_REV", "main")
hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, torch_dtype=torch.float32).to(dev)
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
model = jlens.from_hf(hf, tok)

t0 = time.time()
lens = jlens.fit(model, prompts=prompts, max_seq_len=128, dim_batch=DIMB,
                 checkpoint_path=LENS_OUT + ".ckpt")
lens.save(LENS_OUT)
print(f"[e0] fit done in {time.time()-t0:.0f}s -> {LENS_OUT}", flush=True)

def readout(prompt, position, k=8):
    out = {}
    for use_jac, name in [(True, "jlens"), (False, "logit_lens")]:
        lg, _, _ = lens.apply(model, prompt, positions=[position], use_jacobian=use_jac)
        out[name] = {str(L): [tok.decode([t]) for t in v[0].topk(k).indices]
                     for L, v in sorted(lg.items())}
    return out

cases = {
  "boot_currency": readout("Fact: The currency used in the country shaped like a boot is", -2),
  "legs_spider":   readout("The number of legs on the animal that spins webs is", -2),
}
json.dump({"model": MODEL, "n_prompts": len(prompts), "dim_batch": DIMB, "device": dev,
           "fit_seconds": round(time.time()-t0), "script": __file__, "cases": cases},
          open(EVID, "w"), indent=1)
print(f"[e0] readouts -> {EVID}\n[e0] E0 FIT COMPLETE", flush=True)
