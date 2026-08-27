"""E1 diagnostics: apply a lens (shard or merged) to a held-out document set.
Env: E_LENS (path), E_DOCSET (dir of images; default data/diag_docs), E_TAG.
Outputs artifacts/evidence/tables/<tag>_diag.json:
  - per-layer excess kurtosis of readout logits (band detector)
  - per-layer top-1 agreement with the model's real next-token (motor detector)
  - image-token vs text-token stats separately (C-5)
  - top-10 readout at the last prompt token per image (the receipt-figure data)
  - if other shard lenses exist: pairwise top-1 agreement on this set (saturation proxy)
Forward-only. Any GPU.
"""
import json, os, sys, glob
import torch, transformers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jlens
from jlens_vlm import QwenVLLensModel

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LENS_P = os.environ["E_LENS"]
DOCSET = os.environ.get("E_DOCSET", os.path.join(ROOT, "data", "diag_docs"))
TAG    = os.environ.get("E_TAG", "e1shard0")
MODEL  = os.environ.get("E1_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
REV    = os.environ.get("E_REV", "main")
if not torch.cuda.is_available(): sys.exit("[diag] cuda required")

hf = transformers.AutoModelForImageTextToText.from_pretrained(MODEL, revision=REV, dtype=torch.float32).cuda()
proc = transformers.AutoProcessor.from_pretrained(MODEL, min_pixels=256*28*28, max_pixels=256*28*28)
model = QwenVLLensModel(hf, proc)
lens = jlens.JacobianLens.load(LENS_P)

def excess_kurtosis(x):
    x = x.float(); m = x.mean(-1, keepdim=True); s = x.std(-1, keepdim=True) + 1e-8
    return (((x - m) / s) ** 4).mean(-1) - 3.0

imgs = sorted(glob.glob(os.path.join(DOCSET, "*.png")))
per_layer = {}; readouts = {}; results = {"lens": os.path.basename(LENS_P), "model": MODEL,
    "revision": REV, "torch": torch.__version__, "n_docs": len(imgs)}
for p in imgs:
    prompt = f"IMG::{p}::Transcribe this document as plain text."
    ids = model.encode(prompt)
    with torch.no_grad():
        h = model.forward(ids)                     # [1,T,d]
        true_next = model.unembed(h[:, :-1]).argmax(-1)   # model's own prediction per position
        lg, _, _ = lens.apply(model, prompt, positions=list(range(-min(64, ids.shape[1]-17), 0)))
    span = torch.zeros(ids.shape[1], dtype=torch.bool)
    st, en = model_span = getattr(model, "_last_img_span", (0, 0)) if hasattr(model, "_last_img_span") else (0, 0)
    img_mask = (ids[0] == 151655).cpu()
    true_next = true_next.cpu()
    for L, v in lg.items():                        # v: [P, vocab]
        v = v.float().cpu()
        d = per_layer.setdefault(int(L), {"kurt": [], "kurt_img": [], "kurt_txt": [], "agree": []})
        k = excess_kurtosis(v)
        pos_idx = torch.arange(ids.shape[1] - v.shape[0], ids.shape[1])
        m = img_mask[pos_idx]
        d["kurt"].append(k.mean().item())
        if m.any():  d["kurt_img"].append(k[m].mean().item())
        if (~m).any(): d["kurt_txt"].append(k[~m].mean().item())
        pred = v.argmax(-1)
        tgt = true_next[0, pos_idx.clamp(max=true_next.shape[1]-1)]
        d["agree"].append((pred == tgt).float().mean().item())
    top = lens.apply(model, prompt, positions=[-1])[0]
    readouts[os.path.basename(p)] = {str(L): [proc.tokenizer.decode([t]) for t in v[0].topk(10).indices]
                                     for L, v in sorted(top.items())}
results["per_layer"] = {L: {k: (sum(x)/len(x) if x else None) for k, x in d.items()} for L, d in sorted(per_layer.items())}
results["readouts_last_token"] = readouts

others = [q for q in glob.glob(os.path.join(ROOT, "data", "lenses", "*.e1.shard*.pt")) if q != LENS_P]
if others:
    agree = {}
    other = jlens.JacobianLens.load(sorted(others)[0])
    for p in imgs[:8]:
        prompt = f"IMG::{p}::Transcribe this document as plain text."
        a, _, _ = lens.apply(model, prompt, positions=[-1])
        b, _, _ = other.apply(model, prompt, positions=[-1])
        for L in a:
            if L in b:
                agree.setdefault(int(L), []).append(int(a[L][0].argmax() == b[L][0].argmax()))
    results["shard_agreement_vs_" + os.path.basename(sorted(others)[0])] = {L: sum(v)/len(v) for L, v in sorted(agree.items())}

out = os.path.join(ROOT, "artifacts", "evidence", "tables", f"{TAG}_diag.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(results, open(out, "w"), indent=1)
print(f"[diag] -> {out}", flush=True)
