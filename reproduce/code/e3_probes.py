"""E3: paired lexicalizable/non-lexical probe battery on DocILE.
Captures mean-of-last-32-token residuals at the E2 layers over ~400 labeled pages,
fits ridge probes (70/30 split), then scores every probe DIRECTION for
nameability (loading) and broadcast (b_mlp, b_attn) — the P2 population for H1.
Outputs: E3_probes.json + data/probe_dirs.pt (E4's intervention targets).
"""
import json, os, sys, glob
import torch, transformers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jlens
from jlens_vlm import QwenVLLensModel
from jlens.hooks import ActivationRecorder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYERS = [12, 16, 21, 26, 30]
NDOCS = 400
g = torch.Generator().manual_seed(20260827)
dev = "cuda" if torch.cuda.is_available() else sys.exit("cuda required")

hf = transformers.AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", revision="66285546d2b821cf421d4f5eb2576359d3770cd3",
    dtype=torch.float32).to(dev)
proc = transformers.AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct",
    min_pixels=256*28*28, max_pixels=256*28*28)
model = QwenVLLensModel(hf, proc)
lens = jlens.JacobianLens.load(os.path.join(ROOT, "data/lenses/qwen2.5-vl-3b-instruct.e1.pt"))
_src = getattr(lens, "jacobians", None) or lens.__dict__.get("jacobians") or        {k: v for k, v in lens.__dict__.items() if isinstance(v, dict) and all(hasattr(x, "shape") for x in v.values())}.popitem()[1]
Jmats = {int(L): m.float().to(dev) for L, m in _src.items()}
print(f"[e3] lens layers: {sorted(Jmats)[:3]}..{sorted(Jmats)[-1]}", flush=True)
W_U = model._wu.float().to(dev)

labels = json.load(open(os.path.join(ROOT, "data/e3_labels.json")))
pages = sorted(glob.glob(os.path.join(ROOT, "data/e1_pages/*.png")))
pages = [p for p in pages if os.path.basename(p)[:-4] in labels][:NDOCS]
print(f"[e3] {len(pages)} labeled pages", flush=True)

# ---- capture ----
feats = {L: [] for L in LAYERS}; kept = []
for i, p in enumerate(pages):
    ids = model.encode(f"IMG::{p}::Transcribe this document as plain text.")
    rec = ActivationRecorder(model.layers, LAYERS)
    with torch.no_grad(), rec:
        model.forward(ids)
    for L in LAYERS:
        feats[L].append(rec.activations[L][0, -32:].float().mean(0).cpu())
    kept.append(os.path.basename(p)[:-4])
    if (i+1) % 50 == 0: print(f"[e3] captured {i+1}", flush=True)
X = {L: torch.stack(feats[L]) for L in LAYERS}          # [N, d]

# ---- variables ----
import collections
def cat_targets(key, min_n=8):
    vals = [labels[d][key] for d in kept]
    keep = {v for v, c in collections.Counter(vals).items() if v is not None and c >= min_n}
    classes = sorted(keep)
    y = torch.tensor([classes.index(v) if v in keep else -1 for v in vals])
    return y, classes
def cont_targets(key):
    vals = [labels[d][key] for d in kept]
    y = torch.tensor([float(v) if v is not None else float("nan") for v in vals])
    return y

VARS = [("document_type", "cat", True), ("currency", "cat", True),
        ("cluster_id", "cat", False), ("table_y_centroid", "cont", False),
        ("table_height_frac", "cont", False), ("n_line_items", "cont", False),
        # audit 2026-08-26: height and row count correlate 0.78 in labels; the
        # residualized pair separates "geometry" from "countable rows"
        ("table_height_resid", "cont", False), ("n_line_items_resid", "cont", False)]

RESID = {"table_height_resid": ("table_height_frac", "n_line_items"),
         "n_line_items_resid": ("n_line_items", "table_height_frac")}
def resid_targets(key):
    src, on = RESID[key]
    y, z = cont_targets(src), cont_targets(on)
    m = ~(torch.isnan(y) | torch.isnan(z))
    trm = tr[m[tr]]                       # train-only fit, no val leakage
    zc = z[trm] - z[trm].mean()
    b = ((y[trm] - y[trm].mean()) * zc).sum() / (zc * zc).sum()
    a = y[trm].mean() - b * z[trm].mean()
    r = y - (a + b * z); r[~m] = float("nan")
    return r

def ridge_dir(Xtr, ytr, lam=10.0):
    Xc = Xtr - Xtr.mean(0, keepdim=True)
    yc = ytr - ytr.mean()
    w = torch.linalg.solve(Xc.T @ Xc + lam*torch.eye(Xc.shape[1]), Xc.T @ yc)
    return w

_LN = {}
def loading(d_unit, L):
    if L not in _LN:
        _LN.clear(); torch.cuda.empty_cache()
        V = (W_U @ Jmats[L]); _LN[L] = (V / (V.norm(dim=1, keepdim=True)+1e-9))
    return float((_LN[L] @ d_unit.to(dev)).abs().max().item())
def b_scores(d_unit, L):
    blk = model.layers[L+1]
    D = d_unit[None].to(dev)
    iso = torch.randn(256, 2048, generator=g).to(dev)
    iso = iso / iso.norm(dim=1, keepdim=True)
    with torch.no_grad():
        bm = blk.mlp(blk.post_attention_layernorm(D)).norm() / blk.mlp(blk.post_attention_layernorm(iso)).norm(dim=-1).median()
        Wv, Wo = blk.self_attn.v_proj.weight, blk.self_attn.o_proj.weight
        # GQA-correct (audit 2026-08-26): 16 Q heads, 2 KV heads; q reads KV q//8
        nh = 16; hd = Wo.shape[1] // nh
        n_kv = Wv.shape[0] // hd; grp = nh // n_kv
        def att(vecs):
            v = (Wv @ vecs.T).T.view(vecs.shape[0], n_kv, hd)
            return torch.stack([(Wo[:, q*hd:(q+1)*hd] @ v[:, q//grp].T).norm(dim=0)
                                for q in range(nh)], 1).mean(1)
        ba = att(D)[0] / att(iso).median()
    return float(bm.item()), float(ba.item())

results = {}; probe_dirs = {}
perm = torch.randperm(len(kept), generator=g)
ntr = int(0.7*len(kept))
tr, va = perm[:ntr], perm[ntr:]
for name, kind, named in VARS:
    results[name] = {"named": named, "kind": kind, "per_layer": {}}
    for L in LAYERS:
        Xl = X[L]
        if kind == "cont":
            y = resid_targets(name) if name in RESID else cont_targets(name)
            m = ~torch.isnan(y)
            trm = tr[m[tr]]; vam = va[m[va]]
            if len(trm) < 40 or len(vam) < 15: continue
            w = ridge_dir(Xl[trm], y[trm])
            pred = (Xl[vam] - Xl[trm].mean(0)) @ w + y[trm].mean()
            ss_res = ((pred - y[vam])**2).sum(); ss_tot = ((y[vam]-y[vam].mean())**2).sum()
            score = float(1 - ss_res/ss_tot)
        else:
            y, classes = cat_targets(name)
            m = y >= 0
            trm = tr[m[tr]]; vam = va[m[va]]
            if len(classes) < 2 or len(trm) < 40 or len(vam) < 15: continue
            # one-vs-rest ridge on the largest class -> AUC
            y0 = (y == 0).float()
            w = ridge_dir(Xl[trm], y0[trm])
            s = (Xl[vam] - Xl[trm].mean(0)) @ w
            pos = s[y0[vam] == 1]; neg = s[y0[vam] == 0]
            if len(pos) == 0 or len(neg) == 0: continue
            score = float((pos[:, None] > neg[None, :]).float().mean().item())
        d_unit = (w / w.norm())
        ld = loading(d_unit, L); bm, ba = b_scores(d_unit, L)
        results[name]["per_layer"][L] = {"score": round(score, 3), "loading": round(ld, 3),
                                          "b_mlp": round(bm, 2), "b_attn": round(ba, 2),
                                          "n_train": int(len(trm)), "n_val": int(len(vam))}
        probe_dirs[f"{name}_L{L}"] = d_unit.cpu()
        print(f"[e3] {name} L{L}: score={score:.3f} loading={ld:.3f} B=({bm:.2f},{ba:.2f})", flush=True)

json.dump({"n_docs": len(kept), "layers": LAYERS, "seed": 20260827,
           "note": "language dropped (all-eng corpus, zero variance)",
           "results": results},
          open(os.path.join(ROOT, "artifacts/evidence/tables/E3_probes.json"), "w"), indent=1)
torch.save(probe_dirs, os.path.join(ROOT, "data/probe_dirs.pt"))
print("[e3] E3 COMPLETE", flush=True)
