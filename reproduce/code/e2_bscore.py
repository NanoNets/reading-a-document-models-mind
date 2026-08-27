"""E2: broadcast (B) vs verbalizability (kappa), per prereg H1.
Populations (per layer): P1 MLP-neuron output rows, P3 isotropic random,
P4 covariance-matched random (cov from live doc forwards), P5 J-lens vectors.
(P2 probe directions arrive with E3 and are appended in a second pass.)
Metrics: kappa(d,l)=excess kurtosis of W_U(J_l d); B_mlp=||MLP_{l+1}(post_ln(d))||
normalized by isotropic median; B_attn_gain=mean_h ||W_OV_h d|| normalized same way.
Output: per-(layer,population) arrays -> E2_B_kappa.json. H1 stats computed locally.
"""
import json, os, sys, glob
import torch, transformers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jlens
from jlens_vlm import QwenVLLensModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYERS = [12, 16, 21, 26, 30]          # workspace band per triangulated map
NDIR = 1500
dev = "cuda" if torch.cuda.is_available() else sys.exit("cuda required")
g = torch.Generator(device="cpu").manual_seed(20260826)

hf = transformers.AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", revision="66285546d2b821cf421d4f5eb2576359d3770cd3",
    dtype=torch.float32).to(dev)
proc = transformers.AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct",
    min_pixels=256*28*28, max_pixels=256*28*28)
model = QwenVLLensModel(hf, proc)
lens = jlens.JacobianLens.load(os.path.join(ROOT, "data/lenses/qwen2.5-vl-3b-instruct.e1.pt"))
Jmats = {int(L): m.float().to(dev) for L, m in (lens.__dict__.get("jacobians") or
         {k: v for k, v in torch.load(os.path.join(ROOT, "data/lenses/qwen2.5-vl-3b-instruct.e1.pt"),
          map_location=dev, weights_only=False).items() if hasattr(v, "shape")}).items()}
W_U = model._wu.float().to(dev)        # tied, verified

# --- residual covariance per layer from live doc forwards ---
docs = sorted(glob.glob(os.path.join(ROOT, "data/diag_docs/*.png")))
cov = {L: torch.zeros(2048, 2048, device=dev) for L in LAYERS}
n_pos = 0
from jlens.hooks import ActivationRecorder
for p in docs:
    ids = model.encode(f"IMG::{p}::Transcribe this document as plain text.")
    rec = ActivationRecorder(model.layers, LAYERS)
    with torch.no_grad(), rec:
        model.forward(ids)
    for L in LAYERS:
        h = rec.activations[L][0].float()          # [T, d]
        h = h - h.mean(0, keepdim=True)
        cov[L] += h.T @ h
    n_pos += ids.shape[1]
for L in LAYERS: cov[L] /= max(1, n_pos)
chol = {L: torch.linalg.cholesky(cov[L] + 1e-4*torch.eye(2048, device=dev)) for L in LAYERS}
print(f"[e2] cov from {len(docs)} docs, {n_pos} positions", flush=True)

def unit(x): return x / (x.norm(dim=-1, keepdim=True) + 1e-9)

def kappa_metrics(D, L, chunk=300):    # chunked: [n, vocab] intermediates never exceed chunk x vocab
    kurts, t1s = [], []
    for i in range(0, D.shape[0], chunk):
        V = (W_U @ (Jmats[L] @ D[i:i+chunk].T)).T.float()
        m = V.mean(-1, keepdim=True); sd = V.std(-1, keepdim=True) + 1e-8
        Z = (V - m) / sd
        kurts.append(((Z ** 4).mean(-1) - 3.0)); t1s.append(Z.max(-1).values)
        del V, Z
    return {"kappa": torch.cat(kurts), "top1_z": torch.cat(t1s)}

_LENSNORM = {}
def loading(D, L):
    """J-space loading (the paper's own measure): max |cos(d, v_tok)| over the vocab.
    Provable ceiling: P5 -> 1.0. Analytic null for random d: ~sqrt(2 ln V / dim) ~= 0.11."""
    if L not in _LENSNORM:
        _LENSNORM.clear()                             # single-entry cache: 1.2GB per layer
        torch.cuda.empty_cache()
        V = (W_U @ Jmats[L])                          # [vocab, d]
        _LENSNORM[L] = V / (V.norm(dim=1, keepdim=True) + 1e-9)
    return (_LENSNORM[L] @ D.T).abs().max(0).values   # [n]

def b_mlp(D, L, iso_med=None, raw=False):
    blk = model.layers[L + 1]
    with torch.no_grad():
        out = blk.mlp(D if raw else blk.post_attention_layernorm(D))
    n = out.norm(dim=-1)
    return n if iso_med is None else n / iso_med

def b_attn(D, L, iso_med=None):
    # GQA-correct OV gain (audit 2026-08-26): Qwen2.5-VL-3B has 16 query heads but
    # only 2 KV heads; query head q reads KV head q // (nh // n_kv). The previous
    # version reshaped the 256-dim v-output into 16 fake 16-dim heads and only ever
    # touched o_proj's first 256 columns.
    blk = model.layers[L + 1].self_attn
    Wv, Wo = blk.v_proj.weight, blk.o_proj.weight      # [n_kv*hd, d], [d, nh*hd]
    nh = 16; hd = Wo.shape[1] // nh
    n_kv = Wv.shape[0] // hd; grp = nh // n_kv
    with torch.no_grad():
        v = (Wv @ D.T).T.view(-1, n_kv, hd)            # [n, n_kv, hd]
        gains = torch.stack([(Wo[:, q*hd:(q+1)*hd] @ v[:, q // grp].T).norm(dim=0)
                             for q in range(nh)], 1)   # [n, nh]
    m = gains.mean(1)
    return m if iso_med is None else m / iso_med

res = {}
for L in LAYERS:
    iso = unit(torch.randn(NDIR, 2048, generator=g).to(dev))
    iso_mlp_med = b_mlp(iso, L).median()
    iso_mlp_raw_med = b_mlp(iso, L, raw=True).median()
    iso_att_med = b_attn(iso, L).median()
    pops = {}
    dff = model.layers[L + 1].mlp.down_proj.weight.shape[1]
    idx = torch.randperm(dff, generator=g)[:NDIR]
    pops["P1_mlp_neuron_out"] = unit(model.layers[L].mlp.down_proj.weight.T[idx].float())
    pops["P3_isotropic"] = iso
    pops["P4_cov_matched"] = unit((chol[L] @ torch.randn(2048, NDIR, generator=g).to(dev)).T)
    vidx = torch.randperm(W_U.shape[0], generator=g)[:NDIR]
    pops["P5_jlens_vectors"] = unit((W_U[vidx] @ Jmats[L]).float())
    res[L] = {}
    for name, D in pops.items():
        D = D.to(dev)
        km = kappa_metrics(D, L)
        res[L][name] = {
            "kappa": km["kappa"].cpu().tolist(),
            "top1_z": km["top1_z"].cpu().tolist(),
            "loading": loading(D, L).cpu().tolist(),
            "b_mlp": b_mlp(D, L, iso_mlp_med).cpu().tolist(),
            "b_mlp_raw": b_mlp(D, L, iso_mlp_raw_med, raw=True).cpu().tolist(),
            "b_attn": b_attn(D, L, iso_att_med).cpu().tolist(),
        }
    # IN-JOB CALIBRATION GATE: P5 (one-token directions) must dominate the null on nameability
    import statistics as _st
    for metric in ("top1_z", "loading"):
        p5m = _st.median(res[L]["P5_jlens_vectors"][metric]); p4m = _st.median(res[L]["P4_cov_matched"][metric])
        print(f"[e2] L{L} CALIB {metric}: P5(ceiling)={p5m:.3f} vs P4(null)={p4m:.3f} ratio={p5m/max(p4m,1e-9):.1f}x", flush=True)
    print(f"[e2] L{L} done", flush=True)

json.dump({"layers": LAYERS, "n_dirs": NDIR, "seed": 20260826,
           "lens": "qwen2.5-vl-3b-instruct.e1.pt", "cov_docs": len(docs),
           "results": {str(k): v for k, v in res.items()}},
          open(os.path.join(ROOT, "artifacts/evidence/tables/E2_B_kappa.json"), "w"))
print("[e2] E2 COMPLETE", flush=True)
