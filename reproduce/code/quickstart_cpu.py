"""CPU-safe quickstart: load a released lens and read residual-stream states.

Runs on a laptop, no GPU: the lens file carries the transport matrices J_l for
every decoder layer, and the model's tied unembedding + final RMSNorm weights
are read straight out of the safetensors checkpoint (two tensors — the 3B model
itself is never instantiated and never run).

What it shows:
  1. a layer-0 residual state (a token embedding IS the layer-0 state in this
     architecture) read out through the lens at several depths;
  2. precomputed probe directions from E3 (data/probe_dirs.pt) read out at the
     layer they were fit — the paper's "nameability" measurement in miniature.
     Low-loading directions decode to unrelated tokens; that is the finding,
     not a bug.

Producing NEW states for step 2 (i.e., reading an actual document) requires a
forward pass of the 3B VLM — see the GPU quickstart in README.md.

Usage:  python code/quickstart_cpu.py
Env:    LENS (default data/lenses/qwen2.5-vl-3b-instruct.e1.pt)
"""
import os, sys, json
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code", "vendor-jacobian-lens"))
import jlens

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
REV   = "66285546d2b821cf421d4f5eb2576359d3770cd3"   # pinned (see README)
LENS  = os.environ.get("LENS", os.path.join(ROOT, "data", "lenses", "qwen2.5-vl-3b-instruct.e1.pt"))

lens = jlens.JacobianLens.load(LENS)
print(f"[qs] lens: {os.path.basename(LENS)} | {len(lens.jacobians)} layers | "
      f"d_model={lens.d_model} | fit on {lens.n_prompts} prompts")

# --- unembedding + final norm, straight from the checkpoint (no model) -------
from huggingface_hub import hf_hub_download
from safetensors import safe_open

idx = json.load(open(hf_hub_download(MODEL, "model.safetensors.index.json", revision=REV)))
need = {"model.embed_tokens.weight", "model.norm.weight"}      # head is tied to embed
tensors = {}
for name in need:
    shard = hf_hub_download(MODEL, idx["weight_map"][name], revision=REV)
    with safe_open(shard, framework="pt") as f:
        tensors[name] = f.get_tensor(name).float()
W_U   = tensors["model.embed_tokens.weight"]                   # [vocab, d] (tied)
w_nrm = tensors["model.norm.weight"]                           # [d]

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)

def readout(h, layer, k=8):
    """Top-k tokens the lens says state h (at `layer`) is disposed to produce."""
    v = lens.jacobians[layer].float() @ h.float()               # transport to final basis
    v = v / torch.sqrt((v * v).mean() + 1e-6) * w_nrm           # final RMSNorm
    return [tok.decode([t]) for t in (W_U @ v).topk(k).indices]

# --- 1. a genuine layer-0 residual state: the embedding of " invoice" --------
tid = tok.encode(" invoice", add_special_tokens=False)[0]
h0 = W_U[tid]                                                   # embed row == layer-0 state
for L in (0, 12, 21, 30):
    print(f"[qs] ' invoice' state read at layer {L:>2}: {readout(h0, L)}")

# --- 2. precomputed E3 probe directions at their fit layer -------------------
pd_path = os.path.join(ROOT, "data", "probe_dirs.pt")
if os.path.exists(pd_path):
    dirs = torch.load(pd_path, map_location="cpu", weights_only=True)
    for key in ("currency_L26", "n_line_items_L21", "document_type_L16"):
        if key in dirs:
            L = int(key.rsplit("_L", 1)[1])
            print(f"[qs] probe '{key}' read at layer {L}: {readout(dirs[key], L)}")
print("[qs] done — see README.md for the GPU path that reads real documents.")
