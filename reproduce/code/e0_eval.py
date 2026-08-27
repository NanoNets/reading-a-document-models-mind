"""E0 part 2: run the paper's six bundled lens-eval sets, J-lens vs logit lens.

Usage: E0_LENS=data/lenses/<file>.pt [E0_MODEL=...] python code/e0_eval.py
Metric (per the vendor data/evaluations/README): pass@k = mean over items of
the fraction of `intermediates` whose min-over-layers lens rank <= k, read at
a single position: the last prompt token (poetry: the last newline token).
We report pass@k for k in {1,2,5,10,20,50,100} and a normalized AUC over
log k, for BOTH use_jacobian=True (J-lens) and False (logit lens).
E0 pass criterion (c.f. artifacts/logic/experiments.md): J-lens AUC >= logit
lens AUC on >= 4 of 6 sets.
Applies forward passes only — laptop MPS is sufficient.
"""
import json, math, os, glob
import torch, transformers, jlens

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get("E0_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
LENS  = os.environ["E0_LENS"]
EVDIR = os.path.join(ROOT, "code", "vendor-jacobian-lens", "data", "evaluations")
OUT   = os.path.join(ROOT, "artifacts", "evidence", "tables", "E0_lens_evals.json")
KS = [1, 2, 5, 10, 20, 50, 100]

dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(dev)
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
model = jlens.from_hf(hf, tok)
lens = jlens.JacobianLens.load(LENS) if hasattr(jlens.JacobianLens, "load") else torch.load(LENS, weights_only=False)
print(f"[e0-eval] model={MODEL} lens={LENS} device={dev}", flush=True)

def variants(word):
    """Candidate single-token encodings of an intermediate word."""
    cands = {word, " " + word, word.lower(), " " + word.lower(),
             word.capitalize(), " " + word.capitalize()}
    ids = set()
    for c in cands:
        t = tok.encode(c, add_special_tokens=False)
        if len(t) == 1:
            ids.add(t[0])
    return ids

def readout_position(set_slug, prompt):
    if set_slug == "poetry":
        ids = tok.encode(prompt, add_special_tokens=False)
        nl = [i for i, t in enumerate(ids) if "\n" in tok.decode([t])]
        return nl[-1] - len(ids) if nl else -1   # negative index into sequence
    return -1

def min_rank_over_layers(lens_logits, token_ids):
    """Best (lowest) rank of any variant id, over all layers, at the single position."""
    best = math.inf
    for L, lg in lens_logits.items():
        v = lg[0]                                 # [vocab]
        for tid in token_ids:
            r = int((v > v[tid]).sum().item()) + 1
            best = min(best, r)
    return best

results = {}
for path in sorted(glob.glob(os.path.join(EVDIR, "lens-eval-*.json"))):
    slug = os.path.basename(path)[len("lens-eval-"):-len(".json")]
    data = json.load(open(path))
    items = data["items"] if isinstance(data, dict) and "items" in data else \
            (data[list(data)[0]] if isinstance(data, dict) else data)
    per_mode = {}
    for use_jac, mode in [(True, "jlens"), (False, "logit_lens")]:
        ranks_all = []
        for it in items:
            prompt = it["prompt"]
            if not isinstance(prompt, str):        # multi-turn convention
                prompt = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            pos = readout_position(slug, prompt)
            with torch.no_grad():
                lg, _, _ = lens.apply(model, prompt, positions=[pos], use_jacobian=use_jac)
            item_ranks = []
            for inter in it["intermediates"]:
                ids = variants(inter)
                item_ranks.append(min_rank_over_layers(lg, ids) if ids else None)
            ranks_all.append(item_ranks)
        def pass_at(k):
            fr = [sum(1 for r in irs if r is not None and r <= k) / max(1, len([r for r in irs if r is not None]))
                  for irs in ranks_all if any(r is not None for r in irs)]
            return sum(fr) / len(fr) if fr else 0.0
        pk = {k: round(pass_at(k), 4) for k in KS}
        auc = sum(pk[k] for k in KS) / len(KS)     # simple mean over log-spaced k
        skipped = sum(1 for irs in ranks_all for r in irs if r is None)
        per_mode[mode] = {"pass_at_k": pk, "auc_logk": round(auc, 4),
                          "n_items": len(items), "n_multitoken_skipped": skipped}
        print(f"[e0-eval] {slug:14s} {mode:10s} pass@1={pk[1]:.3f} pass@10={pk[10]:.3f} auc={auc:.3f} skipped={skipped}", flush=True)
    per_mode["jlens_wins"] = per_mode["jlens"]["auc_logk"] >= per_mode["logit_lens"]["auc_logk"]
    results[slug] = per_mode

wins = sum(1 for v in results.values() if v["jlens_wins"])
summary = {"model": MODEL, "lens": os.path.basename(LENS), "device": dev,
           "sets": results, "jlens_wins_of_6": wins,
           "E0_pass_criterion_b": wins >= 4, "ks": KS,
           "note": "min-over-ALL-layers (band not yet identified at E0 stage); "
                   "multi-token intermediates skipped and counted"}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(summary, open(OUT, "w"), indent=1)
print(f"[e0-eval] J-lens wins {wins}/6 -> criterion(b) {'PASS' if wins>=4 else 'FAIL'} -> {OUT}", flush=True)
