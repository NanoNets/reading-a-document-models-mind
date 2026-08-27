"""H1 statistics from E2 B-kappa tables — the generating script for E2_H1_stats*.json.

Checked in 2026-08-26 after the adversarial audit flagged its absence (the numbers were
computed in-session; the definition below reproduces every value in E2_H1_stats_v3.json
to the third decimal, verified by the audit). Definition:
  B         = rank-average of (b_mlp, b_attn) within P1 (or b_mlp_raw for the raw variant)
  top decile= B >= 90th percentile of P1's B (ties included -> n = 150-152)
  floor     = median loading of P4_cov_matched at that layer
  f         = fraction of top-decile P1 directions with loading < floor
  rho       = Spearman(B, loading) over all of P1
Audit additions reported alongside: the below-floor BASE RATE over all of P1 (53.3% at
L21 — the registered 0.25 bar is uninformative), the top-decile EXCESS over base rate,
and a permutation p-value for that excess (20k draws).

Usage: .venv/bin/python code/compute_h1_stats.py [E2_B_kappa.json] [out.json]
"""
import json, os, sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "artifacts/evidence/tables/E2_B_kappa.json")
out_path = sys.argv[2] if len(sys.argv) > 2 else None

def ranks(t):
    r = torch.empty_like(t)
    r[t.argsort()] = torch.arange(len(t), dtype=t.dtype)
    return r

def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ra = (ra - ra.mean()) / ra.std(); rb = (rb - rb.mean()) / rb.std()
    return float((ra * rb).mean())

g = torch.Generator().manual_seed(20260826)
d = json.load(open(src))
res = {}
for L, pops in d["results"].items():
    p1, p4 = pops["P1_mlp_neuron_out"], pops["P4_cov_matched"]
    loading = torch.tensor(p1["loading"], dtype=torch.float64)
    floor = float(torch.tensor(p4["loading"], dtype=torch.float64).median())
    below = (loading < floor)
    base_rate = float(below.float().mean())
    row = {}
    for tag, keys in [("", ("b_mlp", "b_attn")), ("_rawgain", ("b_mlp_raw", "b_attn"))]:
        B = sum(ranks(torch.tensor(p1[k], dtype=torch.float64)) for k in keys) / len(keys)
        top = B >= torch.quantile(B, 0.9)
        f = float(below[top].float().mean())
        row["f" + tag] = round(f, 3)
        if not tag:
            row["rho"] = round(spearman(B, loading), 3)
            n_top = int(top.sum())
            excess = f - base_rate
            # permutation: random n_top-subsets of P1
            draws = torch.stack([below[torch.randperm(len(below), generator=g)[:n_top]].float().mean()
                                 for _ in range(20000)])
            p = float((draws >= f).float().mean())
            row.update(n_top=n_top, base_rate=round(base_rate, 3),
                       excess=round(excess, 3), perm_p=round(p, 5))
    row.update(load_med_P1=round(float(loading.median()), 3),
               load_null=round(floor, 3),
               load_ceiling=round(float(torch.tensor(pops["P5_jlens_vectors"]["loading"]).median()), 3))
    res[L] = row
    print(L, row)

if out_path:
    json.dump(res, open(out_path, "w"), indent=1)
    print("wrote", out_path)
