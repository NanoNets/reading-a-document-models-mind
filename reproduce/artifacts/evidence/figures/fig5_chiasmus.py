"""Figure 5: the fine-tuning chiasmus — maps change early, content changes late.
Reads ONLY artifacts/evidence/tables/E1_twin_vs_base.json (lens-map cosine by layer)
and artifacts/evidence/tables/exploratory/chiasmus_state_cos.json (median cosine of
base-vs-twin residual states, post-image text positions, 10 docs). Regenerate:
  .venv/bin/python artifacts/evidence/figures/fig5_chiasmus.py
The two curves answer the same question — "where did fine-tuning change the model?" —
about two different objects (the lens MAP J_l vs the residual STATES h_l), and they
cross: each metric alone gives the opposite story.
Palette: dataviz reference slots (blue #2a78d6, orange #eb6834), validated.
Status: FINAL (H-TYPEWRITER kill test, 2026-08-26).
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
T = os.path.join(ROOT, "artifacts/evidence/tables")
cosJ = json.load(open(os.path.join(T, "E1_twin_vs_base.json")))["cos_by_layer"]
cosH = json.load(open(os.path.join(T, "exploratory/chiasmus_state_cos.json")))["state_cos_by_layer"]

fig, ax = plt.subplots(figsize=(6.8, 4.0), dpi=200)
Lj = sorted(int(k) for k in cosJ)
ax.plot(Lj, [cosJ[str(L)] for L in Lj], color="#2a78d6", lw=2,
        label="lens map  cos($J_\\ell^{base}$, $J_\\ell^{tuned}$)")
Lh = sorted(int(k) for k in cosH)
ax.plot(Lh, [cosH[str(L)] for L in Lh], color="#eb6834", lw=2,
        label="residual state  cos($h_\\ell^{base}$, $h_\\ell^{tuned}$)")
ax.set_xlabel("layer", fontsize=9)
ax.set_ylabel("base-vs-fine-tune cosine similarity", fontsize=9)
ax.set_ylim(0.80, 1.005)
ax.axhline(1.0, color="#c9c8c2", lw=0.8)
ax.annotate("maps rewired,\nstates near-identical", xy=(4.3, 0.905), fontsize=8,
            color="#52514e", ha="center")
ax.annotate("map preserved,\nstates most divergent", xy=(29.5, 0.868), fontsize=8,
            color="#52514e", ha="center")
ax.legend(frameon=False, fontsize=8.5, loc="lower left")
ax.spines[["top", "right"]].set_visible(False)
for s in ax.spines.values(): s.set_color("#c9c8c2")
ax.grid(axis="y", color="#eceae4", lw=0.6); ax.set_axisbelow(True)
ax.tick_params(colors="#52514e", labelsize=8)
ax.set_title("Fine-tuning rebuilt the reader and kept the typewriter:\n"
             "the transformation changes where the content doesn't, and vice versa",
             fontsize=10, loc="left", color="#0b0b0b")
fig.tight_layout()
out = os.path.join(ROOT, "artifacts/evidence/figures/fig5_chiasmus")
fig.savefig(out + ".png", bbox_inches="tight"); fig.savefig(out + ".svg", bbox_inches="tight")
print("saved", out + ".{png,svg}")
