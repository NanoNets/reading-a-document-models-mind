"""Figure 2: broadcast vs nameability — the H1 dissociation, with its calibration.
Reads ONLY artifacts/evidence/tables/E2_B_kappa.json + E2_H1_stats_v4.json. Regenerate:
  .venv/bin/python artifacts/evidence/figures/fig2_b_vs_loading.py
Palette: dataviz reference slots (blue #2a78d6, orange #eb6834, gray #8a8988), validated.
One point = one direction (500/population subsampled deterministically from 1500).
x = loading (max cosine to any J-lens vector; 1.0 = has a single-token name,
~0.08 = covariance-matched noise floor). y = B_mlp broadcast gain (normed).
Status: FINAL DATA (merged lens, GQA-corrected E2 rerun, seed 20260826); f box shows excess over the below-floor base rate (audit 2026-08-26).
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
T = os.path.join(ROOT, "artifacts/evidence/tables")
d = json.load(open(os.path.join(T, "E2_B_kappa.json")))
h = json.load(open(os.path.join(T, "E2_H1_stats_v4.json")))

LAYERS = ["12", "16", "21", "26", "30"]
POPS = [("P1_mlp_neuron_out", "MLP neuron outputs (real computation)", "#2a78d6", 0.35),
        ("P4_cov_matched",    "covariance-matched random (null)",      "#8a8988", 0.30),
        ("P5_jlens_vectors",  "J-lens vectors (named by construction)", "#eb6834", 0.35)]
STRIDE = 3  # 1500 -> 500 points per population, deterministic

fig, axes = plt.subplots(1, 5, figsize=(12.5, 3.1), dpi=200, sharey=True, sharex=True)
for ax, L in zip(axes, LAYERS):
    for pop, label, color, alpha in POPS:
        r = d["results"][L][pop]
        x = r["loading"][::STRIDE]
        y = [max(v, 0.05) for v in r["b_mlp"][::STRIDE]]
        ax.scatter(x, y, s=4, color=color, alpha=alpha, linewidths=0,
                   label=label if L == "12" else None)
    null = h[L]["load_null"]
    ax.axvline(null, color="#8a8988", lw=0.8, ls=(0, (4, 3)))
    ax.axvline(1.0, color="#eb6834", lw=0.8, ls=(0, (4, 3)))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.045, 1.6); ax.set_ylim(0.05, 20)
    ax.set_title(f"layer {L}", fontsize=9, color="#0b0b0b")
    ax.text(0.97, 0.955, f"f = {h[L]['f']:.2f}\n+{h[L]['excess']:.2f} vs base", transform=ax.transAxes, fontsize=7.5,
            ha="right", va="top", color="#0b0b0b",
            bbox=dict(boxstyle="round,pad=0.25", fc="#f6f5f1", ec="#c9c8c2", lw=0.6))
    ax.spines[["top", "right"]].set_visible(False)
    for s in ax.spines.values(): s.set_color("#c9c8c2")
    ax.grid(axis="y", color="#eceae4", lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(colors="#52514e", labelsize=7.5)

axes[0].set_ylabel("broadcast gain $B_{mlp}$", fontsize=9)
axes[2].set_xlabel("nameability (loading: max cosine to any J-lens vector)", fontsize=9)
axes[0].text(h["12"]["load_null"] * 1.1, 0.062, "noise floor", fontsize=7, color="#52514e")
axes[0].text(0.62, 0.062, "named", fontsize=7, color="#eb6834")
fig.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=3,
           bbox_to_anchor=(0.5, 1.04), markerscale=3)
fig.suptitle("High-broadcast directions sit at the nameability noise floor in the workspace band,\n"
             "and only become nameable at the motor boundary (L26–30)",
             fontsize=10, x=0.01, ha="left", y=1.17, color="#0b0b0b")
fig.tight_layout()
out = os.path.join(ROOT, "artifacts/evidence/figures/fig2_b_vs_loading")
fig.savefig(out + ".png", bbox_inches="tight"); fig.savefig(out + ".svg", bbox_inches="tight")
print("saved", out + ".{png,svg}")
