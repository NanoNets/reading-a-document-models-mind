"""Figure 1: effective dimensionality of the Jacobian lens by normalized depth.
Reads ONLY artifacts/evidence/tables/E1_effdim_band.json + E1_merged_lens_anatomy.json.
Regenerate:
  .venv/bin/python artifacts/evidence/figures/fig1_band_effdim.py
Palette: dataviz reference slots (#2a78d6 blue, #eb6834 orange, #1c8259 green), validated.
Status: FINAL — adds the merged-lens exact W_U-projected curve (the readable-subspace
variant); the shard0 J_l curve is retained (shard-vs-merged agreement 0.999).
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
d = json.load(open(os.path.join(ROOT, "artifacts/evidence/tables/E1_effdim_band.json")))
m = json.load(open(os.path.join(ROOT, "artifacts/evidence/tables/E1_merged_lens_anatomy.json")))

series = [("Document VLM, $J_\\ell$ (Qwen2.5-VL-3B, 36L)", d["vlm_shard0"], "#2a78d6", 35),
          ("Text model, $J_\\ell$ (Qwen2.5-0.5B, 24L)",    d["text_0.5b"],  "#eb6834", 23),
          ("Document VLM, $W_U J_\\ell$ (merged, exact)",  m["exact_WU_effdim"], "#1c8259", 35)]

fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200)
for name, prof, color, nsrc in series:
    Ls = sorted(int(k) for k in prof)
    x = [100 * L / nsrc for L in Ls]
    y = [prof[str(L)] for L in Ls]
    ax.plot(x, y, color=color, lw=2, label=name)

ax.set_xlabel("Depth through source layers (%)", fontsize=9)
ax.set_ylabel("Fraction of dims for 90% of $J_\\ell$ variance", fontsize=9)
ax.set_xlim(0, 100); ax.set_ylim(0, 0.85)
ax.spines[["top", "right"]].set_visible(False)
for s in ax.spines.values(): s.set_color("#c9c8c2")
ax.grid(axis="y", color="#eceae4", lw=0.7); ax.set_axisbelow(True)
ax.tick_params(colors="#52514e", labelsize=8)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax.set_title("Lens dimensionality fans out later, from a deeper collapse,\nin the document VLM", fontsize=10, loc="left", color="#0b0b0b")
fig.tight_layout()
out = os.path.join(ROOT, "artifacts/evidence/figures/fig1_band_effdim")
fig.savefig(out + ".png"); fig.savefig(out + ".svg")
print("saved", out + ".{png,svg}")
