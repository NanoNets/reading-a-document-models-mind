"""Figure 4: what OCR fine-tuning did to the lens — rewiring map + late-band takeover.
Reads ONLY artifacts/evidence/tables/{E1_twin_vs_base.json, e1merged_diag.json,
twinmerged_diag.json}. Regenerate:
  .venv/bin/python artifacts/evidence/figures/fig4_twin_timeline.py
Panel A: per-layer cosine between base and twin lens (where fine-tuning moved things).
Panel B: markup occupancy — fraction of top-10 readout tokens that are output-format
markup (table tags, rules, separators, pure punctuation), receipt docs (n=4), by layer.
NOTE this script is also the reproducible extractor behind the C12 CORRECTION
(LEDGER 2026-08-26): schema first-appearance shifts are modest (Date 7->6, Name 15->9
median, Number 17->15), printed below; the verified reorganization is the late-band
content replacement that Panel B shows.
Palette: dataviz reference slots (blue #2a78d6 base, orange #eb6834 twin), validated.
Status: FINAL DATA (merged lenses, 26-doc diag set; receipt subset n=4).
"""
import json, os, re, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
T = os.path.join(ROOT, "artifacts/evidence/tables")
cos = json.load(open(os.path.join(T, "E1_twin_vs_base.json")))["cos_by_layer"]
base = json.load(open(os.path.join(T, "e1merged_diag.json")))["readouts_last_token"]
twin = json.load(open(os.path.join(T, "twinmerged_diag.json")))["readouts_last_token"]
RECEIPTS = [k for k in base if "receipt" in k]

MARKUP_RE = re.compile(r"^[\s\-=~#*!/|_{}().,:;：`'\"<>\[\]]+$")
def is_markup(t):
    return bool(MARKUP_RE.match(t)) or "<table" in t.lower() or "```" in t

def occupancy(readouts):
    layers = sorted({int(L) for doc in RECEIPTS for L in readouts[doc]})
    return layers, [st.mean(sum(is_markup(t) for t in readouts[doc][str(L)]) /
                            len(readouts[doc][str(L)]) for doc in RECEIPTS)
                    for L in layers]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.5, 3.6), dpi=200)
# ---- Panel A: lens rewiring by layer ----
Ls = sorted(int(k) for k in cos)
axA.plot(Ls, [cos[str(L)] for L in Ls], color="#2a78d6", lw=2)
axA.axhline(1.0, color="#c9c8c2", lw=0.8)
axA.set_xlabel("layer", fontsize=9)
axA.set_ylabel("cosine(base lens, OCR-tuned lens)", fontsize=9)
axA.set_ylim(0.925, 1.003)
axA.annotate("reading/representation\nlayers rewired", xy=(8, 0.952), fontsize=8,
             color="#52514e", ha="center")
axA.annotate("typing map\npreserved", xy=(31.5, 0.9635), fontsize=8,
             color="#52514e", ha="center")
axA.set_title("A — where fine-tuning moved the lens", fontsize=10, loc="left", color="#0b0b0b")

# ---- Panel B: markup occupancy of the readout, base vs twin ----
for readouts, label, color in [(base, "base VLM", "#2a78d6"),
                               (twin, "OCR fine-tune", "#eb6834")]:
    xs, ys = occupancy(readouts)
    axB.plot(xs, ys, color=color, lw=2, label=label)
axB.set_xlabel("layer", fontsize=9)
axB.set_ylabel("markup fraction of top-10 readout", fontsize=9)
axB.set_ylim(0, 1.0)
axB.legend(frameon=False, fontsize=8.5, loc="upper left")
axB.set_title("B — the fine-tune's late band fills with output markup", fontsize=10,
              loc="left", color="#0b0b0b")

for ax in (axA, axB):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ax.spines.values(): s.set_color("#c9c8c2")
    ax.grid(axis="y", color="#eceae4", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#52514e", labelsize=8)

fig.suptitle("OCR fine-tuning rewires the reading layers and hands the late workspace to\n"
             "output-format planning, while preserving the output machinery",
             fontsize=10.5, x=0.01, ha="left", y=1.06, color="#0b0b0b")
fig.tight_layout()
out = os.path.join(ROOT, "artifacts/evidence/figures/fig4_twin_timeline")
fig.savefig(out + ".png", bbox_inches="tight"); fig.savefig(out + ".svg", bbox_inches="tight")
print("saved", out + ".{png,svg}")

# ---- C12-correction extractor: schema first appearances (exact-match, per doc) ----
def first_layers(readouts, tok):
    out = []
    for doc in RECEIPTS:
        for L in sorted(readouts[doc], key=int):
            if any(t.strip() == tok for t in readouts[doc][str(L)]):
                out.append(int(L)); break
    return out
for tok in ["Date", "Name", "Number", "Invoice", "Total"]:
    print(f"  {tok:8s} first layer per receipt doc: base={first_layers(base, tok)} "
          f"twin={first_layers(twin, tok)}")
