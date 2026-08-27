"""Figure 3: the intervention battery — dose-response of band-swap steering.
Reads ONLY artifacts/evidence/tables/{E7_dose_ladder.json, E6_text_control_v2.json,
E4_discriminator.json, E8_nameless_a8.json, E8_nameless_a16.json (last two optional
until they land)}. Regenerate:
  .venv/bin/python artifacts/evidence/figures/fig3_intervention_battery.py
Flip = the graded report margin changes sign relative to the clean run.
alpha = displacement toward the opposite class in class-separation units;
alpha=1 is the source paper's clamp-to-class-mean.
Palette: dataviz reference slots (blue #2a78d6, orange #eb6834, green #1c8259 for the
nameless dial), gray #8a8988 random controls (dashed), validated.
Status: FINAL for text/named-dial panels; nameless-dial series appears when E8 lands.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
T = os.path.join(ROOT, "artifacts/evidence/tables")

def flip_rate(rows, key, clean_key="clean"):
    return 100 * sum(1 for r in rows if (r[key] > 0) != (r[clean_key] > 0)) / len(rows)

# --- text model (E6 v2) ---
e6 = json.load(open(os.path.join(T, "E6_text_control_v2.json")))
tx_a = [a for a in e6["scales"]]
tx_p = [flip_rate(e6["per_doc"], f"probe_a{a:g}") for a in tx_a]
tx_r = [flip_rate(e6["per_doc"], f"rand_a{a:g}") for a in tx_a]

# --- VLM named dial: alpha=1 from the discriminator, ladder from E7 ---
disc = json.load(open(os.path.join(T, "E4_discriminator.json")))
d_flip = 100 * sum(1 for r in disc["per_doc"]
                   if (r["report_type_swap"] > 0) != (r["report_type_clean"] > 0)) / len(disc["per_doc"])
d_rand = 100 * sum(1 for r in disc["per_doc"]
                   if (r["report_type_rand"] > 0) != (r["report_type_clean"] > 0)) / len(disc["per_doc"])
e7 = json.load(open(os.path.join(T, "E7_dose_ladder.json")))
vn_a = [1.0] + [a for a in e7["scales"]]
vn_p = [d_flip] + [flip_rate(e7["per_doc"], f"probe_a{a:g}") for a in e7["scales"]]
vn_r = [d_rand] + [flip_rate(e7["per_doc"], f"rand_a{a:g}") for a in e7["scales"]]

# --- VLM nameless dial (E8, optional until it lands); alpha=1 from original E4 ---
e4 = json.load(open(os.path.join(T, "E4_selectivity.json")))
g_flip = 100 * sum(1 for r in e4["per_doc"]
                   if (r["report_geom_swap"] > 0) != (r["report_geom_clean"] > 0)) / len(e4["per_doc"])
g_rand = 100 * sum(1 for r in e4["per_doc"]
                   if (r["report_geom_rand"] > 0) != (r["report_geom_clean"] > 0)) / len(e4["per_doc"])
nl_a, nl_p, nl_r = [1.0], [g_flip], [g_rand]
for a, fn in [(8, "E8_nameless_a8.json"), (16, "E8_nameless_a16.json")]:
    p = os.path.join(T, fn)
    if os.path.exists(p):
        e8 = json.load(open(p))
        nl_a.append(a)
        nl_p.append(flip_rate(e8["per_doc"], "report_geom_swap", "report_geom_clean"))
        nl_r.append(flip_rate(e8["per_doc"], "report_geom_rand", "report_geom_clean"))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 3.7), dpi=200, sharey=True)
# left: text model
axL.plot(tx_a, tx_p, "o-", color="#2a78d6", lw=2, ms=5, label="language direction")
axL.plot(tx_a, tx_r, "o--", color="#8a8988", lw=1.4, ms=4, label="random direction")
axL.set_title("A — text model (0.5B): report English vs French", fontsize=9.5,
              loc="left", color="#0b0b0b")
# right: VLM
axR.plot(vn_a, vn_p, "o-", color="#eb6834", lw=2, ms=5, label="named dial (document type)")
axR.plot(vn_a, vn_r, "o--", color="#8a8988", lw=1.4, ms=4, label="named dial, random ctrl")
if len(nl_a) > 1:
    axR.plot(nl_a, nl_p, "s-", color="#1c8259", lw=2, ms=5, label="nameless dial (report instrument invalid)")
    axR.plot(nl_a, nl_r, "s--", color="#b3b2ae", lw=1.4, ms=4, label="nameless dial random (also invalid)")
else:
    axR.scatter(nl_a, nl_p, marker="s", color="#1c8259", s=28,
                label="nameless dial (α=1; ladder pending)")
axR.set_title("B — document VLM (3B): report the dial's value", fontsize=9.5,
              loc="left", color="#0b0b0b")

for ax in (axL, axR):
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32]); ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    ax.axvline(1.0, color="#c9c8c2", lw=0.9)
    ax.set_xlabel("dose α (class-separation units; α=1 = source paper's clamp)", fontsize=8.5)
    ax.set_ylim(0, 104)
    ax.spines[["top", "right"]].set_visible(False)
    for s in ax.spines.values(): s.set_color("#c9c8c2")
    ax.grid(axis="y", color="#eceae4", lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(colors="#52514e", labelsize=8)
axL.legend(frameon=False, fontsize=7.8, loc="upper right")
axR.legend(frameon=False, fontsize=7.8, loc="center left")
axL.set_ylabel("verbal report flipped (%)", fontsize=9)
axL.annotate("transported\ndose", xy=(1.0, 96), fontsize=7.5, color="#52514e", ha="center")

fig.suptitle("Band-swap steering is dose-limited, not inert: the transported dose (α=1) moves nothing,\n"
             "α=2 flips the text model completely; the VLM's named dial swaps selectively at α=16 only",
             fontsize=10.5, x=0.01, ha="left", y=1.09, color="#0b0b0b")
fig.tight_layout()
out = os.path.join(ROOT, "artifacts/evidence/figures/fig3_intervention_battery")
fig.savefig(out + ".png", bbox_inches="tight"); fig.savefig(out + ".svg", bbox_inches="tight")
print("saved", out + ".{png,svg}")
print("text probe:", [round(v) for v in tx_p], "rand:", [round(v) for v in tx_r])
print("vlm named probe:", [round(v) for v in vn_p], "rand:", [round(v) for v in vn_r])
print("vlm nameless probe:", [round(v) for v in nl_p], "rand:", [round(v) for v in nl_r])
