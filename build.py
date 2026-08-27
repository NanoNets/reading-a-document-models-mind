#!/usr/bin/env python3
"""Builds index.html — the self-contained Nanonets research-note preview.
Inlines: nn-blog.css (the measured nanonets.com/blog design system), the Nanonets
logo SVG (as data URI), the OCR2 scrubber data, and the figure PNGs.
"""
import base64, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
LW = "/Users/shehral/latent-workspace"
SCRATCH = "/private/tmp/claude-501/-Users-shehral/55bd0236-4999-4610-9b45-5e3e53f30b56/scratchpad"

tpl = open(os.path.join(HERE, "template.html")).read()
tpl = tpl.replace("__NN_BLOG_CSS__", open(os.path.join(HERE, "assets/nn-blog.css")).read())
logo = base64.b64encode(open(os.path.join(HERE, "assets/nanonets.svg"), "rb").read()).decode()
tpl = tpl.replace("__NN_LOGO__", "data:image/svg+xml;base64," + logo)
data = json.load(open(os.path.join(SCRATCH, "scrubber_ocr2.json")))
tpl = tpl.replace("__SCRUBBER_DATA__", json.dumps(data).replace("</", "<\\/"))

def png64(name):
    p = os.path.join(LW, "artifacts/evidence/figures", name + ".png")
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()

for key, fig in [("__FIG_CHIASMUS__", "fig5_chiasmus"),
                 ("__FIG_MARKUP__", "fig4_twin_timeline"),
                 ("__FIG_GRID__", "note_figA_imend_grid"),
                 ("__FIG_DISSOC__", "fig2_b_vs_loading")]:
    tpl = tpl.replace(key, png64(fig))

out = os.path.join(HERE, "index.html")
open(out, "w").write(tpl)
print("built", out, f"{os.path.getsize(out)//1024} KB")
