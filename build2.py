#!/usr/bin/env python3
"""Builds the two publication pages: index.html (research note) + blog.html.
Self-contained: inlines nn-blog.css, the Nanonets logo, scrubber data (real docs),
tapestry/diff components (generated here from verified tables), and figure PNGs.

Usage:
  python3 build2.py            # draft build (noindex + "Internal draft" pill)
  python3 build2.py --publish  # public build (no draft pill; noindex stays by owner directive)
"""
import base64, json, os, html, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LW = "/Users/shehral/latent-workspace"
DATA = os.path.join(HERE, "data")
PUBLISH = "--publish" in sys.argv

css = open(os.path.join(HERE, "assets/nn-blog.css")).read()
logo = "data:image/svg+xml;base64," + base64.b64encode(open(os.path.join(HERE, "assets/nanonets.svg"), "rb").read()).decode()
scrub = json.load(open(os.path.join(DATA, "scrubber_real.json")))

def png64(name):
    p = os.path.join(LW, "artifacts/evidence/figures", name + ".png")
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()

# ---- tapestry component (from cool_genre_vocab, twin, curated verified picks) ----
TAP = [
    ("Receipts", [("print", 15), ("TOTAL", 29), ("=== ===", 30)], "billing schema, then separator art for the totals block"),
    ("Equations", [("{\\", 24), ("$(", 24), ("⟨", 25), ("\\[", 26)], "LaTeX scaffolding — in 4/4 equation pages, 0/22 others"),
    ("Forms", [("Office", 20), ("Print", 22), ("PO", 17)], "form-field vocabulary"),
    ("Handwriting", [("age", 9), ("of", 16), ("tion", 23)], "sounding out the script in fragments — and summoning zero typesetting"),
]
tap_html = '<div class="tapestry">'
for genre, toks, note in TAP:
    chips = "".join(f'<span class="tchip">{html.escape(t)}<i>L{L}</i></span>' for t, L in toks)
    tap_html += f'<div class="trow"><span class="tgenre">{genre}</span><span class="tchips">{chips}</span><span class="tnote">{html.escape(note)}</span></div>'
tap_html += "</div>"

chat_doc = "data:image/png;base64," + base64.b64encode(
    open(os.path.join(LW, "data/showcase_docs/showcase_invoice.png"), "rb").read()).decode()

subs = {
    "__NN_BLOG_CSS__": css, "__NN_LOGO__": logo, "__CHAT_DOC__": chat_doc,
    "__SCRUBBER_DATA__": json.dumps(scrub).replace("</", "<\\/"),
    "__TAPESTRY__": tap_html,
    "__FIG_TWO_PHASE__": png64("pub_two_phase"),
    "__FIG_RAMP_ATT__": png64("pub_ramp_attention"),
    "__FIG_PROBES_BRAND__": png64("pub_probe_ladder_brand"),
    "__FIG_CHIASMUS_BRAND__": png64("pub_chiasmus_brand"),
    "__FIG_DISSOC__": png64("fig2_b_vs_loading"),
    "__DIFFDATA__": json.dumps(json.load(open(os.path.join(DATA, "diffdata_doc05.json")))).replace("</", "<\\/"),
}

for tpl, out in [("note_v2_template.html", "index.html"), ("blog_template.html", "blog.html")]:
    s = open(os.path.join(HERE, tpl)).read()
    for k, v in subs.items():
        s = s.replace(k, v)
    if PUBLISH:
        import re
        # owner directive: pages stay noindex/nofollow even when published
        s = re.sub(r'(<span class="sep">·</span>\s*)?<span class="draft-pill">[^<]*</span>', "", s)
    open(os.path.join(HERE, out), "w").write(s)
    print("built", out, f"{os.path.getsize(os.path.join(HERE, out))//1024} KB", "(PUBLIC)" if PUBLISH else "(draft)")
