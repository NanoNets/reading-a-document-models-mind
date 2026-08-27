# Reading a Document Model's Mind

Nanonets Applied AI Research — interpretability of document vision-language models
using the Jacobian-lens method (Gurnee, Sofroniew, Lindsey et al., Anthropic 2026).

**Internal draft for review — not published.**

## The two pieces

| Page | Audience | Open |
|---|---|---|
| [`index.html`](index.html) | Research note — rich, interactive, full methodology | open locally in any browser |
| [`blog.html`](blog.html) | Blog — non-technical readers | open locally in any browser |

Both pages are fully self-contained (one file each, no build step, no network calls
beyond Google Fonts). The interactive instrument embeds **real measurements** from
Nanonets-OCR2-3B reading real public documents — nothing simulated.

## What's inside

- The first fully validated, full-protocol Jacobian lens fitted on real documents
  over a vision-language model's decoder — and the first base/fine-tune lens pair
  (Qwen2.5-VL-3B-Instruct + Nanonets-OCR2-3B).
- Findings: genre-specific inner vocabularies, the two-phase reader
  (understand→typeset), the polyglot inner voice, the certainty ramp, the
  fine-tuning chiasmus ("new reader, same typewriter"), the 88% nameless majority,
  and more — every number traced to a released evidence table.

## Reproduce everything

[`reproduce/`](reproduce/) contains the fitting + evaluation code, the evidence
tables behind every number on both pages, figure scripts (each page figure
regenerates from tables alone), model cards, pinned requirements, and a step-by-step
[`reproduce.md`](reproduce/reproduce.md). The fitted lenses (~294 MB each) live on
HuggingFace (link in the release cards on the pages) rather than in git.

## Build the pages

```sh
python3 build2.py   # regenerates index.html + blog.html from templates + evidence
```

Apache-2.0 (see reproduce/LICENSE + NOTICE — the vendored lens-fitting engine is
Anthropic's, Apache-2.0, pinned).
