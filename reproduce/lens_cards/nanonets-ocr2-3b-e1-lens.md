---
license: apache-2.0
base_model: nanonets/Nanonets-OCR2-3B
tags:
  - interpretability
  - jacobian-lens
  - vision-language
  - document-ai
  - ocr
---

# Jacobian lens for Nanonets-OCR2-3B (document corpus, E1 twin)

A fitted **Jacobian lens** (Anthropic, `transformer-circuits.pub/2026/workspace`;
reference code `anthropics/jacobian-lens @ 581d3986`, vendored in this release)
over the **text decoder** of `nanonets/Nanonets-OCR2-3B` — an OCR fine-tune of
Qwen2.5-VL-3B — fit on the same document corpus as the base-model lens. The
pair is a controlled "twin" design: same architecture, same fitting corpus,
same instrument; differences between the two lenses isolate what OCR
fine-tuning did.

| | |
|---|---|
| file | `data/lenses/nanonets-ocr2-3b.e1.pt` |
| md5 | `c5e79548e45488fac927065811c37a0d` |
| size | 294 MB (fp16 on disk, fp32 on load) |
| contents | 35 Jacobians `J_l` (2048 x 2048), source layers 0-34 |
| model revision (pinned) | `c3886ff00bb037ce7da24988c9eafaf1fe2bed72` |
| unembedding | tied projection through `embed_tokens` (see checkpoint quirk below) |
| loader | `jlens.JacobianLens.load(path)` from the vendored package |

**Checkpoint quirk (why the revision pin is load-bearing):** this checkpoint
ships no `lm_head.weight` and its config mis-sets `tie_word_embeddings=False`,
so a naive HuggingFace load silently **random-initializes** the output head.
Our adapter (`code/jlens_vlm.py`) detects the untrained head and re-projects
through `embed_tokens` — the intended tied head. All readouts from this lens
must use that adapter (or an equivalent tied projection). If the upstream
config is ever fixed, unpinned loads would change behavior; the pin freezes
what was measured.

## Fitting corpus

500 prompts — the interleaved half (shards 0-3 of 8, merged by running-mean) of
the identical 1000-prompt manifest used for the base lens: ~450 **DocILE
pages** (first page, pdftoppm 26.02.0 at 120 dpi, IDs in
`data/e1_docs_900.json`) with the same **4 rotating OCR instructions**, and
~50 **WikiText text prompts**. Fit fp32, deterministic
(`CUBLAS_WORKSPACE_CONFIG=:4096:8`), `max_seq_len=512`. Half-corpus fitting is
justified by the measured corpus saturation of the base fit (1-vs-3-shard
drift <= 0.001, `E1_saturation_3shard.json`) and the disjoint-shard stability
below.

## Validation

- **E0 instrument gate (shared):** the fitting/readout pipeline reproduces the
  vendor's published lens behavior on a text model — J-lens beats logit lens
  on **6/6** bundled eval sets (`artifacts/evidence/tables/E0_lens_evals.json`).
- **Shard stability (base fit, same pipeline):** disjoint 125-document shards
  agree at **CKA 0.998-0.999 per layer** (`E1_shard_stability.json`).
- **Sanity vs base lens:** per-layer cosine to the base lens is
  **high-but-not-identical** (median 0.940, min 0.930), exactly the signature
  expected for a fine-tune twin rather than a re-fit artifact. Evidence:
  `artifacts/evidence/tables/E1_twin_vs_base.json`, which also records this
  file's md5 (`twin_md5`).

## Intended use

Interpretability research on OCR/document fine-tuning: differencing this lens
against the base-model lens (per-layer rewiring maps, readout-content shifts
over depth, markup-token occupancy), and reading out layer-by-layer token
dispositions of Nanonets-OCR2-3B states over document inputs. Always load the
model at the pinned revision and read out through the tied-head adapter.

## Limitations

- **Corpus-conditional readouts.** `J_l` averages over this document corpus;
  faithfulness on far out-of-distribution inputs is not guaranteed.
- **Single-token vocabulary limitation.** One readout vector per vocabulary
  token, by construction: variables without a single-token name can carry
  causal weight yet stay invisible to the lens. In the accompanying research note
  ("Reading a Document Model's Mind", Shehral, Nanonets Applied AI Research,
  2026) this is measured directly: most of the strongest broadcast signals
  carry no single-token name — the model knows more than the lens can name.
  Do not interpret readout absence as representational absence.
- Fit on half the manifest (500 prompts); the saturation evidence says this is
  converged, but it is not literally the same prompt set as the base lens.
- The lens sees only the text decoder; the vision tower is outside its scope.
- Fit at `max_seq_len=512`; longer contexts are extrapolation.
