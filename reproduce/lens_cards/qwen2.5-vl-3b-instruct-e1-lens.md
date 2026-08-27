---
license: apache-2.0
base_model: Qwen/Qwen2.5-VL-3B-Instruct
tags:
  - interpretability
  - jacobian-lens
  - vision-language
  - document-ai
---

# Jacobian lens for Qwen2.5-VL-3B-Instruct (document corpus, E1)

A fitted **Jacobian lens** (Anthropic, `transformer-circuits.pub/2026/workspace`;
reference code `anthropics/jacobian-lens @ 581d3986`, vendored in this release)
over the **text decoder** of `Qwen/Qwen2.5-VL-3B-Instruct`, fit on a
document-understanding corpus. The lens is the corpus-averaged input-output
Jacobian `J_l = E[dh_final / dh_l]` for every decoder layer `l`; applying
`unembed(J_l @ h)` reads out, as a ranked token list, what the residual-stream
state `h` at layer `l` is disposed to make the model say.

| | |
|---|---|
| file | `data/lenses/qwen2.5-vl-3b-instruct.e1.pt` |
| md5 | `30f68189923fde4a747ef37177cd138f` |
| size | 294 MB (fp16 on disk, fp32 on load) |
| contents | 35 Jacobians `J_l` (2048 x 2048), source layers 0-34 |
| model revision (pinned) | `66285546d2b821cf421d4f5eb2576359d3770cd3` |
| unembedding | model's own tied head (`lm_head`, tied-check cos > 0.5 verified at load) |
| loader | `jlens.JacobianLens.load(path)` from the vendored package |

## Fitting corpus

1000 prompts, fit fp32, deterministic (`CUBLAS_WORKSPACE_CONFIG=:4096:8`, no
sampling), `max_seq_len=512`, 8 disjoint interleaved shards of 125 merged by
running-mean:

- **900 DocILE invoice/form pages** (first page, pdftoppm 26.02.0 at 120 dpi;
  document IDs shipped as `data/e1_docs_900.json` — DocILE itself requires
  free registration and is not redistributed), with **4 rotating OCR
  instructions** ("Transcribe this document as plain text." / "Convert this
  document to markdown, preserving tables." / "Extract all line items from
  this document as a table." / "Read this document and reproduce its full
  text content."), presented as image + instruction chat prompts;
- **100 WikiText text prompts** (`data/prompts_wikitext_1000.json[:100]`).

## Validation

- **E0 instrument gate:** the fitting/readout pipeline reproduces the vendor's
  published lens behavior on `Qwen/Qwen2.5-0.5B-Instruct`: J-lens beats the
  logit lens on **6/6** of the vendor's bundled lens-eval sets (registered
  criterion: >=4/6). Evidence: `artifacts/evidence/tables/E0_lens_evals.json`.
- **Shard stability:** two lenses fit on disjoint 125-document shards agree at
  **CKA 0.998-0.999 per layer** (raw cosine 0.994-0.999). Evidence:
  `artifacts/evidence/tables/E1_shard_stability.json`.
- **Corpus saturation:** the 1-shard vs 3-shard band curve drifted by at most
  0.001 (`E1_saturation_3shard.json`) — the corpus is large enough that the
  merged lens is converged.
- The md5 above is recorded in `E1_twin_vs_base.json` (`base_md5`), tying every
  shipped result table to exactly this file.

## Intended use

Interpretability research on document VLMs: reading out layer-by-layer token
dispositions over document inputs, measuring nameability/loading of residual
directions against the J-lens vector set, and serving as the base-model
reference point for comparisons against OCR fine-tunes (see the companion twin
lens card). Load it with the pinned model revision and the vendored `jlens`.

## Limitations

- **Corpus-conditional readouts.** `J_l` is an *average* over this document
  corpus. Readouts answer "what would this state make the model say, averaged
  over document-OCR contexts" — on far out-of-distribution inputs the
  transport is not guaranteed to be faithful. A lens fit on generic web text
  will differ.
- **Single-token vocabulary limitation.** The lens has one readout vector per
  vocabulary token *by construction*: every concept it can surface has a
  single-token name. Task-critical variables without a single-token name
  (table geometry, row counts, layout schemas) can carry causal weight while
  remaining invisible or garbled in the readout. This is measured directly in the
  accompanying research note ("Reading a Document Model's Mind", Shehral,
  Nanonets Applied AI Research, 2026): most of the strongest broadcast signals
  carry no single-token name. Do not interpret readout absence as
  representational absence.
- The lens sees only the text decoder's residual stream; the vision tower is
  outside its scope (gradients never flow into it during fitting).
- Fit at `max_seq_len=512`; behavior on much longer contexts is extrapolation.
