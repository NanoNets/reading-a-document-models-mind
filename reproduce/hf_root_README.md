---
license: apache-2.0
tags:
  - interpretability
  - jacobian-lens
  - vision-language
  - document-ai
  - ocr
base_model:
  - Qwen/Qwen2.5-VL-3B-Instruct
  - nanonets/Nanonets-OCR2-3B
---

# Jacobian lenses for document vision-language models

The two fitted lenses behind the Nanonets research note **"Reading a Document
Model's Mind"** (Mohammad Ali Shehral, Nanonets Applied AI Research) — as far
as we are aware, the first fully validated, full-protocol Jacobian-lens fit on
real documents over a vision-language model, the first for a document model,
and the first matched base + fine-tune pair.

A Jacobian lens (Gurnee, Sofroniew, Lindsey et al., Anthropic 2026 —
[blog + video](https://www.anthropic.com/research/global-workspace),
[paper](https://transformer-circuits.pub/2026/workspace/)) translates a
model's internal states into ordinary vocabulary, mid-thought — a readout of
what the model is thinking about before it has typed anything. We fitted and
validated the instrument over an open document-model pair: the base model and
its Nanonets document fine-tune. The research note shows what the pair
reveals: per-genre inner vocabularies, the two-phase reader, a polyglot inner
voice, the nameless majority (73–88% of the strongest inner signals carry no
single-token name), and a layer-by-layer X-ray of what fine-tuning changed.

## Files

| path | model (pinned revision) | fit corpus | size | md5 |
|---|---|---|---|---|
| `qwen2.5-vl-3b-instruct/lens.pt` | `Qwen/Qwen2.5-VL-3B-Instruct` @ `6628554…` | 1,000 prompts (900 DocILE document pages + 100 WikiText text passages), 8 interleaved shards merged | 294 MB | `30f68189923fde4a747ef37177cd138f` |
| `nanonets-ocr2-3b/lens.pt` | `nanonets/Nanonets-OCR2-3B` @ `c3886ff…` | 500 prompts (interleaved half of the same manifest), 4 shards merged | 294 MB | `c5e79548e45488fac927065811c37a0d` |

Each lens stores fp16 Jacobians for all 35 inter-layer transports
(`J_l`, 2048×2048) and loads as fp32. Each folder's `README.md` is that
lens's full model card — validation results, intended use, and limitations.
Use the pinned model revisions from the cards: in particular, the OCR2
checkpoint requires the tied-head handling described there.

## Validation (summary)

- Our implementation reproduces Anthropic's published result end-to-end on a
  small text model — clearing the standard logit-lens baseline on **all six**
  of their evaluation sets, with the same margin pattern — before any vision
  fit.
- Two document-lens fits on disjoint halves of the corpus agree at
  **CKA 0.998–0.9999** per layer.
- Every reported measurement carries in-run calibration against noise floors
  and ceilings.

## Use the lenses

Fitting code, the evaluation harness, every evidence table, and the exact
end-to-end reproduction sequence live in the companion GitHub repo:
**[NanoNets/reading-a-document-models-mind](https://github.com/NanoNets/reading-a-document-models-mind)**
(see `reproduce/`). Reading a document through a fitted lens is a single
forward pass on one GPU; re-fitting from scratch takes about 55 GPU-hours.

## Read the findings

- **Research note (the full study):**
  [Reading a Document Model's Mind](https://nanonets.com/research/reading-a-document-models-mind)
  — with the interactive layer-by-layer scanner. A non-technical blog version
  is linked from the note.
- **Method:** the Jacobian lens is Anthropic's instrument — our thanks for
  releasing it with its evaluation suite.

## License

Apache-2.0 (`LICENSE`, `NOTICE`). DocILE documents are not redistributed;
rebuild the fit corpus via the GitHub repo's instructions.

## Citation

If you quote or build on these lenses or the findings, please cite the note:

```bibtex
@misc{shehral2026lens,
  title   = {Reading a Document Model's Mind},
  author  = {Shehral, Mohammad Ali},
  year    = {2026},
  month   = {August},
  url     = {https://nanonets.com/research/reading-a-document-models-mind},
  note    = {Nanonets Applied AI Research. Lenses, code, and evidence tables released under Apache-2.0}
}
```
