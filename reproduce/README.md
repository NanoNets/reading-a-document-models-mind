# Reading a Document Model's Mind — code, lenses, evidence

Code, fitted lenses, and evidence tables behind the Nanonets research note
**"Reading a Document Model's Mind"** (Mohammad Ali Shehral, Nanonets Applied
AI Research).

The Jacobian lens (Gurnee, Sofroniew, Lindsey et al., Anthropic 2026 —
[blog + video](https://www.anthropic.com/research/global-workspace), [paper](https://transformer-circuits.pub/2026/workspace/))
translates a model's internal states into ordinary vocabulary, mid-thought. We
fitted and fully validated it over two document vision-language models — an open
base model and its Nanonets document fine-tune — the first such full-protocol
pair on real documents. This repo lets a stranger with a GPU **re-fit both
lenses, re-run the instrument validation, and regenerate every figure on the
note** from the evidence tables. Where the lens's single-token vocabulary can't
name what the model knows (most of it — see the note's 73–88% finding), that is
measured and reported, not hidden.

### Get the fitted lenses (294 MB each)

The lenses live on HuggingFace, not in git:

```bash
pip install -U huggingface_hub
hf download nanonets/document-vlm-jacobian-lenses --local-dir hf_lenses
mkdir -p data/lenses
cp hf_lenses/qwen2.5-vl-3b-instruct/lens.pt data/lenses/qwen2.5-vl-3b-instruct.e1.pt
cp hf_lenses/nanonets-ocr2-3b/lens.pt       data/lenses/nanonets-ocr2-3b.e1.pt
```

md5s to verify after download: `30f68189923fde4a747ef37177cd138f` (base),
`c5e79548e45488fac927065811c37a0d` (twin).

## Contents

```
data/lenses/qwen2.5-vl-3b-instruct.e1.pt   fitted lens, Qwen2.5-VL-3B-Instruct (base)
data/lenses/nanonets-ocr2-3b.e1.pt         fitted lens, Nanonets-OCR2-3B (OCR fine-tune twin)
code/                                      fitting + evaluation drivers, VLM adapter
code/vendor-jacobian-lens/                 vendored anthropics/jacobian-lens (Apache-2.0, pinned SHA)
data/                                      fit-corpus manifest inputs, E3 labels, probe directions
artifacts/evidence/tables/                 raw result JSONs (every number on the note lives here)
artifacts/evidence/figures/                one script per figure, reads only the tables
lens_cards/                                model cards for the two lenses
METHODS_VERIFIED.md                        the verified numbers table: every figure/claim, with its evidence path
reproduce.md                               exact end-to-end command sequence
```

## Install

Requires Python >= 3.12 (built and tested with 3.13.13; the GPU fits ran on 3.12).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                       # torch 2.13.0, transformers 5.15.1, ...
pip install -e code/vendor-jacobian-lens --no-deps    # the jlens package (pure python)
```

Key pins from `requirements.txt`: `torch==2.13.0`, `transformers==5.15.1`,
`pillow==12.3.0`, `numpy==2.5.2`, `scipy==1.18.1`, `matplotlib==3.11.1`.
The cluster environment that produced the fits used `torch 2.8.0+cu128` on
Python 3.12; every output JSON in `artifacts/evidence/tables/` stamps the
torch/transformers versions and model revision that produced it, so drift is
detectable.

### Model pins (do not unpin)

| artifact | HF id | pinned revision |
|---|---|---|
| base VLM | `Qwen/Qwen2.5-VL-3B-Instruct` | `66285546d2b821cf421d4f5eb2576359d3770cd3` |
| twin (OCR fine-tune) | `nanonets/Nanonets-OCR2-3B` | `c3886ff00bb037ce7da24988c9eafaf1fe2bed72` |
| E0 validation vehicle | `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` |
| lens implementation | `anthropics/jacobian-lens` | `581d398613e5602a5af361e1c34d3a92ea82ba8e` (vendored) |
| page renderer | poppler `pdftoppm` | 26.02.0, `-png -r 120 -f 1 -l 1 -singlefile` |

Why pinning matters here specifically: the Nanonets-OCR2-3B checkpoint ships no
`lm_head.weight` and its config mis-sets `tie_word_embeddings=False`, so an
unpinned load lets HuggingFace random-initialize the head. Our adapter
(`code/jlens_vlm.py`) detects this and re-projects through `embed_tokens` (the
intended tied head). If the upstream config is ever fixed, unpinned
reproductions would silently load a *different* head and get different numbers.
The pins freeze the exact behavior the note measured.

### The lenses

| file | model | fit corpus | size | md5 |
|---|---|---|---|---|
| `data/lenses/qwen2.5-vl-3b-instruct.e1.pt` | base, pinned rev above | 1000 prompts (900 DocILE pages + 100 WikiText), 8 shards merged | 294 MB | `30f68189923fde4a747ef37177cd138f` |
| `data/lenses/nanonets-ocr2-3b.e1.pt` | twin, pinned rev above | 500 prompts (interleaved half of the same manifest: ~450 DocILE + ~50 WikiText), 4 shards merged | 294 MB | `c5e79548e45488fac927065811c37a0d` |

Both store fp16 Jacobians for all 35 inter-layer transports (`J_l`, 2048x2048)
and load as fp32. These md5s are also recorded inside
`artifacts/evidence/tables/E1_twin_vs_base.json` (`base_md5`/`twin_md5`), tying
every downstream table to exactly these files. Full details, validation
results, intended use, and limitations: `lens_cards/`.

## (a) Quickstart — load a lens and read a document

**GPU path (any CUDA GPU, forward-only, ~10 GB):** run from the repo root with
`code/` on `sys.path`:

```python
import sys, torch, transformers, jlens
sys.path.insert(0, "code")
from jlens_vlm import QwenVLLensModel

MODEL, REV = "Qwen/Qwen2.5-VL-3B-Instruct", "66285546d2b821cf421d4f5eb2576359d3770cd3"
hf = transformers.AutoModelForImageTextToText.from_pretrained(
    MODEL, revision=REV, dtype=torch.float32).cuda()
proc = transformers.AutoProcessor.from_pretrained(
    MODEL, min_pixels=256*28*28, max_pixels=256*28*28)
model = QwenVLLensModel(hf, proc)
lens = jlens.JacobianLens.load("data/lenses/qwen2.5-vl-3b-instruct.e1.pt")

prompt = "IMG::/abs/path/to/page.png::Transcribe this document as plain text."
lens_logits, _, _ = lens.apply(model, prompt, positions=[-1])   # last prompt token
for L, lg in sorted(lens_logits.items()):
    print(L, [proc.tokenizer.decode([t]) for t in lg[0].topk(10).indices])
```

This prints, per layer, the ten tokens the lens says the state at the last
prompt position is disposed to produce — the raw material of the note's
"schema preview" readouts. `code/e1_diag.py` is the batch version.

**CPU path (no GPU, tested):** `python code/quickstart_cpu.py` loads the base
lens plus just two tensors from the checkpoint (tied unembedding + final norm —
the 3B model is never instantiated) and reads out (1) a genuine layer-0
residual state (a token embedding) and (2) the precomputed E3 probe directions
shipped in `data/probe_dirs.pt`. Expected output: the token state decodes to
its own name at every depth; the band probe directions decode to unrelated
tokens — the low-nameability finding in miniature. Producing new states for a
real document requires the GPU path above.

## (b) Re-fit both lenses from scratch

Full command sequence with env vars and an optional SLURM template:
**`reproduce.md`**. Summary:

1. **Corpus.** Register for DocILE (free, https://docile.rossum.ai/), download
   the annotated-trainval PDFs. We ship the 900 document IDs
   (`data/e1_docs_900.json`), the renderer (`code/render_pages.py`, pinned
   pdftoppm flags), and the 100 WikiText text prompts. Render, then build the
   manifest with `code/e1_build_manifest.py` (4 OCR instructions rotate
   deterministically across pages).
2. **Shard fits.** `code/e1_fit.py` fits one lens per shard
   (`E1_SHARD=k E1_NSHARDS=8`), fp32, deterministic
   (`CUBLAS_WORKSPACE_CONFIG=:4096:8`, no sampling anywhere in fit or readout).
   Each shard writes a resumable checkpoint every prompt (atomic) and exits
   idempotently if its final `.pt` exists — kill and resubmit freely.
3. **Merge.** `code/e1_merge.py` averages the shard lenses into the canonical
   lens (`MERGE_SLUG`, `MERGE_NSHARDS`).
4. **Budget (fp32).** The base-lens fit cost **~55 H200-hours** (8 shards x
   ~7 h); the twin fit (4 shards) ran on A100-80GB at roughly 2x the per-shard
   time. Budget **~130 GPU-hours (H200-class)** to reproduce the full study —
   both fits plus every evaluation table (E1 diagnostics, E2/E2t, E3, and the
   intervention battery). `E1_DIM_BATCH` auto-scales to GPU memory (32 at
   >=100 GB, 12 at >=70 GB, 4 at >=38 GB); smaller GPUs work, proportionally
   slower.
5. **Check.** `code/e1_diag.py` on a held-out doc set; two disjoint 125-doc
   shards agreed at CKA 0.998-0.999 per layer
   (`artifacts/evidence/tables/E1_shard_stability.json`), so your re-fit should
   land within noise of the shipped lenses.

## (c) Re-run E0 (instrument validation)

E0 reproduces the vendor's published lens behavior on a small text model before
trusting anything downstream — it was a hard gate for this project.

```bash
python code/e0_fit.py            # fits the 0.5B lens; CUDA, MPS, or (slow) CPU
E0_LENS=data/lenses/qwen2.5-0.5b-instruct.e0.pt python code/e0_eval.py
```

`e0_eval.py` runs the six lens-eval sets bundled with the vendored repo
(J-lens vs logit lens, pass@k and AUC over log k). Shipped result: **J-lens
wins 6/6 sets** (`artifacts/evidence/tables/E0_lens_evals.json`); the
registered pass criterion was >=4/6.

## (d) Regenerate every figure

Each figure is a script that reads **only** `artifacts/evidence/tables/*.json`
(all shipped), so this needs no GPU and no model download:

```bash
python artifacts/evidence/figures/fig1_band_effdim.py           # lens effective dimensionality by depth
python artifacts/evidence/figures/fig2_b_vs_loading.py          # broadcast vs nameability (H1 dissociation)
python artifacts/evidence/figures/fig3_intervention_battery.py  # dose-response of band-swap steering
python artifacts/evidence/figures/fig4_twin_timeline.py         # what OCR fine-tuning did to the lens
python artifacts/evidence/figures/fig5_chiasmus.py              # maps change early, content changes late
```

Outputs land next to the scripts as `.png` + `.svg`. To regenerate the H1
statistics table itself from the raw E2 arrays:

```bash
python code/compute_h1_stats.py artifacts/evidence/tables/E2_B_kappa.json out.json
```

Downstream evaluation drivers (`code/e2_bscore.py`, `code/e2t_bscore.py`,
`code/e3_probes.py`) re-produce those tables from the lenses + corpus on a GPU;
see `reproduce.md`.

## Scope

This release covers the lens-fitting and lens-evaluation pipeline and the
evidence behind the note's figures. It does not include raw activation dumps
(too large; regenerate via the scripts) or DocILE-derived images (license;
regenerate via `code/render_pages.py`).

## License

Apache-2.0 (see `LICENSE`, `NOTICE`). The vendored
`code/vendor-jacobian-lens/` is Apache-2.0 by Anthropic, pinned at
`581d3986`; our adapter and drivers are ours. WikiText excerpts are CC BY-SA;
DocILE documents are not redistributed.

## Citation

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
