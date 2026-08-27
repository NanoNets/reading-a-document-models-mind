# Reproduce everything, in order

Plain `python` invocations. Every step also ran as a SLURM job on our cluster;
a generic SLURM template is at the bottom — nothing about the pipeline requires
SLURM. Steps 0-2 need no GPU. Steps 3-6 need CUDA (any modern GPU; see the
budget notes). All commands run from the repo root with the venv active
(README "Install").

Environment variables used throughout:

| var | meaning | default |
|---|---|---|
| `HF_HOME` | HuggingFace cache dir (models download here once) | `~/.cache/huggingface` |
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` | deterministic cuBLAS — **set for every GPU step** | unset |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | avoids fragmentation OOM on long fits | unset |

## 0. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e code/vendor-jacobian-lens --no-deps   # never let pip touch the pinned torch
python -c "import jlens; print('jlens ok')"
```

## 1. Fit corpus (no GPU)

```bash
# 1a. Download DocILE (free registration): https://docile.rossum.ai/
#     You need the annotated-trainval PDFs. Then render our 900 first pages:
DOCILE_PDF_DIR=/path/to/docile/pdfs python code/render_pages.py
# -> data/e1_pages/<doc_id>.png  (pdftoppm 26.02.0, -png -r 120 -f 1 -l 1 -singlefile)

# 1b. Build the 1000-prompt manifest (900 doc prompts, 4 rotating OCR
#     instructions, + 100 WikiText text prompts):
python code/e1_build_manifest.py
# -> data/e1_prompts_manifest.json  (paths point at data/e1_pages; override with E1_PAGES_DIR)
```

## 2. E0 — instrument validation gate (GPU optional; MPS/CPU work, slower)

```bash
python code/e0_fit.py
# env: E0_MODEL (default Qwen/Qwen2.5-0.5B-Instruct), E0_PROMPTS_FILE, E0_N_PROMPTS (1000),
#      E0_DIM_BATCH (32 cuda / 8 otherwise), E0_OUT_TAG (e0), E_REV (model revision)
# -> data/lenses/qwen2.5-0.5b-instruct.e0.pt + artifacts/evidence/tables/e0_readout.json

E0_LENS=data/lenses/qwen2.5-0.5b-instruct.e0.pt python code/e0_eval.py
# -> artifacts/evidence/tables/E0_lens_evals.json ; PASS = J-lens AUC >= logit-lens AUC on >=4/6 sets
```

**Do not proceed if E0 fails.** Shipped run: 6/6.

## 3. E1 — fit the base lens (GPU, the expensive step)

One process per shard; shards are independent — run them in parallel on
separate GPUs, or sequentially on one. fp32; ~7 h per shard on an H200-141GB
at `E1_DIM_BATCH=16` (~55 H200-hours for the 8-shard base fit; ~130 GPU-hours
for the full study including the twin fit and all evaluation tables).
Auto-scaled dim_batch on smaller GPUs is proportionally slower.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for k in 0 1 2 3 4 5 6 7; do
  E1_SHARD=$k E1_NSHARDS=8 E1_DIM_BATCH=16 \
  E1_MODEL=Qwen/Qwen2.5-VL-3B-Instruct \
  E_REV=66285546d2b821cf421d4f5eb2576359d3770cd3 \
  python code/e1_fit.py
done
# other env: E1_MAX_SEQ (512)
# -> data/lenses/qwen2.5-vl-3b-instruct.e1.shard<k>.pt (+ .meta.json with env stamps)
```

Checkpoint/resume: each shard writes `<out>.pt.ckpt` atomically after every
prompt and resumes from it on restart; a finished shard exits immediately
(idempotent). Kill/requeue at will.

```bash
# merge 8 shards -> the canonical lens (minutes)
MERGE_SLUG=qwen2.5-vl-3b-instruct MERGE_NSHARDS=8 \
MERGE_MODEL=Qwen/Qwen2.5-VL-3B-Instruct python code/e1_merge.py
# -> data/lenses/qwen2.5-vl-3b-instruct.e1.pt
```

## 4. E1-twin — fit the OCR fine-tune lens

Same driver, different model. The shipped twin lens used the interleaved
half-corpus (shards 0-3 of 8, 500 prompts) — fitting was corpus-saturated well
before that (see `E1_saturation_3shard.json`); fit all 8 for a superset check
if you have the hours. Runs on A100-80GB (dim_batch auto-drops to 12).

```bash
for k in 0 1 2 3; do
  E1_SHARD=$k E1_NSHARDS=8 E1_MODEL=nanonets/Nanonets-OCR2-3B \
  E_REV=c3886ff00bb037ce7da24988c9eafaf1fe2bed72 \
  python code/e1_fit.py
done
MERGE_SLUG=nanonets-ocr2-3b MERGE_NSHARDS=4 \
MERGE_MODEL=nanonets/Nanonets-OCR2-3B python code/e1_merge.py
# -> data/lenses/nanonets-ocr2-3b.e1.pt
```

Watch the adapter's startup line: it must say `unembed=embed_tokens.T
(config tied; ...)` for this model — that is the tied-head workaround for the
checkpoint's config bug (README "Model pins").

## 5. Diagnostics — compare your re-fit against the shipped lenses

```bash
# held-out doc diagnostics (any ~26 document PNGs; forward-only, any GPU)
E_LENS=data/lenses/qwen2.5-vl-3b-instruct.e1.pt E_DOCSET=/path/to/diag_pngs \
E_TAG=e1merged E_REV=66285546d2b821cf421d4f5eb2576359d3770cd3 python code/e1_diag.py
# -> artifacts/evidence/tables/e1merged_diag.json
# (our diag set: 26 pages spanning DocVQA scans, IAM handwriting, receipts —
#  the exact filenames are keyed in the shipped e1merged_diag.json)
```

Expected: per-layer readout kurtosis and top-1 agreement curves within noise of
the shipped tables; disjoint-corpus shard lenses agreed at CKA 0.998-0.999
(`E1_shard_stability.json`).

## 6. E2/E3 — the note's measurement tables (GPU)

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python code/e2_bscore.py     # broadcast vs nameability populations -> E2_B_kappa.json (seeded, 20260826)
python code/e2t_bscore.py    # same battery on the twin lens        -> E2t_B_kappa.json
python code/e3_probes.py     # paired probe battery on DocILE       -> E3_probes.json + data/probe_dirs.pt
python code/compute_h1_stats.py artifacts/evidence/tables/E2_B_kappa.json \
                                artifacts/evidence/tables/E2_H1_stats_v4.json
```

`e2_bscore.py`/`e2t_bscore.py` need the diag doc set (step 5) for the
covariance-matched null; `e3_probes.py` needs the rendered `data/e1_pages` and
the shipped `data/e3_labels.json`. All three print in-job calibration gates
(P5 ceiling vs P4 null) — if those look wrong, stop and compare env stamps.

## 7. Figures (no GPU)

```bash
for f in artifacts/evidence/figures/fig*.py; do python "$f"; done
```

## Optional: generic SLURM template

```bash
#!/bin/bash
#SBATCH --gres=gpu:1 --cpus-per-task=8 --mem=64GB --time=07:59:00
#SBATCH --array=0-7            # for the shard fits; drop for single jobs
set -e
export HF_HOME=$SCRATCH/hf_cache
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SLURM_SUBMIT_DIR
PY=.venv/bin/python            # call the env python directly; `conda activate`
                               # can silently resolve to base python in batch jobs
$PY -c "import jlens" 2>/dev/null || $PY -m pip install -q -e code/vendor-jacobian-lens --no-deps
E1_SHARD=$SLURM_ARRAY_TASK_ID E1_NSHARDS=8 E1_DIM_BATCH=16 $PY code/e1_fit.py
```

Two portability notes from our runs: (1) compute nodes often need the site
HTTP(S) proxy exported explicitly before the first model download — or pre-warm
`HF_HOME` from a login node; (2) prefer calling the venv's python binary over
environment activation inside batch scripts.
