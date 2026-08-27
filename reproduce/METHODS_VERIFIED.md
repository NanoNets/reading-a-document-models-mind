# Methodology — verified content for the public research note

<!-- Assembled 2026-08-27 by direct extraction from the repo. Every fact carries its source
     path in an HTML comment. All numbers re-read from the raw JSON / code on this date;
     where the LEDGER records a correction (GQA b_attn fix, 55-vs-130 GPU-h, 45-vs-48
     unscoreable intermediates), the corrected value is used. -->

Audience assumption: the reader has **not** read Anthropic's workspace paper. Each method
below gets (a) a plain-English account of what it does and why, then (b) the exact
specifics as implemented here.

**One-paragraph primer.** The Jacobian lens (Anthropic 2026, `transformer-circuits.pub/2026/workspace`,
reference implementation `anthropics/jacobian-lens`) is an instrument for reading a
transformer's intermediate layers. A language model builds its answer gradually across ~35
layers of an internal "residual stream"; the lens asks, for a state at layer L, *"if this
internal state changed slightly, how would the model's final output change?"* — and uses
that sensitivity map (a Jacobian matrix, averaged over a corpus) to translate mid-network
states into ranked lists of vocabulary tokens. We fitted this instrument, for the first
time under the full reference protocol, to two document vision-language models: the base
model `Qwen/Qwen2.5-VL-3B-Instruct` and its OCR fine-tune `nanonets/Nanonets-OCR2-3B`
(a "twin" pair: same architecture, same fitting corpus, so lens differences isolate what
fine-tuning did). <!-- release/README.md; release/lens_cards/*.md; CLAUDE.md -->

---

## 1. Prompting

### 1.1 The fitting prompt set (what the lens averages over)

Plain English: the lens is a *corpus average* — it answers "how do internal states relate
to outputs, averaged over the kind of work this model does." So the fitting corpus has to
look like the model's real workload. Ours is 1,000 prompts: 900 document-OCR tasks and
100 plain-text prompts (the text slice keeps the lens honest on non-visual input).

Exact composition, verified by loading `data/e1_prompts_manifest.json` (1,000 entries):
<!-- data/e1_prompts_manifest.json (counted: 900 IMG entries, 100 text entries) -->

- **900 DocILE pages** (invoices/forms; first page of each document, rendered with
  poppler `pdftoppm` 26.02.0 at 120 dpi, `-png -r 120 -f 1 -l 1 -singlefile`), each paired
  with one of **four rotating OCR instructions**, assigned deterministically by document
  index (`i % 4`, 225 pages each — counts verified):
  <!-- code/e1_build_manifest.py; docs/reproducibility.md PINS; data/e1_prompts_manifest.json -->
  1. `Transcribe this document as plain text.`
  2. `Convert this document to markdown, preserving tables.`
  3. `Extract all line items from this document as a table.`
  4. `Read this document and reproduce its full text content.`
- **100 WikiText text prompts** (`data/prompts_wikitext_1000.json[:100]`), plain encyclopedic
  text, no image. <!-- code/e1_build_manifest.py; data/e1_prompts_manifest.json -->

Prompt convention: multimodal prompts are strings of the form
`IMG::<absolute_image_path>::<instruction>`; anything else is treated as text-only. The
adapter expands `IMG::` prompts through the model's own chat template
(`apply_chat_template(..., add_generation_prompt=True)`) with the image attached, at a
**fixed visual budget** — the processor is loaded with
`min_pixels = max_pixels = 256*28*28`, i.e. every page becomes the same number of image
tokens (~256). <!-- code/jlens_vlm.py; code/e1_fit.py -->

### 1.2 The transcription / diagnostic prompt

One prompt string is reused everywhere a "neutral document context" is needed:

> `Transcribe this document as plain text.`

It is (a) the held-out diagnostic prompt in `e1_diag.py`, (b) the capture prompt for the
E3 probe battery (the forward pass whose residuals the probes read), (c) the prompt under
which E2's covariance-matched null directions are estimated from live forwards, and
(d) the `continuation` condition of the E4 intervention battery.
<!-- code/e1_diag.py line "prompt = f\"IMG::{p}::Transcribe this document as plain text.\""; code/e3_probes.py; code/e2_bscore.py; data/e4_stimuli.json conditions.continuation -->

### 1.3 The E4 report / grading prompts (verbatim)

Plain English: to test whether an internal variable is *used*, you need tasks that force
the model to report it, compute with it, or ignore it. All 80 E4 stimuli share the same
five condition prompts (verified: exactly one unique condition set across all stimuli):
<!-- data/e4_stimuli.json, "conditions" of every stimulus (uniqueness checked) -->

| condition | verbatim prompt | graded on |
|---|---|---|
| `continuation` | `Transcribe this document as plain text.` | 60-token greedy rollout, similarity to clean |
| `anomaly` | `Does this document look like a normal, complete business document? Answer Yes or No.` | Yes-vs-No logit margin |
| `report_geom` | `Does the line-item table cover more than a quarter of the page? Answer Yes or No.` | Yes-vs-No logit margin |
| `compute_geom` | `If the line-item table covers more than a quarter of the page answer A, otherwise answer B.` | A-vs-B logit margin |
| `report_type` | `What kind of document is this? Answer with one word.` | invoice-vs-order logit margin |

Grading detail: a "margin" is `max(logits of positive variants) − max(logits of negative
variants)` at the last position, where the variant sets are all single-token encodings of
the surface forms (`"Yes"`, `" Yes"`, `"yes"`; `"invoice"`, `" invoice"`, `"Invoice"`,
`" Invoice"`; etc.). <!-- code/e4_selectivity.py: toks()/marg(); code/e7_dose.py -->

Stimulus set: 80 DocILE pages, 40/40 split on table height (`tall_split: 40`, median
threshold `table_height_frac = 0.098`), **disjoint from E3's 400 probe-training pages**
(recorded in the file itself: "docs disjoint from E3's 400 (probe-contamination guard)"),
stimulus seed 20260829. <!-- data/e4_stimuli.json top-level fields -->

### 1.4 Where measurements are read (the "last token" convention)

Plain English: in a chat-formatted prompt, the most informative position is the very last
prompt token — the position whose next-token distribution *is* the model's first answer
token. That is where "what is this state about to say" is best defined.

- **Lens readouts** (the per-layer top-10 token lists): `positions=[-1]` — the last prompt
  token after chat templating. <!-- code/e1_diag.py: lens.apply(..., positions=[-1]); release/README.md quickstart -->
- **Diagnostic per-layer statistics** (agreement, kurtosis): the last `min(64, T-17)`
  positions of each prompt, stratified into image-token vs text-token positions (image
  token id 151655). <!-- code/e1_diag.py: positions=list(range(-min(64, ids.shape[1]-17), 0)); img_mask -->
- **E3 probe features**: the **mean of the last 32 positions'** residuals per layer
  (one 2048-d vector per document per layer). <!-- code/e3_probes.py: rec.activations[L][0, -32:].float().mean(0) -->
- **Intervention grading**: final-position logits, `hf(**enc).logits[0, -1]`.
  <!-- code/e4_selectivity.py run(); code/e7_dose.py; code/e10_matched_disp.py -->
- **Fitting itself** averages over positions 16 … T−2: the first 16 positions are excluded
  (attention-sink statistics) and the final position has no next-token target.
  <!-- code/vendor-jacobian-lens/jlens/fitting.py: SKIP_FIRST_N_POSITIONS = 16, valid_position_mask -->

---

## 2. Fitting protocol

### 2.1 What is being fitted

Plain English: for each of the 35 decoder layers, the fit estimates one 2048×2048 matrix
`J_l` — the average sensitivity of the final layer's state to the layer-l state. The
readout is then `lens_l(h) = unembed(J_l · h)`: transport the layer-l state to the final
basis, then decode with the model's own output head. The estimator (Anthropic's
reference code, vendored at a pinned SHA) computes, per prompt, one forward pass with the
prompt replicated `dim_batch` times, then `ceil(2048 / dim_batch)` backward passes; each
backward injects one-hot cotangents at `dim_batch` output dimensions at every valid target
position at once, and the resulting gradient rows (summed over later target positions,
averaged over source positions) are rows of `J_l`. Per-prompt Jacobians are accumulated
as a running mean over the corpus.
<!-- code/vendor-jacobian-lens/jlens/fitting.py module docstring + jacobian_for_prompt() + fit(); code/VENDOR.md -->

The lens is defined over the **text decoder only**; the vision tower is run once per
prompt to produce merged input embeddings (cached, no gradients into the tower), and the
fitting estimator replays those cached embeddings. <!-- code/jlens_vlm.py header comment + encode()/forward() -->

### 2.2 Corpus, sharding, precision

- **Corpus**: the 1,000-prompt manifest of §1.1 (900 DocILE + 100 WikiText — verified counts).
  <!-- data/e1_prompts_manifest.json; code/e1_merge.py meta: "900 DocILE pages + 100 wikitext, 8 shards x 125" -->
- **Sharding**: 8 interleaved shards of 125 prompts (`prompts[shard::8]`), each fitted
  independently on its own GPU. Interleaving (not chunking) means every shard sees the
  same mix of instructions and text prompts. <!-- code/e1_fit.py: [SHARD::NSH], NSHARDS=8 in code/e1_fit.sbatch -->
- **Precision**: full **fp32** model load and fit; lenses stored fp16 on disk (294 MB),
  loaded fp32. <!-- code/e1_fit.py: dtype=torch.float32; release/README.md lens table -->
- **Determinism**: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, no sampling anywhere in fit or
  readout (interventions use greedy decoding). <!-- code/e1_fit.sbatch; docs/reproducibility.md PINS -->
- **Estimator settings**: `max_seq_len=512`; `dim_batch=16` on the H200 shards (set
  explicitly in the sbatch; the driver otherwise auto-scales by GPU memory: 32 at ≥100 GB,
  12 at ≥70 GB, 4 at ≥38 GB, 2 below) → 128 backward passes per prompt at `dim_batch=16`.
  <!-- code/e1_fit.py: MAXSEQ default 512, _auto tiers; code/e1_fit.sbatch: E1_DIM_BATCH=16 -->

### 2.3 Checkpoint / resume

Every prompt, the running Jacobian sum is checkpointed **atomically** (write to temp file,
`os.replace`) with enough state to resume exactly (`n_done`, `next_idx`, and the fit
hyperparameters, which are validated on resume). The shard driver also exits idempotently
if its final `.pt` already exists — shards can be killed and resubmitted freely. This was
load-bearing in practice: the fit survived a SLURM timeout, an OOM (an adapter cache leak,
fixed to a single-entry cache), and a queue-priority shuffle, all by resuming from
checkpoints. <!-- code/vendor-jacobian-lens/jlens/fitting.py: _atomic_save, resume logic, checkpoint_every=1; code/e1_fit.py idempotent exit; code/jlens_vlm.py single-entry cache comment; artifacts/trace/LEDGER.md rows 2026-08-25/26 -->

### 2.4 Merge

The 8 shard lenses are merged into the canonical lens by `JacobianLens.merge` (running-mean
average over the full shard list in one call), producing
`data/lenses/qwen2.5-vl-3b-instruct.e1.pt` plus a provenance meta JSON recording model,
revision, corpus, shard filenames, and torch/transformers versions. Merged-lens results
reproduced shard-0's diagnostics "to the decimal" (agreement ramp and receipt readouts
identical) — the merge is verified, not assumed.
<!-- code/e1_merge.py; artifacts/trace/LEDGER.md row 2026-08-26 "Merged-lens certification" -->

### 2.5 GPU cost

- **Base lens fit: ~55 H200-hours** (8 shards × ~7 h, including failures/restarts).
  An earlier draft claim of "~130 H200-h" conflated the campaign total with the fit; the
  LEDGER correction (2026-08-27) fixes this: base fit ≈55 GPU-h, **full study ≈130 GPU-h**
  (both fits plus every evaluation and intervention table).
  <!-- artifacts/trace/LEDGER.md row 19 ("total ~55 GPU-h incl. failures") + row 2026-08-27 first-claim correction; release/README.md §(b) budget -->
- **Twin lens fit**: 4 shards on A100-80GB at roughly 2× the per-shard time.
  <!-- release/README.md §(b); code/e1_twin_fit.sbatch (--gres=gpu:a100:1, --array=0-3) -->

### 2.6 The twin fit (500-prompt half)

The OCR fine-tune's lens was fitted on **500 prompts: shards 0–3 of the identical 8-shard
manifest** — the interleaved half (~450 DocILE + ~50 WikiText), same instructions, same
settings. Half-corpus fitting is justified quantitatively: the base fit's measured corpus
saturation (1-shard vs 3-shard band-curve drift ≤ 0.001, §3.3) plus a twin-specific guard
(twin shard0-vs-shard1 median cosine 0.9970 ≥ the base floor 0.9966).
<!-- release/lens_cards/nanonets-ocr2-3b-e1-lens.md "Fitting corpus"; code/e1_twin_fit.sbatch: --array=0-3 with E1_NSHARDS=8; artifacts/trace/LEDGER.md row 2026-08-26 "TWIN COMPLETE" (D2 guard) -->

### 2.7 Pinned revisions (and why the pin is load-bearing)

| artifact | id / revision |
|---|---|
| `Qwen/Qwen2.5-VL-3B-Instruct` (base) | `66285546d2b821cf421d4f5eb2576359d3770cd3` |
| `nanonets/Nanonets-OCR2-3B` (twin) | `c3886ff00bb037ce7da24988c9eafaf1fe2bed72` |
| `Qwen/Qwen2.5-0.5B-Instruct` (E0 vehicle) | `7ae557604adf67be50417f59c2c2f167def9a775` |
| `anthropics/jacobian-lens` (instrument) | `581d398613e5602a5af361e1c34d3a92ea82ba8e`, vendored |
| page renderer | poppler `pdftoppm` 26.02.0, `-png -r 120 -f 1 -l 1 -singlefile` |
| cluster env | conda, torch 2.8.0+cu128, Python 3.12 (freeze ships with release) |

<!-- docs/reproducibility.md PINS table; code/VENDOR.md; release/README.md model pins -->

The twin pin is not bureaucracy: the OCR2-3B checkpoint **ships no `lm_head.weight` and
its config mis-sets `tie_word_embeddings=False`**, so a naive HuggingFace load silently
random-initializes the output head. The adapter detects untrained/untied heads with a
cheap cosine check against the embedding rows and re-projects through `embed_tokens` (the
intended tied head); on the transformers version used, HF itself tied the head at load and
the check confirmed it (`tied-check cos=1.000`). If upstream ever fixes the config,
unpinned reproductions would silently load a *different* head and get different numbers —
the pin freezes the measured behavior. <!-- code/jlens_vlm.py _resolve_unembed(); release/README.md "Model pins"; release/lens_cards/nanonets-ocr2-3b-e1-lens.md "Checkpoint quirk"; artifacts/trace/LEDGER.md row 2026-08-25 "Twin unembed guard resolution" -->

---

## 3. Validation ladder

Plain English: before trusting anything the instrument says about a new model class, we
climbed a ladder of checks — does our copy of the instrument reproduce its published
behavior on known ground (E0)? Does the fit converge (shard stability)? Was the corpus
big enough (saturation)? And do the downstream metrics carry their own built-in
positive/negative controls (in-run calibration)?

### 3.1 E0 — instrument gate (hard gate; run before anything else)

Protocol: fit a lens with the untouched vendor code on a small **text** model
(`Qwen/Qwen2.5-0.5B-Instruct`, 1,000 WikiText prompts, `max_seq_len=128` — the vendor's
own spec), then run the **six lens-evaluation sets bundled with the vendor repo**, scoring
J-lens vs the classical logit lens. Metric: for each item, the best (minimum-over-layers)
rank of any single-token encoding of the item's "intermediate" concept, at a single
position (last prompt token; for poetry, the last newline token); pass@k for
k ∈ {1,2,5,10,20,50,100} and a normalized AUC over log k. Registered pass criterion:
J-lens AUC ≥ logit-lens AUC on ≥ 4 of 6 sets. <!-- code/e0_fit.py; code/e0_eval.py -->

Result: **J-lens wins 6/6 sets**, and the margin *pattern* matches the paper's own
reporting — largest wins exactly where the paper reports the lens shines:
typo-correction AUC 0.497 vs 0.124 (**4.0×**), multilingual 0.371 vs 0.176 (**2.1×**),
multihop 0.443 vs 0.331, order-ops 0.287 vs 0.229; poetry and association are near zero
for both instruments at 0.5B (known scale limits) with J-lens still ahead.
<!-- artifacts/evidence/tables/E0_lens_evals.json (all AUC values re-read; 6/6 and criterion flag in file) -->

Bonus finding built into the gate: **48 of the eval items' intermediate concepts have no
single-token encoding at all** and are unscoreable by either lens — the paper's own
evaluation data quantifying the single-token confound this project audits. (An earlier
draft said 45; the verification workflow corrected it to 48 — the sum of
`n_multitoken_skipped` across the six sets: 3+9+14+22+0+0.)
<!-- artifacts/evidence/tables/E0_lens_evals.json n_multitoken_skipped fields; artifacts/trace/LEDGER.md row 2026-08-27 "45→48 transcription error fixed" -->

### 3.2 Shard stability (does the fit converge?)

Two lenses fitted on **disjoint 125-document shards** agree per layer at
**CKA 0.998–0.9999 (median 0.999)**, raw cosine 0.994–1.000. The instrument converges at
125 documents; 8 shards of it is comfortable margin. (Caveat recorded with the result:
agreement measures precision, not accuracy — hence the rest of the ladder.)
<!-- artifacts/evidence/tables/E1_shard_stability.json (35 layers re-read: CKA min 0.998, median 0.9989, max 0.9999) -->

### 3.3 Corpus saturation (was 1,000 prompts enough?)

Three independent 125-doc shard fits agree pairwise at cosine 0.9966–0.9984, and the
band-structure curve (effective dimensionality by depth, §4.4) moves by at most
**|Δ| = 0.001** between a 1-shard and a 3-shard mean — the curve was already converged
before the full corpus finished. The merged 8-shard lens could not and did not move it.
<!-- artifacts/evidence/tables/E1_saturation_3shard.json: three_way_cos, band_drift_1v3_max = 0.00098 -->

### 3.4 Twin sanity

The twin lens is **high-but-not-identical** to the base lens: per-layer cosine median
0.940, minimum 0.9304, rising to 0.9996 at the last layer — the signature expected of a
fine-tune twin rather than a re-fit artifact (identical would mean the fit ignored the
weights; uncorrelated would mean a broken fit).
<!-- artifacts/evidence/tables/E1_twin_vs_base.json: median 0.94, min 0.9304, cos_by_layer L34 = 0.9996 -->

### 3.5 In-run calibration (every measurement carries its own ruler)

Plain English: instead of trusting a metric because it sounds right, each E2 measurement
job embeds a known-answer test. The J-lens's own vectors are nameable **by construction**
(each one *is* a token direction), so they must max out any nameability metric — that is
the ceiling. Covariance-matched random directions must sit at the floor — that is the
null. A metric that fails to separate ceiling from null voids the run, automatically.

- Shipped pattern: nameability ("loading") of the lens-vector population P5 = **1.000
  median at every layer**; covariance-matched null P4 ≈ **0.08** (0.066–0.084 by layer in
  the base model; 0.057–0.079 in the twin) — a ~12× ceiling-to-null separation, printed
  in-job before any hypothesis number is read.
  <!-- code/e2_bscore.py "IN-JOB CALIBRATION GATE"; artifacts/evidence/tables/E2_H1_stats_v4.json load_ceiling/load_null per layer; E2t_H1_stats_v4.json -->
- This gate has teeth: two earlier metric candidates (vocabulary-kurtosis, then top-1-z)
  were **voided by their own embedded calibration** — the ceiling population failed to
  dominate the null — and the runs were discarded and re-designed rather than reported.
  The LEDGER records each casualty. <!-- artifacts/trace/LEDGER.md rows 2026-08-25 "E2 v1 VERDICT WITHHELD", "CALIBRATION ... RETIRED" -->

---

## 4. Measurement definitions

### 4.1 The lens readout

Plain English: take the model's internal state at layer L (a 2048-number vector), push it
through the fitted transport matrix for that layer, then through the model's own output
vocabulary projection. Out comes a score for every vocabulary token: "if the network's
remaining layers act on this state the way they do on average, this is what the model is
disposed to say."

Formula: `lens_l(h) = W_U · RMSNorm(J_l · h)` where `J_l` is the fitted 2048×2048
corpus-average Jacobian for layer l, RMSNorm is the model's own final norm, and `W_U` is
the model's own (tied) unembedding — for the twin, the tied `embed_tokens` projection
(§2.7). <!-- code/vendor-jacobian-lens/jlens/fitting.py docstring: lens_l(h) = unembed(J_l @ h); code/jlens_vlm.py unembed() -->

### 4.2 Broadcast score B (is a direction *listened to*?)

Plain English: a "global workspace" broadcasts — whatever is written into it is read by
many downstream components. The broadcast score asks, from **weights only** (never from
the lens, so the test cannot be circular): if the residual stream moves one unit along
direction d, how loudly do the next layer's components respond, compared to their response
to a random direction?

- `B_mlp(d, l)` = `‖MLP_{l+1}(post_attention_layernorm(d))‖`, normalized by the median of
  the same quantity over isotropic random unit directions. (A raw variant without the
  layernorm is computed alongside; both are reported.)
  <!-- code/e2_bscore.py b_mlp() -->
- `B_attn(d, l)` = mean over the 16 query heads of `‖W_O[:, head] · (W_V d)_kv(head)‖`,
  normalized the same way — **using the GQA-correct layout**: Qwen2.5-VL-3B has 16 query
  heads but only 2 KV heads, and query head q reads KV head `q // 8`. An earlier
  implementation reshaped the 256-dim value output into 16 fake 16-dim heads and touched
  only the first 256 columns of `W_O`; the adversarial audit caught it (CRITICAL-1), the
  metric was rewritten, and **all B_attn-dependent results in this note are from the
  corrected re-runs**. <!-- code/e2_bscore.py b_attn() with GQA comment; artifacts/trace/LEDGER.md rows 2026-08-26 "ADVERSARIAL AUDIT" (CRITICAL-1) and "GQA-CORRECTED E2/E2t + RESIDUALIZED E3 LANDED" -->
- `B` = the rank-average of (`B_mlp`, `B_attn`) within the population being scored.
  <!-- code/compute_h1_stats.py definition block -->

Populations scored (1,500 directions each, per layer, at layers {12, 16, 21, 26, 30} —
the workspace band per the triangulated depth map of §4.4): **P1** MLP neuron output-weight
rows (the model's own working directions), **P3** isotropic random (weak null),
**P4** covariance-matched random (the load-bearing null — random directions with the same
second-order statistics as real residuals, built by Cholesky-transforming Gaussians with
the covariance measured from live document forwards over the 26 diagnostic pages),
**P5** J-lens vectors themselves (ceiling), plus **P2** the E3 probe directions in a
second pass. Seed 20260826. <!-- code/e2_bscore.py: LAYERS, NDIR=1500, populations, chol(cov + 1e-4 I); prereg/H1_broadcast_vs_verbalizability.md "Candidate populations" -->

### 4.3 Nameability ("loading") and the H1 statistics

Plain English: does a direction have a single-token name? Measure the best cosine
similarity between the direction and any of the ~151k token readout vectors of the lens.
A lens vector scores 1.0 by definition; a random direction in 2048 dimensions scores about
0.08–0.11 against the best of 151k candidates.

- `loading(d, l)` = `max over vocab of |cos(d, v_tok(l))|`, where `v_tok(l)` are the rows
  of `W_U · J_l`, unit-normalized. Analytic random-direction expectation
  ≈ `sqrt(2·ln V / d)` ≈ 0.11; the empirical covariance-matched floor is ≈ 0.08.
  <!-- code/e2_bscore.py loading() with docstring -->
- **H1 dissociation statistic** `f` = the fraction of P1's top decile by broadcast B whose
  loading falls **below the covariance-matched null's median** (the "floor"). Reported
  with: Spearman ρ(B, loading) over all of P1; the below-floor **base rate** over all of
  P1; the top-decile **excess** over that base rate; and a 20,000-draw permutation p-value
  for the excess (the audit's additions — the registered f ≥ 0.25 bar alone turned out to
  be uninformative because the base rate is high). Generating script checked into the
  repo; it reproduces every shipped value from the raw arrays.
  <!-- code/compute_h1_stats.py (definitions + audit additions); artifacts/evidence/tables/E2_H1_stats_v4.json -->
- Pre-registration: the H1 criteria (support f ≥ 0.25 with ρ < 0.5; falsify f < 0.10 or
  ρ > 0.7; ambiguous zone in between) and the E4 selectivity predictions (report/compute
  conditions flip ≥ 60%, continuation/anomaly move ≤ 20%) were committed to git on
  **2026-08-24** (commit `06453d5`), before any fit or measurement ran (E1 fit completed
  2026-08-26). Follow-up experiments designed after seeing results (the E6/E7 dose ladder,
  E10) live in files explicitly named `POSTHOC_*`, with their criteria still committed
  before their runs. <!-- prereg/H1_broadcast_vs_verbalizability.md; git log for prereg/ (06453d5 dated 2026-08-24, b98501b/1061dc5 for post-hoc criteria); prereg/POSTHOC_transfer_audit.md -->

### 4.4 Band detectors (where is the "workspace" in depth?)

Plain English: the source paper describes a three-zone depth structure — early "sensory"
layers, a middle low-dimensional "workspace" band, late "motor" layers that type the
actual next token. We located the analogous band in the VLM with three *independent*
detectors, and only claimed a band where they agree:

1. **Effective dimensionality** (`k90/d`): the fraction of singular directions of the
   layer's transport carrying 90% of its variance. Low = the layer funnels through a
   narrow bottleneck (workspace-like); high = it passes (almost) everything (motor-like).
   Two variants: on `J_l` (three-shard mean: 0.069 at L0, plateau ≈ 0.07–0.13 through
   ~L16, rising to 0.799 at L34) and the **exact** paper-analog on `W_U·J_l` (merged
   lens: **0.055 plateau → 0.72 at L34**). The VLM's plateau is *lower and longer* than
   the 0.5B text reference's (≈7% of dims vs ≈11%).
   <!-- artifacts/evidence/tables/E1_effdim_band.json (metric field: "k90/d: fraction of singular directions carrying 90% of J_l variance"); E1_saturation_3shard.json effdim_3shard_mean; E1_merged_lens_anatomy.json exact_WU_effdim; artifacts/trace/LEDGER.md row 2026-08-25 "BAND (effective dimensionality)" -->
2. **Agreement ramp**: per position, does the lens's top-1 token match the model's actual
   next-token prediction? Inside a workspace band it should not (the state is still
   "thinking", not "typing"); at motor layers it must converge. Measured on the merged
   lens over 26 held-out docs: agreement sits at ≈14–21% through the band (0.136–0.206
   over L9–L24) and ramps 31% → 59% over L30 → L34. <!-- code/e1_diag.py agree metric; artifacts/evidence/tables/e1merged_diag.json per_layer.agree (0.31 at L30 → 0.593 at L34); artifacts/trace/LEDGER.md row "diag v2 (FIXED agree metric)" -->
3. **Readout content**: what the top-10 readout at the last prompt token actually says,
   per layer. In the band it names task *schema*; at motor it degenerates into the literal
   next token. Verified example (receipt page, merged base lens): L26 top-10 =
   `Total, Sheet, Yes, Item, Number, ---, Date, Invoice, Name, No`; L30 =
   `Number, Invoice, Here, ---, Name, No, ───, Date, Purchase, Sheet`; L34 = token
   fragments (`N, 1, No, Here, ...`). <!-- artifacts/evidence/tables/e1merged_diag.json readouts_last_token["image_15_receipt.png"] -->

The three detectors triangulate the band to ≈ L10–30 with motor at ≈ L33+, which fixed the
measurement layers {12, 16, 21, 26, 30} used by E2/E3/E4. (A fourth candidate detector —
a kurtosis variant — was *retired* after failing a same-metric control on a known-good
lens; the calibration habit again.) <!-- artifacts/trace/LEDGER.md rows 2026-08-25 "diag v2", "CALIBRATION"; code/e2_bscore.py LAYERS comment -->

### 4.5 Probe protocol (what does the band decodably contain?)

Plain English: a linear probe asks "can a straight-line rule read variable X out of the
layer-L state?" Success means the information is present and linearly available; the probe
direction then becomes a candidate handle for causal tests.

Exact protocol: <!-- code/e3_probes.py -->
- 400 labeled DocILE pages (labels from DocILE's own annotations, `data/e3_labels.json`),
  captured under the neutral transcription prompt; features = mean of the last 32
  positions' residuals, per layer {12, 16, 21, 26, 30}.
- **Ridge regression** (λ = 10) on a **document-level 70/30 train/validation split**
  (seeded permutation, seed 20260827 — no page of a validation document ever appears in
  training).
- Continuous variables scored by validation R²; categorical variables by one-vs-rest ridge
  on the largest class, scored by validation AUC.
- Variables: `document_type`, `currency` (named, single-token answers exist);
  `cluster_id`, `table_y_centroid`, `table_height_frac`, `n_line_items` (candidate
  nameless geometry); plus the audit-mandated **residualized pair**
  (`table_height_resid`, `n_line_items_resid`) — height and row count correlate 0.78 in
  the labels, so each is re-probed after regressing out the other (train-only fit of the
  residualizer, no validation leakage).
- Each probe direction is then itself scored for loading and (GQA-correct) B — the P2
  population — and saved (`data/probe_dirs.pt`) as the intervention handles for E4.

Headline probe results (current, post-audit file): document_type AUC 0.946–0.961
across layers; currency AUC 0.79–0.86; table_height_frac R² 0.41–0.49; n_line_items
R² 0.13–0.33; both residualized variables ≈ 0 — the band encodes **one** shared
table-magnitude variable, not separately-decodable "height" and "row count". Probe-direction
loadings span 0.089–0.151, with the *named* variable (document_type) on top.
<!-- artifacts/evidence/tables/E3_probes.json (all values re-read from the current, residualized, GQA-corrected file); artifacts/trace/LEDGER.md row "GQA-CORRECTED ... C10 ADJUDICATED" -->

### 4.6 Steering protocol (is the variable *used*?)

Plain English: reading is not using. To test use, we overwrite the variable inside the
network mid-forward and watch the behavior. The write is a "band-swap with clamp": at
three workspace layers simultaneously, at every position, the state's coordinate along the
probe direction is set to the value typical of the *opposite* class, leaving all
orthogonal coordinates untouched. A dose knob α scales the applied displacement.

Exact mechanics: <!-- code/e4_selectivity.py hooks; code/e7_dose.py -->
- Intervention layers: **{12, 16, 21}** (the early/mid band; forward hooks on the decoder
  blocks' outputs).
- Clamp: `h ← h + α · (t_opposite − h·d̂) · d̂` per position, where `d̂` is the layer's
  unit probe direction and `t_opposite` is the **class-mean projection** measured in a
  separate clean pass over all 80 stimuli (pass 1 records mean projections per class per
  layer; pass 2 intervenes).
- Grading: the five conditions of §1.3, margins at the last position; continuation graded
  by `difflib.SequenceMatcher` similarity between clean and steered 60-token greedy
  rollouts.
- **Dose ladder** (E7, post-hoc-labeled): α ∈ {2, 4, 8, 16, 32} on the named dial
  (document_type). Context numbers recorded with it: clean class separation ≈ 1.3–1.4
  projection units per layer; mean residual norm ≈ 116–134 — so the α=1 "transported
  dose" (the source paper's own clamp-to-class-mean) moves the state by ~1% of its norm.
  <!-- artifacts/evidence/tables/E7_dose_ladder.json: class_sep, mean_residual_norm; prereg/POSTHOC_transfer_audit.md -->
- Dose results (re-verified from the raw file): probe-direction flips 3/80 (α=2), 4/80
  (α=4), 66/80 (α=8), 70/80 (α=16), 68/80 (α=32) vs plain-random 2, 4, 15, 10, **61**/80 —
  specificity exists at α=8–16 and is destroyed at α=32. Audit correction carried
  forward: the α=8 effect is a one-sided answer collapse (everything → "order");
  **genuine bidirectional class swapping exists only at α=16**. At that effective dose the
  applied displacement is ≈8× the class separation and ≈17% of the residual norm.
  <!-- artifacts/evidence/tables/E7_dose_ladder.json per_doc (flip counts recomputed); artifacts/trace/LEDGER.md rows "ADVERSARIAL AUDIT" (MODERATE-d) and 2026-08-27 verification row (dose sentence 8x/17% @α=16) -->
- **Two random-direction controls**: the original arm clamps a matched random direction to
  the same targets — the audit showed this applies 2–3.9× *larger* displacement than the
  probe arm, so it is valid only as a nonspecific-damage control. **E10** (the prereg C-3
  fix) applies, per position, a displacement of *exactly* the probe arm's magnitude along
  the random direction (fixed random sign per document). Under this fair,
  displacement-matched control: probe 65/80 vs matched-random 5/80 flips at α=8; 68/80 vs
  16/80 at α=16 — the steering is direction-specific, not just damage.
  <!-- code/e10_matched_disp.py (mechanism + header note); artifacts/evidence/tables/E10_matched_disp.json (flip counts recomputed from per_doc) -->
- The pre-registered four-condition selectivity run itself (E4 on the fallback nameless
  variable, α=1): report_geom flips 0/80, compute_geom 6/80 vs random 2/80, anomaly 2/80
  vs 4/80, continuation similarity 0.98 — which triggered the dose-ladder and
  positive-control program above rather than a headline claim.
  <!-- artifacts/evidence/tables/E4_selectivity.json (recomputed); artifacts/trace/LEDGER.md rows "E4 PROPER COMPLETE", "DISCRIMINATOR COMPLETE" -->

---

## 5. Reproducibility — what ships

<!-- release/README.md; release/PREFLIGHT.md; docs/reproducibility.md; artifacts/trace/LEDGER.md row "TIER-1 RELEASE SKELETON" -->

The Tier-1 release tree (`release/`, built and locally tested 2026-08-27) contains:

- **Both fitted lenses** — the highest-leverage artifact; nobody has to re-spend the
  GPU-hours: `qwen2.5-vl-3b-instruct.e1.pt` (md5 `30f68189923fde4a747ef37177cd138f`) and
  `nanonets-ocr2-3b.e1.pt` (md5 `c5e79548e45488fac927065811c37a0d`), 294 MB each, fp16 on
  disk / fp32 on load, 35 Jacobians per lens; the md5s are cross-recorded inside
  `E1_twin_vs_base.json`, tying every result table to exactly these files.
  <!-- release/README.md lens table; artifacts/evidence/tables/E1_twin_vs_base.json base_md5/twin_md5 -->
- **All fitting/evaluation code** (14 drivers + the VLM adapter) and the **vendored
  Anthropic lens implementation** at pinned SHA `581d3986` (Apache-2.0; the whole release
  is Apache-2.0). <!-- release/README.md Contents + License -->
- **The evidence tables** (33 in-scope raw JSONs — every number in the paper) and **one
  script per figure** that reads only those tables (no GPU, no model download needed to
  regenerate every figure), plus `compute_h1_stats.py`, which reproduces the H1 statistics
  file exactly from the raw E2 arrays (tested). <!-- release/README.md §(d); artifacts/trace/LEDGER.md TIER-1 row, tested items (2)(3) -->
- **Corpus manifests, not images**: the 900 DocILE document IDs, the pinned render
  command, the manifest builder, and the 100 WikiText prompts. DocILE requires (free)
  registration and its license bars redistributing renders, so a reproducer downloads
  DocILE and re-renders — standard practice. <!-- docs/reproducibility.md "What reproduces"; release/README.md §(b) -->
- **Seeds and environment stamps**: fits are deterministic (fp32,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, no sampling); every stochastic evaluation carries an
  explicit seed (E2: 20260826; E3: 20260827; E4 stimuli: 20260829; E4/E7/E10 runs:
  20260830); and **every output JSON stamps the torch/transformers versions and model
  revision that produced it**, so environment drift is detectable — the failure mode a
  prior audited project actually hit. <!-- docs/reproducibility.md PINS + environment-drift note; seeds read from each script/JSON -->
- **Lens cards** for both lenses (fit corpus, validation, intended use, limitations —
  including the two limitations a user must not ignore: readouts are corpus-conditional
  averages, and readout absence is not representational absence).
  <!-- release/lens_cards/qwen2.5-vl-3b-instruct-e1-lens.md; release/lens_cards/nanonets-ocr2-3b-e1-lens.md -->
- `reproduce.md` (exact end-to-end command sequence, with SLURM template) and
  `PREFLIGHT.md` (the human gates before publishing).

Status honesty: as of this writing the release tree is **built and locally verified but
not yet public** — the repo has no git remote, and publication waits on the PREFLIGHT
human gates (org/account choice, HF repo + md5 re-verify, DocILE-derivative license
re-check, sign-off, clean-machine sweep). Public-facing text must say "we release" only
after those gates clear. <!-- git remote -v (empty); release/PREFLIGHT.md; artifacts/trace/LEDGER.md rows "VERIFICATION WORKFLOW COMPLETE" item (7) and "TIER-1 RELEASE SKELETON" -->

Not released: raw activation dumps (too large; regenerate via scripts), DocILE-derived
images (license), anything Nanonets-internal. <!-- release/README.md Scope; docs/reproducibility.md -->

---

## Numbers table — every headline metric a reader might quote

| # | metric | value | source (verified) |
|---|---|---|---|
| 1 | Fit corpus | 1,000 prompts = 900 DocILE pages + 100 WikiText | `data/e1_prompts_manifest.json` (counted) |
| 2 | Rotating instructions | 4, exactly 225 pages each | `data/e1_prompts_manifest.json` (counted) |
| 3 | Base-fit sharding | 8 interleaved shards × 125 prompts | `code/e1_fit.py` + `code/e1_fit.sbatch` |
| 4 | Twin fit | 500 prompts (shards 0–3 of 8), 4 shards merged | `release/lens_cards/nanonets-ocr2-3b-e1-lens.md`; `code/e1_twin_fit.sbatch` |
| 5 | Fit precision / determinism | fp32, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, no sampling | `code/e1_fit.py`; `code/e1_fit.sbatch`; `docs/reproducibility.md` |
| 6 | Fit max_seq_len / dim_batch | 512 / 16 (H200 shards; auto-tiered elsewhere) | `code/e1_fit.py`; `code/e1_fit.sbatch` |
| 7 | Positions excluded from fit | first 16 (sink) + final position | `code/vendor-jacobian-lens/jlens/fitting.py` |
| 8 | Base-lens fit cost | ~55 H200-hours (8 × ~7 h, incl. failures) | `artifacts/trace/LEDGER.md` row 2026-08-26; `release/README.md` |
| 9 | Full-study budget | ~130 GPU-hours (H200-class) | `release/README.md`; LEDGER 2026-08-27 correction |
| 10 | Lens size on disk | 294 MB each (fp16), 35 Jacobians of 2048×2048 | `release/README.md`; lens cards |
| 11 | Base lens md5 | `30f68189923fde4a747ef37177cd138f` | `release/README.md`; `E1_twin_vs_base.json` |
| 12 | Twin lens md5 | `c5e79548e45488fac927065811c37a0d` | `release/README.md`; `E1_twin_vs_base.json` |
| 13 | Base model pin | Qwen2.5-VL-3B-Instruct @ `6628554…` | `docs/reproducibility.md` PINS |
| 14 | Twin model pin | Nanonets-OCR2-3B @ `c3886ff…` | `docs/reproducibility.md` PINS |
| 15 | Instrument pin | anthropics/jacobian-lens @ `581d3986` | `code/VENDOR.md` |
| 16 | E0 gate result | J-lens beats logit lens **6/6** sets (criterion ≥4/6) | `artifacts/evidence/tables/E0_lens_evals.json` |
| 17 | E0 margin pattern | typo AUC 0.497 vs 0.124 (4.0×); multilingual 0.371 vs 0.176 (2.1×) | `E0_lens_evals.json` |
| 18 | E0 unscoreable intermediates | 48 (no single-token encoding) — the confound, quantified | `E0_lens_evals.json` (Σ n_multitoken_skipped); LEDGER 45→48 correction |
| 19 | Shard stability | CKA 0.998–0.9999 per layer (median 0.999) on disjoint 125-doc shards | `E1_shard_stability.json` |
| 20 | Corpus saturation | 1-vs-3-shard band drift ≤ 0.001; 3-way cos 0.9966–0.9984 | `E1_saturation_3shard.json` |
| 21 | Twin-vs-base lens cosine | median 0.940, min 0.9304 (early/mid), 0.9996 at L34 | `E1_twin_vs_base.json` |
| 22 | Effective dimensionality (exact, W_U·J_l) | 0.055 plateau → 0.72 at L34 | `E1_merged_lens_anatomy.json` exact_WU_effdim |
| 23 | VLM vs text plateau | ≈7% of dims vs ≈11% (0.5B reference) | `E1_effdim_band.json` |
| 24 | Agreement ramp (merged lens) | 14–20% in band; 31% → 59% over L30→L34 | `e1merged_diag.json` per_layer.agree |
| 25 | Measurement layers | {12, 16, 21, 26, 30}; steering band {12, 16, 21} | `code/e2_bscore.py`; `code/e4_selectivity.py` |
| 26 | E2 populations | 1,500 directions × 5 populations × 5 layers | `code/e2_bscore.py` NDIR |
| 27 | Calibration ceiling / null (loading) | P5 median 1.000 at every layer / P4 ≈ 0.08 (0.066–0.084) | `E2_H1_stats_v4.json`; `code/e2_bscore.py` gate |
| 28 | H1 flagship (base, L16, GQA-corrected) | f = 0.88, excess +0.215 over base rate 0.665, perm p < 1e-4, ρ = −0.316 | `E2_H1_stats_v4.json` |
| 29 | H1 flagship (twin, L16) | f = 0.728, excess +0.277 over base rate 0.451, perm p < 1e-4 | `E2t_H1_stats_v4.json` |
| 30 | H1 registered layer (base, L21) | f = 0.607, excess +0.074, perm p = 0.033 | `E2_H1_stats_v4.json` |
| 31 | Prereg commit | criteria committed 2026-08-24 (`06453d5`), before all fits/runs | git log, `prereg/H1_broadcast_vs_verbalizability.md` |
| 32 | E3 probes | 400 docs, ridge λ=10, mean-last-32 features, doc-level 70/30 split, seed 20260827 | `code/e3_probes.py`; `E3_probes.json` |
| 33 | document_type probe | AUC 0.946–0.961 (best 0.961 @L30), loading up to 0.151 | `E3_probes.json` |
| 34 | currency probe | AUC 0.79–0.86 (0.856 @L12) | `E3_probes.json` |
| 35 | table_height_frac probe | R² 0.41–0.49 (0.491 @L12) | `E3_probes.json` |
| 36 | n_line_items probe | R² 0.13–0.33 (0.331 @L16) | `E3_probes.json` |
| 37 | Residualized probes | R² ≈ 0 for both → one shared table-magnitude variable | `E3_probes.json`; LEDGER "C10 ADJUDICATED" |
| 38 | Probe-direction loadings | span 0.089–0.151; named variable on top | `E3_probes.json` |
| 39 | E4 stimuli | 80 docs, 40/40 tall split (threshold 0.098), disjoint from E3's 400, seed 20260829 | `data/e4_stimuli.json` |
| 40 | E4 @α=1 (prereg run) | report_geom 0/80 flips; compute_geom 6/80 vs 2/80 random; cont. sim 0.98 | `E4_selectivity.json` (recomputed) |
| 41 | Dose ladder (probe vs random flips /80) | α=2: 3 vs 2 · α=4: 4 vs 4 · α=8: 66 vs 15 · α=16: 70 vs 10 · α=32: 68 vs 61 | `E7_dose_ladder.json` (recomputed) |
| 42 | Dose-ladder caveat | α=8 = one-sided collapse; bidirectional swap only at α=16 (≈8× class sep, ≈17% of residual norm) | LEDGER audit row (d) + 2026-08-27 correction |
| 43 | Class separation / residual norm | ≈1.3–1.4 units / ≈116–134 (band layers) | `E7_dose_ladder.json` class_sep, mean_residual_norm |
| 44 | Displacement-matched control (E10) | probe 65/80 vs matched-random 5/80 (α=8); 68/80 vs 16/80 (α=16) | `E10_matched_disp.json` (recomputed) |
| 45 | Diagnostic doc set | 26 held-out cross-domain pages (DocVQA, handwritten, equations, receipts, forms, multilingual zh/ar/ja/fr) | `data/diag_docs/` (listed) |
| 46 | Visual budget | fixed: min_pixels = max_pixels = 256·28·28 (~256 image tokens/page) | `code/e1_fit.py` processor args |
| 47 | Page renderer pin | pdftoppm 26.02.0, 120 dpi, first page, single file | `docs/reproducibility.md` PINS |
| 48 | Release status | built + locally tested; **no public remote yet**; PREFLIGHT gates pending | `release/PREFLIGHT.md`; `git remote -v` (empty) |


<!-- RESOLUTION 2026-08-27: the audit's '24/26 orders at α=16' is ENDPOINT-based (docs ending up reporting the swapped class: 24/26 verified from E7_dose_ladder.json probe_a16>0); the 16/26 recomputed here is SIGN-FLIP-based (vs clean). Both correct; definitions differ because 12/26 orders start misclassified. Use endpoint numbers for the bidirectional-swap claim, flip numbers for totals (70/80). -->
