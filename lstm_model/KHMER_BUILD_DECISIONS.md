# KHMER BUILD — RECOGNITION-UNIT & SPLIT DECISIONS (recorded before the build)
**Recorded:** 2026-07-28 (IST) · **Owner:** Ritu Baskey
**Companion:** `PROSPECTIVE_PREDICTION_KHMER.md` (frozen 2026-07-21, pushed `d3a3130`).
The prereg deliberately left `N_test = TBD (KhmerST recognition split)` and permits a re-file "if
the split changes tok_cov materially." This file fixes those open choices; it does **not** touch the
prediction (16.1 WRR, bands, model), which stays frozen.

## What KhmerST actually provides (measured, not assumed)
`khmerst_data/json_v4/` = 1,544 images, **3,563 regions**, VGG-VIA format, polygon geometry per
region (`all_points_x/y`), transcription in `region_attributes.Name`. Annotations are **line-level
only** — there are no word-level boxes, and Khmer does not space-delimit words *within* a phrase.
Measured token structure (spaces per region):
- **0 spaces (single token): 2,672 regions = 75.0%**
- 1 space: 553 (15.5%) · 2: 222 (6.2%) · ≥3: 116 (3.3%)

## Decision 1 — recognition unit = single-token regions (word-level)
Test units = the **2,672 single-token (0-space) regions**. Each carries its own polygon (so the crop
geometry is real, not inferred) and a single-token transcription (a "word"). This matches the
word-level granularity of the nine benchmark scripts, so the WRR metric means the same thing it does
everywhere else in the paper. Multi-token line regions are **excluded** — recognizing a whole line is
a different, harder task not comparable to the benchmark, and cropping sub-words from a line has no
geometry to support it. The exclusion is principled (granularity match), declared here, and stated in
the paper. Final N after pivot-map filtering is reported in `KHMER_SCORING.md`.

## Decision 2 — image-level train/test split (mirrors the benchmark protocol)
KhmerST ships no recognition train/test split. We split at the **image** level (seeded), so no image
contributes crops to both sides. Synthetic Khmer for Rung B is rendered from the **train-image**
region tokens only (via `synth_multiscript.py`, `khmer` added to `SCRIPT` as `(0x1780, "Khmer")`,
raqm/`ritu_scenetext` env only — coeng stacking must shape correctly); the **test-image**
single-token crops are never a synth source. This is the same discipline the benchmark rungs use
(synth from official-train labels, test on held-out images).

## Decision 3 — tok_cov re-check gate (per the prereg)
The prereg's tok_cov = 89.02 was computed over the **full** KhmerST label set. We recompute it over
the **frozen test set** (single-token test-image regions, pivot-mapped by `khmer_pivot_map.py` v1).
If it moves materially (prereg wording), we file a dated re-file note BEFORE training and score
against the updated input; otherwise we proceed on 89.02 and say so. Either way the point prediction
16.1 is not edited.

## Decision 3 — OUTCOME (built 2026-07-28, `prepare_zeroshot_loso_khmer.py`)
Realized split: 1544 images → 772 train / 772 test (seed 42); 2686 single-token regions of 3563.
**Frozen test set N = 1,252** word crops (after dropping 27 empty-pivot + 60 degenerate boxes);
synth lexicon = 1,911 unique train-image words; **3,240 synth rendered** (fonts KhmerOS + KhmerOSsys,
raqm-shaped). **tok_cov over the frozen test set = 88.01%** vs the prereg's 89.02% (**Δ −1.01**);
codepoint map-rate 94.41% vs 94.48%. Through the frozen instrument (0.583·cov) this moves the point
prediction 16.11 → 15.52, a **−0.59 WRR** shift — far inside the ±1·RMSE band [11.9, 20.3] and the
same rank. **Verdict: NOT material → no re-file; the frozen prediction 16.1 stands and we proceed on
it, noting the −0.59 shift here.** (Recorded in `zeroshot_loso_meta_khmer.json`.)

## Rungs (unchanged from the prereg)
- **Rung A:** train on the nine benchmark scripts only (existing synth+real), no Khmer synth;
  evaluate on the Khmer word-crop test set. Prediction: WRR < 5.
- **Rung B:** + 3,240 synthetic Khmer renderings (1×, is_2x = 0); same evaluation. Prediction: 16.1.

## Build order (CPU steps run now; train steps are GPU-gated behind other users' jobs)
1. [CPU] image-level split → derive single-token word crops from test-image polygons → test set + N.
2. [CPU] tokenize train-image regions → Khmer synth lexicon.
3. [CPU] add `khmer` to `synth_multiscript.py`; render 3,240 synth (raqm env); pivot-map labels.
4. [CPU] recompute tok_cov over the frozen test set; re-file if material.
5. [CPU] build Rung A / Rung B split JSONs.
6. [GPU] train Rung A + Rung B (auto-launch when the shared GPU frees).
7. score vs the frozen prereg → `KHMER_SCORING.md` (+ macros, verify, paper §sec:khmer).
