# KHMER OUT-OF-BENCHMARK (Amendment 4b) — SCORING RECEIPT
**Scored:** 2026-08-01 (IST) · **Owner:** Ritu Baskey
**Preregistration:** `PROSPECTIVE_PREDICTION_KHMER.md` (filed + pushed 2026-07-22, commit
`d3a3130`, on `origin/main`, BEFORE any Khmer recognizer trained anywhere). Training ran
2026-07-28 → 2026-07-31 (`train_zeroshot_loso_rung{A,B}_khmer.log`). **Amendment 4b** of
`PREREGISTRATION.md`.
**Sources of truth:** `result_zs_loso_rung{A,B}_khmer.json` (2/2, produced by
`run_khmer_loso.sh` → `khmer_loso.log`); build receipts `KHMER_BUILD_DECISIONS.md`,
`zeroshot_loso_meta_khmer.json`.

## Setup (as frozen)
Identical instrument to the nine benchmark rungs: Florence-2 grapheme-injected, no BPE,
`train_florence2.py` → `predict_with_conf.py` → `metrics.evaluate_corpus`, unfiltered test split.
**N(test) = 1,252** single-token KhmerST region crops (image-level 50/50 split, seed 42).
Rung A = nine benchmark scripts only (7,700 real train words, **no Khmer of any kind**).
Rung B = Rung A + **3,240 synthetic Khmer** words (1×, two system faces), still **zero real
Khmer images**. Khmer is reached through the hand-curated pivot extension `khmer_pivot_map.py`
v1 (codepoint map-rate 94.41% on the test labels). Test text never read during preparation.

## Realized values
| Rung | N | WRR | CharAcc | CER |
|------|--:|----:|--------:|----:|
| A (no Khmer data at all) | 1252 | **0.00** | 3.56 | 96.44 |
| B (+3,240 synthetic Khmer) | 1252 | **0.80** | 9.41 | 90.59 |

B − A = **+0.80 WRR**, **+5.85 CharAcc**.

## Predictions vs realized (both, hit and miss)
| # | Prediction (frozen 2026-07-22) | Realized | Verdict |
|---|--------------------------------|----------|---------|
| 1 | Rung A WRR **< 5** (coverage-tracking) | 0.00 | **HIT** |
| 2 | *Primary:* Rung B WRR **= 16.1**, ±1·RMSE [11.9, 20.3], ±2·RMSE [7.8, 24.5] | 0.80 | **MISS** — below both bands |

**Miss magnitude:** residual −15.30 WRR = **−3.66 RMSE**. Re-evaluating the frozen instrument at
the realized token coverage (88.01 rather than the filed 89.02) moves the point only to 15.52,
i.e. −3.52 RMSE — the miss is **not** an artifact of the coverage recheck already disclosed in
`KHMER_BUILD_DECISIONS.md`.

**Score: 1 hit / 1 miss.** The structural prediction (Rung-A floor) holds; the quantitative
point prediction fails badly, on the low side.

## Interpretation (per the preregistration's declared caveat — resolved against the instrument)
Caveat 1 of the prereg stated, in advance and in writing, that we expected this point to be
fragile: the coverage term does not generalize across the six held-out Brahmic scripts
(out-of-sample Pearson r = +0.03, p = 0.94; `SCALING_SWEEP_SCORING.md`), and "a low-side miss
would be fully consistent with that finding." That is exactly what happened, and we report it
as a miss rather than reinterpreting it.

This is the instrument's **second** scored failure, and the two are distinct: the scaling sweep
falsified its transported slope *magnitude* in block (5.1× outside the CI), and Khmer falsifies
its *point prediction* on the first script we asked it to forecast out of block. Our position is
that the instrument is **not carried forward as a contribution** — it is reported as a scored,
partly-failed forecast. We state the out-of-block conclusion no wider than the evidence: one
script shows it failed there, not that it must fail on every script outside the blocks.

What survives is narrower and structural:
- **Direction replicates 10/10.** Adding synthetic target-script data through the pivot beats a
  no-target-data control on every script attempted, Khmer included (+0.80 WRR, +5.85 CharAcc).
  The mechanism produces a real, non-zero signal even out of block.
- **The magnitude collapses out of block.** Khmer's Rung B is 4.1% of the nine-script Rung-B
  mean (19.41) and 8.7% of the weakest benchmark script (Tamil, 9.16). This is a boundary
  result: it marks where the method stops being practically useful, not merely where it is worse.

## Context: the test set is hard for everyone (off-the-shelf reference)
Stock Tesseract 5's **Khmer-specific** model — trained on real Khmer, scored on the identical
1,252 crops through the identical pivot/metric path — reaches **2.16 WRR**
(`result_anchor_tesseract_khmer.json`), versus its 14.35 mean across the nine benchmark scripts.
So KhmerST word crops are ~7× harder for a supervised per-script system than the benchmark is.
Our zero-real-image Rung B reaches **37% of that supervised-on-real-Khmer floor**. We report
this as context, not as a rescue: 0.80 WRR is a failure to transfer usefully, and the honest
summary is that both numbers are near the floor.

## Bearing on the "data size beats typology" critique
Khmer is the campaign's cleanest counterexample to the reading that synthetic volume, not
typological relatedness, drives the effect (the reviewer threat tracked in
`COMPETITIVE_POSITIONING_AND_LITERATURE.md`). Khmer received the **same 1× synthetic budget**
(3,240) that produced double-digit WRR on in-block scripts, and had **high nominal token
coverage** (88.01%), yet returned 0.80. Volume and coverage were held up; relatedness was not;
the result collapsed. This is n = 1 script and we state it as such — suggestive, not decisive.

## Where reported
`paper_wacv/sec/4_experiments.tex` §"Out-of-benchmark: Khmer" (`\label{sec:khmer}`); macros in
`paper_wacv/numbers.tex` (`\khmer*`); verified by `verify_wacv_numbers.py`; Fig. 4 Khmer point.

## Campaign prospective-call ledger (updated)
kannada ~15 → 15.42 (hit) · gurmukhi 16.2 → 22.16 (2σ hit, point under-called) · scaling slope
+11.5 → +2.2 (magnitude miss; log-linear *form* and script-invariance confirmed) · architecture
CRNN → 2 hit / 2 miss · **khmer Rung A < 5 → 0.00 (hit); Rung B 16.1 → 0.80 (miss, −3.66 RMSE)
(this file)**.

Every filed call is now scored; none is withdrawn or reinterpreted. We deliberately do not
compress the ledger into a single hit-rate — the calls are not commensurable (rank predictions,
slope magnitudes, per-script points, structural floors), and a pooled ratio would imply a
uniformity the campaign does not have. The misses are the point of filing in advance.
