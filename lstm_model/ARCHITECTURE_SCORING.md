# ARCHITECTURE-GENERALITY (CRNN) — SCORING RECEIPT
**Scored:** 2026-07-28 (IST) · **Owner:** Ritu Baskey
**Preregistration:** `PROSPECTIVE_PREDICTION_ARCHITECTURE.md` (filed + pushed 2026-07-22, commit
`2f21868`, BEFORE any CRNN rung trained). **Amendment 6** of `PREREGISTRATION.md`.
**Sources of truth:** `result_crnn_zs_rung{A,B}_{tamil,telugu,oriya}.json` (6/6, produced
2026-07-22 by `run_crnn_generality.sh` → `crnn_generality.log`; training logs
`train_crnn_zs_rung*_*.log`). Re-derived by `verify_wacv_numbers.py` (101 macros, 0 mismatch).

## Model (as frozen)
CRNN_V3 = ResNet + BiLSTM-512 + CTC, unmodified from the tokenization-law codebase. Vocabulary
built from TRAIN records only; img 64×256, batch 32, Adam lr 1e-3, wd 1e-4, **60 epochs max,
patience 10**, best checkpoint on val. Same `splits_zeroshot_loso_rung{A,B}_{tag}.json` files and
same `metrics.evaluate_corpus` (unfiltered test split, N matches Florence-2) as the Florence-2
rungs. Test text never read during preparation or training.

## Realized values
| Script | Rung A | Rung B | B−A | CRNN-B / Florence-B | Florence-2 B | CRNN CharAcc(B) |
|--------|-------:|-------:|----:|--------------------:|-------------:|----------------:|
| Tamil  | 0.00   | 0.97   | +0.97 | 0.106 | 9.16  | 25.09 |
| Telugu | 0.00   | 1.10   | +1.10 | 0.056 | 19.82 | 31.79 |
| Oriya  | 0.67   | 4.50   | +3.83 | 0.177 | 25.38 | 26.79 |

Mean ratio CRNN-B / Florence-B = **0.11**. Direction B > A: **3/3** (unanimous).

## Predictions vs realized (all four, hits and misses)
| # | Prediction (frozen) | Realized | Verdict |
|---|---------------------|----------|---------|
| 1 | Rung A WRR < 2.0 on all three | 0.0 / 0.0 / 0.67 | **HIT** |
| 2 | *Primary:* Rung B − Rung A ≥ +2.0 on ≥ 2 of 3 | only Oriya (+3.83); Tamil +0.97, Telugu +1.10 | **MISS** (1/3) |
| 3 | Ratio CRNN-B/Florence-B ∈ [0.15, 0.60] on ≥ 2 of 3 (point 0.35) | only Oriya (0.18); Tamil 0.106, Telugu 0.056 | **MISS** (1/3) |
| 4 | Ordering Tamil < Telugu < Oriya (Spearman +1) | 0.97 < 1.10 < 4.50 | **HIT** |

**Score: 2 hit / 2 miss.** The two structural predictions (Rung-A floor; held-out ordering) hold;
the two effect-size predictions (primary lift; magnitude band) miss.

## Interpretation (per the preregistration's declared caveat — not resolved in our favor)
The prereg's Caveat 1 stated in advance that a near-null on prediction 2 would be **ambiguous**
between (a) architecture-dependence of the pivot mechanism and (b) a CRNN capacity/pretraining
floor, and would be reported as such rather than resolved favorably. The realized result is not a
clean null: the **direction replicates 3/3** and Oriya lands squarely in the predicted band, but
Tamil/Telugu collapse to near-floor and the mean magnitude is ~11% of Florence-2's. Character
accuracy is depressed in step (25–32% vs Florence-2's 39–56%), which is evidence *for* the
capacity-floor reading. Conclusion, as filed: **the direction of the effect is not backbone-specific;
its magnitude may be.** The manuscript's backbone-specificity limitation stands.

## Where reported
`paper_wacv/sec/4_experiments.tex` §"Architecture generality: a preregistered CRNN replication"
(`\label{sec:crnn}`); macros in `paper_wacv/numbers.tex` (`\crnn*`); verified by
`verify_wacv_numbers.py`.

## Campaign prospective-call ledger (updated)
kannada ~15 → 15.42 (hit) · gurmukhi 16.2 → 22.16 (2σ hit, point under-called) · scaling slope
+11.5 → +2.2 (magnitude miss) · **architecture CRNN → 2 hit / 2 miss (this file)** · khmer 16.1
(pending, Aug 8 drop-dead).
