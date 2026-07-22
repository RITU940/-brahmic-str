# PROSPECTIVE PREDICTION — ARCHITECTURE GENERALITY (CRNN rungs), filed BEFORE any CRNN rung trains
**Filed:** 2026-07-22 (IST) · **Owner:** Ritu Baskey · **Companion:** `PREREGISTRATION.md` (Amendment 6,
this commit), `PROSPECTIVE_PREDICTION_GURMUKHI.md` / `_SCALING.md` / `_KHMER.md` (instrument lineage).

**Epistemic state at filing:** all 27 Florence-2 LOSO rungs, the 12-point scaling sweep and the
Qwen2.5-VL baseline are observed. **No CRNN has been trained on any zero-shot LOSO split by anyone.**
The CRNN code (`colab_train_v3.py`, `train_crnn_on_florence.py`) exists from the tokenization-law work
but has never been run on these splits. Verification = git history: this file is committed AND pushed
before the first CRNN rung starts.

## Why this experiment exists
The manuscript's phenomenon claim rests entirely on one backbone (Florence-2). "Is this Florence-2
specific?" is a standing reviewer objection we currently answer only with a limitation sentence.
This test converts that limitation into evidence — in either direction.

## Design (frozen here; test text is never read)
- **Scripts:** tamil, telugu, oriya — the three equal-synth (3240, `is_2x = 0`) scripts spanning the
  observed Florence-2 Rung-B range (9.16 / 19.82 / 25.38). Choosing equal-synth scripts removes the
  2×-synth covariate from this comparison.
- **Rungs:** A and B per script (6 runs), using the **same** `splits_zeroshot_loso_rung{A,B}_{tag}.json`
  files the Florence-2 rungs used — same images, same pivot `gt`, same train/val/test partition.
- **Output space:** the shared abugida pivot, identical strings. The CRNN character vocabulary is built
  from **TRAIN records only**, mirroring `prepare_zeroshot_loso.py:build_vocab`. Test text is not read
  at any point of preparation or training.
- **Model/recipe:** `CRNN_V3` (ResNet + BiLSTM-512, CTC) unmodified from the law-paper codebase.
- **Fixed budget, identical for all six runs (declared now):** img 64×256, batch 32, Adam lr 1e-3,
  weight decay 1e-4, **max 60 epochs, early-stop patience 10**, best checkpoint selected on val.
- **Metric:** the same `metrics.evaluate_corpus` used by every Florence-2 rung (WRR = exact-match word
  recognition rate on the held-out script's **real** test images), on the **unfiltered** test split so
  N matches the Florence-2 rungs exactly (tamil 513, telugu 545, oriya 1044).

## Predictions (the confirmatory content of this file)
1. **Rung A (no target synthetic exposure): WRR < 2.0 on all three scripts.**
2. **Primary — replication of the phenomenon across architectures:** Rung B exceeds Rung A by
   **≥ +2.0 WRR on at least 2 of 3 scripts.**
3. **Magnitude:** the CRNN reaches a *fraction* of the Florence-2 transfer level. Predicted ratio
   CRNN-B / Florence-B ∈ **[0.15, 0.60]** on ≥2 of 3 scripts; point estimate 0.35 ⇒
   **tamil ≈ 3.2, telugu ≈ 6.9, oriya ≈ 8.9 WRR.**
4. **Ordering:** the CRNN Rung-B ranking matches the Florence-2 ranking (tamil < telugu < oriya),
   i.e. Spearman ρ = +1 over the three points.

## Declared caveats (filed now, not after the result)
1. **A null on prediction 2 is ambiguous and we say so in advance.** CRNN_V3 is trained from scratch on
   ~10k images with no large-scale pretraining; if all three Rung-B values sit near zero, that is
   consistent with *either* architecture-dependence of the phenomenon *or* a capacity/pretraining floor.
   We will report the ambiguity rather than resolving it in our favour, and the manuscript's limitation
   sentence stays.
2. **This is a replication test, not a fair-fight benchmark.** No claim that CRNN < Florence-2 says
   anything about architectures in general; the budget and pretraining differ by orders of magnitude.
3. **Three scripts, six runs** — a direction test, not a law. No p-value is claimed from n = 3.

## Scoring rule (fixed now)
Report all four predictions against realized values, hits and misses alike, in the same table style as
the earlier receipts. This is the campaign's **fifth** quantitative prospective call
(kannada ~15 → 15.42 hit; gurmukhi 16.2 → 22.16, 2σ hit / point under-called; scaling slope +11.5 → +2.2
miss; khmer 16.1 pending; this file).

```json
{"filed":"2026-07-22","experiment":"architecture_generality_crnn",
 "scripts":["tamil","telugu","oriya"],"rungs":["A","B"],
 "model":"CRNN_V3 (ResNet+BiLSTM-512, CTC), vocab from TRAIN only, 60 epochs max, patience 10",
 "florence_reference":{"tamil":{"A":0.0,"B":9.16},"telugu":{"A":1.28,"B":19.82},"oriya":{"A":0.77,"B":25.38}},
 "pred_rungA":"WRR < 2.0 on all three",
 "pred_primary":"RungB - RungA >= +2.0 WRR on >= 2 of 3",
 "pred_ratio_band":[0.15,0.60],"pred_ratio_point":0.35,
 "pred_points":{"tamil":3.2,"telugu":6.9,"oriya":8.9},
 "pred_ordering":"tamil < telugu < oriya (Spearman +1)",
 "declared_caveat":"a null is ambiguous between architecture-dependence and a capacity floor; reported as ambiguous"}
```
