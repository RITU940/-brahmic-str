# PROSPECTIVE PREDICTION — KHMER RUNG B (filed BEFORE any Khmer model trains)
**Filed:** 2026-07-21 (IST) · **Owner:** Ritu Baskey · **Companion:** `PREREGISTRATION.md`
(Amendment 4b), `PROSPECTIVE_PREDICTION_GURMUKHI.md` (instrument lineage).

**Epistemic state at filing:** all 27 LOSO rungs and the 12-point scaling sweep are observed;
**no Khmer recognizer has been trained by anyone.** KhmerST~\cite{khmerst} labels have been used
ONLY to compute the result-blind coverage input below (labels, not model outputs). Khmer sits
outside the ISCII-aligned Brahmic blocks, so the pivot is extended to it by a hand-curated,
committed correspondence (`khmer_pivot_map.py`, this commit). Verification = git history: this
file and the map are committed AND pushed before the training run.

## Model (frozen; identical instrument used for the Gurmukhi call)
Two-factor least squares on the nine observed Rung-B points (9-point checkpoint):
`WRR = −35.79 + 0.583·tok_cov + 11.48·is_2x_synth`  (RMSE 4.18).
Khmer inputs (result-blind): **tok_cov = 89.02** (token coverage of KhmerST label
grapheme-clusters, mapped to pivot via `khmer_pivot_map.py` v1, by the union vocabulary of the
nine benchmark scripts; codepoint map-rate 94.48%), **synth = 3240 (1×, is_2x = 0)**.
N(test) = to be fixed by the KhmerST recognition split before training.

## Prediction (confirmatory content of this file)
- **Point: 16.1 WRR.**  **±1·RMSE band: [11.9, 20.3].  ±2·RMSE band: [7.8, 24.5].**
- Implied rank among the ten scripts: mid-pack (between Gujarati 15.86 and Telugu 19.82).
- Rung A (no synthetic Khmer, nine benchmark scripts only): **WRR < 5** (coverage-tracking).

## Declared caveats (disclosed now, not after the result)
1. **We expect this point to be fragile, and say so in advance.** The same coverage term does
   *not* generalize across the six held-out Brahmic scripts (out-of-sample Pearson
   $r=+0.03$, $p=0.94$; `SCALING_SWEEP_SCORING.md`). A low-side miss would be fully consistent
   with that finding and with Khmer's out-of-block script, coeng-based stacking, and thin font
   supply (two system faces). We file the instrument's number regardless and score it as-is.
2. **The pivot extension is a best-effort v1.** 94.48% of Khmer codepoints map; a minority of
   register diacritics and diphthong vowels are approximated or dropped (flagged in the map),
   which the coverage statistic counts honestly as misses.
3. **Coverage is computed over the full KhmerST label set** (result-blind), pending the frozen
   recognition test split; if the split changes tok_cov materially we re-file before training and
   note it.

## Scoring rule (fixed now)
Report realized Rung-B WRR vs the point estimate and whether it falls inside the ±1·RMSE and
±2·RMSE bands. A miss is reported exactly like a hit. This is the campaign's fourth quantitative
prospective call (kannada ~15 → 15.42 hit; gurmukhi 16.2, 2σ hit / point under-called; scaling
slope +11.5 → +2.2 miss; this file).

```json
{"filed":"2026-07-21","script":"khmer","rung":"B",
 "model":"WRR = -35.79 + 0.583*tok_cov + 11.48*is_2x_synth (RMSE 4.18; 9-point fit)",
 "inputs":{"tok_cov":89.02,"synth":3240,"is_2x":0,"pivot_map":"khmer_pivot_map.py v1",
           "codepoint_map_rate":94.48,"N_test":"TBD (KhmerST split)"},
 "point":16.1,"band_1rmse":[11.9,20.3],"band_2rmse":[7.8,24.5],
 "rungA_prediction":"WRR < 5",
 "declared_caveat":"coverage does NOT generalize OOS (r=+0.03); low-side miss expected, filed anyway"}
```
