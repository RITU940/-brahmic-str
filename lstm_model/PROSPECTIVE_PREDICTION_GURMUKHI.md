# PROSPECTIVE PREDICTION — GURMUKHI RUNG B (filed BEFORE the rung trains)
**Filed:** 2026-07-09 13:42 IST · **Owner:** Ritu Baskey · **Companion:** `PROSPECTIVE_PREDICTIONS_H3.md`
(047c30c), `PREREGISTRATION.md` §8.

**Epistemic state at filing:** 8 of 9 Rung-B results observed (tamil 9.16, telugu 19.82, kannada
15.42, malayalam 9.69, oriya 25.38, gujarati 15.86, bengali 27.08, devanagari 30.09). Gurmukhi
Rung A is TRAINING at filing time (started 13:33 IST); Rung B has NOT started; no gurmukhi
transfer number has been observed by anyone. Verification = git history of this repository.

## Model (fixed here, before the result)
Two-factor least squares on the 8 observed Rung-B points:
`WRR = −35.47 + 0.570·tok_cov + 12.34·is_2x_synth`  (RMSE 3.97).
Gurmukhi inputs (frozen 2026-06-25, result-blind): tok_cov = 90.7, synth = 3240 (1×), N(test) = 2879.

## Prediction (confirmatory content of this file)
- **Point: 16.2 WRR.**  **±1·RMSE band: [12.2, 20.2].  ±2·RMSE band: [8.2, 24.1].**
- Convergent alternates (declared for robustness, same point within noise): equal-synth-only
  linear fit 16.2; equal-synth median 15.6; nearest-coverage-neighbor (kannada, 90.8) 15.4.
- Implied rank among the nine: 5th–6th (between telugu 19.82 and kannada 15.42).
- Rung A (no-exposure baseline): **WRR < 5** (coverage-tracking, likely 2–4).

## Scoring rule (fixed now)
Report realized Rung-B WRR vs the point estimate and whether it falls inside the ±1·RMSE and
±2·RMSE bands. A miss is reported exactly like a hit. This is the third quantitative prospective
call of the campaign (kannada "~15" → 15.42; devanagari "~25+" committed pre-result but pushed
post-result — scored as weakly prospective; this file is committed AND pushed pre-training).

```json
{"filed":"2026-07-09","script":"gurmukhi","rung":"B",
 "model":"WRR = -35.47 + 0.570*tok_cov + 12.34*is_2x_synth (RMSE 3.97; fit on 8 observed points)",
 "inputs":{"tok_cov":90.7,"synth":3240,"N_test":2879},
 "point":16.2,"band_1rmse":[12.2,20.2],"band_2rmse":[8.2,24.1],
 "rungA_prediction":"WRR < 5"}
```

---
## SCORED — 2026-07-09 22:41 IST (result observed after this file was pushed at 13:44)
**Realized: 22.16 WRR** (CharAcc 44.1, CER 55.9, N=2879).
- vs point 16.2: **miss by +5.96 (≈1.5·RMSE), in the under-prediction direction** (the script
  transferred BETTER than called).
- ±1·RMSE band [12.2, 20.2]: **OUTSIDE** (above by 1.96).
- ±2·RMSE band [8.2, 24.1]: **INSIDE.**
- Rung A < 5: **HIT** (0.59).
Scored per the rule above; reported as-is. Model refit at this declared checkpoint (9 points):
WRR = -35.79 + 0.583·tok_cov + 11.48·is_2x_synth, RMSE 4.18 — instrument for the Khmer prediction.
