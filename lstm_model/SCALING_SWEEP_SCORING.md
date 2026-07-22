# SCALING SWEEP — 12/12 scoring receipt (against the frozen prereg)

**Scored:** 2026-07-21, after the sweep completed 12/12 (`scaling_sweep.log`,
COMPLETE Tue 21 Jul 08:41 IST). **Predictions frozen & pushed before any data:**
`PROSPECTIVE_PREDICTION_SCALING.md` + PREREGISTRATION Amendment 5 (commit 47bc407).
**Instrument (unedited):** `analyze_scaling_law.py`; raw output archived in
`scaling_sweep_scoring_raw.txt`. Result cells = `result_zs_scale{B}_{script}.json`.

## Observed WRR (%) on real test images

| Script | 810 | 1620 | [3240 anchor] | 6480 | 12960 |
|---|---|---|---|---|---|
| malayalam | 4.39 | 7.31 | 9.69 | 11.70 | 14.08 |
| kannada | 11.67 | 16.11 | 15.42 | 19.17 | 19.86 |
| telugu | 13.76 | 16.33 | 19.82 | 21.10 | 23.30 |

(3240 = external LOSO Rung-B anchor, not part of the fit.)

## Verdicts against the fixed scoring rules

**Rule 2 — the faithful doubling step (3240→6480), predicted +11.5/doubling: MISS.**
Observed gains +2.01 / +3.75 / +1.28, mean **+2.35**, outside ±2·RMSE [+3.14, +19.84].
The +11.48/doubling slope (estimated from one 2× jump on bengali+devanagari) does **not**
transport. Reported as a miss per the standing protocol.

**Confirmatory test (Amendment 4a) — swept common slope vs preregistered c=11.48: EXCLUDED.**
Common slope **c = +2.235 WRR/doubling** (95% CI [+1.89, +2.58]); preregistered 11.48 is
5.14× outside the CI (t = −61.6). The *magnitude* prediction is falsified.

**BUT the scaling FORM is confirmed — this is the positive, publishable result.**
WRR is log-linear in synthetic budget per script (R² = 0.997 / 0.906 / 0.999), and the slope
is **script-invariant** (separate-vs-common-slope F(2,6)=0.92, p=0.45 — no evidence slopes
differ). A clean, prospectively-tested law: *each doubling of synthetic exposure buys
~+2.2 WRR, the same for every script tested.*

**Rule 1 — low-end near-collapse: FALSIFIED (transfer is concave in log-budget).**
Linear form predicted collapse toward the Rung-A baseline at 810/1620. Observed: kannada
810 = 11.67 (> pred+2·RMSE 11.14) and telugu 810 = 13.76 (≥ 9.6 rule threshold) both land
**well above** the collapse prediction; malayalam 810 = 4.39 is consistent with it. Net:
a little synthetic exposure already yields double-digit WRR on 2 of 3 scripts → **a little
synthetic exposure goes a long way.** Reported as filed.

**Rule 3 — top step (6480→12960, near-fixed lexicon) vs faithful step: mixed.**
malayalam +2.38 vs +2.01 (comparable → image quantity contributes), telugu +2.20 vs +1.28
(comparable), kannada +0.69 vs +3.75 (**smaller → saturation / lexicon exhaustion**). Both
readings reported per rule.

## Caveat that constrains the paper's framing (from the instrument itself)

The coverage-offset model (`WRR = a + b·tok_cov + c·log2 budget`) fits the 3 swept scripts
at R²=0.98, **but does not generalize**: on the 6 never-fitted scripts the out-of-sample RMSE
is 7.29 (vs 0.77 in-sample) and coverage↔WRR across those 6 alone is r = +0.04, p = 0.94.
The coverage axis has only 3 distinct values, so the offset term is optimistic by construction.
**Do not build the paper's claim on typological coverage as the offset driver.** The *slope*
(within-script replication) is unaffected and is the trustworthy result.

## One-line summary
Magnitude prediction missed (5×), but the pre-registered **form** — WRR log-linear in
synthetic budget with a **script-invariant ~+2.2 WRR/doubling slope** — is confirmed with
tight CIs; low-end collapse is falsified (concave transfer); top-end shows onset of saturation.
