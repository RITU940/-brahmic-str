# PROSPECTIVE PREDICTIONS — H3 (filed BEFORE the results exist)
**Filed:** 2026-07-02 (IST) · **Owner:** Ritu Baskey · **Companion:** `PREREGISTRATION.md` §8
Amendment 1 (same commit).

**What this file is:** falsifiable, timestamped predictions of the zero-shot LOSO Rung-B
outcomes for the **7 scripts whose results do not exist yet**. Verification = the git history
of this repository (this file is committed and pushed before the kannada Rung-B model has even
finished training; any later edit would be visible in `git log -p`). The manuscript will cite
the commit hash.

---

## 1. Epistemic state at filing (what is and is not known)

**Known Rung-B results (2 of 9):**
| script | fertility | tok-cov% | Rung B WRR |
|---|---|---|---|
| tamil  | 6.07 | 88.8 | 9.16 |
| telugu | 5.42 | 92.3 | 19.82 |

**Unknown (7 of 9):** kannada, malayalam, oriya, gujarati, bengali, devanagari, gurmukhi.
(Kannada Rung B crashed at epoch 6 in the 2026-06-30 power outage and was **never evaluated**;
no human or process has observed any of these 7 numbers.)

**Descriptors used below** were all computed result-blind before any Rung-B result existed:
fertility (`script_descriptors.json`, per-language `bpe_fertility_neutral`; script-level value =
mean over the script's BSTD languages: bengali = mean(bengali 3.92, assamese 3.94) = 3.93;
devanagari = mean(hindi 3.40, marathi 3.25) = 3.33; note `odia` in descriptors = `oriya` LOSO tag),
and token coverage (frozen 2026-06-25 in `RESEARCH_STATUS_AND_PATH.md` §5).

---

## 2. PREDICTION P1 — pre-registered primary (fertility, H3 as frozen 2026-06-18)
H3 as pre-registered says **higher fertility ⇒ better transfer**. Predicted Rung-B ranking of
the 7 unknown scripts (best → worst):

> **malayalam (6.02) > kannada (5.29) > oriya (5.17) > gujarati (4.90) > bengali (3.93) >
> devanagari (3.33) > gurmukhi (3.17)**

## 3. PREDICTION P2 — declared directional bet (token coverage)
Mechanistic rationale: transfer requires the shared abugida space to *cover* the held-out
script's grapheme inventory; the 2 known points are consistent with this and not with P1.
Predicted ranking (best → worst):

> **gujarati (97.3) > devanagari (94.3) > oriya (92.6) > kannada (90.8) > gurmukhi (90.7) >
> bengali (87.3) > malayalam (79.2)**

**Our stated expectation at filing: P2 outperforms P1** (this is the bet this file exists to
make checkable). Spearman ρ between the two predicted rankings is −0.25, and they are fully
opposed at the extremes: **malayalam is P1's best and P2's worst** — malayalam alone is close
to decisive between the two hypotheses.

## 4. Exploratory point estimates (declared crude — ranks above are the confirmatory content)
Two-point linear extrapolation on coverage, WRR ≈ 9.16 + 3.05·(tok-cov − 88.8), expect large
errors:

| script | predicted Rung-B WRR (±wide) |
|---|---|
| gujarati | ~35 |
| devanagari | ~26 |
| oriya | ~21 |
| kannada | ~15 |
| gurmukhi | ~15 |
| bengali | ~5 |
| malayalam | ~0–4 |

Also predicted: **every Rung A stays low** (WRR < ~5; it is the no-exposure baseline).

**Known covariates disclosed in advance** (will be reported alongside, not hidden):
devanagari & bengali have 2× synth exposure (6480 vs 3240 images — two BSTD languages each),
which may lift them above the coverage line; their test sets are also much larger and possibly
harder (N = 6042 / 2873 vs ~500–1000). Gurmukhi–kannada coverage differs by only 0.1 points —
their relative order is a coin flip under P2 and we do not stake anything on it.

## 5. Machine-readable form

```json
{
  "filed": "2026-07-02",
  "known_at_filing": {"tamil": 9.16, "telugu": 19.82},
  "unknown_at_filing": ["kannada","malayalam","oriya","gujarati","bengali","devanagari","gurmukhi"],
  "P1_fertility_rank_best_to_worst": ["malayalam","kannada","oriya","gujarati","bengali","devanagari","gurmukhi"],
  "P2_coverage_rank_best_to_worst": ["gujarati","devanagari","oriya","kannada","gurmukhi","bengali","malayalam"],
  "declared_bet": "P2 outperforms P1 (Spearman with realized Rung-B WRR over the 7 scripts)",
  "exploratory_point_estimates_rungB_WRR": {"gujarati": 35, "devanagari": 26, "oriya": 21,
    "kannada": 15, "gurmukhi": 15, "bengali": 5, "malayalam": 2},
  "rungA_prediction": "WRR < 5 for all scripts"
}
```

## 6. Scoring rule (fixed now)
When all 7 results exist: report Spearman ρ(P1, realized) and ρ(P2, realized) over the 7
scripts filed here (the 2 known-at-filing scripts are excluded from the *prospective* score and
reported separately). Whichever loses, the outcome is reported in full per `PREREGISTRATION.md`
§6 — including if both lose.
