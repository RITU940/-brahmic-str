# WACV E&D TRACK FLIP — prepared draft (pending advisor sign-off)
**Prepared:** 2026-07-28 · **Owner:** Ritu Baskey · **Decision owner:** advisor, by **Aug 21**
(enrollment; track is reversible until then, frozen after). Source: `WACV2027_SUBMISSION_COMPLIANCE.md` §4.

## What this is
The paper currently submits to **Algorithms** (`\usepackage[review,algorithms]{wacv}`). Under that
rubric, "the grapheme pivot is precedented; the contribution is empirical" is a legal reject with no
AC backstop and **no rebuttal** in Round 2. The **Evaluations & Datasets** track (new in 2027) is a
better rubric for this paper: its call explicitly solicits new evaluation protocols, prospective
methodologies, negative results, audits, and released resources — which are our rarest assets and
are *invisible* to the Algorithms rubric. This file stages the flip so it can be applied the moment
the advisor signs off, without a scramble.

**Nothing here is wired into `main.tex` yet.** The drafts are parallel files; the live Algorithms
version still compiles unchanged.

## The change is framing, not results. Every number is identical; only emphasis moves.
- **Headline reframed:** from "a method (pivot) that reads unseen scripts" → "an **evaluation
  protocol** + a **predictive instrument** for the zero-real-image setting; the pivot is the
  mechanism we measure, not the headline." The pivot demonstration stays, demoted to evidence.
- **Contribution list reordered:** (1) the zero-real-image LOSO protocol, (2) prospective prediction
  as an evaluation practice scored both ways, (3) controlled attribution (BPE ablation + CRNN), (4)
  system analysis on the benchmark (VLM / off-the-shelf OCR / supervised ceiling) + released resources.

## E&D call → our contribution (verbatim CFP bullets, from the compliance doc §4)
| E&D solicited contribution | Ours |
|---|---|
| "Propose new evaluation protocols, practices, or methodologies" | zero-real-image LOSO rung protocol; prospective-prediction (commit-before-run) methodology |
| "Present negative results, critical analyses" | fertility falsified (−0.80); scaling magnitude missed 5.1×; coverage fails OOS (r=+0.03); CRNN attenuated replication |
| "Rigorous reproduction, auditing, stress-testing" | 101-macro `verify_wacv_numbers.py`; seed-controlled bootstrap CIs; BPE ablation as controlled counterfactual; `verify_bib.py` 28/28 |
| "Systematic analyses of systems on novel datasets" | 9 scripts × rungs + scaling sweep + Khmer out-of-benchmark + CRNN + VLM/OCR/PARSeq anchors |
| "Analysis of strengths, limitations, failure modes of existing systems/benchmarks" | frontier-VLM + off-the-shelf-OCR + supervised-specialist on BSTD under one metric |

## Files prepared
- `sec/0_abstract_ed.tex` — E&D-framed abstract (existing macros only; `%TODO(anchors)` marks the fold-in).
- `sec/1_intro_ed.tex` — E&D-framed intro + reordered 4-item contribution list (`%TODO(khmer)`,
  `%TODO(anchors)`).
Both reference only macros that exist in `numbers.tex` today, so they compile now; the Khmer and
anchor numbers replace the `%TODO` markers as those runs land.

## Apply procedure (on sign-off — ~10 min, reversible)
1. `main.tex` line 5: `\usepackage[review,algorithms]{wacv}` → `\usepackage[review,datasets]{wacv}`.
2. `cp sec/0_abstract.tex sec/0_abstract_algo.bak && cp sec/0_abstract_ed.tex sec/0_abstract.tex`
   (same for `1_intro`). Keep the `.bak` so the flip is one command to revert.
3. Set the OpenReview track to **Evaluations & Datasets** to match the template (they must agree).
4. Rebuild; `verify_wacv_numbers.py` must still pass; re-check ≤8 pages excl. references.
5. Conclusion (`sec/6_conclusion.tex`) — audit its one framing sentence; body sections need no change
   (they are already evidence-first).

## Revert (if advisor chooses Algorithms)
`mv sec/0_abstract_algo.bak sec/0_abstract.tex` (same for intro); restore the package line. Delete the
`_ed` drafts or leave them dormant. No results change either way.

## Risks the flip introduces, and how they're handled
- **E&D reviewers expect a released resource.** Mitigated: we commit to releasing the pipeline,
  splits, verify script, and prereg chain (all exist) via the anonymized artifact mirror (separate
  task) — and the abstract/intro now say so explicitly.
- **"Where is the dataset?"** We are not primarily a *dataset* paper; the contribution is a
  *protocol + instrument + audit*, which the E&D call solicits directly. The framing says this in
  the first paragraph so a reviewer is not left hunting for a dataset headline.
- **Mechanism must read as evidence, not headline method.** Handled by the reorder: the pivot appears
  as "the object the evaluation measures," and the ablation/CRNN as attribution, not as a proposed
  algorithm.

## Status
DRAFT, ready to apply. **Blocked only on the advisor's track sign-off (Aug 21 hard freeze).**
