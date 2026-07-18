# WACV 2027 STRATEGY — the zero-shot cross-script paper (Paper B)
**Researched:** 2026-07-17 (live web research this day) · **Owner:** Ritu Baskey · First author = Ritu.
**Companion docs:** `PUBLICATION_STRATEGY.md` (venue calendar), `COMPETITIVE_POSITIONING_AND_LITERATURE.md`
(landscape as of 06-29; §2 below updates it), `ZEROSHOT_LOSO_LIVE_LOG.md` (results),
`PREREGISTRATION.md` (incl. Amendment 4), `RELATED_WORK_DRAFT.md` (prose).
**Governing rule:** strong results, honestly — never fabricate, never inflate.

---

## 0. VERDICT (the honest answer to "is this worth a top-tier shot?")

**Yes — this is a real WACV-grade paper, and the niche is still open as of 2026-07-17.**
The completed campaign gives us something rare: (a) a *phenomenon* — zero-real-image transfer
to nine unseen Brahmic scripts, 9.2–30.1 WRR from ~0 baselines; (b) a *mechanism test* — the
grapheme pivot is necessary (BPE ablation 2–4× worse); (c) *predictability* — committed-and-pushed
prospective predictions with scored hits (kannada ~15 → 15.42; gurmukhi ±2·RMSE band hit), a
falsified preregistered primary (fertility, −0.80) reported as-is, and a surviving two-factor
account (synth quantity + pivot coverage). No paper in the literature combines these.
"Groundbreaking" is not a property we can declare — it is a property reviewers grant — but the
*prospective-prediction receipts* are our genuinely unusual asset: almost no CV paper can show
git-committed forecasts filed before the runs. That, plus falsification honesty, is the
top-tier differentiator. The gap between "solid WACV paper" and "strong accept" is closed by
finishing three preregistered experiments (§4) and by airtight writing (§5–6), because **Round 2
has NO rebuttal** — the paper must survive first read.

Ceiling check (honest): WACV ≈ 38% acceptance — realistic primary target. If the synth-scaling
dose-response comes out clean AND Khmer transfers, the same manuscript + N-expansion becomes a
credible CVPR 2027 (Nov 15) upgrade candidate; decide Oct 9 when WACV decisions land (no dual
submission).

---

## 1. VENUE INTEL — WACV 2027 (verified on wacv.thecvf.com, 2026-07-17)

- **Round 2: enroll Aug 21 · paper Aug 28 · supplementary Aug 30 · decisions Oct 9** (all AoE).
  Camera-ready Nov 2. Conference Jan 2027.
- **⚠ Round 2 has no rebuttal** — reviews and final decisions arrive together Oct 9. Every
  anticipated objection must be answered inside the 8 pages (or supp). This changes how we write:
  the §Limitations and the 2312.10806 rebuttal paragraph are load-bearing, not optional.
- **Three tracks with different review criteria** (new structure):
  - **Algorithms** — "standard conference criteria: algorithmic novelty and quantified evaluation
    against current, alternative approaches."
  - **Applications** — systems-level innovation, novelty of domain, comparative assessment.
  - **Evaluation & Datasets** — benchmark analysis, dataset development, new evaluation
    protocols, systematic comparisons.
- **Track choice: Algorithms.** Our headline is a method + a predictive finding (pivot space →
  transfer; two-factor model → who transfers), quantified against alternatives (BPE ablation,
  raw Florence-2, frontier VLM, supervised ceilings). E&D is the fallback framing (zero-real-image
  protocol + prediction receipts as "evaluation science") but Algorithms matches the contribution
  and carries more prestige weight. Decide finally at enrollment; nothing in the paper changes.
- **Format:** 8 pages incl. figures/tables + unlimited references; official author-kit template
  mandatory (desk-reject otherwise); double-blind ("beyond reasonable doubt" anonymity);
  supp ≤ 200 MB, optional-to-read.
- **arXiv is explicitly allowed** before/during review (not a prior publication under their dual
  submission policy) → the planned arXiv preprint is compatible; post it once Bbpe + VLM baseline
  are in (late July), so the public version is the strong version.
- Acceptance ≈ 37.8% (2026, 2458 submissions). Round-2-specific rates unpublished.
- WACV has a live OCR/document community (VisionDocs workshop at WACV 2026; a Khmer scene-text
  cross-lingual benchmark was a WACV 2026 *workshop* paper — topical fit confirmed; our job is to
  clear the *main-conference* bar, which the prediction receipts + ablations are for).

---

## 2. FRESH LITERATURE SCAN (2026-07-17) — five new items, NO preemption

Verified additions to `COMPETITIVE_POSITIONING_AND_LITERATURE.md` §2 (all checked at source today):

| Work | id / venue | What it is | Use |
|---|---|---|---|
| **Task-Analogies zero-shot HTR** | arXiv 2604.09713 (Apr 2026) | Zero-*real*-data synthetic-to-real HTR via task arithmetic — but **five Latin-script languages only**; unseen *script* explicitly left to future work (their fn. 1) | Closest new neighbor. Cite prominently: independent confirmation the zero-real-data regime matters; we cross the script boundary they don't. |
| **UnionST** | arXiv 2602.06450, **CVPR 2026** | Synthetic-data engine for STR; quality > quantity claims; Latin-centric | Cite in synth-data related work. Their quality-vs-quantity result is a *within-script, real-data-available* regime; ours is cross-script zero-real. No conflict; sharpens our framing of the synth-budget factor. |
| **GlotOCR Bench** | arXiv 2604.12978 (Apr 2026) | 2026 frontier VLMs (incl. Qwen3-VL, Gemini, GPT-4) still fail beyond a handful of Unicode scripts; document-focused | **Gift for motivation** ("the script gap is real in 2026") + calibrates Amendment-4c expectations. Cite in intro ¶1. |
| **Manchu VLM OCR** | arXiv 2507.06761 (Jul 2025) | Qwen2.5-VL/LLaMA-3.2 fine-tuned on 60k synthetic Manchu words → 93% real word acc | Mechanism support: synthetic-only VLM fine-tuning works for one low-resource script. Single script, no transfer, no prediction, documents. Cite + differentiate. |
| **Universal Khmer TR** | arXiv 2603.00702 (Feb 2026) | Cross-*modality* (print/hand/scene) Khmer recognition | Cite in the Khmer section (4b); orthogonal (modality, not script transfer). KhmerST remains our eval target. |

Also rechecked today: the 06-29 threat map is unchanged — no new work does zero-real-image
transfer to an unseen script, none predicts cross-script transfer from script properties.
**arXiv 2312.10806 (data-size > typology) remains the #1 reviewer threat**; our rebuttal is now
*stronger* than in June because our own data agrees with them on the data-size axis (synth
quantity dominates) while isolating a structural coverage effect at fixed budget — we don't
fight their finding, we *decompose* it. Fertility-refutation also now aligns us with the
"Beyond Fertility" line rather than against it. Honest science turned both threats into allies.

---

## 3. WHAT THE PAPER IS (the 8-page story)

**Title direction:** "Reading Unseen Scripts: Predictable Zero-Real-Image Transfer for Brahmic
Scene Text" (final wording later; must contain *unseen script* + *zero real images* + *predictable*).

**Claims (each with its evidence artifact):**
1. **Phenomenon.** A Florence-2-class VLM fine-tuned in a shared grapheme pivot space on 7–8
   source scripts + synthetic-only target images reads *real* scene text in a script it has never
   seen a real image of: 9.16–30.09 WRR across all nine BSTD scripts (baselines 0–5.5).
   [18 rungs, result JSONs, verified metric = BSTD protocol]
2. **Mechanism.** The pivot is necessary: stock-BPE ablation collapses transfer 2–4× at equal
   CharAcc on telugu — the pivot converts character ability into word reads. [Bbpe ×9 — 2 done,
   7 to finish, §4.1]
3. **Predictability.** Transfer is forecastable from two measurable, training-free quantities:
   synthetic-exposure budget and pivot-space token coverage. Prospective receipts: kannada "~15"
   → 15.42; gurmukhi 16.2 [8.2–24.1] → 22.16 (outer band hit, miss reported); preregistered
   primary (fertility) refuted at −0.80 and reported. Dose-response sweep §4.2 turns the
   2×-synth covariate into a designed result. [047c30c chain + Amendment 4a]
4. **Boundary.** Out-of-benchmark, out-of-India test on Khmer (different Unicode block, coeng
   stacking) with a prediction filed before the run. Success = generality; near-zero = honest
   boundary of transfer. Either way it's a result. [§4.3]
5. **Context.** Frontier VLMs (Qwen2.5-VL) score near the floor on the same nine test sets under
   the same metric — the gap our recipe closes is real in 2026. [§4.4 + GlotOCR cite]

**What we do NOT claim:** per-script SOTA (supervised PARSeq ≈73% is a different axis — real
labeled target data vs our zero); that fertility predicts transfer (we refuted our own prereg
primary and say so); that the law (Part ①) is the headline — it appears as the motivation for
the grapheme pivot, compressed to ~½ page + pointer to the journal-length treatment.

---

## 4. EXPERIMENT PRIORITY STACK (GPU is the binding resource; A5000 ≈ 6–8 h/rung post-fix)

| # | Experiment | Cost | Value for acceptance | Status / call |
|---|---|---|---|---|
| 4.1 | **Finish Bbpe ×7** (kannada re-runs from scratch; sentinel guard verified) | ~35 GPU-h (~1.5–2 d) | **Mandatory.** A 2/9 ablation is a standing reviewer objection; 9/9 makes claim 2 airtight. Preregistered (Amendment 3). | **Resume immediately.** |
| 4.2 | **Synth-scaling sweep** (mal/kan/tel × {810,1620,3240*,6480,12960}) = 12 new runs | ~12 × 7 h ≈ 3.5–4 d GPU (+ CPU rendering for 6480/12960 — start now, free) | **Highest scientific value.** Converts the dominant factor from covariate to designed dose-response; tests the frozen c≈+12.3/doubling prediction. This is what lifts "nice application" → "predictive science." Preregistered (4a). | Queue right after Bbpe. |
| 4.3 | **Khmer out-of-benchmark** (engineering: U+17D2 coeng in segmenter, KhmerST word crops; then A+B rungs) | ~3 d engineering (CPU, parallel) + ~14 GPU-h | High wow + generality; prediction receipt #4 (file `PROSPECTIVE_PREDICTION_KHMER.md` from the frozen 9-pt model BEFORE training). Timeboxed: if the segmenter/crops aren't clean by Aug 8, ship without it (paper stands on 1–3+5). | Engineering starts now, CPU-only. |
| 4.4 | **Frontier-VLM baseline** (Qwen2.5-VL-7B zero-shot on 9 test sets, fixed prompt, our metric) | Inference only, ~hours in GPU gaps | Cheap and expected by a 2026 WACV audience ("why not just prompt a VLM?"). Preregistered (4c) with the upper-bound caveat. | Slot into gaps between rungs. |
| 4.5 | Rung C few-shot slope | ~9 short runs | Nice-to-have; supp material at best for 8pp. | Only if GPU idle after 4.1–4.4. |
| 4.6 | N-expansion for the law | large | Journal-scope (Pattern Recognition version), not WACV. | Defer. |

Rough GPU math: 35 h (4.1) + ~90 h (4.2) + ~14 h (4.3) + slack ≈ **6–8 GPU-days** → done by
~Aug 5–10 with contention margin. Writing overlaps throughout. Feasible.

**Hard dates:** internal full-draft freeze **Aug 14** → verify-numbers script + polish pass →
enroll **Aug 21** → submit **Aug 28**. arXiv once 4.1+4.4 land (~Jul 27–31), per §1.

---

## 5. HOW WE LOSE, AND THE COUNTER FOR EACH (write these INTO the paper — no rebuttal round)

1. **"Absolute WRR is low (9–30%)."** → Different-axis framing sentence BEFORE every table
   (zero real target images vs supervised 73% with real data); Rung A ≈ 0 and raw Florence-2 = 0
   anchors; VLM baseline near floor; CharAcc 34–58% shows the signal is real.
2. **"2312.10806 showed data size beats typology."** → §3-of-COMPETITIVE doc rebuttal, upgraded:
   we *confirm* the quantity effect on the target side (dose-response, designed) and *decompose*
   it from a structural coverage effect at fixed budget. Their finding is our §5.1 opening cite,
   not our enemy.
3. **"Fertility was your preregistered primary and it failed."** → Feature, not bug: falsification
   reported with receipts; the field's fertility-skeptic line (2510.09947, TokSuite) independently
   agrees; the surviving account was itself tested prospectively (gurmukhi). This paragraph is
   the paper's honesty signature — reviewers reward it or at minimum can't attack it.
4. **"Only 2/9 BPE ablations"** → closed by 4.1. **"Is it Florence-2-specific?"** → honest
   limitation + the law's 12-model CRNN evidence suggests architecture-generality; explicitly
   scoped as future work. **"Synthetic renderer bias?"** → fc-match multi-font + blank
   verification + released pipeline; UnionST cited as the quality frontier.
5. **"Bengali test contamination?"** (ritu1 doc-level split issue) → this paper evaluates on
   BSTD only for the nine transfer scripts; the ritu1 half is not part of Paper B's eval. One
   clarifying sentence in supp. [[bengali-split-ritu1-leak]]
6. **Anonymity trap:** the manuscript cites our own git receipts (047c30c etc.). For double-blind:
   cite as "prospective predictions were filed in a version-controlled archive before each run;
   hashes in supp / anonymized repo (e.g. Anonymous GitHub)". NO repo URL with the name `ritu1`,
   no owner name in receipts shown to reviewers. Camera-ready de-anonymizes.
7. **No AI wording anywhere** per standing rule; verify with grep before every commit of paper
   text. [[no-ai-attribution]]

---

## 6. MANUSCRIPT PLAN (8 pp, WACV author kit)

- §1 Intro (1.0 pp): script gap in 2026 (GlotOCR) → zero-real-image question → contributions 1–5.
  Fig 1 = method/pivot diagram (adapt `fig0_method`/prof-abstract pivot figure).
- §2 Related (0.75 pp): compress `RELATED_WORK_DRAFT.md` 7 subsections → 5 paragraphs;
  add the five §2 cites; 2312.10806 gets its own paragraph.
- §3 Method (1.25 pp): pivot space construction, synth pipeline, rungs protocol, descriptors.
- §4 Experiments (3.0 pp): Fig 2 = 9-script A/B/Bbpe bars; Fig 3 = dose-response curves (4.2);
  Fig 4 = prediction-receipts timeline (unique figure: filed-vs-realized with bands — nobody
  else has this figure); Tab 1 = main 27-rung table; Tab 2 = VLM baseline + supervised anchors;
  Khmer subsection (4.3) if it lands.
- §5 Analysis (1.0 pp): two-factor model, fertility refutation, horse-race, equal-synth
  decomposition.
- §6 Limitations + §7 Conclusion (0.75 pp): synthetic→real gap, single architecture, N=9,
  Gujarati residual, honest scope.
- Supp: prereg/amendment chain, all per-script details, extra figures, verify-script output.
- **`verify_wacv_numbers.py`** from day one (mirror the IJDAR one) — every number in the tex
  re-derived from result JSONs.

---

## 7. SOURCES (verified 2026-07-17)
- WACV 2027 CFP/tracks: https://wacv.thecvf.com/Conferences/2027/CallForPapers
- WACV 2027 dates: https://wacv.thecvf.com/Conferences/2027/Dates
- WACV 2027 author guide (8pp, template, blind, arXiv-OK): https://wacv.thecvf.com/Conferences/2027/AuthorGuides
- WACV 2026 stats 37.8%: https://papercopilot.com/statistics/wacv-statistics/
- Task-Analogies HTR: https://arxiv.org/abs/2604.09713 · UnionST: https://arxiv.org/abs/2602.06450
- GlotOCR: https://arxiv.org/abs/2604.12978 · Manchu: https://arxiv.org/abs/2507.06761
- Universal Khmer: https://arxiv.org/abs/2603.00702 · KhmerST: https://arxiv.org/abs/2410.18277
- WACV2026W Khmer spotting: https://openaccess.thecvf.com/content/WACV2026W/VisionDocs/papers/Nom_Cross-Lingual_Transfer_for_Complex_Scripts_A_Benchmark_on_End-to-End_Khmer_WACVW_2026_paper.pdf
