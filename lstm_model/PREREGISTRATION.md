# Pre-Registration — Tokenization-Granularity Law + Zero-Shot Cross-Script STR

**Frozen:** 2026-06-18, **before** any `checkpoints_law_*` / `fusion_law_*` results
existed (the Dravidian training run launched the same day; no law-branch WRRs had been
observed at freeze time). This document fixes the hypotheses, the independent variables,
the dependent variables, and the statistical tests *in advance*. Deviations must be
logged in §8 with justification.

**Why this exists:** the central claim is a *predictive relationship*. The credibility
of such a claim collapses if descriptors or tests are chosen after seeing the outcomes
(p-hacking / HARKing). Fixing them now is the defense. See the integrity standards in
`GROUNDBREAKING_RESEARCH_PLAN.md`.

---

## 1. Hypotheses (stated before results)

- **H1 (polarity).** Whether the grapheme branch beats the BPE branch on a script is
  predicted by a single, pre-chosen script descriptor (primary: **neutral fertility** =
  mean GPT-2 byte-level-BPE tokens per grapheme cluster). Prediction: grapheme wins
  **iff** fertility exceeds a threshold lying between Hindi (3.40, BPE won in prior runs)
  and Bengali (3.92, grapheme won).
- **H2 (fusion gain).** The dual-branch fusion gain (fusion WRR − best single-branch WRR)
  **increases with branch disagreement**, and disagreement is itself predicted by
  fertility. Operationally: fusion gain correlates positively with fertility and with
  measured per-sample disagreement.
- **H3 (unifying / the spine).** The same descriptor that predicts H1/H2 also predicts
  **which held-out script transfers best in the zero-shot setting** (Part ②): higher-
  fertility, structurally-shared scripts transfer more.

**Directional, falsifiable prediction (Finding 2 from §9 of the tracker):** the four
Dravidian scripts (Tamil 6.07, Malayalam 6.02, Telugu 5.42, Kannada 5.29) should show
the **largest** grapheme-branch wins. If they do not, H1 in its simple-threshold form
is falsified (see §6).

## 2. Independent variables (descriptors) — the pre-specified horse-race

All computed from BSTD **train** labels only, **tokenizer-/result-blind**, by
`measure_script_descriptors.py` (already produces several; the starred ones to be added
BEFORE reading law results). The point is to test fertility *against* competitors, not
to assume it wins — this directly answers the "fertility is a weak predictor" literature
(Beyond Fertility / STRR, arXiv 2510.09947).

1. **neutral_fertility** — mean GPT-2 BPE tokens / grapheme cluster. *(PRIMARY — fixed in advance.)*
2. **bytes_per_cluster** — mean UTF-8 bytes / cluster (tokenizer-free fertility proxy).
3. **grapheme_entropy** — Shannon entropy of the cluster distribution.
4. **conjunct_density** — fraction of clusters containing a virama.
5. **clusters_per_word**, **chars_per_cluster** — script-complexity controls.
6. ★ **bpe_cluster_fragmentation** — fraction of grapheme clusters the neutral BPE
   splits into >1 token (a cleaner per-cluster cousin of fertility).
7. ★ **strr** — Single-Token Retention Rate: fraction of whole words encoded as a single
   neutral-BPE token (the metric arXiv 2510.09947 proposes as better than fertility).

**Explicitly NOT used (honesty):** morphological-consistency F1 / morphological edit
distance require gold morpheme segmentation we do not have for these scripts; we will
say so rather than approximate and over-claim.

## 3. Dependent variables (from `fusion_law_<lang>.log` + `conf_law_*`)

- **polarity** ∈ {grapheme, bpe} = sign of (grapheme WRR − BPE WRR). *(H1)*
- **grapheme_advantage** = grapheme WRR − BPE WRR (signed, continuous).
- **fusion_gain** = fusion WRR − max(grapheme WRR, BPE WRR). *(H2)*
- **disagreement** = fraction of test samples where the two branches' top-1 outputs
  differ (per-sample; thousands of points — this is where the statistical power lives).
- For Part ②: zero-shot WRR per held-out script at Rungs A/B/C.

## 4. Design controls

- **Equal training budget:** all languages capped to the **b1800** splits
  (1,620 train / 180 val; FULL official test). Prevents the script effect from being
  confounded by data volume (Hindi had 3× Bengali's data — Finding 4). Fixed seed 42.
- **Identical training recipe** across all languages/branches (same LoRA r, epochs, lr,
  batch size; only the tokenizer differs between branches).
- **One uncapped sensitivity run** (a high-fertility + a low-fertility script trained on
  all available data) to confirm the relationship is not an artifact of the cap.

## 5. Statistical analysis plan (fixed in advance)

- **H1:** logistic regression of polarity on each descriptor (one at a time);
  **leave-one-script-out cross-validation**; report per-descriptor LOSO accuracy.
  Primary test uses `neutral_fertility`; others reported as the horse-race.
- **H2:** linear regression of `fusion_gain` on `neutral_fertility` and on
  `disagreement`; report R², slope, and **bootstrap 95% CIs** (resampling scripts).
- **Per-sample power:** because N_scripts is small (~9–13), back every per-script claim
  with per-sample analysis — paired bootstrap / McNemar on the test samples for
  branch-vs-branch and fusion-vs-best.
- **Multiple comparisons:** the horse-race tests several descriptors; report all tried
  (no silent dropping) and treat only the pre-registered primary as confirmatory; the
  rest are exploratory.
- **H3:** rank-correlation (Spearman) between the descriptor and zero-shot transfer WRR
  across held-out scripts.

## 6. Success / falsification criteria (committed now)

- **Law supported if:** the primary descriptor predicts polarity for **≥8/9 Brahmic
  scripts under LOSO** AND explains `fusion_gain` with **R² > ~0.6** (bootstrap CI
  excluding 0 slope).
- **Law falsified / downgraded if:** Dravidian scripts do **not** show the largest
  grapheme wins, OR LOSO polarity accuracy ≤ chance, OR fusion-gain slope CI includes 0.
  In that case we report the negative result honestly and pivot the framing to "fusion
  helps regardless, but is not predictable from fertility" (still publishable, weaker).
- A competitor descriptor (STRR / fragmentation) beating fertility is **not** a failure —
  it is a finding; we report whichever predicts best.

## 7. Part ② (zero-shot) protocol — fixed

- **Shared abugida space** via deterministic Unicode rules (`build_shared_grapheme_space.py`,
  to be written). Leave-one-script-out: train on 8 Brahmic scripts, test the 9th.
- **Three rungs, all reported:** A = zero target data; B (headline) = + synthetic
  font-rendered target words, no real images; C = + 50–100 real target words (slope).
- **Baselines:** Florence-2 zero-shot (already 0.0 on disk) and a BPE model under the
  same protocol.
- **Success bar:** clearly-above-baseline Rung-B WRR on ≥1 held-out script, and H3
  rank-correlation positive.

## 8. Deviation log (append-only)

- *(none yet — freeze 2026-06-18)*
- **2026-07-02 — AMENDMENT 1 (H3 descriptor horse-race made explicit).** Filed when exactly
  **2/9 Rung-B results were known** (tamil WRR 9.16, telugu WRR 19.82; kannada Rung B crashed
  at epoch 6 in a power outage and was never evaluated — no other Rung-B value has been
  observed by anyone). §5 already defines H3 as Spearman(descriptor, Rung-B WRR) with
  `neutral_fertility` as the confirmatory primary. We now state explicitly that the full §2
  descriptor set **plus the token-coverage / type-coverage signals** (computed 2026-06-25,
  result-blind, from source-vocab coverage of the held-out script's test tokens; values frozen
  in `RESEARCH_STATUS_AND_PATH.md` §5 before any Rung-B result existed) enter the H3 horse-race
  as exploratory competitors. Fertility remains the sole confirmatory predictor; all
  competitors are reported, no silent dropping (§5 multiple-comparisons rule applies).
  **Disclosed honestly:** at filing time the 2 known points trend *against* fertility
  (higher-fertility Tamil transferred worse) and *with* token coverage. The remaining 7 Rung-B
  results (kannada, malayalam, oriya, gujarati, bengali, devanagari, gurmukhi) are unobserved
  at filing; rank predictions for them are committed in `PROSPECTIVE_PREDICTIONS_H3.md` in the
  same git commit as this amendment. Per §6, a competitor beating fertility is a finding, not
  a failure, and will be reported as such.
- **2026-07-02 — AMENDMENT 2 (visual-similarity control descriptor added to H3 horse-race).**
  To directly test the visual-similarity hypothesis of arXiv 2312.10806 ("visual/appearance
  similarity, not typology, drives cross-lingual STR transfer") with data rather than argument:
  define `visual_similarity(script)` = mean pairwise cosine similarity between the held-out
  script's rendered glyph images and the source scripts' rendered glyph images, embedded with
  the **frozen** Florence-2 vision encoder (no training; same fonts as the synth pipeline).
  Result-blind by construction (uses no WRR). Enters the H3 horse-race as exploratory. If it
  out-predicts the structural descriptors, we report that honestly.
- **2026-07-02 — AMENDMENT 3 (BPE baseline implementation detail).** §7 pre-registered "a BPE
  model under the same protocol" as a baseline; the orchestrator as implemented ran only
  grapheme-space rungs. Implemented now as **Rung Bbpe**: identical splits, training recipe,
  seed, and evaluation as Rung B, but stock Florence-2 tokenizer (no grapheme-vocab injection).
  Runs for all 9 scripts after the main 18 rungs complete (tamil, telugu first). At filing, the
  grapheme Rung-B numbers for tamil/telugu are known but their BPE counterparts are not; for
  the other 7 scripts neither is known. Purpose: test whether the shared-grapheme pivot space
  is necessary for zero-real-image transfer, vs. any synthetic fine-tuning sufficing.
- **2026-07-02 — housekeeping (non-analytic).** Kannada Rung B is retrained from scratch after
  the 2026-06-30 power outage (partial epoch-6 checkpoint set aside, never evaluated). The
  orchestrator's resume check was hardened (train-completion sentinel) so a mid-training crash
  can never cause evaluation of an undertrained checkpoint. `HF_HUB_OFFLINE=1` set (model is
  fully cached locally; removes network as a failure mode). None of these affect analysis.
- **2026-07-09 — AMENDMENT 4 (three Part-② extensions, designs fixed BEFORE any of them runs).**
  Filed with 8/9 Rung-B results observed (gurmukhi pending; its quantitative prediction is already
  filed in `PROSPECTIVE_PREDICTION_GURMUKHI.md`, committed and pushed before its training). None of
  the experiments below has started at filing; the Bbpe phase (Amendment 3) has also not yet begun.
  Motivating observation, disclosed at 047c30c filing and realized since: the 2×-synthetic-exposure
  covariate dominates (the two 2×-synth scripts, bengali 27.08 and devanagari 30.09, are the top two
  results), fertility is refuted (Spearman −0.88 at 8 points), and coverage orders transfer within a
  fixed synth budget (+0.77 among the six equal-synth scripts).

  **(a) Synthetic-exposure scaling study.** Scripts fixed now: **malayalam, kannada, telugu**
  (chosen to span token coverage 79.2 / 90.8 / 92.3 with the three smallest test sets 547/720/545 —
  cheap evaluation; all three are 1×-synth scripts so the existing 3240-image Rung-B result is
  reused as the middle point). Budgets: **{810, 1620, 3240*, 6480, 12960}** synthetic images
  (*existing). Smaller budgets = random subset of the existing 3240 set (seed 42); larger budgets =
  additional words rendered by the SAME pipeline, fonts, and verification (`prepare_zeroshot_loso.py`
  protocol). Training recipe identical to Rung B in every other respect (same LoRA config, lr,
  batch size, 15 epochs, seed 42). DV: Rung-B WRR (CharAcc/CER reported alongside). Analysis fixed
  in advance: per-script fit WRR = α + β·log2(synth), and joint two-factor fit
  WRR = a + b·tok_cov + c·log2(synth/3240); report R² and bootstrap CIs for β and c.
  **Declared point prediction to test:** the natural 2× experiment estimates c ≈ +12.3 WRR per
  doubling (fit of 2026-07-09, `PROSPECTIVE_PREDICTION_GURMUKHI.md`); we test whether the swept c
  is compatible. Saturation/concavity, if seen, is reported as-is; a flat curve falsifies the
  synth-quantity account and is reported per §6.

  **(b) Out-of-benchmark validation on Khmer.** Purpose: test whether the recipe extends beyond
  BSTD and beyond India — Khmer is a Brahmi-descended abugida in a different Unicode block
  (U+1780–17FF, coeng-based stacking) with an independent benchmark (KhmerST, ACCV 2024;
  WildKhmerST if suitable word crops can be derived). Protocol: train on ALL NINE BSTD scripts as
  sources (Khmer is outside BSTD, so this is not LOSO; stated openly) + 3240 synthetic Khmer images
  from the same pipeline (fonts via fc-match with per-image blank verification; the grapheme
  segmenter's virama rule must handle U+17D2 — an engineering check, timeboxed to 3 days, done
  BEFORE the prediction is filed). Evaluation: real Khmer test images; primary metric CER, with
  word-level WRR where word crops are derivable (deviation from the BSTD word-crop protocol is
  forced by the line-level source data and disclosed here). **Prediction protocol fixed now:**
  before the Khmer rung trains, compute Khmer token coverage result-blind (same 2026-06-25 method)
  and file `PROSPECTIVE_PREDICTION_KHMER.md` from the frozen two-factor model, committed and pushed
  first. A near-zero result is the boundary-of-transfer finding and is reported per §6.

  **(c) Frontier-VLM baseline.** To quantify the script gap in current-generation models: evaluate
  one or two open vision-language models (Qwen2.5-VL-7B-Instruct; optionally a Qwen3-VL size that
  fits inference on free GPU windows) zero-shot on the nine BSTD test sets (and the Khmer test set
  if (b) runs), with a fixed simple prompt ("Read the text in the image."), greedy decoding, and
  the SAME normalization/WRR metric as all our results. Declared caveat: these models' training
  data is unknown and may contain real text in these scripts, so they are an UPPER-bound reference
  for off-the-shelf capability, not a like-for-like comparison to our zero-real-target protocol;
  results reported whatever they show, including any script where the frontier model is strong.

  **(d) Standing prediction protocol.** The two-factor model (coefficients frozen in
  `PROSPECTIVE_PREDICTION_GURMUKHI.md`, refit only at declared checkpoints after new results are
  public) is the designated instrument for all future per-script predictions; each prediction is
  filed in its own `PROSPECTIVE_PREDICTION_<script>.md`, committed and pushed before the run it
  predicts, with point, ±1·RMSE and ±2·RMSE bands, and misses reported exactly like hits.

- **2026-07-18 — AMENDMENT 5 (scaling-study word-pool constraint + fixed implementation).**
  Filed BEFORE any Amendment-4a run exists (0/12 trained; data generation for the sweep starts
  after this amendment is committed). While implementing 4a we checked the word source and found
  a constraint the 4a filing missed: synthetic words come from the target language's BSTD
  **train-split text** (never test text — rendering test-split words would leak test vocabulary
  and is forbidden, unchanged), and the full train pools are **malayalam 2,393 / kannada 2,208 /
  telugu 2,215 records** (the base 3,240-image set already uses 1,620 of them at 2 renders
  each). The declared budgets {6480, 12960} therefore cannot be met by "additional words" alone
  (6480 needs 3,240 word-slots; 12960 needs 6,480). Fixed implementation, declared now:
  1. **810 / 1620:** prefix of a single seed-42 shuffle of the existing 3,240-image list
     (⇒ nested subsets 810 ⊂ 1620 ⊂ 3240), as filed in 4a.
  2. **6480:** existing 3,240 + 3,240 NEW images rendered by the same pipeline/fonts/verification
     (`prepare_zeroshot_loso.py` functions imported, not reimplemented), new-render rng =
     fresh `random.Random(42)`. New images use **previously-unused train records first**
     (588–773 per language, ×2 renders each = "additional words" as 4a intended), then top up by
     cycling the full train pool. Fonts are asserted identical to the recorded per-script font
     lists in `zeroshot_loso_meta_*.json`; blank-rate assert <20% unchanged.
  3. **12960:** the 6480 set + 6,480 further renders cycling the full pool (no new words exist).
     Nested: 6480 ⊂ 12960.
  4. **Disclosed interpretation rule, fixed in advance:** the 3240→6480 step is the faithful
     test of the natural experiment's per-doubling coefficient (it adds mostly new words, as
     bengali/devanagari did); the 6480→12960 step raises render quantity at (near-)fixed
     lexicon, so a flattening there cannot distinguish synth-quantity saturation from lexicon
     exhaustion — both readings will be reported. Lexicon sizes reported per rung.
  5. **Run order, fixed now:** budget phases 6480 → 1620 → 810 → 12960, each over
     (malayalam, kannada, telugu). Rationale: the declared-c test first; cheap rungs next;
     the longest rung last. Training recipe/args identical to Rung B (same orchestrator recipe,
     grapheme injection ON, per-budget vocab built from that budget's train set by the same
     `build_vocab`).
  6. Per-rung point predictions from the 9-pt two-factor instrument (+11.48·log2(synth/3240)
     extension, RMSE 4.18 bands) are filed in `PROSPECTIVE_PREDICTION_SCALING.md`, committed and
     pushed before the first sweep run trains; where the linear-in-log2 extrapolation falls below
     the script's Rung-A baseline it is clipped to that baseline, with the clip disclosed there.
