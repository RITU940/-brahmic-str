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
