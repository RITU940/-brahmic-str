# Groundbreaking Research Plan & Living Tracker
**Project:** Tokenization for Brahmic Scripts — a predictive law + zero-shot cross-script scene-text recognition
**Owner:** Ritu Baskey
**Created:** 2026-06-16 · **Last updated:** 2026-06-21
**Status legend:** ⬜ not started · 🟡 in progress · ✅ done · ⚠️ blocked/risk

> This is the single source of truth for the "groundbreaking paper" pivot. Every
> number here was cross-checked against files on disk (latest: 2026-06-18, see §8).
> Update the checkboxes and the "Last updated" date as work proceeds.

---

## ★ DECISION LOCKED (2026-06-18)

**Scope:** Pursue the **FULL ①+② unified paper** (the Law + zero-shot cross-script),
to **top-tier quality**. *Explicit owner directive: time and GPU are NOT constraints;
optimize purely for strength, correctness, and publishability at a high-tier venue.*

**Thesis (one sentence — the spine of the paper):**
> *We can read Brahmic scripts for which we have zero labeled images, and we can
> predict in advance — from a single measurable property of the writing system —
> which scripts that works for and whether grapheme or subword tokenization wins.*

**Why this and not a narrower paper (decided after a literature stress-test, §11):**
- Fusion alone is pre-empted (MGP-STR, IJCV Jan 2026 — learnable multi-granularity
  fusion). It is a *component* here, never the headline.
- "The Law" alone walks into the live *fertility-is-a-weak-predictor* literature
  (Beyond Fertility / STRR, arXiv 2510.09947). Survivable only with the horse-race
  + statistical rigor in `PREREGISTRATION.md`.
- Zero-shot alone ≈ transliteration transfer, already done in NLP (TransliCo).
- The **unpublished, hard-to-preempt asset is the LINK**: a script property that
  *predicts transfer to unseen scripts* (§4.3 unifying figure). That figure is the paper.

**Working venue targets:** IJCV / Pattern Recognition / IJDAR (top tier); arXiv
preprint first for priority. (Internship "deadline" reframed: arXiv preprint is the
real near-term deliverable; journal acceptance comes months later — review cycles.)

---

## ★ RIGOR & INTEGRITY STANDARDS (binding — this is what keeps integrity questions off the work)

1. **Pre-register before peeking.** The analysis plan (hypotheses, descriptors,
   dependent variables, statistical tests, success/falsification criteria) is fixed
   in `PREREGISTRATION.md` BEFORE any law-branch results are read. No post-hoc
   descriptor selection.
2. **Every number → a file.** Each figure/table value traces to a result JSON on
   disk (extend the §8 cross-check log). No estimated, rounded-from-memory, or
   "plausible" numbers anywhere.
3. **Real citations only.** No invented references. Every cited paper is verified
   (arXiv id / DOI) before it enters the manuscript.
4. **Report negatives.** Rung-A near-zero results, descriptors that DON'T predict,
   the ROVER loss — all stay in. Honesty is the armor.
5. **Statistical teeth for small N.** 9 scripts is too few for a "law" on its own:
   back per-script claims with per-sample disagreement (large N), leave-one-script-out
   CV, bootstrap CIs. Never call it a physical "law" in the prose — "predictive relationship".
6. **Reproducible.** Fixed seeds, committed splits, runnable scripts end-to-end.
7. **Human-quality prose.** Manuscript is written to read like a domain expert wrote
   it: specific, hedged where uncertain, no generic LLM filler.

## ★ SESSION CONTINUITY (read this first in a new session)

When resuming this work, in order: (1) read this file top-to-bottom; (2) read
`PREREGISTRATION.md` (the fixed analysis plan); (3) read `PROF_REPORT.md` for the
prior (pre-pivot) results; (4) check §8 cross-check log + §9 progress log for the
latest verified disk state; (5) re-verify any number it intends to use against the
named result file before relying on it. The auto-memory index (`MEMORY.md`) also
points here.

---

## 0. Plain-English summary (for non-CS readers)

We are no longer just saying "our trick improves Bengali OCR." Two existing papers
(MGP-STR, IJCV 2025; GraDeT-HTR, EMNLP 2025) already published pieces of that idea,
so it is *not* groundbreaking on its own.

Instead we are going after **two new things at once**:

1. **A LAW (the science).** We will prove that a single, *measurable* property of a
   writing system *predicts* (a) whether the "grapheme" way of reading beats the
   "subword" way, and (b) how much you gain by combining them. If one number can
   predict this across ~10 different scripts, that is a *scientific finding* others
   will cite and reuse — not a one-off trick.

2. **A NEW ABILITY (the wow).** We will build one model that can read a script it has
   **never seen a single labeled photo of**, by exploiting the fact that all Indian
   "Brahmic" scripts are built from the same underlying building blocks. This directly
   attacks the real-world problem: the lowest-resource scripts have *no* training data.

The law explains *why* the zero-shot ability works → one tight, top-tier paper.

---

## 1. The contribution, precisely stated

**Title (working):** *A Predictive Law of Tokenization Granularity for Brahmic Scripts,
and Zero-Shot Scene-Text Recognition on Scripts with No Labeled Data.*

**Part ① — The Law.**
> Define **fertility** = average number of subword (BPE) tokens the standard tokenizer
> uses to encode one *grapheme cluster* of a script. Hypothesis:
> (a) the grapheme branch beats the BPE branch **iff** fertility > a threshold; and
> (b) the **fusion gain** (combining both branches) grows with branch *disagreement*,
> which is itself predicted by fertility.
>
> Seed evidence already on disk: polarity **flips** between Bengali (grapheme wins:
> 57.3 vs 52.4) and Hindi (BPE wins: 56.9 vs 52.6), yet fusion gains in *both*.

**Part ② — Zero-shot cross-script.**
> Map every script's grapheme clusters into a **shared abugida primitive space**
> (consonant + vowel-sign + virama structure, via deterministic Unicode rules).
> Train on N−1 scripts; read a held-out script with **zero labeled real images**.
> Report three honesty rungs (§4.2).

**Why reviewers can't call it incremental:**
- vs **MGP-STR** (IJCV 2025, confidence + learnable fusion of granularities): they have
  no *theory of when/why* granularities help, and no cross-script transfer.
- vs **GraDeT-HTR** (EMNLP 2025, grapheme tokenizer for Bengali handwritten): single
  script, single language, no law, no zero-shot, not scene text, not a VLM.

---

## 2. Datasets (VERIFIED on disk 2026-06-16)

### 2.1 Primary dataset: BSTD Recognition (Bharat Scene Text Dataset)
- **Location:** `benchmarks/bstd_recognition.zip` (847 MB, 106,490 files) and the
  already-extracted `benchmarks/bstd/Recognition/` (train + test image folders +
  `train_recognition_data.json`, `test_recognition_data.json`).
- **Label format (verified):** JSON dict, `filename → {path, language, text}`.
  77,715 train label entries total.
- **Contents (image counts straight from the zip):**

  | Language | Script | Train | Test | Already prepared? |
  |---|---|---|---|---|
  | Bengali | Bengali | 4,936 | 1,368 | ✅ `florence2_splits_bstd.json` |
  | Assamese | Bengali | 2,627 | 1,505 | ✅ `florence2_splits_bstd_assamese.json` |
  | Hindi | Devanagari | 14,927 | 4,846 | ✅ `florence2_splits_bstd_hindi.json` |
  | Marathi | Devanagari | 3,917 | 1,196 | ⬜ |
  | Gujarati | Gujarati | 1,884 | 1,015 | ⬜ |
  | Punjabi | Gurmukhi | 8,310 | 2,879 | ⬜ |
  | Kannada | Kannada | 2,208 | 720 | ⬜ |
  | Malayalam | Malayalam | 2,393 | 547 | ⬜ |
  | Odia | Odia | 3,148 | 1,044 | ⬜ |
  | Tamil | Tamil | 2,029 | 513 | ⬜ |
  | Telugu | Telugu | 2,215 | 545 | ⬜ |
  | English (Latin control) | Latin | 29,121 | 12,568 | ⬜ |

- **Distinct Brahmic scripts = 9** (Bengali, Devanagari, Gujarati, Gurmukhi, Kannada,
  Malayalam, Odia, Tamil, Telugu) across **11 Indic languages**, **+ Latin control**.
  → **13 data points for the law**, **9 scripts for leave-one-out zero-shot.**

### 2.2 Secondary dataset: our in-house Bengali set (independent test)
- `Bengali/` (images) + `Bengali_gt/` (labels), split in `florence2_splits.json`:
  **train 3,207 / val 397 / test 370** (3,974 clean labeled, leakage-free, by-document).
- Role: an *independent* Bengali test set we collected → strengthens external validity.
  **Not used for the law/zero-shot core; kept as a robustness check.**

### 2.3 Synthetic (for Part ② Rung B only)
- Generated from Unicode fonts (no collection needed). Existing generators:
  `generate_synthetic.py`, `generate_synthetic_curriculum.py`.

---

## 3. Do we need MORE images? — DECISION: **No.** (with one methodological caveat)

**Verdict:** We do **not** collect or add real images. Justification:
1. The law needs **breadth (many scripts)**, which we already have (13 points).
2. Zero-shot needs **zero** target images by design.
3. The paper's thesis is *low-resource behavior*; more data per script would dilute it.
4. Absolute per-script SOTA is **not** our claim (PARSeq on 8.48M synth already holds
   that) — so chasing more data is the wrong race.

**Methodological caveat (the opposite of "add data"):** to make the law clean, we
**equalize the training budget across all languages** so any difference is caused by
the *script*, not by data quantity.
- **Common train budget:** 2,000 words/language (fixed seed). Languages with fewer
  (none below 1,884) use all available; English is *down*-sampled from 29k to 2,000.
- **Validation:** ~250–500/language. **Test:** always the FULL official test set.
- We also run one **uncapped** sensitivity check to confirm the law survives budget
  changes. `prepare_bstd_lang.py` already supports a `train_budget` argument.

---

## 4. Methodology & experiments

### 4.1 Part ① — measuring the law
For **each** of the 13 entries:
1. **Build script descriptors BEFORE seeing results** (pre-registered, anti-cheating):
   - `fertility` = mean BPE tokens per grapheme cluster (primary descriptor)
   - `grapheme_entropy` = Shannon entropy of grapheme-cluster distribution
   - `conjunct_density` = fraction of clusters containing a virama/conjunct
   - `bpe_oov_rate` = fraction of grapheme clusters with no clean BPE coverage
   - New script to write: `measure_script_descriptors.py` (no GPU; pure text stats).
2. **Train two branches** at the common budget: BPE (`mode=STANDARD`) and grapheme
   (`mode=GRAPHEME`) — `train_florence2.py` + per-language grapheme vocab from
   `build_grapheme_vocab_lang.py`.
3. **Evaluate** WRR / CharAcc / CER + per-sample confidences (`conf_*.json`) →
   compute **fusion gain** and **polarity** (which branch wins).
4. **Fit the law:** regress polarity (classification) and fusion-gain (regression) on
   the descriptors. Report R², which single descriptor wins, leave-one-script-out CV.
   - **Success bar:** one descriptor predicts polarity for ≥8/9 scripts AND explains
     fusion gain with R² > ~0.6.

### 4.2 Part ② — zero-shot cross-script
1. **Shared abugida space:** map each script's clusters to common phonetic primitives
   via deterministic Unicode rules. Tools: `aksharamukha` / `indic-transliteration`.
   New script: `build_shared_grapheme_space.py`.
2. **Leave-one-script-out:** train shared-grapheme model on 8 Brahmic scripts, test on
   the 9th with **zero real target images**. Repeat for each held-out script.
3. **Three honesty rungs (report ALL):**
   - **Rung A:** zero target data at all (pure structural transfer). Likely low; any
     non-trivial number is news.
   - **Rung B (HEADLINE):** + synthetic font-rendered target words (no real images).
   - **Rung C:** + 50–100 real target words (few-shot) to show the slope.
4. **Baselines to beat:** Florence-2 zero-shot (already 0.0% — on disk), and a BPE
   model under the same protocol (should fail to transfer).
   - **Success bar:** clearly-above-baseline WRR in Rung B on ≥1 held-out script.

### 4.3 The unifying result
Show the law from ① *predicts which held-out scripts transfer best* in ② (high-fertility,
structurally-shared scripts transfer more). One figure tying the two halves = the paper's
spine.

---

## 5. Task tracker (update these as you go)

### Phase A — Priority preprint (protect the idea) ⬜
- ⬜ A1. Polish current results into an arXiv preprint (reuse `paper/main.tex`)
- ⬜ A2. Submit to arXiv (free, immediate, international visibility)

### Phase B — The Law (Part ①) 🟡
- ✅ B1. Write `measure_script_descriptors.py`; output `script_descriptors.json` (no GPU) — DONE 2026-06-16, see §10.
        **UPDATED 2026-06-21:** added the two pre-registered ★ horse-race descriptors
        (`strr`, `bpe_cluster_fragmentation`) BEFORE reading full law results. Result: both
        are DEGENERATE within Brahmic (STRR=0 for every Indic script; fragmentation ≈0.93–0.98
        saturated) → only `bpe_fertility_neutral` discriminates. See §9 (2026-06-21).
- ✅ B2. Equal-budget (1,800) splits for ALL 12 langs — DONE 2026-06-16
        (`florence2_splits_bstd_<lang>_b1800.json`; each = 1,620 train / 180 val / full test).
        Used 1,800 (just below smallest lang Gujarati 1,884) so all 12 are EXACTLY matched.
        Extracted the 9 missing languages' images from the zip first.
- ✅ B3. Grapheme vocabs for ALL 12 langs — DONE 2026-06-16
        (`grapheme_vocab_<lang>_b1800.json`; e.g. Malayalam 790, Telugu 701, English 85 graphemes).
- 🟡 B4. Train BPE + grapheme branches at b1800 on server3 — **4/12 DONE** (Dravidian:
        tamil, telugu, kannada, malayalam — finished 2026-06-18; verified bug-free 2026-06-21).
        **Remaining 8 LAUNCHED 2026-06-21** (`law_run_rest.log`, PID 3026179, resumable,
        skips done). Order: bengali assamese hindi marathi gujarati punjabi odia english.
- 🟡 B5. Eval + confidences + fusion per language → 4/12 done (`conf_*_law_<lang>.json`,
        `fusion_law_<lang>.log`); auto-run by the launcher for the remaining 8.
- 🟡 B6. Fit + validate the law — `fit_law.py` WRITTEN + tested + staged to server3 (numpy-only:
        LOSO threshold horse-race + fertility regression w/ bootstrap CI per PREREG §5).
        **PREVIEW UPDATED on 9/12 (2026-06-22):** GUJARATI = FIRST MISS — fert 4.90 predicted
        grapheme but BPE won by 0.39 (26.70 vs 27.09, a near-tie/zero advantage). Polarity now
        **8/9 LOSO (88.9%) — STILL meets the ≥8/9 bar.** Fertility still wins horse-race (89% vs
        ≤78% all competitors). H2: grapheme_adv R²=0.72 CI[+2.04,+7.40]; fusion_gain R²=0.71
        CI[+1.01,+3.03], both exclude 0. Fusion still helped Gujarati (+6.5). HONEST NOTE/LEAD:
        Gujarati has lowest conjunct_density (0.097) of the high-fert scripts → advantage may
        shrink for structurally simpler scripts even at high fertility (paper refinement, not a
        failure). Report negative honestly. UPDATE on 10/12 (+punjabi, fert 3.17→BPE as predicted):
        polarity 9/10, fertility LOSO 90%, grapheme_adv R²=0.77, fusion_gain R²=0.73.
        ✅ FIGURE done: `make_law_figure.py` (system python3+matplotlib, reads law_fit_results.json)
        → figures/law_main.pdf/.png — Panel(a) fertility vs grapheme-advantage w/ flip zone + law
        line + R²; Panel(b) descriptor horse-race (fertility 90% highlighted). Re-run on all 12.
- ⬜ B7. Uncapped HEADLINE runs — **DECIDED 2026-06-21: YES, do them** (owner approved).
        Separate uncapped, fully-converged per-script models → strong absolute-WRR headline
        table; capped b1800 models stay ONLY for the clean law comparison. Two labeled purposes.

### Phase C — Zero-shot (Part ②) 🟡
- ✅ C1. `build_shared_grapheme_space.py` — DONE + VERIFIED round-trip 99.9–100% (2026-06-22, §9)
- 🟡 C2. Multi-script training data in shared space — PILOT done (`prepare_zeroshot_pilot.py`;
        held-out Tamil, sources telugu+kannada+malayalam; Rung A/B splits+vocabs+synth built,
        staged to server3). Pre-GPU coverage signal 87.9%. Full N-script version: TODO.
- 🟡 C3. Leave-one-script-out training — PILOT auto-chained (`run_pilot_chain.sh` PID 3156898/900
        → `run_zeroshot_pilot_server3.sh tamil` after law batch). Full LOSO over all scripts: TODO.
- 🟡 C4. Rung A/B/C eval — pilot does Rung A + Rung B on Tamil; Rung C (few-shot) + per-script: TODO.
- ⬜ C5. Tie to the law (§4.3) — the unifying figure (needs pilot results + full law fit first)

### Phase D — Manuscript ⬜
- ⬜ D1. Rewrite around the two-part contribution
- ⬜ D2. Regenerate all figures from result files (extend `make_figures.py`)
- ⬜ D3. Target venue submission (see §7)

---

## 6. Compute budget (reality check)
- Per training run (from logs): STANDARD ≈ 2.7 h, GRAPHEME ≈ 5 h (15 epochs, Florence-2-base, LoRA).
- Part ①: ~18 new runs → ~70–100 GPU-hours.
- Part ②: ~9 leave-one-out runs (multi-script, larger) → ~30–50 GPU-hours.
- **Total ≈ 100–150 GPU-hours.** On one consumer GPU = weeks, not days.
- **Local GPU (verified 2026-06-16):** 1× NVIDIA RTX 4000 Ada, 20 GB VRAM — enough for
  Florence-2-base LoRA. BUT currently 100% busy with an UNRELATED project (`diff3f`,
  PartNet 3D meshes, PID 4123629). OCR env is free; wait for the GPU or use server #2.
- **GPU server #2 (verified 2026-06-16):** `server3` = host `cvpr-gamma`, 192.168.57.100,
  user `ujjwalb`, Ubuntu 20.04. 1× **NVIDIA RTX A5000, 24 GB** (driver 550, CUDA 12.4).
  Currently ~10 GB free / 100% util (two python jobs running). Good for Florence-2 LoRA.
  ⚠️ It's a SEPARATE machine reached by PASSWORD ssh → the assistant cannot drive it
  directly. To use it: (a) set up an ssh KEY from this box to server3, copy the repo+data
  (`rsync`), recreate the `ritu_scenetext` env, then run `run_law_training.sh` there; or
  (b) user runs the runner on server3 manually. Decision pending.
- **Training env (verified):** `/home/ujjwal/miniconda3/envs/ritu_scenetext/bin/python`
  (transformers 4.44.2, torch 2.5.1+cu121). Launch template: `run_crossscript.sh`.
- ⚠️ OPEN ITEM: confirm server #2 specs + free hours/day → fix how many scripts we attempt.

---

## 7. Publication strategy
- **Now:** arXiv preprint of current results → priority + visibility.
- **Then:** the strong ①+② paper → **Pattern Recognition / IJDAR / IJCV** (top tier).
- **Fallback (MVP-groundbreaking):** Part ① alone on 6–7 scripts is already a novel
  finding and a strong Q1 submission. Part ② is the amplifier, not a single point of
  failure.
- Honesty (rungs, negative results, calibration caveats) is the armor that gets ambitious
  work past reviewers.

---

## 8. Cross-check log (what was verified, when)
- 2026-06-16: BSTD zip language inventory + counts — read directly from
  `unzip -l benchmarks/bstd_recognition.zip` (table in §2.1).
- 2026-06-16: BSTD label format — read `train_recognition_data.json` (77,715 entries,
  dict `filename → {path, language, text}`).
- 2026-06-16: existing prepared splits — `florence2_splits*.json` (counts in §2).
- 2026-06-16: current result numbers (WRR/CharAcc/CER) — `results_*_ours.json`,
  `results_*_bstd.json`, `eval_metrics_v3.json`, `significance_report.json`,
  `parseq_report.json`, `calibration_report.json` (all match the prior report).
- 2026-06-16: trained checkpoints present — 11 `checkpoints_*` dirs, ~2.6 GB each.
- 2026-06-21: 4 Dravidian capped runs verified on server3 — WRRs read from
  `law_run_dravidian.log`; grapheme-vocab injection verified per-language correct (train==eval
  injected-token counts; vs `grapheme_vocab_<lang>_b1800.json` sizes). `checkpoints_law_<lang>_{standard,grapheme}`
  present for tamil/telugu/kannada/malayalam.
- 2026-06-21: remaining-8 run confirmed RUNNING on server3 (`law_run_rest.log`, PID 3026179).
- 2026-06-21: `script_descriptors.json` regenerated locally with `strr` + `bpe_cluster_fragmentation`
  added; values read from the script's own printed table (STRR=0 all Brahmic; frag 0.93–0.98).

---

## 9. Progress log & preliminary findings

### ⭐ SESSION HANDOFF — 2026-06-25 (NEW SESSION: READ THIS + `RESEARCH_STATUS_AND_PATH.md` FIRST) ⭐
**Machine moved:** we are now WORKING ON server3 directly (`cvpr-gamma`), not via ssh from
isiserver (isiserver storage was full). Working dir `/c/ujjwalb/ritu1/lstm_model`.
**See the new master doc `RESEARCH_STATUS_AND_PATH.md`** — it has the brutal honest
literature assessment (verified precedents: MGP-STR, GraDeT-HTR, STRR, zero-shot Chinese
stroke/radical line, distance-predicts-transfer), the reject-resistant PATH to top tier,
and all verified env facts (stale split paths, font fix, env).

**What happened this session:**
- **Part ① Law = confirmed COMPLETE & supported** (8/9 polarity, LOSO 90.9%, fusion_gain
  R²=0.70). Pilot zero-shot Tamil **Rung B WRR=9.94** confirmed (beats 0.0 baseline).
- **Built + launched the FULL 9-script leave-one-SCRIPT-out zero-shot** (the H3 unifying
  experiment). New: `prepare_zeroshot_loso.py`, `run_zeroshot_loso.sh`; all 9 datasets
  generated & verified (0% blank synth; coverage 79–97%). **Chain RUNNING** → `zeroshot_loso.log`.
- **Fixed two silent-failure bugs** before launch: (a) split files store stale
  `/home/ujjwal/ritu1` image paths (images really at local `benchmarks/`); (b) no Noto fonts
  here → old renderer made tofu. Both fixed in `prepare_zeroshot_loso.py`.
- **Monitor:** `pgrep -af run_zeroshot_loso; grep "RESULT LOSO" zeroshot_loso.log`. Resumable.

---

### ⭐ SESSION HANDOFF — 2026-06-23 ⭐
**Where we are RIGHT NOW:**
- **Part ① (the Law) = DONE & SUPPORTED.** All 12 b1800 runs finished. On the 9 Brahmic
  scripts: polarity **8/9** (only Gujarati misses), fertility LOSO 90.9%, fusion_gain R²=0.70,
  grapheme_adv R²=0.68 — BOTH pre-registered §6 criteria met. English = Latin control (kept
  out of the fit). Full numbers in the "2026-06-23 LAW BATCH COMPLETE" entry below.
- **Part ② pilot = RUNNING on server3.** Held-out Tamil; Rung A training now
  (`checkpoints_zeroshot_rungA_tamil`), then Rung B auto-runs. Output → `zeroshot_pilot.log`.

**⚠️ CRITICAL — the live watcher died with the old session.** The pilot itself keeps
running on server3 (independent nohup: `run_pilot_chain.sh` PID 3156900 → `run_zeroshot_pilot_server3.sh`).
**You must CHECK IT MANUALLY**, e.g.:
```
ssh ujjwalb@192.168.57.100 'cd /c/ujjwalb/ritu1/lstm_model && tail -15 zeroshot_pilot.log && grep "RESULT RUNG" zeroshot_pilot.log'
```
Look for two `[RESULT RUNG A/B tamil] WRR=...` lines + `ZEROSHOT PILOT COMPLETE`. Baseline:
Florence-2 raw zero-shot = 0.0; supervised Tamil std=28.07/grph=36.84. **Rung B clearly
beating 0.0 = the groundbreaking direction is real.** (If you want a notifier, re-arm a Monitor
on `zeroshot_pilot.log` for `ZEROSHOT PILOT COMPLETE`.)

**NEW FILES THIS SESSION (all in `lstm_model/`):**
- `build_shared_grapheme_space.py` — Brahmic→pivot mapping (`--verify` = 99.9–100% round-trip). DONE.
- `prepare_zeroshot_pilot.py` — builds Rung A/B splits+vocabs+synth. (sources telugu+kannada+malayalam → tamil).
- `run_zeroshot_pilot_server3.sh`, `run_pilot_chain.sh` — pilot runner + post-law auto-chain.
- `fit_law.py` — numpy-only law fit (LOSO horse-race + bootstrap CI). Run: `python fit_law.py --logdir law_logs`.
- `make_law_figure.py` — money figure (system `python3`, matplotlib). → `figures/law_main.png`.
- FIXED `fusion_analysis.py` (oracle now uses `normalize_bengali`). Added strr+bpe_cluster_fragmentation to `measure_script_descriptors.py`.
- Results: `law_logs/` (12 fusion logs), `law_fit_results_brahmic.json`, `figures/law_main.png`.

**NEXT STEPS (in order):** (1) collect pilot Rung A/B WRR when done. (2) If Rung B >0: extend
to full leave-one-script-out over more held-out scripts + the unifying figure (law predicts
transfer, H3/§4.3). (3) B7 uncapped headline runs (owner approved). (4) Rung C few-shot.
(5) Manuscript (Phase D). **Server3:** `ssh ujjwalb@192.168.57.100`, dir `/c/ujjwalb/ritu1/lstm_model`,
env `/c/ujjwalb/.conda/envs/ritu_scenetext/bin/python`. Never touch other users' jobs.



### 2026-06-23 — LAW BATCH COMPLETE (all 12) — PRE-REGISTERED LAW SUPPORTED
Final capped b1800 results (fixed oracle). Odia grapheme +14.27 (biggest win); English
(Latin control) grapheme +1.83.

Full table (fertility / BPE / grapheme / adv / winner): Odia 5.17/33.1/47.4/+14.3/grph,
Tamil 6.07/28.1/36.8/+8.8/grph, Malayalam 6.02/15.2/20.8/+5.7/grph, Telugu 5.42/21.3/26.8/+5.5/grph,
Bengali 3.92/27.1/30.0/+2.9/grph, Kannada 5.29/21.8/23.8/+1.9/grph, Assamese 3.94/26.7/27.5/+0.8/grph,
**Gujarati 4.90/27.1/26.7/−0.4/BPE (MISS)**, Hindi 3.40/40.8/29.8/−11.0/BPE,
Marathi 3.25/48.7/36.8/−11.9/BPE, Punjabi 3.17/54.8/44.3/−10.5/BPE,
*English(Latin CONTROL) 0.44/49.0/50.8/+1.8/grph*.

**PRE-REGISTERED VERDICT (PREREG §6) — BOTH criteria MET on the 9 Brahmic scripts
(English = Latin control, excluded from the law fit as pre-specified):**
- Polarity: **8/9 Brahmic scripts correct** (only Gujarati misses) — meets ≥8/9. Per-lang
  LOSO 10/11 = 90.9%. Fertility WINS the horse-race (90.9% vs ≤81.8% all competitors).
- fusion_gain **R²=0.698**, slope CI[+1.05,+2.67] excl. 0 — meets R²>~0.6.
- grapheme_advantage R²=0.678 (Brahmic). Saved: `law_fit_results_brahmic.json` (primary),
  figure `figures/law_main.png` (N=11).
**HONEST NUANCES (keep in paper):** (1) Gujarati near-tie miss (lowest conjunct_density 0.097).
(2) English control ALSO slightly favored grapheme (+1.8) → benefit not purely abugida; including
English in the linear fit drops R² 0.68→0.26 (it's an out-of-domain outlier at fert 0.44) — so it
stays a CONTROL, not a fit point (pre-registered, not post-hoc). (3) Odia +14.3 strong outlier above
the line → relationship monotonic, not perfectly linear. Pipeline: law_logs → fit_law.py →
make_law_figure.py (all reproducible). Pilot auto-launched 04:51 IST (Rung A Tamil training).

### 2026-06-22 — PART ② ZERO-SHOT PILOT BUILT + STAGED (auto-runs after law batch)
De-risking the groundbreaking claim early (owner directive). Built and verified the
foundation of Part ② while the law batch runs.

**(1) Shared abugida space — `build_shared_grapheme_space.py` (NEW, no GPU).**
Deterministic, reversible mapping of every Brahmic script into one pivot (Devanagari-
offset) space via the ISCII-aligned Indic Unicode blocks (corresponding letters share
the same in-block offset). `to_pivot` / `from_pivot` / `pivot_graphemes`. Dandas
(U+0964/65, script=Common) are passed through; OM/abbreviation deliberately NOT (their
offsets collide with real letters, e.g. Assamese RA U+09F0 & Gurmukhi TIPPI U+0A70 at
0x70). **VERIFIED round-trip (`--verify`): 99.9–100% char / 99.5–100% word for all 9
scripts** (Assamese/Bengali/Marathi/Telugu = 100.00%). Residual = genuine mixed-script
label contamination in BSTD GT (e.g. "भारतीय" inside a Kannada label) — correctly NOT
collapsible. This was the make-or-break correctness gate for Part ②; it PASSED.

**(2) Pilot data — `prepare_zeroshot_pilot.py` (NEW).** Leave-one-script-out, held-out
**TAMIL** (fertility 6.07 → law predicts transfer should work BEST here; unseen Dravidian
glyphs), sources = **telugu+kannada+malayalam** (Dravidian neighbours). Built:
splits_zeroshot_rungA_tamil.json (4860 src train / 540 val / 513 real Tamil test, all in
pivot), rungB (+3240 synthetic Tamil words, Noto-font rendered, NO real Tamil images →
8100 train), + source-only & +synth grapheme vocabs (pivot). Eval in pivot space (==
target-script WRR since round-trip lossless).
**>>> PRE-GPU FEASIBILITY SIGNAL (strong): the source-only pivot vocab already covers
87.9% of Tamil's test grapheme-token occurrences (65.9% of distinct units) with ZERO
Tamil training.** Output-side transfer is real; the remaining gap is visual (unseen
glyphs) → exactly what Rung B's synthetic rendering addresses.

**(3) Staged + auto-chained on server3.** Data rsynced, image paths rewritten
(/home/ujjwal/ritu1→/c/ujjwalb/ritu1), 0 missing verified. `run_zeroshot_pilot_server3.sh`
trains Rung A then Rung B (grapheme mode, pivot vocab) → pivot WRR on real Tamil test.
`run_pilot_chain.sh` (PID 3156900, RUNNING) waits for the law "LAW TRAINING COMPLETE"
marker, then auto-launches the pilot → `zeroshot_pilot.log`. Baselines: Florence-2 raw
zero-shot = 0.0; supervised Tamil b1800 std=28.07 / grph=36.84 (context ceilings).
Expectation: Rung A (zero target images) low-but-maybe-nonzero; Rung B (synthetic) is the
headline — must clearly beat 0.0 to call the direction validated.

### 2026-06-22 — POLARITY FLIP CONFIRMED (4 more langs done; law holding 8/8)
Remaining-8 run progressing on server3. New capped b1800 results (fixed oracle):

| Lang | Script | Fertility | BPE/Std WRR | Grapheme WRR | Grapheme adv. | Winner |
|---|---|---|---|---|---|---|
| Assamese | Bengali | 3.94 | 26.71 | 27.51 | +0.80 | grapheme |
| Bengali | Bengali | 3.92 | 27.12 | 30.04 | +2.92 | grapheme |
| Hindi | Devanagari | 3.40 | 40.78 | 29.82 | −10.96 | **BPE** |
| Marathi | Devanagari | 3.25 | 48.66 | 36.79 | −11.87 | **BPE** |

**KEY:** with the 4 Dravidian (all fertility >5, grapheme wins), polarity is now correct
for **8/8 scripts** under the pre-registered threshold (grapheme wins iff fertility > ~3.9;
crossover sits exactly between Hindi 3.40 and Bengali 3.92, as H1 predicted). Meets the
≥8/9 success bar already. Fusion gains in all 8 (H2 holding). NOTE these Bengali/Hindi
numbers are the proper b1800-capped versions (supersede the OLD uncapped §1 numbers for
the law fit). Remaining: gujarati (training), punjabi, odia, english. Predicted: gujarati
4.90 + odia 5.17 → grapheme; punjabi 3.17 + english 0.44 → BPE. If held = 12/12.
Per-lang DONE times: bengali 10:44PM, assamese 02:12AM, hindi 06:34AM, marathi 10:09AM.

### 2026-06-21 — SESSION SUMMARY (read this first for current state)
**Who/what:** verification + horse-race + launch of the remaining law runs. No results
were inflated; the session's whole point was to check honesty and push the *real* levers.

**(1) Verified the 4 Dravidian capped results are REAL (no bug).** A scare in the eval
logs — `"Injecting Bengali grapheme tokens..."` printed while evaluating Tamil — turned
out to be a **cosmetic hardcoded print string** in `inject_grapheme_tokens`
(`train_florence2.py:268`). The actual vocab loaded is per-language correct: injected
token counts match each language's own `grapheme_vocab_<lang>_b1800.json`
(tamil 553 of 579, telugu 678/703, kannada 656/678, malayalam 771/792), and **train-time
and eval-time injection counts are identical** → grapheme branch trustworthy.

**Verified Dravidian capped (b1800) results** (from `law_run_dravidian.log`, full test sets):

| Script | Fertility | BPE/Std WRR | Grapheme WRR | Grapheme adv. | Fusion WRR | Fusion gain | Oracle |
|---|---|---|---|---|---|---|---|
| Tamil | 6.07 | 28.07 | 36.84 | **+8.77** | 46.39 | +9.55 | 49.12 |
| Malayalam | 6.02 | 15.17 | 20.84 | **+5.67** | 26.69 | +5.85 | 25.59 ⚠️ |
| Telugu | 5.42 | 21.28 | 26.79 | **+5.51** | 33.76 | +6.97 | 35.78 |
| Kannada | 5.29 | 21.81 | 23.75 | **+1.94** | 30.83 | +7.08 | 34.03 |

**Preliminary law signal (STRONG):** grapheme wins in ALL 4, and grapheme advantage is
**monotonic in fertility** (Tamil>Malayalam>Telugu>Kannada ⇒ Spearman ρ=1.0 on these 4) —
exactly the pre-registered Finding-2 / H1 prediction, made before any number was seen.
✅ **ANOMALY RESOLVED (2026-06-21):** Malayalam fusion (26.69) > oracle (25.59) was a
**metric-mismatch BUG in `fusion_analysis.py`**, not a data problem. The oracle used
`(s or '').strip()` while WRR (`metrics.evaluate_corpus`) uses `normalize_bengali`
(NFC + zero-width ZWJ/ZWNJ removal + whitespace collapse) — so the oracle applied a
STRICTER correctness test than the WRR it bounds, understating it everywhere (visible
only on Malayalam). FIXED: oracle now calls `normalize_bengali` (committed local +
pushed to server3). Verified: Malayalam oracle → **28.52** (≥ fusion 26.69 ✓);
Standard/Grapheme/Fusion WRR UNCHANGED → the law's headline numbers are unaffected.
TODO: re-run `fusion_analysis.py` for **tamil** at the end (its fusion ran just before
the fix push); telugu/kannada/malayalam auto-corrected by the in-progress run.
⚠️ **NOTE:** prior Bengali/Hindi WRRs (§1) are from OLD *uncapped* runs — NOT comparable
to b1800; the law fit must use the b1800 Bengali/Hindi from the remaining-8 run.

**(2) Launched the remaining 8 capped runs** on server3 (GPU was idle, 23 GB free):
`nohup bash run_law_training_server3.sh > law_run_rest.log 2>&1 &` (PID 3026179).
Resumable, skips the 4 done. Order: bengali assamese hindi marathi gujarati punjabi odia
english. ETA ~1.5–2 days. Monitor: `ssh ujjwalb@192.168.57.100 'cd /c/ujjwalb/ritu1/lstm_model; tail law_run_rest.log; ls -d checkpoints_law_*'`.

**(3) Added the 2 pre-registered ★ descriptors** (`strr`, `bpe_cluster_fragmentation`) to
`measure_script_descriptors.py` and regenerated `script_descriptors.json` (local, no GPU),
BEFORE full law results exist (honest horse-race order). **Finding:** both are degenerate
within Brahmic — **STRR = 0.000 for every Indic script** (English 0.276), fragmentation
≈ 0.93–0.98 (saturated). Only `bpe_fertility_neutral` (3.17–6.07) discriminates. This
directly answers the "fertility is a weak predictor" literature: its proposed replacement
(STRR) is *non-discriminative* for Brahmic scene text. Reported as a horse-race outcome.

**(4) Strategic decisions locked by owner this session:**
- **Direction = "both, no compromise":** finish the law fully (all 9 scripts, capped +
  uncapped headline, full stats) → THEN Part ② zero-shot. Sequential.
- **Uncapped headline runs = YES** (B7 above): strong absolute WRR for credibility, kept
  separate from the capped law models. The capped low WRR (15–37%) is BY DESIGN (1,620-word
  budget) and is NOT the contribution — do not "fix" it by un-capping the law models.
- **Reframing recorded:** the groundbreaking novelty is Part ② (zero-real-image cross-script
  transfer predicted by the law), NOT bigger OCR numbers. Law = the explanation, not the headline.

**Immediate next actions (for the next session):**
1. Monitor `law_run_rest.log`; when done, collect 8× `fusion_law_<lang>.log` + `conf_*_law_*`.
2. ✅ DONE: fixed the Malayalam oracle>fusion bug (`fusion_analysis.py` metric mismatch).
   Remaining sub-task: re-run `fusion_analysis.py --std std_law_tamil --grph grph_law_tamil
   --self _none_` once GPU run frees up (tamil's oracle still on old value in its log).
3. Write `fit_law.py` (logistic reg of polarity + LOSO + bootstrap CI per PREREG §5) and
   produce the law figure/table once all 9 b1800 points are in.
4. Set up + launch uncapped headline runs (B7).
5. Only then start Part ② (`build_shared_grapheme_space.py`, PREREG §7).

### 2026-06-16 — Phase B1 done: script descriptors measured (NO GPU)

### 2026-06-16 — Phase B1 done: script descriptors measured (NO GPU)
Script: `measure_script_descriptors.py` → `script_descriptors.json`.
Measured over BSTD train labels for all 12 languages.

**Neutral fertility = mean GPT-2 byte-level-BPE tokens per grapheme cluster**
(GPT-2 = the English-centric byte-level BPE Florence-2/BART inherit; the FAIR
cross-script reference). Higher = the default tokenizer fragments the script more
= grapheme tokenization should help more.

| Language | Script | n_words | neutral fertility | bytes/cluster | conjunct density |
|---|---|---|---|---|---|
| Tamil | Tamil | 2,029 | **6.07** | 6.08 | 0.325 |
| Malayalam | Malayalam | 2,393 | **6.02** | 6.06 | 0.265 |
| Telugu | Telugu | 2,215 | 5.42 | 5.43 | 0.194 |
| Kannada | Kannada | 2,208 | 5.29 | 5.42 | 0.197 |
| Odia | Odia | 3,137 | 5.17 | 5.20 | 0.149 |
| Gujarati | Gujarati | 1,884 | 4.90 | 4.91 | 0.097 |
| Assamese | Bengali | 2,627 | 3.94 | 5.11 | 0.127 |
| **Bengali** | Bengali | 4,936 | **3.92** | 5.22 | 0.140 |
| **Hindi** | Devanagari | 14,927 | **3.40** | 5.34 | 0.132 |
| Marathi | Devanagari | 3,917 | 3.25 | 5.16 | 0.104 |
| Punjabi | Gurmukhi | 8,310 | 3.17 | 4.75 | 0.015 |
| English (control) | Latin | 29,119 | 0.44 | 1.00 | 0.000 |

**Finding 1 (the law's first confirmation):** neutral fertility orders the already-
trained scripts CORRECTLY. Bengali (3.92, grapheme branch WINS +4.9 ours / +2.4 BSTD)
sits just ABOVE Hindi (3.40, BPE branch WINS −4.3) — i.e. the polarity flip happens
right at this fertility boundary. One single variable already explains the known flip.

**Finding 2 (sharp falsifiable prediction):** the four Dravidian scripts (Tamil,
Malayalam, Telugu, Kannada; fertility 5.3–6.1) should show the LARGEST grapheme wins.
This is the headline experiment to run in B4. If it holds, the law is real.

**Finding 3 (methodology catch — important):** the in-repo "standard" tokenizer
(71,254 tokens) was secretly EXTENDED with Bengali tokens, so it is NOT a fair
cross-script reference (it made Bengali look like fertility 1.46). Fixed by switching
the primary descriptor to the neutral GPT-2 tokenizer. The biased value is kept as
`bpe_fertility_experiment` for transparency only.

**Finding 4 (confounder → confirms the equal-budget design):** Hindi has 14,927 train
words vs Bengali's 4,936 (3×). So Hindi's BPE-win could be partly a DATA-SIZE effect,
not purely a script effect. → We MUST cap all languages to a common ~2,000-word budget
(§3) before fitting the law, else script and data-volume are confounded. The plan
already calls for this; the data now proves it is necessary, not optional.

### 2026-06-16 — Phase B2 + B3 done: equal-budget data ready for all 12 languages
- Extracted the 9 missing languages' images from `bstd_recognition.zip` (all 12 now on disk).
- Added non-destructive `--out_suffix` to `prepare_bstd_lang.py` and `build_grapheme_vocab_lang.py`.
- Built 12 equal-budget splits (`*_b1800.json`, 1,620 train / 180 val each) + 12 vocabs.
- Wrote `run_law_training.sh` (the B4 launcher; resumable, GPU-memory-aware).
- **Status: everything up to the GPU step is DONE.** B4 just needs a free GPU.

### 2026-06-18 — Phase B4 LAUNCHED on server3 (GPU resolved)
**Decision:** run on **server3 (RTX A5000, 24 GB)** — owner approved ("go ahead with
server3, don't touch anyone else's stuff").

**Env (verified working, NO rebuild needed):** the existing conda env
`/c/ujjwalb/.conda/envs/ritu_scenetext` already has the exact stack —
**Python 3.10.20, torch 2.5.1+cu121 (CUDA True), transformers 4.44.2, peft 0.13.0**,
sees the A5000. (A half-started Miniconda install this session was abandoned/cleaned;
nothing was installed wrongly — it failed clean under `set -e`.)

**Data staged on server3 (verified):**
- rsynced all 12 `florence2_splits_bstd_*_b1800.json`, 12 `grapheme_vocab_*_b1800.json`,
  `script_descriptors.json`, and latest scripts (`train_florence2.py`,
  `predict_with_conf.py`, `fusion_analysis.py`, `build_grapheme_vocab_lang.py`,
  `dataset.py`, `metrics.py`, `measure_script_descriptors.py`).
- **Rewrote 50,346 image paths** in the server3 split copies
  (`/home/ujjwal/ritu1` → `/c/ujjwalb/ritu1`); spot-check 0 missing.
- All 12 BSTD language image folders present (train+test) on server3.
- Smoke test (tamil `--check_data_only`): 1620/180/513 train/val/test, 0 missing. ✅

**Launcher:** `run_law_training_server3.sh` (server3-path + `ritu_scenetext` python +
`HF_HOME=/c/ujjwalb/.cache/huggingface`; otherwise identical logic to
`run_law_training.sh`). **Dravidian-first order** (tamil telugu kannada malayalam,
then the other 8) so Finding-2's sharp prediction is tested first.

**Running now:** `nohup bash run_law_training_server3.sh tamil telugu kannada malayalam`
→ `law_run_dravidian.log` on server3 (launched 2026-06-18 ~13:22 IST).
- To resume monitoring in a new session:
  `ssh ujjwalb@192.168.57.100 'cd /c/ujjwalb/ritu1/lstm_model; tail law_run_dravidian.log; ls -d checkpoints_law_*'`
  (key-based SSH from this box works; A5000 host = cvpr-gamma, 192.168.57.100).
- After the 4 Dravidian finish + look sane: launch the remaining 8 with
  `nohup bash run_law_training_server3.sh > law_run_rest.log 2>&1 &` (resumable; skips done).

### Next decisions / actions
- Monitor Dravidian runs; confirm Finding 2 (largest grapheme wins at high fertility).
- Then B5/B6: fit the law on all points → law figure/table.
- Parallel no-GPU work available: descriptor horse-race (add STRR + alternatives to
  `measure_script_descriptors.py`) per `PREREGISTRATION.md`.

---

## 10. Glossary (for quick reference)
- **WRR** — Word Recognition Rate: % of words read 100% correctly.
- **CER** — Character Error Rate: lower is better.
- **Grapheme cluster** — one visual letter-unit in Indic scripts (base + signs/conjuncts).
- **BPE / subword** — the default way LLM tokenizers chop text; built mostly for English.
- **Fertility** — how many BPE pieces it takes to write one grapheme cluster.
- **Fusion** — run two model branches, keep the more-confident answer.
- **Abugida** — a writing system where consonants carry an inherent vowel (all Brahmic scripts).
- **Zero-shot** — reading a script with no labeled real training images of it.
- **LoRA** — cheap fine-tuning: train tiny adapters, keep the big model frozen.

---

## 11. Literature stress-test (done 2026-06-18 — positioning + threats)

Verified against the live literature so reviewers can't surprise us. Each entry =
what it is, and how OUR claim survives it.

| Work | What it does | Why we are still novel / the threat |
|---|---|---|
| **MGP-STR** (IJCV, Jan 2026; arXiv 2307.13244 / 2209.03592) | Learnable fusion of char/BPE/WordPiece **inside one Latin ViT**. | Closest prior art to *fusion*. **Threat: fusion is no longer a headline.** We keep it as a component; novelty is the LAW + zero-shot + cross-script, none of which MGP-STR has. Must cite prominently and differentiate in the intro. |
| **GraDeT-HTR** (EMNLP 2025) | Grapheme tokenizer + decoder-only transformer, **Bengali handwritten**. | Single script/lang, no VLM, no scene text, no law, no transfer. |
| **Beyond Fertility / STRR** (arXiv 2510.09947, 2025) | Argues **fertility is a weak predictor** of LLM downstream perf; proposes STRR + morphological metrics. | **Biggest threat to Part ①.** Mitigation: our dependent variable is OCR-specific (granularity-advantage sign + fusion gain), NOT generic LLM quality; and we *horse-race* fertility against STRR/conjunct-density/entropy rather than betting on fertility. Pre-registered in `PREREGISTRATION.md`. |
| **IndicSuperTokenizer** (arXiv 2511.03237, 2025); **BrahmicTokenizer-131K** (arXiv 2605.29379, 2026) | SOTA Indic LLM tokenizers (optimize fertility). | Orthogonal: they *build* a tokenizer; we study *when granularity matters for recognition* and *predict it*. Cite as evidence the topic is hot + as alternative descriptors. |
| **TransliCo** (arXiv 2401.06620) + transliteration zero-shot transfer | Shared/transliterated space for cross-lingual NLP transfer. | Zero-shot cross-script is **not virgin in NLP**. Our novelty is precise: first **zero-real-image** transfer for **scene-text VLMs** via abugida primitives, with transfer **predicted by the law**. |
| **Zero-shot OCR Sinhala/Tamil** (arXiv 2507.18264); **Nayana** VLM OCR | Low-resource / zero-shot OCR exists and is active. | Confirms the problem matters; none predicts transfer from a script descriptor or ties it to a granularity law. |
| **BSTD** (arXiv 2511.23071, Nov 2025; PARSeq ~73% avg WRR) | The benchmark we use; absolute SOTA holder (8.48M synth pretrain). | We do NOT chase per-script SOTA (their data scale wins). Our axis is orthogonal: predictability + zero-data transfer. |

**Net:** the unified ①+② thesis is novel and defensible IF (a) MGP-STR is cited and
out-positioned, (b) the law is presented as a *horse-raced predictive relationship*
with honest CIs (not "fertility, the one true predictor"), and (c) zero-shot is framed
as zero-real-image VLM transfer predicted by the law. All three are in the plan.
</content>
</invoke>
