# RESEARCH STATUS & PATH TO A TOP-TIER PAPER
**Project:** A Predictive Law of Tokenization Granularity for Brahmic Scripts + Zero-Shot
Cross-Script Scene-Text Recognition.
**Owner:** Ritu Baskey · **This doc created:** 2026-06-25 · **Machine:** server3 (`cvpr-gamma`, RTX A5000 24 GB)

> **NEW SESSION: READ `ZEROSHOT_LOSO_LIVE_LOG.md` FIRST** (live run state — updated
> every session), then this file, then `GROUNDBREAKING_RESEARCH_PLAN.md` (full history) and
> `PREREGISTRATION.md` (frozen analysis plan **+ §8 amendments of 2026-07-02**). Every claim
> below was checked against files on disk on 2026-06-25; run-state facts here are SUPERSEDED
> by the live log.
>
> **2026-07-02 additions** (details in the live log §9 and `PREREGISTRATION.md` §8):
> `PROSPECTIVE_PREDICTIONS_H3.md` (timestamped rank predictions for the 7 unfinished Rung-B
> scripts, commit `047c30c`; fertility-P1 vs coverage-P2, malayalam decisive) ·
> Rung **Bbpe** BPE-baseline phase appended to `run_zeroshot_loso.sh` (27 rungs total now) ·
> `compute_visual_similarity.py` → `visual_similarity_descriptors.json` (visual-similarity
> control for the H3 horse-race) · orchestrator resume bug fixed (`.train_done` sentinel) ·
> `COMPETITIVE_POSITIONING_AND_LITERATURE.md` + `PUBLICATION_STRATEGY.md` for positioning/venues.

---

## 0. THE ONE-SENTENCE PAPER (the spine — do not lose this)
> *We can read a Brahmic script for which we have ZERO labeled real images, and we can
> predict in advance — from one measurable property of the writing system (fertility /
> abugida structure) — which unseen scripts that transfer works best for.*

The genuinely hard-to-preempt asset is **predict-then-transfer to an unseen *script***.
The "law" is the *explanation*; the zero-shot capability is the *wow*. Fusion is only a
component (MGP-STR already owns fusion). **Frame the paper around Part ②, explained by Part ①.**

---

## 1. BRUTAL HONEST ASSESSMENT (done 2026-06-25 via live literature search)
**Current strength = solid, publishable (Findings / workshop / Pattern Recognition / IJDAR).
NOT yet "groundbreaking / internationally-known."** Reason: every pillar is individually
precedented; novelty is in the *combination + the Brahmic-STR instantiation*, and the
empirics are thin where reviewers attack. To clear the top bar we must make the zero-shot
capability *strong across many scripts* and fix N=9. See §2 (precedents) and §3 (path).

### Verified precedents a reviewer WILL cite (with our differentiation)
| Threat (real, verified) | What it is | How we survive / differentiate |
|---|---|---|
| **MGP-STR** (IJCV Jan 2026; arXiv 2307.13244) | Learnable fusion of char/BPE/WordPiece in one Latin ViT | Fusion is NOT our headline; it's a component. We add the *law* + *cross-script zero-shot*, neither of which MGP-STR has. Must cite up front. |
| **GraDeT-HTR** (EMNLP 2025; arXiv 2509.18081) | Grapheme tokenizer beats BPE, **Bengali** handwritten | "Grapheme>BPE for Indic" is already known. Our novelty = a *predictive rule for WHEN it flips* across 9 scripts + zero-shot. Single-script, no law, no transfer. |
| **BnGraphemizer / Grapheme Pair Encoding** | Grapheme units beat byte-BPE on Tamil/Sinhala/Hindi | Same as above — direction known; prediction + transfer is ours. |
| **Beyond Fertility / STRR** (arXiv 2510.09947) | Argues fertility is a WEAK predictor; proposes STRR | Biggest threat to Part ①. Mitigation already in: our DV is OCR-specific (granularity-advantage sign + fusion gain), and we **horse-race** fertility vs STRR/entropy/conjunct-density. NB: STRR is **degenerate (=0) for every Indic script** in our data — itself a finding. |
| **Zero-shot Chinese via stroke/radical decomposition** (1) Stroke-level arXiv 2106.11613, (2) STAR 2210.08490, (3) LERRNet 2025, (4) HierCode 2024 | Recognize UNSEEN Chinese characters by decomposing into shared sub-units | **Closest conceptual prior art to Part ②.** Differentiation: they transfer to unseen *characters within ONE script*; we transfer to an entirely unseen ***script*** (different Unicode block, different glyphs) in a scene-text VLM, AND predict which scripts transfer. Must cite + position explicitly. |
| **Distance predicts transfer** (lang/embedding similarity → transfer; many papers) | A measurable property predicting cross-lingual transfer is well established in NLP | Our H3 ("descriptor predicts transfer") is a *known principle* applied to a new place (OCR/STR, script-level). Frame as instantiation, not as discovering the principle. |
| **BSTD** (arXiv 2511.23071, Nov 2025; PARSeq ~73% avg WRR, up to 92% English) | The benchmark we use; SOTA holder via 8.48M synth pretrain | We do NOT chase per-script SOTA. Orthogonal axis: predictability + zero-real-image transfer. |

**Net:** the unified ①+② thesis is novel & defensible IF (a) MGP-STR + the Chinese
stroke/radical line are cited and out-positioned, (b) the law is a *horse-raced predictive
relationship* with honest CIs (never "the one true predictor"), (c) zero-shot is framed as
zero-real-image cross-SCRIPT VLM transfer *predicted by the law*, and (d) we deliver strong,
multi-script zero-shot numbers (not one script at ~10%).

---

## 2. WHERE WE ARE RIGHT NOW (verified on disk 2026-06-25)

### ✅ Part ① — The Law: COMPLETE & SUPPORTED
- All 12 b1800 runs done. Fit: `law_fit_results_brahmic.json` (N=11: 9 Brahmic + Assamese/Bengali both Bengali-script counted; English=Latin control excluded). Figure: `figures/law_main.png/.pdf`.
- **Polarity 8/9 Brahmic correct** (only Gujarati misses, −0.39 near-tie). Per-lang LOSO **90.9%**. Fertility WINS the horse-race (LOSO 90.9% vs ≤81.8% all competitors; STRR degenerate).
- **fusion_gain R²=0.698**, slope CI [+1.05,+2.67] excl. 0. **grapheme_advantage R²=0.678.** Both pre-registered §6 criteria MET.
- Honest nuances to keep in paper: Gujarati near-tie miss (lowest conjunct_density 0.097); English control also slightly favored grapheme (+1.8); Odia +14.3 is a strong-but-above-line outlier (monotonic, not perfectly linear).

### 🟢 Part ② — Zero-shot: PILOT WORKED, FULL LOSO NOW RUNNING
- **Pilot (held-out Tamil, Dravidian sources):** Rung A WRR=0.00, **Rung B WRR=9.94** (vs Florence-2 raw zero-shot 0.0; supervised Tamil std 28.07 / grph 36.84). Direction validated.
- **FULL 9-script leave-one-SCRIPT-out LAUNCHED 2026-06-25 11:27 IST** → `zeroshot_loso.log` (PID 1283525 at launch; resumable). This produces the **H3 unifying result**.

---

## 3. THE PATH TO TOP-TIER (priority order — this is the plan)
1. **[RUNNING] Full 9-script zero-shot LOSO** → the unifying figure: *fertility/structure
   predicts which unseen scripts transfer best* (H3). This single experiment decides
   top-tier vs Findings. Collect `result_zs_loso_rung{A,B}_*.json` when done.
2. **Rung C (few-shot slope):** add 50–100 real target words per script → show the
   transfer slope from 0 real images upward. Cheap, strong for reviewers.
3. **Fix N=9 (the law's fatal weakness):** 9 points cannot carry the word "law". Add
   synthetic-only Brahmic/SE-Asian scripts (Sinhala, Thai, Lao, Khmer, Myanmar, Tibetan)
   to push N≈15–20, plus large-N **per-sample** disagreement stats + bootstrap CIs.
   (Pre-register the extension in `PREREGISTRATION.md` §8 deviation log BEFORE fitting.)
4. **B7 uncapped headline runs** (owner-approved): fully-converged per-script models for a
   strong absolute-WRR table, kept SEPARATE from the capped law models (two labeled purposes).
5. **Ship an artifact people adopt:** release the shared-abugida-pivot space + the
   zero-real-image Indic STR benchmark protocol + code. Benchmarks get cited regardless of numbers.
6. **Manuscript (Phase D):** reframe spine on Part ② explained by Part ①; cite/out-position
   MGP-STR + Chinese stroke/radical line; horse-race framing for the law; all numbers → result files.
- **Venue realism:** current → Findings/workshop/PR/IJDAR. With #1+#2+#3 strong → real shot
  at CVPR/ICCV/IJCV or ACL/EMNLP main. "International" comes from strong paper + adopted
  benchmark/code, rarely one result alone.
- **Falsification honesty (keep us un-rejectable):** if zero-shot stays ~10% across scripts,
  REPORT it and reframe as "structural output-transfer is real but visually bottlenecked;
  synthetic rendering is necessary" — still publishable, weaker. Do not inflate.

---

## 4. VERIFIED ENVIRONMENT FACTS (so a new session doesn't re-derive / repeat mistakes)
- **We are ON server3** now (`cvpr-gamma`), working dir `/c/ujjwalb/ritu1/lstm_model`, `HOME=/c/ujjwalb`.
- **Python env:** `/c/ujjwalb/.conda/envs/ritu_scenetext/bin/python` (3.10, torch 2.5.1+cu121, transformers 4.44.2, peft 0.13.0). `export HF_HOME=/c/ujjwalb/.cache/huggingface`.
- **GPU:** RTX A5000, 24 GB. Shared with an UNRELATED job `python train.py` (PID 551517, user=ujjwalb, dir `Vansh/multihop_memory_vqa`) using ~4.6 GB — **NOT ours, do not touch.**
- **⚠️ STALE PATHS:** all `florence2_splits_bstd_*_b1800.json` store `/home/ujjwal/ritu1/...`
  image paths that **do NOT resolve here** (that dir is gone). Real images live locally at
  `benchmarks/bstd/Recognition/` (106,461 files). `prepare_zeroshot_loso.py` rewrites paths
  to local (`localize()`); `train_florence2.py` also re-resolves on load. If you write new
  splits, anchor paths at `benchmarks/...`.
- **⚠️ FONTS:** NO Noto Indic fonts here. The pilot's `resolve_font('Noto Sans X')` would
  silently fall back to **DejaVu = tofu boxes**. `prepare_zeroshot_loso.py` fixes this via
  `fc-match/fc-list :lang=<code>` + multi-font + per-image blank verification. libraqm
  (complex shaping) IS available (verified: Hindi conjuncts, Odia, Punjabi, Malayalam,
  Gujarati, Bengali, Odia synth all render correct glyphs). Per-script fonts found: Tamil
  (Lohit/Samyak), Telugu (Lohit/Pothana/Vemana), Kannada (Lohit/Navilu/Gubbi), Malayalam
  (Rachana/Manjari/...), Bengali (Mukti/Likhan/Lohit), Devanagari (Gargi/sahadeva/kalimati/...),
  Gujarati (Rasa/padmaa/Samyak/Lohit), Gurmukhi (Lohit/Saab/...), Oriya (utkal/Lohit/FreeSerif).
- **Grapheme-token injection** prints "Injecting Bengali grapheme tokens..." for EVERY
  language — cosmetic hardcoded string in `train_florence2.py:268`, harmless (verified
  2026-06-21; correct per-language vocab is loaded).

---

## 5. NEW FILES THIS SESSION (2026-06-25)
- **`prepare_zeroshot_loso.py`** — full leave-one-SCRIPT-out data builder. Font-fixed,
  path-fixed, balanced sources (700 train / 90 val per source lang, seed 42), multi-language
  targets per script, synth render verification. `--script X` | `--all`.
- **`run_zeroshot_loso.sh`** — 9-script train+eval orchestrator (Rung A then B), resumable,
  GPU-aware. `nohup bash run_zeroshot_loso.sh > zeroshot_loso.log 2>&1 &`.
- **Data generated for all 9 scripts** (verified, 0% blank synth):
  `splits_zeroshot_loso_rung{A,B}_<tag>.json`, `grapheme_vocab_zeroshot_loso_rung{A,B}_<tag>.json`,
  `synth_zeroshot_loso_<tag>/`, `zeroshot_loso_meta_<tag>.json`.
  tag = script.lower(): bengali devanagari gujarati gurmukhi kannada malayalam oriya tamil telugu.

### Pre-GPU coverage signal (source-vocab covers held-out script's test tokens, ZERO training)
| Script | tok-cov% | type-cov% | test imgs | synth |
|---|---|---|---|---|
| Gujarati | 97.3 | 84.1 | 1015 | 3240 |
| Devanagari | 94.3 | 62.9 | 6042 | 6480 |
| Oriya | 92.6 | 77.6 | 1044 | 3240 |
| Telugu | 92.3 | 83.7 | 545 | 3240 |
| Kannada | 90.8 | 78.3 | 720 | 3240 |
| Gurmukhi | 90.7 | 60.1 | 2879 | 3240 |
| Tamil | 88.8 | 66.5 | 513 | 3240 |
| Bengali | 87.3 | 69.2 | 2873 | 6480 |
| Malayalam | 79.2 | 69.1 | 547 | 3240 |

---

## 6. HOW TO RESUME / MONITOR (exact commands)
```bash
cd /c/ujjwalb/ritu1/lstm_model
# is the LOSO chain alive?
pgrep -af run_zeroshot_loso ; tail -30 zeroshot_loso.log
# results so far:
grep "RESULT LOSO" zeroshot_loso.log ; cat result_zs_loso_rung*_*.json 2>/dev/null
# if it died, just relaunch (resumable, skips finished rungs):
nohup bash run_zeroshot_loso.sh > zeroshot_loso.log 2>&1 &
# regenerate any dataset (no GPU):
/c/ujjwalb/.conda/envs/ritu_scenetext/bin/python prepare_zeroshot_loso.py --script Tamil
```
When all 18 rungs (9 scripts × A/B) are done: build the **H3 unifying figure** — scatter
held-out-script fertility (from `script_descriptors.json`) vs Rung B zero-shot WRR
(`result_zs_loso_rungB_*.json`), report Spearman ρ (PREREG §5 H3). That figure + the law
figure = the paper's two money shots.

## 7. IMMEDIATE NEXT STEPS (in order)
1. Wait for / monitor the LOSO chain; collect `result_zs_loso_*` (≈ several days, resumable).
2. Build H3 unifying figure + Spearman(fertility, zero-shot WRR). Expect higher-fertility/
   higher-coverage scripts to transfer better (Gujarati/Devanagari/Telugu vs Malayalam).
3. Rung C few-shot (50–100 real words/script) → slope.
4. Pre-register + run the N-expansion synthetic scripts (Sinhala/Thai/Lao/Khmer/Myanmar/Tibetan).
5. B7 uncapped headline runs.
6. Manuscript (Phase D) with the reframed spine + full citation positioning from §1.
