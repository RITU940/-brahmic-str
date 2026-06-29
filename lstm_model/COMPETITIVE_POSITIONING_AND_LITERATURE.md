# COMPETITIVE POSITIONING & LITERATURE (live web research)
**Project:** A Predictive Law of Tokenization Granularity for Brahmic Scripts + Zero-Shot
Cross-Script Scene-Text Recognition.
**Owner:** Ritu Baskey · **Doc created:** 2026-06-29 (web research done same day) · **Machine:** server3 (`cvpr-gamma`)

> Companion to `RESEARCH_STATUS_AND_PATH.md` (where-we-are) and `GROUNDBREAKING_RESEARCH_PLAN.md`
> (full history). THIS file = the verified competitive landscape + the exact rebuttals to use
> in Related Work and in the rebuttal phase. Every precedent below was checked on the live web
> on 2026-06-29 (arXiv/venue confirmed). Use this when writing §Related Work and §Limitations.

---

## 0. ONE-PARAGRAPH VERDICT (the honest read after the search)
The **combination** we are building — (a) a *predictive law* (one measurable script property
predicts grapheme-vs-BPE advantage) + (b) *zero-real-image transfer to an entirely unseen
**script*** (different Unicode block) + (c) the *Brahmic* instantiation — **does not exist in
the literature.** Every individual pillar is precedented; the unification is not. The niche is
**open**. Risk has shifted away from "is it novel" (it is) toward **framing/positioning** — one
2023 cross-lingual-STR paper directly challenges the H3 premise and must be out-positioned, and
our absolute zero-shot numbers must be framed as a *different axis* than supervised STR, not a
capability race. The remaining scientific unknown (unchanged) is whether the **H3 correlation**
comes out strong once 5–6 Rung-B scripts finish.

---

## 1. LIVE STATUS SNAPSHOT (so this doc is self-contained) — verified on disk 2026-06-29
### Part ① The Law — DONE (`law_fit_results_brahmic.json`, N=11)
- `grapheme_advantage` vs fertility: **R²=0.678**, slope 6.41, **95% CI [3.37, 9.21]** (excludes 0).
- `fusion_gain` vs fertility: **R²=0.698**, slope 1.80, **95% CI [1.05, 2.67]** (excludes 0).
- Polarity **8/9** correct (only Gujarati misses, −0.39 near-tie).
- Horse race (per-script LOSO direction-accuracy): **fertility 90.9%** vs all competitors ≤81.8%
  (conjunct_density 81.8 / spearman 0.77; clusters_per_word 81.8 / 0.77; grapheme_entropy 81.8 / 0.05;
  bytes_per_cluster 72.7; bpe_frag 36.4; **STRR degenerate = 0 for every Indic script**).
### Part ② Zero-shot LOSO — RUNNING (PID 1283525 at last check), 18 rungs (9 scripts × A/B)
- **Tamil:** Rung A WRR 0.0 (CharAcc 4.8) · **Rung B WRR 9.16** (CharAcc 39.0, CER 61.0) ✅
- **Telugu:** Rung A WRR 1.28 (CharAcc 23.32) · Rung B training (epoch ~11/15 at last check)
- Remaining 7 scripts pending. ~1.3 days/script ⇒ full set ~mid-July 2026.
- Baseline anchors: Florence-2 raw zero-shot = 0.0; supervised Tamil (our b1800) = 28.07 BPE / 36.84 grph.

---

## 2. THE VERIFIED COMPETITIVE LANDSCAPE (cite + out-position each)

### 2.1 Direct precedents (in current status doc — all CONFIRMED real on 2026-06-29)
| Work | Venue / id (verified) | What it actually does | Our differentiation (say this) |
|---|---|---|---|
| **MGP-STR** | IJCV; arXiv **2307.13244** (~94% on Latin benchmarks) | Learnable fusion of char/BPE/WordPiece in one Latin ViT | Fusion is a *component*, not our headline. We add the *law* + *cross-script zero-shot*. Cite up front. |
| **GraDeT-HTR** | EMNLP 2025; arXiv **2509.18081** | Grapheme tokenizer beats BPE, **Bengali handwritten**, decoder-only | "Grapheme>BPE for one Indic script" already known. Ours = *predictive rule for WHEN it flips across 9 scripts* + transfer. Single-script, no law, no transfer. |
| **BnGraphemizer** | (Bengali HTR) | Trie-based grapheme tokenizer, Bengali HTR | Same direction; we add prediction + transfer. |
| **Beyond Fertility / STRR** | **NeurIPS 2025 *Workshop*** (Eval LLM Lifecycle); arXiv **2510.09947**; code github.com/tafseer-nayeem/STRR | Argues fertility is a weak tokenizer metric **for LLMs**; proposes STRR (single-token retention rate). 6 tokenizers × 7 langs. | **Threat is softer than feared:** it's a *workshop* paper in the *LLM-tokenizer-fairness* domain, NOT OCR. Fertility-as-OCR-output-predictor is under-explored. We also *horse-race* fertility vs STRR and find **STRR degenerate (=0) for all Indic** — itself a finding. |
| **Zero-shot Chinese via stroke/radical** | (1) arXiv 2106.11613, (2) STAR 2210.08490, (3) LERRNet 2025, (4) **HierCode** arXiv 2403.13761 | Recognize UNSEEN Chinese *characters* by decomposing into shared sub-units | Closest conceptual prior art to Part ②. They transfer to unseen *characters within ONE script*; we transfer to an entirely unseen ***script*** (diff Unicode block, diff glyphs) in a scene-text VLM, AND predict which scripts transfer. Cite + position explicitly. |
| **BSTD (Bharat Scene Text)** | arXiv **2511.23071** (Nov 2025); 6,582 imgs / 126,292 words / 11 Indian langs + English; github Bhashini-IITJ | The benchmark we use. PARSeq synth-only ≈47% avg WRR (32–92); fine-tuned ≈73% (56–92). | We do NOT chase per-script SOTA. Orthogonal axis: predictability + zero-real-image transfer. Fresh benchmark = good timing. |

### 2.2 NEW work surfaced by the search (NOT yet in status doc — ADD THESE)
| Work | Venue / id (verified) | What it does | Why it matters to us |
|---|---|---|---|
| **Cross-Lingual Learning in Multilingual STR** | arXiv **2312.10806** | Studies *what predicts cross-lingual transfer in STR*. Concludes: (1) **dataset size of the high-resource lang matters MORE than typological similarity**; (2) for STR, **visual/appearance similarity > linguistic typology**. | ⚠️ **THE #1 THREAT TO H3.** Directly says "typology doesn't predict STR transfer; data size does." MUST cite & out-position (see §3). Single most important finding of the search. |
| **MOoSE** (Multi-Orientation Sharing Experts) | ICDAR 2024; arXiv **2407.18616**; github lancercat/moose | **Open-set STR**: recognizes UNSEEN *characters* (e.g. Japanese kana) across orientations. Recognition, not detection. | Same family as the Chinese line: unseen *characters within a script*, not unseen *script*. Cite, differentiate, do not fear. |
| **Separate Scene Text Detector for Unseen Scripts** | arXiv **2307.15991** | Zero-shot **text DETECTION** for unseen scripts via stroke embeddings. | Sounds competitive but it's **detection (localizing), not recognition (reading)**. Clears a lane — mention to pre-empt, no threat. |
| **Open-set Text Recognition via Character-Context Decoupling / Label-to-Prototype** | arXiv 2204.05535 / 2203.05179 | Earlier open-set recognition baselines (unseen characters). | Background for the open-set recognition lineage; cite briefly. |

---

## 3. THE #1 THREAT IN DETAIL + OUR REBUTTAL (arXiv 2312.10806)
**Their claim (will be quoted against us):** "In multilingual STR, cross-lingual transfer is driven
by *dataset size of the high-resource language* and by *visual/appearance similarity*, NOT by
linguistic typology." ⇒ A reviewer says: *"H3 is dead on arrival — typology doesn't predict STR
transfer; this 2023 paper already showed data size does."*

**Our three-part rebuttal (deploy ALL of them, in Related Work AND Limitations):**
1. **Fertility ≠ linguistic typology.** Our predictor is a **script-structural / output-granularity**
   property of the *writing system* (how many grapheme units a word packs), not word-order/family
   typology. Their negative result is about *typology*; it does not bind a *structural* predictor.
2. **We control for the confound they identified.** Their driver is *data size*. Our LOSO pipeline
   uses **balanced sources (700 train / 90 val per source language, seed 42)** — data size is held
   ~constant across source languages, so we isolate the variable they said was confounded.
3. **Different regime.** They study transfer *with real labeled target data* among scripts. We study
   **zero *real* images of the target**, transferring across a **different Unicode block** via shared
   abugida output structure + synthetic rendering. Their setting ≠ ours.
**Net:** Cited & out-positioned ⇒ this becomes a *strength* ("we answered the obvious objection").
Ignored ⇒ rejection. **This paragraph must exist in the manuscript.**

---

## 4. ABSOLUTE-NUMBERS REALITY CHECK (for §Experiments framing + the prof meeting)
On the SAME benchmark (BSTD), **supervised** PARSeq: ≈47% avg WRR synth-only, ≈73% fine-tuned;
low-resource Telugu/Malayalam ≈50% even supervised. Our zero-real-image Tamil = **9.16%**.
- **Do NOT** let comparisons sit on equal footing — they use real labeled target training data; we use **zero**.
- **Frame:** ours is a *different axis* (predictability + zero-real-image transfer), proof-of-mechanism
  (9.16 vs Florence-2 raw 0.0, CharAcc 39%), not a capability race. State this explicitly before any table.
- Telugu Rung A already at 23% CharAcc is an encouraging sign the floor is not stuck at ~10%.

---

## 5. NOVELTY MAP (what is ours vs known) — one-glance
- KNOWN: grapheme>BPE for Indic (GraDeT, BnGraphemizer) · learnable granularity fusion (MGP-STR) ·
  zero-shot to unseen *characters* (Chinese stroke/radical, MOoSE) · "a property predicts transfer"
  as an NLP principle · zero-shot unseen-script *detection* (2307.15991) · data-size/visual-similarity
  drive STR transfer (2312.10806).
- **OURS (unclaimed):** a **predictive, horse-raced law** for *when* grapheme beats BPE *across scripts*
  (sign + magnitude) **⊕** zero-**real-image** transfer to an unseen **script** (Unicode-block-level)
  in a scene-text VLM **⊕** **the law predicts which unseen scripts transfer best (H3)** **⊕** Brahmic.

---

## 6. ACTION ITEMS THIS RESEARCH CREATES (fold into the plan)
1. **[Writing — mandatory]** Add a Related-Work paragraph citing & out-positioning **2312.10806**
   with the 3-part rebuttal (§3). Highest priority; a top-venue reviewer WILL raise it.
2. **[Writing]** Add **MOoSE (2407.18616)** + the open-set recognition lineage (2204.05535, 2203.05179)
   to the "unseen units" paragraph alongside the Chinese stroke/radical line; differentiate = unseen
   *script* not unseen *character*.
3. **[Writing]** Add **2307.15991** as "detection ≠ recognition" pre-emption (one sentence).
4. **[Framing]** Put the §4 "different axis, not a capability race" sentence ahead of every zero-shot table.
5. **[Framing]** Use the "Beyond Fertility is a *workshop/LLM* paper, not OCR" point to claim the
   fertility-as-OCR-predictor white space explicitly.
6. **[Unchanged, still decisive]** Finish LOSO → measure **H3 Spearman(fertility, Rung-B WRR)** at
   5–6 scripts. The literature confirms the spot is right; whether there's gold is still in the GPU.
7. **[Unchanged]** N-expansion (Sinhala/Thai/Lao/Khmer/Myanmar/Tibetan) to fix N=9 — pre-register first.

---

## 7. SOURCES (verified 2026-06-29)
- MGP-STR — https://arxiv.org/abs/2307.13244
- GraDeT-HTR — https://arxiv.org/html/2509.18081
- Beyond Fertility / STRR (NeurIPS 2025 Workshop) — https://arxiv.org/abs/2510.09947 · code https://github.com/tafseer-nayeem/STRR
- Cross-Lingual Learning in Multilingual STR (#1 threat) — https://arxiv.org/html/2312.10806v1
- MOoSE open-set STR — https://arxiv.org/abs/2407.18616 · code https://github.com/lancercat/moose
- Separate Detector for Unseen Scripts (detection only) — https://arxiv.org/abs/2307.15991
- Open-set recognition baselines — https://arxiv.org/pdf/2204.05535 · https://arxiv.org/pdf/2203.05179
- Bharat Scene Text Dataset (BSTD) — https://arxiv.org/abs/2511.23071 · code https://github.com/Bhashini-IITJ/BharatSceneTextDataset
- HierCode zero-shot Chinese — https://arxiv.org/pdf/2403.13761
