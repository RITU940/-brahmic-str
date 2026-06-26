# PUBLICATION STRATEGY — Brahmic-STR Tokenization Law + Zero-Shot Cross-Script
**Researched:** 2026-06-26 (live venue data) · **Owner:** Ritu Baskey · First author = Ritu.
**Companion docs:** `RESEARCH_STATUS_AND_PATH.md` (science status), `ZEROSHOT_LOSO_LIVE_LOG.md`
(the running experiment), `PREREGISTRATION.md`. **Rule that governs everything:** honest results only.

---

## 0. THE HONEST ONE-PARAGRAPH VERDICT
You actually have **two papers**, at two maturity levels:
- **Paper A — Bengali grapheme-fusion STR** (mostly done; draft `paper/main.tex` exists). Solid,
  honest, replicated on a public benchmark. **Will get accepted at a topical venue** (ICDAR/IJDAR/
  workshop/Findings). Not CVPR-main material on its own (fusion is pre-empted by MGP-STR).
- **Paper B — The tokenization-granularity *law* + zero-shot cross-script transfer** (the
  "groundbreaking" one; the LOSO run feeds this). Its ceiling is decided by the experiments
  **running right now.** If multi-script zero-shot is strong + H3 correlation holds + N is expanded
  → real shot at **WACV / CVPR / ACL-EMNLP main**. If zero-shot stays ~10% and N=9 → **Pattern
  Recognition / IJDAR / Findings / workshop** tier (still a real, citable publication).

**Will it get accepted?** Yes — *somewhere good* is highly likely because the work is honest and
on a real problem. *Top-tier (CVPR/ICCV/IJCV/ACL-main)* is **conditional** on the running results
being strong and on fixing N=9. Nobody can promise a top-tier accept in advance; the levers that
move the odds are concrete and listed in §4.

---

## 1. CALENDAR — VENUES THAT FIT YOUR TIMELINE (as of 2026-06-26)
Run finishes ≈ **July 4–5** (9-day LOSO). Add Rung C + H3 + writing. So the feasible windows:

| Venue | Type / fit | Submission deadline | Notify | Accept rate | Verdict for you |
|---|---|---|---|---|---|
| **WACV 2027** (Orlando, Jan 4–8) | CV conf, CORE-A; apps-friendly | **Round 2: Aug 28 '26** (reg Aug 21) | Oct 9 '26 | ~37.8% (2026) | **★ Primary conference target.** Realistic, reputable, fast. |
| **arXiv** | preprint, priority | anytime | n/a | n/a | **★ Do first** (~mid-July) for priority + visibility. |
| **Pattern Recognition** (Elsevier) | Q1 journal, IF **7.6** | rolling (no deadline) | months | — | **★ Strong journal home;** lets you include N-expansion. |
| **IJDAR** (Springer/IAPR) | document-analysis journal, IF 2.5 | rolling | **median 7 days to 1st decision** | — | Topical home, fastest decision; lower IF. |
| **CVPR 2027** (June) | CV flagship | abstract **Nov 15 '26**, paper ~Nov 22 | Feb '27 | ~22–25% | Stretch upgrade *only if* WACV-level results exceed expectations. |
| **ICDAR 2027** (Kuala Lumpur, Aug 18–22) | IAPR document-analysis flagship; biennial | ~Feb–Mar '27 (TBA) | mid-'27 | competitive/topical | Best *topical* fit; has an **IJDAR journal track**. Far out. |
| **AACL-IJCNLP 2026** (Hengqin, Nov) | NLP; tokenization angle | direct **Jul 15 '26** (ARR May 25 passed) | Sep 7 '26 | — | Too rushed (3 wks); skip unless Paper-B framing is NLP-first. |
| **AAAI 2027** (Montréal, Feb) | broad AI | abstract Jul 20 / paper **Jul 27 '26** | — | ~23% | Too soon + too broad; skip. |
| **EMNLP 2026** | NLP main | **PASSED** (May 25) | — | ~20% | Missed; aim EMNLP/ACL 2027 if NLP route. |
| **NeurIPS 2026 Eval&Datasets** | benchmark track | **PASSED** (May 6) | — | — | Missed; ICLR 2027 D&B is the next benchmark slot. |

---

## 2. RECOMMENDED PLAN (two-track, standard & low-risk)
**Track 1 — get priority + a sure publication moving (now → August):**
1. **arXiv preprint** the moment the 9-script LOSO + H3 figure exist (~mid-July). Free, immediate,
   international, establishes priority over the MGP-STR / GraDeT-HTR / Chinese-zero-shot line.
2. **Paper A (Bengali fusion)** → submit to **IJDAR** (7-day first decision, topical) or hold for
   **ICDAR 2027**. This bankrolls a guaranteed publication while Paper B matures.

**Track 2 — aim the big paper as high as the results honestly allow:**
3. Target **WACV 2027 Round 2 (Aug 28 '26)** for **Paper B** *if* by mid-August you have: full
   9-script zero-shot + H3 (Spearman) + Rung C slope. WACV is the sweet spot — CORE-A prestige,
   ~38% accept, decision by Oct, in-person Jan 2027.
4. If results slip or you want the complete story (incl. N-expansion to ~15–20 scripts) →
   **Pattern Recognition** (IF 7.6, no deadline) instead. Same paper, no rush, higher completeness.
5. Keep **CVPR 2027 (Nov 15)** as a stretch upgrade only if WACV-grade results clearly over-deliver.

**Do NOT** dual-submit the same paper to two peer-reviewed venues at once (desk-reject). arXiv is
fine alongside any of them. Pick one peer-reviewed home at a time.

---

## 3. WILL IT GET ACCEPTED? — HONEST ODDS (conditional on results)
Bands assume the paper is well-written, honestly reported, and positions the precedents (§5).

| Scenario after experiments | arXiv | WACV / ICDAR / IJDAR / PR | CVPR-ICCV / IJCV / ACL-main |
|---|---|---|---|
| Zero-shot strong (multi-script, clean H3 ρ) + N expanded to ~15–20 | certain | **high** | **plausible (~coin-flip at top CV)** |
| Zero-shot modest (~10%) but H3 ρ clearly positive, N=9 | certain | **moderate–high** | low–moderate |
| Zero-shot ~10%, H3 weak/noisy | certain | **moderate** (reframe as honest negative + benchmark) | low |

The single biggest swing factor is **H3**: a clean "fertility predicts which unseen scripts
transfer" correlation is the headline that lifts this from "nice application" to "predictive
science." Absolute WRR matters far less than that correlation.

---

## 4. WHAT'S REQUIRED TO BE SUBMISSION-READY (the checklist)
**Experiments (status):**
- [ ] Full 9-script zero-shot LOSO, Rungs A+B — *running now* (1/18 done). [`ZEROSHOT_LOSO_LIVE_LOG.md`]
- [ ] **H3 figure** + Spearman(fertility, RungB WRR) — the money shot.
- [ ] **Rung C** few-shot slope (50–100 real words/script) — cheap, very persuasive to reviewers.
- [x] **The law** (Part ①): 12 b1800 runs, LOSO 90.9%, fusion_gain R²=0.70. (done)
- [ ] **Fix N=9** for "law": N-expansion to ~15–20 via synthetic-only scripts
      (Sinhala/Thai/Lao/Khmer/Myanmar/Tibetan) + per-sample bootstrap CIs. *(pre-register first)*
- [ ] **B7 uncapped** headline runs → strong absolute-WRR table (kept separate from capped law runs).

**Rigor & honesty (reviewers attack here):**
- Bootstrap 95% CIs (resample scripts); per-sample McNemar/paired-bootstrap; report the full
  descriptor horse-race (fertility vs STRR vs fragmentation/entropy) — no silent dropping.
- Limitations section incl. the synthetic→real domain gap and Gujarati near-tie; falsification honesty.

**Artifacts (these get you cited regardless of numbers):**
- Release **code**, the **shared abugida pivot space**, and the **zero-real-image Indic STR
  benchmark protocol**. A benchmark + code is often what makes a paper "internationally known."

**Writing & format:**
- Spine = **Part ② (zero-shot) explained by Part ① (the law)**. Cite/out-position **MGP-STR**
  (IJCV 2026), **GraDeT-HTR** (EMNLP 2025), **Chinese stroke/radical zero-shot** line, **Beyond
  Fertility/STRR**, **BSTD** — see §5.
- Figures: `law_main`, **H3 scatter**, architecture/method diagram.
- WACV/CVPR = double-blind, 8 pages + refs, CVPR LaTeX template, OpenReview, optional supp, rebuttal.
- ICDAR = Springer LNCS, ~15 pp, double-blind. Journals (PR/IJDAR/IJCV) = no hard page limit,
  rolling, expect a major-revision round. Disclose compute (1× RTX A5000).

---

## 5. RELATED-WORK POSITIONING (must cite & differentiate — from the status doc)
- **MGP-STR** (IJCV Jan 2026): learnable multi-granularity fusion in one Latin ViT. *Fusion is a
  component, not our headline; we add the law + cross-script zero-shot.* Cite up front.
- **GraDeT-HTR** (EMNLP 2025): grapheme>BPE, Bengali handwritten. *Single script, no law, no transfer.*
- **Chinese stroke/radical zero-shot** (arXiv 2106.11613, STAR 2210.08490, HierCode 2024, LERRNet 2025):
  closest prior art to Part ②. *They transfer to unseen characters within one script; we transfer to
  an unseen **script**, predicted by the law.*
- **Beyond Fertility / STRR** (arXiv 2510.09947): argues fertility is weak. *We horse-race it and show
  STRR is degenerate (=0) for every Indic script — itself a finding.*
- **BSTD** (arXiv 2511.23071): the benchmark; PARSeq ~73%. *We don't chase per-script SOTA; orthogonal
  axis = predictability + zero-real-image transfer.*

---

## 6. SOURCES (live, 2026-06-26)
- WACV 2027 dates: https://wacv.thecvf.com/Conferences/2027/Dates
- WACV acceptance (37.8%, 2026): https://papercopilot.com/statistics/wacv-statistics/
- CVPR 2027 (abstract Nov 15 '26): https://mlciv.com/ai-deadlines/conference/?id=cvpr27
- ICDAR 2027 (KL, Aug 18–22 '27): https://icdar2027.org/important-dates
- EMNLP 2026 (deadline passed May 25): https://2026.emnlp.org/
- AACL-IJCNLP 2026 (direct Jul 15 '26): https://2026.aaclnet.org/calls/main_conference_papers/
- AAAI 2027 (paper Jul 27 '26): https://aaai.org/conference/aaai/aaai-27/
- Pattern Recognition IF 7.6: https://research.com/journal/pattern-recognition
- IJDAR (IF 2.5, 7-day first decision): https://link.springer.com/journal/10032
- NeurIPS 2026 Eval&Datasets (deadline passed May 6): https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets
