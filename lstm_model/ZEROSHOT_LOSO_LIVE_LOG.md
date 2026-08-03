# ZERO-SHOT LOSO — LIVE RUN LOG & SESSION HANDOFF
**Purpose:** single source of truth for the *running* 9-script zero-shot LOSO experiment
(Part ② of the Brahmic-STR paper). Each new session starts by reading **this file first**,
then `RESEARCH_STATUS_AND_PATH.md` (overall research status) and `PREREGISTRATION.md`.
This file is updated as each rung completes.

**Owner:** Ritu Baskey · **Machine:** server3 (`cvpr-gamma`, RTX A5000 24 GB) ·
**Started:** 2026-06-25 11:27 IST · **Last updated:** 2026-07-18 morning (**RUN RESUMED 2026-07-17 22:36 IST** after the 07-10 stop — GPU was free again; Bbpe now 5/9: kannada 13.33, malayalam 4.2, oriya 14.94 landed overnight, gujarati training; guardian + @reboot + monitor re-armed; WACV strategy doc + full manuscript draft started, see §12)

---

## 0. TL;DR FOR A NEW SESSION
- A long (~9-day, resumable) run: `run_zeroshot_loso.sh`, log `zeroshot_loso.log`.
  **27 rungs** = 9 held-out scripts × {Rung A, Rung B, Rung **Bbpe**} — the Bbpe BPE-baseline
  phase (same protocol, stock tokenizer) was appended 2026-07-02 per PREREGISTRATION §7/§8;
  it runs AFTER the main 18, tamil/telugu first. **✅ ALIVE again** — after the ~Jun 30 power
  outage + ~6 days of GPU contention, the §9 watcher auto-relaunched the run 2026-07-06 12:03 (see §11).
- **Done so far (20/27 — MAIN 18 COMPLETE; Bbpe 2/9, pivot confirmed necessary). ⏸ RUN STOPPED 2026-07-10 13:30 (owner request, GPU ceded; recovery stack disarmed — resume manually, see rung log):** Tamil A=0.0/**B=9.16**, Telugu A=1.28/**B=19.82**, Kannada A=2.78/**B=15.42**,
  Malayalam A=0.18/**B=9.69**, Oriya A=0.77/**B=25.38**, Gujarati A=3.05/**B=15.86** (P2's predicted
  best — missed, 4th), Bengali A=1.39/**B=27.08** (NEW BEST from 2nd-lowest coverage — the 2×-synth
  covariate disclosed at filing dominates), Devanagari A=5.46 (⚠ breaches the "<5" line by 0.46).
  **At 7 B points: coverage +0.29 raw / +0.77 among equal-synth scripts; fertility −0.82 (refuted).**
  Emerging story: synth quantity ≫ coverage (within budget) ≫ fertility/vis-sim (nothing).
  Devanagari B (2× synth, cov 94.3) expected HIGH ~25+ — the pattern's next test, ~17:00 IST.
- **✅ RESOLVED 2026-07-06:** the §9 relaunch watcher fired at 12:03 (18.6 GB free), relaunched the
  run itself with **`HF_HUB_OFFLINE=1`** baked in, and exited. Kannada B re-trained from scratch in
  ~6¾ h (vs ~1–1.5 days pre-fix — the `num_workers` speedup is CONFIRMED, ~1.8–1.9 batch/s).
- **2026-07-02 paper-hardening (all committed, see §9 for detail):** PREREGISTRATION §8 amendments
  filed (H3 horse-race incl. coverage + visual-similarity; BPE baseline; outage housekeeping) while
  7/9 Rung-B results are unobserved; **`PROSPECTIVE_PREDICTIONS_H3.md`** commits rank predictions
  for the 7 unknown scripts (commit `047c30c` = prospectivity proof; **malayalam is the decisive
  script** between fertility-P1 and coverage-P2); orchestrator resume bug FIXED (would have evaluated
  kannada-B's partial ep6 ckpt); `compute_visual_similarity.py` computes the vissim descriptor
  (CPU-only, frozen encoder) → `visual_similarity_descriptors.json`.
- **2026-07-03 (see §10):** full relaunch path VERIFIED end-to-end (sentinels, data, offline HF cache
  incl. banglish tokenizer, bash-5 Bbpe path, disk); working-tree copy of THIS FILE was found silently
  reverted to a pre-07-02 version and restored from HEAD (**lesson: `git diff` the handoff docs at
  session start — git HEAD is the source of truth**); `visual_similarity_descriptors.json` committed
  result-blind (`c5e7c28`); 2-page prof summary added (`prof_abstract/`, `8c75e97`+`909b161`).
- **The pipeline was audited correct on 2026-06-26** (fonts/paths/training/eval/metric — see §3).
- **The headline is H3** (does fertility predict which scripts transfer), *not* absolute WRR.
- **Owner directive:** strong/top-tier results but **honestly — never fabricate or inflate.**
  Maximize via correctness + legitimate method/framing only.

- **NEW 2026-06-29 — dataloader speedup committed (`54d7a45`):** `num_workers 0→8` + `prefetch_factor=4`
  + `persistent_workers` in `train_florence2.py`. The A5000 was dataloader-starved (util 7–60%, our
  job uses only ~11.3 GB). **Result-neutral** (same seed/order). **Applies from the kannada rung onward**;
  tamil/telugu were pre-fix. Watch kannada's batch/s (monitor reports it) to confirm the gain.
- **NEW 2026-06-29 — competitive-positioning doc `COMPETITIVE_POSITIONING_AND_LITERATURE.md`** (committed
  `fe68c76`): verified field landscape + the #1 reviewer threat (arXiv 2312.10806, "data-size>typology")
  with a 3-part rebuttal. **Read it before writing Related Work.**
- **NEW 2026-06-29 — server1 (`gpu`) scoped, NOT yet used:** 3× Tesla P6 16 GB, user `ujjwal`. Our job
  fits (~11.3 GB) so batch unchanged, BUT P6 ≈ ¼–⅓ A5000 speed, and `gpu` does not resolve from server3.
  **Decision deferred** until kannada's post-fix speed shows whether server1 is even needed. See §8.

**First thing to do in a new session:** re-arm monitoring (the previous session's monitor
dies when the session ends — the *run itself* keeps going independently). See §6.

---

## 1. THE EXPERIMENT (what each rung means)
For each held-out Brahmic script, train two models in the shared abugida pivot space and
evaluate **pivot WRR on that script's REAL test images** (which the model never trained on):
- **Rung A** = trained on **source scripts only** (7000 train / 900 val, 7 source langs).
  The held-out script is *never seen in any form*. → **Rung A WRR ≈ 0 is the expected baseline.**
- **Rung B** = Rung A **+ synthetic target-script images** (3240 synth; still **ZERO real**
  target images). This is the **transfer number** — the gain from synthetic exposure in the
  shared space. Pilot (Tamil): Rung A=0.00, **Rung B=9.94** WRR.

Reference points (Tamil, from pilot / prior runs): Florence-2 raw zero-shot = 0.0;
supervised Tamil std = 28.07, grapheme = 36.84. So Rung B's job is to climb from 0 toward
the supervised ceiling **using no real target images.**

**Script order:** `tamil → telugu → kannada → malayalam → oriya → gujarati → bengali →
devanagari → gurmukhi`. Pre-fix each Rung B took ~1–1.5 days (dataloader-bound); expect faster from
kannada onward after the `num_workers` speedup (`54d7a45`).

---

## 2. LIVE STATUS & RESULTS TABLE
Updated as rungs finish. WRR/CharAcc/CER in %. (— = not done yet.)

| # | Script | Rung A WRR | Rung B WRR | CharAcc (B) | CER (B) | N(test) | Status |
|---|--------|-----------|-----------|-------------|---------|---------|--------|
| 1 | tamil      | 0.0 | **9.16** | 39.0 | 61.0 | 513  | ✅ done |
| 2 | telugu     | 1.28 | **19.82** | 54.43 | 45.57 | 545  | ✅ done |
| 3 | kannada    | 2.78 | **15.42** | 46.37 | 53.63 | 720  | ✅ done (B re-run after outage) |
| 4 | malayalam  | 0.18 | **9.69** | 34.57 | 65.43 | 547  | ✅ done — decisive rung: P1 flagship FAILED, P2 tier confirmed |
| 5 | oriya      | 0.77 | **25.38** | 56.37 | 43.63 | 1044 | ✅ done — BEST transfer yet; high-coverage script lands on top, as P2 requires |
| 6 | gujarati   | 3.05 | **15.86** | 38.49 | 61.51 | 1015 | ✅ done — MISSED est ~35; rank 3rd not 1st (tie w/ kannada). Honest dent in P2 |
| 7 | bengali    | 1.39 | **27.08** | 57.77 | 42.23 | 2873 | ✅ done — NEW BEST from 2nd-lowest coverage; the disclosed 2×-synth covariate dominates |
| 8 | devanagari | 5.46 | **30.09** | 57.16 | 42.84 | 6042 | ✅ done — NEW BEST; 2nd 2×-synth script at #1–2, as the pattern implied |
| 9 | gurmukhi   | 0.59 | **22.16** | 44.1 | 55.9 | 2879 | ✅ done — prediction scored: outer band HIT [8.2–24.1], inner band missed (+6.0, under-called) |

**Raw result so far:** `result_zs_loso_rungA_tamil.json` =
`{"rung":"A","script":"tamil","N":513,"WRR":0.0,"CharAcc":4.8,"CER":95.2}`

**Append-only rung log:**
- `2026-06-25 23:08` — [RESULT LOSO RUNG A tamil] N=513 WRR=0.0 CharAcc=4.8 CER=95.2 (expected baseline).
- `2026-06-26 ~14:00` — [RESULT LOSO RUNG B tamil] N=513 WRR=9.16 CharAcc=39.0 CER=61.0 — **transfer works**: 0→9.16 WRR, 4.8→39.0 CharAcc with ZERO real Tamil images (vs pilot B=9.94, consistent).
- `2026-06-28 ~04:52` — [RESULT LOSO RUNG A telugu] N=545 WRR=1.28 CharAcc=23.32 CER=76.68 — Rung A baseline (slightly above ~0; some incidental cross-script coverage). Rung B started same time, training now.
- `2026-06-29 ~21:18` — [RESULT LOSO RUNG B telugu] N=545 WRR=19.82 CharAcc=54.43 CER=45.57 — **strong transfer, 2× Tamil** (Tamil B=9.16). 1.28→19.82 WRR, 23.3→54.4 CharAcc with ZERO real Telugu images. Second H3 data point.
- `2026-06-30 ~05:30` — [RESULT LOSO RUNG A kannada] N=720 WRR=2.78 CharAcc=19.25 CER=80.75 — Rung A baseline. Kannada Rung B started 05:56, batch ~1.3/s.
- `2026-06-30 ~10:04` — **RUN DIED** mid-Kannada-Rung-B at ep6/15 (best_model ep6 saved). Machine lost power → network down (HF DNS-resolution errors flood the tail) → orchestrator + training killed. No Kannada-B result JSON; **B re-runs from scratch on relaunch.** Discovered 2026-07-02.
- `2026-07-06 12:03` — **AUTO-RELAUNCH WORKED:** `gpu_relaunch_watcher.sh` saw 18,595 MiB free (≥12,000), relaunched `run_zeroshot_loso.sh` itself via nohup (PID 1937079) and exited. No manual step involved — the §9/§10 recovery infra worked unattended.
- `2026-07-06 18:48` — [RESULT LOSO RUNG B kannada] N=720 WRR=15.42 CharAcc=46.37 CER=53.63 — third H3 point; full from-scratch re-train + eval in ~6¾ h (speedup confirmed at scale). **Prospective hit:** `PROSPECTIVE_PREDICTIONS_H3.md` (result-blind, commit `047c30c`) staked exploratory estimate "kannada ~15".
- `2026-07-07 08:45` — malayalam Rung A training started (orchestrator's `wait_gpu` rode out overnight contention, down to ~1 GB free). 1.8–1.9 batch/s, ep9/15 by 11:26 — healthy. **Malayalam is the decisive H3 script** (P1-fertility's best vs P2-coverage's worst).
- `2026-07-07 ~13:30` — [RESULT LOSO RUNG A malayalam] N=547 WRR=0.18 CharAcc=2.74 CER=97.26 — expected ~0 baseline; LOWEST Rung A of the 4 so far (tamil 0.0 < mal 0.18 < telugu 1.28 < kannada 2.78), consistent with malayalam having the lowest token coverage (79.2). Rung B (the decisive rung) started 13:36, 1.6 batch/s — result expected ~20:30–21:30 IST tonight.
- `2026-07-07 23:20` — [RESULT LOSO RUNG A oriya] N=1044 WRR=0.77 CharAcc=10.09 CER=89.91 — 4th straight Rung A < 5, as predicted ("every Rung A stays low", 047c30c). Rung B oriya (H3 point #5 — the prereg figure threshold) started 23:20 at 2.0 batch/s; result expected ~05:30–07:00 IST 2026-07-08. P2 estimate for oriya: ~21.
- `2026-07-08 04:28` — [RESULT LOSO RUNG B oriya] N=1044 WRR=25.38 CharAcc=56.37 CER=43.63 — **best transfer of the campaign**; P2 estimate was ~21. **H3 at 5 points: Spearman(coverage)=+0.90, Spearman(fertility)=−0.90** (n=5 one-sided p≈0.04 descriptively; confirmatory test remains the 7-script prospective score, now 3/7 observed with P2's internal order perfect: oriya>kannada>malayalam, and P1's exactly reversed). Prereg ≥5–6-point threshold reached → H3 figure buildable; building at 6 points once gujarati B lands. Gujarati Rung A started 04:28, 2.0 batch/s.
- `2026-07-08 ~08:05` — [RESULT LOSO RUNG A gujarati] N=1015 WRR=3.05 CharAcc=15.71 CER=84.29 — 5th straight Rung A < 5 as predicted; highest baseline of the five, consistent with gujarati's highest token coverage (97.3, incidental-coverage effect). Rung B started 08:09 (2.1 batch/s) — **P2's predicted best transferrer (est ~35)**; result expected ~14:30–15:30 IST.
- `2026-07-08 13:23` — [RESULT LOSO RUNG B gujarati] N=1015 WRR=15.86 CharAcc=38.49 CER=61.51 — **first real miss for P2**: predicted BEST (crude est ~35), realized 3rd of 6 (behind oriya 25.38, telugu 19.82; statistical tie with kannada 15.42). **H3 at 6 points: token-coverage +0.77, type-coverage +0.60, fertility −0.71, vis-sim −0.14** — coverage still clearly leads, but n=6 one-sided p<0.05 needs 0.829, so descriptively not significant at 6; the 7-script prospective score can still clear the 0.786 two-sided bar IF devanagari lands high (dev per-estimate → ρ(P2)≈0.86; dev mid-table → below threshold). **Devanagari is now the pivotal remaining script.** Honest note: gujarati underperforming its 97.3 coverage suggests coverage saturates at the top / other factors bind (test-set difficulty? font/style gap?) — investigate AFTER all 9 results are in (no mid-race analysis changes); report per prereg §6 either way. Bengali Rung A started 13:23 (2× synth, N(test)=2873 — covariates disclosed in 047c30c).
- `2026-07-08 16:53` — [RESULT LOSO RUNG A bengali] N=2873 WRR=1.39 CharAcc=17.7 CER=82.3 — 6th straight Rung A < 5 as predicted. Rung B started 16:53 (1.9 batch/s; 13,480 train samples with 2× synth — disclosed covariate); result expected ~02:00–02:30 IST 2026-07-09. P2 crude est for bengali: ~5 (its coverage 87.3 is 2nd-lowest); a mid-teens result would suggest the 2× synth covariate lifts it, exactly as flagged in 047c30c.
- `2026-07-09 01:26` — [RESULT LOSO RUNG B bengali] N=2873 WRR=27.08 CharAcc=57.77 CER=42.23 — **NEW BEST, and a plot twist**: P2's crude est was ~5 (coverage 87.3, 2nd-lowest). The covariate DISCLOSED AT FILING (047c30c §4: bengali & devanagari have 2× synth, 6480 vs 3240, "may lift them above the coverage line") did exactly that. **H3 at 7 points: coverage +0.29, fertility −0.82** (raw, all scripts). Descriptive decomposition per the disclosed covariate: **among the six equal-synth (3240) scripts, coverage holds +0.77**; the one 2×-synth script observed sits far above its coverage line. Emerging honest picture: **synthetic-exposure quantity is the dominant factor, pivot-space coverage orders transfer within a fixed synth budget, fertility is decisively refuted** (−0.82). NB this partially agrees with arXiv 2312.10806's data-size view — for TARGET-side synthetic quantity — while still showing a structural coverage effect at fixed budget; the declared 7-script scoring (§6 of 047c30c) will be reported unchanged regardless. Implied expectation for devanagari (2× synth, cov 94.3): HIGH, ~25+.
- `2026-07-09 06:01` — [RESULT LOSO RUNG A devanagari] N=6042 WRR=5.46 CharAcc=31.57 CER=68.43 — highest Rung-A baseline (tracks its 94.3 coverage). **⚠ Honest note: this marginally BREACHES the prospective "every Rung A < 5" line (047c30c §5)** — score it 7/8 correct with one 0.46-point breach, reported as-is. Rung B (2× synth, the pattern's next test) started 06:01 at 1.6 batch/s; result expected ~16:30–17:30 IST.
- `2026-07-09 13:32` — [RESULT LOSO RUNG B devanagari] N=6042 WRR=30.09 CharAcc=57.16 CER=42.84 — **NEW BEST**; the 2nd 2×-synth script lands at the top, as the two-factor picture implied. Honest timing note: the "expected HIGH ~25+" line was committed 11:10 IST (39474d5, before this result existed at 13:32) but only PUSHED after — treat that one as weakly prospective; the gurmukhi prediction below is committed AND pushed before its Rung B trains. **H3 at 8 points: coverage +0.43 raw, fertility −0.88; equal-synth-six coverage +0.77 unchanged.** Two-factor OLS on all 8 (WRR ~ coverage + 2×synth): coverage +0.57 WRR/pt, 2×-synth bonus +12.3 WRR, RMSE 3.97 — both 2×-synth residuals ≈ 0.
- `2026-07-09 13:42` — **PROSPECTIVE_PREDICTION_GURMUKHI.md filed** (gurmukhi Rung A training, Rung B NOT started, no gurmukhi transfer number observed by anyone): **point 16.2 WRR, ±1·RMSE [12.2, 20.2], ±2·RMSE [8.2, 24.1]**; four independent estimators converge (two-factor 16.2, equal-synth linear 16.2, equal-synth median 15.6, nearest-neighbor kannada 15.4). Also: gurmukhi Rung A < 5. Third quantitative prospective call of the campaign (kannada "~15"→15.42 hit; devanagari "~25+"→30.09 weakly-prospective hit).
- `2026-07-09 17:23` — [RESULT LOSO RUNG A gurmukhi] N=2879 WRR=0.59 CharAcc=12.3 CER=87.7 — inside the prediction file's "Rung A < 5"; baselines now 8/9 correct with one 0.46-pt breach (devanagari). **Rung B started 17:23 — the 16.2 [12.2–20.2] call was pushed at 13:44, 3h39m before training began.** Result expected ~23:15 IST tonight; main 18 then complete and the Bbpe phase starts automatically.
- `2026-07-09 22:41` — [RESULT LOSO RUNG B gurmukhi] N=2879 WRR=22.16 CharAcc=44.1 CER=55.9 — **MAIN 18 RUNGS COMPLETE.** Prediction scored per the fixed rule (see PROSPECTIVE_PREDICTION_GURMUKHI.md): point 16.2 missed by +5.96 (under-called, ≈1.5·RMSE); ±1·RMSE band missed by 1.96; **±2·RMSE band [8.2–24.1] HIT**; Rung A < 5 HIT (0.59). **Final campaign stats (9 B points): coverage +0.37 raw / +0.61 equal-synth-7; fertility −0.80. Declared 7-script prospective bet (047c30c §6): ρ(P2-coverage)=+0.32 vs ρ(P1-fertility)=−0.71 → bet WON (P2 ≫ P1), and neither ranking alone is significant — the honest verdict is the two-factor account** (9-pt checkpoint refit: WRR = −35.79 + 0.583·tok_cov + 11.48·is_2x, RMSE 4.18 — the instrument for the Khmer prediction). Transfer range 9.16–30.09 WRR across all nine scripts, every baseline ~0–5. Bbpe phase started itself 22:41 (tamil first, 2.0 batch/s).
- `2026-07-10 04:03` — [RESULT LOSO RUNG Bbpe tamil] N=513 WRR=2.34 CharAcc=30.45 CER=69.55 — **ablation evidence #1: the grapheme pivot matters.** Stock tokenizer, identical protocol/data/seed: 2.34 vs grapheme-pivot 9.16 WRR (3.9× worse).
- `2026-07-10 09:17` — [RESULT LOSO RUNG Bbpe telugu] N=545 WRR=10.28 CharAcc=55.1 CER=44.9 — **evidence #2:** 10.28 vs 19.82 (1.9× worse). Notably CharAcc is nearly EQUAL (55.1 vs 54.4): BPE reads characters but fails whole words — the pivot converts character-level ability into word-level reads. Bbpe kannada training (2.2 batch/s).
- `2026-07-10 13:30` — **RUN STOPPED INTENTIONALLY (owner request — GPU ceded to a labmate).** State at stop: 20/27 rungs done (main 18 complete + Bbpe tamil/telugu); Bbpe kannada killed at ep12/15 — it re-runs from scratch on resume (the `.train_done` sentinel guard makes the partial ckpt harmless). **The entire auto-recovery stack was DISARMED with it:** guardian killed, `@reboot` crontab line removed, session monitor stopped — the run will NOT self-resume. GPU verified free (54 MiB). **To resume the remaining 7 Bbpe rungs (~35 GPU-h):** `cd lstm_model && nohup bash run_zeroshot_loso.sh > zeroshot_loso.log 2>&1 &` (resumable, skips all finished rungs), then re-arm `.monitor/loso_guardian.sh` detached + re-add the crontab line + re-arm the session monitor.
- `2026-07-17 22:36` — **RUN RESUMED** (owner directed a research-then-work session on the WACV paper; GPU found fully free, 54 MiB). Old log backed up (`.monitor/zeroshot_loso.log.pre_relaunch_20260717.bak`); partial Bbpe-kannada ckpt set aside (`...partial_ep12_gpucede_20260710`); orchestrator relaunched (skipped the 20 finished rungs correctly); guardian re-armed detached + `@reboot` crontab line re-added + session monitor armed.
- `2026-07-18 ~00:00–06:13` — [RESULT LOSO RUNG Bbpe kannada] N=720 WRR=13.33 CharAcc=48.76 · [RESULT LOSO RUNG Bbpe malayalam] N=547 WRR=4.2 CharAcc=31.16 · [RESULT LOSO RUNG Bbpe oriya] N=1044 WRR=14.94 CharAcc=50.64 — **Bbpe 5/9. Direction 5/5 for the grapheme pivot, but the margin VARIES: ratios now 3.9× (tamil), 2.3× (malayalam), 1.9× (telugu), 1.7× (oriya), 1.16× (kannada).** Honest note: kannada's gap is small (13.33 vs 15.42) and its Bbpe CharAcc is actually HIGHER (48.76 vs 46.37) — the "pivot converts char ability into word reads" story holds, but "2–4×" is no longer the honest range; paper wording softened to "every script, up to 3.9×" pending 9/9. Gujarati Bbpe training (started 10:02 IST).
- `2026-07-07 19:37` — [RESULT LOSO RUNG B malayalam] N=547 WRR=9.69 CharAcc=34.57 CER=65.43 — **THE DECISIVE RESULT.** Honest scoring: **P1 (fertility, prereg primary) flagship FAILED** — P1 ranked malayalam BEST of the 7 unknowns (fertility 6.02); realized bottom-tier, far below kannada (15.42) and telugu (19.82) which P1 ranked beneath it. **P2 (coverage, declared bet) directionally RIGHT** — predicted malayalam WORST; realized bottom-tier ✓ — but its crude estimate "~0–4" undershot (9.69). The malayalam>tamil inversion (9.69 vs 9.16) is a statistical tie (Δ≈3 words of ~530; two-proportion z≈0.3). Descriptive Spearman(coverage, B-WRR) over the 4 observed points = 0.8; confirmatory scoring stays as declared (Spearman over the 7 unknowns when all exist; 2/7 observed). **Phenomenon-claim gift:** even the predicted-worst script gains 0.18→9.69 with zero real images — transfer does not collapse on low-coverage scripts. Oriya Rung A started 19:37.

---

## 3. PIPELINE AUDIT (done 2026-06-26 — why we can trust the numbers)
All verified against files on disk this session:
- **Fonts / no tofu:** Tamil synth = 3240 imgs, 0% blank (per `zeroshot_loso_meta_tamil.json`);
  token coverage 88.8%, type coverage 66.5%. `prepare_zeroshot_loso.py` fixes fonts via
  `fc-match`/multi-font + per-image blank verification. libraqm shaping available.
- **Paths:** real images resolved locally at `benchmarks/bstd/Recognition/`; stale
  `/home/ujjwal/...` paths are rewritten by `localize()` / re-resolved on load.
- **Training healthy:** Rung B Tamil loss 10.0 (ep1) → 0.06 (ep12). Converging normally.
  (NB: very low train loss is on *synthetic* data — synthetic→real domain gap is the main
  honest reason WRR may stay modest. Property to report, not a bug.)
- **Eval correctness** (`predict_with_conf.py`): loads `best_model` adapter (val-selected,
  converged), evaluates the `['test']` split = the held-out script's **real** images,
  beam search, `max_new_tokens=64`.
- **Metric** (`metrics.py:evaluate_corpus`): WRR = exact match after NFC + zero-width-char
  removal + whitespace collapse = IndicSTR12 / BSTD protocol (directly comparable to BSTD
  PARSeq ~73%). CharAcc = (1 − corpus_CER). Fair, not unfairly strict, not buggy.

**Conclusion:** no silent mistake is costing us numbers. Whatever comes out is real.

---

## 4. THE HEADLINE = H3 (this is what makes it top-tier, not absolute WRR)
Zero-shot-to-unseen-script WRR will be modest (~10%-ish). The paper's money shot is:
> **Does a writing system's fertility/structure PREDICT which unseen scripts transfer best?**

When ≥ ~5–6 Rung B results exist, build the H3 figure:
- X = held-out-script **fertility** (from `script_descriptors.json`),
- Y = **Rung B WRR** (from `result_zs_loso_rungB_*.json`),
- report **Spearman ρ** (PREREG §5 H3). Expect higher-fertility/higher-coverage scripts
  (Gujarati/Devanagari/Telugu) to transfer better than low-coverage (Malayalam).

This correlation is publishable **even if individual WRRs are low.** Pre-GPU coverage
signal (token-cov%, ZERO training) for ordering intuition: Gujarati 97.3, Devanagari 94.3,
Oriya 92.6, Telugu 92.3, Kannada 90.8, Gurmukhi 90.7, Tamil 88.8, Bengali 87.3, Malayalam 79.2.

---

## 5. HONEST PATH TO "IMPRESSIVE" (owner directive: strong but honest)
Legitimate levers only — never fabricate:
1. **Correctness first** (done §3) — bugs that tank WRR are the cheapest honest win.
2. **H3 correlation** (§4) — the real headline; strong even at low WRR.
3. **Rung C few-shot slope** — add 50–100 *real* target words/script → show transfer slope
   from 0 real images upward. Cheap, very persuasive for reviewers.
4. **B7 uncapped headline runs** — fully-converged per-script models for a strong absolute
   table (kept SEPARATE from capped law models).
5. **N-expansion** (Sinhala/Thai/Lao/Khmer/Myanmar/Tibetan synthetic) to fix N=9 for "the law."
6. If a script is weak → **investigate why** (domain gap / coverage / font), fix honestly,
   report it either way. Falsification honesty keeps the paper un-rejectable.

---

## 6. RESUME / RE-ARM / MONITOR (exact commands)
```bash
cd /c/ujjwalb/ritu1/lstm_model
PY=/c/ujjwalb/.conda/envs/ritu_scenetext/bin/python

# Is the run alive? what's done?
pgrep -af 'run_zeroshot_loso[.]sh'
grep 'RESULT LOSO' zeroshot_loso.log
ls result_zs_loso_rung*_*.json
tail -20 zeroshot_loso.log

# Which rung is training now + health:
pgrep -af train_florence2
tail -5 train_zeroshot_loso_rung*_*.log

# If the run DIED (and < 18 results): relaunch — it's resumable (skips finished rungs):
nohup bash run_zeroshot_loso.sh > zeroshot_loso.log 2>&1 &
#   (or a subset: nohup bash run_zeroshot_loso.sh malayalam oriya > zeroshot_loso.log 2>&1 &)
```
**Re-arm the per-rung monitor in a new session** (previous session's monitor dies on exit;
the run does not). A ready-made watcher lives at **`.monitor/loso_monitor.sh`** (git-ignored,
inside this repo). Keep it running in a spare terminal (tmux/screen), persistent:
```bash
bash /c/ujjwalb/ritu1/lstm_model/.monitor/loso_monitor.sh
```
It emits each new `RESULT LOSO` line, the batch/s of each new rung (to confirm the speedup),
and warns once if the orchestrator dies or all 18 rungs finish.

**⚠️ Working-file rule (owner directive 2026-06-29):** keep ALL scratch/temp/working files under
`ritu1/` (e.g. `.monitor/`). NEVER write scratch files into
`/c/ujjwalb/Antik/...`, which is a **labmate's** project folder. Never read or write outside `ritu1/`.

**Env:** `export HF_HOME=/c/ujjwalb/.cache/huggingface`. GPU shared with UNRELATED jobs from other
users (`Vansh/multihop_memory_vqa` and others, ~11 GB total at last scan) — do not touch them.
`wait_gpu` needs ≥12 GB free. Our training itself uses ~11.3 GB at batch_size=4.

---

## 7. NEXT STEPS (in order)
1. Let the 18 rungs finish (monitor reports each). Collect `result_zs_loso_*`.
2. At ≥5–6 Rung B results → build H3 figure + Spearman(fertility, Rung B WRR).
3. Rung C few-shot slope. 4. B7 uncapped. 5. N-expansion. 6. Manuscript (Phase D) —
   spine = Part ② explained by Part ①; cite/out-position MGP-STR + Chinese stroke/radical line.

---

## 8. SESSION 2026-06-29 — WHAT CHANGED (full detail for a new session)
**Commits this session (branch `main`):**
- `fe68c76` — `COMPETITIVE_POSITIONING_AND_LITERATURE.md` (verified lit landscape + rebuttals)
  + Telugu Rung A result/config + training-log update.
- `54d7a45` — dataloader speedup in `train_florence2.py` (`num_workers 0→8`, `prefetch_factor=4`,
  `persistent_workers`, `TOKENIZERS_PARALLELISM=false`). **Result-neutral**; applies from kannada on.
- `.gitignore` added (`.monitor/`).

**Speed diagnosis (why the fix):** A5000 our-job memory ≈ 11.3 GB at batch_size=4; AMP already on;
GPU util was 7–60% ⇒ dataloader-bound at `num_workers=0`. Fix overlaps image decode + processor
preprocessing with GPU compute. Expect a meaningful speedup on Rung B (the critical path).
**VERIFY on kannada's batch/s** via the monitor — that number decides the server1 question below.

**server1 (`gpu`) — investigated, decision DEFERRED:**
- Host shows as `gpu`, user `ujjwal`, **3× Tesla P6, 16 GB each** (free at scan: GPU0 ~12.5, GPU1 ~14.4,
  GPU2 ~14.9 GB; small jobs from OTHER users on all three — coexist, do not evict).
- Driver 555.42 / CUDA 12.5 present (torch cu121 OK; P6 = Pascal sm_61, supported). For our purposes it
  is a FRESH box: no `ritu_scenetext` env, no repo clone, no data.
- Our training fits in P6 free memory at batch_size=4 (~11.3 GB) ⇒ **no science change**, BUT P6 ≈ ¼–⅓
  A5000 speed (Pascal; crippled FP16 ⇒ effectively FP32). So the marginal win is modest.
- **Blocker:** `gpu` does NOT resolve from server3 (`192.168.57.100`); owner reached it from
  `192.168.199.98` — subnets may differ. To use it: need its **routable IP + SSH user**, then create
  env (conda + torch 2.5.1 cu121 + transformers 4.44.2 + peft), `git pull` repo, rsync ~1.25 GB data
  (BSTD 1 GB + synth 189 MB + splits/vocab 39 MB), and launch a **disjoint subset**, e.g.
  `bash run_zeroshot_loso.sh kannada malayalam oriya gujarati` (resumable; server3 keeps the rest).
- **Plan:** set up server1 ONLY if kannada's post-fix speed shows server3 alone is too slow.

**Honest-results stance reaffirmed (owner pushed hard for top-tier "do anything"):** the agreed path is
to maximize *legitimate* levers and make the paper robust to ANY H3 outcome (if H3 is weak, pivot the
spine to "zero-real-image cross-script transfer works + the law + released benchmark/artifact"). Never
fabricate/cherry-pick/drop scripts — that is the fastest way to lose a top-tier venue. See
[[honest-strong-results]] and `COMPETITIVE_POSITIONING_AND_LITERATURE.md` §0.

---

## 9. SESSION 2026-07-02 — POWER OUTAGE POST-MORTEM + RELAUNCH PLAN
**What happened:** discovered the run had been **dead ~2 days**. Between 2026-06-29 and 2026-06-30
it completed **Telugu Rung B = 19.82** (strong) and **Kannada Rung A = 2.78**, then the box lost
power ~Jun 30 10:04 during **Kannada Rung B (ep6/15)**. The training log tail is a wall of
`huggingface.co` DNS-resolution failures — that was the **network dropping as the machine went down**,
NOT a pipeline bug. Orchestrator + training process both gone; no Kannada-B result JSON.

**State now:** box is back up but **contended** — other users' jobs hold ~16.3/24.5 GB (98% util),
only ~8.2 GB free; our job needs ~11.3 GB + `wait_gpu` wants ≥12 GB. So we **wait for headroom.**

**Relaunch plan (auto):** **`.monitor/gpu_relaunch_watcher.sh`** polls GPU free memory every 3 min
and, once ≥12 GB frees, **relaunches the run ITSELF via nohup** (needs no internet, no manual step; guards
against double-launch; logs to `.monitor/gpu_relaunch.log`). Resumable — skips finished rungs,
re-runs Kannada B onward. **Two hardening changes at relaunch:**
1. `export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` — Florence-2 base is cached locally, so a network
   blip can never again flood the log / risk the run. Result-neutral.
2. Re-arm `.monitor/loso_monitor.sh` so a silent death is caught in minutes, not days.

Exact relaunch (run from `lstm_model/`, after GPU frees — offline env now baked into the script):
```bash
cd /c/ujjwalb/ritu1/lstm_model
nohup bash run_zeroshot_loso.sh > zeroshot_loso.log 2>&1 &
```

**H3 so far (2 Rung B points):** Tamil 9.16, Telugu 19.82 — Telugu (higher token-cov 92.3 vs Tamil
88.8) transfers much better, directionally consistent with H3. Need ≥5–6 for the Spearman.
⚠️ NB: the 2 points trend AGAINST fertility-as-predictor (Tamil has HIGHER fertility but LOWER
transfer) and WITH coverage — handled honestly via the §8 amendments below.

**LATER SAME SESSION (2026-07-02) — paper-hardening changes, all committed:**
1. **PREREGISTRATION.md §8 amendments filed** (while 7/9 Rung-B results are still unobserved):
   (1) H3 descriptor horse-race made explicit (fertility stays confirmatory primary; coverage +
   §2 set + visual-similarity are declared competitors, all reported); (2) visual-similarity
   descriptor defined (frozen Florence-2 vision encoder on rendered glyphs — answers arXiv
   2312.10806 with data); (3) BPE baseline (Rung Bbpe) implementation logged.
2. **`PROSPECTIVE_PREDICTIONS_H3.md` created** — timestamped rank predictions for the 7 unknown
   Rung-B scripts under P1 (fertility) and P2 (coverage), + declared bet (P2 wins), + scoring
   rule. The git commit/push hash is the proof of prospectivity. **Malayalam is the decisive
   script** (P1's best, P2's worst). Manuscript will cite the commit hash.
3. **`run_zeroshot_loso.sh` upgraded:** (a) `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` baked in;
   (b) resume bug FIXED — old check skipped training if `best_model` existed, which would have
   silently EVALUATED KANNADA-B'S PARTIAL EP6 CHECKPOINT on relaunch; now uses a
   `.train_done` sentinel + skips whole rung when its result JSON exists + train-failure guard;
   (c) **BPE-baseline phase appended** (Rung `Bbpe` × 9 scripts, tamil/telugu first, identical
   protocol minus grapheme injection) — the ablation that causally links Part ① to Part ②.
   Total rungs now **27** (monitor EXPECTED updated). Partial kannada ckpt set aside as
   `checkpoints_zeroshot_loso_rungB_kannada.partial_ep6_poweroutage_20260630`; `.train_done`
   sentinels touched in the 5 completed ckpt dirs.
4. **`compute_visual_similarity.py` written & run (CPU-only, never touches the LOSO GPU):**
   top-50 grapheme clusters per script (same `segment_graphemes_indic` as the descriptor
   pipeline) × 3 fontconfig-verified fonts (same `resolve_fonts` as the synth pipeline) →
   ~150 glyph images/script → frozen Florence-2 vision encoder (mean-pooled, L2-normed) →
   9×9 mean-pairwise-cosine matrix → `visual_similarity_descriptors.json`. Enters the H3
   horse-race as the exploratory visual-similarity control (§8 Amendment 2 / answers arXiv
   2312.10806 with data).
5. Still TODO: Rung C; H3 figure at ≥5–6 points; N-expansion (pre-register first).

---

## 10. SESSION 2026-07-03 — VERIFICATION, INCIDENT FIX, PROF SUMMARY (GPU still blocked)

**Run state all day:** DEAD (since Jun 30), GPU contended the whole session (~7.9 GB free vs 12 needed;
other users' jobs unchanged). `.monitor/gpu_relaunch_watcher.sh` confirmed ALIVE and polling every 3 min
— it relaunches the run itself when ≥12 GB frees, no manual step needed. A watcher tails
`.monitor/gpu_relaunch.log` for the relaunch outcome and then execs `loso_monitor.sh`.

**⚠️ INCIDENT — stale worktree copy of this file:** at session start, the working-tree copy of
`ZEROSHOT_LOSO_LIVE_LOG.md` was found silently REVERTED to a pre-2026-07-02 state (27→18 rungs, §8/§9
hardening notes gone, `WRRare` typo introduced; mtime 3 min after the last commit — the previous session
apparently saved a stale buffer over it). Diff-verified a strict regression, backed up to
`.monitor/ZEROSHOT_LOSO_LIVE_LOG.stale_worktree_20260703.md.bak`, restored via `git checkout`.
**Standing lesson: at session start, `git diff` the handoff docs; a dirty worktree on them likely
means corruption, and git HEAD is the source of truth.**

**Relaunch path VERIFIED end-to-end (not assumed):**
- Resume logic traced: 5 finished rungs skip via result JSONs; all 5 `.train_done` sentinels on disk;
  kannada-B dir absent (partial renamed aside) → retrains from scratch; internal `wait_gpu` re-guards.
- Offline mode PROVEN: with `HF_HUB_OFFLINE=1`, processor + trust-remote-code config + full 231M-param
  Florence-2 weights + banglish tokenizer (vocab 71,254) all load from cache, CPU-only; no `.incomplete`
  blobs. Tokenizer behavior identical to finished rungs.
- Data complete for all 22 remaining rungs (36 splits/vocab files, 9 synth dirs exact counts);
  bash 5.0.17 OK for the empty-`GRAPH_ARGS` Bbpe path (`set -u`); disk 194 GB free vs ~60 needed;
  old `zeroshot_loso.log` backed up (`.monitor/zeroshot_loso.log.pre_relaunch_20260703.bak`) since
  relaunch truncates it.

**Committed this session (all pushed):**
- `c5e7c28` — `visual_similarity_descriptors.json` committed BEFORE any further Rung-B result exists
  (completes the result-blind evidence chain of `047c30c`). NB: its dynamic range is tight (8/9 scripts
  within 0.917–0.930; Gurmukhi apart at 0.876) — rankings noise-sensitive; report as-is if it loses.
- `8c75e97` + `909b161` — **`prof_abstract/`**: 2-page abstract-style PDF for the supervisor
  (`RESEARCH_SUMMARY_RITU.pdf` + build scripts + 2 new figures: pivot-space diagram, zero-shot bar
  chart). Answers: which scripts attempted (all 9 supervised; zero-shot 2 done + kannada mid-run),
  what the shared representation is (Unicode offset pivot), which script held out for Telugu 19.8
  (Telugu itself). To refresh after new results: edit data lists in `make_fig_zeroshot.py`, re-run
  the two fig scripts + `make_pdf.py` (system python3; fpdf2 in user site). A blanket `*.png`
  gitignore rule hides new PNGs — `git add -f` figure files intentionally committed.

**Honest publishability read (2026-07-03):** publishable at a good venue ~certain given honesty +
prereg; STRONG (CORE-A) is a coin flip decided by the 7 unobserved Rung-B results — a clean rank
correlation for EITHER declared predictor keeps the top-tier story; noisy/no ordering → benchmark +
negative-result paper (moderate tier). Timeline (WACV R2 Aug 28) is the biggest practical risk while
the GPU stays contended.

**server1 fallback (if contention drags past ~this weekend):** revive §8 plan — 3× Tesla P6 16 GB.
P6 is Pascal (sm_61, crippled FP16 ⇒ FP32): expect ~4–6× slower per rung than the post-fix A5000
(Rung B ≈ 1.5–2.5 days/script on one P6), BUT three cards run three scripts in parallel and a slow
free GPU beats a fast blocked one. Setup needed: routable IP + SSH from owner (host `gpu` does not
resolve from server3), conda env (torch 2.5.1 cu121 / transformers 4.44.2 / peft 0.13), git clone,
rsync ~1.25 GB data. Priority subset: `bash run_zeroshot_loso.sh malayalam oriya gujarati`
(malayalam = decisive script; disjoint from server3's resumable set; result JSONs merge trivially).
---

## 11. SESSION 2026-07-07 — AUTO-RELAUNCH WORKED; KANNADA B = 15.42 (PROSPECTIVE "~15" HIT)

**Run state:** ALIVE. The recovery infrastructure built in §9/§10 worked unattended, end-to-end:
`gpu_relaunch_watcher.sh` fired 2026-07-06 12:03 (18,595 MiB free ≥ 12,000), relaunched the
orchestrator via nohup (PID 1937079), logged `RELAUNCHED OK`, and exited. Resume logic behaved
exactly as verified in §10: the 5 finished rungs were skipped via their result JSONs and kannada-B
re-trained from scratch (partial ep6 ckpt had been set aside). Result landed 18:48 the same day.

**Kannada Rung B = 15.42 WRR / 46.37 CharAcc / 53.63 CER (N=720).**
- **H3 now has 3 points, perfectly rank-ordered by token coverage (P2):** telugu 92.3 → 19.82,
  kannada 90.8 → 15.42, tamil 88.8 → 9.16. Fertility (P1) remains directionally wrong on the
  observed points (tamil = highest fertility of the three, lowest transfer).
- **Prospective scoring:** `PROSPECTIVE_PREDICTIONS_H3.md` (commit `047c30c`, filed before this
  result existed) gave the exploratory point estimate **kannada ~15 → realized 15.42**. Cite the
  commit hash in the manuscript; this is exactly the evidence chain that file exists to create.
- Speedup confirmed at scale: full Rung-B train+eval ≈ 6¾ h (pre-fix ~1–1.5 days), 1.8–1.9 batch/s
  ⇒ server3 alone is fast enough for the WACV timeline IF contention stays moderate; server1 (§8)
  stays a fallback only.

**Now training:** malayalam Rung A (started 2026-07-07 08:45 after `wait_gpu` rode out overnight
contention; ep9/15, loss ~0.28, ~11.2 GB, healthy at 11:26). Malayalam Rung B — **the decisive
script** (P1's predicted best vs P2's predicted worst, P2 crude estimate ~0–4 WRR) — expected
within ~a day, contention permitting. Then oriya → gurmukhi, then the 9 Bbpe rungs.

**Monitor re-armed** this session (`loso_monitor.sh`, persistent). ⚠️ NB: the relaunch
truncated `zeroshot_loso.log`, so the monitor's counter only sees post-relaunch RESULT lines
(max 22 of 27): completion will surface as `[WARN orchestrator STOPPED] 22/27`, not `[DONE]` —
interpret accordingly.

**Committed this session:** kannada-B result + conf JSONs and this log update.
**Deliberately NOT committed:**
`training_log_florence2_grapheme.json` — it is overwritten in place by the live malayalam-A
training (a mid-epoch snapshot would be misleading); commit it at a rung boundary. NB the
kannada-B in-memory training curve in that file has already been overwritten by malayalam-A;
the per-rung text log `train_zeroshot_loso_rungB_kannada.log` retains the full curve.

**Standing auto-recovery armed 2026-07-07 (later same session, owner request):**
`gpu_relaunch_watcher.sh` is ONE-SHOT (exits after a single relaunch, or immediately if the run is
alive), so a new **`.monitor/loso_guardian.sh`** now provides standing protection: a flock'd loop
(max one instance) that, whenever the orchestrator is dead with <27 result JSONs on disk, invokes
the watcher (waits for ≥12 GB free, relaunches detached, exits) and then resumes guarding; it
retires itself once all 27 results exist. Armed two ways: (a) running detached NOW (own session via
setsid, survives disconnect/session end); (b) **user crontab `@reboot` entry** ⇒ a power outage now
auto-recovers with no human in the loop (cron is active; `run_zeroshot_loso.sh` is env-self-contained
— absolute conda python, `HF_HOME` + offline flags exported inside the script — so cron's minimal
env is fine). Verified live: flock dedupe (second launch exits), watcher's double-launch guard
(exits when orchestrator alive). Log: `.monitor/guardian.log`.
**⚠️ When the 27-rung run fully completes: remove the crontab line (`crontab -l`, then `crontab -r`
or edit) — the guardian retires itself, but the @reboot entry would re-arm a stale watcher-wait
on the next boot.**

---

## 12. SESSION 2026-07-17/18 — WACV PIVOT: STRATEGY + RESUME + MANUSCRIPT

**Owner directive:** research thoroughly whether the work clears the top-tier bar and what WACV
accepts, then work. Outcomes:
1. **`WACV_STRATEGY.md`** (new) — venue intel verified live (R2: enroll Aug 21 / paper Aug 28 /
   decisions Oct 9, **NO rebuttal in R2**; three tracks, we target **Algorithms**; 8pp + unlimited
   refs; arXiv explicitly allowed), fresh preemption scan (five new 2026 cites, **no preemption** —
   closest is Task-Analogies HTR 2604.09713, Latin-only, leaves unseen scripts to future work),
   honest verdict + experiment priority stack + how-we-lose list. Read it first next session.
2. **Run resumed + recovery stack re-armed** (see rung log above). Remaining after Bbpe: Amendment-4a
   scaling sweep (needs 6480/12960 synth rendering FIRST — CPU job, not yet started), 4c VLM baseline
   (inference in GPU gaps), 4b Khmer (engineering timeboxed, drop-dead ~Aug 8).
3. **`paper_wacv/` manuscript started on the official WACV 2027 author kit** (downloaded from
   wacv.thecvf.com; compiles LOCALLY via `/c/ujjwalb/ritu1/.tools/tectonic main.tex` — no Overleaf
   round-trip needed; 7pp review-format PDF builds clean, 0 undefined citations).
   Structure: `numbers.tex` (every campaign number as a macro — future verify_wacv_numbers.py
   regenerates it), `sec/0_abstract..6_conclusion`, `main.bib` (28 entries pulled from the arXiv API
   by `make_bib.py` — titles/authors machine-fetched, not hand-typed). Full prose drafted for
   abstract/intro/related/method/analysis/limitations/conclusion; experiments section has the real
   9-script table + 5/9 Bbpe + red TODO slots for scaling/Khmer/VLM/figures.
   **Anonymity rules honored:** no names, receipts cited as "version-controlled archive, hashes in
   supp, anonymized for review". **TODO next:** figures 1/2/4 (make_wacv_figs.py), verify script,
   scaling-sweep prep (render 6480/12960 synth for mal/kan/tel), VLM baseline runner.
4. Devanagari/gurmukhi/bengali/gujarati Bbpe expected over ~the next day; at 9/9 update
   numbers.tex ratios + abstract/intro wording (grep for `TODO(bbpe)`).

---

## 13. SESSION 2026-07-18 (cont.) — AMENDMENT 5, SCALING SWEEP ARMED, VLM PREP, FULL AUTOMATION CHAIN

**Everything below was set up while Bbpe gujarati trained (ep6/15, 3.2 batch/s). The remaining
campaign now runs UNATTENDED end-to-end; a fresh session mainly needs to READ results and update
the paper. State of automation:**

```
LOSO (4 Bbpe rungs left) ──done(27/27)──▶ chain: VLM baseline (if .vlm_ready) ──▶ scaling sweep (12 runs)
   ▲ loso_guardian + @reboot                 ▲ .monitor/scale_chain.sh            ▲ scale_guardian + @reboot
```

1. **PREREGISTRATION Amendment 5 filed + pushed 47bc407 BEFORE any sweep data existed.** Reason:
   4a's "additional words" is impossible at the top budget — BSTD full-train pools are only
   mal 2393 / kan 2208 / tel 2215 records vs 3240/6480 word-slots needed. Fixed design: nested
   subsets 810⊂1620⊂3240⊂6480⊂12960; 6480 = +3240 new renders (previously-UNUSED train records
   ×2 first = "additional words", then cycled top-up); 12960 = +6480 more cycled renders (no new
   words exist — 6480→12960 is render-quantity at fixed lexicon, interpretation rule disclosed).
   Run order declared: 6480 → 1620 → 810 → 12960, each over (mal, kan, tel). Test text NEVER read.
2. **`PROSPECTIVE_PREDICTION_SCALING.md` filed + pushed (same commit):** 12 per-rung points from
   the frozen 9-pt instrument (+11.48/doubling, clip at Rung A, ±4.18/±8.36 bands) + fixed
   falsification rules (checkable: e.g. malayalam-810 ≥ ~8.5 falsifies linearity at the low end).
3. **Data BUILT & verified (`prepare_scaling_sweep.py`):** 29,160 new images (0 failures, fonts
   asserted identical to original meta), 24 splits+vocab files (`splits_zs_scale{B}_{tag}.json`),
   provenance in `scaling_sweep_meta_{tag}.json`. Spot-checked a render visually: correctly
   shaped Malayalam (raqm env used — NEVER system python for rendering).
4. **`run_scaling_sweep.sh`** mirrors the audited LOSO orchestrator exactly (result-JSON skip +
   `.train_done` sentinel + wait_gpu + same train/eval/metric); emits `[RESULT SCALE {B} {tag}]`
   → `result_zs_scale{B}_{tag}.json`. **`.monitor/scale_chain.sh`** (ARMED, flock'd) waits for
   LOSO 27/27 + orchestrator dead, then runs the VLM baseline in the free-GPU window, then
   launches the sweep. **`.monitor/scale_guardian.sh`** (ARMED + 2nd `@reboot` crontab line)
   relaunches the chain if everything dies with <12 scale results; retires at 12.
5. **VLM baseline (Amendment 4c) READY:** isolated venv `/c/ujjwalb/ritu1/.tools/vlm_venv`
   (transformers 4.51.3 over the SAME torch 2.5.1cu121 — the ritu_scenetext env was NOT touched:
   4.44.2 lacks Qwen2.5-VL and upgrading mid-campaign could break Florence-2 training).
   `eval_vlm_baseline.py` = declared protocol (fixed prompt, greedy, same metric, native-script
   gt) + a DISCLOSED lenient secondary score (substring/token credit — biased in the VLM's favor,
   anti-strawman). `run_vlm_baseline.sh` refuses to start unless ≥20 GB GPU free. Qwen2.5-VL-7B
   (~16 GB) downloading in background → touches `.monitor/.vlm_ready` on success; if absent the
   chain skips VLM (rerun manually later — resumable per tag).
6. **Khmer (4b) derisked:** KhmerST GitLab repo verified publicly clonable
   (gitlab.com/vannkinhnom123/khmerst, 1,544 images, line-level polygons → word-crop derivation
   is the disclosed engineering step). Not cloned yet — dedicated timeboxed session, drop-dead
   ~Aug 8, PROSPECTIVE_PREDICTION_KHMER.md must be pushed BEFORE its training.
7. **Watch items:** disk at 95% (106 GB free at session; sweep ckpts ~30 GB + Qwen 16 GB fit,
   but check `df` each session; pruning old epoch_N snapshot subdirs of COMPLETED rungs is the
   safe lever if needed — never touch best_model or .train_done). Monitor line counters are
   log-relative (log truncated at each relaunch) — completion may surface as
   `[WARN orchestrator STOPPED] N/27`; TRUST the result-JSON count (`ls result_zs_*|wc -l`).
8. **Next session checklist:** (a) `ls result_zs_loso_rung*|wc -l` (27?) and
   `ls result_zs_scale*|wc -l` (12?); (b) fill `paper_wacv/numbers.tex` TODOs (grep `TODO(`),
   update Bbpe wording at 9/9 + scaling Fig. 3 at 12/12, score PROSPECTIVE_PREDICTION_SCALING;
   (c) `result_vlm_qwen25_*` → paper §4.5 table; (d) figures (`make_wacv_figs.py` to write) +
   `verify_wacv_numbers.py`; (e) Khmer session; (f) commit results at rung boundaries.

---

## 14. SESSION 2026-07-19 — LOSO 27/27 DONE; SWEEP RUNNING; PAPER NUMBERS LOCKED + VERIFIED

1. **LOSO COMPLETE 27/27 at 03:16 IST** (orchestrator log tail + 27 result JSONs). Bbpe 9/9:
   direction 9/9 for the grapheme pivot (exact two-sided sign test p=0.0039); B/Bbpe ratios
   1.08× (devanagari) … 3.91× (tamil). The "2–4×" framing is retired everywhere; wording is now
   "all nine scripts, 1.08–3.91×".
2. **Exploratory finding (post hoc, labeled as such in the paper):** fertility predicts the
   BPE *penalty* even though it failed to predict transfer level — Spearman(fertility, B/Bbpe)
   = +0.73, exact permutation p=0.031; on the absolute difference it drops to +0.28 (denominator
   confound disclosed). Added to §5.1 of the manuscript as a candidate regularity, not a law.
3. **Scaling sweep auto-launched by scale_chain at 03:20** (guardian relaunch path worked).
   2/12 at this session: mal-6480 = 11.7 WRR (prereg point 21.2 — faithful-step gain +2.0 vs
   predicted +11.5, outside ±2·RMSE for the script), kan-6480 = 19.17 (point 26.9, gain +3.75,
   outside ±1·RMSE). Two consecutive undershoots → saturation/lexicon-exhaustion signal per
   Amendment 5.4's disclosed interpretation rule; VERDICT WAITS for the 3-script mean (telugu
   6480 training since ~13:20). Score PROSPECTIVE_PREDICTION_SCALING.md only at 12/12.
4. **Manuscript updated to 9/9:** all TODO(bbpe) resolved (abstract, intro ×2, Table 1 cells,
   §4.2 rewritten, §5.1 exploratory paragraph). Contribution 2 retitled "The pivot is the
   mechanism" (1.08× devanagari makes "necessary" an overclaim). New macros: \wrrP*/\chaP* ×4,
   \bpeRatioMin/Max, \bpeSignP, \spearFertPenalty{,P,Diff}. Compiles: 7 pp, 0 unresolved refs.
5. **verify_wacv_numbers.py WRITTEN AND PASSING: 69 macros re-derived from result JSONs,
   0 mismatches** — including independent reproduction of spearFert −0.80, the two-factor fit
   (−35.79 / 0.583 / 11.48, RMSE 4.18), covRaw +0.37 / covEqual +0.61, and the exact
   permutation p for the exploratory correlation. Run before every submission build.
6. **Housekeeping:** pruned epoch_* snapshots of all 29 completed rungs (sanctioned lever;
   best_model + .train_done kept) → disk 77→114 GB free. Stale loso_guardian @reboot line
   removed from crontab (27/27). The two .partial_* forensic dirs left untouched.
7. **Qwen download was INCOMPLETE (shards 1–2/5 missing) — that's why .vlm_ready was never
   touched and the chain skipped the VLM window.** Fixed: .monitor/resume_qwen_download.sh
   resuming in background (touches .vlm_ready after verifying all shards against the index);
   .monitor/vlm_chain.sh ARMED (+ @reboot line): waits for sweep 12/12 + orchestrator dead +
   .vlm_ready, then runs run_vlm_baseline.sh in the freed GPU; retires on
   result_vlm_qwen25_done.
8. **Next session:** (a) at 12/12 — score the scaling prereg (hits AND misses), write §4.2
   dose–response + Fig 3; (b) confirm result_vlm_qwen25_* landed → §4.5 table; (c) figures
   (make_wacv_figs.py: Fig 2 bars from result JSONs, Fig 4 receipts timeline) + bootstrap CIs
   for Table 1; (d) Khmer session (clone KhmerST, crops, PROSPECTIVE_PREDICTION_KHMER.md pushed
   BEFORE training; drop-dead ~Aug 8); (e) freeze Aug 14 → enroll Aug 21 → submit Aug 28.

---

## 15. HANDOFF FOR NEXT SESSION (filed 2026-07-19 ~14:00 IST) — STATE, AUTOMATION, EXACT COMMANDS

**Read §14 first for what changed today. This section = everything a cold session needs.**

### 15.1 What is running RIGHT NOW (nothing needs a human until 12/12)
- **Sweep 2/12 done** (mal-6480 = 11.7, kan-6480 = 19.17), **telugu-6480 training** since ~13:20
  (PID family 4093400, ritu_scenetext env). ~5 h/run incl. eval ⇒ **12/12 expected ~Jul 21 late**.
  Declared order (Amendment 5): 6480 → 1620 → 810 → 12960, each over (mal, kan, tel).
- **Qwen2.5-VL-7B resume download** running (`.monitor/resume_qwen_download.sh`, launched 13:33);
  the two missing shards were ~2.9 GB fetched at filing. On verified completion it touches
  `.monitor/.vlm_ready`. If it died: rerun the script — idempotent, resumes blobs.
- **`.monitor/vlm_chain.sh` ARMED** (flock + @reboot): waits for scale 12/12 + orchestrator dead
  + `.vlm_ready`, then runs `run_vlm_baseline.sh` (refuses <20 GB GPU free), touches
  `result_vlm_qwen25_done`, retires. Logs → `.monitor/vlm_chain.log`, `vlm_baseline.log`.
- **Crontab now (loso_guardian line REMOVED at 27/27):**
  `@reboot .monitor/scale_guardian.sh` · `@reboot .monitor/vlm_chain.sh`

### 15.2 Morning-check commands
```bash
cd /c/ujjwalb/ritu1/lstm_model
ls result_zs_scale*_*.json | wc -l          # 12 = sweep done
ls result_vlm_qwen25_* 2>/dev/null          # per-tag VLM results (after sweep)
tail -5 .monitor/vlm_chain.log .monitor/scale_guardian.log scaling_sweep.log
ls .monitor/.vlm_ready                      # exists = model verified on disk
python3 verify_wacv_numbers.py              # must stay 69/69 (extend for new macros)
df -h /c                                    # was 114 GB free after §14 pruning
```

### 15.3 At 12/12 — scoring is PRE-COMMITTED, follow it mechanically
Score against `PROSPECTIVE_PREDICTION_SCALING.md` (frozen rules, commit 47bc407):
per-rung points, clip-to-RungA rule at the low end, faithful-step rule = 3-script MEAN gain in
[+7.3, +15.7] hits / outside ±2·RMSE misses. Current tally: mal-6480 gain +2.0 and kan-6480
+3.75 both undershoot ⇒ if telugu follows, the linear form is falsified at the top and the
disclosed Amendment-5.4 interpretation (lexicon exhaustion vs render count) becomes §4.2's
finding. Misses are reported exactly like hits. Then: numbers.tex scaling macros + Fig 3 +
extend verify_wacv_numbers.py; VLM numbers → §4.5 table + macros.

### 15.4 Paper state (commit 90c65fc, pushed)
`paper_wacv/` compiles (tectonic, 7 pp, 0 unresolved refs); 9/9 Bbpe numbers locked; remaining
red TODOs = scaling / VLM / Khmer / Fig 1–4 / bootstrap CIs — nothing else.
`verify_wacv_numbers.py` 69/69 PASS; run before every build; extend it with every macro added.
Figures TODO: `make_wacv_figs.py` (Fig 2 bars from result JSONs, Fig 3 dose–response, Fig 4
receipts timeline — the unique figure). Any Indic text rendering: ritu_scenetext env ONLY (raqm).

### 15.5 Deadlines (WACV_STRATEGY.md §1/§4)
Khmer drop-dead ~Aug 8 (PROSPECTIVE_PREDICTION_KHMER.md committed+PUSHED before training) ·
arXiv ~Jul 27–31 once VLM lands · internal freeze Aug 14 · enroll Aug 21 · **submit Aug 28** ·
decisions Oct 9 (no rebuttal).

### 15.6 Version-control policy for this repo (unchanged, restated for a cold session)
- Commit at rung boundaries: result+conf JSONs, paper sources, live-log sections. Push same day
  (prospectivity receipts require pushed hashes).
- `.monitor/` is gitignored BY POLICY (recovery scripts, locks, logs); automation is documented
  here instead — §15.7 has verbatim copies of the two scripts added today.
- Synth image dirs + checkpoints stay ignored; splits/vocab JSONs are tracked (3cdd232 policy).
- **Do NOT commit Paper-A (IJDAR) files from this workflow.** Currently-dirty Paper-A files to
  leave alone: PAPER_HANDOFF.md, PROF_REPORT.md, draw_method_figure.py, figures/fig1–fig9*,
  make_figures.py, make_fig_replication.py, paper/ (incl. main.tex open in the IDE),
  cover_letter, train_florence2.py + training_log_florence2_*.json, verify_paper_numbers.py,
  conf_std_law_*, ensemble_control_*, audit_supplement.py, dataset_stats.json,
  learned_fusion_*, make_fig7_qualitative.py, paper_figures/, bengali_lab_dataset*.
- Before every commit that touches docs/paper text: grep the diff for AI/assistant wording.
- Never rewrite pushed history (breaks the 047c30c receipt chain).

### 15.7 Disaster-recovery copies of today's untracked automation (.monitor/ is gitignored)

`.monitor/vlm_chain.sh`:
```bash
#!/usr/bin/env bash
cd /c/ujjwalb/ritu1/lstm_model || exit 1
LOG=.monitor/vlm_chain.log
exec 9>.monitor/vlm_chain.lock
flock -n 9 || { echo "[$(date '+%F %T')] another vlm_chain holds the lock — exiting." >> "$LOG"; exit 0; }
[ -f result_vlm_qwen25_done ] && { echo "[$(date '+%F %T')] already done — retiring." >> "$LOG"; exit 0; }
echo "[$(date '+%F %T')] armed: waiting for sweep 12/12 + .vlm_ready" >> "$LOG"
while true; do
  n=$(ls result_zs_scale*_*.json 2>/dev/null | wc -l)
  if [ "$n" -ge 12 ] && ! pgrep -f 'run_scaling_sweep[.]sh' >/dev/null 2>&1 \
     && [ -f .monitor/.vlm_ready ]; then break; fi
  sleep 300
done
echo "[$(date '+%F %T')] sweep complete (${n}/12), model ready — running VLM baseline" >> "$LOG"
bash run_vlm_baseline.sh >> vlm_baseline.log 2>&1 \
  && { touch result_vlm_qwen25_done; echo "[$(date '+%F %T')] VLM baseline DONE" >> "$LOG"; } \
  || echo "[$(date '+%F %T')] VLM baseline FAILED — rerun manually (resumable per tag)" >> "$LOG"
exit 0
```

`.monitor/resume_qwen_download.sh`:
```bash
#!/usr/bin/env bash
cd /c/ujjwalb/ritu1/lstm_model || exit 1
export HF_HOME=/c/ujjwalb/.cache/huggingface
VPY=/c/ujjwalb/ritu1/.tools/vlm_venv/bin/python
$VPY - <<'PYEOF'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct")  # resumes partial blobs
import os, json
idx = json.load(open(os.path.join(p, "model.safetensors.index.json")))
shards = sorted(set(idx["weight_map"].values()))
missing = [s for s in shards if not os.path.exists(os.path.join(p, s))]
assert not missing, f"still missing: {missing}"
print(f"all {len(shards)} shards present at {p}")
PYEOF
if [ $? -eq 0 ]; then
  touch .monitor/.vlm_ready
  echo "[$(date '+%F %T')] qwen download verified — .vlm_ready touched"
else
  echo "[$(date '+%F %T')] qwen download still incomplete — rerun this script"
  exit 1
fi
```

---

## 16. HANDOFF FOR NEXT SESSION (filed 2026-07-22 ~16:30 IST) — WACV SUBMISSION PHASE

**§15 is stale runtime-wise; read this section first. Everything experimental except Khmer and the
running CRNN test is DONE. The campaign is now a submission project with a hard date: Aug 28.**

### 16.1 What changed since §15 (sessions of Jul 21 and Jul 22)
1. **Scaling sweep scored 12/12** (`SCALING_SWEEP_SCORING.md`, instrument `analyze_scaling_law.py`,
   raw in `scaling_sweep_scoring_raw.txt`). Magnitude **MISSED**: common slope **+2.235
   WRR/doubling** (95% CI [+1.89, +2.58]) vs the preregistered 11.48 — 5.1x outside the CI. But the
   **form is confirmed**: log-linear in budget (R^2 0.997/0.906/0.999) with a script-invariant slope
   (F(2,6)=0.92, p=0.45). Low-end collapse **falsified** (transfer is concave); kannada shows
   top-end saturation. Coverage-offset model does **not** generalize (OOS RMSE 7.29 vs 0.77
   in-sample; r=+0.03 on the six never-fitted scripts) — the paper must not rest on typology.
2. **Qwen2.5-VL-7B baseline done + fairly rescored** (`rescore_vlm_extracted.py`,
   `result_vlm_qwen25_extracted_*.json`): our Rung B beats the 7B VLM on **8/9** (VLM leads only
   Devanagari).
3. **Paper reframed** around pivot-mechanism + VLM context; scaling law reported as a prospective
   result **with its miss**; typology demoted to a disclosed negative. Per-script bootstrap CIs
   added (`bootstrap_cis.py`, seed 20260721; 6/9 pivot CIs strictly above BPE). Four real figures
   (`make_wacv_figs.py` -> figs 1-4; Wong colorblind palette + redundant markers, already compliant).
4. **Khmer prereg filed and PUSHED before any Khmer training** (`d3a3130`,
   `PROSPECTIVE_PREDICTION_KHMER.md`, point 16.1 WRR) and **KhmerST cloned** (`khmerst_data/repo`,
   7.7 GB, HEAD ff64017, 1,544 images / 31 jsons). **No Khmer build step has run yet.**
5. **WACV 2027 process researched from primary sources** ->
   `WACV2027_SUBMISSION_COMPLIANCE.md`. The four findings that change plans:
   (a) supplementary **may not contain results on additional datasets**, and R2 has no revision —
   so **Khmer must be inside the Aug 28 PDF or it ships without Khmer** (Aug 8 drop-dead confirmed);
   (b) *"you should not cite your public codebase"* — printed commit hashes deanonymize us
   (repo is public), so the receipts need SHA-256 commitments in the review version;
   (c) dual submission = 20% overlap with anything submitted elsewhere during Aug 28-Oct 9;
   (d) **author list is frozen at enrollment, Aug 21, permanently.**
6. **Track recommendation: Evaluations & Datasets** (new in 2027). Its call explicitly solicits new
   evaluation protocols, negative results, auditing and systematic analyses — i.e. our receipts and
   falsifications, which are invisible to the Algorithms rubric. Under Algorithms, "the pivot is
   precedented, the contribution is empirical" is a legal reject with **no AC backstop and no
   rebuttal**. Working assumption = E&D; final flip at enrollment (`\usepackage[review,datasets]{wacv}`).
   **Owner sign-off still needed.**
7. **Amendment 6 filed + pushed (`2f21868`) before any CRNN trained**
   (`PROSPECTIVE_PREDICTION_ARCHITECTURE.md`): CRNN_V3 on the same LOSO splits/pivot space/metric,
   tamil+telugu+oriya x A/B, vocab from TRAIN only, frozen 60-epoch budget. Answers the standing
   "is it Florence-2-specific?" objection in either direction; a null is declared ambiguous
   (capacity floor vs architecture-dependence) **in advance**.
8. **Two correctness fixes (`1860818`):** `\spearVissim` was a stale 6-point -0.14 with a TODO
   inside the 2312.10806 rebuttal paragraph -> re-derived over all nine points as **-0.12** and
   added to the verify script (**88 macros, 0 mismatches**). `verify_bib.py` audits every
   `main.bib` entry against the arXiv API with 429/5xx backoff — **28/28 exist** with matching
   titles/authors/years (florence2 + univkhmer confirmed on arxiv.org after rate limiting).
9. **Dual-submission overlap measured (`257b29c`, `overlap_audit.py`): 0.29% / 0.27% directional,
   0.00% in every WACV section, no shared figures.** Total shared substance = three Paper-A numbers
   (polarity 8/9, LOSO 90.9%, R^2 0.678) in two sentences. **Both papers can be submitted.**
   Open item: `sec/2_related.tex` says "Our preliminary study" with NO citation — replace with the
   author-kit's anonymized parallel-submission citation + anonymized IJDAR PDF in the supp.

### 16.2 What is RUNNING right now
- `run_crnn_generality.sh` (detached, `crnn_generality.log`), 6 rungs, ~20 min each.
  **1/6 scored: CRNN Rung A tamil = WRR 0.0 / CharAcc 13.38 / N=513** (prediction 1 holds so far).
  Rung B tamil training since 16:20. **All six expected by ~18:10 IST 2026-07-22.**
- Nothing else. GPU otherwise free after that; disk 77 GB free (96% used) — watch before Khmer synth.

### 16.3 Next actions, in priority order
1. **Score the CRNN rungs against `PROSPECTIVE_PREDICTION_ARCHITECTURE.md`** (all four predictions,
   hits and misses alike) -> new subsection + macros + extend `verify_wacv_numbers.py`.
2. **Khmer build** — the only item with a hard external deadline (**Aug 8**): add khmer to
   `synth_multiscript.py` (raqm env ONLY), render synth, apply `khmer_pivot_map.py`, derive
   single-word test crops from the line-level polygons, train A+B, score vs 16.1.
3. **PARSeq (IndicPhotoOCR, env `indicphotoocr`, `eval_indicphotoocr_parseq.py` pattern) on the nine
   LOSO test sets** = per-script supervised ceiling under OUR metric; then **Tesseract** (not
   installed; conda-forge, all nine Indic traineddata exist) = the "why not off-the-shelf OCR?"
   answer. Both inference-only.
4. **Hash-commitment scheme** for the receipts (SHA-256 + timestamps in supp; real hashes at
   camera-ready) and the **anonymized parallel-submission citation** for IJDAR.
5. **Writing surgery** (after the experiments land, so the abstract is written once): reviewer-facing
   preemption subsections, first-page decisiveness, negatives ordered as rigor, "what we do not
   claim", data-asset/ethics paragraph, remove the last `\TODO` (khmer slot), `\cref` audit.
6. **Owner decisions needed:** track sign-off (E&D vs Algorithms) and the **final author list before
   Aug 21** — irreversible after enrollment.

### 16.4 State-check commands
```bash
cd /c/ujjwalb/ritu1/lstm_model
ls result_crnn_zs_rung*.json | wc -l          # 6 = architecture test complete
tail -5 crnn_generality.log
/c/ujjwalb/.conda/envs/ritu_scenetext/bin/python verify_wacv_numbers.py   # must stay 88/88
/c/ujjwalb/.conda/envs/ritu_scenetext/bin/python overlap_audit.py         # must stay < 20%
/c/ujjwalb/.conda/envs/ritu_scenetext/bin/python verify_bib.py --quiet    # 28/28
nvidia-smi; df -h /c; git log --oneline -5
```

### 16.5 Dates (unchanged, from `WACV2027_SUBMISSION_COMPLIANCE.md`)
Khmer drop-dead **Aug 8** · internal freeze **Aug 14** · enroll (author list FINAL) **Aug 21** ·
submit **Aug 28** · supp **Aug 30** · reviews+decisions **Oct 9** (no rebuttal) · camera-ready
**Nov 2** · author registration recorded by **Nov 17** or the paper is dropped · conference
**Jan 4-8 2027**, Disney Springs FL (start the US visa process the day acceptance lands).

---

## 17. HANDOFF (2026-07-28) — CRNN scored+in-paper; Khmer BUILT + training armed (GPU-gated)

**Session summary.** Repo had sat idle 6 days. Two things landed:

### 17.1 CRNN architecture generality (Amendment 6) — DONE, pushed `db956b7`
All 6 rungs (finished 07-22, were unscored) scored vs the frozen prereg → **2 hit / 2 miss**
(P1 Rung-A floor HIT, P4 ordering HIT; P2 +2.0-lift MISS, P3 ratio-band MISS — only oriya clears
them). Direction B>A replicates 3/3 but magnitude ~11% of Florence-2, char-acc depressed in step →
reported as the prereg's pre-declared **ambiguous** verdict (architecture-dependence vs capacity
floor), backbone-limitation kept. New `\label{sec:crnn}` subsection + 13 `\crnn*` macros +
`ARCHITECTURE_SCORING.md`; `verify_wacv_numbers.py` now **101 macros, 0 mismatch**.

### 17.2 Khmer (Amendment 4b) — BUILT this session; training GPU-gated and armed
- **Recognition unit decided by data** (`KHMER_BUILD_DECISIONS.md`): 75% of KhmerST regions
  (2686/3563) are single-token → those are the word-level test set. Image-level 50/50 split (seed 42).
- **Built** by `prepare_zeroshot_loso_khmer.py` (raqm env, hard `features.check("raqm")` gate):
  **N_test = 1,252** word crops (`khmer_test_crops/`, git-excluded), **3,240 synth**
  (`synth_zeroshot_loso_khmer/`, git-ignored), sources = all 11 Brahmic langs (7,700 train).
  Splits `splits_zeroshot_loso_rung{A,B}_khmer.json`, vocabs, `zeroshot_loso_meta_khmer.json`.
- **tok_cov recheck (prereg gate): 88.01% vs 89.02%, Δ−1.01 → instrument point 16.11→15.52
  (−0.59 WRR, inside ±1 RMSE). NOT material → no re-file; frozen prediction 16.1 stands.**
- **Training armed, GPU-gated:** `run_khmer_loso.sh` (A+B only, NO BPE; `train_florence2.py`
  grapheme-injected → `predict_with_conf.py` → `evaluate_corpus`) launched detached; it self-gates on
  `wait_gpu` (≥12 GB free) so it never preempts labmate `harsh`'s jobs (GPU currently ~3 GB free).
  Guardian `.monitor/khmer_guardian.sh` (relaunches runner if it dies, retires at 2 results) running
  + `@reboot` cron added. Watch: `tail -f khmer_loso.log`; results → `result_zs_loso_rung{A,B}_khmer.json`.
- **When results land:** score vs prereg 16.1 → `KHMER_SCORING.md` + `\khmer*` macros + extend
  `verify_wacv_numbers.py` + fill `sec/4_experiments.tex` §sec:khmer (remove the `\TODO`) +
  `numbers.tex` khmer TODO + Fig-4 khmer point. Rung-A prediction: WRR < 5.

### 17.3 Anchors (Amendment: supervised ceiling + off-the-shelf floor) — DRAFTED + RUNNING
- **Tesseract floor** `eval_tesseract_anchor.py` (stock Tesseract 5 LSTM, psm 8; binary in the
  loca_accnt env, tessdata has all 9 Indic + **khm**): CPU, RUNNING now (`tesseract_anchor.log`,
  niced). Scored in pivot space (to_pivot / khmer_to_pivot → evaluate_corpus), identical pipeline to
  our model. Early: tamil 15.4 / telugu 13.76 / kannada 12.08 WRR (off-the-shelf per-script models
  edge our zero-real-image Rung-B on some scripts — honest reference, frame by the data). Outputs
  `result_anchor_tesseract_{script}.json` (+khmer).
- **PARSeq ceiling** `eval_parseq_anchor.py` (IndicPhotoOCR specialist, imports in ritu_scenetext;
  9 scripts, no khmer; oriya→odia, devanagari→hindi, gurmukhi→punjabi). GPU. Runner
  `run_parseq_anchor.sh` launched detached, **chained after Khmer** (waits for both khmer results,
  then GPU≥6GB) so it never contends with our own training. Outputs `result_anchor_parseq_{script}.json`.
- **When both land:** add `\parseq*`/`\tess*` per-script macros + a floor/ceiling column or table to
  `sec/4_experiments.tex` (the "why not off-the-shelf OCR?" + supervised-ceiling answer), extend
  verify, commit results.

### 17.4 Still open
E&D track flip + anonymized artifact mirror + SHA-256 commitments, 8-page compression/desk-reject
surgery. **Owner: track sign-off + author list by Aug 21.**

---

## 18. SESSIONS 2026-08-01 and 2026-08-03 — CAMPAIGN CLOSED; REVIEWER-RISK PASS

Filed 2026-08-03. §18.1–18.3 reconstruct the 08-01 session (which shipped five commits but no log
entry); §18.4–18.6 are the 08-03 reviewer-risk pass.

### 18.1 Khmer scored — Amendment 4b closed (commit `e3a0904`)
Trained 07-28 → 07-31 on the GPU-gated runner. N=1252. **Rung A 0.00 / Rung B 0.80 WRR**
(+5.85 CharAcc). Prereg point 16.1 → **miss at −3.66·RMSE**; Rung-A<5 → **hit**. Re-evaluating the
frozen instrument at the realized coverage (88.01 vs filed 89.02) moves the point to 15.52, so the
miss is not an artifact of the disclosed recheck. Receipt: `KHMER_SCORING.md`. Reported as a miss;
the instrument is **not carried forward as a contribution**. What survives: direction replicates
**10/10**, and the magnitude collapse locates the method's boundary. Doubles as the campaign's
cleanest counter to "data size beats typology" (same 1× budget, 88% coverage, still 0.80; n=1).

### 18.2 Anchors landed + contamination checked (`e3a0904`, `557ea38`, `5a5d8a2`)
Tesseract off-the-shelf floor (10 scripts incl. Khmer; macro-avg **14.35**, we win **8/9**, losing
only Tamil) and IndicPhotoOCR PARSeq supervised ceiling (macro-avg **66.26**; we recover **0.29**,
i.e. real labels buy **3.4×**). Merged with the VLM into one reference table. **`ANCHOR_SPLIT_HYGIENE.md`**
answers the train-on-test objection before it is raised: **0 source-image overlap** between BSTD
`Recognition/{train,test}` on all nine scripts, and our measured Tamil 76.61 sits at the published
~73 rather than far above it.

### 18.3 E&D flip + supplement (`7d340b4`, `26d2c0b`)
`\usepackage[review,datasets]{wacv}`; E&D abstract/intro swapped in with `*_algo.bak` kept for a
one-command revert (`ED_TRACK_FLIP.md`). `build_artifact_bundle.sh` → anonymised
`wacv_supplementary.zip` (6.2 MB, whitelist-not-blacklist, paths/owner scrubbed, `SHA256SUMS.txt`),
which backs the release claim **inside the supplement** and removes the need for a public mirror.
**Not logged at the time and left uncommitted for two days:** retitle, abstract rewrite, §3.4
honesty paragraph, §5 hedging, rebuilt figures. Folded into this session's commit.

### 18.4 ⚠ THE REPO IS PRIVATE — the prospectivity receipt is weaker than the paper claimed
Verified 08-03: `RITU940/-brahmic-str` returns **404 anonymously** (API and web); the account's 8
public repos do not include it; **GH Archive** (which records events only for public repos) has
**zero** events in the ±2 h window around the 07-22 prereg push, so it was private then too; and
`git log --format=%G?` returns **N** — the commits are **unsigned**. Software Heritage has never
archived it. Consequences: (a) printed hashes could not have deanonymised us anyway; (b) **no third
party ever observed the filings**, and git author/committer dates are settable, so publishing at
camera-ready yields provenance, **not proof**. `sec/3_method.tex` previously claimed order "becomes
independently verifiable against the public archive at camera-ready" — unsupportable, now rewritten
to state the limit and to name the fix a future campaign should use. **Nothing can be repaired
retroactively**: every run finished by 07-31, so an OSF/OpenTimestamps stamp made now proves
existence *after* the runs. Owner decision 08-03: **honest disclosure only**, no public mirror.

### 18.5 Reviewer-risk pass (owner's seven-item list)
1. **Title** → "Reading Unseen Scripts: A Zero-Real-Image Evaluation Protocol and a Scored
   Forecasting Campaign for Brahmic Scene Text" (the old "Predictable…" contradicted our own
   negative record).
2. **Abstract** 378 → 347 words; defense (protocol, released harness, "a tenth script needs a
   correspondence map and a font") fully established in ¶1, misses confined to ¶3.
3. **"Is this a dataset paper?"** preempted in three places: title, abstract ¶1 ("not a new
   dataset"), and the intro's contribution-class paragraph citing the track charter.
5. **§5 compressed**: printed correlations 12 → 9; the exploratory fertility→BPE-penalty ρ=+0.73
   paragraph cut to one sentence marked unregistered and "used for nothing here". The two-factor
   **fit equation is deliberately kept** — every later forecast was drawn from it, so removing it
   would make the scorecard unauditable — but is now printed "for audit" and then discounted.
6. Tamil losing to Tesseract is stated outright in §4.6 ("losing only Tamil"); family and CRNN
   caveats already in Limitations. No change.
7. **Figures, all four checked at final print size**: committed `fig3_scaling.pdf` printed the slope
   annotation **through** the Telugu legend entry (fixed: horizontal legend on top, annotation
   bottom-right, marker meanings moved to the caption); the same edit had moved fig4's
   "under-predicted" label into *its* legend (fixed); both figures overran the column by 1.35 pt
   (now `width=\linewidth`); and the scorecard figure was floating onto the **references page**
   (float moved earlier in `sec/5_analysis.tex`).

### 18.6 Dual-submission disclosure closed
`sec/2_related.tex` no longer says "Our preliminary study" with no citation. It now cites
`\cite{parallelsub}` — the author kit's prescribed anonymous form ("Authors. … Anonymized copy
supplied as supplemental material") — and states that the two papers share no experiment.
**`main.bib` is 29 entries now (was 28).**

### 18.7 State and what is still open
Build: **0 overfull, 0 errors, verify 155/155 macros, references start p8** (content fits with ~1
page of headroom regained). Nothing of ours is running; the GPU is the labmate's.
**Open, all owner-side or submission-day:**
- **Aug 21 enrollment** — author list frozen permanently; advisor sign-off on the E&D track (the tex
  is already flipped, and the OpenReview track must match); real paper ID replaces `\wacvPaperID`.
- **`\textcolor{red}{TBD}`** in the `parallelsub` bib note — fill the IJDAR submission ID, and add
  the anonymised Paper-A PDF to the supplement as `parallel_submission.pdf`.
- **Rebuild `wacv_supplementary.zip`** at submission (the current one predates these edits).
- Confirm Paper-A's actual submission status before Aug 28 (the 20% rule binds Aug 28 – Oct 9).

---

## 19. SESSION 2026-08-04 — FULL-PAPER AUDIT (15 findings, all fixed)

Owner asked for a whole-paper recheck. Read every section against the result JSONs. Four were real
errors, not polish:

1. **Conclusion contradicted §3.3.** It still read "every call is auditable by commit hash" — the
   claim §3.3 had just retracted (private repo, unsigned commits). Now: "every call ships in the
   supplement with its scoring receipt, under the provenance limits stated in Sec.~3.3."
2. **Limitations was false at the low end.** "\wrrBmin--\wrrBmax\% clears the OCR floor
   (\tessMean\%)" — Tamil (9.16) is *below* 14.35, which the paper admits two pages earlier. Now
   scoped to the macro-average with Tamil named as the exception.
3. **§4.3 mis-stated the character-accuracy pattern.** Claimed BPE "matches" pivot CharAcc on
   Telugu; it exceeds it (55.10 vs 54.43). Checked all nine: BPE ≥ pivot on exactly Telugu,
   Kannada, Devanagari. Reworded to "equals or exceeds" over those three.
4. **Table 1 caption vs §4.3 disagreed** on what \bpeCIdisjoint means — caption said the Rung-B CI
   clears the BPE *point*, text said the *interval*. `bootstrap_cis.py` computes `blo > phi`
   (interval disjointness), so the caption was understating the measurement. Caption corrected.

Typos/rendering: intro contribution 4 was missing "of nine" after \vlmScriptsWon; Table 2's caption
printed **"~~73%"** (`${\sim}\parseqFinetuned\%$` where the macro already carries `{\sim}`);
"most-favourable" ×2 normalised to US spelling.

Precision: "hardest script (Malayalam)" → "script with least to borrow" (Tamil actually scores
lower, 9.16 < 9.69); three bare correlations in §2 given their $\rho=$ labels; thousands separators
made uniform (comma-free, matching the verified N macros); "effectively solved" hedge restored in
the abstract.

**Verification-claim integrity — the important one.** §3.3 promises the harness re-derives *every
derivable number*, but four derived values sat in the prose as literals the harness never checked:
the 1.16× near-parity ceiling, the 0.46 Rung-A breach, the 1.5σ Gurmukhi miss and the 0.0% VLM raw
primary. All four macro-ised; `verify_wacv_numbers.py` extended with a near-parity check, a
Gurmukhi-σ derivation from the filed ±1σ band, and a VLM raw-primary check that *asserts*
uniformity across all nine rather than trusting the literal. **155 → 158 macros, 0 mismatches.**

Clean on inspection: anonymity (no name/affiliation/repo URL/local path in any .tex or .bib), all 28
real bib entries verified against arXiv (`parallelsub` flagged MANUAL by design), no
cited-but-undefined keys, no repeated words, no unbalanced math, 0 overfull boxes, 0 undefined refs.
CRNN lifts re-checked against the raw JSONs (Rung A is exactly 0.0 on Tamil/Telugu, so reusing the
Rung-B macro as the lift is arithmetically correct; Oriya has its own \crnnLiftOriya). Rung-A really
does track coverage (Spearman +0.68, computed this session). Content ends p8, refs p8–10.

`make_overleaf_zip.sh` added: the committed `wacv_paper_overleaf.zip` had been **stale since Aug 1
17:55** — old title, 28-entry bib, the colliding fig3 — so uploading it would have shipped the draft
we were fixing. The zip is now a gitignored build product; re-run the script after any edit.

**Still open (owner/submission-day):** red `TBD` submission ID + `parallel_submission.pdf` in the
supp; `\wacvPaperID`; rebuild `wacv_supplementary.zip`; advisor track sign-off and the **Aug 21**
author freeze. Optional: `fig2_bars.pdf` is orphaned (dropped in the 8-page fit) and there is now
~1 page of headroom to reinstate it.

### 19.1 Fig. 2 reinstated (2026-08-04, owner request)
The per-script A/B$_{\mathrm{bpe}}$/B bar chart is back as **Fig. 2** (numbering now: 1 pivot,
2 bars, 3 scaling, 4 receipts). It ships as a **`figure*`** (full width): the PDF is 6.9 in natural
and `\textwidth` is 6.875 in, so the old single-column `figure` was scaling it to ~47% and rendering
the tick labels at ~3 pt — that, not the page budget alone, is why it read badly before.

**It did not fit for free.** A full-width float cost more than the page had spare: content ran to
p9. Reclaimed by (a) `fig_bars` figsize 2.5 → **1.4 in** (bars stay legible; fonts are absolute, so
only the plot area compresses), (b) a 2-line caption, and (c) cutting eight passages that each
**duplicate a statement kept elsewhere** — 4.1 "Table 1 is the campaign's core"; 4.2's "Both hold at
once" flourish; 4.3's fertility-margin sentence (now carried by 5.1's post-hoc line); 4.5's CRNN
closing restatement; 4.7's repeat of the GlotOCR count from the intro; §2's near-verbatim copy of
3.3's opening; 5.3's closing paragraph compressed (its three miss figures are restated in the
conclusion); and the conclusion's "we offer the recipe" line (duplicates the abstract's last
sentence). No evidence, number or caveat was removed — only restatement.

**Content now ends exactly at the bottom of p8, refs p9–10: ZERO slack left.** Any future addition
must be paid for by a cut. verify 158/158, 0 overfull, 0 undefined.
