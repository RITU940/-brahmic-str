# ZERO-SHOT LOSO — LIVE RUN LOG & SESSION HANDOFF
**Purpose:** single source of truth for the *running* 9-script zero-shot LOSO experiment
(Part ② of the Brahmic-STR paper). A new-session agent should read **this file first**,
then `RESEARCH_STATUS_AND_PATH.md` (overall research status) and `PREREGISTRATION.md`.
This file is updated as each rung completes.

**Owner:** Ritu Baskey · **Machine:** server3 (`cvpr-gamma`, RTX A5000 24 GB) ·
**Started:** 2026-06-25 11:27 IST · **Last updated:** 2026-07-07 (session: auto-relaunch WORKED 2026-07-06; Kannada B=15.42 — prospective "~15" hit; malayalam Rung A training; monitor re-armed)

---

## 0. TL;DR FOR A NEW SESSION
- A long (~9-day, resumable) run: `run_zeroshot_loso.sh`, log `zeroshot_loso.log`.
  **27 rungs** = 9 held-out scripts × {Rung A, Rung B, Rung **Bbpe**} — the Bbpe BPE-baseline
  phase (same protocol, stock tokenizer) was appended 2026-07-02 per PREREGISTRATION §7/§8;
  it runs AFTER the main 18, tamil/telugu first. **✅ ALIVE again** — after the ~Jun 30 power
  outage + ~6 days of GPU contention, the §9 watcher auto-relaunched the run 2026-07-06 12:03 (see §11).
- **Done so far (6/27):** Tamil A=0.0, **Tamil B=9.16**, Telugu A=1.28, **Telugu B=19.82**,
  Kannada A=2.78, **Kannada B=15.42** — all 3 Rung-B points rank-ordered exactly by token coverage (P2).
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
| 4 | malayalam  | — | — | — | — | 547  | 🟡 Rung A training (started 2026-07-07 08:45) |
| 5 | oriya      | — | — | — | — | 1044 | queued |
| 6 | gujarati   | — | — | — | — | 1015 | queued |
| 7 | bengali    | — | — | — | — | 2873 | queued |
| 8 | devanagari | — | — | — | — | 6042 | queued |
| 9 | gurmukhi   | — | — | — | — | 2879 | queued |

**Raw result so far:** `result_zs_loso_rungA_tamil.json` =
`{"rung":"A","script":"tamil","N":513,"WRR":0.0,"CharAcc":4.8,"CER":95.2}`

**Append-only rung log:**
- `2026-06-25 23:08` — [RESULT LOSO RUNG A tamil] N=513 WRR=0.0 CharAcc=4.8 CER=95.2 (expected baseline).
- `2026-06-26 ~14:00` — [RESULT LOSO RUNG B tamil] N=513 WRR=9.16 CharAcc=39.0 CER=61.0 — **transfer works**: 0→9.16 WRR, 4.8→39.0 CharAcc with ZERO real Tamil images (vs pilot B=9.94, consistent).
- `2026-06-28 ~04:52` — [RESULT LOSO RUNG A telugu] N=545 WRR=1.28 CharAcc=23.32 CER=76.68 — Rung A baseline (slightly above ~0; some incidental cross-script coverage). Rung B started same time, training now.
- `2026-06-29 ~21:18` — [RESULT LOSO RUNG B telugu] N=545 WRR=19.82 CharAcc=54.43 CER=45.57 — **strong transfer, 2× Tamil** (Tamil B=9.16). 1.28→19.82 WRR, 23.3→54.4 CharAcc with ZERO real Telugu images. Second H3 data point.
- `2026-06-30 ~05:30` — [RESULT LOSO RUNG A kannada] N=720 WRR=2.78 CharAcc=19.25 CER=80.75 — Rung A baseline. Kannada Rung B started 05:56, batch ~1.3/s.
- `2026-06-30 ~10:04` — **RUN DIED** mid-Kannada-Rung-B at ep6/15 (best_model ep6 saved). Machine lost power → network down (HF DNS-resolution errors flood the tail) → orchestrator + training killed. No Kannada-B result JSON; **B re-runs from scratch on relaunch.** Discovered 2026-07-02.
- `2026-07-06 12:03` — **AUTO-RELAUNCH WORKED:** `gpu_relaunch_watcher.sh` saw 18,595 MiB free (≥12,000), relaunched `run_zeroshot_loso.sh` itself via nohup (PID 1937079) and exited. No human/agent involved — the §9/§10 recovery infra worked unattended.
- `2026-07-06 18:48` — [RESULT LOSO RUNG B kannada] N=720 WRR=15.42 CharAcc=46.37 CER=53.63 — third H3 point; full from-scratch re-train + eval in ~6¾ h (speedup confirmed at scale). **Prospective hit:** `PROSPECTIVE_PREDICTIONS_H3.md` (result-blind, commit `047c30c`) staked exploratory estimate "kannada ~15".
- `2026-07-07 08:45` — malayalam Rung A training started (orchestrator's `wait_gpu` rode out overnight contention, down to ~1 GB free). 1.8–1.9 batch/s, ep9/15 by 11:26 — healthy. **Malayalam is the decisive H3 script** (P1-fertility's best vs P2-coverage's worst).

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
inside this repo). Arm it with the Monitor tool, persistent:
```bash
bash /c/ujjwalb/ritu1/lstm_model/.monitor/loso_monitor.sh
```
It emits each new `RESULT LOSO` line, the batch/s of each new rung (to confirm the speedup),
and warns once if the orchestrator dies or all 18 rungs finish.

**⚠️ Working-file rule (owner directive 2026-06-29):** keep ALL scratch/temp/working files under
`ritu1/` (e.g. `.monitor/`). Do NOT use the harness default scratchpad — it points into
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

## 8. SESSION 2026-06-29 — WHAT CHANGED (full detail for a new agent)
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
and, once ≥12 GB frees, **relaunches the run ITSELF via nohup** (needs no internet, no agent; guards
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
— it relaunches the run itself when ≥12 GB frees, no agent/session needed. A session monitor watches
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

**Monitor re-armed** this session (session Monitor → `loso_monitor.sh`). ⚠️ NB: the relaunch
truncated `zeroshot_loso.log`, so the monitor's counter only sees post-relaunch RESULT lines
(max 22 of 27): completion will surface as `[WARN orchestrator STOPPED] 22/27`, not `[DONE]` —
interpret accordingly.

**Committed this session:** kannada-B result + conf JSONs, this log update, and
`.claude/settings.json` (allows background sessions to edit this working copy directly —
the live-log workflow requires it; owner-approved). **Deliberately NOT committed:**
`training_log_florence2_grapheme.json` — it is overwritten in place by the live malayalam-A
training (a mid-epoch snapshot would be misleading); commit it at a rung boundary. NB the
kannada-B in-memory training curve in that file has already been overwritten by malayalam-A;
the per-rung text log `train_zeroshot_loso_rungB_kannada.log` retains the full curve.
