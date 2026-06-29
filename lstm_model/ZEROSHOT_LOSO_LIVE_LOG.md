# ZERO-SHOT LOSO — LIVE RUN LOG & SESSION HANDOFF
**Purpose:** single source of truth for the *running* 9-script zero-shot LOSO experiment
(Part ② of the Brahmic-STR paper). A new-session agent should read **this file first**,
then `RESEARCH_STATUS_AND_PATH.md` (overall research status) and `PREREGISTRATION.md`.
This file is updated as each rung completes.

**Owner:** Ritu Baskey · **Machine:** server3 (`cvpr-gamma`, RTX A5000 24 GB) ·
**Started:** 2026-06-25 11:27 IST · **Last updated:** 2026-06-29 (session: lit/positioning, dataloader speedup, server1 scoped)

---

## 0. TL;DR FOR A NEW SESSION
- A long (~9-day, resumable) run is **in progress**: `run_zeroshot_loso.sh`, PID **1283525**,
  log `zeroshot_loso.log`. **18 rungs** = 9 held-out scripts × {Rung A, Rung B}.
- **Done so far (3/18):** Tamil A=0.0, **Tamil B=9.16** (transfer works), Telugu A=1.28.
  **Telugu Rung B training now** (~ep11/15 as of 2026-06-29 ~11:45).
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
| 2 | telugu     | 1.28 | — | — | — | 545  | 🟢 Rung B training (~ep11/15) |
| 3 | kannada    | — | — | — | — | 720  | queued |
| 4 | malayalam  | — | — | — | — | 547  | queued |
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
