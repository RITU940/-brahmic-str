# Pocket Handoff — Bengali Scene-Text OCR Paper (quick-resume notes)

## 1. Goal
Produce a **strong, honest, first-authored journal/conference METHOD paper** (not a survey)
on **Bengali scene-text word recognition** by fine-tuning the Florence-2 VLM, with several
*combined* novelties. Internship deadline: **manuscript-complete (all experiments + figures)
by 2026-06-30**; acceptance can come later. First author = Ritu (the user).

**Hard rules (do not violate):**
- **Results must be HONEST** — no fabrication, no cherry-picking, no inflated numbers. The user
  values correctness; a reviewer will catch fake numbers. Report leakage-free metrics only.
- **Do NOT kill other users' GPU jobs.** A lab user (`adminis+`) grabs the shared GPU. Always
  poll `nvidia-smi --query-gpu=memory.free` and require **>=12 GB free** before launching
  training (a 4 GB guard caused an OOM crash mid-run once). Set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Long runs go through `nohup ... &` with a log file; check the log tail for
  completion (a bare `&` gives no notification).
- PEFT eval gotcha: load adapter with `PeftModel.from_pretrained(bare_resized_base, ckpt)` —
  do NOT call `get_peft_model` first or it silently loads nothing. (Already correct in code.)

## 2. The contributions (the "story")
1. **Grapheme-cluster tokenization inside a VLM** — inject ~640 Bengali grapheme tokens into
   Florence-2's vocab (+ Banglish BPE tokenizer base). Aligns subword units with the script's
   orthographic units.
2. **Dual-tokenization confidence fusion** (the key discovery) — BPE and grapheme decoders make
   *complementary* errors; parameter-free max-beam-confidence fusion recovers >half the oracle gap.
   **Replicates on the public BSTD benchmark**, not just our data — this is the reviewer-convincing result.
3. **Grapheme-rarity-weighted synthetic curriculum** — oversample words containing rare conjuncts
   (weight = sum 1/(freq[g]+1)), render with Noto Bengali fonts. Single biggest lever (+5 WRR).
4. **Semi-supervised self-training** — pseudo-label 3,120 unlabeled crops via beam
   `sequences_scores` confidence (2,441 accepted at conf 0.75). Helps char-level metrics.
5. **Public-benchmark validation** — BSTD (Bharat Scene Text) Bengali, 1,368 test words.

## 3. Verified results (leakage-free, by-document split)
**Our test (370 words):**
| Model | WRR | Char-Acc | 1-NED | CER |
|---|---|---|---|---|
| Zero-shot Florence-2 | 0.0 | -5.9 | — | 106 |
| CRNN (CTC baseline) | 44.10 | 69.38 | — | 30.6 |
| Standard (BPE) | 52.43 | — | — | — |
| Grapheme | 57.30 | — | — | — |
| Self-train | 57.03 | 78.04 | — | 21.96 |
| **Synthaug (rarity curriculum)** | **62.16** | **80.51** | **82.93** | 19.49 |
| Fusion (BPE+grapheme) | 59.19 | — | — | — |
| **Fusion 4-way (all models)** | **64.59** | — | — | — |
| Oracle ceiling (4-way) | 70.81 | — | — | — |

**BSTD public test (1,368 words; trained on ours+BSTD):**
Standard 57.38 → Grapheme 59.80 → **Fusion 62.13** (oracle 66.45). PARSeq reference ~82%.

**Headline framing:** character-accuracy **80.5%** / 1-NED **82.9%**; **+12.2 WRR** over the VLM
baseline; fusion advantage **replicated on an independent public benchmark**. Do NOT claim
word-level 80-90% — it is not achievable honestly on this data (oracle is 70.8%); models make
correlated errors on hard/blurry/rare-conjunct crops.

## 4. Dataset facts
- `lstm_model/Bengali/` = 7,378 crop images; `lstm_model/Bengali_gt/` = 7,378 GT txt (1:1).
- Only **~3,970 cleanly labeled** (3,120 are 0-byte = unlabeled self-training pool; 284 are `###`).
- Splits: `florence2_splits.json` = 3207/397/370 by document (no leakage). Unlabeled pool:
  `unlabeled_pool.json` (3,120). BSTD: `florence2_splits_bstd.json` (4443/493/1368).
  Synthetic-augmented train: `florence2_splits_synthaug.json` (8,207 = 3,207 real + 5,000 synth).

## 5. Key files
- `train_florence2.py` — train/eval. Flags: `--use_graphemes --evaluate --splits_file --ckpt_dir`.
  Config: 15 epochs, batch 4, lr 1e-4, lora_r 16. (`--resume` declared but NOT implemented.)
- `prepare_florence2_data.py` — builds splits by document.
- `self_training_pseudolabel.py` — pseudo-label unlabeled pool.
- `generate_synthetic_curriculum.py` — rarity-weighted synthetic data + synthaug splits.
- `predict_with_conf.py` — writes `conf_<tag>.json` ({gt,pred,conf}) per test set.
- `fusion_analysis.py` — max-conf fusion + oracle. (Extend to N-way for the 4-model fusion.)
- `prepare_bstd_bengali.py`, `zeroshot_florence2_eval.py`, `train_crnn_on_florence.py` — baselines.
- Result files: `results_*_ours.json`, `results_{std,grph}_combined.json`, `conf_*_ours.json`,
  `fusion_*.log`. Checkpoints: `checkpoints_*_{standard,grapheme}`, `checkpoints_synthaug_grapheme`.

## 6. What's DONE vs TODO (updated 2026-06-10 night)
DONE: data consolidation, leakage-free splits, all model training (standard/grapheme/self-train/
synthaug/combined), all evals, fusion (3-way + 4-way), baselines (zero-shot/CRNN/Tesseract-FT/BSTD).
DONE 2026-06-10 (horizon broadening + paper assets):
- **Calibration analysis** (`calibration_analysis.py` -> `calibration_report.json`): conf AUROC
  0.88-0.91, ECE 0.35-0.41; selective prediction 83.0% acc @70% cov, 89.7% @50% (fusion4).
- **ROVER char-voting ablation** (`rover_fusion.py`): 64.05 vs max-conf 64.59 -> simple rule wins.
  NOTE: with 2 models ROVER reduces mathematically to max-conf.
- **Cross-script**: Assamese REPLICATES (std 31.63 / grph 36.35 / fusion 38.54 / oracle 43.79,
  n=1505). Hindi (Devanagari) training overnight via `run_crossscript_resume.sh` (machine reboot
  killed first run; assamese grapheme = converged epoch-12 best; `--resume` flag is dead code).
- **Significance** (`significance_tests.py` -> `significance_report.json`): grapheme>BPE
  significant everywhere (p=.041/.030/5e-5); fusion>best-single significant on public sets
  (p=.0024 Bengali, .0085 Assamese), NOT powered on our n=370 (p=.20) — stated honestly in paper.
- **All 7 figures** in `figures/` (`make_figures.py`, system python3; computes fusion live;
  PIL+raqm renders Bengali correctly). fig6 rarity buckets: fusion 39.1 vs BPE 32.6 in <5 bucket.
- **Manuscript draft v1**: `paper/main.tex` (Overleaf-ready) + `paper/README.md` (number->file
  audit map). Four [TBD-HINDI] slots to fill. `PROF_REPORT.md` = plain-language report for prof.
TODO (the remaining work):
- Fill [TBD-HINDI] in paper + add hindi to fig2/fig3 `extra_langs` + rerun `make_figures.py`
  + rerun `significance_tests.py` (auto-detects conf_*_hindi.json).
- Architecture/method diagram (fig for Sec 3) — draw.io or TikZ.
- Polish manuscript, prof feedback, submit-ready by June 30.
- **Optional accuracy lifts** (all honest, leave oracle headroom): scale synthetic 5k->30-50k,
  Florence-2-large backbone, 4-way fusion on BSTD (needs synthaug/selftrain inference on BSTD).
- Draft a short reply to the professor re: a low-cost/virtual/Indian conference + free arXiv
  preprint for priority (funding is tight).

## 7. Environment
- Python: `/home/ujjwal/miniconda3/envs/ritu_scenetext/bin/python`
- Workdir: `/home/ujjwal/ritu1/lstm_model`
- Model: `microsoft/Florence-2-base`, task prompt `<OCR>`, Banglish tokenizer
  `RocketFuel810/florence2-banglish-tokenizer` (vocab 71,254 -> 71,894 after grapheme inject).
- Metrics keys are UPPERCASE in JSON: `WRR`, `CER`, `char_accuracy`, `1-NED`.
