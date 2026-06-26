# Project Report: Bengali / Indic Scene-Text Recognition with Florence-2
**Ritu Baskey — internship project report, 2026-06-10**
*(Every number in this report comes from result files on disk in `lstm_model/`; nothing is estimated.)*

---

## 1. What the project is about

**Goal:** teach an AI model to read Bengali text in real-world photos (shop
signs, banners, posters — "scene text"), where current systems perform poorly,
and turn the method into a journal paper.

**Why it's hard:**
1. **Very little data.** English scene-text systems train on millions of
   labeled words; for Bengali only a few thousand exist publicly.
2. **The script itself.** Bengali uses *grapheme clusters*: a base letter plus
   vowel signs and *conjuncts* (two or three consonants fused into one shape,
   e.g. ক্ষ, শ্রী). Modern AI language models use "BPE" tokenizers built
   mostly for English text, which slice these clusters into fragments that
   don't correspond to anything visible in the image.

**Core idea of the paper:** make the model's tokenizer *script-aware*
(teach it whole grapheme clusters), and then exploit the discovery that the
BPE version and the grapheme version of the same model make *different
mistakes* — combining them by confidence beats both.

---

## 2. The datasets (exact, verified numbers)

### 2.1 Our own Bengali dataset (collected/annotated in the lab)
| Fact | Number |
|---|---|
| Total cropped word images | 7,378 |
| With non-empty labels | 4,258 |
| Labeled "###" (illegible — must be excluded) | 284 |
| **Clean usable labeled crops** | **≈3,970** |
| Unlabeled crops (used for semi-supervised learning) | 3,120 |
| Distinct word labels | 2,169 |
| Train / validation / test split | 3,207 / 397 / 370 |

**Important data work done first:** I audited every image against every label
file. The old train/test split had *leakage* (crops from the same original
photo in both train and test), which inflates scores. I rebuilt the splits
**by document** (all crops from one photo stay on one side). Honest scores
dropped at first — that is expected and correct.

### 2.2 BSTD — public benchmark (Bharat Scene Text Dataset, IIT Jodhpur)
Used so reviewers can verify our claims on data we didn't create.
| Language | Script | Train | Test |
|---|---|---|---|
| Bengali | Bengali | 4,443 (+493 val) | 1,368 |
| Assamese | Bengali | 2,365 (+262 val) | 1,505 |
| Hindi | Devanagari | 4,500 budget (+500 val) | 4,846 |

(Hindi train was subsampled to 5,000 with a fixed random seed so all
languages are compared at a similar "low-resource" scale; test sets are
always kept full.)

### 2.3 Synthetic data
~5,000 computer-rendered Bengali word images, where words containing **rare
graphemes** are deliberately over-sampled (weight = Σ 1/(frequency+1) over
the word's graphemes). This targets exactly the conjuncts the real data lacks.

---

## 3. The model and methods

**Base model:** Microsoft **Florence-2-base** — a "vision-language model"
(VLM): it looks at an image and *generates* text, like a small GPT with eyes.
We fine-tune it cheaply with **LoRA** (small trainable adapters; the full
model stays frozen except embeddings/output layer).

**The five method components:**
1. **Standard fine-tune (BPE branch):** Florence-2 + a Bengali-English BPE
   tokenizer (71,254 tokens).
2. **Grapheme branch:** we inject **640 Bengali grapheme-cluster tokens**
   into the tokenizer, so conjuncts become single units the model can output
   directly.
3. **Self-training:** the model labels the 3,120 unlabeled crops itself; we
   keep only the 2,441 labels it is ≥75% confident about and retrain.
4. **Synthetic rarity curriculum:** train with the rare-grapheme-weighted
   synthetic images (Section 2.3), then real data.
5. **Dual-tokenization fusion (the main discovery):** run multiple branches;
   each outputs its answer with a confidence score (from beam search);
   **keep the answer of whichever branch is more confident**. No training,
   no extra parameters.

---

## 4. Results (all on the clean, leakage-free test sets)

### 4.1 Our Bengali test set (370 images)
| System | Word accuracy (WRR) | Character accuracy |
|---|---|---|
| Tesseract OCR (fine-tuned) — classic baseline | 24.8 | 45.5 |
| CRNN — standard deep-learning OCR baseline | 44.1 | 69.4 |
| Florence-2 zero-shot (no fine-tuning) | 0.0 | — |
| Florence-2 fine-tuned (BPE) | 52.4 | 76.9 |
| + grapheme tokenization | 57.3 | 75.7 |
| + self-training | 57.0 | 78.0 |
| + synthetic curriculum (best single model) | **62.2** | **80.5** |
| **Fusion of all four models** | **64.6** | **82.8** |
| *Oracle (theoretical ceiling: magically pick the best model per image)* | *70.8* | — |

Total improvement over the standard fine-tune: **+12.2 word-accuracy points**.

### 4.2 Public benchmark — BSTD Bengali (1,368 test images)
| System | WRR |
|---|---|
| Florence-2 BPE | 57.4 |
| Florence-2 grapheme | 59.8 |
| **Fusion** | **62.1** (oracle 66.4) |

The same pattern on data we didn't collect → the discovery is not a fluke.

### 4.3 Cross-script generalization (running tonight)
| Test set | BPE | Grapheme | Fusion | Oracle |
|---|---|---|---|---|
| BSTD Assamese (1,505) | 31.6 | 36.3 | **38.5** | 43.8 |
| BSTD Hindi (4,846) | 56.9 | 52.6 | **61.7** | 63.1 |

Assamese has only 2,365 training words — the hardest low-resource setting —
and the pattern still holds (+4.7 grapheme, +2.2 fusion). Hindi (different
script, Devanagari) gave the most interesting result of the whole project:
the polarity FLIPS — there the BPE branch is stronger (Devanagari is well
covered by BPE vocabularies) — yet fusion gains the MOST (+4.8 over the best
single model, recovering 77% of the theoretical ceiling; p < 0.0001).
Conclusion: which tokenization wins depends on the script, but the two views
are always complementary, and confidence reliably picks the right one.

### 4.4 Head-to-head with the strongest specialist model (added 2026-06-12)

Reviewers will ask: "how do you compare to the best existing system?" The
BSTD authors released their own PARSeq recognizer — a specialist OCR model
**pre-trained on 8.48 MILLION synthetic Bengali word images** and then
fine-tuned on BSTD. We ran their released model on both test sets with the
exact same scoring code as everything else (no GPU needed — done on CPU):

| Test set | Their PARSeq | Our fusion | Verdict |
|---|---|---|---|
| Our Bengali test (370) | 66.8 | 64.6 | **Statistical tie** (p = 0.50) |
| BSTD Bengali test (1,368) | 77.3 | 62.1 | They win in-domain (expected) |

How to present this honestly (and we do, in §4.4 of the paper):
- On THEIR benchmark, their model wins clearly — 8.5 million training images
  beats our ~12 thousand. No surprise, and we say so.
- On OUR independent test set, their giant model is **statistically
  indistinguishable from ours** — even though we trained on about
  **1000× less data**. That is the data-efficiency headline.
- Our contribution is *orthogonal* to data scale: it's about tokenization
  inside vision-language models, and could be combined with their synthetic
  pre-training in future work.

### 4.5 Why fusion works (calibration analysis)
- The models' confidence genuinely predicts correctness: **AUROC 0.88–0.91**
  (1.0 = perfect, 0.5 = coin flip). That's why "trust the more confident
  model" works.
- **Selective prediction:** if the system may say "not sure" and pass hard
  images to a human, accuracy on what it does answer is **83.0% at 70%
  coverage** and **89.7% at 50% coverage**. This is the honest route to
  "80–90%" numbers and is exactly how real digitization pipelines operate.
- Caveat we report: confidences are *over-confident in absolute terms*
  (ECE ≈ 0.35), so we use them only for ranking, never as probabilities.

### 4.6 Honest negative result (strengthens the paper)
We tried a fancier fusion (ROVER character-level voting, the classic speech-
recognition method): **64.05 vs 64.59** for our simple max-confidence rule.
Reviewers like seeing that the simple design choice was tested, not assumed.

### 4.7 Error analysis
Bucketing test words by how rare their rarest grapheme is in training data:
all models degrade on rare-conjunct words (<5 training occurrences:
~33% for BPE), the synthetic curriculum dominates the mid-frequency buckets
it targets, and fusion helps most in the rarest bucket (39.1 vs 32.6).

---

## 5. What exists right now (deliverables)

| Deliverable | Where | Status |
|---|---|---|
| 7 publication-quality figures (PDF+PNG) | `lstm_model/figures/` | ✅ done |
| Full LaTeX manuscript draft v1 | `lstm_model/paper/main.tex` | ✅ complete (Hindi + PARSeq comparison included) |
| Verification README (every number → source file) | `lstm_model/paper/README.md` | ✅ |
| All trained models | `lstm_model/checkpoints_*` | ✅ (Hindi tonight) |
| Reproducible scripts (training, eval, fusion, calibration, figures) | `lstm_model/*.py`, `run_*.sh` | ✅ |
| Project brief for collaborators | `lstm_model/PAPER_HANDOFF.md` | ✅ |

**Figures:** (1) main results bar chart, (2) cross-dataset replication,
(3) error-complementarity breakdown (11.9% of images only the grapheme model
solves; 8.1% only BPE), (4) risk–coverage curves, (5) calibration/AUROC,
(6) grapheme-rarity buckets, (7) qualitative examples with real crops.

---

## 6. Novelty and positioning (literature check done)

Two close works exist and are cited honestly:
- **GraDeT-HTR (EMNLP 2025):** grapheme tokenizer for Bengali **handwritten**
  text — not a VLM, not scene text, no fusion.
- **MGP-STR (ECCV 2022):** fuses BPE/WordPiece granularities **inside one
  English model** — not script-aware, not separately trained, not low-resource.

**Our defensible claim:** *the first grapheme-aware dual-tokenization
confidence fusion in a vision-language model for low-resource Indic scene
text* — demonstrated on 3 languages / 2 scripts with a calibration-based
explanation of the mechanism.

---

## 7. Remaining timeline (internship ends June 30)

| When | What |
|---|---|
| ✅ June 11 | Hindi training + fusion done; all numbers in paper; figs 2–3 regenerated |
| ✅ June 12 | Specialist PARSeq head-to-head done (CPU only) and written into paper |
| This week | Polish manuscript, prof's feedback round |
| By June 30 | Complete submittable manuscript |

**If asked "why not 90% accuracy?":** with ~4k labeled words, even the oracle
over all four models is 70.8% — remaining errors are blur/occlusion/extreme
fonts that all models share. Character-level accuracy is 82.8%, and selective
prediction reaches 89.7% — those are the legitimate high numbers, and we
report the trade-off openly instead of inflating.
