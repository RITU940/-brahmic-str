# Bengali Scene-Text OCR — Progress Report

**Date:** 2026-06-07
**Author:** Ritu Baskey
**Target:** Strong journal (write to *Pattern Recognition* quality; submit to an attainable Q1/Q2 — Expert Systems with Applications / IJDAR / Neurocomputing / ACM TALLIP). Manuscript-complete by June 30; submission to follow.

---

## 1. Working title & contribution

**Grapheme-Aware, Semi-Supervised Vision-Language Modeling for Low-Resource Bengali Scene Text Recognition.**

Four contributions, each tied to a current open problem in the 2025 literature:

1. **Grapheme-cluster tokenization injected into a Vision-Language Model** (Florence-2), with a parameter-efficient (LoRA + `modules_to_save`) recipe that makes vocabulary extension trainable. Distinct from GraDeT-HTR (EMNLP 2025), which is handwritten text with a from-scratch decoder.
2. **Semi-supervised self-training** on 3,120 unlabeled real crops (pseudo-label → confidence-filter → retrain) — turns missing labels into a contribution.
3. **Grapheme-frequency-aware synthetic curriculum** that over-samples rare conjuncts, pretrained synthetic→real (attacks the synthetic-to-real gap named as open in the 2025 non-Latin STR review).
4. **Cross-script generalization** to Assamese (shares the Bengali script) + evaluation on public benchmarks **IndicSTR12 / BSTD** for comparability with PARSeq.

## 2. Dataset — verified and consolidated this week

| Item | Count |
|---|---|
| Total real Bengali scene-text crops | **7,378** (images ↔ GT, 1:1, integrity-checked) |
| Cleanly **labeled** (after dropping 284 `###` + 3,120 empty) | **3,974** |
| — Train / Val / Test (split **by document ID**, no leakage) | **3,207 / 397 / 370** |
| Unlabeled crops reserved for self-training | **3,120** |
| Distinct words / characters / graphemes | 2,169 / 148 / 762 |

Data was scattered across wrong locations (crops inside a checkpoint folder, GT duplicated in three places) and the old split was built on a smaller, partly-leaky set. All consolidated to a canonical layout; splits regenerated; 68 stale/leaky artifacts archived.

## 3. Status

- ✅ Dataset verified, consolidated, leakage-free splits regenerated, workspace cleaned.
- 🔄 **Retraining** standard-BPE and grapheme Florence-2 on the clean, larger set (in progress).
- ⏭ Next: semi-supervised self-training + synthetic curriculum; public-benchmark eval; full baseline table (zero-shot VLM, CRNN, PARSeq, commercial OCR); ablations; figures; manuscript.

## 4. Honest framing for the timeline

A strong-journal paper *published* by June 30 is not feasible (review cycles are months). Realistic internship deliverable: a **complete, submittable, first-authored manuscript** with all experiments, public-benchmark results, ablations and figures by June 30. We also intend a free arXiv preprint to establish priority, and — if funding permits — a parallel conference submission (low-cost options: virtual presentation, Indian venues, or a workshop).

## 5. Decisions requested from the advisor

1. Confirm venue target (recommend ESWA or IJDAR; *Pattern Recognition* as a stretch).
2. Approve the expanded scope (semi-supervised + synthetic + cross-script).
3. Conference in parallel? Any travel/registration funding possible, or target a virtual/low-cost venue?
