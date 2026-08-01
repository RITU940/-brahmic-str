# ANCHOR SPLIT HYGIENE — is the supervised ceiling contaminated?
**Checked:** 2026-08-01 (IST) · **Owner:** Ritu Baskey · **Companion:** `KHMER_SCORING.md`

## The question a reviewer will ask
Our supervised-ceiling anchor is the IndicPhotoOCR PARSeq per-language recognizer — built by the
**BSTD authors themselves**. Our LOSO test crops are the **BSTD `Recognition/test` split**. If
those specialists were trained on images that also appear in our test set, the "ceiling" is a
train-on-test number and means nothing. The first result (Tamil, **76.61 WRR**, vs our Rung-B
9.16) is high enough that the question must be answered before the number enters the paper.

## Check 1 — BSTD train/test are disjoint at the source-image level
Crops are named `<SOURCE_IMAGE>_<region_idx>.jpg`, so a shared photo is detectable even when no
crop filename repeats. Comparing `Recognition/train/<lang>` against `Recognition/test/<lang>`:

| script (BSTD dir) | train crops | test crops | filename overlap | **source-image overlap** |
|---|--:|--:|--:|--:|
| tamil (tamil) | 2029 | 513 | 0 | **0** |
| telugu (telugu) | 2215 | 545 | 0 | **0** |
| kannada (kannada) | 2208 | 720 | 0 | **0** |
| malayalam (malayalam) | 2393 | 547 | 0 | **0** |
| oriya (odia) | 3148 | 1044 | 0 | **0** |
| gujarati (gujarati) | 1884 | 1015 | 0 | **0** |
| bengali (bengali) | 4936 | 1368 | 0 | **0** |
| devanagari (hindi) | 14927 | 4846 | 0 | **0** |
| gurmukhi (punjabi) | 8310 | 2879 | 0 | **0** |

**Total source-image overlap across the nine: 0.** No test photo contributes a training crop.

## Check 2 — the measured ceiling matches the published supervised number
The BSTD paper reports PARSeq at ${\sim}73\%$ average WRR after fine-tuning on real labeled
images (`\parseqFinetuned` in `numbers.tex`, cited from~`bstd`). We measure **76.61** on Tamil
through our own pivot + metric path. Consistent with the published figure rather than inflated
far above it — what a leaked test set would produce.

## Check 3 — the two augmented test sets are ours, not theirs
Our Bengali (N=2873) and Devanagari (N=6042) test sets are larger than BSTD's test split
(1368 / 4846) because they include our own lab-collected crops. Those images are not in BSTD at
all, so the specialist cannot have trained on them either. This *lowers* the anchor's advantage on
those two scripts if anything; it cannot inflate it.

## Conclusion
The supervised ceiling is **legitimate, not contaminated**. It remains a *different-axis* number —
it presupposes exactly the real labeled target-script data whose absence defines our setting — and
the paper frames it that way. No caveat about test-set leakage is warranted, and we do not imply
one.

Reproduce: the per-script counts above come from listing `benchmarks/bstd/Recognition/{train,test}/<lang>`
and comparing crop-name stems (`'_'.join(name.split('_')[:-1])`).
