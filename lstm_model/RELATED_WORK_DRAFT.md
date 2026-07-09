# Related Work — draft prose for the zero-shot paper (port to LaTeX §2)
**Draft:** 2026-07-09 · working text, citations by arXiv id (BibTeX later). Each subsection ends
with the differentiation sentence that must survive editing. Companion positioning notes at the end.

---

## 2.1 Multilingual scene-text recognition and Indic benchmarks

Scene-text recognition (STR) is mature for Latin script, where modern recognizers exceed 90% word
accuracy on standard benchmarks, but coverage collapses outside a handful of scripts. IndicSTR12
(2403.08007) and the Bharat Scene Text Dataset (BSTD, 2511.23071) established real-image benchmarks
for Indian languages; on BSTD, a PARSeq recognizer reaches ~47% average word recognition rate (WRR)
when trained on synthetic data only and ~73% after fine-tuning on real labeled images — numbers that
presuppose labeled data in the target script. GlotOCR Bench (2604.12978) recently quantified the
wider problem: across 100+ Unicode scripts, even frontier vision-language models read fewer than
thirty, and on unfamiliar scripts they emit random output or substitute characters from scripts they
know. *Our work targets exactly this regime — scripts for which no real labeled images are available
at all — and asks not only whether they can be read, but how well, predicted in advance.*

## 2.2 OCR in the VLM era

General-purpose vision-language models now lead text extraction on documents (e.g., Qwen-VL family,
2502.13923; GOT-OCR 2.0; DeepSeek-OCR), and a parallel line shows small unified specialist models
(0.5–3B parameters) matching or beating far larger generalists on reading tasks. Multilingual
coverage, however, remains thin: Qwen3-VL advertises OCR in 32 languages, and independent
evaluations report Qwen2.5-VL producing non-decodable Unicode for Indic outputs, leading to its
exclusion from Hindi and Bengali assistive-technology evaluations (2606.25084). *We show that a
0.23B-parameter open VLM (Florence-2), equipped with a structural grapheme pivot and rendered
synthetic data, reads Brahmic scripts that current frontier models garble — and we report a
frontier-VLM baseline on the same test sets to quantify that gap directly.*

## 2.3 Tokenization granularity for complex scripts

For scripts with large or compositional inventories, output granularity matters. MGP-STR (2307.13244)
fuses character, BPE, and WordPiece predictions inside one Latin ViT; GraDeT-HTR (2509.18081) and
BnGraphemizer show grapheme-level tokenization beating byte-BPE for Bengali handwriting. The
"Beyond Fertility" line (2510.09947) argues fertility is a weak tokenizer-quality metric for LLMs
and proposes STRR. Our Part ① confirms the grapheme-vs-BPE choice is not arbitrary but *predictable*:
across nine Brahmic scripts under equalized budgets, neutral fertility predicts which branch wins
(8/9 polarity, leave-one-script-out 90.9%) and how much fusion helps (R²≈0.70), while STRR is
degenerate (identically zero) for every Indic script — itself evidence that LLM-tokenizer metrics do
not transport to OCR. *Prior work establishes that granularity matters for single scripts; we
contribute the cross-script rule for when it matters, and then test whether the same structure
enables transfer.*

## 2.4 Reading unseen units vs. unseen scripts

A substantial line recognizes unseen *characters within a known script* by decomposing them into
shared sub-units: stroke- and radical-based zero-shot Chinese recognition (2106.11613; STAR,
2210.08490; HierCode, 2403.13761), and open-set recognition of novel characters across orientations
(MOoSE, 2407.18616; also 2204.05535, 2203.05179). Separately, zero-shot *detection* of unseen
scripts localizes but does not read text (2307.15991). In speech recognition, shared byte-level text
representations let ASR systems cover languages with unseen scripts (Maestro-U, 2210.10027) — a
cross-domain analogue of our pivot. *All of these transfer within a script or stop short of
recognition; to our knowledge no prior system reads real scene text of an entirely unseen script —
different Unicode block, different glyph inventory — from zero real target images, let alone with
its accuracy predicted beforehand.*

## 2.5 What predicts cross-lingual transfer

In NLP, transfer is known to correlate with lexical and subword overlap between source and target
(e.g., 2310.10378; 2405.12413), and language-relatedness alone is a poor guide. For STR specifically,
2312.10806 concludes that source dataset size and visual similarity, not linguistic typology, drive
cross-lingual transfer. Our findings sharpen rather than contradict this picture, in a regime that
work did not study (zero real target images, synthetic-only exposure): (i) a typology-style
structural property — fertility — indeed fails as a transfer predictor (Spearman −0.88 across nine
scripts), exactly as a data-centric view expects; (ii) the quantity of *synthetic target exposure*
dominates, a target-side data-size effect; (iii) within a fixed synthetic budget, coverage of the
shared grapheme space orders transfer (+0.77) — the OCR instantiation of the subword-overlap
principle; and (iv) rendered-glyph visual similarity, measured with a frozen encoder, predicts
nothing in our data (−0.14), separating our coverage effect from the visual-similarity account.
*Because our leave-one-script-out design balances source data exactly, the confound their study
identified is controlled by construction, and each of these claims was preregistered or filed as a
prospective prediction before the corresponding results existed.*

## 2.6 Scaling laws and performance prediction

Empirical scaling laws are established for supervised STR: performance follows smooth power laws in
model and data size for seen scripts (CVPR 2024, 2401.00028), and predicting model capability before
training is an active methodological line in language modeling (observational scaling laws;
data-mixing laws). *We extend scaling-law methodology to a regime it has not touched — zero-real-data
transfer to unseen scripts — and use the fitted law prospectively: the accuracy of the final held-out
script (and of an out-of-benchmark script, Khmer) is committed to a public repository, with error
bands, before the corresponding model is trained.*

## 2.7 Rigor: preregistration in empirical ML

The ML reproducibility literature identifies preregistration and negative-result reporting as the
most relevant practical remedies for result manipulation, HARKing, and seed cherry-picking
(aaai.70002; 2405.02200; 2311.18807), while noting that adoption at full study scale is rare. This
study was run under a frozen preregistration with an append-only amendment log; rank and point
predictions for unobserved results were committed and pushed in advance (verifiable by commit hash);
the preregistered primary hypothesis was falsified and is reported as such. *To our knowledge this
is the first fully preregistered, prospectively scored transfer study in text recognition; we offer
the protocol itself as a reusable contribution.*

---

## Positioning notes (must survive into the final text)

1. **The 2312.10806 paragraph is mandatory** (§2.5 above carries it). Deploy all three limbs:
   fertility ≠ typology-as-family-tree but is typology-adjacent and *loses honestly*; source data
   size is balanced by design; our regime (zero real target images) is one they did not study. Our
   synth-quantity finding *agrees* with their data-size thesis on the target side — framed as
   refinement, not conflict, it converts the #1 threat into supporting literature.
2. **"Different axis" sentence before every absolute-number table:** supervised BSTD numbers use
   real labeled target data; ours use none. The comparison figure is the frontier-VLM baseline, not
   supervised SOTA.
3. **Instantiation framing for coverage:** never claim discovery of "overlap predicts transfer" —
   claim its first OCR/scene-text instantiation, in a *designed* pivot space, with the
   visual-similarity control.
4. **The word "groundbreaking" never appears in the manuscript.** Claims stay one notch below what
   the evidence supports; the prediction receipts do the talking.
5. Disclose compute (single RTX A5000) and the synthetic→real domain gap in Limitations; keep the
   Gujarati over-coverage miss and the devanagari Rung-A 5.46 baseline breach visible in the text.
