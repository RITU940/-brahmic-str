# Assembles RESEARCH_SUMMARY_RITU.pdf — 2-page abstract-style overview for Prof.
# Every number sourced from result files on disk (see the docs they mirror).
from fpdf import FPDF

BASE = "/c/ujjwalb/ritu1/lstm_model"
OUT = f"{BASE}/prof_abstract/RESEARCH_SUMMARY_RITU.pdf"
DV = "/usr/share/fonts/truetype/dejavu"

INK, SEC, MUT = (11, 11, 11), (82, 81, 78), (137, 135, 129)
BLUE, RULE = (24, 79, 149), (225, 224, 217)

pdf = FPDF("P", "mm", "A4")
pdf.set_margins(13, 12, 13)
pdf.set_auto_page_break(True, 12)
pdf.add_font("dv", "", f"{DV}/DejaVuSans.ttf")
pdf.add_font("dv", "B", f"{DV}/DejaVuSans-Bold.ttf")
pdf.add_font("dv", "I", f"{DV}/DejaVuSans-Oblique.ttf")
CW = pdf.w - 26  # content width


def rule(gap_before=1.2, gap_after=2.0):
    pdf.ln(gap_before)
    pdf.set_draw_color(*RULE)
    pdf.set_line_width(0.25)
    pdf.line(13, pdf.get_y(), 13 + CW, pdf.get_y())
    pdf.ln(gap_after)


def head(kicker, title):
    pdf.set_font("dv", "B", 9.2)
    pdf.set_text_color(*BLUE)
    pdf.cell(11, 5.0, kicker)
    pdf.set_text_color(*INK)
    pdf.cell(0, 5.0, title, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(0.4)


def para(txt, size=8.6, color=SEC, style="", lh=3.85, indent=0):
    pdf.set_font("dv", style, size)
    pdf.set_text_color(*color)
    pdf.set_x(13 + indent)
    pdf.multi_cell(CW - indent, lh, txt)
    pdf.ln(0.65)


# ─────────────────────────────── PAGE 1 ───────────────────────────────
pdf.add_page()
pdf.set_font("dv", "B", 14.5)
pdf.set_text_color(*INK)
pdf.multi_cell(CW, 6.4, "Can a Model Read a Script It Has Never Seen?",
               new_x="LMARGIN", new_y="NEXT")
pdf.set_font("dv", "", 10.2)
pdf.set_text_color(*SEC)
pdf.multi_cell(CW, 4.8, "A predictive tokenization law and zero-shot cross-script transfer for Brahmic scene-text recognition",
               new_x="LMARGIN", new_y="NEXT")
pdf.ln(0.8)
pdf.set_font("dv", "", 7.8)
pdf.set_text_color(*MUT)
pdf.multi_cell(CW, 3.6, "Ritu Baskey  ·  research summary  ·  10 July 2026  ·  every number here comes from a result file on disk and is reproducible; nothing is estimated or projected",
               new_x="LMARGIN", new_y="NEXT")
rule()

pdf.set_font("dv", "B", 8.6); pdf.set_text_color(*INK)
pdf.cell(0, 4.2, "ABSTRACT", new_x="LMARGIN", new_y="NEXT"); pdf.ln(0.4)
para("Scene-text recognition — reading shop signs, banners and posters in photographs — is data-starved for Indian scripts: "
     "collecting labeled photographs for all nine Brahmic scripts is prohibitively expensive. This project builds a way around "
     "that bottleneck in three connected results. (1) A script-aware tokenization + confidence-fusion method lifts Bengali "
     "scene-text word accuracy from 52.4% to 64.6% and statistically ties a specialist model trained on roughly 1000× more data. "
     "(2) An empirical law: one pre-registered, computable property of a writing system (BPE fertility — how many subword tokens "
     "a standard tokenizer spends per written character unit) predicts, before any training, which tokenization wins on which "
     "script — correct on 8 of 9 Brahmic scripts, 90.9% under leave-one-script-out validation. (3) A shared \"abugida pivot "
     "space\" that maps all nine scripts into one output vocabulary, in which a model trained on eight scripts reads the ninth "
     "— a script it has never seen one real photograph of — at 9.2–30.1% exact-word accuracy, now demonstrated on ALL NINE "
     "scripts (base model: 0.0% on every one; no-exposure baselines ≤ 5.5%). The running ablation shows the stock tokenizer "
     "transfers 2–4× worse under the identical protocol — the pivot space is what carries the transfer.", size=8.7, color=INK)
rule()

head("1", "Scripts and data attempted so far")
para("All nine Brahmic scripts have been trained and evaluated in the supervised study, via the 11 Indic languages of the public "
     "BSTD benchmark (IIT Jodhpur): Bengali script (Bengali + Assamese), Devanagari (Hindi + Marathi), Gurmukhi (Punjabi), "
     "Gujarati, Oriya (Odia), Tamil, Telugu, Kannada and Malayalam — plus English (Latin) as a control. Added to this: our own "
     "lab-annotated Bengali set (7,378 crops; I re-built its train/test split after finding photo-level leakage in the old one), "
     "and font-rendered synthetic data for every script. In total 44 fine-tuned model variants, all on a single shared RTX A5000. "
     "The zero-shot study (§4–5) covers the same nine scripts: all nine complete as of 9 July (18 experiments); the "
     "BPE-ablation phase is running now.")

head("2", "Result 1 — script-aware tokenization + fusion (the completed Bengali paper)")
para("Base model Florence-2 (a small vision-language model), fine-tuned with LoRA. Injecting whole grapheme-cluster tokens "
     "(so a fused conjunct — two or three consonants written as one shape — is a single output unit) beats the stock BPE "
     "tokenizer; the two make different mistakes, and picking "
     "the more confident answer per image beats both. On our Bengali test set: Tesseract 24.8 → CRNN 44.1 → fine-tuned BPE 52.4 → "
     "best single model 62.2 → fusion 64.6% (+12.2 over standard fine-tuning). Replicated on public BSTD: Bengali 62.1, "
     "Assamese 38.5, Hindi 61.7 — on Hindi the polarity flips (BPE wins there), yet fusion still gains +4.8 (p < 0.0001). "
     "Head-to-head with BSTD's specialist PARSeq model (pre-trained on 8.48 million synthetic images): statistical tie on our "
     "independent test set (66.8 vs 64.6, p = 0.50) with ~1000× less training data. Model confidence genuinely predicts "
     "correctness (AUROC 0.88–0.91): in selective mode the system reaches 89.7% accuracy at 50% coverage. Full draft exists.")

head("3", "Result 2 — the tokenization-granularity law (why Hindi flips: it is predictable)")
para("That polarity flip is not noise — it is the phenomenon. With hypotheses, descriptors and statistics frozen in a "
     "pre-registration on 18 June (before any result existed), all 11 languages were trained under identical budgets "
     "(1,620 images each) in both tokenizations:", lh=3.8)
pdf.image(f"{BASE}/figures/law_main.png", x=13, w=CW)
pdf.ln(1.2)
para("One number predicts the winner: grapheme tokenization wins exactly where fertility is high (8/9 Brahmic scripts correct; "
     "the single miss, Gujarati, is a −0.39 near-tie). Under leave-one-script-out validation fertility scores 90.9% vs ≤ 82% for "
     "six competitor descriptors — including STRR, the predictor recent literature proposes instead, which degenerates to zero "
     "on every Indic script (itself a reportable finding). The fusion gain also grows with fertility (R² = 0.70; bootstrap CI on "
     "the slope excludes 0, pre-registered criteria met).", size=7.9, color=MUT, lh=3.5)

# ─────────────────────────────── PAGE 2 ───────────────────────────────
pdf.add_page()
head("4", "The shared representation: one \"abugida pivot space\" for all nine scripts")
para("What is the shared representation? A deterministic, invertible Unicode offset map. Because the Brahmic scripts descend "
     "from a common ancestor, their Unicode blocks are aligned: the same sound sits at the same offset in every block. Shifting "
     "each character by a fixed per-script constant maps every script into one shared block (Devanagari's), losslessly:", lh=3.7)
pdf.image(f"{BASE}/prof_abstract/fig_pivot.png", x=13 + (CW - 148) / 2, w=148)
pdf.ln(0.8)
para("A model trained to output pivot text learns ONE output vocabulary shared by all scripts: a script it never saw still lands "
     "on output units it already knows from the other eight — the structural prerequisite for zero-shot transfer. Round-trip "
     "fidelity was verified word-by-word over the full benchmark before use.", size=7.9, color=MUT, lh=3.4)

head("5", "Result 3 (complete) — reading a held-out script with zero real images, all nine scripts")
para("Leave-one-script-out protocol, pre-registered: hold out one script entirely; train on 7,000 real photos of the ten other "
     "languages (in pivot space). Rung A adds nothing else — the held-out script is never seen in any form. Rung B additionally "
     "adds 3,240 synthetic font-rendered words of the held-out script — still zero real photographs of it. Evaluate exact-word "
     "accuracy on the held-out script's real test photos.", lh=3.7)
para("So, to answer directly: when Telugu is read at 19.8%, Telugu itself was the held-out script. That model was trained on "
     "Bengali, Assamese, Hindi, Marathi, Gujarati, Punjabi, Kannada, Malayalam, Odia and Tamil, plus computer-rendered Telugu "
     "only; it never saw a single real Telugu photograph, and was tested on 545 real ones. The base model reads them at 0.0%.",
     style="B", color=INK, lh=3.7)
pdf.image(f"{BASE}/prof_abstract/fig_zeroshot.png", x=13 + (CW - 136) / 2, w=136)
pdf.ln(0.8)
para("Transfer is real and large on every script: word accuracy 9.2–30.1% from synthetic renders alone, i.e. 20–85% of each "
     "script's supervised reference (median ≈ 48%), with character accuracy up to 58%. The two scripts trained with double "
     "synthetic exposure (Bengali, Devanagari — a covariate disclosed in advance) are the top two results: transfer scales with "
     "synthetic quantity, which the pre-registered scaling study will now measure deliberately. The honest remaining gap is "
     "synthetic-to-real domain shift, which the few-shot rung (adding just 50–100 real words) will measure next.", size=7.9, color=MUT, lh=3.4)

head("6", "The falsifiable bet currently on the table")
para("The bet is now resolved, and the loser is reported in full: fertility — the pre-registered primary — is refuted as a "
     "transfer predictor (rank correlation −0.80 across all nine scripts; its decisive prediction, Malayalam best, landed near "
     "the bottom). Coverage won the committed head-to-head (+0.32 vs −0.71 over the seven predicted scripts) and orders transfer "
     "within a fixed synthetic budget (+0.61); synthetic quantity dominates overall. The visual-similarity control committed "
     "result-blind (c5e7c28) also fails to predict (−0.14), answering the literature's \"appearance drives transfer\" objection "
     "with data. The method now makes numerical calls: Gurmukhi's accuracy was predicted (16.2, band 8.2–24.1) and pushed to git "
     "3½ hours before its training began; it realized 22.16 — inside the committed band, and the miss direction (better than "
     "called) is reported as-is. Next call: Khmer, on an external benchmark.", lh=3.7)

head("7", "Status and trajectory")
para("20 of 27 pre-registered experiments are complete: the main 18 finished on 9 July; the BPE ablation is 2/9 in (stock "
     "tokenizer transfers 2–4× worse — the pivot is necessary). Next, all pre-registered in Amendment 4 before running: finish "
     "the ablation (~2 days), a synthetic-exposure scaling study on three scripts, an out-of-benchmark validation on Khmer "
     "(different country, different Unicode block, accuracy to be predicted and pushed before training), a frontier-VLM "
     "baseline, and the few-shot rung. Plan: arXiv preprint ~20 July; main paper to WACV 2027 (Round 2, 28 Aug); the completed "
     "Bengali paper to IJDAR now.", lh=3.7)

rule(0.6, 1.4)
pdf.set_font("dv", "I", 7.2)
pdf.set_text_color(*MUT)
pdf.multi_cell(CW, 3.15,
    "Integrity: hypotheses/descriptors/tests pre-registered before results (frozen 18 Jun; amendments git-timestamped); "
    "splits leakage-audited; pipeline audited end-to-end; negative results reported in full (fertility refuted as transfer "
    "predictor; Gujarati near-tie; STRR degenerate). All claims trace to result files in the repo.")

pdf.output(OUT)
print("pages:", pdf.pages_count if hasattr(pdf, "pages_count") else pdf.page, "->", OUT)
