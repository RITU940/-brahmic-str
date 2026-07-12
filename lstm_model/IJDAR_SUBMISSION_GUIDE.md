# IJDAR SUBMISSION GUIDE — full procedure, first submission
**Paper:** Paper A — Script-Aware Dual-Tokenization Fusion (draft `paper/main.tex`)
**Target:** IJDAR (Springer/IAPR), submitted into the open special issue
"Computer Vision Systems for Document Analysis and Recognition" (deadline 2026-09-20).
**Researched:** 2026-07-12 (live pages). Companion: `PUBLICATION_STRATEGY.md`.

---

## 0. THE PIPELINE AT A GLANCE
submit → technical check (days) → editor desk screen (~1 wk; desk-reject possible)
→ reviewers invited → under review (1–4 mo) → decision letter
→ revision round(s) (97% of accepted papers have ≥1) → accept
→ licence signing + proofs → Online First publication (DOI, citable) → issue assignment.
Realistic wall-clock: acceptance ≈ late 2026–early 2027 if submitted July–Aug 2026.

**Key URLs**
- Journal home: https://link.springer.com/journal/10032
- Submission guidelines: https://link.springer.com/journal/10032/submission-guidelines
- Direct submission (Snapp system): https://submission.nature.com/new-submission/10032/3

---

## 1. BEFORE SUBMISSION — checklist
1. **Author list & order** — settle with supervisor (draft still says "co-authors TBD").
   Ritu = first author. Decide corresponding author (handles ALL journal contact,
   signs licence; needs a reliable, ideally institutional, email).
2. **ORCID** — free 16-digit researcher ID, https://orcid.org (5 min). Recommended
   at submission; get one for each author if possible.
3. **Manuscript format** (from the guidelines page):
   - Springer Nature LaTeX template, `[iicol]` two-column option; editable source
     files (the .tex + figures) are MANDATORY at every submission and revision.
   - **≤ 20 pages** (two-column, incl. figures); longer needs cover-letter justification.
   - Abstract **150–250 words** (current draft ~280 — trim). **4–6 keywords.**
   - Title page: title, all authors + affiliations (dept, institution, city, country),
     corresponding author marked with email, ORCIDs. IJDAR is **not double-blind** —
     names stay on the manuscript.
   - References: **numbered square brackets [1]**, DOIs as full links where available.
   - Figures: halftones ≥300 dpi, combination art ≥600 dpi, line art ≥1200 dpi;
     EPS/TIFF (or embedded in PDF via LaTeX); sans-serif labels; cite in order.
   - **Statements and Declarations** section before references: Competing Interests
     ("The authors have no competing interests…"), Funding, **Data Availability**
     (where code/data live — commit to a release), Ethics/Consent = not applicable.
   - Springer LLM policy: LLMs cannot be authors; AI-assisted copy editing needs
     no declaration.
4. **Cover letter (1 page):** what the paper shows (3–4 sentences), why it fits
   IJDAR + the CV-systems special issue, the delta vs GraDeT-HTR / MGP-STR,
   confirmation the work is original and not under review elsewhere, corresponding
   author contact. If >20 pp, justify here.
5. **arXiv:** Springer explicitly allows preprints **before or at submission**
   (not "prior publication"). Post Paper A to arXiv on submission day; disclose the
   arXiv ID/DOI in the submission form. NEVER post the final published PDF anywhere.
6. **Numbers audit:** every number in the paper traces to a result file on disk
   (standing rule). Springer runs automated plagiarism screening at technical check.

## 2. SUBMITTING (Snapp, step by step)
1. Journal page → **"Submit manuscript"** → Snapp; register/log in
   (use the email that will be the corresponding address).
2. Upload manuscript: **.zip of LaTeX source** (auto-compiles to PDF) or PDF + source.
3. Snapp auto-extracts title/abstract/authors → **verify every field**. Add each
   co-author with affiliation + email — co-authors receive verification emails
   they must acknowledge.
4. Fill declarations tabs: funding, competing interests, data availability,
   authorship confirmation, policy agreements.
5. **Special issue:** when asked about collections/special issues, select
   *"Computer Vision Systems for Document Analysis and Recognition"*.
6. Upload cover letter; some forms ask for suggested (non-conflicted) reviewers —
   people from the BSTD / IndicSTR / GraDeT-HTR orbit, no co-authors/colleagues.
7. Review the compiled PDF, **Submit**. Afterwards: tracking dashboard shows status
   (technical check → editor assigned → reviewers invited → under review → decision).

## 3. AFTER SUBMISSION — statuses & etiquette
- **Technical check bounce-backs** (missing declaration, figure dpi) are normal fixes,
  not rejections.
- IJDAR's "median 7 days to first decision" = the desk-screen stage. Full review:
  expect **1–4 months of "under review"**. Do not email the editor before ~3 months;
  then one polite status inquiry is acceptable.
- **Decision letters:** Reject (can't resubmit same paper) · Major revision ·
  Minor revision · Accept. **Major revision is the normal successful path** —
  it means "acceptable if you fix these".

## 4. REVISION ROUND (how to win it)
- Deadline typically 1–3 months (extensions granted on request).
- Write a **point-by-point response letter**: quote every reviewer comment, respond
  to each (do the experiment where feasible — the seed-ensemble control is
  pre-designed for exactly this), state what changed and where (page/line).
  Polite and factual always, even to a hostile review. Disagreement is allowed
  if argued with evidence.
- Submit revised source + response letter (+ a change-highlighted PDF helps)
  through the same Snapp record. Second round is usually faster.

## 5. ACCEPTANCE → PUBLICATION
1. Corresponding author signs the **Licence to Publish**.
2. **Open access choice:** decline (subscription route = **no fee**). The optional
   OA APC is £2,290 / $3,290 / €2,590 — unnecessary; arXiv provides free visibility.
3. **Proofs** arrive for approval — check numbers/names/affiliations carefully,
   respond within the stated window (typically days).
4. Paper appears **Online First** with a DOI (fully citable) before issue assignment.
5. Self-archiving: arXiv v1 (submitted version) stays up legally; the published
   PDF may never be posted; the accepted manuscript may be posted after a 12-month
   embargo (subscription route).

## 6. HARD RULES
- **No dual submission** — while under review at IJDAR, the paper may not be under
  review anywhere else (arXiv excepted; it is not a venue).
- Only the corresponding author communicates with the journal.
- Keep the supervisor in the loop at every stage; he must approve the submission.
