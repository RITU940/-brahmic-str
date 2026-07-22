# WACV 2027 — PROCESS, REVIEW CRITERIA, POLICY TRAPS, AND OUR COMPLIANCE PLAN
**Researched:** 2026-07-22 (live fetch of every official WACV 2027 page this day) · **Owner:** Ritu Baskey
**Primary sources:** wacv.thecvf.com/Conferences/2027/{CallForPapers, Dates, AuthorGuides, ReviewerGuidelines,
ACGuides, SACGuides} + the local author kit (`paper_wacv/wacv-2027-author-kit-template/`).
**Companion:** `WACV_STRATEGY.md` (venue verdict/how-we-lose), `COMPETITIVE_POSITIONING_AND_LITERATURE.md`.
**Rule:** quotes below are verbatim from the official pages; everything else is marked as inference.

---

## 1. THE PROCESS (exact mechanics of the round we are entering)

WACV 2027 runs **two rounds, journal-style**: *"WACV 2027 will follow a two-round review process that
is similar to journal submissions to provide the authors with an additional chance to defend and/or
revise their submissions."*

**Round 1** (closed — deadline was Jun 26 '26; we did not submit): ≥3 reviewers → AC recommendation of
**Accept / Revise and Resubmit / Reject**. R&R papers keep their paper ID, submit a revised PDF + 1-page
rebuttal, and *"The revised version will be sent to the same reviewers and ACs, after which the final
decisions will be made."*

**Round 2 = our round.** All dates **AoE**:

| Milestone | Date | Consequence for us |
|---|---|---|
| New paper registration (enrollment) | **Aug 21 '26** | **Author list is frozen here** (see §3.4) |
| Paper submission | **Aug 28 '26** | Everything that will be reviewed must be in this PDF |
| Supplementary material | **Aug 30 '26** | Cannot carry new results (see §3.1) |
| Reviews **and** decisions | **Oct 09 '26** | No rebuttal, no revision, no reviewer dialogue |
| Camera-ready | **Nov 02 '26** | De-anonymize; add acknowledgments |
| Author registration recorded | **Nov 17 '26** | Else *"omitted from proceedings"* |
| Conference | **Jan 4–8 '27**, Disney Springs, FL | Jan 6–8 main conf |

*"Completely new Round 2 submissions will follow a similar review process as Round 1, with the crucial
difference that Round 2 papers will not have a rebuttal and revision step."* Decisions are **Accept or
Reject only**.

**Also:** *"Rejected round 1 papers which are submitted to round 2 as new papers will be desk rejected."*
(Not applicable to us — we never submitted to R1. Our planned arXiv preprint does not create this problem;
arXiv is explicitly not a publication.)

**Decision machinery:** ≥3 reviewers → **AC pairs** (*"AC Pairs must meet to discuss the papers and reach
an accept, revise or reject decision"*) → SACs *"calibrating decisions across ACs"* and *"consult with the
program chairs (PCs) on particularly borderline or difficult paper decisions."* No published quota or
target acceptance rate at any level. WACV 2026: 2,458 submissions, ≈37.8% accept (round-level rates are
not published — **unknown**, do not assume R2 is easier or harder).

**Inference (ours):** with two ACs and no rebuttal, the paper must be *self-evidently* correct and
important to a fast reader who reads the reviews plus our abstract/intro/figures. Anything that needs a
rebuttal to survive is already lost.

---

## 2. WHAT REVIEWERS ARE TOLD — the parts that help us and the parts that can kill us

### 2.1 Explicitly in our favour (quote these to ourselves while writing)
- *"the fact that a proposed method does not exceed the state-of-the-art accuracy on an existing benchmark
  dataset is not grounds for rejection by itself. Rather, it is important to weigh both the novelty and
  potential impact of the work alongside the reported performance."* → our 9–30 WRR vs supervised ~73 is
  survivable **if** the different-axis framing is unmissable.
- *"We recommend that you embrace novel, brave applications and concepts, even if they have not been
  tested on many datasets."* → protects N=9(+Khmer) scope.
- *"Claims in a review that the submitted work 'has been done before' MUST be backed up with specific
  references and an explanation of how closely they are related."* → a lazy "grapheme tokenizers exist"
  reject is out-of-policy **if** we have already named and differentiated the closest works ourselves.
- *"Failing to cite an arXiv paper or failing to beat its performance SHOULD NOT be SOLE grounds for
  rejection"* and reviewers *"SHOULD NOT reject a paper solely because another paper with a similar idea
  has already appeared on arXiv."* → covers Task-Analogies HTR (2604.09713, Apr 2026) and any July-2026
  preprint that lands before Aug 28.
- *"Minor flaws that can be easily corrected should not be a reason to reject a paper."*
- Reviewers *"may optionally check this code to ensure the paper's results are reproducible and
  trustworthy"* → our verify-script + result JSONs are a trust asset if shipped as anonymized supp.

### 2.2 What gets papers marked down
- Not *"technically sound"*; no *"application or algorithmic contribution"* (track-dependent, §4).
- *"Inadequate citations of data assets or prior work."* → BSTD and KhmerST must be cited as data assets
  with versions/licences.
- Ethics: personal data / human subjects without IRB clearance; use of **withdrawn datasets**
  (DukeMTMC-ReID, MS-Celeb-1M named) → reviewers are told to escalate to ACs/PCs, not decide themselves.
- Reviews must be *"a list of strengths and weaknesses"*; short reviews are called out as irresponsible —
  **inference:** expect substantive, itemised criticism from at least one reviewer, and expect the AC pair
  to weigh the *written* weaknesses more than the score.

### 2.3 Track-specific criteria (verbatim)
- **Algorithms:** *"algorithmic novelty and quantified evaluation against current, alternative approaches."*
- **Applications:** *"systems-level innovation, novelty of the domain and comparative assessment"* — and the
  AC guide tells ACs to **overrule** reviewers who reject an Applications paper merely for lacking
  algorithmic novelty.
- **Evaluations & Datasets (NEW in 2027):** *"must advance the science and practice of evaluation in
  computer vision including the development and use of datasets and other resources."* The AC guide
  acknowledges *uncertainty* with this new track and asks ACs to take that into account.

---

## 3. THE FIVE POLICY TRAPS THAT APPLY TO **THIS** PAPER

### 3.1 Supplementary cannot carry Khmer (or any new result) — **hard planning constraint**
Verbatim: supplementary *"may not include results on additional datasets, results obtained with an improved
version of the method, or an updated or corrected version of the submission PDF."* Combined with
"R2 has no rebuttal and revision step", this means:

> **Everything — Khmer §4.4, Fig. 4 Khmer point, every number — must be inside the 8-page PDF uploaded
> Aug 28.** The Aug 30 supp deadline is *not* a two-day extension for content, and there is no later
> opportunity. The existing **Khmer drop-dead of ~Aug 8** is therefore correct and non-negotiable; if the
> Khmer rung is not scored by then, we ship without it and the prereg file is scored in the journal version.

Supp *can* hold: proofs, extra figures/tables, deeper analysis **of experiments already in the paper**,
videos, and code (≤200 MB, PDF or ZIP; main PDF ≤50 MB).

### 3.2 The prospective-receipt hashes are an anonymity hazard — **desk-reject class**
Standard: reviewers must not be able to reasonably infer authorship; *"Violation of any of these guidelines
may lead to desk rejection."* FAQ, verbatim: *"To preserve anonymity, you should not cite your public
codebase."*

Our repo (`github.com/RITU940/-brahmic-str`) is **public**, and our differentiator is git commit hashes.
A hash printed in the paper or supp is a one-search deanonymisation channel (commit pages are crawlable,
and the hash appears in our public history).

**Plan (do all three):**
1. Keep the current anonymised wording in the body (*"filed in a version-controlled archive before each
   run; hashes in supplementary"*) but **do not print resolvable hashes** in the review version.
2. Instead publish a **cryptographic commitment**: SHA-256 digests of each prediction file plus the
   filing timestamps, in the supp. A digest proves the content existed and matches at camera-ready without
   pointing at a named repo.
3. At camera-ready (Nov 2) swap digests → real hashes + repo URL.
Alternative if we want reviewers to browse: an **Anonymous-GitHub / anonymised Zenodo mirror**; only if the
mirror carries no author strings anywhere (README, LICENSE, commit authors, JSON provenance fields).

### 3.3 Dual submission vs the IJDAR paper — **20% rule, active during Aug 28–Oct 9**
Verbatim: authors confirm *"no publication substantially similar in content (defined as having 20 percent
or more overlap) has been or will be registered or submitted to this or another conference, workshop, or
journal during the review period."* IJDAR Paper-A is a journal submission alive in exactly that window.

**Actions:** (a) keep the Part-① law material in Paper-B to the planned ≤½ page of motivation, and measure
the overlap (text + figures + tables) before submission; (b) use the mechanism the official kit prescribes
for exactly this case, verbatim from `sec/1_intro.tex`:
> *"If you are making a submission to another conference at the same time, which covers similar or
> overlapping material, you may need to refer to that submission in order to explain the differences...
> include the anonymized parallel submission as supplemental material and cite it as
> [1] Authors. 'The frobnicatable foo filter', F&G 2014 Submission ID 324, Supplied as supplemental
> material fg324.pdf."*
So: cite Paper-A as **"Authors. IJDAR submission ID <id>, supplied as supplemental material"** and ship an
**anonymised** copy of it in the supp ZIP. This simultaneously satisfies "explain the difference" and
avoids a non-anonymous self-citation. (c) Never write "our journal paper".

### 3.4 Enrollment freezes the author list — **Aug 21, and it is final**
Verbatim: *"Authors cannot be added or deleted after the paper enrollment deadlines, but only reordered"*;
*"The author list is considered final after the paper's earliest submission deadline."* Same rule that bit
Paper-A's Tier-3 authorship item. **Decide the full author list (advisor, any collaborator) before Aug 21.**

### 3.5 LLM policy + fabricated citations — **automatic-rejection class**
Verbatim: *"We welcome authors to use any tool that is suitable for preparing high-quality papers... we
expect papers to fully describe their methodology, and any tool that is important to that methodology,
including the use of LLMs, should be described also"*; *"It is not a defense to a charge of plagiarism or
of inaccuracy to argue that 'an LLM did it'. You are responsible for what you submit."* Papers *"which cite
non-existent material"* face rejection, potentially without review, and *"All text will be subjected to the
plagiarism checker."*

Nothing in our methodology is LLM-assisted in the disclosable sense (Florence-2 and Qwen2.5-VL are the
*objects of study* and are fully described), so no disclosure statement is owed. What **is** owed: a
**bibliography existence audit** — every entry in `main.bib` (28 machine-fetched from the arXiv API + the
hand-added venue strings) verified to resolve to a real paper with correct authors/venue/year, because a
single hallucinated-looking reference is a rejection risk under this policy. Write `verify_bib.py`.

---

## 4. TRACK DECISION — Algorithms vs Evaluations & Datasets

The template selects the track (`\usepackage[review,algorithms|applications|datasets]{wacv}`), and the
OpenReview track must match. Reversible until enrollment (Aug 21).

**The E&D bullet list reads like a description of our paper** (verbatim bullets):
- *"Propose new evaluation protocols, practices, or methodologies"* ← our zero-real-image rung protocol +
  prospective-prediction receipts.
- *"Present negative results, critical analyses, and use-case-inspired evaluations"* ← fertility refuted
  (−0.80), scaling magnitude missed 5.1×, coverage fails out-of-sample (r=+0.03), Devanagari baseline
  breach reported.
- *"Provide rigorous reproduction, auditing, and stress-testing of prior evaluations"* ← 87-macro verify
  script, seed-controlled re-derivation, BPE ablation as a controlled counterfactual.
- *"Systematic analyses of systems on novel datasets"* ← 9 scripts × 3 rungs + Khmer out-of-benchmark.
- *"Analysis of strengths, limitations, or failure modes of existing benchmarks"* ← the frontier-VLM
  comparison on BSTD under one metric.

**Algorithms** judges *"algorithmic novelty and quantified evaluation against current, alternative
approaches."* We satisfy the second half strongly (BPE ablation, Qwen2.5-VL, supervised anchors, Rung A).
The first half is the exposure: the grapheme pivot itself is precedented (GraDeT-HTR, BnGraphemizer), so a
reviewer can write *"the tokenizer idea is known; the contribution is empirical"* — and unlike the
Applications track, **there is no AC instruction to overrule that objection**, and no rebuttal to answer it.

**Recommendation (mine, to be confirmed with the advisor):** submit to **Evaluations & Datasets**, framed
as *"a protocol and a predictive instrument for zero-real-image cross-script transfer, with the forecasts
filed before the runs and scored including the misses."* Reasons: our rarest asset (prospective receipts +
scored failures) is an explicitly solicited E&D contribution but is *invisible* to the Algorithms rubric;
low absolute WRR is a non-issue under an evaluation rubric; and the first-year E&D pool is likely thinner
and its ACs have been told to be accommodating. Risks to manage if we go E&D: (a) reviewers may expect a
released resource — so we commit to releasing the synthetic-rendering pipeline, the splits/vocab files, the
Khmer word-crop derivation, and the verify script (all already exist); (b) the mechanism/ablation content
must be framed as *evidence*, not as the headline method.
**Choose Algorithms only if** we decide the pivot-based recipe is being sold as a method — in which case
the intro must lead with the recipe, not the receipts.

---

## 5. FORMATTING / DESK-REJECT CHECKLIST (verbatim rules, then our status)

Rule: *"Papers that are not properly anonymized, or do not use the template, or have more than eight pages
(excluding references) will be rejected without review."*

| # | Requirement | Our status (verified 2026-07-22) |
|---|---|---|
| 1 | ≤8 pages excl. references, WACV style | Content ends on p8 where **References** begin → compliant but **<1 page of slack** for Khmer §4.4 + fig |
| 2 | Official WACV 2027 kit, correct track option | `\usepackage[review,algorithms]{wacv}` ✓ (switch if track changes) |
| 3 | Anonymous: no names, no acknowledgments, no self-identifying links | Title block anonymous ✓; **acknowledgments must stay absent until camera-ready** ("Q: Are acknowledgements OK? A: No. Leave them for the final copy.") |
| 4 | No identifying metadata | `pdfinfo main.pdf` → Creator "LaTeX with hyperref", Producer "xdvipdfmx", **no author, no paths** ✓ |
| 5 | Self-citation in third person | *"Blind review means that you do not use the words 'my' or 'our' when citing previous work."* — audit §2/§5 for "our earlier"/"our law" phrasing |
| 6 | No visible TODO/placeholder text | **1 `\TODO` still live** in `sec/4_experiments.tex:122` (Khmer slot) + `numbers.tex` TODO comments |
| 7 | Paper ID filled in | `\def\wacvPaperID{*****}` — set to the real ID after enrollment |
| 8 | Supp ≤200 MB PDF/ZIP, anonymized code, no new results | Not built yet — see §6 |
| 9 | Colour-blind-safe figures (kit explicitly asks) | Paper-A figures were fixed; **Paper-B figs 1–4 not yet audited** |
| 10 | `\cref` for cross-references, 9-pt references | Kit convention; audit before freeze |
| 11 | Abstract ≤5000 chars on OpenReview; PDF ≤50 MB | Fine (187 KB) |

---

## 6. WHAT TO DO, IN ORDER (added to the campaign plan)

**Before Aug 8 (Khmer drop-dead, unchanged)**
1. Khmer build → train → eval → score against the pushed 16.1 prereg. In or out by Aug 8; §3.1 forbids a
   late add.

**Before Aug 14 (internal freeze)**
2. **Track decision** (§4) with the advisor; if E&D, rewrite abstract/intro contribution list and switch
   the template option.
3. **Anonymity pass**: replace any resolvable commit hash with SHA-256 commitments (§3.2); grep the whole
   `paper_wacv/` tree for `ritu`, `RITU940`, `brahmic-str`, `github`, institution names, and first-person
   self-citation.
4. **`verify_bib.py`**: assert every `main.bib` entry resolves (arXiv API / DOI / DBLP) with matching
   title+authors+year (§3.5).
5. Recompute and verify **`\spearVissim`** at 9 points and add it to `verify_wacv_numbers.py` — it is
   currently a stale 6-point number sitting inside the 2312.10806 rebuttal paragraph, and that paragraph is
   load-bearing.
6. **Overlap audit vs Paper-A** (§3.3); prepare the anonymised IJDAR PDF for the supp and the
   "Authors, submission ID …" citation.
7. **Data-asset paragraph**: BSTD + KhmerST cited with version and licence, one sentence on the ritu1
   split caveat, one sentence that all images come from public benchmarks (no new human-subject data,
   no IRB trigger, no withdrawn datasets).
8. Colour-blind audit of figs 1–4; remove the last `\TODO`.

**Aug 21 (enrollment)** — author list final; register the paper ≥1 week before Aug 28; set the real paper
ID in `main.tex`.

**Aug 28 (submission)** — `verify_wacv_numbers.py` must pass 100% on the exact PDF being uploaded; final
page check (≤8 excl. refs); upload supp ZIP Aug 30 (anonymised code + result JSONs + prereg digests +
anonymised parallel submission).

**After Oct 9** — accept → camera-ready Nov 2 (de-anonymise, add acknowledgments, real hashes, repo URL);
**author registration must be recorded by Nov 17** or the paper is dropped from the proceedings; note the
US-visa lead time from India for a Jan 4–8 conference in Florida — start that the day acceptance lands.

---

## 7. SOURCES (all fetched 2026-07-22)
- Call for Papers — https://wacv.thecvf.com/Conferences/2027/CallForPapers
- Dates — https://wacv.thecvf.com/Conferences/2027/Dates
- Author Guidelines (policies, FAQs, ethics, LLM) — https://wacv.thecvf.com/Conferences/2027/AuthorGuides
- Reviewer Guidelines — https://wacv.thecvf.com/Conferences/2027/ReviewerGuidelines
- Area Chair Guidelines — https://wacv.thecvf.com/Conferences/2027/ACGuides
- Senior Area Chair Guidelines — https://wacv.thecvf.com/Conferences/2027/SACGuides
- Author kit (local) — `paper_wacv/wacv-2027-author-kit-template/` (README, `sec/1_intro.tex` blind-review
  section, `sec/2_formatting.tex`)
- WACV size/acceptance history — https://papercopilot.com/statistics/wacv-statistics/
