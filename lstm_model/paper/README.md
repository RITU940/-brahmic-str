# Paper draft — how to build and what's verified

**Draft v2 (2026-07-12), IJDAR submission format.** Compile on
[Overleaf](https://overleaf.com): open the *Springer Nature LaTeX Template*
from the Overleaf gallery, replace its `main.tex` with `paper/main.tex`,
and upload the `figures/` folder (keep the `../figures/` layout or drop the
figure files next to `main.tex` and delete the `\graphicspath` line).
Compiler: pdfLaTeX. Class: `sn-jrnl` with `[iicol]` (two-column).

**Before submission fill in `main.tex`:** author list, affiliation block,
funding declaration (all marked `FILL`).

**Every number in the tables/prose is computed from result files in
`lstm_model/` and can be re-verified in one command:**

```
python3 verify_paper_numbers.py   # recomputes all numbers, PASS/FAIL per claim
python3 make_figures.py           # regenerates figs 1-7
python3 make_fig_replication.py   # regenerates fig 8
python3 cascade_analysis.py       # regenerates cascade_report.json
```

**Scorer note (v2):** all numbers — tables, oracles, complementarity,
AUROC/ECE, selective prediction, buckets — now use the single scorer in
`metrics.py` (NFC + zero-width strip + whitespace collapse). In v1 the
oracle/complementarity/calibration numbers had been computed with a raw
string comparison; the unified numbers (v2) are equal or slightly better.

| Claim in paper | Source |
|---|---|
| Table 1 rows (WRR/CharAcc/CER/1-NED) | `results_{std,grph,selftrain,synthaug}_ours.json`, `eval_metrics_v3.json` (CRNN/Tesseract) |
| Fusion 64.6 / 82.8, oracle 71.9 | `verify_paper_numbers.py` over `conf_*_ours.json` |
| Table 2 cross-dataset rows + oracles | `verify_paper_numbers.py` over `conf_{std,grph}_{ours,bstd,assamese,hindi}.json` |
| Table 3 twelve-language replication (12/12, mean +5.0) | `fusion_replication_law.json` (from `conf_{std,grph}_law_*.json`) |
| Table 4 cascade (66–85% of gain at t=0.95) | `cascade_report.json` (`cascade_analysis.py`) |
| AUROC 0.89–0.91 / ECE 0.32–0.41 / risk-coverage 84.9@70, 91.4@50 | `calibration_report.json` (`calibration_analysis.py`) |
| Complementarity 11.9% / 7.0% | `verify_paper_numbers.py` |
| Rarity buckets (51.6 vs 40.5 in <5; n=126/98/103/43) | `make_figures.py` fig6 logic under unified scorer |
| ROVER 64.05 vs 64.59 | `rover_result.json`, rescored with unified scorer |
| McNemar p-values + bootstrap CIs | `verify_paper_numbers.py` (cross-checks `significance_report.json`) |
| Injected token counts 640/528/981 | `conf_grph.log`, `conf_assamese_grph.log`, `conf_hindi_grph.log` ("Added N new grapheme tokens") — v1's 593/1003 were cluster-inventory sizes, corrected in v2 |
| Table PARSeq (66.8/84.6 ours; 77.3/92.9 BSTD; p=0.50 / <1e-4) | `parseq_report.json` + `verify_paper_numbers.py` over `conf_parseq_{ours,bstd}.json` |
| PARSeq trained on 8.48M synthetic + BSTD; reports 0.82 in-domain | BSTD paper arXiv:2511.23071 (Tables 4 & 7) |
