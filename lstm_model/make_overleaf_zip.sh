#!/usr/bin/env bash
# Rebuild wacv_paper_overleaf.zip from the CURRENT paper_wacv sources.
#
#   bash make_overleaf_zip.sh
#
# Ships only what Overleaf needs to compile: the seven live sec/ files (the
# *_ed drafts and *_algo.bak track-flip alternates stay in git, not in the
# upload), the kit's .sty/.bst, and the figures. No build artefacts, no PDF.
# Re-run this after ANY edit -- an out-of-date zip silently uploads an older
# title/abstract, which is exactly how a stale draft reaches a reviewer.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
SRC=paper_wacv
ZIP=$PWD/wacv_paper_overleaf.zip

for f in main.tex preamble.tex numbers.tex main.bib wacv.sty ieeenat_fullname.bst; do
  [ -f "$SRC/$f" ] || { echo "MISSING $SRC/$f"; exit 1; }
done

rm -f "$ZIP"
( cd "$SRC" && zip -q -r "$ZIP" \
    main.tex preamble.tex numbers.tex main.bib wacv.sty ieeenat_fullname.bst \
    sec/0_abstract.tex sec/1_intro.tex sec/2_related.tex sec/3_method.tex \
    sec/4_experiments.tex sec/5_analysis.tex sec/6_conclusion.tex \
    figs/fig1_pivot.pdf figs/fig2_bars.pdf figs/fig3_scaling.pdf figs/fig4_receipts.pdf )

echo "wrote $ZIP"
unzip -l "$ZIP" | tail -3
echo
echo "Title going up:"
unzip -p "$ZIP" main.tex | grep -A1 '\\title'
