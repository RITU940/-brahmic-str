#!/usr/bin/env python3
"""
H3 analysis (the paper's headline test)
=======================================
Does a script descriptor (PRIMARY: neutral fertility) PREDICT how well an unseen
script transfers in the zero-real-image setting (Rung-B WRR)?

Per PREREGISTRATION.md sec.5 H3: Spearman rank-correlation between the descriptor and
zero-shot transfer WRR across held-out scripts.

Run with SYSTEM python3 (has numpy/scipy/matplotlib), like make_figures.py:
    python3 h3_analysis.py

Auto-discovers result_zs_loso_rungB_*.json, so it is safe to run at ANY stage of the
LOSO sweep; it just uses however many scripts have finished. Needs >=3 points for a
correlation, >=5-6 for a credible line.
"""
import json, glob, os, sys
import numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))

# LOSO held-out script tag -> key in script_descriptors.json
# (oriya->odia, gurmukhi->punjabi, devanagari->hindi: same script, descriptor named by language)
TAG2DESC = {
    'tamil': 'tamil', 'telugu': 'telugu', 'kannada': 'kannada', 'malayalam': 'malayalam',
    'oriya': 'odia', 'gujarati': 'gujarati', 'bengali': 'bengali',
    'devanagari': 'hindi', 'gurmukhi': 'punjabi',
}
PRIMARY = 'bpe_fertility_neutral'                       # pre-registered primary descriptor
DESCRIPTORS = ['bpe_fertility_neutral', 'conjunct_density', 'grapheme_entropy',
               'bpe_cluster_fragmentation', 'strr']      # the horse-race

def load_json(p):
    with open(p) as f:
        return json.load(f)

desc = load_json(os.path.join(BASE, 'script_descriptors.json'))

rows = []
for f in sorted(glob.glob(os.path.join(BASE, 'result_zs_loso_rungB_*.json'))):
    r = load_json(f)
    tag = r['script']
    dkey = TAG2DESC.get(tag)
    if dkey is None or dkey not in desc:
        print(f"[warn] no descriptor mapping for '{tag}', skipping", file=sys.stderr)
        continue
    d = desc[dkey]
    cov = None
    metap = os.path.join(BASE, f'zeroshot_loso_meta_{tag}.json')
    if os.path.exists(metap):
        cov = load_json(metap).get('target_token_coverage_by_source_vocab_pct')
    row = {'tag': tag, 'desc_key': dkey, 'WRR': r['WRR'],
           'CharAcc': r.get('CharAcc'), 'coverage': cov}
    for k in DESCRIPTORS:
        row[k] = d.get(k)
    rows.append(row)

n = len(rows)
print(f"\n=== H3 zero-shot transfer analysis — {n} held-out script(s) with Rung-B results ===")
if n == 0:
    print("No Rung-B results yet. Re-run after the first scripts finish.")
    sys.exit(0)

hdr = f"{'script':<11}{'descKey':<9}{'fertility':>10}{'cover%':>8}{'WRR':>7}{'CharAcc':>9}"
print(hdr); print('-' * len(hdr))
for r in sorted(rows, key=lambda x: -(x['WRR'] or 0)):
    cov = r['coverage'] if r['coverage'] is not None else float('nan')
    ca = r['CharAcc'] if r['CharAcc'] is not None else float('nan')
    print(f"{r['tag']:<11}{r['desc_key']:<9}{r[PRIMARY]:>10.3f}{cov:>8.1f}{r['WRR']:>7.2f}{ca:>9.1f}")

def corr(xname, xs, ys, yname):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        print(f"  {xname:>26} vs {yname:<8}: n={len(pairs)} (need >=3)")
        return
    xx, yy = zip(*pairs)
    rho, p = spearmanr(xx, yy)
    star = '  <-- primary' if xname == PRIMARY and yname == 'WRR' else ''
    print(f"  {xname:>26} vs {yname:<8}: Spearman rho={rho:+.3f}  p={p:.3f}  (n={len(pairs)}){star}")

if n >= 3:
    print("\n--- Horse-race: Spearman(descriptor, Rung-B WRR) ---")
    for k in DESCRIPTORS:
        corr(k, [r[k] for r in rows], [r['WRR'] for r in rows], 'WRR')
    corr('coverage', [r['coverage'] for r in rows], [r['WRR'] for r in rows], 'WRR')
    print("\n--- Same predictors vs Rung-B CharAcc (denser DV, less floored than WRR) ---")
    for k in (PRIMARY, 'coverage'):
        corr(k, [r.get(k) for r in rows], [r['CharAcc'] for r in rows], 'CharAcc')
else:
    print(f"\n[only {n} script(s) so far — need >=3 for any correlation, >=5-6 for a credible H3 line]")

# ---- scatter: fertility vs Rung-B WRR, labelled by script ----
fig, ax = plt.subplots(figsize=(6, 4.5))
xs = [r[PRIMARY] for r in rows]; ys = [r['WRR'] for r in rows]
ax.scatter(xs, ys, s=60, color='#2b6cb0', zorder=3)
for r in rows:
    ax.annotate(r['tag'], (r[PRIMARY], r['WRR']), xytext=(4, 4),
                textcoords='offset points', fontsize=8)
ax.set_xlabel('neutral fertility (BPE tokens / grapheme cluster)')
ax.set_ylabel('zero-shot Rung-B WRR (%)')
title = 'H3: fertility vs zero-real-image transfer'
if n >= 3:
    rho, p = spearmanr(xs, ys)
    title += f'   (Spearman rho={rho:+.2f}, p={p:.2f}, n={n})'
ax.set_title(title, fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
os.makedirs(os.path.join(BASE, 'figures'), exist_ok=True)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(BASE, 'figures', f'h3_zeroshot_transfer.{ext}'), dpi=150)
print("\nSaved figures/h3_zeroshot_transfer.pdf/.png")
