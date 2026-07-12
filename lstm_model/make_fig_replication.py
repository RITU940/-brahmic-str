#!/usr/bin/env python3
"""fig8: fixed-budget fusion replication across the 12 law languages.
Reads fusion_replication_law.json (verify_paper_numbers.py) and renders a
dot plot: BPE branch, grapheme branch, max-confidence fusion per language,
sorted by fusion WRR. Same style/palette as make_figures.py.
    python3 make_fig_replication.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, 'figures')

plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 110})
C = {'std': '#4878CF', 'grph': '#EE854A', 'fus': '#D65F5F', 'orc': '#999999'}

rows = json.load(open(os.path.join(BASE, 'fusion_replication_law.json')))
langs = sorted(rows, key=lambda l: rows[l]['fus'])

fig, ax = plt.subplots(figsize=(6.6, 4.4))
for y, lang in enumerate(langs):
    r = rows[lang]
    lo = min(r['bpe'], r['grph'], r['fus'])
    ax.plot([lo, r['fus']], [y, y], color='#CCCCCC', lw=1.4, zorder=1)
    ax.scatter(r['bpe'], y, marker='o', s=42, color=C['std'], zorder=3)
    ax.scatter(r['grph'], y, marker='s', s=40, color=C['grph'], zorder=3)
    ax.scatter(r['fus'], y, marker='D', s=44, color=C['fus'], zorder=4)
    gain = r['fus'] - max(r['bpe'], r['grph'])
    ax.annotate(f'+{gain:.1f}', (r['fus'], y), xytext=(7, -3.5),
                textcoords='offset points', fontsize=9, color='#555555')

labels = {'english': 'English (control)'}
ax.set_yticks(range(len(langs)))
ax.set_yticklabels([labels.get(l, l.capitalize()) for l in langs])
ax.set_xlabel('WRR (%)')
ax.set_xlim(10, 66)
ax.grid(axis='x', color='#EEEEEE', lw=0.8, zorder=0)
ax.scatter([], [], marker='o', s=42, color=C['std'], label='BPE branch')
ax.scatter([], [], marker='s', s=40, color=C['grph'], label='grapheme branch')
ax.scatter([], [], marker='D', s=44, color=C['fus'], label='fusion')
ax.legend(loc='lower right', frameon=False, fontsize=10)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(FIG, f'fig8_replication.{ext}'),
                bbox_inches='tight', dpi=300 if ext == 'png' else None)
print('saved figures/fig8_replication.pdf/.png')
