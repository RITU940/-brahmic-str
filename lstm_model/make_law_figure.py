"""
make_law_figure.py  --  the Part-① "money figure" + descriptor horse-race.
==========================================================================
Reads law_fit_results.json (produced by fit_law.py) and renders:
  Panel A: neutral fertility  vs  grapheme advantage (grapheme WRR - BPE WRR),
           points labelled by script, coloured by which branch wins, the polarity
           line at y=0, the fitted law line + R², and the predicted-flip band.
  Panel B: leave-one-script-out polarity accuracy per descriptor (the horse-race),
           the pre-registered primary (neutral_fertility) highlighted.

Run with SYSTEM python3 (has matplotlib):  python3 make_law_figure.py
Outputs figures/law_main.pdf/.png (300 dpi).
"""
import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, 'figures'); os.makedirs(FIG, exist_ok=True)

d = json.load(open(os.path.join(BASE, 'law_fit_results.json')))
rows = d['rows']
H2 = d.get('H2_grapheme_advantage', {})
horse = d.get('H1_horserace', {})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={'width_ratios': [1.45, 1]})

# ── Panel A: the law ──────────────────────────────────────────────────────────
win_c, lose_c = '#1a7a2e', '#c0392b'
xs = [r['fertility'] for r in rows]
ys = [r['grapheme_advantage'] for r in rows]
for r in rows:
    win = r['polarity'] == 1
    axA.scatter(r['fertility'], r['grapheme_advantage'], s=90, zorder=3,
                facecolor=(win_c if win else 'white'),
                edgecolor=(win_c if win else lose_c), linewidth=1.8)
    axA.annotate(r['lang'].title(), (r['fertility'], r['grapheme_advantage']),
                 textcoords='offset points', xytext=(7, 4), fontsize=8.5, color='#333333')

axA.axhline(0, color='#888888', lw=1, ls='--', zorder=1)
axA.text(min(xs), 0.4, 'grapheme wins  ↑', fontsize=8, color=win_c, va='bottom')
axA.text(min(xs), -0.4, 'subword wins  ↓', fontsize=8, color=lose_c, va='top')

# predicted-flip band: between highest-fertility subword-win and lowest-fertility grapheme-win
gw = [r['fertility'] for r in rows if r['polarity'] == 1]
bw = [r['fertility'] for r in rows if r['polarity'] == 0]
if gw and bw:
    lo, hi = max(bw), min(gw)
    if lo > hi:
        lo, hi = hi, lo
    axA.axvspan(lo, hi, color='#f0c419', alpha=0.18, zorder=0)
    axA.text((lo + hi) / 2, max(ys) * 0.95, 'predicted\nflip zone', fontsize=8,
             ha='center', va='top', color='#9a7d0a')

# fitted law line
if H2:
    s, b, r2 = H2['slope'], H2['intercept'], H2['r2']
    xr = [min(xs) - 0.2, max(xs) + 0.2]
    axA.plot(xr, [s * x + b for x in xr], color='#1a56b0', lw=2, zorder=2,
             label=f'law fit: adv = {s:.1f}·fertility {b:+.1f}\n$R^2$ = {r2:.2f}')
    axA.legend(loc='lower right', fontsize=8.5, framealpha=0.9)

axA.set_xlabel('neutral fertility  (GPT-2 BPE tokens / grapheme cluster)', fontsize=10)
axA.set_ylabel('grapheme advantage  (grapheme WRR − subword WRR)', fontsize=10)
axA.set_title(f'(a)  One number predicts the tokenization winner  (N={d["n"]})', fontsize=11)
axA.grid(True, alpha=0.25)

# ── Panel B: horse-race ───────────────────────────────────────────────────────
labels = list(horse.keys())
accs = [horse[l]['loso_acc'] * 100 for l in labels]
order = sorted(range(len(labels)), key=lambda i: accs[i])
labels = [labels[i] for i in order]; accs = [accs[i] for i in order]
colors = ['#1a56b0' if l.startswith('neutral_fertility') else '#BBBBBB' for l in labels]
axB.barh(range(len(labels)), accs, color=colors)
axB.set_yticks(range(len(labels))); axB.set_yticklabels(labels, fontsize=8.5)
axB.axvline(50, color='#c0392b', ls=':', lw=1)
axB.text(50, -0.7, 'chance', color='#c0392b', fontsize=7.5, ha='center')
for i, v in enumerate(accs):
    axB.text(v + 1, i, f'{v:.0f}%', va='center', fontsize=8)
axB.set_xlim(0, 109)
axB.set_xlabel('leave-one-script-out polarity accuracy (%)', fontsize=10)
axB.set_title('(b)  Descriptor horse-race', fontsize=11)

fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(FIG, f'law_main.{ext}'), bbox_inches='tight',
                dpi=300 if ext == 'png' else None)
plt.close(fig)
print(f"saved figures/law_main.pdf/.png  (N={d['n']}, fertility LOSO="
      f"{horse.get('neutral_fertility*',{}).get('loso_acc',0)*100:.0f}%, "
      f"adv R2={H2.get('r2',float('nan')):.2f})")
