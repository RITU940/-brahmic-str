"""
Paper figures for the grapheme-aware dual-tokenization paper.
Run with SYSTEM python3 (matplotlib + PIL/raqm for Bengali shaping):
    python3 make_figures.py
All Florence-2 / fusion / oracle numbers are computed live from
conf_*.json + results_*.json — no hardcoded model numbers.
Outputs PDF + PNG (300 dpi) into figures/.
"""
import os, sys, json, glob, unicodedata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from metrics import evaluate_corpus
from build_grapheme_vocab_lang import segment_graphemes_indic

FIG = os.path.join(BASE, 'figures')
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 110})
C = {'std': '#4878CF', 'grph': '#EE854A', 'self': '#6ACC65',
     'syn': '#956CB4', 'fus': '#D65F5F', 'orc': '#999999'}

norm = lambda s: (s or '').strip()


def load_conf(tag):
    return json.load(open(os.path.join(BASE, f'conf_{tag}.json')))


def fuse(tag_list):
    """max-conf fusion preds + gts + per-sample correctness of each model."""
    D = [load_conf(t) for t in tag_list]
    n = len(D[0])
    gts = [r['gt'] for r in D[0]]
    fused, fconf = [], []
    for i in range(n):
        c, p = max(((d[i]['conf'], d[i]['pred']) for d in D), key=lambda x: x[0])
        fused.append(p); fconf.append(c)
    oracle = 100 * sum(1 for i in range(n)
                       if any(norm(gts[i]) == norm(d[i]['pred']) for d in D)) / n
    return D, gts, fused, fconf, oracle


def wrr(gts, preds):
    return evaluate_corpus(gts, preds, verbose=False)


def savefig(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'{name}.{ext}'), bbox_inches='tight',
                    dpi=300 if ext == 'png' else None)
    plt.close(fig)
    print(f'  saved figures/{name}.pdf/.png')


# ---------------------------------------------------------------- data: ours
TAGS4 = ['std_ours', 'grph_ours', 'selftrain_ours', 'synthaug_ours']
D4, gts_o, fused_o, fconf_o, oracle_o = fuse(TAGS4)
m_fus_o = wrr(gts_o, fused_o)
res = {t: json.load(open(os.path.join(BASE, f'results_{t}.json'))) for t in
       ['std_ours', 'grph_ours', 'selftrain_ours', 'synthaug_ours',
        'std_bstd', 'grph_bstd', 'zeroshot_ours']}
crnn = json.load(open(os.path.join(BASE, 'eval_metrics_v3.json')))

# ------------------------------------------------- fig 1: main results bars
def fig1():
    rows = [
        ('Tesseract (fine-tuned)', crnn['tess_ft']['word_acc'], crnn['tess_ft']['char_acc'], '#BBBBBB'),
        ('CRNN (CTC)', crnn['crnn_v3']['word_acc'], crnn['crnn_v3']['char_acc'], '#BBBBBB'),
        ('Florence-2 zero-shot', res['zeroshot_ours']['WRR'], res['zeroshot_ours']['char_accuracy'], '#888888'),
        ('Florence-2 fine-tuned (BPE)', res['std_ours']['WRR'], res['std_ours']['char_accuracy'], C['std']),
        ('+ grapheme tokenization', res['grph_ours']['WRR'], res['grph_ours']['char_accuracy'], C['grph']),
        ('+ self-training', res['selftrain_ours']['WRR'], res['selftrain_ours']['char_accuracy'], C['self']),
        ('+ synthetic curriculum', res['synthaug_ours']['WRR'], res['synthaug_ours']['char_accuracy'], C['syn']),
        ('Dual-tokenization fusion', m_fus_o['WRR'], m_fus_o['char_accuracy'], C['fus']),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = range(len(rows))
    ax.bar(x, [r[1] for r in rows], 0.62, color=[r[3] for r in rows], zorder=3)
    ax.scatter(x, [r[2] for r in rows], marker='D', s=28, color='#222222',
               zorder=4, label='Character accuracy')
    for i, r in enumerate(rows):
        ax.text(i, r[1] + 1.2, f'{r[1]:.1f}', ha='center', fontsize=9)
    ax.axhline(oracle_o, ls='--', lw=1, color=C['orc'], zorder=2)
    ax.text(len(rows) - 0.45, oracle_o + 1, f'oracle {oracle_o:.1f}',
            ha='right', fontsize=9, color='#555555')
    ax.set_xticks(list(x))
    ax.set_xticklabels([r[0] for r in rows], rotation=28, ha='right', fontsize=9)
    ax.set_ylabel('Word recognition rate (%)')
    ax.set_ylim(0, 92)
    ax.legend(frameon=False, loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.25, zorder=0)
    savefig(fig, 'fig1_main_results')


# --------------------------------------- fig 2: cross-dataset / cross-script
def fig2(extra_langs=()):
    sets = [('Our Bengali\n(370)', ['std_ours', 'grph_ours']),
            ('BSTD Bengali\n(1368)', ['std_bstd', 'grph_bstd']),
            ('BSTD Assamese\n(1505)', ['std_assamese', 'grph_assamese'])]
    for lang, label in extra_langs:
        sets.append((label, [f'std_{lang}', f'grph_{lang}']))
    groups = []
    for label, tags in sets:
        D, gts, fused, _, orc = fuse(tags)
        groups.append((label,
                       wrr(gts, [r['pred'] for r in D[0]])['WRR'],
                       wrr(gts, [r['pred'] for r in D[1]])['WRR'],
                       wrr(gts, fused)['WRR'], orc))
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    import numpy as np
    xs = np.arange(len(groups))
    w = 0.2
    ax.bar(xs - 1.5 * w, [g[1] for g in groups], w, color=C['std'], label='BPE', zorder=3)
    ax.bar(xs - 0.5 * w, [g[2] for g in groups], w, color=C['grph'], label='Grapheme', zorder=3)
    ax.bar(xs + 0.5 * w, [g[3] for g in groups], w, color=C['fus'], label='Fusion', zorder=3)
    ax.bar(xs + 1.5 * w, [g[4] for g in groups], w, color='white',
           edgecolor=C['orc'], hatch='//', label='Oracle', zorder=3)
    for i, g in enumerate(groups):
        for k, v in enumerate(g[1:]):
            ax.text(xs[i] + (k - 1.5) * w, v + 0.8, f'{v:.1f}', ha='center', fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    ax.set_ylabel('Word recognition rate (%)')
    ax.set_ylim(0, max(g[4] for g in groups) * 1.18)
    ax.legend(frameon=False, ncol=4, fontsize=9, loc='upper center',
              bbox_to_anchor=(0.5, 1.14))
    ax.grid(axis='y', alpha=0.25, zorder=0)
    savefig(fig, 'fig2_cross_dataset')


# -------------------------------------- fig 3: complementarity breakdown
def fig3(extra_langs=()):
    sets = [('Our Bengali', ['std_ours', 'grph_ours']),
            ('BSTD Bengali', ['std_bstd', 'grph_bstd']),
            ('BSTD Assamese', ['std_assamese', 'grph_assamese'])]
    for lang, label in extra_langs:
        sets.append((label.replace('\n', ' '), [f'std_{lang}', f'grph_{lang}']))
    cats = ['Both correct', 'Only grapheme', 'Only BPE', 'Neither']
    cols = ['#6ACC65', C['grph'], C['std'], '#CCCCCC']
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for i, (label, (t1, t2)) in enumerate(sets):
        a, b = load_conf(t1), load_conf(t2)
        n = len(a)
        ok1 = [norm(r['gt']) == norm(r['pred']) for r in a]
        ok2 = [norm(r['gt']) == norm(r['pred']) for r in b]
        v = [sum(x and y for x, y in zip(ok1, ok2)),
             sum((not x) and y for x, y in zip(ok1, ok2)),
             sum(x and (not y) for x, y in zip(ok1, ok2)),
             sum((not x) and (not y) for x, y in zip(ok1, ok2))]
        v = [100 * c / n for c in v]
        left = 0
        for val, col, cat in zip(v, cols, cats):
            ax.barh(i, val, left=left, color=col, height=0.55,
                    label=cat if i == 0 else None)
            if val > 6:
                ax.text(left + val / 2, i, f'{val:.0f}%', ha='center',
                        va='center', fontsize=8.5)
            left += val
    ax.set_yticks(range(len(sets)))
    ax.set_yticklabels([s[0] for s in sets], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel('Share of test images (%)')
    ax.legend(frameon=False, ncol=4, fontsize=8, loc='upper center',
              bbox_to_anchor=(0.5, 1.18))
    savefig(fig, 'fig3_complementarity')


# ----------------------------------------------- fig 4: risk-coverage curve
def fig4():
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    curves = [('Fusion (4 models)', fconf_o,
               [norm(g) == norm(p) for g, p in zip(gts_o, fused_o)], C['fus'], 2.2),
              ('Synthetic-curriculum model', [r['conf'] for r in D4[3]],
               [norm(r['gt']) == norm(r['pred']) for r in D4[3]], C['syn'], 1.4),
              ('BPE model', [r['conf'] for r in D4[0]],
               [norm(r['gt']) == norm(r['pred']) for r in D4[0]], C['std'], 1.4)]
    for label, confs, corr, col, lw in curves:
        order = sorted(range(len(confs)), key=lambda i: -confs[i])
        accs, covs, c = [], [], 0
        for k, i in enumerate(order, 1):
            c += corr[i]
            covs.append(100 * k / len(order))
            accs.append(100 * c / k)
        ax.plot(covs, accs, color=col, lw=lw, label=label)
    # mark the selective-prediction operating points of the fusion curve
    order = sorted(range(len(fconf_o)), key=lambda i: -fconf_o[i])
    corr_f = [norm(g) == norm(p) for g, p in zip(gts_o, fused_o)]
    for cov in (50, 70):
        k = int(round(cov / 100 * len(order)))
        acc = 100 * sum(corr_f[i] for i in order[:k]) / k
        ax.scatter([cov], [acc], color=C['fus'], zorder=5, s=30)
        ax.annotate(f'{acc:.1f}% @ {cov}% cov', (cov, acc),
                    textcoords='offset points', xytext=(8, 6), fontsize=9)
    ax.set_xlabel('Coverage (%) — share of images answered')
    ax.set_ylabel('Accuracy on answered images (%)')
    ax.set_xlim(10, 100); ax.set_ylim(40, 100)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    savefig(fig, 'fig4_risk_coverage')


# -------------------------------------------- fig 5: reliability + AUROC bar
def fig5():
    rep = json.load(open(os.path.join(BASE, 'calibration_report.json')))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    ax = axes[0]
    bins = rep['fusion4_ours']['reliability_bins']
    xs = [i / 10 + 0.05 for i in range(10)]
    accs = [b.get('accuracy') for b in bins]
    ns = [b.get('n', 0) for b in bins]
    ax.plot([0, 1], [0, 1], ls='--', color='#999999', lw=1)
    ax.bar(xs, [a if a is not None else 0 for a in accs], 0.09,
           color=C['fus'], alpha=0.85, edgecolor='white', zorder=3)
    for x, n, a in zip(xs, ns, accs):
        if n:
            ax.text(x, (a or 0) + 0.03, str(n), ha='center', fontsize=7, color='#444444')
    ax.set_xlabel('Predicted confidence'); ax.set_ylabel('Empirical accuracy')
    ax.set_title(f"Reliability (fusion)  ECE={rep['fusion4_ours']['ECE']:.2f}", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax = axes[1]
    names = ['std_ours', 'grph_ours', 'selftrain_ours', 'synthaug_ours', 'fusion4_ours']
    labels = ['BPE', 'Grapheme', 'Self-train', 'Synth-curr.', 'Fusion']
    aurocs = [rep[n]['AUROC'] for n in names]
    cols = [C['std'], C['grph'], C['self'], C['syn'], C['fus']]
    ax.bar(range(len(names)), aurocs, 0.6, color=cols, zorder=3)
    for i, a in enumerate(aurocs):
        ax.text(i, a + 0.008, f'{a:.2f}', ha='center', fontsize=8.5)
    ax.axhline(0.5, ls='--', lw=1, color='#999999')
    ax.text(len(names) - 0.4, 0.51, 'chance', fontsize=8, color='#777777', ha='right')
    ax.set_xticks(range(len(names))); ax.set_xticklabels(labels, fontsize=8.5, rotation=20)
    ax.set_ylabel('AUROC (confidence → correctness)')
    ax.set_ylim(0.45, 1.0)
    ax.set_title('Confidence ranks correctness', fontsize=10)
    ax.grid(axis='y', alpha=0.25, zorder=0)
    fig.tight_layout()
    savefig(fig, 'fig5_calibration')


# -------------------------------- fig 6: WRR by grapheme-rarity bucket
def fig6():
    sp = json.load(open(os.path.join(BASE, 'florence2_splits.json')))
    from collections import Counter
    freq = Counter()
    for part in ('train', 'val'):
        for r in sp[part]:
            freq.update(segment_graphemes_indic(r['gt'].strip()))
    buckets = [(0, 5, '<5'), (5, 20, '5-19'), (20, 100, '20-99'), (100, 10**9, '≥100')]

    def bucket_of(word):
        f = min((freq.get(g, 0) for g in segment_graphemes_indic(word)), default=0)
        for lo, hi, lab in buckets:
            if lo <= f < hi:
                return lab

    models = [('BPE', [r['pred'] for r in D4[0]], C['std']),
              ('Grapheme', [r['pred'] for r in D4[1]], C['grph']),
              ('Synth-curriculum', [r['pred'] for r in D4[3]], C['syn']),
              ('Fusion', fused_o, C['fus'])]
    labs = [b[2] for b in buckets]
    counts = {lab: 0 for lab in labs}
    for g in gts_o:
        counts[bucket_of(norm(g))] += 1
    import numpy as np
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    xs = np.arange(len(labs))
    w = 0.19
    for k, (name, preds, col) in enumerate(models):
        vals = []
        for lab in labs:
            idx = [i for i, g in enumerate(gts_o) if bucket_of(norm(g)) == lab]
            vals.append(100 * sum(norm(gts_o[i]) == norm(preds[i]) for i in idx) / max(1, len(idx)))
        ax.bar(xs + (k - 1.5) * w, vals, w, color=col, label=name, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{lab}\n(n={counts[lab]})' for lab in labs], fontsize=9)
    ax.set_xlabel('Rarest grapheme in word — training-set frequency')
    ax.set_ylabel('Word recognition rate (%)')
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(axis='y', alpha=0.25, zorder=0)
    savefig(fig, 'fig6_rarity_buckets')


# ------------------------------------------ fig 7: qualitative examples grid
def fig7():
    from PIL import Image, ImageDraw, ImageFont
    fonts = (glob.glob('/usr/share/fonts/truetype/noto/NotoSerifBengali-Regular.ttf') or
             glob.glob('/usr/share/fonts/truetype/noto/NotoSerifBengali-Medium.ttf') or
             glob.glob('/usr/share/fonts/truetype/noto/NotoSerifBengali*.ttf'))
    font_b = ImageFont.truetype(fonts[0], 26)
    font_s = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    sp = json.load(open(os.path.join(BASE, 'florence2_splits.json')))
    test = sp['test']
    a, b = D4[0], D4[1]   # BPE, grapheme
    assert all(test[i]['gt'].strip() == a[i]['gt'].strip() for i in range(len(a))), \
        'conf rows not aligned with test split order'
    ok = lambda r: norm(r['gt']) == norm(r['pred'])
    okf = [norm(g) == norm(p) for g, p in zip(gts_o, fused_o)]
    grapheme_fix = [i for i in range(len(a)) if not ok(a[i]) and ok(b[i]) and okf[i]]
    bpe_fix = [i for i in range(len(a)) if ok(a[i]) and not ok(b[i]) and okf[i]]
    fails = [i for i in range(len(a)) if not okf[i]]
    grapheme_fix.sort(key=lambda i: -len(norm(gts_o[i])))
    bpe_fix.sort(key=lambda i: -len(norm(gts_o[i])))
    picks = grapheme_fix[:3] + bpe_fix[:2] + fails[:1]

    CW, CH, PAD = 460, 240, 12
    cols_n, rows_n = 3, 2
    canvas = Image.new('RGB', (cols_n * CW, rows_n * CH), 'white')
    draw = ImageDraw.Draw(canvas)
    for k, i in enumerate(picks):
        cx, cy = (k % cols_n) * CW, (k // cols_n) * CH
        path = test[i]['image']
        try:
            im = Image.open(path).convert('RGB')
            im.thumbnail((CW - 2 * PAD, 96))
            canvas.paste(im, (cx + PAD, cy + PAD))
        except Exception as e:
            draw.text((cx + PAD, cy + PAD), f'[image: {e}]', fill='red', font=font_s)
        y = cy + PAD + 100
        rows = [('GT', a[i]['gt'], 'black'),
                ('BPE', a[i]['pred'], '#1a56b0' if ok(a[i]) else '#c0392b'),
                ('Grapheme', b[i]['pred'], '#1a56b0' if ok(b[i]) else '#c0392b'),
                ('Fusion', fused_o[i], '#1a7a2e' if okf[i] else '#c0392b')]
        for tag, txt, col in rows:
            draw.text((cx + PAD, y), f'{tag}: ', fill='#555555', font=font_s)
            draw.text((cx + PAD + 110, y - 6), txt or '∅', fill=col, font=font_b)
            y += 32
    canvas.save(os.path.join(FIG, 'fig7_qualitative.png'), dpi=(300, 300))
    print('  saved figures/fig7_qualitative.png')


if __name__ == '__main__':
    print(f'Fusion (ours, 4-way): WRR {m_fus_o["WRR"]:.2f}  CharAcc '
          f'{m_fus_o["char_accuracy"]:.2f}  oracle {oracle_o:.2f}')
    hindi = ([('hindi', 'BSTD Hindi\n(4846)')]
             if os.path.exists(os.path.join(BASE, 'conf_std_hindi.json')) else [])
    fig1(); fig2(extra_langs=hindi); fig3(extra_langs=hindi)
    fig4(); fig5(); fig6(); fig7()
    print('ALL FIGURES DONE')
