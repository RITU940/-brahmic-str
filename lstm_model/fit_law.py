"""
fit_law.py  --  Part ① statistical fit (PREREGISTRATION.md §5).
================================================================
Fits and validates the predictive tokenization-granularity law from the per-language
results, using ONLY the pre-registered analysis:
  H1 (polarity): for each descriptor, leave-one-SCRIPT-out (LOSO) threshold classifier
                 accuracy + point-biserial / Spearman with grapheme_advantage.
                 Primary descriptor = neutral_fertility; the rest are the horse-race.
  H2 (fusion):   linear regression of fusion_gain (and grapheme_advantage) on fertility,
                 R², slope, bootstrap-95% CI (resampling SCRIPTS).

NO external deps beyond numpy (scipy/sklearn intentionally avoided — must run on server3).
Inputs:  script_descriptors.json  +  law_logs/fusion_law_<lang>.log  (or --logdir).
Output:  law_fit_results.json  + printed report.

Usage:  PY=.../python ; $PY fit_law.py --logdir law_logs
"""
import os, re, sys, json, glob, argparse
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(42)

DESCRIPTORS = [   # (json key, label) — primary first
    ('bpe_fertility_neutral', 'neutral_fertility*'),
    ('bytes_per_cluster', 'bytes_per_cluster'),
    ('grapheme_entropy', 'grapheme_entropy'),
    ('conjunct_density', 'conjunct_density'),
    ('bpe_cluster_fragmentation', 'bpe_frag'),
    ('strr', 'strr'),
    ('clusters_per_word', 'clusters_per_word'),
]


def parse_fusion_log(path):
    txt = open(path).read()
    def grab(label):
        m = re.search(label + r'\s+WRR:\s*([\d.]+)', txt)
        return float(m.group(1)) if m else None
    return {'bpe': grab('Standard'), 'grph': grab('Grapheme'),
            'fusion': grab(r'FUSION \(max-conf\)'), 'oracle': grab('ORACLE ceiling')}


def pearson(x, y):
    if np.std(x) == 0 or np.std(y) == 0:   # constant descriptor (e.g. STRR=0) -> undefined
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return float('nan')
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def loso_threshold_acc(x, y):
    """Leave-one-script-out 1D threshold classifier accuracy (direction learned on train)."""
    n = len(x); correct = 0; preds = []
    for i in range(n):
        xi = np.delete(x, i); yi = np.delete(y, i)
        xs = np.sort(np.unique(xi))
        cands = [xs[0] - 1] + [(xs[j] + xs[j + 1]) / 2 for j in range(len(xs) - 1)] + [xs[-1] + 1]
        best = (-1, None, 1)
        for t in cands:
            for d in (1, -1):
                pred = (xi > t).astype(int) if d == 1 else (xi < t).astype(int)
                acc = float((pred == yi).mean())
                if acc > best[0]:
                    best = (acc, t, d)
        _, t, d = best
        p = int(x[i] > t) if d == 1 else int(x[i] < t)
        preds.append(p); correct += int(p == y[i])
    return correct / n, preds


def lin_fit(x, y):
    s, b = np.polyfit(x, y, 1)
    yh = s * x + b
    ss_res = float(np.sum((y - yh) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return float(s), float(b), r2


def bootstrap_slope_ci(x, y, B=10000):
    n = len(x); slopes = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(np.unique(x[idx])) < 2:
            continue
        slopes.append(np.polyfit(x[idx], y[idx], 1)[0])
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logdir', default='law_logs')
    ap.add_argument('--descriptors', default=os.path.join(BASE, 'script_descriptors.json'))
    a = ap.parse_args()

    desc = json.load(open(a.descriptors, encoding='utf-8'))
    rows = []
    for path in sorted(glob.glob(os.path.join(BASE, a.logdir, 'fusion_law_*.log'))):
        lang = re.search(r'fusion_law_(.+)\.log', os.path.basename(path)).group(1)
        if lang not in desc:
            continue
        r = parse_fusion_log(path)
        if r['bpe'] is None or r['grph'] is None:
            continue
        rows.append({
            'lang': lang, 'script': desc[lang].get('script', '?'),
            'fertility': desc[lang].get('bpe_fertility_neutral'),
            'bpe': r['bpe'], 'grph': r['grph'], 'fusion': r['fusion'],
            'grapheme_advantage': round(r['grph'] - r['bpe'], 2),
            'polarity': int(r['grph'] > r['bpe']),
            'fusion_gain': round((r['fusion'] - max(r['grph'], r['bpe'])), 2) if r['fusion'] else None,
            'desc': desc[lang],
        })

    n = len(rows)
    print(f"\n=== LAW FIT  (N = {n} languages){'  [PARTIAL]' if n < 12 else ''} ===")
    print(f"{'lang':<10}{'script':<11}{'fert':>6}{'BPE':>7}{'grph':>7}{'adv':>7}"
          f"{'win':>6}{'fus_gain':>9}")
    for r in sorted(rows, key=lambda z: -(z['fertility'] or 0)):
        print(f"{r['lang']:<10}{r['script']:<11}{r['fertility']:>6.2f}{r['bpe']:>7.2f}"
              f"{r['grph']:>7.2f}{r['grapheme_advantage']:>+7.2f}"
              f"{'grph' if r['polarity'] else 'bpe':>6}"
              f"{(r['fusion_gain'] if r['fusion_gain'] is not None else float('nan')):>9.2f}")

    y_pol = np.array([r['polarity'] for r in rows])
    adv = np.array([r['grapheme_advantage'] for r in rows], float)

    # ── H1: descriptor horse-race ────────────────────────────────────────────
    print("\n--- H1: polarity prediction (leave-one-script-out) + corr with advantage ---")
    print(f"{'descriptor':<20}{'LOSO_acc':>9}{'pred_ok':>9}{'pearson_adv':>13}{'spearman_adv':>13}")
    horse = {}
    for key, label in DESCRIPTORS:
        vals = [r['desc'].get(key) for r in rows]
        if any(v is None for v in vals) or len(set(y_pol)) < 2:
            continue
        x = np.array(vals, float)
        acc, preds = loso_threshold_acc(x, y_pol)
        pear = pearson(x, adv)
        spear = spearman(x, adv)
        horse[label] = {'loso_acc': acc, 'pearson_adv': pear, 'spearman_adv': spear}
        print(f"{label:<20}{acc*100:>8.1f}%{sum(int(p==yy) for p,yy in zip(preds,y_pol)):>6}/{n:<2}"
              f"{pear:>13.3f}{spear:>13.3f}")

    # ── H2: fusion-gain & advantage regression on fertility ──────────────────
    fert = np.array([r['fertility'] for r in rows], float)
    out = {'n': n, 'rows': [{k: r[k] for k in ('lang','script','fertility','bpe','grph',
            'fusion','grapheme_advantage','polarity','fusion_gain')} for r in rows],
           'H1_horserace': horse}
    print("\n--- H2: regression on neutral_fertility ---")
    for name, yv in [('grapheme_advantage', adv),
                     ('fusion_gain', np.array([r['fusion_gain'] for r in rows], float))]:
        if np.any(np.isnan(yv)):
            print(f"{name}: (missing values, skipped)"); continue
        s, b, r2 = lin_fit(fert, yv)
        lo, hi = bootstrap_slope_ci(fert, yv)
        sig = "CI excludes 0 OK" if (lo > 0 or hi < 0) else "CI includes 0 (ns)"
        out[f'H2_{name}'] = {'slope': s, 'intercept': b, 'r2': r2, 'slope_CI95': [lo, hi]}
        print(f"{name:<20} slope={s:+.3f}  R2={r2:.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  {sig}")

    json.dump(out, open(os.path.join(BASE, 'law_fit_results.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f"\nsaved law_fit_results.json")
    print("Pre-registered success bar: primary descriptor LOSO polarity >=8/9 AND "
          "fusion_gain R2>~0.6 with slope CI excluding 0.")


if __name__ == '__main__':
    main()
