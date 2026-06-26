"""
Statistical significance for the paper (CPU-only, from conf_*.json files).
- McNemar's exact test (binomial) on paired word-level correctness:
    fusion vs best single model, and grapheme vs BPE, per dataset.
- 95% bootstrap confidence intervals (10k resamples, seed 42) for each WRR
  and for the fusion-minus-best-single difference.
Writes significance_report.json and prints a summary.
"""
import os, sys, json, random
from math import comb

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from metrics import normalize_bengali as norm_canon

norm = lambda s: norm_canon(s or '')


def load(tag):
    d = json.load(open(os.path.join(BASE, f'conf_{tag}.json')))
    return d


def correct(rows):
    return [norm(r['gt']) == norm(r['pred']) for r in rows]


def fuse(tag_list):
    D = [load(t) for t in tag_list]
    n = len(D[0])
    fused = []
    for i in range(n):
        _, p = max(((d[i]['conf'], d[i]['pred']) for d in D), key=lambda x: x[0])
        fused.append({'gt': D[0][i]['gt'], 'pred': p})
    return D, fused


def mcnemar_exact(corr_a, corr_b):
    """Exact two-sided McNemar (binomial on discordant pairs)."""
    b = sum(1 for x, y in zip(corr_a, corr_b) if x and not y)
    c = sum(1 for x, y in zip(corr_a, corr_b) if (not x) and y)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return b, c, min(1.0, p)


def boot_ci(corr, n_boot=10000, seed=42):
    rng = random.Random(seed)
    n = len(corr)
    stats = []
    for _ in range(n_boot):
        s = sum(corr[rng.randrange(n)] for _ in range(n))
        stats.append(100 * s / n)
    stats.sort()
    return stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot)]


def boot_diff_ci(corr_a, corr_b, n_boot=10000, seed=42):
    rng = random.Random(seed)
    n = len(corr_a)
    stats = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        stats.append(100 * (sum(corr_a[i] for i in idx) - sum(corr_b[i] for i in idx)) / n)
    stats.sort()
    return stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot)]


def analyze(name, tags, best_single_tag):
    D, fused = fuse(tags)
    by_tag = dict(zip(tags, D))
    out = {'n': len(fused)}
    cf = correct(fused)
    cb = correct(by_tag[best_single_tag])
    wrr = lambda c: round(100 * sum(c) / len(c), 2)
    out['fusion_WRR'] = wrr(cf)
    out['fusion_CI95'] = boot_ci(cf)
    out['best_single'] = best_single_tag
    out['best_single_WRR'] = wrr(cb)
    b, c, p = mcnemar_exact(cf, cb)
    out['fusion_vs_best'] = {'fusion_only_correct': c, 'single_only_correct': b,
                             'mcnemar_p': round(p, 5),
                             'diff_CI95': boot_diff_ci(cf, cb)}
    # grapheme vs BPE (first two tags by convention std, grph)
    c_std, c_g = correct(by_tag[tags[0]]), correct(by_tag[tags[1]])
    b2, c2, p2 = mcnemar_exact(c_g, c_std)
    out['grapheme_vs_bpe'] = {'grapheme_WRR': wrr(c_g), 'bpe_WRR': wrr(c_std),
                              'grapheme_only': b2, 'bpe_only': c2,
                              'mcnemar_p': round(p2, 5),
                              'diff_CI95': boot_diff_ci(c_g, c_std)}
    print(f"\n=== {name} (n={out['n']}) ===")
    print(f"  grapheme {out['grapheme_vs_bpe']['grapheme_WRR']} vs BPE "
          f"{out['grapheme_vs_bpe']['bpe_WRR']}  "
          f"p={out['grapheme_vs_bpe']['mcnemar_p']}  "
          f"diff CI95 {tuple(round(x,1) for x in out['grapheme_vs_bpe']['diff_CI95'])}")
    print(f"  fusion {out['fusion_WRR']} (CI95 {tuple(round(x,1) for x in out['fusion_CI95'])}) "
          f"vs best single [{best_single_tag}] {out['best_single_WRR']}  "
          f"p={out['fusion_vs_best']['mcnemar_p']}  "
          f"diff CI95 {tuple(round(x,1) for x in out['fusion_vs_best']['diff_CI95'])}")
    return out


def main():
    rep = {}
    rep['ours_4way'] = analyze('Our Bengali — 4-way fusion',
                               ['std_ours', 'grph_ours', 'selftrain_ours', 'synthaug_ours'],
                               'synthaug_ours')
    rep['ours_2way'] = analyze('Our Bengali — 2-way fusion',
                               ['std_ours', 'grph_ours'], 'grph_ours')
    rep['bstd_bengali'] = analyze('BSTD Bengali — 2-way fusion',
                                  ['std_bstd', 'grph_bstd'], 'grph_bstd')
    rep['bstd_assamese'] = analyze('BSTD Assamese — 2-way fusion',
                                   ['std_assamese', 'grph_assamese'], 'grph_assamese')
    if os.path.exists(os.path.join(BASE, 'conf_std_hindi.json')):
        # on Hindi the BPE/standard branch is the stronger single model
        rep['bstd_hindi'] = analyze('BSTD Hindi — 2-way fusion',
                                    ['std_hindi', 'grph_hindi'], 'std_hindi')
    json.dump(rep, open(os.path.join(BASE, 'significance_report.json'), 'w'), indent=2)
    print('\nsaved significance_report.json')


if __name__ == '__main__':
    main()
