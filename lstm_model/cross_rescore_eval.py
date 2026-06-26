"""
Cross-tokenizer agreement rescoring — offline evaluation (CPU phase).
Reads cross_scores_<tag>.json and compares selection rules:
  1. best single branch (top-1 of each model)
  2. max-conf word fusion (baseline = current paper method)
  3. PoE agreement: argmax_c mean_m avg_logprob_m(c)         [equal weight]
  4. PoE + generation prior: agreement + log of best beam prob
  5. oracle over the n-best UNION (new, higher ceiling than top-1 oracle)
Reports WRR/char metrics via the canonical scorer.
"""
import os, sys, json, argparse, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import evaluate_corpus, normalize_bengali as N

BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_tag', required=True)
    a = ap.parse_args()
    D = json.load(open(os.path.join(BASE, f'cross_scores_{a.in_tag}.json')))
    tags = sorted(D[0]['xs'].keys())
    gts = [r['gt'] for r in D]
    n = len(D)

    def report(name, preds):
        m = evaluate_corpus(gts, preds, verbose=False)
        print(f"{name:34s} WRR {m['WRR']:6.2f} | CharAcc {m['char_accuracy']:6.2f} "
              f"| 1-NED {m['1-NED']:6.2f}")
        return m

    results = {}
    # 1. each branch's own top-1 (highest gen_conf among its own candidates)
    for t in tags:
        preds = []
        for r in D:
            g = r['gen_conf'].get(t) or {}
            preds.append(max(g, key=g.get) if g else '')
        results[f'single_{t}'] = report(f'single {t}', preds)

    # 2. max-conf word fusion baseline
    preds = []
    for r in D:
        best, bc = '', -1
        for t in tags:
            g = r['gen_conf'].get(t) or {}
            if g:
                c = max(g, key=g.get)
                if g[c] > bc:
                    bc, best = g[c], c
        preds.append(best)
    results['fusion_maxconf'] = report('FUSION max-conf (baseline)', preds)

    # 3. PoE agreement (equal-weight mean of avg logprobs)
    preds = []
    for r in D:
        if not r['candidates']:
            preds.append(''); continue
        sc = {c: sum(r['xs'][t].get(c, -99.0) for t in tags) / len(tags)
              for c in r['candidates']}
        preds.append(max(sc, key=sc.get))
    results['poe_agreement'] = report('PoE agreement (xs only)', preds)

    # 4. PoE + generation prior (add log best-beam-prob, small weight sweep)
    import math
    for w in (0.25, 0.5, 1.0):
        preds = []
        for r in D:
            if not r['candidates']:
                preds.append(''); continue
            sc = {}
            for c in r['candidates']:
                xs = sum(r['xs'][t].get(c, -99.0) for t in tags) / len(tags)
                gp = max((r['gen_conf'][t].get(c, 0.0) for t in tags), default=0.0)
                sc[c] = xs + w * math.log(max(gp, 1e-6)) / 10
            preds.append(max(sc, key=sc.get))
        results[f'poe_plus_prior_w{w}'] = report(f'PoE + gen prior (w={w})', preds)

    # 5. union oracle
    orc = 100 * sum(1 for r in D
                    if any(N(c) == N(r['gt']) for c in r['candidates'])) / n
    print(f"{'ORACLE over n-best union':34s} WRR {orc:6.2f}")
    results['union_oracle_WRR'] = orc

    json.dump(results, open(os.path.join(BASE, f'cross_rescore_results_{a.in_tag}.json'), 'w'),
              indent=2)
    print(f"saved cross_rescore_results_{a.in_tag}.json")


if __name__ == '__main__':
    main()
