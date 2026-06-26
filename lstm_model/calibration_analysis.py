"""
Calibration & selective-prediction analysis of beam confidences.
================================================================
Questions a reviewer will ask about max-confidence fusion:
  1. Is beam confidence actually calibrated (does conf ~ P(correct))?  -> ECE, reliability bins
  2. Does fusion work *because* confidence ranks correctness?          -> AUROC of conf vs correct
  3. Selective prediction: if the system may abstain, what accuracy
     at what coverage?                                                 -> risk-coverage curve
All computed from existing conf_<tag>.json files (no GPU).
Outputs: calibration_report.json + printed table.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = os.path.dirname(os.path.abspath(__file__))

def load(tag):
    return json.load(open(os.path.join(BASE, f'conf_{tag}.json')))

def correctness(rows):
    return [1 if (r['gt'] or '').strip() == (r['pred'] or '').strip() else 0 for r in rows]

def ece(confs, corr, n_bins=10):
    n = len(confs)
    total = 0.0
    bins = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i in range(n) if lo < confs[i] <= hi] if b > 0 else \
              [i for i in range(n) if lo <= confs[i] <= hi]
        if not idx:
            bins.append({'bin': f'{lo:.1f}-{hi:.1f}', 'n': 0})
            continue
        acc = sum(corr[i] for i in idx) / len(idx)
        avg_c = sum(confs[i] for i in idx) / len(idx)
        total += (len(idx) / n) * abs(acc - avg_c)
        bins.append({'bin': f'{lo:.1f}-{hi:.1f}', 'n': len(idx),
                     'avg_conf': round(avg_c, 3), 'accuracy': round(acc, 3)})
    return total, bins

def auroc(confs, corr):
    pos = [c for c, y in zip(confs, corr) if y == 1]
    neg = [c for c, y in zip(confs, corr) if y == 0]
    if not pos or not neg:
        return float('nan')
    wins = sum((1.0 if p > q else 0.5 if p == q else 0.0) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))

def risk_coverage(confs, corr):
    order = sorted(range(len(confs)), key=lambda i: -confs[i])
    pts, correct_so_far = [], 0
    for k, i in enumerate(order, 1):
        correct_so_far += corr[i]
        pts.append({'coverage': k / len(order), 'accuracy': correct_so_far / k})
    # report accuracy at fixed coverages
    out = {}
    for cv in (0.5, 0.7, 0.8, 0.9, 1.0):
        best = min(pts, key=lambda p: abs(p['coverage'] - cv))
        out[f'acc@{int(cv*100)}%cov'] = round(100 * best['accuracy'], 2)
    return out, pts

def main():
    report = {}
    sets = {
        'ours': ['std_ours', 'grph_ours', 'selftrain_ours', 'synthaug_ours'],
        'bstd': ['std_bstd', 'grph_bstd'],
    }
    for ds, tags in sets.items():
        for tag in tags:
            f = os.path.join(BASE, f'conf_{tag}.json')
            if not os.path.exists(f):
                continue
            rows = load(tag)
            confs = [r['conf'] for r in rows]
            corr = correctness(rows)
            e, bins = ece(confs, corr)
            a = auroc(confs, corr)
            rc, _ = risk_coverage(confs, corr)
            report[tag] = {'n': len(rows), 'WRR': round(100 * sum(corr) / len(corr), 2),
                           'ECE': round(e, 4), 'AUROC': round(a, 4),
                           'risk_coverage': rc, 'reliability_bins': bins}
            print(f"{tag:16s} n={len(rows):5d} WRR={report[tag]['WRR']:6.2f} "
                  f"ECE={e:.4f} AUROC={a:.4f}  {rc}")

    # fusion-level selective prediction (4-way on ours): conf of the WINNING model
    tags4 = ['std_ours', 'grph_ours', 'selftrain_ours', 'synthaug_ours']
    if all(os.path.exists(os.path.join(BASE, f'conf_{t}.json')) for t in tags4):
        D = {t: load(t) for t in tags4}
        n = len(D[tags4[0]])
        gts = [r['gt'] for r in D[tags4[0]]]
        fused_pred, fused_conf = [], []
        for i in range(n):
            c, p = max(((D[t][i]['conf'], D[t][i]['pred']) for t in tags4), key=lambda x: x[0])
            fused_pred.append(p); fused_conf.append(c)
        corr = [1 if (gts[i] or '').strip() == (fused_pred[i] or '').strip() else 0 for i in range(n)]
        e, bins = ece(fused_conf, corr)
        a = auroc(fused_conf, corr)
        rc, _ = risk_coverage(fused_conf, corr)
        report['fusion4_ours'] = {'n': n, 'WRR': round(100 * sum(corr) / n, 2),
                                  'ECE': round(e, 4), 'AUROC': round(a, 4),
                                  'risk_coverage': rc, 'reliability_bins': bins}
        print(f"{'fusion4_ours':16s} n={n:5d} WRR={report['fusion4_ours']['WRR']:6.2f} "
              f"ECE={e:.4f} AUROC={a:.4f}  {rc}")

    json.dump(report, open(os.path.join(BASE, 'calibration_report.json'), 'w'),
              ensure_ascii=False, indent=2)
    print("Saved calibration_report.json")

if __name__ == '__main__':
    main()
