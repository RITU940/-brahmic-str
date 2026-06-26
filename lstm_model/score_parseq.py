"""
Score the IndicPhotoOCR PARSeq predictions (conf_parseq_{ours,bstd}.json)
and compare head-to-head against our fusion system on the same images:
WRR/CharAcc/1-NED via metrics.evaluate_corpus, exact McNemar on paired
correctness, and 95% bootstrap CI on the WRR difference.
Run in the main env (no torch needed):  python score_parseq.py
Writes parseq_report.json.
"""
import os, sys, json

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from metrics import evaluate_corpus, normalize_bengali
from significance_tests import fuse, correct, mcnemar_exact, boot_diff_ci, boot_ci

FUSION_TAGS = {
    'ours': ['std_ours', 'grph_ours', 'selftrain_ours', 'synthaug_ours'],
    'bstd': ['std_bstd', 'grph_bstd'],
}


def main():
    rep = {}
    for which, tags in FUSION_TAGS.items():
        path = os.path.join(BASE, f'conf_parseq_{which}.json')
        if not os.path.exists(path):
            print(f'-- conf_parseq_{which}.json missing, skipping')
            continue
        parseq = json.load(open(path))
        _, fused = fuse(tags)
        assert len(parseq) == len(fused), f'{which}: {len(parseq)} vs {len(fused)}'
        # sanity: same gt order
        for a, b in zip(parseq, fused):
            assert normalize_bengali(a['gt']) == normalize_bengali(b['gt'])

        m_p = evaluate_corpus([r['gt'] for r in parseq], [r['pred'] for r in parseq])
        m_f = evaluate_corpus([r['gt'] for r in fused], [r['pred'] for r in fused])
        c_p, c_f = correct(parseq), correct(fused)
        b, c, p = mcnemar_exact(c_f, c_p)   # b: fusion-only, c: parseq-only
        out = {
            'n': len(parseq),
            'parseq': {k: round(m_p[k], 2) for k in ('WRR', 'char_accuracy', '1-NED', 'CER')},
            'parseq_WRR_CI95': boot_ci(c_p),
            'fusion': {k: round(m_f[k], 2) for k in ('WRR', 'char_accuracy', '1-NED', 'CER')},
            'fusion_only_correct': b, 'parseq_only_correct': c,
            'mcnemar_p': p,
            'fusion_minus_parseq_WRR_CI95': boot_diff_ci(c_f, c_p),
        }
        rep[which] = out
        print(f"\n=== {which} (n={out['n']}) ===")
        print(f"  PARSeq  WRR {out['parseq']['WRR']}  CharAcc {out['parseq']['char_accuracy']}  1-NED {out['parseq']['1-NED']}")
        print(f"  Fusion  WRR {out['fusion']['WRR']}  CharAcc {out['fusion']['char_accuracy']}  1-NED {out['fusion']['1-NED']}")
        print(f"  McNemar p={out['mcnemar_p']:.2g}  diff CI95 "
              f"{tuple(round(x, 1) for x in out['fusion_minus_parseq_WRR_CI95'])} "
              f"(fusion-only {b}, parseq-only {c})")
    json.dump(rep, open(os.path.join(BASE, 'parseq_report.json'), 'w'), indent=2)
    print('\nsaved parseq_report.json')


if __name__ == '__main__':
    main()
