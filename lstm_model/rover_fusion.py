"""
Character-level ROVER-style voting fusion (vs word-level max-conf fusion).
==========================================================================
Word-level max-conf fusion must pick ONE model's whole word. But models often
get *different parts* of the same word right. ROVER aligns the candidate
strings at the character level (pivot = highest-confidence candidate) and
votes per position, weighted by model confidence — so the fused word can be
better than every individual candidate.

Reads conf_<tag>.json files; reports word-level and char-level metrics for
each model, max-conf fusion, and ROVER fusion.
"""
import os, sys, json, argparse
from difflib import SequenceMatcher
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import evaluate_corpus

BASE = os.path.dirname(os.path.abspath(__file__))


def align_to_pivot(pivot, cand):
    """Return list (len(pivot)+1) of per-slot chars from cand aligned to pivot.
    Slot i in [0, len(pivot)) = char aligned to pivot[i] ('' = deletion).
    Insertions are appended to the preceding slot's string."""
    slots = [''] * (len(pivot) + 1)
    sm = SequenceMatcher(None, pivot, cand, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal':
            for k in range(i2 - i1):
                slots[i1 + k] = cand[j1 + k]
        elif op == 'replace':
            seg = cand[j1:j2]
            n = i2 - i1
            for k in range(n):
                slots[i1 + k] = seg[k] if k < len(seg) else ''
            if len(seg) > n:
                slots[i2 - 1] += seg[n:]
        elif op == 'delete':
            for k in range(i1, i2):
                slots[k] = ''
        elif op == 'insert':
            slots[i1] = seg = cand[j1:j2] + slots[i1]
    return slots


def rover_one(cands, confs):
    """cands: list of strings, confs: list of weights. Vote per pivot slot."""
    order = sorted(range(len(cands)), key=lambda i: -confs[i])
    pivot = cands[order[0]]
    if not pivot:
        return max(zip(confs, cands))[1]
    aligned = []
    for i in order:
        if cands[i] == pivot:
            aligned.append((list(pivot) + [''], confs[i]))
        else:
            aligned.append((align_to_pivot(pivot, cands[i]), confs[i]))
    out = []
    for s in range(len(pivot)):
        votes = {}
        for slots, w in aligned:
            votes[slots[s]] = votes.get(slots[s], 0.0) + w
        out.append(max(votes.items(), key=lambda kv: kv[1])[0])
    return ''.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+',
                    default=['std_ours', 'grph_ours', 'selftrain_ours', 'synthaug_ours'])
    ap.add_argument('--out', default='rover_result.json')
    a = ap.parse_args()

    D = {t: json.load(open(os.path.join(BASE, f'conf_{t}.json'))) for t in a.tags}
    n = len(D[a.tags[0]])
    gts = [r['gt'] for r in D[a.tags[0]]]

    def report(name, preds):
        m = evaluate_corpus(gts, preds, verbose=False)
        print(f"{name:22s} WRR {m['WRR']:6.2f} | CharAcc {m['char_accuracy']:6.2f} "
              f"| CER {m['CER']:6.2f} | 1-NED {m['1-NED']:6.2f}")
        return m

    for t in a.tags:
        report(t, [r['pred'] for r in D[t]])

    # max-conf (word-level) baseline fusion
    maxconf = []
    for i in range(n):
        maxconf.append(max(((D[t][i]['conf'], D[t][i]['pred']) for t in a.tags),
                           key=lambda x: x[0])[1])
    report('FUSION max-conf', maxconf)

    rover = []
    for i in range(n):
        cands = [D[t][i]['pred'] for t in a.tags]
        confs = [D[t][i]['conf'] for t in a.tags]
        rover.append(rover_one(cands, confs))
    m = report('FUSION ROVER (char)', rover)

    # oracle
    norm = lambda s: (s or '').strip()
    orc = 100 * sum(1 for i in range(n)
                    if any(norm(gts[i]) == norm(D[t][i]['pred']) for t in a.tags)) / n
    print(f"{'ORACLE (word pick)':22s} WRR {orc:6.2f}")

    json.dump({'tags': a.tags, 'rover_preds': rover, 'maxconf_preds': maxconf, 'gts': gts,
               'rover_metrics': m},
              open(os.path.join(BASE, a.out), 'w'), ensure_ascii=False, indent=2)
    print(f"saved {a.out}")


if __name__ == '__main__':
    main()
