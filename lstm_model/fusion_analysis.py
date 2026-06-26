"""
Dual-tokenization fusion: combine standard-BPE and grapheme Florence-2 decoders
by beam confidence. Parameter-free (max-confidence) -> no overfitting.
Reports each model, the fusion, and the oracle ceiling, using the project metric.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import evaluate_corpus, normalize_bengali

BASE = os.path.dirname(os.path.abspath(__file__))

def load(tag):
    f = os.path.join(BASE, f'conf_{tag}.json')
    return json.load(open(f)) if os.path.exists(f) else None

def wrr(gts, preds):
    return evaluate_corpus(gts, preds, verbose=False)['WRR']

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--std', default='std_ours'); ap.add_argument('--grph', default='grph_ours')
    ap.add_argument('--self', dest='self_tag', default='selftrain_ours')
    a = ap.parse_args()
    std = load(a.std); grph = load(a.grph); self_ = load(a.self_tag)
    assert std and grph, f"need conf_{a.std}.json and conf_{a.grph}.json"
    n = len(std)
    gts = [r['gt'] for r in std]
    S = [r['pred'] for r in std]; cS = [r['conf'] for r in std]
    G = [r['pred'] for r in grph]; cG = [r['conf'] for r in grph]

    # Parameter-free max-confidence fusion (BPE + grapheme)
    fused = [G[i] if cG[i] >= cS[i] else S[i] for i in range(n)]

    print(f"N={n}")
    print(f"Standard           WRR: {wrr(gts,S):.2f}")
    print(f"Grapheme           WRR: {wrr(gts,G):.2f}")
    print(f"FUSION (max-conf)  WRR: {wrr(gts,fused):.2f}")

    # add self-train into a 3-way fusion if available
    if self_:
        T = [r['pred'] for r in self_]; cT = [r['conf'] for r in self_]
        fused3 = []
        for i in range(n):
            cands = [(cS[i], S[i]), (cG[i], G[i]), (cT[i], T[i])]
            fused3.append(max(cands, key=lambda x: x[0])[1])
        print(f"Self-train         WRR: {wrr(gts,T):.2f}")
        print(f"FUSION3 (max-conf) WRR: {wrr(gts,fused3):.2f}")

    # oracle ceiling -- MUST use the SAME normalization as the WRR metric
    # (normalize_bengali: NFC + zero-width removal + whitespace collapse), otherwise
    # the oracle applies a stricter correctness test than WRR and can fall BELOW the
    # fusion WRR it is meant to upper-bound (observed for Malayalam: 25.59 < 26.69).
    def norm(s): return normalize_bengali(s or '')
    cor = lambda P: [norm(gts[i]) == norm(P[i]) for i in range(n)]
    oS, oG = cor(S), cor(G)
    orc = 100*sum(1 for i in range(n) if oS[i] or oG[i])/n
    print(f"ORACLE ceiling     WRR: {orc:.2f}")

    json.dump({'fused_maxconf': fused, 'gts': gts},
              open(os.path.join(BASE, 'fusion_predictions.json'), 'w'), ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
