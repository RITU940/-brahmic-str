#!/usr/bin/env python3
"""Independent re-computation of every headline number in paper/main.tex
from the per-image prediction files (conf_*.json).

Run:  python3 verify_paper_numbers.py
Prints PASS/FAIL against the values quoted in the draft, then the
12-language fixed-budget fusion replication (law runs) and the
confidence-cascade analysis.
"""
import json
import math
import sys

from metrics import normalize_bengali, edit_distance


def load(name):
    with open(name) as f:
        return json.load(f)


def correct(rec):
    return normalize_bengali(rec["gt"]) == normalize_bengali(rec["pred"])


def wrr(recs):
    return 100.0 * sum(correct(r) for r in recs) / len(recs)


def corpus_char_acc(recs):
    ed = tot = 0
    for r in recs:
        g, p = normalize_bengali(r["gt"]), normalize_bengali(r["pred"])
        ed += edit_distance(g, p)
        tot += len(g)
    return 100.0 * (1.0 - ed / tot)


def fuse(*models):
    """Max-confidence fusion across aligned per-image lists."""
    out = []
    for recs in zip(*models):
        out.append(max(recs, key=lambda r: r["conf"]))
    return out


def oracle(*models):
    n = len(models[0])
    return 100.0 * sum(any(correct(m[i]) for m in models) for i in range(n)) / n


def mcnemar_exact(a, b):
    """Exact two-sided McNemar on paired correctness of model lists a, b."""
    x = sum(1 for r, s in zip(a, b) if correct(r) and not correct(s))
    y = sum(1 for r, s in zip(a, b) if not correct(r) and correct(s))
    n, k = x + y, min(x, y)
    if n == 0:
        return 1.0
    p = sum(math.comb(n, i) for i in range(k + 1)) / (1 << n)
    return min(1.0, 2.0 * p)


def auroc(recs):
    pairs = sorted((r["conf"], correct(r)) for r in recs)
    pos = sum(1 for _, c in pairs if c)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    # rank-sum (ties get average rank)
    ranks, i = {}, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[k] = avg
        i = j
    rank_pos = sum(ranks[k] for k, (_, c) in enumerate(pairs) if c)
    return (rank_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def selective(recs, coverage):
    keep = sorted(recs, key=lambda r: -r["conf"])[: max(1, round(len(recs) * coverage))]
    return wrr(keep)


def check(label, got, want, tol=0.06):
    ok = abs(got - want) <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {label:52s} paper={want:<8g} recomputed={got:.2f}")
    return ok


def main():
    fails = 0

    print("== Our Bengali test set (370) ==")
    std = load("conf_std_ours.json")
    grph = load("conf_grph_ours.json")
    st = load("conf_selftrain_ours.json")
    syn = load("conf_synthaug_ours.json")
    assert len(std) == len(grph) == len(st) == len(syn) == 370
    assert all(a["gt"] == b["gt"] == c["gt"] == d["gt"] for a, b, c, d in zip(std, grph, st, syn))
    fours = fuse(std, grph, st, syn)
    twos = fuse(std, grph)
    fails += not check("BPE WRR", wrr(std), 52.4)
    fails += not check("grapheme WRR", wrr(grph), 57.3)
    fails += not check("self-train WRR", wrr(st), 57.0)
    fails += not check("synth-curriculum WRR", wrr(syn), 62.2)
    fails += not check("4-model fusion WRR", wrr(fours), 64.6)
    fails += not check("4-model fusion CharAcc", corpus_char_acc(fours), 82.8)
    fails += not check("4-model oracle", oracle(std, grph, st, syn), 70.8)
    fails += not check("2-model fusion WRR", wrr(twos), 59.2)
    fails += not check("2-model oracle", oracle(std, grph), 63.5)
    only_g = 100.0 * sum(correct(g) and not correct(s) for g, s in zip(grph, std)) / 370
    only_s = 100.0 * sum(correct(s) and not correct(g) for g, s in zip(grph, std)) / 370
    fails += not check("only-grapheme-correct %", only_g, 11.9)
    fails += not check("only-BPE-correct %", only_s, 8.1)
    fails += not check("AUROC BPE", auroc(std), 0.91, tol=0.006)
    fails += not check("AUROC grapheme", auroc(grph), 0.88, tol=0.006)
    fails += not check("AUROC fusion(4)", auroc(fours), 0.88, tol=0.006)
    fails += not check("selective WRR @70%", selective(fours, 0.70), 83.0, tol=0.3)
    fails += not check("selective WRR @50%", selective(fours, 0.50), 89.7, tol=0.3)
    print(f"  McNemar grapheme-vs-BPE p = {mcnemar_exact(grph, std):.4f}  (paper 0.041)")
    best = grph if wrr(grph) >= wrr(std) else std
    print(f"  McNemar fusion4-vs-best-single p = {mcnemar_exact(fours, syn):.4f}  (paper 0.20 vs best branch)")

    for tag, files, ns, want in [
        ("BSTD Bengali", ("conf_std_bstd.json", "conf_grph_bstd.json"), 1368,
         dict(bpe=57.4, grph=59.8, fus=62.1, orc=66.4, p_gb=0.030, p_fb=0.0024)),
        ("BSTD Assamese", ("conf_std_assamese.json", "conf_grph_assamese.json"), 1505,
         dict(bpe=31.6, grph=36.3, fus=38.5, orc=43.8, p_gb=5e-5, p_fb=0.0085)),
        ("BSTD Hindi", ("conf_std_hindi.json", "conf_grph_hindi.json"), 4846,
         dict(bpe=56.9, grph=52.6, fus=61.7, orc=63.1, p_gb=1e-4, p_fb=1e-4)),
    ]:
        print(f"== {tag} ({ns}) ==")
        s, g = load(files[0]), load(files[1])
        assert len(s) == len(g) == ns
        f2 = fuse(s, g)
        fails += not check("BPE WRR", wrr(s), want["bpe"])
        fails += not check("grapheme WRR", wrr(g), want["grph"])
        fails += not check("fusion WRR", wrr(f2), want["fus"])
        fails += not check("oracle", oracle(s, g), want["orc"])
        best = g if wrr(g) >= wrr(s) else s
        print(f"  McNemar grph-vs-BPE p = {mcnemar_exact(g, s):.2e} (paper ~{want['p_gb']:g}); "
              f"fusion-vs-best p = {mcnemar_exact(f2, best):.2e} (paper ~{want['p_fb']:g})")

    print("== PARSeq comparison ==")
    pq_o, pq_b = load("conf_parseq_ours.json"), load("conf_parseq_bstd.json")
    b_bstd = fuse(load("conf_std_bstd.json"), load("conf_grph_bstd.json"))
    fails += not check("PARSeq ours WRR", wrr(pq_o), 66.8)
    fails += not check("PARSeq BSTD WRR", wrr(pq_b), 77.3)
    print(f"  McNemar PARSeq-vs-fusion (ours) p = {mcnemar_exact(pq_o, fours):.3f} (paper 0.50)")
    print(f"  McNemar PARSeq-vs-fusion (BSTD) p = {mcnemar_exact(pq_b, b_bstd):.2e} (paper <1e-4)")

    print(f"\n>>> AUDIT: {fails} FAILURES <<<" if fails else "\n>>> AUDIT: ALL CHECKS PASS <<<")

    # ---------------- 12-language fixed-budget replication ----------------
    print("\n== 12-language fixed-budget (b1800) fusion replication ==")
    print(f"{'lang':12s} {'N':>6s} {'BPE':>6s} {'grph':>6s} {'fus':>6s} {'orc':>6s} "
          f"{'p(g-b)':>9s} {'p(f-best)':>9s}")
    import glob
    rows = {}
    for gf in sorted(glob.glob("conf_grph_law_*.json")):
        lang = gf[len("conf_grph_law_"):-len(".json")]
        g, s = load(gf), load(f"conf_std_law_{lang}.json")
        f2 = fuse(s, g)
        best = g if wrr(g) >= wrr(s) else s
        rows[lang] = dict(N=len(g), bpe=wrr(s), grph=wrr(g), fus=wrr(f2),
                          orc=oracle(s, g), p_gb=mcnemar_exact(g, s),
                          p_fb=mcnemar_exact(f2, best))
        r = rows[lang]
        print(f"{lang:12s} {r['N']:6d} {r['bpe']:6.2f} {r['grph']:6.2f} {r['fus']:6.2f} "
              f"{r['orc']:6.2f} {r['p_gb']:9.2e} {r['p_fb']:9.2e}")
    with open("fusion_replication_law.json", "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    beats = sum(1 for r in rows.values() if r["fus"] >= max(r["bpe"], r["grph"]))
    print(f"fusion >= best single branch in {beats}/{len(rows)} languages "
          f"(saved fusion_replication_law.json)")

    # ---------------- confidence cascade ----------------
    print("\n== Cascade: BPE first, grapheme only if conf < t (our Bengali set) ==")
    print(f"{'t':>5s} {'WRR':>6s} {'2nd-pass%':>9s}")
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01):
        out, second = [], 0
        for s, g in zip(std, grph):
            if s["conf"] >= t:
                out.append(s)
            else:
                second += 1
                out.append(max((s, g), key=lambda r: r["conf"]))
        print(f"{t:5.2f} {wrr(out):6.2f} {100.0 * second / len(std):9.1f}")


if __name__ == "__main__":
    main()
