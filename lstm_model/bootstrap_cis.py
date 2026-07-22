#!/usr/bin/env python3
"""Canonical, seeded bootstrap of per-script 95% WRR confidence intervals.

Source of truth for the \ciB* / \ciP* / \bpeCIdisjoint macros in
paper_wacv/numbers.tex (the pure-python verify_wacv_numbers.py cannot reproduce
numpy's RNG, so this seeded script is the reproducibility contract).

  python3 bootstrap_cis.py            # print the macro block + the disjoint-CI stat
Percentile bootstrap, 5000 resamples over test samples, seed 20260721.
Point estimates are asserted to match the committed result_zs_loso_* WRRs.
"""
import json, sys
import numpy as np

sys.path.insert(0, ".")
from metrics import compute_wrr

SCR = ["tamil", "telugu", "kannada", "malayalam", "oriya",
       "gujarati", "bengali", "devanagari", "gurmukhi"]
SEED, B = 20260721, 5000


def per_sample(conf):
    d = json.load(open(conf))
    return np.array([compute_wrr(r["gt"], r["pred"]) for r in d], dtype=float)


def ci(x, rng):
    idx = rng.integers(0, len(x), size=(B, len(x)))
    m = x[idx].mean(1) * 100.0
    return round(float(np.percentile(m, 2.5)), 1), round(float(np.percentile(m, 97.5)), 1)


def main():
    rng = np.random.default_rng(SEED)
    disjoint = 0
    out = []
    for s in SCR:
        xb = per_sample(f"conf_zs_loso_rungB_{s}.json")
        xp = per_sample(f"conf_zs_loso_rungBbpe_{s}.json")
        resB = json.load(open(f"result_zs_loso_rungB_{s}.json"))["WRR"]
        resP = json.load(open(f"result_zs_loso_rungBbpe_{s}.json"))["WRR"]
        assert abs(xb.mean() * 100 - resB) < 0.25, f"{s}: B point {xb.mean()*100:.2f} != {resB}"
        assert abs(xp.mean() * 100 - resP) < 0.25, f"{s}: Bbpe point {xp.mean()*100:.2f} != {resP}"
        blo, bhi = ci(xb, rng)
        plo, phi = ci(xp, rng)
        disjoint += blo > phi
        out.append((s, blo, bhi, plo, phi))

    print(f"% --- per-script 95pct bootstrap CIs (bootstrap_cis.py, seed {SEED}, {B} resamples) ---")
    for s, blo, bhi, plo, phi in out:
        print(f"\\newcommand{{\\ciB{s}}}{{[{blo:.1f},\\,{bhi:.1f}]}}")
    for s, blo, bhi, plo, phi in out:
        print(f"\\newcommand{{\\ciP{s}}}{{[{plo:.1f},\\,{phi:.1f}]}}")
    print(f"\\newcommand{{\\bpeCIdisjoint}}{{{disjoint}}}  % scripts whose Rung-B CI lies strictly above the BPE CI")
    print(f"\n%% {disjoint}/9 scripts: pivot 95% CI strictly above BPE 95% CI", file=sys.stderr)


if __name__ == "__main__":
    main()
