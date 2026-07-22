#!/usr/bin/env python3
r"""Re-derive every derivable macro in paper_wacv/numbers.tex from the result JSONs.

Run before every submission build:
    python3 verify_wacv_numbers.py
Exit 0 = every checked macro matches its source-of-truth derivation.
Exit 1 = at least one mismatch (printed as FAIL).

Sources of truth:
  result_zs_loso_rung{A,B,Bbpe}_{script}.json   — WRR / CharAcc / N per rung
  law_fit_results_brahmic.json                  — per-language fertility (script mean)
Coverage macros (\cov*) are declared result-blind inputs (computed 2026-06-25);
they are read from numbers.tex and used as inputs to the correlation/fit checks.
"""
import json, re, sys, math, itertools

TEX = "paper_wacv/numbers.tex"
SCRIPTS = ["tamil", "telugu", "kannada", "malayalam", "oriya",
           "gujarati", "bengali", "devanagari", "gurmukhi"]
TWOX = {"bengali", "devanagari"}  # disclosed 2x synthetic budget


def tex_macros(path):
    txt = open(path).read()
    out = {}
    for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", txt):
        out[m.group(1)] = m.group(2)
    return out


def num(s):
    try:
        return float(s.replace("{\\sim}", "").replace("+", ""))
    except ValueError:
        return None


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (sx * sy)


def solve(A, y):
    """Least-squares solve of (A^T A) b = A^T y via Gaussian elimination (pure python)."""
    n, p = len(A), len(A[0])
    XtX = [[sum(A[k][i] * A[k][j] for k in range(n)) for j in range(p)] for i in range(p)]
    Xty = [sum(A[k][i] * y[k] for k in range(n)) for i in range(p)]
    M = [XtX[i][:] + [Xty[i]] for i in range(p)]
    for col in range(p):
        piv = max(range(col, p), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        for r in range(p):
            if r != col:
                f = M[r][col] / M[col][col]
                M[r] = [a - f * b for a, b in zip(M[r], M[col])]
    return [M[i][p] / M[i][i] for i in range(p)]


def linreg_r2(xs, ys):
    """Simple OLS y~x, return R^2 (pure python)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return 1 - ss_res / ss_tot


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    num_ = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num_ / den


NUMWORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
           6: "six", 7: "seven", 8: "eight", 9: "nine"}


def ols_two_factor(cov, is2x, wrr):
    # WRR = a + b*cov + c*is2x, least squares via normal equations
    n = len(wrr)
    X = [[1.0, cov[i], is2x[i]] for i in range(n)]
    XtX = [[sum(X[k][i] * X[k][j] for k in range(n)) for j in range(3)] for i in range(3)]
    Xty = [sum(X[k][i] * wrr[k] for k in range(n)) for i in range(3)]
    # gaussian elimination
    A = [row[:] + [Xty[i]] for i, row in enumerate(XtX)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(A[r][col]))
        A[col], A[piv] = A[piv], A[col]
        for r in range(3):
            if r != col:
                f = A[r][col] / A[col][col]
                A[r] = [a - f * b for a, b in zip(A[r], A[col])]
    beta = [A[i][3] / A[i][i] for i in range(3)]
    resid = [wrr[k] - (beta[0] + beta[1] * cov[k] + beta[2] * is2x[k]) for k in range(n)]
    rmse = math.sqrt(sum(r * r for r in resid) / n)
    return beta, rmse


def main():
    tex = tex_macros(TEX)
    res = {}
    for s in SCRIPTS:
        for rung, key in [("A", "A"), ("B", "B"), ("Bbpe", "P")]:
            res[(key, s)] = json.load(open(f"result_zs_loso_rung{rung}_{s}.json"))

    fert_by_script = {}
    for row in json.load(open("law_fit_results_brahmic.json"))["rows"]:
        fert_by_script.setdefault(row["script"].lower(), []).append(row["fertility"])
    fert = {k: sum(v) / len(v) for k, v in fert_by_script.items()}
    if "odia" in fert and "oriya" not in fert:
        fert["oriya"] = fert["odia"]

    checks = []  # (macro, tex_value_str, derived_float, decimals)

    def add(name, derived, decimals=2):
        checks.append((name, tex.get(name), derived, decimals))

    for s in SCRIPTS:
        add(f"wrrA{s}", res[("A", s)]["WRR"])
        add(f"wrrB{s}", res[("B", s)]["WRR"])
        add(f"wrrP{s}", res[("P", s)]["WRR"])
        add(f"chaB{s}", res[("B", s)]["CharAcc"])
        add(f"chaP{s}", res[("P", s)]["CharAcc"])
        add(f"n{s}", res[("B", s)]["N"], 0)

    wrrB = [res[("B", s)]["WRR"] for s in SCRIPTS]
    wrrP = [res[("P", s)]["WRR"] for s in SCRIPTS]
    add("wrrBmin", min(wrrB))
    add("wrrBmax", max(wrrB))

    ratios = [b / p for b, p in zip(wrrB, wrrP)]
    add("bpeRatioMin", min(ratios))
    add("bpeRatioMax", max(ratios))
    n_pos = sum(1 for b, p in zip(wrrB, wrrP) if b > p)
    assert n_pos == 9, f"sign test premise violated: only {n_pos}/9 positive"
    add("bpeSignP", 2 * 0.5 ** 9, 4)

    fvals = [fert[s] for s in SCRIPTS]
    add("spearFert", spearman(fvals, wrrB))
    rho_ratio = spearman(fvals, ratios)
    add("spearFertPenalty", rho_ratio)
    add("spearFertPenaltyDiff", spearman(fvals, [b - p for b, p in zip(wrrB, wrrP)]))

    # exact two-sided permutation p for the fertility-penalty correlation
    hits = 0
    total = 0
    for perm in itertools.permutations(range(9)):
        total += 1
        if abs(spearman(fvals, [ratios[i] for i in perm])) >= abs(rho_ratio) - 1e-12:
            hits += 1
    add("spearFertPenaltyP", hits / total, 3)

    cov = [num(tex[f"cov{s}"]) for s in SCRIPTS]  # declared result-blind inputs
    add("spearCovRaw", spearman(cov, wrrB))
    eq = [i for i, s in enumerate(SCRIPTS) if s not in TWOX]
    add("spearCovEqual", spearman([cov[i] for i in eq], [wrrB[i] for i in eq]))

    beta, rmse = ols_two_factor(cov, [1.0 if s in TWOX else 0.0 for s in SCRIPTS], wrrB)
    add("tfIntercept", beta[0])
    add("tfCovCoef", beta[1], 3)
    add("tfTwoxBonus", beta[2])
    add("tfRMSE", rmse)

    # ---- Qwen2.5-VL-7B extraction re-score (Amendment 4c, disclosed secondary) ----
    vlm = {s: json.load(open(f"result_vlm_qwen25_extracted_{s}.json")) for s in SCRIPTS}
    vw = {s: vlm[s]["WRR_extracted"] for s in SCRIPTS}
    vn = {s: vlm[s]["N"] for s in SCRIPTS}
    for s in SCRIPTS:
        add(f"vlmWrr{s.capitalize()}", vw[s])
    assert vw["tamil"] == max(vw[s] for s in SCRIPTS if s not in ("bengali", "devanagari"))
    add("vlmMacro", sum(vw[s] for s in SCRIPTS) / 9)
    add("vlmMicro", sum(vw[s] * vn[s] for s in SCRIPTS) / sum(vn[s] for s in SCRIPTS))
    scripts_won = sum(1 for s in SCRIPTS if res[("B", s)]["WRR"] > vw[s])  # word macro, checked below

    # ---- synthetic-exposure scaling sweep (Amendment 5); 3240 anchor = external Rung-B ----
    SWEEP = ["malayalam", "kannada", "telugu"]
    BUD = [810, 1620, 6480, 12960]
    sobs = {(s, b): json.load(open(f"result_zs_scale{b}_{s}.json"))["WRR"]
            for s in SWEEP for b in BUD}
    r2s = {s: linreg_r2([math.log2(b) for b in BUD], [sobs[(s, b)] for b in BUD]) for s in SWEEP}
    add("scaleRtwoMax", max(r2s.values()), 3)
    add("scaleRtwoMin", min(r2s.values()), 3)
    # common (script-invariant) slope: per-script intercepts + one shared log2-budget slope
    rows = [(i, math.log2(b), sobs[(s, b)]) for i, s in enumerate(SWEEP) for b in BUD]
    dmat = [[1.0 if r[0] == k else 0.0 for k in range(len(SWEEP))] + [r[1]] for r in rows]
    common_slope = solve(dmat, [r[2] for r in rows])[-1]
    add("scaleSlope", common_slope, 3)
    add("scaleSlopeRatio", num(tex["scalePreregC"]) / common_slope, 1)
    faithful = [sobs[(s, 6480)] - res[("B", s)]["WRR"] for s in SWEEP]
    add("scaleFaithfulObs", sum(faithful) / len(faithful))
    # out-of-sample: coverage vs WRR across the 6 scripts never used in the sweep fit
    oos = [s for s in SCRIPTS if s not in SWEEP]
    add("covOOSr", pearson([num(tex[f"cov{s}"]) for s in oos],
                           [res[("B", s)]["WRR"] for s in oos]))
    # declared/statistical macros NOT re-derived here (need scipy dists; verified via
    # analyze_scaling_law.py): scaleSlopeCI, scaleFp, covOOSp, scalePreregC, scaleFaithfulPred.

    fails = 0
    for name, tex_val, derived, dec in checks:
        if tex_val is None:
            print(f"FAIL  {name:22s} missing from numbers.tex (derived {derived})")
            fails += 1
            continue
        t = num(tex_val)
        d = round(derived, dec)
        # macros are printed at varying precision; match at the tex value's own precision
        tol = 10 ** -(len(tex_val.split(".")[1]) if "." in tex_val else 0) / 2 + 1e-9
        ok = t is not None and abs(t - derived) <= max(tol, 10 ** -dec / 2 + 1e-9)
        print(f"{'ok  ' if ok else 'FAIL'}  {name:22s} tex={tex_val:<10s} derived={d}")
        fails += 0 if ok else 1

    # word-valued macro: scripts where our Rung-B beats the VLM (VLM leads only Devanagari)
    won_word = NUMWORD.get(scripts_won, str(scripts_won))
    tv = tex.get("vlmScriptsWon")
    ok = tv == won_word
    print(f"{'ok  ' if ok else 'FAIL'}  {'vlmScriptsWon':22s} tex={tv!s:<10s} derived={won_word} (count {scripts_won}/9)")
    fails += 0 if ok else 1

    print(f"\n{len(checks) + 1} macros checked, {fails} mismatch(es).")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
