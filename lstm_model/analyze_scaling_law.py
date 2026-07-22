"""Fit the synthetic-exposure scaling law from the Amendment-5 sweep.

Reads result_zs_scale{B}_{script}.json (sweep-internal points only; the 3240 point is
the external LOSO Rung-B anchor and is reported separately, not fitted) and asks:
  1. is WRR log-linear in synthetic budget, per script?
  2. is the slope the same across scripts (script-invariant), by F-test?
  3. does a separable model  WRR = a + b*tok_cov + c*log2(budget)  fit, and is the
     preregistered c = +11.48 inside the confidence interval?
"""
import glob
import json

import numpy as np
from scipy import stats

TOK_COV = {"malayalam": 79.18, "kannada": 90.80, "telugu": 92.30}
ANCHOR_3240 = {"malayalam": 9.69, "kannada": 15.42, "telugu": 19.82}  # external LOSO Rung-B
PREREG_C = 11.48

obs = {}
for f in glob.glob("result_zs_scale*_*.json"):
    d = json.load(open(f))
    obs[(d["script"], int(d["rung"].replace("scale", "")))] = d["WRR"]

scripts = ["malayalam", "kannada", "telugu"]
budgets = [810, 1620, 6480, 12960]

print("=" * 74)
print("OBSERVED (sweep-internal; 3240 anchor shown in brackets, not fitted)")
print("=" * 74)
for s in scripts:
    row = " ".join(
        f"{b}:{obs[(s, b)]:>6.2f}" if (s, b) in obs else f"{b}:{'--':>6}" for b in budgets
    )
    print(f"{s:<11} cov={TOK_COV[s]:5.2f}  {row}   [3240:{ANCHOR_3240[s]:.2f}]")

# ---- 1. per-script log-linear fits -------------------------------------------------
print()
print("=" * 74)
print("1. PER-SCRIPT FITS:  WRR = intercept + slope * log2(budget)")
print("=" * 74)
per_script = {}
for s in scripts:
    pts = [(np.log2(b), obs[(s, b)]) for b in budgets if (s, b) in obs]
    if len(pts) < 3:
        print(f"{s:<11} only {len(pts)} points — skipped")
        continue
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    r = stats.linregress(x, y)
    per_script[s] = (x, y, r)
    dof = len(x) - 2
    tcrit = stats.t.ppf(0.975, dof)
    lo, hi = r.slope - tcrit * r.stderr, r.slope + tcrit * r.stderr
    print(
        f"{s:<11} n={len(x)}  slope={r.slope:+6.3f} WRR/doubling  "
        f"95%CI[{lo:+.3f},{hi:+.3f}]  R2={r.rvalue**2:.4f}"
    )

# ---- 2. common-slope test (ANCOVA) -------------------------------------------------
print()
print("=" * 74)
print("2. IS THE SLOPE SCRIPT-INVARIANT?  (separate slopes vs one common slope)")
print("=" * 74)
X_all, y_all, sidx = [], [], []
for i, s in enumerate(scripts):
    if s not in per_script:
        continue
    x, y, _ = per_script[s]
    X_all += list(x)
    y_all += list(y)
    sidx += [i] * len(x)
X_all, y_all, sidx = np.array(X_all), np.array(y_all), np.array(sidx)
present = sorted(set(sidx))

# restricted: per-script intercepts, ONE shared slope
Dr = np.zeros((len(y_all), len(present) + 1))
for j, i in enumerate(present):
    Dr[sidx == i, j] = 1.0
Dr[:, -1] = X_all
br, *_ = np.linalg.lstsq(Dr, y_all, rcond=None)
rss_r = float(((y_all - Dr @ br) ** 2).sum())

# full: per-script intercepts AND per-script slopes
Df = np.zeros((len(y_all), 2 * len(present)))
for j, i in enumerate(present):
    m = sidx == i
    Df[m, j] = 1.0
    Df[m, len(present) + j] = X_all[m]
bf, *_ = np.linalg.lstsq(Df, y_all, rcond=None)
rss_f = float(((y_all - Df @ bf) ** 2).sum())

df1 = Df.shape[1] - Dr.shape[1]
df2 = len(y_all) - Df.shape[1]
F = ((rss_r - rss_f) / df1) / (rss_f / df2) if df2 > 0 and rss_f > 0 else float("nan")
p = 1 - stats.f.cdf(F, df1, df2) if np.isfinite(F) else float("nan")
c_common = br[-1]
se_c = float(np.sqrt(rss_r / (len(y_all) - Dr.shape[1]) * np.linalg.pinv(Dr.T @ Dr)[-1, -1]))
tcrit = stats.t.ppf(0.975, len(y_all) - Dr.shape[1])
print(f"separate slopes: {[f'{bf[len(present)+j]:+.3f}' for j in range(len(present))]}")
print(f"common slope c = {c_common:+.3f} WRR/doubling  "
      f"95%CI[{c_common - tcrit*se_c:+.3f},{c_common + tcrit*se_c:+.3f}]")
print(f"F({df1},{df2}) = {F:.3f}, p = {p:.4f}  -> "
      f"{'slopes DIFFER' if p < 0.05 else 'NO evidence slopes differ (script-invariant)'}")
t_pre = (c_common - PREREG_C) / se_c
print(f"vs preregistered c = {PREREG_C}: t = {t_pre:.2f}, "
      f"{'EXCLUDED by 95% CI' if abs(t_pre) > tcrit else 'inside 95% CI'} "
      f"(ratio {PREREG_C/c_common:.2f}x)")

# ---- 3. separable model with coverage ----------------------------------------------
print()
print("=" * 74)
print("3. SEPARABLE MODEL:  WRR = a + b*tok_cov + c*log2(budget)")
print("=" * 74)
cov = np.array([TOK_COV[scripts[i]] for i in sidx])
M = np.column_stack([np.ones_like(y_all), cov, X_all])
beta, *_ = np.linalg.lstsq(M, y_all, rcond=None)
pred = M @ beta
rss = float(((y_all - pred) ** 2).sum())
tss = float(((y_all - y_all.mean()) ** 2).sum())
n, k = len(y_all), M.shape[1]
sigma2 = rss / (n - k)
se = np.sqrt(np.diag(sigma2 * np.linalg.pinv(M.T @ M)))
tc = stats.t.ppf(0.975, n - k)
names = ["intercept a", "coverage b ", "log2budget c"]
for nm, bb, ss in zip(names, beta, se):
    print(f"  {nm} = {bb:+8.3f}  95%CI[{bb - tc*ss:+.3f},{bb + tc*ss:+.3f}]")
print(f"  R2 = {1 - rss/tss:.4f}   RMSE = {np.sqrt(rss/n):.3f} WRR   n = {n}")
print("  residuals:")
for i, s in enumerate(scripts):
    m = sidx == i
    if m.any():
        print(f"    {s:<11} {'  '.join(f'{r:+5.2f}' for r in (y_all - pred)[m])}")

print()
print("  held-out check — predict the external 3240 anchor (never fitted):")
for s in scripts:
    if s not in per_script:
        continue
    ph = beta[0] + beta[1] * TOK_COV[s] + beta[2] * np.log2(3240)
    a = ANCHOR_3240[s]
    print(f"    {s:<11} predicted {ph:5.2f}   anchor {a:5.2f}   resid {a - ph:+5.2f}")

# ---- 4. Rule 3: does the top step saturate? ----------------------------------------
print()
print("=" * 74)
print("4. PREREG RULE 3 — top step 6480->12960 (near-fixed lexicon) vs faithful step")
print("=" * 74)
for s in scripts:
    if (s, 12960) in obs and (s, 6480) in obs:
        top = obs[(s, 12960)] - obs[(s, 6480)]
        faithful = obs[(s, 6480)] - ANCHOR_3240[s]
        print(f"  {s:<11} top step {top:+5.2f}   faithful step {faithful:+5.2f}   "
              f"-> {'comparable: image QUANTITY drives transfer' if top >= 0.7*faithful else 'smaller: saturation/lexicon exhaustion'}")
    elif (s, 12960) not in obs:
        print(f"  {s:<11} 12960 rung still running")

# ---- 5. OUT-OF-SAMPLE: does the fitted law extrapolate to the 6 UNSWEPT scripts? ----
# Each unswept script has exactly one observation: its LOSO Rung-B result, at the
# synth budget recorded in zeroshot_loso_meta_<script>.json. If coverage really sets
# the offset, the law fit on {malayalam,kannada,telugu} should predict these.
print()
print("=" * 74)
print("5. OUT-OF-SAMPLE CHECK — 6 scripts never used in the fit")
print("=" * 74)
HELDOUT = {  # script: (tok_cov, synth budget, observed Rung-B WRR)
    "bengali":    (87.34, 6480, 27.08),
    "devanagari": (94.35, 6480, 30.09),
    "gujarati":   (97.25, 3240, 15.86),
    "gurmukhi":   (90.68, 3240, 22.16),
    "oriya":      (92.64, 3240, 25.38),
    "tamil":      (88.78, 3240, 9.16),
}
resid = []
print(f"{'script':<12}{'cov':>7}{'budget':>8}{'pred':>8}{'obs':>8}{'resid':>8}")
for s, (cv, bud, o) in HELDOUT.items():
    ph = beta[0] + beta[1] * cv + beta[2] * np.log2(bud)
    resid.append(o - ph)
    print(f"{s:<12}{cv:>7.2f}{bud:>8}{ph:>8.2f}{o:>8.2f}{o - ph:>+8.2f}")
resid = np.array(resid)
print(f"\n  out-of-sample RMSE = {np.sqrt((resid**2).mean()):.2f} WRR"
      f"   (in-sample RMSE = {np.sqrt(rss/n):.2f})")
print(f"  mean |resid| = {np.abs(resid).mean():.2f}, range [{resid.min():+.2f}, {resid.max():+.2f}]")
r_cov = stats.pearsonr([HELDOUT[s][0] for s in HELDOUT], [HELDOUT[s][2] for s in HELDOUT])
print(f"  coverage vs WRR across the 6 held-out scripts alone: "
      f"r = {r_cov[0]:+.3f}, p = {r_cov[1]:.3f}")
print("""
  READ THIS BEFORE BUILDING THE PAPER ON THE OFFSET TERM:
  the coverage axis in the fit has only THREE distinct values, so the two coverage
  parameters (a, b) are nearly saturated by construction and the in-sample R2 is
  optimistic. The budget slope is estimated from within-script replication and is
  NOT affected by this; the offset model is.""")
