#!/usr/bin/env python3
"""Cascaded two-branch inference: decode with the BPE branch first and
invoke the grapheme branch only when BPE beam confidence falls below a
threshold t (then keep the more confident of the two outputs).

Answers the doubled-inference-cost question for the fusion paper: how much
of the full-fusion WRR gain survives at what fraction of second decodes.
Same rule and thresholds for every dataset (nothing tuned on test).

Run:  python3 cascade_analysis.py     -> prints table, saves cascade_report.json
"""
import json

from metrics import normalize_bengali

SETS = {
    "ours_bengali": ("conf_std_ours.json", "conf_grph_ours.json"),
    "bstd_bengali": ("conf_std_bstd.json", "conf_grph_bstd.json"),
    "bstd_assamese": ("conf_std_assamese.json", "conf_grph_assamese.json"),
    "bstd_hindi": ("conf_std_hindi.json", "conf_grph_hindi.json"),
}
THRESHOLDS = (0.80, 0.90, 0.95, 0.99)


def ok(r):
    return normalize_bengali(r["gt"]) == normalize_bengali(r["pred"])


def wrr(recs):
    return 100.0 * sum(ok(r) for r in recs) / len(recs)


def main():
    report = {}
    print(f"{'dataset':14s} {'BPE':>6s} {'fus':>6s} | " +
          " | ".join(f"t={t:.2f}: WRR 2nd% gain%" for t in THRESHOLDS))
    for name, (fs, fg) in SETS.items():
        std = json.load(open(fs))
        grph = json.load(open(fg))
        assert len(std) == len(grph)
        full = [max(p, key=lambda r: r["conf"]) for p in zip(std, grph)]
        w_std, w_full = wrr(std), wrr(full)
        row = {"n": len(std), "wrr_bpe": round(w_std, 2), "wrr_fusion": round(w_full, 2),
               "points": []}
        cells = []
        for t in THRESHOLDS:
            out, second = [], 0
            for s, g in zip(std, grph):
                if s["conf"] >= t:
                    out.append(s)
                else:
                    second += 1
                    out.append(max((s, g), key=lambda r: r["conf"]))
            w = wrr(out)
            frac = 100.0 * second / len(std)
            gain = 100.0 * (w - w_std) / (w_full - w_std) if w_full > w_std else 100.0
            row["points"].append({"t": t, "wrr": round(w, 2),
                                  "second_pass_pct": round(frac, 1),
                                  "gain_retained_pct": round(gain, 1)})
            cells.append(f"{w:6.2f} {frac:5.1f} {gain:5.1f}")
        report[name] = row
        print(f"{name:14s} {w_std:6.2f} {w_full:6.2f} | " + " | ".join(cells))
    with open("cascade_report.json", "w") as f:
        json.dump(report, f, indent=1)
    print("Saved cascade_report.json")


if __name__ == "__main__":
    main()
