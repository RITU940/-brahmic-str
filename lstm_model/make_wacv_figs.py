#!/usr/bin/env python3
"""Generate the WACV figures from the result JSONs (reproducible).

  figs/fig2_bars.pdf     — per-script transfer: Rung A / B_bpe / B (WRR)
  figs/fig3_scaling.pdf  — synthetic-budget dose-response, observed vs preregistered

Run in the `omni` conda env (matplotlib). Colorblind-safe Wong palette.
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "paper_wacv", "figs")
os.makedirs(OUT, exist_ok=True)

# Wong colorblind-safe palette
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#E69F00", "#009E73", "#999999"
plt.rcParams.update({
    "font.size": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "legend.frameon": False, "pdf.fonttype": 42,
})

SCRIPTS = ["tamil", "telugu", "kannada", "malayalam", "oriya",
           "gujarati", "bengali", "devanagari", "gurmukhi"]


def wrr(rung, s):
    return json.load(open(os.path.join(BASE, f"result_zs_loso_rung{rung}_{s}.json")))["WRR"]


# ---------------------------------------------------------------- Fig 2: bars
def fig_bars():
    order = sorted(SCRIPTS, key=lambda s: wrr("B", s))  # ascending by transfer
    A = [wrr("A", s) for s in order]
    P = [wrr("Bbpe", s) for s in order]
    B = [wrr("B", s) for s in order]
    x = range(len(order))
    w = 0.27
    fig, ax = plt.subplots(figsize=(6.9, 2.5))
    ax.bar([i - w for i in x], A, w, label="Rung A (source only)", color=GREY)
    ax.bar([i for i in x], P, w, label=r"Rung B$_{\mathrm{bpe}}$ (stock tokenizer)", color=ORANGE)
    ax.bar([i + w for i in x], B, w, label="Rung B (grapheme pivot)", color=BLUE)
    ax.set_xticks(list(x))
    ax.set_xticklabels([s.capitalize() for s in order], rotation=20, ha="right")
    ax.set_ylabel("WRR (%)")
    ax.set_ylim(0, max(B) * 1.15)
    ax.legend(ncol=3, loc="upper left", fontsize=7.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, "fig2_bars.pdf"))
    plt.close(fig)


# ------------------------------------------------------- Fig 3: dose-response
def fig_scaling():
    SWEEP = {"malayalam": BLUE, "kannada": ORANGE, "telugu": GREEN}
    BUD = [810, 1620, 6480, 12960]
    anchor3240 = {s: wrr("B", s) for s in SWEEP}  # external LOSO Rung-B anchor
    # preregistered point predictions (PROSPECTIVE_PREDICTION_SCALING.md, clipped)
    PRED = {"malayalam": {6480: 21.2, 12960: 32.7},
            "kannada": {6480: 26.9, 12960: 38.4},
            "telugu": {6480: 31.3, 12960: 42.8}}

    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    for s, c in SWEEP.items():
        obs = {b: json.load(open(os.path.join(BASE, f"result_zs_scale{b}_{s}.json")))["WRR"]
               for b in BUD}
        xs = [math.log2(b) for b in BUD]
        ys = [obs[b] for b in BUD]
        # observed points + log-linear fit
        n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
        beta = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / sum((a - mx) ** 2 for a in xs)
        alpha = my - beta * mx
        xr = [math.log2(700), math.log2(15000)]
        ax.plot(xr, [alpha + beta * x for x in xr], "-", color=c, lw=1.1, zorder=1)
        ax.plot(xs, ys, "o", color=c, ms=4.5, label=s.capitalize(), zorder=3)
        ax.plot(math.log2(3240), anchor3240[s], "D", color=c, ms=4.5,
                mfc="white", mew=1.0, zorder=3)  # external anchor
        # preregistered predictions (open squares) — the gap is the magnitude miss
        px = [math.log2(b) for b in PRED[s]]
        py = [PRED[s][b] for b in PRED[s]]
        ax.plot(px, py, "s", color=c, ms=4.5, mfc="none", mew=1.0, zorder=2)

    ax.set_xticks([math.log2(b) for b in [810, 1620, 3240, 6480, 12960]])
    ax.set_xticklabels(["810", "1.6k", "3.2k", "6.5k", "13k"])
    ax.set_xlabel("synthetic images (log scale)")
    ax.set_ylabel("WRR (%)")
    ax.set_ylim(0, 46)
    # two mini-legends: scripts (colors) and marker meanings
    leg1 = ax.legend(loc="upper left", fontsize=7, title="held-out script", title_fontsize=7)
    ax.add_artist(leg1)
    from matplotlib.lines import Line2D
    marks = [Line2D([], [], marker="o", color="k", ls="", ms=4.5, label="observed"),
             Line2D([], [], marker="D", color="k", ls="", ms=4.5, mfc="w", label="Rung-B anchor"),
             Line2D([], [], marker="s", color="k", ls="", ms=4.5, mfc="none", label="preregistered")]
    ax.legend(handles=marks, loc="lower right", fontsize=6.5)
    ax.text(0.03, 0.62, "slope: pre-reg $+11.5$,\nobserved $+2.2$/doubling",
            transform=ax.transAxes, fontsize=6.5, color="#333333")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig3_scaling.pdf"))
    plt.close(fig)


# --------------------------------------------------- Fig 4: prospective receipts
def fig_receipts():
    """Predicted vs observed for every quantitative forecast filed pre-run.
    LOSO point calls land on the diagonal (hits); the scaling rungs fall off it
    (low budget above = concave; high budget below = the magnitude miss)."""
    RMSE = 4.18
    # preregistered scaling predictions (PROSPECTIVE_PREDICTION_SCALING.md, clipped)
    PRED = {"malayalam": {810: 0.18, 1620: 0.18, 6480: 21.2, 12960: 32.7},
            "kannada":   {810: 2.78, 1620: 3.9, 6480: 26.9, 12960: 38.4},
            "telugu":    {810: 1.28, 1620: 8.3, 6480: 31.3, 12960: 42.8}}
    sx, sy = [], []
    for s in PRED:
        for b in PRED[s]:
            sx.append(PRED[s][b])
            sy.append(json.load(open(os.path.join(BASE, f"result_zs_scale{b}_{s}.json")))["WRR"])
    # LOSO point calls filed before their rung trained
    loso = [("Kannada", 15.0, 15.42), ("Gurmukhi", 16.2, 22.16)]

    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    lim = 46
    ax.plot([0, lim], [0, lim], "-", color="#444444", lw=0.9, zorder=1)
    ax.fill_between([0, lim], [-RMSE, lim - RMSE], [RMSE, lim + RMSE],
                    color=GREY, alpha=0.25, lw=0, zorder=0, label=r"$\pm1$ RMSE")
    ax.scatter(sx, sy, s=26, color=BLUE, zorder=3, label="scaling rungs (pre-reg)")
    for name, px, py in loso:
        ax.scatter([px], [py], s=44, color=ORANGE, marker="D", zorder=4,
                   edgecolor="k", linewidth=0.4)
        ax.annotate(name, (px, py), textcoords="offset points", xytext=(6, -2), fontsize=6.5)
    ax.scatter([], [], s=44, color=ORANGE, marker="D", label="LOSO point calls")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("preregistered prediction (WRR %)")
    ax.set_ylabel("realized WRR (%)")
    ax.text(24, 6, "over-predicted\n(slope miss)", fontsize=6, color="#555555", ha="center")
    ax.text(9, 20, "under-predicted\n(concave low end)", fontsize=6, color="#555555", ha="center")
    ax.legend(loc="upper left", fontsize=6.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig4_receipts.pdf"))
    plt.close(fig)


# ------------------------------------------------ Fig 1: shared pivot mechanism
def fig_pivot():
    """Schematic: the ISCII offset alignment collapses the same letter across
    Brahmic scripts onto one pivot code, so a held-out script's graphemes land on
    codes already seen from other scripts. Uses romanized labels + hex (no Indic
    shaping) so the numeric alignment -- the actual mechanism -- is explicit."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(3.35, 2.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    # source scripts (top) : (name, KA codepoint, block base, colour, held_out)
    rows = [("Devanagari", 0x0915, 0x0900, GREY, False),
            ("Telugu",     0x0C15, 0x0C00, GREY, False),
            ("Bengali",    0x0995, 0x0980, GREY, False),
            ("Tamil",      0x0B95, 0x0B80, ORANGE, True)]   # held-out
    xs = [1.0, 3.3, 5.6, 7.9]
    box_w, box_h, ytop = 1.9, 1.15, 8.2
    for (name, cp, base, col, held), x in zip(rows, xs):
        ax.add_patch(FancyBboxPatch((x, ytop), box_w, box_h, mutation_scale=0.02,
                     boxstyle="round,pad=0.02", fc="white", ec=col, lw=1.6 if held else 1.0))
        ax.text(x + box_w / 2, ytop + box_h * 0.62, name, ha="center", va="center",
                fontsize=6.8, color=col, fontweight="bold" if held else "normal")
        ax.text(x + box_w / 2, ytop + box_h * 0.24, f"KA = U+{cp:04X}", ha="center",
                va="center", fontsize=6.2, fontfamily="monospace")
        if held:
            ax.text(x + box_w / 2, ytop + box_h + 0.35, "held-out\n(zero real images)",
                    ha="center", va="bottom", fontsize=5.6, color=ORANGE)

    # pivot box (bottom)
    pw, py = 3.0, 2.4
    px = 5 - pw / 2
    ax.add_patch(FancyBboxPatch((px, py), pw, 1.25, mutation_scale=0.02,
                 boxstyle="round,pad=0.02", fc="#EAF2FA", ec=BLUE, lw=1.8))
    ax.text(5, py + 0.85, "shared pivot", ha="center", fontsize=7.2, color=BLUE, fontweight="bold")
    ax.text(5, py + 0.42, "KA = U+0915", ha="center", fontsize=6.6, fontfamily="monospace")

    for (name, cp, base, col, held), x in zip(rows, xs):
        ax.add_patch(FancyArrowPatch((x + box_w / 2, ytop - 0.05), (5, py + 1.30),
                     arrowstyle="-|>", mutation_scale=8, lw=1.3 if held else 0.8,
                     color=col, alpha=0.95 if held else 0.55,
                     connectionstyle="arc3,rad=0.0"))
    ax.text(5, 5.5, r"pivot $=$ 0x0900 $+$ (cp $-$ block base)", ha="center",
            fontsize=6.4, color="#333333", style="italic")
    ax.text(5, 1.75, "held-out graphemes land on codes\nalready seen from source scripts",
            ha="center", va="top", fontsize=6.0, color="#444444")
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT, "fig1_pivot.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_pivot()
    fig_bars()
    fig_scaling()
    fig_receipts()
    print("wrote figs/fig1_pivot.pdf, fig2_bars.pdf, fig3_scaling.pdf, fig4_receipts.pdf")
