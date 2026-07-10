# Renders fig_zeroshot.png — zero-shot LOSO results, all nine scripts (system python3).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, SEC, MUT, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
A_C, B_C = "#86b6ef", "#2a78d6"   # ordinal blue pair (validated)

scripts = ["Tamil", "Telugu", "Kannada", "Malayalam", "Oriya", "Gujarati",
           "Bengali\u2020", "Devanagari\u2020", "Gurmukhi"]
rungA = [0.0, 1.28, 2.78, 0.18, 0.77, 3.05, 1.39, 5.46, 0.59]
rungB = [9.16, 19.82, 15.42, 9.69, 25.38, 15.86, 27.08, 30.09, 22.16]
sup   = [46.39, 33.76, 30.83, 26.69, 52.39, 33.60, 31.72, 47.89, 57.83]
ntest = [513, 545, 720, 547, 1044, 1015, 2873, 6042, 2879]

fig, ax = plt.subplots(figsize=(12.8, 4.6), dpi=300)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
bw = 0.32
xs = range(len(scripts))

for i in xs:
    xa, xb = i - bw / 2 - 0.02, i + bw / 2 + 0.02
    ax.bar(xa, rungA[i], bw, color=A_C, zorder=3)
    ax.annotate(f"{rungA[i]:.1f}", (xa, rungA[i] + 0.8), ha="center",
                fontsize=7.5, color=SEC, zorder=4)
    ax.bar(xb, rungB[i], bw, color=B_C, zorder=3)
    ax.annotate(f"{rungB[i]:.1f}", (xb, rungB[i] + 0.8), ha="center",
                fontsize=8.5, fontweight="bold", color=INK, zorder=4)
    ax.hlines(sup[i], i - 0.35, i + 0.35, colors=MUT, linestyles=(0, (4, 3)),
              linewidth=1.3, zorder=3)
    ax.annotate(f"{sup[i]:.0f}", (i + 0.38, sup[i]), ha="left", va="center",
                fontsize=7, color=MUT, zorder=4)

ax.set_xticks(list(xs))
ax.set_xticklabels([f"{s}\n({n})" for s, n in zip(scripts, ntest)],
                   fontsize=8.8, color=INK)
ax.set_ylabel("word recognition rate (%)", fontsize=10, color=SEC)
ax.set_ylim(0, 62)
ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(BASE)
ax.tick_params(colors=MUT, labelcolor=INK)

import matplotlib.patches as mpatches
import matplotlib.lines as mlines
ax.legend(handles=[
    mpatches.Patch(color=A_C, label="Rung A — held-out script never seen at all"),
    mpatches.Patch(color=B_C, label="Rung B — + synthetic renders only (ZERO real images)"),
    mlines.Line2D([], [], color=MUT, linestyle=(0, (4, 3)),
                  label="supervised fusion on 1,620 real images (reference)"),
], loc="upper left", fontsize=8.5, frameon=False)
ax.set_title("Reading each held-out script with zero real training images — all nine scripts complete",
             fontsize=12, color=INK, loc="left", pad=12)
ax.annotate("Base VLM (Florence-2) raw zero-shot = 0.0 on every script.  (n) = real test photos.  "
            "\u2020 trained with 2\u00d7 synthetic exposure (6,480 renders; two BSTD languages — disclosed in advance).",
            (0.0, -0.26), xycoords="axes fraction", fontsize=8, color=MUT)
ax.annotate("Gurmukhi was called in advance: 16.2 (band 8.2\u201324.1) committed to git 3\u00bd hours before its training began; realized 22.16.",
            (0.0, -0.33), xycoords="axes fraction", fontsize=8, color=MUT)

fig.tight_layout()
fig.savefig("/c/ujjwalb/ritu1/lstm_model/prof_abstract/fig_zeroshot.png",
            bbox_inches="tight", facecolor="white")
print("fig_zeroshot.png done")
