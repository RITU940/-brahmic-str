# Renders fig_zeroshot.png — zero-shot LOSO results so far (system python3).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, SEC, MUT, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
A_C, B_C = "#86b6ef", "#2a78d6"   # ordinal blue pair (validated)

scripts = ["Tamil", "Telugu", "Kannada"]
rungA = [0.0, 1.28, 2.78]
rungB = [9.16, 19.82, None]           # kannada B in training
sup = [46.39, 33.76, 30.83]           # supervised fusion, 1,620 real target imgs
ntest = [513, 545, 720]

fig, ax = plt.subplots(figsize=(8.4, 4.3), dpi=300)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
bw = 0.30
xs = range(len(scripts))

for i in xs:
    xa, xb = i - bw / 2 - 0.02, i + bw / 2 + 0.02
    ax.bar(xa, rungA[i], bw, color=A_C, zorder=3)
    ax.annotate(f"{rungA[i]:.1f}", (xa, rungA[i] + 0.8), ha="center",
                fontsize=9, color=SEC, zorder=4)
    if rungB[i] is not None:
        ax.bar(xb, rungB[i], bw, color=B_C, zorder=3)
        ax.annotate(f"{rungB[i]:.1f}", (xb, rungB[i] + 0.8), ha="center",
                    fontsize=10, fontweight="bold", color=INK, zorder=4)
    else:
        ax.annotate("in\ntraining", (xb, 1.0), ha="center", va="bottom",
                    fontsize=8.5, color=MUT, style="italic", zorder=4)
    # supervised reference segment
    ax.hlines(sup[i], i - 0.33, i + 0.33, colors=MUT, linestyles=(0, (4, 3)),
              linewidth=1.4, zorder=3)
    ax.annotate(f"supervised {sup[i]:.0f}", (i, sup[i] + 1.0), ha="center",
                fontsize=8, color=MUT, zorder=4)

ax.set_xticks(list(xs))
ax.set_xticklabels([f"{s}\n({n} real test photos)" for s, n in zip(scripts, ntest)],
                   fontsize=10, color=INK)
ax.set_ylabel("word recognition rate (%)", fontsize=10, color=SEC)
ax.set_ylim(0, 52)
ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(BASE)
ax.tick_params(colors=MUT, labelcolor=INK)

import matplotlib.patches as mpatches
import matplotlib.lines as mlines
leg = ax.legend(handles=[
    mpatches.Patch(color=A_C, label="Rung A — held-out script never seen at all"),
    mpatches.Patch(color=B_C, label="Rung B — + synthetic renders only (ZERO real images)"),
    mlines.Line2D([], [], color=MUT, linestyle=(0, (4, 3)),
                  label="supervised fusion on 1,620 real images (upper reference)"),
], loc="upper right", fontsize=8.5, frameon=False)
ax.set_title("Reading a held-out script with zero real training images",
             fontsize=12, color=INK, loc="left", pad=12)
ax.annotate("Base VLM (Florence-2) raw zero-shot = 0.0 on every Brahmic script.",
            (0.0, -0.24), xycoords="axes fraction", fontsize=8.5, color=MUT)

fig.tight_layout()
fig.savefig("/c/ujjwalb/ritu1/lstm_model/prof_abstract/fig_zeroshot.png",
            bbox_inches="tight", facecolor="white")
print("fig_zeroshot.png done")
