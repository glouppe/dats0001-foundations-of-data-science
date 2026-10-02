"""What a 95% confidence interval promises, in lecture 4: colonies of penguins
simulated from the model at a known mu*, and the interval each colony gives.

The parameters are round numbers close to the penguin estimates, so that the
true value is known, which it never is for real data.

Usage: uv run python scripts/figures_lec4_intervals.py
"""

import matplotlib.pyplot as plt
import numpy as np

GREY = "#354046"
LIGHT = "#b8c0c6"
BLUE = "#0173b2"
RED = "#c0392b"
MU, SIGMA, N = 4200, 800, 342      # mu* and sigma* of the simulation, colony size
COLONIES = 50

plt.rcParams.update({"font.size": 12, "text.color": GREY, "axes.labelcolor": GREY,
                     "xtick.color": GREY, "ytick.color": GREY, "axes.edgecolor": LIGHT,
                     "mathtext.fontset": "cm"})


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(MU, SIGMA, size=(COLONIES, N))
    mean = x.mean(axis=1)
    half = 1.96 * x.std(axis=1, ddof=1) / np.sqrt(N)
    miss = np.abs(mean - MU) > half

    fig, ax = plt.subplots(figsize=(5.8, 3.4), dpi=200)
    k = np.arange(COLONIES)
    for color, sel in ((BLUE, ~miss), (RED, miss)):
        ax.errorbar(k[sel], mean[sel], yerr=half[sel], fmt="o", ms=2.5, lw=1,
                    color=color, capsize=0)
    ax.plot([-1, COLONIES], [MU, MU], color=GREY, lw=1, ls=(0, (4, 3)))
    ax.text(COLONIES + .5, MU, r"$\mu^* = %d$ g" % MU, ha="left", va="center",
            fontsize=11)
    ax.set_xlabel("Simulated colony of %d penguins" % N)
    ax.set_ylabel("Mean body mass (g)")
    ax.set_xticks([])
    ax.set_xlim(-1, COLONIES + 8)
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig("figures/lec4/confidence-intervals.svg", bbox_inches="tight",
                facecolor="white")
    print("wrote figures/lec4/confidence-intervals.svg: %d of %d intervals miss mu*"
          % (miss.sum(), COLONIES))
