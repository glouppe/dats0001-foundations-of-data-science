"""The confidence interval of lecture 4: how it is built from the likelihood
ratio of the penguin data, and what it promises, on colonies of penguins
simulated from the model at a known mu*.

The parameters are round numbers close to the penguin estimates, so that the
true value is known, which it never is for real data.

Usage: uv run python scripts/figures_lec4_intervals.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

GREY = "#354046"
LIGHT = "#b8c0c6"
BLUE = "#0173b2"
RED = "#c0392b"
MU, SIGMA, N = 4200, 800, 342      # mu* and sigma* of the simulation, colony size
COLONIES = 50

plt.rcParams.update({"font.size": 12, "text.color": GREY, "axes.labelcolor": GREY,
                     "xtick.color": GREY, "ytick.color": GREY, "axes.edgecolor": LIGHT,
                     "mathtext.fontset": "cm"})


def likelihood_ratio():
    """lambda(mu) for the penguins, sigma fixed to its estimate, and the set it cuts."""
    x = pd.read_csv("data/penguins.csv")["body_mass_g"].dropna().to_numpy()
    n, mu_hat, sigma_hat = len(x), x.mean(), x.std()
    c = chi2.ppf(.95, df=1)
    mu = np.linspace(4040, 4360, 400)
    lam = n * (mu - mu_hat) ** 2 / sigma_hat ** 2
    lo, hi = mu_hat + np.array([-1, 1]) * np.sqrt(c) * sigma_hat / np.sqrt(n)

    fig, ax = plt.subplots(figsize=(5.8, 3.0), dpi=200)
    ax.plot(mu, lam, color=BLUE, lw=1.8)
    ax.axhline(c, color=RED, lw=1.2, ls=(0, (4, 3)))
    ax.text(mu_hat, c + .25, "$c = %.2f$" % c, color=RED, ha="center", va="bottom",
            fontsize=11)
    ax.fill_between([lo, hi], 0, c, color=BLUE, alpha=.15, linewidth=0)
    ax.plot([lo, hi], [0, 0], color=BLUE, lw=4, solid_capstyle="butt")
    for v in (lo, hi):
        ax.plot([v, v], [0, c], color=BLUE, lw=.8)
    ax.plot(mu_hat, 0, "o", ms=5, color=GREY, zorder=3)
    ax.text(mu_hat, .35, r"$\hat{\mu}$", ha="center", va="bottom", fontsize=12)
    ax.set_xticks([lo, mu_hat, hi], ["%.0f" % lo, "%.0f" % mu_hat, "%.0f" % hi])
    ax.set_xlabel(r"$\mu$ (g)")
    ax.set_ylabel(r"$\lambda(\mu; \mathbf{x}_\mathrm{obs})$")
    ax.set_xlim(mu[0], mu[-1])
    ax.set_ylim(0, 12)
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig("figures/lec4/likelihood-ratio.svg", bbox_inches="tight",
                facecolor="white")
    print("wrote figures/lec4/likelihood-ratio.svg: [%.0f, %.0f] g" % (lo, hi))


def coverage():
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


if __name__ == "__main__":
    likelihood_ratio()
    coverage()
