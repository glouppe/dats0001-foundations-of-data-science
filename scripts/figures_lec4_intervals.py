"""The confidence interval of lecture 4: how it is built from the likelihood
ratio of the penguin data, how the distribution of that ratio can be simulated
instead of derived, and what the interval promises, on colonies of penguins
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


def simulated_statistic(simulations=10000, candidates=49, per_candidate=2000):
    """The interval by brute force, in two figures: the simulated distribution of
    lambda at one candidate mu and its 95% quantile c(mu); then c(mu) at every
    candidate, against the observed lambda(mu; x_obs), which it cuts into the interval."""
    x = pd.read_csv("data/penguins.csv")["body_mass_g"].dropna().to_numpy()
    n, mu_hat, sigma = len(x), x.mean(), x.std()
    rng = np.random.default_rng(0)

    def lam_sim(mu, m):
        return n * (rng.normal(mu, sigma, size=(m, n)).mean(axis=1) - mu) ** 2 / sigma ** 2

    def tidy(ax):
        for side in ("right", "top"):
            ax.spines[side].set_visible(False)

    # step 1: the simulated distribution of lambda at one candidate, and its quantile
    lam = lam_sim(mu_hat, simulations)
    c = np.quantile(lam, .95)
    fig, ax = plt.subplots(figsize=(5.8, 2.6), dpi=200)
    bins = np.linspace(0, 10, 51)
    w = 1 / (simulations * (bins[1] - bins[0]))
    ax.hist(lam[lam <= c], bins=bins, weights=np.full((lam <= c).sum(), w), color=BLUE,
            alpha=.35, label="%d simulated data sets" % simulations)
    ax.hist(lam[lam > c], bins=bins, weights=np.full((lam > c).sum(), w), color=RED, alpha=.5)
    grid = np.linspace(.02, 10, 400)
    ax.plot(grid, chi2.pdf(grid, df=1), color=GREY, lw=1.2, label=r"$\chi^2_1$")
    ax.axvline(c, color=RED, lw=1.2, ls=(0, (4, 3)))
    ax.text(c + .2, .9, "$c(\\hat{\\mu}) = %.2f$" % c, color=RED, fontsize=10)
    ax.text(c + 1.2, .14, "5%", color=RED, fontsize=10)
    ax.set_xlabel(r"$\lambda(\hat{\mu}; \mathbf{x})$, $\mathbf{x} \sim p(\mathbf{x} \mid \hat{\mu})$")
    ax.set_ylabel("density")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.2)
    ax.legend(frameon=False, loc="center right")
    tidy(ax)
    fig.tight_layout()
    fig.savefig("figures/lec4/simulated-statistic.svg", bbox_inches="tight", facecolor="white")

    # step 2: c(mu) at every candidate, against the observed lambda(mu; x_obs)
    mus = np.linspace(4060, 4345, candidates)
    cs = np.array([np.quantile(lam_sim(m, per_candidate), .95) for m in mus])
    gap = n * (mu_hat - mus) ** 2 / sigma ** 2 - cs        # negative inside the interval
    cross = [np.interp(0, sorted([gap[k], gap[k + 1]]),
                       [mus[k], mus[k + 1]] if gap[k] < gap[k + 1] else [mus[k + 1], mus[k]])
             for k in range(len(mus) - 1) if np.sign(gap[k]) != np.sign(gap[k + 1])]
    lo, hi = min(cross), max(cross)
    fine = np.linspace(4040, 4365, 400)
    fig, ax = plt.subplots(figsize=(5.8, 2.6), dpi=200)
    ax.plot(fine, n * (mu_hat - fine) ** 2 / sigma ** 2, color=BLUE, lw=1.6,
            label=r"$\lambda(\mu; \mathbf{x}_\mathrm{obs})$")
    ax.plot(mus, cs, "o", ms=3, color=RED, label=r"$c(\mu)$, simulated at each $\mu$")
    ax.plot(mu_hat, c, "o", ms=9, mfc="none", mec=RED, mew=1.4, zorder=4)
    ax.annotate(r"$c(\hat{\mu})$", (mu_hat, c), (mu_hat - 70, 7), color=RED, fontsize=10,
                ha="center", arrowprops=dict(arrowstyle="->", color=RED, lw=.9, shrinkB=6))
    ax.axvspan(lo, hi, color=BLUE, alpha=.12, lw=0)
    ax.set_xticks([lo, mu_hat, hi], ["%.0f" % lo, "%.0f" % mu_hat, "%.0f" % hi])
    ax.set_xlabel(r"$\mu$ (g)")
    ax.set_xlim(fine[0], fine[-1])
    ax.set_ylim(0, 12)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, 1), ncols=2, fontsize=9)
    tidy(ax)
    fig.tight_layout()
    fig.savefig("figures/lec4/simulated-interval.svg", bbox_inches="tight", facecolor="white")
    print("wrote figures/lec4/simulated-statistic.svg, simulated-interval.svg: "
          "c = %.3f, C(x_obs) = [%.0f, %.0f]" % (c, lo, hi))


def coverage():
    rng = np.random.default_rng(0)
    x = rng.normal(MU, SIGMA, size=(COLONIES, N))
    mean = x.mean(axis=1)
    half = 1.96 * x.std(axis=1) / np.sqrt(N)          # sigma-hat, the MLE
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
    simulated_statistic()
    coverage()
