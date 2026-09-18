"""The prior predictive check of lecture 4: colonies of penguins simulated from
the prior, against the one actually measured.

A pooled histogram of many draws hides a prior that is far too tight on sigma,
since its spread comes from mu. Simulating whole datasets shows it at once.

Usage: uv run python scripts/figures_lec4_prior.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GREY = "#354046"
LIGHT = "#b8c0c6"
BLUE = "#0173b2"
N_COLONIES = 5

plt.rcParams.update({"font.size": 12, "text.color": GREY, "axes.labelcolor": GREY,
                     "xtick.color": GREY, "ytick.color": GREY, "axes.edgecolor": LIGHT,
                     "mathtext.fontset": "cm"})


def colonies(rng, mass, scale2):
    """Simulate N_COLONIES datasets of the observed size, from the prior."""
    mu = rng.normal(*mass, size=N_COLONIES)
    sigma2 = rng.uniform(0, scale2, size=N_COLONIES)
    return rng.normal(mu[:, None], np.sqrt(sigma2)[:, None], size=(N_COLONIES, N))


def strip(ax, sample, y, color, size=4):
    ax.plot(sample, y + .18 * (np.random.rand(len(sample)) - .5), "o",
            ms=size, alpha=.25, color=color, markeredgewidth=0)


if __name__ == "__main__":
    observed = pd.read_csv("data/penguins.csv")["body_mass_g"].dropna().to_numpy()
    N = len(observed)
    rng = np.random.default_rng(1)
    np.random.seed(1)

    priors = [(r"$\mathcal{N}(\mu \mid 5000, 2000^2)$, $\mathrm{Uniform}(\sigma^2 \mid 0, 100)$",
               (5000, 2000), 100),
              (r"$\mathcal{N}(\mu \mid 4000, 1000^2)$, $\mathrm{Uniform}(\sigma^2 \mid 0, 1500^2)$",
               (4000, 1000), 1500 ** 2)]

    fig, axes = plt.subplots(2, 1, figsize=(5.8, 4.4), dpi=200, sharex=True)
    for ax, (title, mass, scale2) in zip(axes, priors):
        for k, sample in enumerate(colonies(rng, mass, scale2)):
            strip(ax, sample, k + 1, BLUE)
        strip(ax, observed, 0, GREY)
        ax.set_title(title, loc="left", fontsize=11, pad=4)
        ax.set_yticks(range(N_COLONIES + 1))
        ax.set_yticklabels(["measured"] + ["simulated"] * N_COLONIES, fontsize=9)
        ax.set_ylim(-.6, N_COLONIES + .6)
        ax.set_xlim(-1000, 11000)
    axes[-1].set_xlabel("Body mass (g)")
    fig.tight_layout()
    fig.savefig("figures/lec4/prior-predictive-check.png", bbox_inches="tight",
                facecolor="white")
    print("wrote figures/lec4/prior-predictive-check.png")
