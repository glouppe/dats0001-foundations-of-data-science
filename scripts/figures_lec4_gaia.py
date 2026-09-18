"""Figures of the Gaia example of lecture 4: what a parallax is, what Gaia
measures, and what the posterior distance of a star looks like.

`data/gaia-dr3-sample.csv` is a random sample of 5000 stars of Gaia DR3, pulled
from the ESA archive with

    SELECT TOP 20000 source_id, parallax, parallax_error, phot_g_mean_mag
    FROM gaiadr3.gaia_source
    WHERE random_index < 4000000 AND parallax IS NOT NULL
      AND phot_g_mean_mag IS NOT NULL

Usage: uv run python scripts/figures_lec4_gaia.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc, Circle

GREY = "#354046"
LIGHT = "#b8c0c6"
BLUE = "#0173b2"
RED = "#c0392b"
L = 1.35          # length scale of the prior, in kpc

plt.rcParams.update({"font.size": 12, "text.color": GREY, "axes.labelcolor": GREY,
                     "xtick.color": GREY, "ytick.color": GREY, "axes.edgecolor": LIGHT,
                     "mathtext.fontset": "cm"})


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print("wrote", path)


def geometry():
    """What a parallax is: the angle 1 au subtends at the star. Not to scale."""
    fig, ax = plt.subplots(figsize=(5.8, 2.6), dpi=200)
    ax.set_axis_off()
    ax.set_aspect("equal")

    au, star = 1.25, (4.6, 0)
    ax.add_patch(Circle((0, 0), au, facecolor="none", edgecolor=LIGHT, ls=(0, (4, 4))))
    ax.plot(0, 0, "o", ms=9, color="#e8a33d", zorder=3)
    ax.text(0, -.28, "Sun", ha="center", va="top", fontsize=11)

    for y in (au, -au):
        ax.plot(0, y, "o", ms=6, color=BLUE, zorder=3)
        ax.plot([0, star[0]], [y, 0], color=GREY, lw=.9, zorder=1)
    ax.text(.2, au + .12, "Earth in June", fontsize=10, va="bottom")
    ax.text(.2, -au - .12, "Earth in December", fontsize=10, va="top")

    ax.annotate("", star, (0, 0), zorder=1,
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=.9, shrinkA=6, shrinkB=9))
    ax.text(star[0] * .3, .16, "$r$", ha="center", va="bottom", fontsize=13)
    ax.plot(*star, "*", ms=17, color=GREY, zorder=3)
    ax.text(star[0] + .4, 0, "star", ha="left", va="center", fontsize=11)

    ax.annotate("", (0, au), (0, 0), arrowprops=dict(arrowstyle="<->", color=GREY, lw=.9))
    ax.text(-.12, au / 2, "1 au", ha="right", va="center", fontsize=11)

    angle = np.degrees(np.arctan2(au, star[0]))
    ax.add_patch(Arc(star, 3.2, 3.2, theta1=180 - angle, theta2=180,
                     edgecolor=GREY, lw=.9))
    half = np.radians(angle / 2)
    ax.text(star[0] - 1.85 * np.cos(half), 1.85 * np.sin(half), r"$\varpi$",
            ha="center", va="center", fontsize=13)

    ax.set_xlim(-1.5, 5.6)
    ax.set_ylim(-2.1, 2.1)
    save(fig, "figures/lec4/parallax-geometry.svg")


def measurements(df):
    """What Gaia reports: a parallax and its uncertainty, for every star."""
    fig, ax = plt.subplots(figsize=(5.8, 3.8), dpi=200)
    negative = df.parallax < 0
    ax.scatter(df.phot_g_mean_mag[~negative], df.parallax[~negative],
               s=5, alpha=.3, color=BLUE, linewidths=0, label="measured $\\varpi > 0$")
    ax.scatter(df.phot_g_mean_mag[negative], df.parallax[negative],
               s=5, alpha=.5, color=RED, linewidths=0, label="measured $\\varpi < 0$")
    ax.axhline(0, color=GREY, lw=.8)
    ax.set_xlabel("$G$ magnitude (fainter to the right)")
    ax.set_ylabel(r"Parallax $\varpi$ (mas)")
    ax.set_ylim(-2.5, 4)
    ax.legend(markerscale=3, loc="upper left")
    fig.tight_layout()
    save(fig, "figures/lec4/gaia-parallaxes.png")


def pick(df, error, snr=None, negative=False):
    """A real star of the sample, close to a given precision."""
    d = df[df.parallax < 0] if negative else df[df.parallax > 0]
    d = d[(d.parallax_error - error).abs() < .05 * error]
    if snr is not None:
        d = d.iloc[((d.parallax / d.parallax_error) - snr).abs().argsort()]
    return d.iloc[0]


def posterior(r, varpi, sigma):
    """p(r | varpi, sigma) up to a constant, for the exponential prior."""
    return prior(r) * np.exp(-.5 * ((varpi - 1 / r) / sigma) ** 2)


def prior(r, length=L):
    """An exponentially decreasing space density, with length scale L."""
    return r ** 2 * np.exp(-r / length) / (2 * length ** 3)


def posteriors(df):
    """Three stars of the sample: the parallax speaks, or the prior does."""
    stars = [pick(df, error=.02, snr=40), pick(df, error=.3, snr=2),
             pick(df, error=.4, negative=True)]
    titles = ["a well measured parallax", "a noisy parallax",
              "a negative parallax, nothing to invert"]

    r = np.linspace(.02, 8, 2000)
    fig, axes = plt.subplots(3, 1, figsize=(5.8, 5.1), dpi=200, sharex=True)
    for ax, star, title in zip(axes, stars, titles):
        p = posterior(r, star.parallax, star.parallax_error)
        ax.plot(r, prior(r) / prior(r).max(), color=LIGHT, lw=1.4, label="prior")
        ax.fill_between(r, p / p.max(), color=BLUE, alpha=.25)
        ax.plot(r, p / p.max(), color=BLUE, lw=1.6, label="posterior")
        if star.parallax > 0:
            ax.axvline(1 / star.parallax, color=RED, lw=1.2, ls=(0, (4, 3)),
                       label=r"$1/\varpi$")
        ax.set_yticks([])
        ax.set_title(r"%s: $\varpi = %.2f \pm %.2f$ mas"
                     % (title, star.parallax, star.parallax_error),
                     loc="left", fontsize=11, pad=4)
    axes[0].legend(loc="upper right", fontsize=10, frameon=False, ncols=3)
    axes[1].set_ylabel("density")
    axes[-1].set_xlabel("Distance $r$ (kpc)")
    fig.tight_layout()
    save(fig, "figures/lec4/gaia-posteriors.svg")


def log_marginal(df, lengths, r=np.linspace(1e-3, 40, 3000)):
    """log p(varpi_1:N | sigma_1:N, L), the latent distances integrated out."""
    varpi = df.parallax.to_numpy()[:, None]
    sigma = df.parallax_error.to_numpy()[:, None]
    likelihood = np.exp(-.5 * ((varpi - 1 / r) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    out = []
    for length in lengths:
        marginal = np.trapezoid(likelihood * prior(r, length), r, axis=1)
        out.append(np.log(np.maximum(marginal, 1e-300)).sum())
    return np.array(out)


def length_scale(df):
    """The upper level: what the 5000 stars together say about the Galaxy."""
    lengths = np.linspace(.4, 2.6, 60)
    curve = log_marginal(df, lengths)
    best = lengths[curve.argmax()]

    fig, ax = plt.subplots(figsize=(5.8, 3.1), dpi=200)
    ax.plot(lengths, curve - curve.max(), color=BLUE, lw=1.8)
    ax.axvline(best, color=RED, lw=1.2, ls=(0, (4, 3)))
    ax.text(best + .06, -180, r"$\hat{L} = %.2f$ kpc" % best, color=RED, fontsize=11)
    ax.set_xlabel("Length scale $L$ (kpc)")
    ax.set_ylabel(r"$\log p(\varpi_{1:N} \mid \sigma_{1:N}, L)$, relative")
    ax.set_ylim(-1200, 60)
    fig.tight_layout()
    save(fig, "figures/lec4/gaia-length-scale.svg")
    print("  maximum marginal likelihood at L = %.2f kpc" % best)


if __name__ == "__main__":
    df = pd.read_csv("data/gaia-dr3-sample.csv")
    snr = df.parallax / df.parallax_error
    print("%d stars, %.0f%% negative, %.0f%% below a signal-to-noise ratio of 5"
          % (len(df), 100 * (df.parallax < 0).mean(), 100 * (snr < 5).mean()))
    geometry()
    measurements(df)
    posteriors(df)
    length_scale(df)
