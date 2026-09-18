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
L_MAX = 5.0       # the prior on the length scale is flat on (0, L_MAX], in kpc

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


def prior(r, length):
    """An exponentially decreasing space density, with length scale L."""
    return r ** 2 * np.exp(-r / length) / (2 * length ** 3)


def likelihood(r, varpi, sigma):
    """p(varpi | r, sigma), the measurement model, as a function of r."""
    return np.exp(-.5 * ((varpi - 1 / r) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def posterior_length(df, lengths, r=np.geomspace(1e-2, 40, 2000)):
    """p(L | varpi_1:N, sigma_1:N), up to a constant, under a flat prior on L."""
    varpi = df.parallax.to_numpy()[:, None]
    sigma = df.parallax_error.to_numpy()[:, None]
    like = likelihood(r, varpi, sigma)
    log = np.array([np.log(np.maximum(np.trapezoid(like * prior(r, length), r, axis=1),
                                      1e-300)).sum() for length in lengths])
    density = np.exp(log - log.max())
    return density / np.trapezoid(density, lengths)


def summarize(lengths, density):
    """The mode and the standard deviation of a posterior on a grid."""
    mean = np.trapezoid(lengths * density, lengths)
    var = np.trapezoid((lengths - mean) ** 2 * density, lengths)
    return lengths[density.argmax()], var ** .5


def posteriors(df, lengths, density):
    """Three stars of the sample: the parallax speaks, or the Galaxy does."""
    stars = [pick(df, error=.02, snr=40), pick(df, error=.3, snr=2),
             pick(df, error=.4, negative=True)]
    titles = ["a well measured parallax", "a noisy parallax",
              "a negative parallax, nothing to invert"]

    r = np.linspace(.02, 8, 2000)
    # p(r_i | everything) integrates the Galaxy model over the posterior of L
    population = np.trapezoid(prior(r[None, :], lengths[:, None]) * density[:, None],
                              lengths, axis=0)
    fig, axes = plt.subplots(3, 1, figsize=(5.8, 5.1), dpi=200, sharex=True)
    for ax, star, title in zip(axes, stars, titles):
        p = likelihood(r, star.parallax, star.parallax_error) * population
        ax.plot(r, population / population.max(), color=LIGHT, lw=1.4,
                label="before the parallax")
        ax.fill_between(r, p / p.max(), color=BLUE, alpha=.25)
        ax.plot(r, p / p.max(), color=BLUE, lw=1.6, label="after the parallax")
        if star.parallax > 0:
            ax.axvline(1 / star.parallax, color=RED, lw=1.2, ls=(0, (4, 3)),
                       label=r"$1/\varpi$")
        ax.set_yticks([])
        ax.set_title(r"%s: $\varpi = %.2f \pm %.2f$ mas"
                     % (title, star.parallax, star.parallax_error),
                     loc="left", fontsize=11, pad=4)
    axes[1].legend(loc="upper right", fontsize=10, frameon=False)
    axes[1].set_ylabel("density")
    axes[-1].set_xlabel("Distance $r$ (kpc)")
    fig.tight_layout()
    save(fig, "figures/lec4/gaia-posteriors.svg")


def length_scale(lengths, density):
    """What the 5000 stars together say about the length scale of the Galaxy."""
    mode, sd = summarize(lengths, density)
    fig, ax = plt.subplots(figsize=(5.8, 3.1), dpi=200)
    ax.plot(lengths, density, color=BLUE, lw=1.8)
    ax.fill_between(lengths, density, color=BLUE, alpha=.2)
    ax.axvline(mode, color=RED, lw=1.2, ls=(0, (4, 3)))
    ax.annotate(r"$%.2f \pm %.3f$ kpc" % (mode, sd), (mode, density.max()),
                (mode + .18, density.max() * .82), color=RED, fontsize=11,
                arrowprops=dict(arrowstyle="-", color=RED, lw=.8))
    ax.set_xlabel("Length scale $L$ (kpc)")
    ax.set_ylabel(r"$p(L \mid \varpi_{1:N}, \sigma_{1:N})$")
    ax.set_yticks([])
    ax.set_xlim(lengths[0], lengths[-1])
    fig.tight_layout()
    save(fig, "figures/lec4/gaia-length-scale.svg")
    print("  L = %.3f +- %.3f kpc, from a prior flat on (0, %.0f]" % (mode, sd, L_MAX))


if __name__ == "__main__":
    df = pd.read_csv("data/gaia-dr3-sample.csv")
    snr = df.parallax / df.parallax_error
    print("%d stars, %.0f%% negative, %.0f%% below a signal-to-noise ratio of 5"
          % (len(df), 100 * (df.parallax < 0).mean(), 100 * (snr < 5).mean()))
    geometry()
    measurements(df)
    lengths = np.unique(np.concatenate([np.linspace(.4, 2.6, 90),
                                        np.linspace(.90, 1.15, 260)]))
    density = posterior_length(df, lengths)
    posteriors(df, lengths, density)
    length_scale(lengths, density)
