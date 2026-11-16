"""Figures of lecture 8: three graphical models, in the style and at the scale of lectures 4
to 7, and the plots of the lecture. The running example is the Gaussian mixture of the penguin
bills (lecture 4, Example 2), fitted by EM with full covariance matrices; the ELBO pictures are
computed exactly on a one-parameter mixture of the flipper lengths. Every plot is W inches wide,
shown at .width-70, as in lecture 7.

Usage: uv run python scripts/figures_lec8.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
from scipy.stats import multivariate_normal, norm  # noqa: E402

from figures_lec4 import (GAP, GREY, MARGIN, R, R_SQ, STEP, UNIT, arrow, box,  # noqa: E402
                          figure, hyper, measure, node, plate, rescale)
from matplotlib.patches import Rectangle  # noqa: E402

WIDTH = 246.5568 / 72 / UNIT    # canvas width of the lecture 4 figures, in drawing units
LIGHT = "#b8c0c6"
BLUE = "#0173b2"
ORANGE = "#de8f05"
GREEN = "#029e73"
COLORS = np.array([plt.matplotlib.colors.to_rgb(c) for c in (BLUE, ORANGE, GREEN)])
W = 6.4

plt.rcParams.update({"font.size": 12, "legend.fontsize": 10, "axes.titlesize": 12,
                     "text.color": GREY, "axes.labelcolor": GREY,
                     "xtick.color": GREY, "ytick.color": GREY, "axes.edgecolor": LIGHT,
                     "mathtext.fontset": "cm"})


def tidy(ax):
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)


def save(fig, name):
    path = "figures/lec8/%s" % name
    fig.savefig(path, facecolor="white", dpi=200)
    plt.close(fig)
    print("wrote", path)


# Graphical models ----------------------------------------------------------------------

def square(ax, xy, label):
    """A fixed, non-random quantity: a small filled square with its label on the right."""
    ax.add_patch(Rectangle((xy[0] - R_SQ, xy[1] - R_SQ), 2 * R_SQ, 2 * R_SQ,
                           facecolor=GREY, edgecolor=GREY, lw=.7, zorder=2))
    ax.text(xy[0] + .22, xy[1], label, va="center", ha="left", fontsize=17)


def lvm():
    """A latent z and an observation x, both depending on the parameters theta."""
    fig, ax = figure()
    node(ax, (0, STEP), r"$\mathbf{z}$")
    node(ax, (0, 0), r"$\mathbf{x}$", observed=True)
    arrow(ax, (0, STEP), (0, 0))
    theta = (R + GAP + 3 * R_SQ, STEP / 2)
    square(ax, theta, r"$\theta$")
    arrow(ax, theta, (0, STEP), r_start=R_SQ)
    arrow(ax, theta, (0, 0), r_start=R_SQ)
    return fig, ax, "figures/lec8/lvm.svg"


def gmm():
    """The Gaussian mixture fitted by maximum likelihood: its parameters are fixed squares."""
    fig, ax = figure()
    observations = plate(ax, [box((0, STEP)), box((0, 0))], "$N$")
    node(ax, (0, STEP), r"$z_i$")
    node(ax, (0, 0), r"$\mathbf{x}_i$", observed=True)
    arrow(ax, (0, STEP), (0, 0))
    hyper(ax, (0, STEP), r"$\boldsymbol{\pi}$", observations[2])
    hyper(ax, (0, 0), r"$\boldsymbol{\mu}_{1:K}, \mathbf{\Sigma}_{1:K}$", observations[2])
    return fig, ax, "figures/lec8/gmm-model.svg"


def hierarchical():
    """Parameters theta with a prior of hyperparameter alpha, shared by N observations."""
    fig, ax = figure()
    observations = plate(ax, [box((0, STEP)), box((0, 0))], "$N$")
    node(ax, (0, STEP), r"$\mathbf{z}_i$")
    node(ax, (0, 0), r"$\mathbf{x}_i$", observed=True)
    arrow(ax, (0, STEP), (0, 0))
    theta = (0, observations[3] + GAP + R)
    node(ax, theta, r"$\theta$")
    arrow(ax, theta, (0, STEP))
    hyper(ax, theta, r"$\alpha$", R)
    return fig, ax, "figures/lec8/hierarchical-lvm.svg"


def graphical_models():
    for fig, ax, path in [lvm(), gmm(), hierarchical()]:
        x0, y0, x1, y1 = measure(fig, ax)
        width = max(WIDTH, x1 - x0 + 2 * MARGIN)
        centre = (x0 + x1) / 2
        rescale(fig, ax, centre - width / 2, centre + width / 2, y0 - MARGIN, y1 + MARGIN)
        fig.savefig(path, facecolor="white")
        plt.close(fig)
        print("wrote %-30s %.2f x %.2f in" % (path, *fig.get_size_inches()))


# EM on the penguin bills ---------------------------------------------------------------

def bills():
    d = pd.read_csv("data/penguins.csv").dropna(subset=["bill_length_mm", "bill_depth_mm"])
    return d[["bill_length_mm", "bill_depth_mm"]].to_numpy()


def densities(x, pi, mu, cov):
    """pi_k N(x_i | mu_k, Sigma_k), as an N x K array."""
    return np.stack([pi[k] * multivariate_normal(mu[k], cov[k]).pdf(x)
                     for k in range(len(pi))], 1)


def em(x, K=3, seed=0, steps=100):
    """EM for a Gaussian mixture with full covariances. Returns the parameters and the
    responsibilities at every iteration, and the log-likelihoods."""
    rng = np.random.default_rng(seed)
    N = len(x)
    mu = x[rng.choice(N, K, replace=False)]
    cov = np.array([np.cov(x.T)] * K) / 4
    pi = np.ones(K) / K
    history = []
    for t in range(steps + 1):
        p = densities(x, pi, mu, cov)
        r = p / p.sum(1, keepdims=True)                          # E-step
        history.append((pi, mu, cov, r, np.log(p.sum(1)).sum()))
        Nk = r.sum(0)                                            # M-step
        mu = (r.T @ x) / Nk[:, None]
        cov = np.array([((r[:, k, None] * (x - mu[k])).T @ (x - mu[k])) / Nk[k]
                        for k in range(K)])
        pi = Nk / N
    return history


def ellipse(ax, mean, cov, color):
    """The contour at two standard deviations of a 2D Gaussian."""
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    for lw, fill in [(0, True), (1.4, False)]:
        ax.add_patch(Ellipse(mean, 4 * np.sqrt(vals[1]), 4 * np.sqrt(vals[0]), angle=angle,
                             facecolor=color if fill else "none", alpha=.12 if fill else 1,
                             edgecolor=color, lw=lw, zorder=1))


def bills_axes(ax):
    ax.set_xlabel("bill length (mm)")
    ax.set_ylabel("bill depth (mm)")
    ax.set_xlim(31, 61)
    ax.set_ylim(12.5, 22)
    tidy(ax)


def bills_figure(x):
    fig, ax = plt.subplots(figsize=(W, 3.6), dpi=200)
    ax.scatter(*x.T, s=12, color=GREY, alpha=.7, lw=0)
    bills_axes(ax)
    fig.tight_layout()
    save(fig, "penguin-bills.png")


def state(ax, x, step, title=True):
    pi, mu, cov, r, ll = step
    ax.scatter(*x.T, s=12, c=np.clip(r @ COLORS, 0, 1), alpha=.8, lw=0, zorder=2)
    for k in range(len(pi)):
        ellipse(ax, mu[k], cov[k], COLORS[k])
    bills_axes(ax)


def em_figures(x):
    history = em(x, seed=0)
    for j, t in enumerate([0, 1, 3, 10, 30, 100]):
        fig, ax = plt.subplots(figsize=(W, 3.6), dpi=200)
        state(ax, x, history[t])
        ax.set_title("iteration %d, log-likelihood %.0f" % (t, history[t][4]))
        fig.tight_layout()
        save(fig, "em-bills-%d.png" % j)
    print("  log-likelihood: %s" % ", ".join("%.1f" % h[4] for h in history[:5]),
          "...", "%.1f" % history[-1][4])

    local = em(x, seed=2)
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.0), dpi=200, sharey=True)
    for ax, h in zip(axes, [history[-1], local[-1]]):
        state(ax, x, h)
        ax.set_title("log-likelihood %.0f" % h[4])
    axes[1].set_ylabel("")
    fig.tight_layout()
    save(fig, "em-local.png")
    print("  local maximum: %.1f against %.1f" % (local[-1][4], history[-1][4]))


# The ELBO, computed exactly on a one-parameter mixture --------------------------------

def flippers():
    d = pd.read_csv("data/penguins.csv").dropna(subset=["flipper_length_mm"])
    return d["flipper_length_mm"].to_numpy()


FIXED = dict(pi=np.array([.6, .4]), mu2=217., sigma=7.)


def joint(x, theta):
    """log pi_k N(x_i | mu_k, sigma), N x K, with mu_1 = theta the one free parameter."""
    mus = np.array([theta, FIXED["mu2"]])
    return np.log(FIXED["pi"]) + norm.logpdf(x[:, None], mus, FIXED["sigma"])


def loglik(x, theta):
    j = joint(x, theta)
    m = j.max(1, keepdims=True)
    return (m[:, 0] + np.log(np.exp(j - m).sum(1))).sum()


def posterior(x, theta):
    j = joint(x, theta)
    j -= j.max(1, keepdims=True)
    q = np.exp(j)
    return q / q.sum(1, keepdims=True)


def elbo(x, q, theta):
    return (q * (joint(x, theta) - np.log(np.clip(q, 1e-300, None)))).sum()


def elbo_figures(x):
    thetas = np.linspace(166, 214, 400)
    ll = np.array([loglik(x, t) for t in thetas])
    lo, hi = ll.max() - 750, ll.max() + 90

    def panel(name, theta_t, q, label=r"$\theta^{(t)}$", arrow_to=None, gap=False):
        fig, ax = plt.subplots(figsize=(W, 3.2), dpi=200)
        bound = np.array([elbo(x, q, t) for t in thetas])
        ax.plot(thetas, ll, color=BLUE, lw=1.8, label=r"$\log p(\mathbf{x} \mid \theta)$")
        ax.plot(thetas, bound, color=ORANGE, lw=1.8, label=r"$\mathcal{L}(q, \theta)$")
        ax.axvline(theta_t, color=GREY, lw=.8, ls=(0, (4, 3)))
        ax.plot([theta_t], [elbo(x, q, theta_t)], "o", color=ORANGE, ms=5, zorder=4)
        ax.plot([theta_t], [loglik(x, theta_t)], "o", color=BLUE, ms=5, zorder=4)
        ticks, labels = [theta_t], [label]
        if gap:
            ax.annotate("", (theta_t - .8, elbo(x, q, theta_t)), (theta_t - .8, loglik(x, theta_t)),
                        arrowprops=dict(arrowstyle="<|-|>", color=GREY, lw=.9, mutation_scale=8))
            ax.text(theta_t - 1.6, (elbo(x, q, theta_t) + loglik(x, theta_t)) / 2,
                    r"$\mathrm{KL}$",
                    va="center", ha="right", fontsize=11)
        if arrow_to is not None:
            new = thetas[np.argmax(bound)]
            ax.axvline(new, color=GREY, lw=.8, ls=(0, (1, 2)))
            ax.plot([new], [bound.max()], "o", color=ORANGE, ms=5, zorder=4)
            ax.annotate("", (new, ll.max() + 30), (theta_t, ll.max() + 30),
                        arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1, mutation_scale=9))
            ax.text((theta_t + new) / 2, ll.max() + 45, "M-step", ha="center", fontsize=10)
            ticks.append(new)
            labels.append(r"$\theta^{(t+1)}$")
        ax.set_xticks(ticks, labels)
        ax.set_yticks([])
        ax.set_xlim(thetas[0], thetas[-1])
        ax.set_ylim(lo, hi)
        ax.set_xlabel(r"$\theta$")
        ax.legend(frameon=False, loc="upper left")
        tidy(ax)
        fig.subplots_adjust(left=.03, right=.98, bottom=.2, top=.95)
        save(fig, name)

    theta_t = 176.
    panel("em-elbo-gap.png", theta_t, posterior(x, 196.), gap=True)
    q = posterior(x, theta_t)
    panel("em-elbo-nogap.png", theta_t, q)
    panel("em-elbo-mstep.png", theta_t, q, arrow_to=True)
    new = thetas[np.argmax([elbo(x, q, t) for t in thetas])]
    panel("em-elbo-next.png", new, posterior(x, new), label=r"$\theta^{(t+1)}$")


if __name__ == "__main__":
    graphical_models()
    x = bills()
    bills_figure(x)
    em_figures(x)
    elbo_figures(flippers())
