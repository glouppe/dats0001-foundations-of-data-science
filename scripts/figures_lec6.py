"""Figures of lecture 6: the two graphical models, in the style and at the scale of
lectures 4 and 5, and the plots of the lecture, redrawn from the data and seeds of
nb06a (random walks) and nb06b (Bayesian linear regression with Metropolis-Hastings).
The three trace figures of the diagnostics are simulated here, on known targets.

Usage: uv run python scripts/figures_lec6.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import corner  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from scipy.stats import multivariate_normal, norm  # noqa: E402

from figures_lec4 import (COL, FS, GREY, MARGIN, OBSERVED, R, STEP, UNIT, arrow, box,  # noqa: E402
                          figure, hyper, measure, node, plate, rescale)

WIDTH = 246.5568 / 72 / UNIT    # canvas width of the lecture 4 figures, in drawing units
LIGHT = "#b8c0c6"
BLUE = "#0173b2"
ORANGE = "#de8f05"
RED = "#c0392b"
GREEN = "#029e73"

plt.rcParams.update({"font.size": 12, "text.color": GREY, "axes.labelcolor": GREY,
                     "xtick.color": GREY, "ytick.color": GREY, "axes.edgecolor": LIGHT,
                     "mathtext.fontset": "cm"})


def tidy(ax):
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)


def save(fig, name):
    path = "figures/lec6/%s" % name
    fig.savefig(path, facecolor="white", bbox_inches="tight", pad_inches=.03, dpi=200)
    plt.close(fig)
    print("wrote", path)


# Graphical models --------------------------------------------------------------------

def markov_chain():
    """x_{t-1} -> x_t -> x_{t+1}, with dots on both sides."""
    fig, ax = figure()
    gap = 1.5
    xs = [-gap, 0, gap]
    for x, t in zip(xs, ["t-1", "t", "t+1"]):
        ax.add_patch(Circle((x, 0), R, facecolor="white", edgecolor=GREY, lw=1.15, zorder=2))
        ax.text(x, 0, r"$x_{%s}$" % t, ha="center", va="center", zorder=3, fontsize=FS - 3)
    for a, b in zip(xs, xs[1:]):
        arrow(ax, (a, 0), (b, 0))
    for side in (-1, 1):
        dots = (side * (gap + 1.0), 0)
        ax.text(*dots, r"$\cdots$", ha="center", va="center", fontsize=FS)
        if side < 0:
            arrow(ax, dots, (xs[0], 0), r_start=.32)
        else:
            arrow(ax, (xs[-1], 0), dots, r_end=.32)
    return fig, ax, "figures/lec6/markov-chain.svg"


def regression_model():
    """Bayesian linear regression: weights w and log sigma, shared by the N pairs (x_i, y_i)."""
    fig, ax = figure()
    node(ax, (0, STEP), r"$x_i$", observed=True)
    node(ax, (0, 0), r"$y_i$", observed=True)
    arrow(ax, (0, STEP), (0, 0))
    pairs = plate(ax, [box((0, STEP)), box((0, 0))], "$N$")
    left = pairs[0] - .55 - R
    w, s = (left, STEP), (left, 0)
    node(ax, w, r"$\mathbf{w}$")
    ax.add_patch(Circle(s, R, facecolor="white", edgecolor=GREY, lw=1.15, zorder=2))
    ax.text(*s, r"$\log \sigma$", ha="center", va="center", zorder=3, fontsize=FS - 5)
    arrow(ax, w, (0, 0))
    arrow(ax, s, (0, 0))
    hyper(ax, w, r"$\tau_1$", left - R, side="left")
    hyper(ax, s, r"$\tau_2$", left - R, side="left")
    return fig, ax, "figures/lec6/example1.svg"


def graphical_models():
    for fig, ax, path in [markov_chain(), regression_model()]:
        x0, y0, x1, y1 = measure(fig, ax)
        width = max(WIDTH, x1 - x0 + 2 * MARGIN)
        centre = (x0 + x1) / 2
        rescale(fig, ax, centre - width / 2, centre + width / 2, y0 - MARGIN, y1 + MARGIN)
        fig.savefig(path, facecolor="white")
        plt.close(fig)
        print("wrote %-28s %.2f x %.2f in" % (path, *fig.get_size_inches()))


# Random walks (nb06a) ----------------------------------------------------------------

def walk(N, T, kappa=0.0):
    """nb06a: x_1 ~ N(0, 1), x_{t+1} = x_t - kappa x_t + eps, eps ~ N(0, 1)."""
    x = np.zeros((N, T))
    x[:, 0] = np.random.normal(0, 1, size=N)
    for t in range(1, T):
        x[:, t] = x[:, t - 1] - kappa * x[:, t - 1] + np.random.normal(0, 1, size=N)
    return x


def random_walks():
    """The four random-walk figures, with nb06a's seed and sequence of draws."""
    np.random.seed(42)
    paths = walk(20, 100)
    long = walk(10000, 1000)
    paths_mr = walk(20, 100, kappa=.1)
    long2 = walk(10000, 1000)
    long_mr = walk(10000, 1000, kappa=.1)
    t = np.arange(1, 101)

    for name, x in (("gaussian_random_walk_1d.png", paths),
                    ("gaussian_random_walk_1d_drift.png", paths_mr)):
        fig, ax = plt.subplots(figsize=(5.8, 3.2), dpi=200)
        for row in x:
            ax.plot(t, row, color=BLUE, lw=.8, alpha=.45)
        ax.set_xlabel("$t$")
        ax.set_ylabel("$x_t$")
        ax.set_xlim(1, 100)
        ax.set_ylim(-14, 20)
        tidy(ax)
        fig.tight_layout()
        save(fig, name)

    tt = np.arange(1, 1001)
    fig, ax = plt.subplots(figsize=(5.8, 3.2), dpi=200)
    ax.plot(tt, long.var(axis=0), color=BLUE, lw=1.4, label="10,000 simulated walks")
    ax.plot(tt, tt, color=GREY, lw=1, ls=(0, (4, 3)), label="$1 + (t - 1)\\sigma^2$")
    ax.set_xlabel("$t$")
    ax.set_ylabel("variance of $x_t$")
    ax.set_xlim(1, 1000)
    ax.legend(frameon=False, loc="upper left")
    tidy(ax)
    fig.tight_layout()
    save(fig, "gaussian_random_walk_1d_variance.png")

    stationary = 1 / (2 * .1 - .1 ** 2)
    fig, ax = plt.subplots(figsize=(5.8, 3.2), dpi=200)
    ax.plot(tt, long2.var(axis=0), color=BLUE, lw=1.4, label="random walk")
    ax.plot(tt, long_mr.var(axis=0), color=ORANGE, lw=1.4, label="mean-reverting walk")
    ax.axhline(stationary, color=ORANGE, lw=1, ls=(0, (4, 3)))
    ax.text(1000, stationary * 1.25, r"$\sigma^2 / (2\kappa - \kappa^2) = %.2f$" % stationary,
            ha="right", va="bottom", color=ORANGE, fontsize=10)
    ax.set_yscale("log")
    ax.set_xlabel("$t$")
    ax.set_ylabel("variance of $x_t$")
    ax.set_xlim(1, 1000)
    ax.legend(frameon=False, loc="upper left")
    tidy(ax)
    fig.tight_layout()
    save(fig, "gaussian_random_walk_variance_comparison.png")
    print("  stationary variance %.2f, simulated %.2f" % (stationary, long_mr.var(axis=0)[-1]))


# Bayesian linear regression (nb06b) ----------------------------------------------------

THETA_TRUE = np.array([-3.0, 0.0, np.log(5.0)])     # slope, intercept, log sigma
TAU1 = TAU2 = 3.0


def regression_data():
    """nb06b: ten inputs uniform on [-10, 10], y = -3 x + 0 + N(0, 5^2), seed 777."""
    np.random.seed(777)
    x = np.sort(np.random.rand(10) * 20 - 10)
    y = THETA_TRUE[0] * x + THETA_TRUE[1] + np.random.randn(10) * np.exp(THETA_TRUE[2])
    return x, y


def log_posterior(theta, x, y):
    w0, w1, log_s = theta
    return (norm.logpdf(y, w0 * x + w1, np.exp(log_s)).sum()
            + norm.logpdf(theta[:2], 0, TAU1).sum() + norm.logpdf(log_s, 0, TAU2))


def mh(theta0, n, log_target, q, seed=0):
    """nb06b's Metropolis-Hastings, with its seed and order of draws."""
    np.random.seed(seed)
    theta, lp = theta0, log_target(theta0)
    out, accepts = [], 0
    for _ in range(n):
        prop = q(theta)
        u = np.random.rand()
        lp_prop = log_target(prop)
        if np.log(u) < lp_prop - lp:
            theta, lp = prop, lp_prop
            accepts += 1
        out.append(theta)
    return np.array(out), accepts / n


def regression():
    x, y = regression_data()
    thetas, rate = mh(np.zeros(3), 50000, lambda th: log_posterior(th, x, y),
                      lambda th: th + multivariate_normal.rvs(mean=np.zeros(3), cov=.1))
    print("  regression: acceptance rate %.2f" % rate)
    xs = np.linspace(-10, 10, 200)

    fig, ax = plt.subplots(figsize=(5.8, 3.4), dpi=200)
    ax.plot(xs, THETA_TRUE[0] * xs + THETA_TRUE[1], color=GREY, lw=1.2, ls=(0, (4, 3)),
            label="true line, $y = -3x$")
    ax.scatter(x, y, s=28, color=ORANGE, zorder=3, label="observations")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(frameon=False, loc="upper right")
    tidy(ax)
    fig.tight_layout()
    save(fig, "example1-data.png")

    labels = [r"$w_0$ (slope)", r"$w_1$ (intercept)", r"$\log \sigma$"]
    fig, axes = plt.subplots(3, 1, figsize=(5.8, 4.2), dpi=200, sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(thetas[:, j], color=BLUE, lw=.4)
        ax.axhline(THETA_TRUE[j], color=RED, lw=1, ls=(0, (4, 3)),
                   label="value used to simulate the data" if j == 0 else None)
        ax.set_ylabel(labels[j], fontsize=10)
        tidy(ax)
    axes[-1].set_xlabel("iteration $t$")
    axes[-1].set_xlim(0, len(thetas))
    fig.legend(frameon=False, loc="upper center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .95))
    save(fig, "example1-traces.png")

    warm = thetas[1000:]
    fig = corner.corner(warm, labels=[r"$w_0$", r"$w_1$", r"$\log \sigma$"],
                        truths=THETA_TRUE, truth_color=RED, color=BLUE,
                        quantiles=[.05, .5, .95], label_kwargs=dict(fontsize=12))
    fig.set_size_inches(5.2, 5.2)
    save(fig, "example1-corner.png")

    rng = np.random.default_rng(0)
    draws = warm[rng.choice(len(warm), 4000, replace=False)]
    ypred = (draws[:, :1] * xs + draws[:, 1:2]
             + np.exp(draws[:, 2:3]) * rng.standard_normal((len(draws), len(xs))))
    lo, med, hi = np.percentile(ypred, [5, 50, 95], axis=0)
    fig, ax = plt.subplots(figsize=(5.8, 3.4), dpi=200)
    ax.fill_between(xs, lo, hi, color=BLUE, alpha=.2, lw=0, label="5-95% band")
    ax.plot(xs, med, color=BLUE, lw=1.6, label="median")
    ax.plot(xs, THETA_TRUE[0] * xs + THETA_TRUE[1], color=GREY, lw=1.2, ls=(0, (4, 3)),
            label="true line")
    ax.scatter(x, y, s=28, color=ORANGE, zorder=3, label="observations")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    tidy(ax)
    fig.tight_layout()
    save(fig, "example1-pp.png")


# Diagnostics (simulated) ---------------------------------------------------------------

def rw_mh(log_target, x0, step, n, rng):
    """Random-walk Metropolis-Hastings in one dimension."""
    x, lp = x0, log_target(x0)
    out = np.empty(n)
    for k in range(n):
        prop = x + step * rng.standard_normal()
        lp_prop = log_target(prop)
        if np.log(rng.random()) < lp_prop - lp:
            x, lp = prop, lp_prop
        out[k] = x
    return out


def autocorrelation_time(chain, max_lag=2000):
    """tau_int with a simple window: sum of autocorrelations until the first negative one."""
    c = chain - chain.mean()
    acf = np.correlate(c, c, mode="full")[len(c) - 1:len(c) - 1 + max_lag] / (c.var() * len(c))
    stop = np.argmax(acf < 0) if (acf < 0).any() else max_lag
    return 1 + 2 * acf[1:stop].sum()


def two_traces(chains, titles, name, ylim=None):
    fig, axes = plt.subplots(2, 1, figsize=(5.8, 3.8), dpi=200, sharex=True)
    for ax, chain, title in zip(axes, chains, titles):
        ax.plot(chain, color=BLUE, lw=.4)
        ax.set_title(title, loc="left", fontsize=10, pad=3)
        if ylim:
            ax.set_ylim(*ylim)
        tidy(ax)
    axes[-1].set_xlabel("iteration $t$")
    axes[-1].set_xlim(0, len(chains[0]))
    fig.tight_layout()
    save(fig, name)


def diagnostics():
    rng = np.random.default_rng(3)
    gauss = lambda x: -.5 * x ** 2
    n = 10000

    slow = rw_mh(gauss, 0.0, .1, n, rng)
    fast = rw_mh(gauss, 0.0, 2.4, n, rng)
    t_slow, t_fast = autocorrelation_time(slow), autocorrelation_time(fast)
    two_traces([slow, fast],
               [r"proposal std 0.1: high autocorrelation, $\tau_\mathrm{int} \approx %.0f$" % t_slow,
                r"proposal std 2.4: low autocorrelation, $\tau_\mathrm{int} \approx %.0f$" % t_fast],
               "trace-autocorrelation.png", ylim=(-4.5, 4.5))
    print("  tau_int: %.0f and %.1f, ESS %.0f and %.0f" % (t_slow, t_fast, n / t_slow, n / t_fast))

    bimodal = lambda x: np.logaddexp(norm.logpdf(x, 1, .4), norm.logpdf(x, 5, .4))
    poor = rw_mh(bimodal, 1.0, .5, n, rng)
    good = rw_mh(bimodal, 1.0, 4.0, n, rng)
    two_traces([poor, good],
               ["proposal std 0.5: poor mixing, the chain rarely crosses between modes",
                "proposal std 4: good mixing"],
               "trace-mixing.png", ylim=(-1, 7))

    a = rw_mh(gauss, -25.0, .1, n, rng)
    b = rw_mh(gauss, 25.0, .1, n, rng)
    fig, ax = plt.subplots(figsize=(5.8, 2.8), dpi=200)
    ax.plot(a, color=BLUE, lw=.4, label="chain started at $-25$")
    ax.plot(b, color=ORANGE, lw=.4, label="chain started at $25$")
    ax.set_xlabel("iteration $t$")
    ax.set_xlim(0, n)
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    tidy(ax)
    fig.tight_layout()
    save(fig, "trace-convergence.png")


if __name__ == "__main__":
    graphical_models()
    random_walks()
    regression()
    diagnostics()
