"""Figures of lecture 6: the two graphical models, in the style and at the scale of
lectures 4 and 5, and the plots of the lecture, redrawn from the data and seeds of
nb06a (random walks) and nb06b (Bayesian linear regression with Metropolis-Hastings).
The three trace figures of the diagnostics and the animations of MH and HMC are simulated here, on
known targets; simulation-based calibration is run on the regression model of nb06b.

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


# Doeblin condition ---------------------------------------------------------------------

def doeblin_overlap():
    """m-step transition densities of a Gaussian random walk with steps of standard deviation 1.5,
    truncated to [-3, 3], from the starting
    states -3, -2, ..., 3, for m = 1 and m = 4 (top). Their pointwise minimum over all starting states
    is delta * nu, shaded; delta is its area. Below, nu itself, the common part rescaled to area 1,
    on its own scale."""
    g = np.linspace(-3, 3, 601)
    dx = g[1] - g[0]
    step = 1.5                                       # standard deviation of a step
    K = np.exp(-.5 * ((g[None, :] - g[:, None]) / step) ** 2)
    K /= K.sum(1, keepdims=True) * dx                # rows are densities on the grid
    starts = np.arange(-3, 4)
    fig, axes = plt.subplots(2, 2, figsize=(5.8, 3.9), dpi=200, sharex=True,
                             gridspec_kw=dict(height_ratios=[2, 1.2]))
    Pm = np.eye(len(g)) / dx
    for m, col in zip(range(1, 5), [0, None, None, 1]):
        Pm = Pm @ K * dx
        if col is None:
            continue
        ax, axn = axes[0, col], axes[1, col]
        common = Pm.min(0)
        delta = common.sum() * dx
        ax.fill_between(g, common, color=LIGHT, lw=0, label=r"common part $\delta \nu$")
        for x0 in starts:
            row = Pm[np.argmin(np.abs(g - x0))]
            if x0 == -3:
                ax.plot(g, row, color=BLUE, lw=1.4, label="$x_t = -3$")
            elif x0 == 3:
                ax.plot(g, row, color=ORANGE, lw=1.4, label="$x_t = 3$")
            else:
                ax.plot(g, row, color=GREY, lw=.6, alpha=.6,
                        label="$x_t = -2, \\ldots, 2$" if x0 == -2 else None)
        ax.set_title(r"$m = %d$, $\delta = %.2f$" % (m, delta), fontsize=10, pad=3)
        axn.fill_between(g, common / delta, color=GREEN, alpha=.25, lw=0)
        axn.plot(g, common / delta, color=GREEN, lw=1.2)
        axn.set_xlabel("$x_{t+m}$")
        axn.text(.02, .95, r"$\nu$", transform=axn.transAxes, ha="left", va="top", fontsize=11, color=GREEN)
        for a in (ax, axn):
            a.set_xlim(-3, 3)
            a.set_yticks([])
            tidy(a)
            a.spines["left"].set_visible(False)
        print("  m = %d: delta = %.4f" % (m, delta))
    axes[0, 1].sharey(axes[0, 0])
    axes[1, 1].sharey(axes[1, 0])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, .07, 1, 1))
    save(fig, "doeblin-overlap.png")


def doeblin():
    """Two copies of a chain that restarts from nu = N(0, 1) with probability delta at each step
    and otherwise takes a Gaussian random-walk step, started at -10 and 10 and driven by the
    same random numbers: they stay 20 apart until the first restart, and coincide from then on."""
    delta, T = .05, 80
    rng = np.random.default_rng(3)      # first restart at t = 21, near its mean 1 / delta
    restart = rng.random(T) < delta
    nu, eps = rng.standard_normal(T), rng.standard_normal(T)
    chains = np.empty((2, T))
    chains[:, 0] = -10, 10
    for t in range(1, T):
        prev = chains[:, t - 1]
        chains[:, t] = nu[t] if restart[t] else prev + eps[t]
    first = np.argmax(restart[1:]) + 2    # in the time index t = 1, ..., T of the plot

    fig, ax = plt.subplots(figsize=(5.8, 2.6), dpi=200)
    ax.axvline(first, color=LIGHT, lw=1, ls=(0, (4, 3)))
    ax.text(first + .8, 9, "first restart from $\\nu$", fontsize=10, va="top")
    ax.plot(np.arange(1, T + 1), chains[0], color=BLUE, lw=1.2, label="started at $-10$")
    ax.plot(np.arange(1, T + 1), chains[1], color=ORANGE, lw=1.2, label="started at $10$")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x_t$")
    ax.set_xlim(1, T)
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    tidy(ax)
    fig.tight_layout()
    save(fig, "doeblin.png")
    print("  first restart at t = %d" % first)


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
    for k in range(20):
        ax.plot(xs, draws[k, 0] * xs + draws[k, 1], color=BLUE, lw=.6, alpha=.5,
                label=r"$y = w_0 x + w_1$, posterior draws" if k == 0 else None)
    ax.plot(xs, med, color=BLUE, lw=1.6, label="median")
    ax.plot(xs, THETA_TRUE[0] * xs + THETA_TRUE[1], color=GREY, lw=1.2, ls=(0, (4, 3)),
            label="true line")
    ax.scatter(x, y, s=28, color=ORANGE, zorder=3, label="observations")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(frameon=False, loc="lower left", fontsize=10)
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


def autocorrelation_figure():
    """Autocorrelation rho_k against the lag k for the two chains of trace-autocorrelation.png (same
    seed and order of draws), with the area under each curve shaded up to the window that defines
    tau_int; tau_int = 1 + 2 sum_k rho_k is twice the shaded area plus one."""
    rng = np.random.default_rng(3)
    gauss = lambda x: -.5 * x ** 2
    n, max_lag = 10000, 1500
    chains = [rw_mh(gauss, 0.0, .1, n, rng), rw_mh(gauss, 0.0, 2.4, n, rng)]
    fig, ax = plt.subplots(figsize=(5.8, 2.8), dpi=200)
    for chain, color, std in zip(chains, [BLUE, ORANGE], ["0.1", "2.4"]):
        c = chain - chain.mean()
        acf = np.correlate(c, c, mode="full")[n - 1:n - 1 + max_lag] / (c.var() * n)
        stop = np.argmax(acf < 0) if (acf < 0).any() else max_lag
        tau = 1 + 2 * acf[1:stop].sum()
        lags = np.arange(max_lag)
        ax.fill_between(lags[:stop], acf[:stop], color=color, alpha=.25, lw=0)
        ax.plot(lags, acf, color=color, lw=1.2 if color == BLUE else .7,
                label=r"proposal std %s: $\tau_\mathrm{int} \approx %s$, ESS $\approx %d$"
                % (std, "%.0f" % tau if tau > 10 else "%.1f" % tau, round(n / tau, -1)))
    ax.axhline(0, color=LIGHT, lw=.8)
    ax.set_xlim(0, max_lag)
    ax.set_xlabel("lag $k$")
    ax.set_ylabel(r"$\rho_k$")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    tidy(ax)
    fig.tight_layout()
    save(fig, "autocorrelation.png")


def diagnostics():
    rng = np.random.default_rng(3)
    gauss = lambda x: -.5 * x ** 2
    n = 10000

    slow = rw_mh(gauss, 0.0, .1, n, rng)
    fast = rw_mh(gauss, 0.0, 2.4, n, rng)
    t_slow, t_fast = autocorrelation_time(slow), autocorrelation_time(fast)
    two_traces([slow, fast],
               ["proposal std 0.1: high autocorrelation",
                "proposal std 2.4: low autocorrelation"],
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


# Metropolis-Hastings at work ---------------------------------------------------------------

def hmc_contours():
    """Contour lines of H in the (x, v) plane for the two-mode 1-D target of the HMC animation, drawn
    as the middle panel of the animation: two paths of the dynamics (blue) on their contour lines
    (dashed), around one mode at low H and around both at higher H, joined by a new v (orange)."""
    logp = lambda x: np.logaddexp(np.log(.4) + norm.logpdf(x, -1.5, .6), np.log(.6) + norm.logpdf(x, 1.5, .7))
    grad = lambda x, h=1e-5: (logp(x + h) - logp(x - h)) / (2 * h)
    H = lambda x, v: -logp(x) + .5 * v ** 2
    gx = np.linspace(-3.5, 3.5, 300)
    gv = np.linspace(-3.5, 3.5, 300)
    GX, GV = np.meshgrid(gx, gv)
    HH = H(GX, GV)
    x0, v_low, v_high = -1.5, 1.0, 2.3
    fig, ax = plt.subplots(figsize=(3.2, 3), dpi=200)
    ax.contour(gx, gv, HH, levels=np.linspace(1.2, 6, 7), colors=LIGHT, linewidths=.7)
    for v0 in (v_low, v_high):
        ax.contour(gx, gv, HH, levels=[H(x0, v0)], colors=GREY, linewidths=1, linestyles="dashed")
        path, x, v = [(x0, v0)], x0, v0
        for _ in range(3000):
            v = v + .005 * grad(x)
            x = x + .01 * v
            v = v + .005 * grad(x)
            path.append((x, v))
        ax.plot(*np.array(path).T, color=BLUE, lw=2)
    ax.plot([x0, x0], [v_low, v_high], color=ORANGE, lw=2)
    ax.plot(x0, v_low, "o", color=BLUE, ms=6)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$v$")
    ax.set_title("$(x, v)$ plane", fontsize=11, pad=3)
    tidy(ax)
    save(fig, "hmc-contours.png")


def hmc_dynamics_animation():
    """Six HMC iterations on a two-mode 1-D target, from x = -1.5, with 30 leapfrog steps of size 0.1:
    the samples on the target, the path in the (x, v) plane over contours of H, and H against the
    leapfrog step. Each draw of v (orange) keeps x and changes H; each path (blue) keeps
    H nearly constant. The chain crosses to the other mode on the iteration with the largest H."""
    from PIL import Image
    logp = lambda x: np.logaddexp(np.log(.4) + norm.logpdf(x, -1.5, .6), np.log(.6) + norm.logpdf(x, 1.5, .7))
    grad = lambda x, h=1e-5: (logp(x + h) - logp(x - h)) / (2 * h)
    H = lambda x, v: -logp(x) + .5 * v ** 2
    eps, L, K = .1, 30, 6
    rng = np.random.default_rng(4)
    x, v_end = -1.5, 0.
    jumps, paths, samples = [], [], [x]          # jumps: (x, v before, v after); paths: (L + 1, 2)
    for _ in range(K):
        v = rng.standard_normal()
        jumps.append((x, v_end, v))
        path = [(x, v)]
        xx, vv = x, v
        for _ in range(L):
            vv = vv + eps / 2 * grad(xx)
            xx = xx + eps * vv
            vv = vv + eps / 2 * grad(xx)
            path.append((xx, vv))
        path = np.array(path)
        paths.append(path)
        if np.log(rng.random()) < H(x, v) - H(xx, vv):
            x, v_end = xx, vv
        else:
            v_end = v
        samples.append(x)

    gx = np.linspace(-3.5, 3.5, 300)
    gv = np.linspace(-3.5, 3.5, 300)
    GX, GV = np.meshgrid(gx, gv)
    HH = H(GX, GV)
    Hmax = max(H(*p.T).max() for p in paths)

    def draw(k, phase, j):
        """Iteration k; phase "jump" with fraction j of the momentum draw, or "path" up to step j."""
        for a in axes:
            a.cla()
        x0, va, vb = jumps[k]
        v_now = va + j * (vb - va) if phase == "jump" else None
        cur = (x0, v_now) if phase == "jump" else paths[k][j]
        done = k + (phase == "path" and j == L)
        ax = axes[0]
        ax.plot(gx, np.exp(logp(gx)), color=GREY, lw=1.2)
        ax.plot(samples[:done + 1], np.full(done + 1, .012), "|", color=BLUE, ms=12, mew=1.5)
        ax.plot(cur[0], np.exp(logp(cur[0])), "o", color=BLUE, ms=8)
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(0, .42)
        ax.set_yticks([])
        ax.set_xlabel("$x$")
        ax.set_title(r"target $\tilde{\pi}(x)$ and samples", fontsize=11, pad=3)
        ax = axes[1]
        ax.contour(gx, gv, HH, levels=np.linspace(1.2, 6, 7), colors=LIGHT, linewidths=.7)
        for i in range(k):
            ax.plot([jumps[i][0]] * 2, jumps[i][1:], color=ORANGE, lw=1, alpha=.35)
            ax.plot(*paths[i].T, color=BLUE, lw=1, alpha=.35)
        ax.plot([x0, x0], [va, vb if phase == "path" else v_now], color=ORANGE, lw=2)
        if phase == "path":
            ax.contour(gx, gv, HH, levels=[H(x0, vb)], colors=GREY, linewidths=1, linestyles="dashed")
            ax.plot(*paths[k][:j + 1].T, color=BLUE, lw=2)
        ax.plot(*cur, "o", color=BLUE, ms=6)
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$v$")
        ax.set_title("$(x, v)$ plane", fontsize=11, pad=3)
        ax = axes[2]
        for i in range(k + 1):
            s0 = i * L
            hb = H(jumps[i][0], jumps[i][2])
            ha = H(jumps[i][0], jumps[i][1])
            top = hb if (i < k or phase == "path") else ha + j * (hb - ha)
            ax.plot([s0, s0], [ha, top], color=ORANGE, lw=2)
            n = L if i < k else (j if phase == "path" else 0)
            if n:
                ax.plot(s0 + np.arange(n + 1), H(*paths[i][:n + 1].T), color=BLUE, lw=2)
        ax.set_xlim(-2, K * L + 2)
        ax.set_ylim(0, Hmax + .8)
        ax.set_xlabel("leapfrog step")
        ax.set_title("$H(x, v)$", fontsize=11, pad=3)
        ax.legend(handles=[plt.Line2D([], [], color=ORANGE, lw=2, label="redraw $v$"),
                           plt.Line2D([], [], color=BLUE, lw=2, label="path at constant $H$")],
                  frameon=False, loc="lower right", fontsize=9)
        for a in axes:
            tidy(a)
        fig.tight_layout()

    images, durations = [], []
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=100,
                             gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
    frames = []
    for k in range(K):
        frames += [(k, "jump", j, 120) for j in (.25, .5, .75)] + [(k, "jump", 1., 700)]
        frames += [(k, "path", j, 80) for j in range(0, L + 1, 2)]
        frames[-1] = frames[-1][:3] + (700,)
    frames[-1] = frames[-1][:3] + (3000,)
    for k, phase, j, ms in frames:
        draw(k, phase, j)
        fig.canvas.draw()
        images.append(Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
                      .convert("RGB").quantize(colors=64, method=Image.Quantize.MEDIANCUT))
        durations.append(ms)
    plt.close(fig)
    path = "figures/lec6/hmc-dynamics.gif"
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)
    print("wrote", path, "(%d frames)" % len(images))


# Two-mode target, shared by the Metropolis-Hastings and HMC animations
MIX_W = np.array([.4, .6])
MIX_MU = np.array([[-1.5, -1.0], [1.5, 1.2]])
MIX_COV = np.array([[[.5, .3], [.3, .5]], [[.6, -.2], [-.2, .4]]])
MIX_PREC = np.linalg.inv(MIX_COV)
MIX_START = np.array([0.0, 3.0])


def mix_comp_logs(x):
    """log w_k + log N(x | mu_k, cov_k) for both components, x of shape (..., 2)."""
    d = x[..., None, :] - MIX_MU
    return (np.log(MIX_W) - .5 * np.einsum("...ki,kij,...kj->...k", d, MIX_PREC, d)
            - .5 * np.log(np.linalg.det(MIX_COV)) - np.log(2 * np.pi))


def mix_log_target(x):
    return np.logaddexp.reduce(mix_comp_logs(x), axis=-1)


def mix_grad_log_target(x):
    logs = mix_comp_logs(x)
    resp = np.exp(logs - np.logaddexp.reduce(logs))
    return -np.einsum("k,kij,kj->i", resp, MIX_PREC, x - MIX_MU)


def run_mh(n=5000, step=1.0, seed=1, start=MIX_START):
    rng = np.random.default_rng(seed)
    x = np.asarray(start, float)
    chain, props, accepted = [x], [], []
    for _ in range(n):
        prop = x + step * rng.standard_normal(2)
        acc = np.log(rng.random()) < mix_log_target(prop) - mix_log_target(x)
        props.append(prop)
        accepted.append(acc)
        if acc:
            x = prop
        chain.append(x)
    return dict(name="Metropolis-Hastings", chain=np.array(chain), props=np.array(props),
                accepted=np.array(accepted), paths=None)


def run_hmc(n=5000, eps=.2, L=15, seed=1):
    rng = np.random.default_rng(seed)
    x = MIX_START
    chain, props, accepted, paths = [x], [], [], []
    for _ in range(n):
        v = rng.standard_normal(2)
        xs, vs = x.copy(), v + eps / 2 * mix_grad_log_target(x)
        path = [xs]
        for i in range(L):
            xs = xs + eps * vs
            vs = vs + (eps if i < L - 1 else eps / 2) * mix_grad_log_target(xs)
            path.append(xs)
        h0 = -mix_log_target(x) + .5 * v @ v
        h1 = -mix_log_target(xs) + .5 * vs @ vs
        acc = np.log(rng.random()) < h0 - h1
        props.append(xs)
        accepted.append(acc)
        paths.append(np.array(path))
        if acc:
            x = xs
        chain.append(x)
    return dict(name="Hamiltonian Monte Carlo", chain=np.array(chain), props=np.array(props),
                accepted=np.array(accepted), paths=paths)


def draw_corner(ax, ax1, ax2, run, t, step_by_step):
    """Corner plot of the first t iterations of a run: contours of the target, samples, path of
    the chain, and, when step_by_step, the last proposal accepted (green) or rejected (red)."""
    lim = (-4, 4)
    g = np.linspace(*lim, 200)
    G = np.stack(np.meshgrid(g, g), -1)
    ax.contour(g, g, np.exp(mix_log_target(G)), levels=5, colors=LIGHT, linewidths=.8)
    seen = run["chain"][:t + 1]
    ax.plot(seen[:, 0], seen[:, 1], ".", color=BLUE, ms=3 if t < 300 else 1.5, alpha=.5 if t < 300 else .25)
    path = seen if t <= 300 else seen[-100:]
    ax.plot(path[:, 0], path[:, 1], color=BLUE, lw=.8 if step_by_step else .5, alpha=.9 if step_by_step else .6)
    ax.plot(*seen[-1], "o", color=BLUE, ms=6)
    if step_by_step:
        cur, prop, acc = run["chain"][t - 1], run["props"][t - 1], run["accepted"][t - 1]
        c = GREEN if acc else RED
        if run["paths"] is None:
            ax.plot([cur[0], prop[0]], [cur[1], prop[1]], color=c, lw=1, ls=(0, (3, 2)))
        else:
            traj = run["paths"][t - 1]
            ax.plot(traj[:, 0], traj[:, 1], color=ORANGE, lw=2, zorder=5)
        ax.plot(*prop, "o" if acc else "x", color=c, ms=6, mew=1.5, mfc="none" if acc else None, zorder=6)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    tidy(ax)
    bins = np.linspace(*lim, 41)
    for d, a in ((0, ax1), (1, ax2)):
        marg = sum(MIX_W[k] * norm.pdf(g, MIX_MU[k, d], np.sqrt(MIX_COV[k, d, d])) for k in range(2))
        if d == 0:
            a.hist(seen[:, 0], bins=bins, density=True, color=LIGHT)
            a.plot(g, marg, color=GREY, lw=1.2)
            a.set_ylim(0, .8)
        else:
            a.hist(seen[:, 1], bins=bins, density=True, color=LIGHT, orientation="horizontal")
            a.plot(marg, g, color=GREY, lw=1.2)
            a.set_xlim(0, .8)
        a.axis("off")


def animate(runs, path, n=5000, slow=40):
    """GIF of corner plots of several runs side by side, iteration by iteration for the first
    `slow` iterations, then faster up to n."""
    from PIL import Image
    frames_t = list(range(1, slow + 1)) + sorted(set(np.geomspace(slow + 1, n, 50).astype(int)))
    durations = [500] * slow + [120] * (len(frames_t) - slow - 1) + [3000]
    fig = plt.figure(figsize=(4.6 * len(runs), 4.6), dpi=100)
    outer = fig.add_gridspec(1, len(runs), wspace=.25)
    panels = []
    for k in range(len(runs)):
        gs = outer[k].subgridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3], wspace=.05, hspace=.05)
        ax = fig.add_subplot(gs[1, 0])
        panels.append((ax, fig.add_subplot(gs[0, 0], sharex=ax), fig.add_subplot(gs[1, 1], sharey=ax)))
    images = []
    for t in frames_t:
        for run, (ax, ax1, ax2) in zip(runs, panels):
            for a in (ax, ax1, ax2):
                a.cla()
            draw_corner(ax, ax1, ax2, run, t, t <= slow)
            label = "$t = %d$\nacceptance\nrate %.0f%%" % (t, 100 * run["accepted"][:t].mean())
            if len(runs) > 1:
                label = run["name"].replace(" ", "\n", 1) + "\n\n" + label
            ax2.text(.05, 1.33, label, transform=ax2.transAxes, ha="left", va="top", fontsize=10, color=GREY)
        fig.canvas.draw()
        images.append(Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
                      .convert("RGB").quantize(colors=64, method=Image.Quantize.MEDIANCUT))
    plt.close(fig)
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)
    print("wrote", path, "(%d frames)" % len(images))


def mh_animation():
    """Random-walk MH on the two-mode target, as an animated corner plot."""
    mh = run_mh()
    print("  MH: acceptance rate %.2f" % mh["accepted"].mean())
    animate([mh], "figures/lec6/mh.gif")


def mh_hmc_animation():
    """The same MH chain next to HMC (eps = 0.2, L = 15) on the same target, from the same start,
    with the leapfrog trajectories of the first iterations in orange."""
    mh, hmc = run_mh(), run_hmc()
    for r in (mh, hmc):
        right = (r["chain"][1:].sum(1) > 0).mean()
        print("  %s: acceptance rate %.2f, fraction in the mode of weight 0.6: %.2f"
              % (r["name"], r["accepted"].mean(), right))
    animate([mh, hmc], "figures/lec6/mh-hmc.gif")


def mh_hmc_traces():
    """Trace plots of the chains of the two animations: x_1 and x_2 for MH alone, then MH and HMC
    side by side on shared axes, with the means of the two modes as dashed lines."""
    mh, hmc = run_mh(), run_hmc()
    labels = ["$x_1$", "$x_2$"]

    def trace(ax, chain, d, legend=False):
        for k in range(2):
            ax.axhline(MIX_MU[k, d], color=GREY, lw=.8, ls=(0, (4, 3)),
                       label="means of the two modes" if legend and k == 0 else None)
        ax.plot(chain[:, d], color=BLUE, lw=.4)
        ax.set_ylim(-4, 4)
        ax.set_xlim(0, len(chain) - 1)
        tidy(ax)

    # MH alone: the full trace of x_1, and a zoom where rejections show as flat segments
    t0, t1 = 2000, 2100
    fig, axes = plt.subplots(2, 1, figsize=(5.8, 3.6), dpi=200)
    trace(axes[0], mh["chain"], 0, legend=True)
    axes[0].axvspan(t0, t1, color=ORANGE, alpha=.25, lw=0)
    axes[0].set_ylabel(labels[0])
    axes[0].set_xlabel("iteration $t$")
    t = np.arange(t0, t1 + 1)
    for k in range(2):
        axes[1].axhline(MIX_MU[k, 0], color=GREY, lw=.8, ls=(0, (4, 3)))
    axes[1].step(t, mh["chain"][t0:t1 + 1, 0], where="post", color=BLUE, lw=1)
    axes[1].set_xlim(t0, t1)
    axes[1].set_ylim(-4, 4)
    axes[1].set_ylabel(labels[0])
    axes[1].set_xlabel("iteration $t$ (zoom on the orange window)")
    tidy(axes[1])
    fig.legend(frameon=False, loc="upper center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .94))
    save(fig, "mh-traces.png")

    # MH and HMC side by side: full traces of x_1 and zooms on the same window
    fig, axes = plt.subplots(2, 2, figsize=(5.8, 3.6), dpi=200, sharey=True)
    for j, run in enumerate((mh, hmc)):
        trace(axes[0, j], run["chain"], 0, legend=j == 0)
        axes[0, j].axvspan(t0, t1, color=ORANGE, alpha=.25, lw=0)
        axes[0, j].set_title(run["name"], fontsize=10, pad=3)
        axes[0, j].set_xlabel("iteration $t$", fontsize=9)
        for k in range(2):
            axes[1, j].axhline(MIX_MU[k, 0], color=GREY, lw=.8, ls=(0, (4, 3)))
        axes[1, j].step(t, run["chain"][t0:t1 + 1, 0], where="post", color=BLUE, lw=1)
        axes[1, j].set_xlim(t0, t1)
        axes[1, j].set_ylim(-4, 4)
        axes[1, j].set_xlabel("iteration $t$ (zoom)", fontsize=9)
        tidy(axes[1, j])
        for a in axes[:, j]:
            a.tick_params(labelsize=8)
    for d in range(2):
        axes[d, 0].set_ylabel(labels[0])
    fig.legend(frameon=False, loc="upper center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .92))
    save(fig, "mh-hmc-traces.png")


def split_rhat(chains):
    """Rank-normalized split-R-hat (Vehtari et al., 2021) of chains of shape (M, T)."""
    from scipy.stats import rankdata
    half = chains.shape[1] // 2
    z = np.concatenate([chains[:, :half], chains[:, half:2 * half]])
    m, n = z.shape
    z = norm.ppf((rankdata(z).reshape(z.shape) - 3 / 8) / (m * n + 1 / 4))
    means = z.mean(1)
    B = n / (m - 1) * ((means - means.mean()) ** 2).sum()
    W = z.var(1, ddof=1).mean()
    return np.sqrt(((n - 1) / n * W + B / n) / W)


def rhat_figure():
    """Traces of x_1 for four MH chains on the two-mode target, started at the four corners, after
    200, 1000 and 10000 iterations, with the split-R-hat of the iterations shown."""
    starts = [(-3, -3), (3, 3), (-3, 3), (3, -3)]
    chains = np.array([run_mh(n=10000, seed=10 + k, start=s0)["chain"][1:, 0] for k, s0 in enumerate(starts)])
    fig, axes = plt.subplots(3, 1, figsize=(5.8, 4.4), dpi=200)
    for ax, T in zip(axes, [200, 1000, 10000]):
        for chain, color in zip(chains, [BLUE, ORANGE, GREEN, RED]):
            ax.plot(np.arange(1, T + 1), chain[:T], color=color, lw=.6 if T <= 1000 else .3, alpha=.9)
        r = split_rhat(chains[:, :T])
        ax.set_title(r"%d iterations, $\hat{R} = %.3f$" % (T, r), loc="left", fontsize=10, pad=3)
        ax.set_xlim(0, T)
        ax.set_ylim(-4, 4)
        ax.set_ylabel("$x_1$")
        tidy(ax)
        print("  R-hat after %d iterations: %.4f" % (T, r))
    axes[-1].set_xlabel("iteration $t$")
    fig.tight_layout()
    save(fig, "rhat.png")


# Simulation-based calibration ------------------------------------------------------------

def calibration_intervals():
    """90% credible intervals of a Gaussian mean in 50 simulated experiments: theta* ~ N(0, 1), five
    observations x_i ~ N(theta*, 1), exact posterior N(sum x / 6, 1 / 6); next to it, an
    overconfident posterior with intervals half as wide. Intervals that miss theta* are red."""
    rng = np.random.default_rng(5)
    n_exp, n, z = 50, 5, norm.ppf(.95)
    theta = rng.standard_normal(n_exp)
    x = theta[:, None] + rng.standard_normal((n_exp, n))
    mean, sd = x.sum(1) / (n + 1), np.sqrt(1 / (n + 1))
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 3.1), dpi=200, sharey=True)
    k = np.arange(n_exp)
    for ax, width, title in zip(axes, [1, .5], ["exact posterior", "overconfident posterior"]):
        half = width * z * sd
        miss = np.abs(mean - theta) > half
        for color, sel in ((BLUE, ~miss), (RED, miss)):
            ax.vlines(k[sel], (mean - half)[sel], (mean + half)[sel], color=color, lw=1.2)
        ax.plot(k, theta, "o", color=ORANGE, ms=2.5, zorder=3, label=r"$\theta^*$")
        ax.set_title("%s\n%d misses out of %d" % (title, miss.sum(), n_exp), fontsize=10, pad=3)
        ax.set_xticks([])
        ax.set_xlabel("simulated experiment", fontsize=9)
        tidy(ax)
        print("  %s: %d misses" % (title, miss.sum()))
    axes[0].set_ylabel(r"$\theta$")
    handles = [plt.Line2D([], [], color=BLUE, lw=1.2), plt.Line2D([], [], color=RED, lw=1.2),
               plt.Line2D([], [], color=ORANGE, marker="o", ls="", ms=3)]
    fig.legend(handles, ["90% interval containing $\\theta^*$", "90% interval missing $\\theta^*$", "$\\theta^*$"],
               frameon=False, loc="lower center", fontsize=9, ncol=3)
    fig.tight_layout(rect=(0, .1, 1, 1))
    save(fig, "calibration-intervals.png")


def log_posterior_batch(theta, x, y, bug=False):
    """Unnormalized log posterior of the regression for S data sets at once, theta (S, 3), y (S, N).
    With bug=True, the likelihood takes sigma^2 as its scale, a common mistake with scipy."""
    w0, w1, log_s = theta[:, :1], theta[:, 1:2], theta[:, 2:3]
    scale = np.exp(2 * log_s) if bug else np.exp(log_s)
    log_lik = (-.5 * ((y - w0 * x - w1) / scale) ** 2 - np.log(scale)).sum(1)
    return log_lik - .5 * (theta[:, :2] ** 2).sum(1) / TAU1 ** 2 - .5 * theta[:, 2] ** 2 / TAU2 ** 2


def sbc_ranks(S=1000, L=99, thin=100, warm=4000, bug=False, seed=0):
    """Ranks of the true parameters among L posterior draws, for S simulations. Each simulation
    draws theta from the prior and y at the inputs of nb06b, then runs random-walk MH whose
    proposal is adapted during the warm-up only, and keeps one draw in `thin`."""
    rng = np.random.default_rng(seed)
    x, _ = regression_data()
    N = len(x)
    truth = np.column_stack([rng.normal(0, TAU1, (S, 2)), rng.normal(0, TAU2, S)])
    y = truth[:, :1] * x + truth[:, 1:2] + np.exp(truth[:, 2:3]) * rng.standard_normal((S, N))

    # start at least squares, with a proposal from its covariance
    X = np.column_stack([x, np.ones(N)])
    w = np.linalg.lstsq(X, y.T, rcond=None)[0].T
    s2 = ((y - w @ X.T) ** 2).sum(1) / (N - 2)
    theta = np.column_stack([w, .5 * np.log(s2)])
    cov = np.zeros((S, 3, 3))
    cov[:, :2, :2] = np.linalg.inv((X.T @ X)[None] / s2[:, None, None] + np.eye(2) / TAU1 ** 2)
    cov[:, 2, 2] = 1 / (2 * N)
    chol = np.linalg.cholesky(2.38 ** 2 / 3 * cov)

    lp = log_posterior_batch(theta, x, y, bug)
    window, draws = [], []
    for k in range(warm + L * thin):
        if k in (500, 1000, 2000, 3000):
            W = np.stack(window, 1)
            ok = (np.diff(W, axis=1) != 0).any(2).mean(1) > .1
            Wc = W - W.mean(1, keepdims=True)
            emp = np.einsum("ski,skj->sij", Wc, Wc) / (W.shape[1] - 1)
            emp = np.where(ok[:, None, None], emp + 1e-12 * np.eye(3), np.eye(3))
            chol = np.where(ok[:, None, None], np.linalg.cholesky(2.38 ** 2 / 3 * emp), .3 * chol)
            window = []
        prop = theta + np.einsum("sij,sj->si", chol, rng.standard_normal((S, 3)))
        lp_prop = log_posterior_batch(prop, x, y, bug)
        accept = np.log(rng.random(S)) < lp_prop - lp
        theta[accept], lp[accept] = prop[accept], lp_prop[accept]
        if k < warm:
            window.append(theta.copy())
        elif (k - warm) % thin == thin - 1:
            draws.append(theta.copy())
    return (np.stack(draws, 1) < truth[:, None, :]).sum(1)


def calibration():
    from scipy.stats import binom
    S, L, bins = 1000, 99, 20
    labels = [r"$w_0$ (slope)", r"$w_1$ (intercept)", r"$\log \sigma$"]
    rows = [(sbc_ranks(S, L), "correct likelihood"),
            (sbc_ranks(S, L, bug=True), r"bug: $\sigma^2$ in place of $\sigma$ in the likelihood")]
    lo, hi = binom.ppf([.005, .995], S, 1 / bins)
    fig, axes = plt.subplots(2, 3, figsize=(5.8, 3.6), dpi=200, sharex=True, sharey="row")
    for (ranks, title), row in zip(rows, axes):
        for j, ax in enumerate(row):
            ax.hist(ranks[:, j], bins=bins, range=(0, L + 1), color=BLUE)
            ax.axhspan(lo, hi, color=GREY, alpha=.2, lw=0, zorder=3)
            ax.axhline(S / bins, color=GREY, lw=.8, ls=(0, (4, 3)))
            ax.set_xlim(0, L + 1)
            ax.set_yticks([])
            ax.tick_params(labelsize=9)
            tidy(ax)
            ax.spines["left"].set_visible(False)
        row[0].set_title(title, loc="left", fontsize=10, pad=3)
    axes[0, 0].set_ylim(0, 2.5 * S / bins)
    for ax, label in zip(axes[-1], labels):
        ax.set_xlabel("rank of " + label, fontsize=9)
    fig.tight_layout()
    save(fig, "sbc.png")

    # expected coverage of the central credible intervals, read off the same ranks: theta* lies in
    # the central interval of level alpha when its rank is within the central fraction alpha of 0..L
    alphas = np.linspace(0, 1, 101)
    band = np.array([binom.ppf([.005, .995], S, a) / S for a in alphas])
    fig, axes = plt.subplots(1, 3, figsize=(5.8, 2.7), dpi=200, sharey=True)
    for j, ax in enumerate(axes):
        ax.fill_between(alphas, band[:, 0], band[:, 1], color=GREY, alpha=.2, lw=0)
        ax.plot([0, 1], [0, 1], color=GREY, lw=.8, ls=(0, (4, 3)))
        for (ranks, title), color in zip(rows, [BLUE, ORANGE]):
            u = (ranks[:, j] + .5) / (L + 1)              # position of theta* among the draws, in (0, 1)
            cover = [(np.abs(u - .5) <= a / 2).mean() for a in alphas]
            ax.plot(alphas, cover, color=color, lw=1.4, label=title if j == 0 else None)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel(r"credible level $\alpha$", fontsize=9)
        ax.set_title(labels[j], fontsize=10, pad=3)
        ax.tick_params(labelsize=8)
        tidy(ax)
    axes[0].set_ylabel("coverage", fontsize=9)
    fig.legend(frameon=False, loc="lower center", fontsize=9, ncol=2)
    fig.tight_layout(rect=(0, .13, 1, 1))
    save(fig, "coverage.png")


if __name__ == "__main__":
    graphical_models()
    random_walks()
    doeblin_overlap()
    doeblin()
    mh_animation()
    regression()
    diagnostics()
    autocorrelation_figure()
    hmc_contours()
    hmc_dynamics_animation()
    mh_hmc_animation()
    mh_hmc_traces()
    rhat_figure()
    calibration_intervals()
    calibration()
