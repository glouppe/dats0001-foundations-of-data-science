"""Figures of lecture 7: the two graphical models of Newcomb's data, in the style and at the
scale of lectures 4 to 6, and the plots of the lecture, redrawn from the data and models of
nb07a (posterior predictive checks and residuals of the Gaussian model), nb07b (the t model)
and nb07c (polynomial regression, with nb07c's data and seed).

Usage: uv run python scripts/figures_lec7.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import corner  # noqa: E402
import emcee  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import gamma, norm, t  # noqa: E402

from figures_lec4 import (GREY, MARGIN, R, STEP, UNIT, arrow, box, figure,  # noqa: E402
                          measure, node, plate, rescale)

WIDTH = 246.5568 / 72 / UNIT    # canvas width of the lecture 4 figures, in drawing units
LIGHT = "#b8c0c6"
BLUE = "#0173b2"
ORANGE = "#de8f05"
GREEN = "#029e73"

plt.rcParams.update({"font.size": 12, "legend.fontsize": 10, "axes.titlesize": 12,
                     "text.color": GREY, "axes.labelcolor": GREY,
                     "xtick.color": GREY, "ytick.color": GREY, "axes.edgecolor": LIGHT,
                     "mathtext.fontset": "cm"})

# Every plot is W inches wide, shown at .width-70, with the fonts above: one scale for all.
# A smaller figure keeps that scale when shown at .width-NN with NN = 70 * width / W.
W = 6.4


def inches(width_class):
    """The width of a figure shown at .width-NN at the common scale."""
    return W * width_class / 70

# Newcomb's 66 measurements of the passage time of light, in deviations from 24800 ns (nb07a).
NEWCOMB = np.array([28, 26, 33, 24, 34, -44, 27, 16, 40, -2, 29, 22, 24, 21, 25,
                    30, 23, 29, 31, 19, 24, 20, 36, 32, 36, 28, 25, 21, 28, 29,
                    37, 25, 28, 26, 30, 32, 36, 26, 30, 22, 36, 23, 27, 27, 28,
                    27, 31, 27, 26, 33, 26, 32, 32, 24, 39, 28, 24, 25, 32, 25,
                    29, 27, 28, 29, 16, 23])
XLABEL = "passage time $-$ 24800 (ns)"


def tidy(ax):
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)


def save(fig, name):
    path = "figures/lec7/%s" % name
    fig.savefig(path, facecolor="white", dpi=200)
    plt.close(fig)
    print("wrote", path)


# Graphical models ----------------------------------------------------------------------

def gaussian_model():
    """mu and sigma, shared by the N measurements x_i."""
    fig, ax = figure()
    node(ax, (0, 0), r"$x_i$", observed=True)
    measurements = plate(ax, [box((0, 0))], "$N$")
    left = measurements[0] - .55 - R
    for label, y in [(r"$\mu$", .7), (r"$\sigma$", -.7)]:
        node(ax, (left, y), label)
        arrow(ax, (left, y), (0, 0))
    return fig, ax, "figures/lec7/newcomb-model.svg"


def t_model():
    """mu and sigma shared by the x_i; nu shared by the noise terms epsilon_i."""
    fig, ax = figure()
    node(ax, (0, STEP), r"$\epsilon_i$")
    node(ax, (0, 0), r"$x_i$", observed=True)
    arrow(ax, (0, STEP), (0, 0))
    measurements = plate(ax, [box((0, STEP)), box((0, 0))], "$N$")
    left = measurements[0] - .55 - R
    node(ax, (left, STEP), r"$\nu$")
    arrow(ax, (left, STEP), (0, STEP))
    for label, y in [(r"$\mu$", .3), (r"$\sigma$", -.9)]:
        node(ax, (left, y), label)
        arrow(ax, (left, y), (0, 0))
    return fig, ax, "figures/lec7/newcomb-model2.svg"


def graphical_models():
    for fig, ax, path in [gaussian_model(), t_model()]:
        x0, y0, x1, y1 = measure(fig, ax)
        width = max(WIDTH, x1 - x0 + 2 * MARGIN)
        centre = (x0 + x1) / 2
        rescale(fig, ax, centre - width / 2, centre + width / 2, y0 - MARGIN, y1 + MARGIN)
        fig.savefig(path, facecolor="white")
        plt.close(fig)
        print("wrote %-28s %.2f x %.2f in" % (path, *fig.get_size_inches()))


# Models of Newcomb's data (nb07a, nb07b) ------------------------------------------------

def log_prior_gaussian(theta):
    mu, sigma = theta
    if not (-1000 <= mu <= 1000 and .1 <= sigma <= 1000):
        return -np.inf
    return -np.log(2000) - np.log(999.9)


def log_posterior_gaussian(theta, x):
    lp = log_prior_gaussian(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + norm.logpdf(x, theta[0], theta[1]).sum()


def log_posterior_t(theta, x):
    """theta = (mu, sigma, log nu), with nu ~ Gamma(2, scale 10) and its Jacobian."""
    mu, sigma, log_nu = theta
    lp = log_prior_gaussian((mu, sigma))
    if not np.isfinite(lp):
        return -np.inf
    nu = np.exp(log_nu)
    ll = t.logpdf(x, df=nu, loc=mu, scale=sigma).sum()
    if not np.isfinite(ll):
        return -np.inf
    return lp + gamma.logpdf(nu, a=2, scale=10) + log_nu + ll


def sample(log_posterior, start, x, steps=20000, walkers=16, seed=0):
    rng = np.random.default_rng(seed)
    start = minimize(lambda th: -log_posterior(th, x), start, method="Nelder-Mead").x
    pos = start + 1e-2 * rng.standard_normal((walkers, len(start)))
    sampler = emcee.EnsembleSampler(walkers, len(start), log_posterior, args=(x,))
    sampler.random_state = np.random.RandomState(seed).get_state()
    sampler.run_mcmc(pos, steps)
    tau = sampler.get_autocorr_time()
    return sampler.get_chain(discard=1000, thin=int(np.max(tau)), flat=True)


def corner_plot(thetas, labels, name, width_class):
    """A corner plot at the common scale, shown at .width-`width_class`."""
    size = inches(width_class)
    fig = plt.figure(figsize=(size, size), dpi=200)
    corner.corner(thetas, labels=labels, color=BLUE, quantiles=[.025, .5, .975], fig=fig,
                  max_n_ticks=3, label_kwargs=dict(fontsize=12))
    fig.subplots_adjust(left=.2, bottom=.2, right=.97, top=.97, wspace=.08, hspace=.08)
    save(fig, name)


def data_figure():
    fig, ax = plt.subplots(figsize=(W, 2.6), dpi=200)
    ax.hist(NEWCOMB, bins=np.arange(-46, 44, 2), color=BLUE, lw=0)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel("measurements")
    tidy(ax)
    fig.tight_layout()
    save(fig, "newcomb-data.png")


def replicate(thetas, rng):
    """One replicated data set of N measurements for each posterior draw."""
    return thetas[:, :1] + thetas[:, 1:2] * rng.standard_normal((len(thetas), len(NEWCOMB)))


def replicates_figure(reps):
    """The observed data among 19 replicated data sets."""
    fig, axes = plt.subplots(4, 5, figsize=(W, 4.6), dpi=200, sharex=True, sharey=True)
    bins = np.arange(-50, 66, 5)
    for k, ax in enumerate(axes.flat):
        observed = k == 7
        ax.hist(NEWCOMB if observed else reps[k], bins=bins,
                color=ORANGE if observed else BLUE, lw=0)
        ax.set_yticks([])
        ax.set_xticks([-50, 0, 50])
        tidy(ax)
        ax.spines["left"].set_visible(False)
        if observed:
            ax.set_title("observed", color=ORANGE, pad=2)
    fig.supxlabel(XLABEL, fontsize=12)
    fig.tight_layout(h_pad=.6, w_pad=.4)
    save(fig, "newcomb-replicates.png")


def statistic_panel(ax, values, observed, title):
    ax.hist(values, bins=30, color=BLUE, lw=0, density=True,
            label="replicated data sets")
    ax.axvline(observed, color=ORANGE, lw=2, label="observed data")
    p = np.mean(values >= observed)
    ax.set_title("%s, $p = %.2f$" % (title, p))
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    tidy(ax)
    return p


def statistics_figures(reps):
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.8), dpi=200)
    p_mean = statistic_panel(axes[0], reps.mean(1), NEWCOMB.mean(), r"mean $T(\mathbf{x})$")
    p_var = statistic_panel(axes[1], reps.var(1), NEWCOMB.var(), r"variance $T(\mathbf{x})$")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="lower center",
               ncol=2)
    fig.tight_layout(rect=(0, .1, 1, 1))
    save(fig, "newcomb-replicate-stats.png")

    fig, ax = plt.subplots(figsize=(W, 2.8), dpi=200)
    p_min = statistic_panel(ax, reps.min(1), NEWCOMB.min(), r"minimum $T(\mathbf{x})$")
    ax.set_xlabel(XLABEL)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(.06, 1))
    fig.tight_layout()
    save(fig, "newcomb-replicate-mins.png")
    print("  Bayesian p-values: mean %.3f, variance %.3f, minimum %.3f" % (p_mean, p_var, p_min))


def residuals_figures(thetas, reps):
    """Standardized residuals of the observed and of the replicated data, each replicated
    data set standardized with the posterior draw it was simulated from."""
    r_obs = ((NEWCOMB[None, :] - thetas[:, :1]) / thetas[:, 1:2]).ravel()
    r_rep = ((reps - thetas[:, :1]) / thetas[:, 1:2]).ravel()

    fig, (ax, qq) = plt.subplots(1, 2, figsize=(W, 3.2), dpi=200,
                                 gridspec_kw=dict(width_ratios=[1.45, 1]))
    bins = np.linspace(-7, 4, 89)
    ax.hist(r_rep, bins=bins, density=True, color=BLUE, alpha=.5, lw=0,
            label="replicated data")
    ax.hist(r_obs, bins=bins, density=True, histtype="step", color=ORANGE, lw=1.6,
            label="observed data")
    ax.set_xlabel("standardized residual $r_i$")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    tidy(ax)

    q = np.linspace(.005, .995, 199)
    lim = (-7, 3.5)
    qq.plot(lim, lim, color=GREY, lw=1, ls=(0, (4, 3)))
    qq.plot(np.quantile(r_rep, q), np.quantile(r_obs, q), "o", ms=2.5, color=BLUE)
    qq.set_xlim(*lim)
    qq.set_ylim(*lim)
    qq.set_aspect("equal")
    qq.set_xticks([-6, -3, 0, 3])
    qq.set_yticks([-6, -3, 0, 3])
    qq.set_xlabel("replicated quantiles")
    qq.set_ylabel("observed quantiles")
    tidy(qq)
    fig.tight_layout()
    save(fig, "newcomb-residuals.png")


def newcomb():
    data_figure()
    thetas = sample(log_posterior_gaussian, [26., 10.], NEWCOMB)
    print("  Gaussian model: %d draws, mu %.2f (sd %.2f), 95%% equal-tailed [%.2f, %.2f]"
          % (len(thetas), thetas[:, 0].mean(), thetas[:, 0].std(),
             *np.percentile(thetas[:, 0], [2.5, 97.5])))
    corner_plot(thetas, [r"$\mu$", r"$\sigma$"], "newcomb-posterior.png", 45)

    rng = np.random.default_rng(1)
    draws = thetas[rng.choice(len(thetas), 4000, replace=False)]
    reps = replicate(draws, rng)
    replicates_figure(reps)
    statistics_figures(reps)
    residuals_figures(draws, reps)

    thetas = sample(log_posterior_t, [26., 5., np.log(3.)], NEWCOMB)
    thetas[:, 2] = np.exp(thetas[:, 2])
    print("  t model: %d draws, mu %.2f, sigma %.2f, nu median %.2f"
          % (len(thetas), thetas[:, 0].mean(), thetas[:, 1].mean(), np.median(thetas[:, 2])))
    corner_plot(thetas, [r"$\mu$", r"$\sigma$", r"$\nu$"], "newcomb-model2-posterior.png", 55)


# Polynomial regression (nb07c) ----------------------------------------------------------

def f_true(x):
    return .5 * x ** 3 - x ** 2 + 2 * x + 1


def polynomial(x, coeffs):
    return sum(c * x ** k for k, c in enumerate(coeffs))


def polynomials():
    np.random.seed(123)
    x = np.linspace(-3, 3, 20)
    y = f_true(x) + np.random.normal(0, 3, size=len(x))

    def neg_log_posterior(coeffs):
        return (.5 * np.sum((y - polynomial(x, coeffs)) ** 2) / 3 ** 2
                + .5 * np.sum(coeffs ** 2) / 10 ** 2)

    xs = np.linspace(-3, 3, 200)
    fig, ax = plt.subplots(figsize=(W, 3.2), dpi=200)
    ax.plot(xs, f_true(xs), color=GREY, lw=1.2, ls=(0, (4, 3)), label="true function")
    for degree, color in [(3, BLUE), (7, GREEN)]:
        coeffs = minimize(neg_log_posterior, np.zeros(degree + 1), method="BFGS").x
        ax.plot(xs, polynomial(xs, coeffs), color=color, lw=1.6,
                label="degree %d, MAP" % degree)
    ax.scatter(x, y, s=28, color=ORANGE, zorder=3, label="observations")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(frameon=False, loc="upper left")
    tidy(ax)
    fig.tight_layout()
    save(fig, "polynomial-fit.png")


# Cross-validation -----------------------------------------------------------------------

def kfold(k=5):
    fig, ax = plt.subplots(figsize=(W, 2.6), dpi=200)
    for run in range(k):
        y = k - 1 - run
        for fold in range(k):
            held = fold == run
            ax.add_patch(Rectangle((fold, y + .12), .94, .76, lw=0,
                                   facecolor=BLUE if held else LIGHT))
        ax.text(-.15, y + .5, "split %d" % (run + 1), ha="right", va="center")
    ax.add_patch(Rectangle((0, -1), .3, .45, facecolor=LIGHT, lw=0))
    ax.text(.4, -.78, "training", va="center")
    ax.add_patch(Rectangle((1.9, -1), .3, .45, facecolor=BLUE, lw=0))
    ax.text(2.3, -.78, "validation", va="center")
    ax.set_xlim(-1.3, k)
    ax.set_ylim(-1.1, k)
    ax.set_axis_off()
    ax.set_position((0, 0, 1, 1))
    save(fig, "kfold.png")


if __name__ == "__main__":
    graphical_models()
    newcomb()
    polynomials()
    kfold()
