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


# Newcomb's experiment ------------------------------------------------------------------

def newcomb_animation(distance=3721., time=24.83e-6, turns=230., exaggerate=4, path=None):
    """Newcomb's rotating-mirror measurement (1882), as a GIF. Light leaves the rotating mirror
    at Fort Myer, on the heights west of the Potomac, is sent back by a fixed mirror at the base
    of the Washington Monument, 3721 m away, and returns about 24.83 microseconds later.
    Meanwhile the mirror, a square steel prism driven by air turbines at 200 to 250 turns per
    second, has turned by an angle alpha, so the returning light leaves at 2 alpha from its way
    in. A beam splitter below the mirror sends the returning light to the eyepiece, where the
    image of the slit appears shifted by an amount set by 2 alpha (the layout of Carter & Carter,
    2002, Fig. 2). The time is alpha / omega. The profile is schematic, with heights exaggerated;
    the angles of the close-up are drawn `exaggerate` times larger than the real ones."""
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Arc, FancyArrowPatch, Polygon, Rectangle
    from PIL import Image
    path = path or "figures/lec7/newcomb-experiment.gif"
    omega = 2 * np.pi * turns                       # rad/s
    alpha = omega * time                            # rad, about 2 degrees
    shown = exaggerate * alpha
    PRE, TRIP, POST = 8, 56, 12                     # frames before, during and after the trip
    GROUND, WATER, STEEL = "#e6e2da", "#cfe1ee", "#9aa5ad"
    station = np.array([0., .62])                   # the rotating mirror, on the heights
    fixed = np.array([distance - 60, .05])          # the fixed mirror, at the Monument's base
    fig = plt.figure(figsize=(W, 3.0), dpi=150)
    ax = fig.add_axes((0, .02, .64, .96))
    mx = fig.add_axes((.64, .02, .36, .96))

    def ground(ax):
        x = np.linspace(-700, 4100, 400)
        heights = (.55 * np.exp(-((x + 80) / 520) ** 2) + .02
                   - .14 * ((x > 1150) & (x < 2050)) * np.sin(np.pi * np.clip((x - 1150) / 900, 0, 1)) ** .3)
        ax.fill_between(x, -.6, heights, color=GROUND, lw=0, zorder=1)
        ax.plot(x, heights, color=GREY, lw=.7, zorder=2)
        river = (x > 1150) & (x < 2050)
        ax.fill_between(x[river], heights[river], -.02, color=WATER, lw=0, zorder=2)
        ax.text(1600, -.25, "Potomac", ha="center", fontsize=9, style="italic", color=BLUE, zorder=3)

    def monument(ax):
        x0, w, h = distance, 55, 1.55
        ax.add_patch(Polygon([(x0 - w, .02), (x0 + w, .02), (x0 + .68 * w, h), (x0, h + .14),
                              (x0 - .68 * w, h)], closed=True, facecolor="white",
                             edgecolor=GREY, lw=.8, zorder=3))
        ax.text(x0, h + .3, "Washington\nMonument", ha="center", va="bottom", fontsize=9, zorder=3)

    def pulse(ax, a, b, f, back=False):
        """A short bright pulse at fraction f of the way from a to b, with a fading trail."""
        head = a + f * (b - a)
        tail = a + max(f - .07, 0) * (b - a)
        pts = np.linspace(tail, head, 12)
        segs = np.stack([pts[:-1], pts[1:]], 1)
        lc = LineCollection(segs, colors=[(*plt.matplotlib.colors.to_rgb(ORANGE), a_)
                                          for a_ in np.linspace(.05, 1, 11)], lw=3, zorder=6,
                            capstyle="round")
        ax.add_collection(lc)
        ax.plot(*head, "o", color=ORANGE, ms=4.5, zorder=7)

    def draw(stage, u):
        ax.cla()
        mx.cla()
        # Profile: Fort Myer to the Monument, heights exaggerated.
        ground(ax)
        monument(ax)
        ax.add_patch(Rectangle(station - (35, .05), 70, .1, facecolor=GREY, lw=0, zorder=4))
        ax.text(*(station + (0, .22)), "Fort Myer", ha="center", va="bottom", fontsize=9)
        ax.add_patch(Rectangle(fixed - (8, .09), 16, .18, facecolor=GREY, lw=0, zorder=4))
        ax.plot(*np.array([station, fixed]).T, color=ORANGE, lw=.7, alpha=.25, zorder=5)
        elapsed = {0: 0., 1: u * time, 2: time}[stage]
        if stage == 1:
            f = 2 * u if u < .5 else 2 - 2 * u
            if u < .5:
                pulse(ax, station, fixed, f)
            else:
                pulse(ax, fixed, station, 1 - f)
        ax.annotate("", (distance, -.48), (0, -.48),
                    arrowprops=dict(arrowstyle="<|-|>", color=GREY, lw=.7, mutation_scale=7))
        ax.text(distance / 2, -.44, "3721 m", ha="center", va="bottom", fontsize=9,
                bbox=dict(facecolor=GROUND, edgecolor="none", pad=1))
        ax.text(-620, 2.05, r"$t = %5.2f\ \mu\mathrm{s}$" % (elapsed * 1e6), fontsize=13,
                va="top")
        ax.text(-620, 1.7, r"$\alpha = %4.2f^\circ$" % np.degrees(omega * elapsed),
                fontsize=13, va="top")
        if stage == 2 and u > .3:
            ax.text(-620, 1.33, r"$t = \alpha / \omega$, $\omega \approx %d$ turns/s" % turns,
                    fontsize=10, va="top", color=ORANGE)
        ax.set_xlim(-700, 4100)
        ax.set_ylim(-.6, 2.15)
        ax.set_axis_off()

        # Close-up of the rotating mirror: a square steel prism, angles exaggerated.
        phi = shown * {0: (u - 1) * PRE / TRIP, 1: u, 2: 1}[stage]
        p0 = np.array([.1, -.1])                    # middle of the face in use, before turning
        c, s_ = np.cos(shown), np.sin(shown)
        a, b = c * p0[0] - s_ * p0[1], s_ * p0[0] + c * p0[1]
        n = np.array([np.cos(-np.pi / 4 + shown), np.sin(-np.pi / 4 + shown)])
        q = np.array([a + (b - p0[1]) * n[1] / n[0], p0[1]])   # where the returning light lands
        src = np.array([p0[0], -.8])
        # Beam splitter: a half-silvered plate at 45 degrees below the mirror. Light from the
        # slit goes through it; the returning light is partly reflected towards the eyepiece.
        g0, ng = np.array([p0[0], -.4]), np.array([1., -1.]) / np.sqrt(2)
        dr = (lambda d: d - 2 * (d @ n) * n)(np.array([-1., 0]))       # back from the mirror
        h = q + ((g0 - q) @ ng) / (dr @ ng) * dr                         # on the beam splitter
        de = dr - 2 * (dr @ ng) * ng                                     # towards the eyepiece
        xe = -.25
        land = h + (xe - h[0]) / de[0] * de                              # image in the eyepiece
        mx.plot(*np.array([src, p0]).T, color=ORANGE, lw=.7, alpha=.25)
        mx.plot([p0[0], .95], [p0[1], p0[1]], color=ORANGE, lw=.7, alpha=.25)
        mx.plot([g0[0], xe], [g0[1], g0[1]], color=GREY, lw=.6, ls=(0, (3, 2)))
        if stage == 0:
            pulse(mx, src, p0, u)
        elif stage == 1 and u < .08:
            pulse(mx, p0, np.array([.95, p0[1]]), u / .08)
        elif stage == 1 and u > .92:
            pulse(mx, np.array([.95, p0[1]]), q, (u - .92) / .08)
        elif stage == 2:
            mx.plot([.95, q[0]], [p0[1], q[1]], color=ORANGE, lw=1, alpha=.5)
            l1, l2 = np.linalg.norm(h - q), np.linalg.norm(land - h)
            r = u * (l1 + l2)
            path = [q, q + min(r, l1) / l1 * (h - q)]
            if r > l1:
                path.append(h + (r - l1) / l2 * (land - h))
            mx.plot(*np.array(path).T, color=ORANGE, lw=2, zorder=6)
            if u > .3:
                angle = np.degrees(np.arctan2(dr[1], dr[0])) % 360
                mx.add_patch(Arc(q, .44, .44, theta1=270, theta2=angle, color=ORANGE, lw=1.4))
                mid = np.radians((270 + angle) / 2)
                mx.text(q[0] + .16, q[1] - .1, r"$2\alpha$",
                        ha="center", va="center", fontsize=12, color=ORANGE)
            if u > .95:
                mx.add_patch(Rectangle((xe - .005, land[1] - .035), .03, .07, facecolor=ORANGE,
                                       lw=0, zorder=7))
                mx.annotate("", (xe - .1, land[1]), (xe - .1, g0[1]),
                            arrowprops=dict(arrowstyle="<|-|>", color=ORANGE, lw=1,
                                            mutation_scale=7, shrinkA=0, shrinkB=0), zorder=7)
        corners = np.array([(.2 * np.cos(np.pi / 2 * k + phi), .2 * np.sin(np.pi / 2 * k + phi))
                            for k in range(4)])
        mx.add_patch(Polygon(corners, closed=True, facecolor=STEEL, edgecolor=GREY, lw=1.1,
                             zorder=4))
        # The angle alpha turned since the light left: a fixed reference and a mark on the prism.
        ref = np.radians(135)
        mx.plot([.24 * np.cos(ref), .55 * np.cos(ref)], [.24 * np.sin(ref), .55 * np.sin(ref)],
                color=GREY, lw=.7, ls=(0, (3, 2)), zorder=5)
        if phi > 0:
            mx.plot([.2 * np.cos(ref + phi), .55 * np.cos(ref + phi)],
                    [.2 * np.sin(ref + phi), .55 * np.sin(ref + phi)], color=GREY, lw=1.1, zorder=5)
            mx.add_patch(Arc((0, 0), .9, .9, theta1=135, theta2=135 + np.degrees(phi),
                             color=ORANGE, lw=1.4, zorder=5))
            mid = ref + phi / 2
            mx.text(.6 * np.cos(mid), .6 * np.sin(mid), r"$\alpha$", ha="center",
                    va="center", fontsize=12, color=ORANGE)
        mx.add_patch(FancyArrowPatch((.3, .14), (.06, .33), connectionstyle="arc3,rad=.45",
                                     arrowstyle="-|>", mutation_scale=9, color=GREY, lw=.8))
        # Slit, beam splitter and eyepiece with its scale.
        mx.add_patch(Rectangle(src - (.07, .025), .14, .05, facecolor=GREY, lw=0))
        mx.text(src[0] + .1, src[1], "slit", ha="left", va="center", fontsize=9)
        mx.plot(*np.array([g0 - (.12, .12), g0 + (.12, .12)]).T, color=BLUE, lw=2.2,
                alpha=.6, solid_capstyle="butt", zorder=5)
        mx.text(g0[0] + .2, g0[1] - .22, "beam\nsplitter", ha="left", va="center", fontsize=9)
        mx.add_patch(Rectangle((xe - .14, -.66), .14, .56, facecolor="white", edgecolor=GREY,
                               lw=.9, zorder=3))
        for k, y in enumerate(np.linspace(-.6, -.2, 9)):
            mx.plot([xe - .04 if k % 4 else xe - .07, xe], [y, y], color=GREY, lw=.6, zorder=4)
        mx.text(xe - .15, -.72, "eyepiece", ha="left", va="top", fontsize=9)
        mx.text(.94, p0[1] + .04, "to the\nMonument", ha="right", va="bottom", fontsize=9)
        mx.text(.94, -.9, r"angles $\times %d$" % exaggerate, fontsize=9, color=GREY,
                ha="right", va="bottom", style="italic")
        mx.text(-.15, .52, "rotating mirror", fontsize=9, va="top")
        mx.set_xlim(-.5, .96)
        mx.set_ylim(-.92, .55)
        mx.set_aspect("equal")
        mx.set_axis_off()

    plan = ([(0, u) for u in np.linspace(0, 1, PRE)] + [(1, u) for u in np.linspace(0, 1, TRIP)]
            + [(2, u) for u in np.linspace(0, 1, POST)])
    durations = [70] * (len(plan) - 1) + [4000]
    images = []
    for stage, u in plan:
        draw(stage, u)
        fig.canvas.draw()
        images.append(Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
                      .convert("RGB").quantize(colors=96, method=Image.Quantize.MEDIANCUT))
    plt.close(fig)
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0,
                   optimize=True)
    print("wrote", path, "(%d frames); alpha = %.2f deg, c = %.0f km/s"
          % (len(images), np.degrees(alpha), 2 * distance / time / 1e3))

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


def hpd(samples, mass=.95):
    """The narrowest interval holding a fraction `mass` of the samples."""
    d = np.sort(samples)
    k = int(np.floor(mass * len(d)))
    j = np.argmin(d[k:] - d[:len(d) - k])
    return d[j], d[j + k]


def intervals_figure(thetas):
    """Posterior histograms of mu and sigma, with the 95% equal-tailed and HPD intervals."""
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.9), dpi=200)
    for ax, values, label in zip(axes, thetas.T, [r"$\mu$", r"$\sigma$"]):
        ax.hist(values, bins=60, density=True, color=LIGHT, lw=0)
        top = ax.get_ylim()[1]
        for (lo, hi), color, name, y in [(np.percentile(values, [2.5, 97.5]), BLUE,
                                          "equal-tailed", 1.06), (hpd(values), ORANGE, "HPD", 1.16)]:
            ax.plot([lo, hi], [y * top] * 2, color=color, lw=3, solid_capstyle="butt",
                    label="95%% %s" % name)
        ax.set_ylim(0, 1.22 * top)
        ax.set_xlabel(label)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        tidy(ax)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, .1, 1, 1))
    save(fig, "newcomb-intervals.png")
    for values, name in zip(thetas.T, ["mu", "sigma"]):
        print("  %s: equal-tailed [%.2f, %.2f], HPD [%.2f, %.2f]"
              % (name, *np.percentile(values, [2.5, 97.5]), *hpd(values)))


def predictive_figure(gaussian, student):
    """The data against the posterior predictive densities of both models, on a log scale."""
    xs = np.linspace(-60, 60, 600)
    pg = norm.pdf(xs[:, None], gaussian[:, 0], gaussian[:, 1]).mean(1)
    pt = t.pdf(xs[:, None], student[:, 2], student[:, 0], student[:, 1]).mean(1)
    fig, ax = plt.subplots(figsize=(W, 2.9), dpi=200)
    ax.hist(NEWCOMB, bins=np.arange(-60, 62, 2), density=True, color=LIGHT, lw=0,
            label="measurements")
    ax.plot(xs, pg, color=BLUE, lw=1.6, label="Gaussian model")
    ax.plot(xs, pt, color=GREEN, lw=1.6, label="$t$ model")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, .3)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel("predictive density")
    ax.legend(frameon=False, loc="upper left")
    tidy(ax)
    fig.tight_layout()
    save(fig, "newcomb-predictive.png")


def laplace_figure(student):
    """Posterior of log nu under the t model, against the marginal of the Laplace approximation."""
    from numdifftools import Hessian
    neg = lambda th: -log_posterior_t(th, NEWCOMB)  # noqa: E731
    mode = minimize(neg, [27., 4., 1.], method="Nelder-Mead",
                    options=dict(xatol=1e-8, fatol=1e-10, maxiter=20000)).x
    cov = np.linalg.inv(Hessian(neg)(mode))
    values = np.log(student[:, 2])
    xs = np.linspace(-.5, 3, 400)
    fig, ax = plt.subplots(figsize=(W, 2.9), dpi=200)
    ax.hist(values, bins=60, range=(-.5, 3), density=True, color=LIGHT, lw=0,
            label="posterior samples (MCMC)")
    ax.plot(xs, norm.pdf(xs, mode[2], np.sqrt(cov[2, 2])), color=BLUE, lw=1.6,
            label="Laplace approximation")
    ax.axvline(mode[2], color=GREY, lw=1, ls=(0, (4, 3)), label="mode")
    ax.set_xlabel(r"$\log \nu$")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    tidy(ax)
    fig.tight_layout()
    save(fig, "laplace.png")


def occam():
    """MacKay's picture: the marginal likelihood of a simple and of a flexible model, over all
    possible data sets."""
    xs = np.linspace(0, 10, 500)
    simple = norm.pdf(xs, 3, .9)
    flexible = .5 * norm.pdf(xs, 3.5, 2.6) + .5 * norm.pdf(xs, 6.5, 2.6)
    fig, ax = plt.subplots(figsize=(W, 2.6), dpi=200)
    ax.plot(xs, simple, color=BLUE, lw=1.8, label="simple model")
    ax.plot(xs, flexible, color=GREEN, lw=1.8, label="flexible model")
    ax.axvline(3.4, color=ORANGE, lw=1.6)
    ax.text(3.55, .43, r"$\mathbf{x}_\mathrm{obs}$", color=ORANGE)
    ax.set_xlabel("all possible data sets")
    ax.set_ylabel(r"$p(\mathbf{x} \mid \mathcal{M})$")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, .48)
    ax.legend(frameon=False, loc="upper right")
    tidy(ax)
    fig.tight_layout()
    save(fig, "occam.png")


def elpd_figure(k=5, seed=42):
    """Pointwise log predictive densities of 5-fold cross-validation, t model minus Gaussian."""
    from scipy.special import logsumexp
    from sklearn.model_selection import KFold
    lppd = np.zeros((2, len(NEWCOMB)))
    for train, test in KFold(n_splits=k, shuffle=True, random_state=seed).split(NEWCOMB):
        g = sample(log_posterior_gaussian, [26., 10.], NEWCOMB[train], steps=6000)
        s = sample(log_posterior_t, [26., 5., np.log(3.)], NEWCOMB[train], steps=6000)
        x = NEWCOMB[test][None, :]
        ll = [norm.logpdf(x, g[:, :1], g[:, 1:2]),
              t.logpdf(x, np.exp(s[:, 2:3]), s[:, :1], s[:, 1:2])]
        for j in range(2):
            lppd[j, test] = logsumexp(ll[j], axis=0) - np.log(len(ll[j]))
    diff = lppd[1] - lppd[0]
    print("  ELPD: Gaussian %.0f, t %.0f; two lowest points account for %.0f of %.0f"
          % (lppd[0].sum(), lppd[1].sum(), np.sort(diff)[-2:].sum(), diff.sum()))
    fig, ax = plt.subplots(figsize=(W, 2.9), dpi=200)
    ax.axhline(0, color=LIGHT, lw=1)
    ax.scatter(NEWCOMB, diff, s=22, color=BLUE, zorder=3)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel("$t$ minus Gaussian")
    ax.set_title("log predictive density of each held-out measurement")
    tidy(ax)
    fig.tight_layout()
    save(fig, "elpd-pointwise.png")


def pvalue_figure(mu=24., sigma=10.8, n=20000, seed=2):
    """The ordinary p-value: the mean of data sets simulated with theta fixed, against the
    observed mean, with the tail beyond it shaded."""
    rng = np.random.default_rng(seed)
    means = (mu + sigma * rng.standard_normal((n, len(NEWCOMB)))).mean(1)
    observed = NEWCOMB.mean()
    p = np.mean(means >= observed)
    bins = np.linspace(mu - 4.5 * sigma / np.sqrt(len(NEWCOMB)),
                       mu + 4.5 * sigma / np.sqrt(len(NEWCOMB)), 61)
    fig, ax = plt.subplots(figsize=(W, 2.8), dpi=200)
    counts, edges, patches = ax.hist(means, bins=bins, density=True, color=BLUE, lw=0,
                                     label=r"simulated data sets, $\mu = %g$, $\sigma = %g$"
                                     % (mu, sigma))
    for left, patch in zip(edges[:-1], patches):
        if left >= observed:
            patch.set_facecolor(ORANGE)
    ax.axvline(observed, color=GREY, lw=1.6, label="observed data")
    ax.text(observed + .25, .6 * counts.max(), "$p = %.2f$" % p, color=ORANGE)
    ax.set_ylim(0, 1.45 * counts.max())
    ax.set_xlabel(r"mean $T(\mathbf{x})$ (ns above 24800 ns)")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    tidy(ax)
    fig.tight_layout()
    save(fig, "pvalue.png")
    print("  p-value of the mean at mu = %g, sigma = %g: %.3f" % (mu, sigma, p))


def newcomb():
    data_figure()
    pvalue_figure()
    thetas = sample(log_posterior_gaussian, [26., 10.], NEWCOMB)
    print("  Gaussian model: %d draws, mu %.2f (sd %.2f), 95%% equal-tailed [%.2f, %.2f]"
          % (len(thetas), thetas[:, 0].mean(), thetas[:, 0].std(),
             *np.percentile(thetas[:, 0], [2.5, 97.5])))
    corner_plot(thetas, [r"$\mu$", r"$\sigma$"], "newcomb-posterior.png", 45)
    intervals_figure(thetas)
    gaussian = thetas

    rng = np.random.default_rng(1)
    draws = thetas[rng.choice(len(thetas), 4000, replace=False)]
    reps = replicate(draws, rng)
    replicates_figure(reps)
    statistics_figures(reps)
    residuals_figures(draws, reps)

    thetas = sample(log_posterior_t, [26., 5., np.log(3.)], NEWCOMB)
    thetas[:, 2] = np.exp(thetas[:, 2])
    predictive_figure(gaussian, thetas)
    laplace_figure(thetas)
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
    newcomb_animation()
    newcomb()
    polynomials()
    occam()
    kfold()
    elpd_figure()
