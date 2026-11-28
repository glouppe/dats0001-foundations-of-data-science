"""Figures of lecture 9: the graphical model, in the style and at the scale of lectures 4 to 8,
and the plots of the lecture. Every plot is W inches wide, shown at .width-70, as in lectures 7
and 8.

- the Gaussian fitted to a two-mode distribution by the forward and by the reverse KL divergence;
- the mean-field Gaussian that maximizes the ELBO for a correlated Gaussian posterior, computed
  exactly (each factor has variance 1 / Lambda_jj, Lambda the precision matrix);
- the projectile of nb01 as the guiding example: posterior trajectories and the posterior of
  (v, alpha) by emcee; ADVI with a mean-field and a full-rank Gaussian; amortized VI with a
  conditional normalizing flow, for three landing distances.

Usage: uv run python scripts/figures_lec9.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import norm  # noqa: E402

from figures_lec4 import (GREY, MARGIN, STEP, UNIT, arrow, figure, measure,  # noqa: E402
                          node, rescale)

WIDTH = 246.5568 / 72 / UNIT    # canvas width of the lecture 4 figures, in drawing units
LIGHT = "#b8c0c6"
BLUE = "#0173b2"
ORANGE = "#de8f05"
GREEN = "#029e73"
W = 6.4

plt.rcParams.update({"font.size": 12, "legend.fontsize": 10, "axes.titlesize": 12,
                     "text.color": GREY, "axes.labelcolor": GREY,
                     "xtick.color": GREY, "ytick.color": GREY, "axes.edgecolor": LIGHT,
                     "mathtext.fontset": "cm"})


def tidy(ax):
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)


def save(fig, name):
    path = "figures/lec9/%s" % name
    fig.savefig(path, facecolor="white", dpi=200)
    plt.close(fig)
    print("wrote", path)


# Graphical model ------------------------------------------------------------------------

def graphical_model():
    """Latent variables z, which gather all the unknowns, and observed variables x."""
    fig, ax = figure()
    node(ax, (0, STEP), r"$\mathbf{z}$")
    node(ax, (0, 0), r"$\mathbf{x}$", observed=True)
    arrow(ax, (0, STEP), (0, 0))
    x0, y0, x1, y1 = measure(fig, ax)
    width = max(WIDTH, x1 - x0 + 2 * MARGIN)
    centre = (x0 + x1) / 2
    rescale(fig, ax, centre - width / 2, centre + width / 2, y0 - MARGIN, y1 + MARGIN)
    path = "figures/lec9/lvm.svg"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("wrote %-30s %.2f x %.2f in" % (path, *fig.get_size_inches()))


# Forward and reverse KL ------------------------------------------------------------------

def kl_directions():
    """A two-mode p and the Gaussians q that minimize KL(p || q) and KL(q || p), on a grid."""
    zs = np.linspace(-6, 6, 4001)
    dz = zs[1] - zs[0]
    p = .5 * norm.pdf(zs, -2, .6) + .5 * norm.pdf(zs, 2, .6)
    mean = np.sum(zs * p) * dz                                 # forward KL: moment matching
    sd = np.sqrt(np.sum((zs - mean) ** 2 * p) * dz)

    def reverse(params):
        q = norm.pdf(zs, params[0], np.exp(params[1]))
        return np.sum(q * (np.log(q + 1e-300) - np.log(p + 1e-300))) * dz

    m, log_s = minimize(reverse, [1., 0.], method="Nelder-Mead").x
    fig, ax = plt.subplots(figsize=(W, 2.6), dpi=200)
    ax.fill_between(zs, p, color=LIGHT, lw=0, label=r"$p$")
    ax.plot(zs, norm.pdf(zs, mean, sd), color=ORANGE, lw=1.8,
            label=r"$\arg\min_q \, \mathrm{KL}(p \,\|\, q)$")
    ax.plot(zs, norm.pdf(zs, m, np.exp(log_s)), color=BLUE, lw=1.8,
            label=r"$\arg\min_q \, \mathrm{KL}(q \,\|\, p)$")
    ax.set_xlabel(r"$z$")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-5.5, 6)
    ax.legend(frameon=False, loc="upper right")
    tidy(ax)
    fig.tight_layout()
    save(fig, "kl-directions.png")


# The variational family ------------------------------------------------------------------

def variational_family():
    """The family Q as a region of the space of distributions over z, the posterior outside it,
    and level sets of the KL divergence to the posterior (circles, schematically). Gradient
    steps on phi move q from q(z; phi_0) to q(z; phi*), where a level set touches Q."""
    centre, a, b = np.array([-.6, 0.]), 1.7, .95
    post = np.array([1.9, .5])
    ts = np.linspace(0, 2 * np.pi, 2000)
    rim = centre + np.column_stack([a * np.cos(ts), b * np.sin(ts)])
    star = rim[np.argmin(np.linalg.norm(rim - post, axis=1))]
    d = np.linalg.norm(star - post)
    fig, ax = plt.subplots(figsize=(W, 2.6), dpi=200)
    for k, r in enumerate([d, d + .45, d + .9]):
        ax.add_patch(plt.Circle(post, r, facecolor="none", edgecolor=ORANGE, lw=1.,
                                ls=(0, (3, 2)), alpha=1 - .3 * k))
    ax.add_patch(Ellipse(centre, 2 * a, 2 * b, facecolor=BLUE, alpha=.12, lw=0))
    ax.add_patch(Ellipse(centre, 2 * a, 2 * b, facecolor="none", edgecolor=BLUE, lw=1.4))
    ax.text(centre[0] - .95 * a, centre[1] + .75 * b, r"$\mathcal{Q}$", color=BLUE, fontsize=16)
    start = np.array([-1.7, -.45])
    control = star - 1.2 * (post - star) / d                  # arrive along the gradient
    w = np.linspace(0, 1, 100)[:, None]
    path = (1 - w) ** 2 * start + 2 * w * (1 - w) * control + w ** 2 * star
    ax.plot(*path.T, color=BLUE, lw=1.4)
    ax.plot(*start, "o", color=BLUE, ms=5, mfc="white")
    ax.text(*(start + [0, .14]), r"$q(\mathbf{z}; \phi_0)$", color=BLUE, ha="center", va="bottom")
    ax.plot(*star, "o", color=BLUE, ms=7)
    ax.text(*(star + [-.08, .1]), r"$q(\mathbf{z}; \phi^*)$", color=BLUE, ha="right", va="bottom")
    ax.plot(*post, "o", color=GREY, ms=7)
    ax.text(*(post + [.12, .12]), r"$p(\mathbf{z} \mid \mathbf{x})$", color=GREY)
    ax.set_xlim(-2.6, 3.4)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.tight_layout()
    save(fig, "variational-family.png")


# Mean field ------------------------------------------------------------------------------

def gaussian_ellipse(ax, cov, color, label, ls="-"):
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    for k in (1, 2):
        ax.add_patch(Ellipse((0, 0), 2 * k * np.sqrt(vals[1]), 2 * k * np.sqrt(vals[0]),
                             angle=angle, facecolor="none", edgecolor=color, lw=1.6, ls=ls,
                             label=label if k == 1 else None))


def mean_field():
    """A correlated Gaussian posterior and the mean-field Gaussian that maximizes the ELBO."""
    cov = np.array([[1., .9], [.9, 1.]])
    precision = np.linalg.inv(cov)
    q = np.diag(1 / np.diag(precision))
    fig, ax = plt.subplots(figsize=(W, 3.), dpi=200)
    gaussian_ellipse(ax, cov, GREY, r"posterior $p(\mathbf{z} \mid \mathbf{x})$")
    gaussian_ellipse(ax, q, BLUE, r"mean field $q(z_1) \, q(z_2)$")
    ax.set_aspect("equal")
    ax.set_xlim(-6.5, 4.2)
    ax.set_ylim(-2.4, 2.4)
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, loc="upper left")
    tidy(ax)
    fig.tight_layout()
    save(fig, "mean-field.png")
    print("  mean field: sd %.2f against marginal sd 1" % np.sqrt(q[0, 0]))


# Change of variables ---------------------------------------------------------------------

def change_of_variables():
    """u ~ N(0, 1) and z = f(u) = u + 0.8 tanh(2u), drawn as in a textbook: f in the middle,
    p(u) below it and p(z) on its left. Two intervals of u and their images under f carry the
    same mass: where f is steep (around 0), the image is wider and the density of z lower."""
    f = lambda u: u + .8 * np.tanh(2 * u)
    df = lambda u: 1 + 1.6 / np.cosh(2 * u) ** 2
    us = np.linspace(-3, 3, 1000)
    fig = plt.figure(figsize=(W, 3.6), dpi=200)
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 2.6], height_ratios=[2.4, 1],
                            wspace=.05, hspace=.08)
    curve = fig.add_subplot(grid[0, 1])
    below = fig.add_subplot(grid[1, 1], sharex=curve)
    left = fig.add_subplot(grid[0, 0], sharey=curve)
    curve.plot(us, f(us), color=GREY, lw=1.6)
    curve.text(2.1, f(2.1) - .9, r"$z = f(u)$", color=GREY, ha="left")
    below.plot(us, norm.pdf(us), color=GREY, lw=1.6)
    left.plot(norm.pdf(us) / df(us), f(us), color=BLUE, lw=1.6)
    for (a, b), color in [((-.2, .2), ORANGE), ((1.2, 1.6), GREEN)]:
        seg = np.linspace(a, b, 50)
        below.fill_between(seg, norm.pdf(seg), color=color, alpha=.6, lw=0)
        left.fill_betweenx(f(seg), norm.pdf(seg) / df(seg), color=color, alpha=.6, lw=0)
        curve.fill_between(seg, -4, f(seg), color=color, alpha=.15, lw=0)
        curve.fill_betweenx(f(seg), -3, seg, color=color, alpha=.15, lw=0)
    curve.set_xlim(-3, 3)
    curve.set_ylim(f(-3), f(3))
    left.set_xlim(.45, 0)
    below.set_ylim(0, .45)
    below.set_xlabel(r"$u$")
    left.set_ylabel(r"$z$")
    below.text(-2.9, .3, r"$p(u)$", color=GREY)
    left.text(.43, f(2.6), r"$p(z)$", color=BLUE)
    for ax in (curve, below, left):
        tidy(ax)
    curve.tick_params(labelbottom=False, labelleft=False)
    below.set_yticks([])
    left.set_xticks([])
    curve.spines["left"].set_visible(False)
    curve.spines["bottom"].set_visible(False)
    curve.tick_params(length=0)
    left.spines["left"].set_visible(False)
    left.spines["right"].set_visible(True)
    left.yaxis.tick_right()
    left.tick_params(axis="y", labelright=False, length=0)
    left.yaxis.set_label_position("left")
    fig.subplots_adjust(left=.06, right=.98, top=.97, bottom=.14)
    save(fig, "change-of-variables.png")


def change_of_variables_2d():
    """A grid of the u plane and its image under a nonlinear f. A small square of area dV around
    u goes to a curved quadrilateral, close to the parallelogram spanned by the columns of the
    Jacobian J_f(u) times the sides of the square, of area |det J_f(u)| dV."""
    f = lambda u1, u2: (u1 + .35 * np.sin(1.3 * u2), u2 + .25 * u1 ** 2)
    def jac(u1, u2):
        return np.array([[1., .35 * 1.3 * np.cos(1.3 * u2)], [.5 * u1, 1.]])
    fig, (left, right) = plt.subplots(1, 2, figsize=(W, 2.8), dpi=200)
    ticks = np.linspace(-2, 2, 9)
    t = np.linspace(-2, 2, 200)
    for c in ticks:
        left.plot(t, np.full_like(t, c), color=LIGHT, lw=.8)
        left.plot(np.full_like(t, c), t, color=LIGHT, lw=.8)
        right.plot(*f(t, np.full_like(t, c)), color=LIGHT, lw=.8)
        right.plot(*f(np.full_like(t, c), t), color=LIGHT, lw=.8)
    u0, h = np.array([.5, .5]), .5
    sq = np.array([[0, 0], [h, 0], [h, h], [0, h], [0, 0]]) + u0
    left.fill(*sq.T, color=ORANGE, alpha=.6, lw=0)
    side = np.concatenate([np.linspace(sq[k], sq[k + 1], 30) for k in range(4)])
    right.fill(*f(*side.T), color=ORANGE, alpha=.6, lw=0)
    J = jac(*u0)
    par = np.array([[0, 0], J @ [h, 0], J @ [h, h], J @ [0, h], [0, 0]]) + np.array(f(*u0))
    right.plot(*par.T, color=BLUE, lw=1.4, ls=(0, (3, 2)))
    right.text(par[1, 0] + .15, par[1, 1] + .05, r"$|\det J_f(\mathbf{u})| \, dV$", color=BLUE,
               va="center", bbox=dict(facecolor="white", edgecolor="none", pad=1))
    left.text(u0[0] + h + .1, u0[1] + h / 2, r"$dV$", color=ORANGE, va="center",
              bbox=dict(facecolor="white", edgecolor="none", pad=1))
    left.set_title(r"$\mathbf{u}$")
    right.set_title(r"$\mathbf{z} = f(\mathbf{u})$")
    for ax in (left, right):
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for side_ in ("left", "bottom", "right", "top"):
            ax.spines[side_].set_visible(False)
    fig.tight_layout()
    save(fig, "change-of-variables-2d.png")


def normalizing_flow():
    """z_0 ~ N(0, 1) through two invertible functions: the density after each one, by the change
    of variables, divided by the derivative of each function along the way."""
    f1 = lambda u: u + .8 * np.tanh(2 * u)
    df1 = lambda u: 1 + 1.6 / np.cosh(2 * u) ** 2
    f2 = lambda u: u + .9 * np.tanh(3 * (u - 1.5))
    df2 = lambda u: 1 + 2.7 / np.cosh(3 * (u - 1.5)) ** 2
    z0 = np.linspace(-3.5, 3.5, 2000)
    d0 = norm.pdf(z0)
    z1, d1 = f1(z0), d0 / df1(z0)
    z2, d2 = f2(z1), d1 / df2(z1)
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.2), dpi=200, sharey=True)
    for ax, (z, d), title in zip(axes, [(z0, d0), (z1, d1), (z2, d2)],
                                 [r"$p(z_0)$", r"$q(z_1)$, $z_1 = g_1(z_0)$", r"$q(z_2)$, $z_2 = g_2(z_1)$"]):
        ax.fill_between(z, d, color=BLUE, alpha=.25, lw=0)
        ax.plot(z, d, color=BLUE, lw=1.6)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        tidy(ax)
    fig.tight_layout()
    save(fig, "normalizing-flow.png")


def advi_transform():
    """A posterior on z > 0 (a Gamma), its version in zeta = log z, the Gaussian in zeta that
    minimizes the reverse KL, and that Gaussian mapped back to z."""
    from scipy.stats import gamma
    zetas = np.linspace(-4, 3, 3001)
    dzeta = zetas[1] - zetas[0]
    p_zeta = gamma.pdf(np.exp(zetas), 2.) * np.exp(zetas)

    def reverse(params):
        q = norm.pdf(zetas, params[0], np.exp(params[1]))
        return np.sum(q * (np.log(q + 1e-300) - np.log(p_zeta + 1e-300))) * dzeta

    m, log_s = minimize(reverse, [0., 0.], method="Nelder-Mead").x
    zs = np.linspace(1e-3, 7, 2000)
    fig, (left, right) = plt.subplots(1, 2, figsize=(W, 2.6), dpi=200)
    left.fill_between(zs, gamma.pdf(zs, 2.), color=LIGHT, lw=0,
                      label=r"posterior $p(z \mid \mathbf{x})$")
    left.plot(zs, norm.pdf(np.log(zs), m, np.exp(log_s)) / zs, color=BLUE, lw=1.8,
              label=r"$q$")
    left.set_xlabel(r"$z > 0$")
    right.fill_between(zetas, p_zeta, color=LIGHT, lw=0)
    right.plot(zetas, norm.pdf(zetas, m, np.exp(log_s)), color=BLUE, lw=1.8)
    right.set_xlabel(r"$\zeta = \log z$")
    left.legend(frameon=False, loc="upper right")
    left.set_title("constrained")
    right.set_title("unconstrained, where $q$ is Gaussian")
    for ax in (left, right):
        ax.set_yticks([])
        tidy(ax)
    fig.tight_layout()
    save(fig, "advi-transform.png")


# The projectile ---------------------------------------------------------------------------
# The ball of nb01, thrown from 1.5 m with speed v and angle alpha, without drag. Its landing
# distance has a closed form, equal to nb01's fine Euler steps; the noise sd is 0.5 m and the
# priors are uniform, v on (0, 10) and alpha on (0, pi/2). The variational families live in
# zeta = (logit(v / 10), logit(alpha / (pi / 2))), as in ADVI.

G, H, SIGMA, V_MAX, A_MAX = 9.81, 1.5, .5, 10., np.pi / 2


def landing(v, alpha, xp=np):
    return v * xp.cos(alpha) / G * (v * xp.sin(alpha) + xp.sqrt((v * xp.sin(alpha)) ** 2 + 2 * G * H))


def flight(v, alpha, n=60):
    t_end = (v * np.sin(alpha) + np.sqrt((v * np.sin(alpha)) ** 2 + 2 * G * H)) / G
    t = np.linspace(0, t_end, n)
    return v * np.cos(alpha) * t, H + v * np.sin(alpha) * t - G * t ** 2 / 2


def x_observed():
    return landing(5., np.pi / 4) + SIGMA * np.random.default_rng(1).standard_normal()


def mcmc_projectile(x_obs, seed=0, steps=6000):
    import emcee

    def log_posterior(theta):
        v, alpha = theta
        if not (0 < v < V_MAX and 0 < alpha < A_MAX):
            return -np.inf
        return norm.logpdf(x_obs, landing(v, alpha), SIGMA)

    rng = np.random.default_rng(seed)
    start = np.column_stack([rng.uniform(1, 9, 32), rng.uniform(.1, 1.4, 32)])
    sampler = emcee.EnsembleSampler(32, 2, log_posterior)
    sampler.random_state = np.random.RandomState(seed).get_state()
    sampler.run_mcmc(start, steps)
    return sampler.get_chain(discard=1000, thin=10, flat=True)


def projectile_axes(ax):
    ax.set_xlabel(r"$v$ (m/s)")
    ax.set_ylabel(r"$\alpha$ (rad)")
    ax.set_xlim(0, V_MAX)
    ax.set_ylim(0, A_MAX)
    tidy(ax)


def projectile_setup():
    """Left: trajectories drawn from the posterior, all landing near x_obs. Right: the posterior
    of (v, alpha), sampled by MCMC."""
    x_obs = x_observed()
    post = mcmc_projectile(x_obs)
    print("  projectile: x_obs %.2f m, %d MCMC draws" % (x_obs, len(post)))
    rng = np.random.default_rng(0)
    fig, (left, right) = plt.subplots(1, 2, figsize=(W, 2.8), dpi=200,
                                      gridspec_kw=dict(width_ratios=[1.3, 1]))
    for v, alpha in post[rng.choice(len(post), 40, replace=False)]:
        left.plot(*flight(v, alpha), color=BLUE, lw=.8, alpha=.5)
    left.plot([x_obs], [0], "o", color=ORANGE, ms=6, zorder=3)
    left.text(x_obs, -.35, r"$x_\mathrm{obs}$", color=ORANGE, ha="center", va="top")
    left.axhline(0, color=LIGHT, lw=1)
    left.set_xlabel("distance (m)")
    left.set_ylabel("height (m)")
    left.set_ylim(-1.1, None)
    left.spines["left"].set_bounds(0, 5)
    left.set_yticks([0, 2, 4])
    tidy(left)
    right.scatter(*post[rng.choice(len(post), 4000, replace=False)].T, s=3, color=GREY, lw=0,
                  alpha=.4)
    projectile_axes(right)
    right.set_title(r"$p(v, \alpha \mid x_\mathrm{obs})$")
    fig.tight_layout()
    save(fig, "projectile.png")


def adam(grad, params, steps, lr=.01, seed=0):
    """Plain Adam on a pytree of parameters, with a fresh key at every step."""
    import jax
    import jax.numpy as jnp
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    key = jax.random.PRNGKey(seed)
    for t in range(1, steps + 1):
        key, sub = jax.random.split(key)
        g = grad(params, sub)
        m = jax.tree_util.tree_map(lambda a, b: .9 * a + .1 * b, m, g)
        v = jax.tree_util.tree_map(lambda a, b: .999 * a + .001 * b ** 2, v, g)
        params = jax.tree_util.tree_map(
            lambda p, a, b: p + lr * (a / (1 - .9 ** t)) / (jnp.sqrt(b / (1 - .999 ** t)) + 1e-8),
            params, m, v)
    return params


def log_joint_zeta(zeta, x, jnp):
    """log p(x, zeta), with the uniform priors carried to zeta by the logistic maps."""
    import jax
    s = jax.nn.sigmoid(zeta)
    v, alpha = V_MAX * s[..., 0], A_MAX * s[..., 1]
    log_jac = jnp.sum(jnp.log(s) + jnp.log1p(-s), -1) + np.log(V_MAX * A_MAX)
    log_prior = -np.log(V_MAX * A_MAX)
    mean = landing(v, alpha, jnp)
    return log_prior + log_jac - .5 * ((x - mean) / SIGMA) ** 2 - np.log(SIGMA * np.sqrt(2 * np.pi))


def to_theta(zeta):
    s = 1 / (1 + np.exp(-zeta))
    return np.column_stack([V_MAX * s[:, 0], A_MAX * s[:, 1]])


def advi_projectile():
    """ADVI on x_obs with a mean-field and a full-rank Gaussian in zeta, against MCMC."""
    import jax
    import jax.numpy as jnp
    x_obs = x_observed()
    post = mcmc_projectile(x_obs)
    fits = {}
    for name in ("mean field", "full rank"):
        def elbo(params, key, full=name == "full rank"):
            eps = jax.random.normal(key, (64, 2))
            L = jnp.diag(jnp.exp(params["log_s"]))
            if full:
                L = L + jnp.tril(params["off"], -1)
            zeta = params["mu"] + eps @ L.T
            entropy = jnp.sum(params["log_s"])
            return jnp.mean(log_joint_zeta(zeta, x_obs, jnp)) + entropy
        grad = jax.jit(jax.grad(elbo))
        params = {"mu": jnp.zeros(2), "log_s": jnp.zeros(2), "off": jnp.zeros((2, 2))}
        params = adam(grad, params, 4000, lr=.02)
        L = np.diag(np.exp(np.asarray(params["log_s"])))
        if name == "full rank":
            L = L + np.tril(np.asarray(params["off"]), -1)
        eps = np.random.default_rng(0).standard_normal((3000, 2))
        fits[name] = to_theta(np.asarray(params["mu"]) + eps @ L.T)
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.8), dpi=200, sharey=True)
    post = post[np.random.default_rng(0).choice(len(post), 4000, replace=False)]
    for ax, (name, q) in zip(axes, fits.items()):
        ax.scatter(*post.T, s=3, color=LIGHT, lw=0, label="MCMC")
        ax.scatter(*q.T, s=3, color=BLUE, lw=0, alpha=.5, label="ADVI, " + name)
        projectile_axes(ax)
        ax.set_title("ADVI, " + name)
    axes[1].set_ylabel("")
    fig.tight_layout()
    save(fig, "advi-projectile.png")


def amortized_flow(xs=(2., 4., 6.)):
    """Amortized VI with a conditional normalizing flow q(zeta | x): eight affine coupling layers
    whose scale and shift are small networks of (the other coordinate, x). Trained with the
    ELBO averaged over landing distances drawn from the prior predictive, then evaluated at three
    distances against MCMC."""
    import jax
    import jax.numpy as jnp

    K, HIDDEN = 8, 64
    keys = jax.random.split(jax.random.PRNGKey(0), K)

    def init_layer(key):
        k1, k2, k3 = jax.random.split(key, 3)
        return {"W1": jax.random.normal(k1, (2, HIDDEN)) * .3, "b1": jnp.zeros(HIDDEN),
                "W2": jax.random.normal(k2, (HIDDEN, HIDDEN)) * .1, "b2": jnp.zeros(HIDDEN),
                "W3": jax.random.normal(k3, (HIDDEN, 2)) * .01, "b3": jnp.zeros(2)}

    params = [init_layer(k) for k in keys]

    def net(p, a, x):
        h = jnp.tanh(jnp.stack([a, (x - 4.) / 2.], -1) @ p["W1"] + p["b1"])
        h = jnp.tanh(h @ p["W2"] + p["b2"])
        out = h @ p["W3"] + p["b3"]
        return 2. * jnp.tanh(out[..., 0] / 2.), out[..., 1]

    def flow(params, z0, x):
        """z0 -> zeta, and log |det J| of the whole map."""
        z, log_det = z0, 0.
        for k, p in enumerate(params):
            i, j = (0, 1) if k % 2 == 0 else (1, 0)
            log_scale, shift = net(p, z[..., i], x)
            zj = z[..., j] * jnp.exp(log_scale) + shift
            z = z.at[..., j].set(zj)
            log_det = log_det + log_scale
        return z, log_det

    def elbo(params, key):
        k1, k2, k3 = jax.random.split(key, 3)
        theta = jax.random.uniform(k1, (256, 2)) * jnp.array([V_MAX, A_MAX])
        x = landing(theta[:, 0], theta[:, 1], jnp) + SIGMA * jax.random.normal(k2, (256,))
        z0 = jax.random.normal(k3, (256, 2))
        zeta, log_det = flow(params, z0, x)
        log_q = jnp.sum(-.5 * z0 ** 2, -1) - np.log(2 * np.pi) - log_det
        return jnp.mean(log_joint_zeta(zeta, x, jnp) - log_q)

    grad = jax.jit(jax.grad(elbo))
    params = adam(grad, params, 20000, lr=1e-3)
    final = float(np.mean([elbo(params, k) for k in jax.random.split(jax.random.PRNGKey(9), 20)]))
    print("  amortized flow: ELBO %.3f" % final)

    fig, axes = plt.subplots(1, len(xs), figsize=(W, 2.4), dpi=200, sharey=True)
    for ax, x in zip(axes, xs):
        post = mcmc_projectile(x, steps=4000)
        z0 = np.random.default_rng(0).standard_normal((2000, 2))
        zeta, _ = flow(params, jnp.asarray(z0), jnp.full(2000, x))
        q = to_theta(np.asarray(zeta))
        ax.scatter(*post[np.random.default_rng(0).choice(len(post), 3000, replace=False)].T, s=2,
                   color=LIGHT, lw=0)
        ax.scatter(*q.T, s=2, color=BLUE, lw=0, alpha=.4)
        projectile_axes(ax)
        ax.set_title(r"$x = %g$ m" % x)
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.tight_layout()
    save(fig, "amortized-flow.png")

if __name__ == "__main__":
    graphical_model()
    kl_directions()
    variational_family()
    mean_field()
    change_of_variables()
    change_of_variables_2d()
    normalizing_flow()
    advi_transform()
    projectile_setup()
    advi_projectile()
    amortized_flow()
