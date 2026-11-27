"""Figures of lecture 9: the graphical model, in the style and at the scale of lectures 4 to 8,
and the plots of the lecture. Every plot is W inches wide, shown at .width-70, as in lectures 7
and 8.

- the Gaussian fitted to a two-mode distribution by the forward and by the reverse KL divergence;
- the mean-field Gaussian that maximizes the ELBO for a correlated Gaussian posterior, computed
  exactly (each factor has variance 1 / Lambda_jj, Lambda the precision matrix);
- ADVI with a mean-field Gaussian against emcee, on the projectile of nb01, with the model,
  observation and optimization of nb09-advi (Euler steps of 0.1 s, seed 777).

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
    """u ~ N(0, 1) and z = f(u) = u + 0.8 tanh(2u). Two intervals of u and their images carry
    the same mass: where f stretches (around 0), the density of z is lower."""
    f = lambda u: u + .8 * np.tanh(2 * u)
    df = lambda u: 1 + 1.6 / np.cosh(2 * u) ** 2
    us = np.linspace(-3.5, 3.5, 1000)
    fig, (left, right) = plt.subplots(1, 2, figsize=(W, 2.6), dpi=200, sharey=True)
    left.plot(us, norm.pdf(us), color=GREY, lw=1.6)
    right.plot(f(us), norm.pdf(us) / df(us), color=BLUE, lw=1.6)
    for (a, b), color in [((-.15, .15), ORANGE), ((1.2, 1.5), GREEN)]:
        seg = np.linspace(a, b, 50)
        left.fill_between(seg, norm.pdf(seg), color=color, alpha=.5, lw=0)
        right.fill_between(f(seg), norm.pdf(seg) / df(seg), color=color, alpha=.5, lw=0)
    left.set_title(r"$p(u)$")
    right.set_title(r"$p(z) = p(u) \, |f'(u)|^{-1}$")
    left.set_xlabel(r"$u$")
    right.set_xlabel(r"$z = f(u)$")
    for ax in (left, right):
        ax.set_yticks([])
        ax.set_xlim(-3.5, 3.5) if ax is left else ax.set_xlim(f(-3.5), f(3.5))
        tidy(ax)
    fig.tight_layout()
    save(fig, "change-of-variables.png")


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
                                 [r"$p(z_0)$", r"$q(z_1)$, $z_1 = f_1(z_0)$", r"$q(z_2)$, $z_2 = f_2(z_1)$"]):
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


# ADVI on the projectile ------------------------------------------------------------------

def trajectory_end(v, alpha, dt=0.1, G=9.81):
    """Where the ball lands, with the Euler steps of nb09-advi."""
    v_x, v_y = v * np.cos(alpha), v * np.sin(alpha)
    x, y = 0., 1.5
    while y > 0:
        v_y += dt * -G
        x += dt * v_x
        y += dt * v_y
    return x


def advi_projectile():
    import emcee
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsc

    def landing(v, alpha, dt=0.1, G=9.81):
        v_x, v_y = v * jnp.cos(alpha), v * jnp.sin(alpha)
        x, y = 0., 1.5
        while y > 0:
            v_y += dt * -G
            x += dt * v_x
            y += dt * v_y
        return x

    sigmoid = jax.nn.sigmoid

    def elbo(vp, x_obs, eps):
        zeta = vp[0] + jnp.exp(vp[1]) * eps
        v = 10. * sigmoid(zeta[0])
        alpha = np.pi / 2 * sigmoid(zeta[1])
        out = jnp.log(1 / 10.) + jnp.log(10. * sigmoid(zeta[0]) * (1 - sigmoid(zeta[0])))
        out += jnp.log(2 / np.pi) + jnp.log(np.pi / 2 * sigmoid(zeta[1]) * (1 - sigmoid(zeta[1])))
        out += jsc.stats.norm.logpdf(x_obs, loc=landing(v, alpha), scale=.5)
        out += 1. + np.log(2 * np.pi) + jnp.sum(vp[1])        # entropy of q, two dimensions
        return out

    grad_elbo = jax.grad(elbo)
    np.random.seed(777)                                        # as in nb09-advi
    x_obs = trajectory_end(5., np.pi / 4) + .5 * np.random.randn()
    vp = jnp.array([np.random.randn(2), np.random.randn(2)])
    _ = np.random.randn(100, 2)                                # the test draws of nb09-advi
    for _epoch in range(50):
        g = jnp.zeros_like(vp)
        for eps in np.random.randn(25, 2):
            g += grad_elbo(vp, x_obs, eps)
        vp += .02 * g / 25
    zeta = np.asarray(vp[0]) + np.exp(np.asarray(vp[1])) * np.random.randn(4000, 2)
    advi = np.column_stack([10. / (1 + np.exp(-zeta[:, 0])),
                            np.pi / 2 / (1 + np.exp(-zeta[:, 1]))])

    def log_posterior(theta):
        v, alpha = theta
        if not (0 < v < 10 and 0 < alpha < np.pi / 2):
            return -np.inf
        return norm.logpdf(x_obs, trajectory_end(v, alpha), .5)

    rng = np.random.default_rng(0)
    start = np.column_stack([rng.uniform(4, 9, 32), rng.uniform(.3, 1.2, 32)])
    sampler = emcee.EnsembleSampler(32, 2, log_posterior)
    sampler.random_state = np.random.RandomState(0).get_state()
    sampler.run_mcmc(start, 4000)
    mcmc = sampler.get_chain(discard=1000, thin=10, flat=True)
    print("  projectile: x_obs %.2f, %d MCMC draws" % (x_obs, len(mcmc)))

    fig, ax = plt.subplots(figsize=(W, 3.2), dpi=200)
    ax.scatter(*mcmc.T, s=4, color=LIGHT, lw=0, label="MCMC (emcee)")
    ax.scatter(*advi[:2000].T, s=4, color=BLUE, lw=0, alpha=.6, label="ADVI, mean field")
    ax.set_xlabel(r"$v$ (m/s)")
    ax.set_ylabel(r"$\alpha$ (rad)")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, np.pi / 2)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, 1), ncol=2, markerscale=3)
    tidy(ax)
    fig.tight_layout()
    save(fig, "advi-projectile.png")


if __name__ == "__main__":
    graphical_model()
    kl_directions()
    mean_field()
    change_of_variables()
    normalizing_flow()
    advi_transform()
    advi_projectile()
