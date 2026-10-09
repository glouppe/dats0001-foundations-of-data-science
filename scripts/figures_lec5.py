"""Figures of lecture 5: the two graphical models, redrawn in the style of
lecture 4, and two plots comparing a continuous-time process with its
discrete-time version, without noise (an ODE) and with noise (an SDE).

The drawing functions and constants come from figures_lec4.py, and the canvas
has the width of the lecture 4 figures, so that both lectures show their
graphical models at the same scale with a single .width-NN class.

Usage: uv run python scripts/figures_lec5.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from figures_lec4 import (FS, GREY, MARGIN, OBSERVED, R, STEP, UNIT, arrow, box,  # noqa: E402
                          figure, measure, node, plate, rescale)

WIDTH = 246.5568 / 72 / UNIT    # canvas width of the lecture 4 figures, in drawing units


def static():
    """A static latent variable model: theta points at z_i and at x_i."""
    fig, ax = figure()
    node(ax, (0, STEP), r"$\mathbf{z}_i$")
    node(ax, (0, 0), r"$\mathbf{x}_i$", observed=True)
    arrow(ax, (0, STEP), (0, 0))
    observations = plate(ax, [box((0, STEP)), box((0, 0))], "$N$")
    theta = (observations[2] + .34 + R, STEP)
    node(ax, theta, r"$\theta$")
    arrow(ax, theta, (0, STEP))
    arrow(ax, theta, (0, 0))
    return fig, ax, "figures/lec5/lvm.svg"


def chain():
    """A state-space model: the states form a Markov chain, each emits an observation."""
    fig, ax = figure()
    gap = 1.5                                   # tighter than COL, to fit the canvas
    xs = [-gap, 0, gap]
    for x, t in zip(xs, ["t-1", "t", "t+1"]):
        for y, v in ((STEP, "z"), (0, "x")):
            ax.add_patch(Circle((x, y), R, facecolor=OBSERVED if v == "x" else "white",
                                edgecolor=GREY, lw=1.15, zorder=2))
            # time indices are wider than i: a smaller label keeps them inside the node
            ax.text(x, y, r"$\mathbf{%s}_{%s}$" % (v, t), ha="center", va="center",
                    zorder=3, fontsize=FS - 3)
        arrow(ax, (x, STEP), (x, 0))
    for a, b in zip(xs, xs[1:]):
        arrow(ax, (a, STEP), (b, STEP))
    for side in (-1, 1):
        dots = (side * (gap + 1.0), STEP)
        ax.text(*dots, r"$\cdots$", ha="center", va="center", fontsize=FS)
        if side < 0:
            arrow(ax, dots, (xs[0], STEP), r_start=.32)
        else:
            arrow(ax, (xs[-1], STEP), dots, r_end=.32)
    return fig, ax, "figures/lec5/sm.svg"


BLUE = "#0173b2"
LIGHT = "#b8c0c6"
KAPPA, MU, SIGMA, Z0 = .8, 0.0, .6, 2.0     # decay rate, equilibrium, diffusion, start
T, DT, FINE = 6.0, .5, 1e-3                  # horizon, discrete step, simulation step


def discretization(noise):
    """A continuous path and the discrete-time model with step DT, driven by the same
    noise: the decay ODE when noise is False, the Ornstein-Uhlenbeck SDE otherwise."""
    rng = np.random.default_rng(3)
    n = int(round(T / FINE))
    t = np.arange(n + 1) * FINE
    dw = rng.normal(0, np.sqrt(FINE), n) * (SIGMA if noise else 0)
    z = np.empty(n + 1)
    z[0] = Z0
    for k in range(n):                       # Euler-Maruyama on a very fine grid
        z[k + 1] = z[k] - KAPPA * (z[k] - MU) * FINE + dw[k]
    every = int(round(DT / FINE))
    td = t[::every]
    w = np.add.reduceat(dw, np.arange(0, n, every))   # the same noise, summed per step
    zd = np.empty(len(td))
    zd[0] = Z0
    for k in range(1, len(td)):
        zd[k] = zd[k - 1] - KAPPA * (zd[k - 1] - MU) * DT + w[k - 1]

    fig, ax = plt.subplots(figsize=(5.8, 2.6), dpi=200)
    ax.axhline(MU, color=LIGHT, lw=.8, ls=(0, (4, 3)))
    ax.plot(t, z, color=GREY, lw=1.2, label=r"continuous time, $z(t)$")
    ax.plot(td, zd, "o-", color=BLUE, ms=4, lw=.9,
            label=r"discrete time, $z_t$, $\Delta t = %g$" % DT)
    ax.set_xlabel("$t$")
    ax.set_xlim(0, T)
    ax.set_ylim(-1.2, 2.3)
    ax.set_yticks([MU], [r"$\mu$"])
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    path = "figures/lec5/%s.svg" % ("sde-discretization" if noise else "ode-discretization")
    fig.savefig(path, facecolor="white")
    print("wrote", path)


if __name__ == "__main__":
    discretization(noise=False)
    discretization(noise=True)
    for fig, ax, path in [static(), chain()]:
        x0, y0, x1, y1 = measure(fig, ax)
        width = max(WIDTH, x1 - x0 + 2 * MARGIN)
        centre = (x0 + x1) / 2
        rescale(fig, ax, centre - width / 2, centre + width / 2, y0 - MARGIN, y1 + MARGIN)
        fig.savefig(path, facecolor="white")
        print("wrote %-24s %.2f x %.2f in" % (path, *fig.get_size_inches()))
