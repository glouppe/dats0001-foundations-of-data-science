"""Figures of lecture 5: the two graphical models, redrawn in the style of
lecture 4, and two plots comparing the continuous-time Lorenz system with its
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

# Lorenz system: a chaotic flow, whose trajectories fill a butterfly-shaped attractor.
S, RHO, B = 10.0, 28.0, 8 / 3
T, DT, FINE, SIGMA = 15.0, .01, 1e-4, 3.0     # horizon, discrete step, simulation step, noise


def drift(z):
    return np.array([S * (z[1] - z[0]), z[0] * (RHO - z[2]) - z[1], z[0] * z[1] - B * z[2]])


def lorenz(noise):
    """The Lorenz system: a continuous path, simulated on a very fine grid, and the
    discrete-time model with step DT, driven by the same noise; in 3d, and z1 over time."""
    z0 = np.array([1.0, 1.0, 1.0])
    for _ in range(int(5 / FINE)):            # burn-in, to start on the attractor
        z0 = z0 + drift(z0) * FINE
    rng = np.random.default_rng(0)
    n = int(round(T / FINE))
    dw = rng.normal(0, np.sqrt(FINE), (n, 3)) * (SIGMA if noise else 0)
    z = np.empty((n + 1, 3))
    z[0] = z0
    for k in range(n):                        # Euler-Maruyama on a very fine grid
        z[k + 1] = z[k] + drift(z[k]) * FINE + dw[k]
    every = int(round(DT / FINE))
    w = np.add.reduceat(dw, np.arange(0, n, every), axis=0)   # the same noise, per step
    zd = np.empty((len(w) + 1, 3))
    zd[0] = z0
    for k in range(len(w)):
        zd[k + 1] = zd[k] + drift(zd[k]) * DT + w[k]
    t, td = np.arange(n + 1) * FINE, np.arange(len(zd)) * DT

    fig = plt.figure(figsize=(7.6, 3.0), dpi=200)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.15, 1])
    ax = fig.add_subplot(grid[0], projection="3d")
    ax.plot(*z.T, color=GREY, lw=.45, label="continuous time")
    ax.plot(*zd.T, color=BLUE, lw=.45, alpha=.85, label=r"discrete time, $\Delta t = %g$" % DT)
    ax.view_init(18, -58)
    ax.set_box_aspect(None, zoom=1.25)
    ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis.line.set_color(LIGHT)
    ax.set_xlabel("$z_1$", labelpad=-12), ax.set_ylabel("$z_2$", labelpad=-12)
    ax.set_zlabel("$z_3$", labelpad=-12)

    ax2 = fig.add_subplot(grid[1])
    ax2.plot(t, z[:, 0], color=GREY, lw=.8)
    ax2.plot(td, zd[:, 0], color=BLUE, lw=.8)
    ax2.set_xlabel("$t$")
    ax2.set_ylabel("$z_1$")
    ax2.set_xlim(0, T)
    for side in ("right", "top"):
        ax2.spines[side].set_visible(False)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, loc="upper center",
               ncols=2, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .9))
    path = "figures/lec5/%s.svg" % ("sde-discretization" if noise else "ode-discretization")
    fig.savefig(path, facecolor="white")
    gap = np.abs(z[::every, 0] - zd[:, 0])
    print("wrote", path, "trajectories part at t = %.2f" % td[np.argmax(gap > 5)])


if __name__ == "__main__":
    lorenz(noise=False)
    lorenz(noise=True)
    for fig, ax, path in [static(), chain()]:
        x0, y0, x1, y1 = measure(fig, ax)
        width = max(WIDTH, x1 - x0 + 2 * MARGIN)
        centre = (x0 + x1) / 2
        rescale(fig, ax, centre - width / 2, centre + width / 2, y0 - MARGIN, y1 + MARGIN)
        fig.savefig(path, facecolor="white")
        print("wrote %-24s %.2f x %.2f in" % (path, *fig.get_size_inches()))
