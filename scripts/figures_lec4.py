"""Redraw the graphical models of lecture 4, in one consistent style.

Usage: uv run python scripts/figures_lec4.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

GREY = "#354046"
OBSERVED = "#cfe0f3"
EDGE = "#354046"
plt.rcParams.update({"font.size": 13, "text.color": GREY, "mathtext.fontset": "cm"})


def figure(width, height):
    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    ax.set_axis_off()
    ax.set_aspect("equal")
    return fig, ax


def node(ax, xy, label, observed=False, r=.42):
    ax.add_patch(Circle(xy, r, facecolor=OBSERVED if observed else "white",
                        edgecolor=EDGE, lw=1.4, zorder=2))
    ax.text(*xy, label, ha="center", va="center", zorder=3, fontsize=14)


def square(ax, xy, label, side=.26, dx=.42, ha="left"):
    x, y = xy
    ax.add_patch(Rectangle((x - side / 2, y - side / 2), side, side,
                           facecolor=OBSERVED, edgecolor=EDGE, lw=1.2, zorder=2))
    ax.text(x + (dx if ha == "left" else -dx), y, label, ha=ha, va="center", fontsize=14)


def arrow(ax, start, end, r_start=.42, r_end=.42):
    """An arrow between two shapes, clipped to their radii in data units."""
    (x0, y0), (x1, y1) = start, end
    dx, dy = x1 - x0, y1 - y0
    norm = (dx ** 2 + dy ** 2) ** .5
    ux, uy = dx / norm, dy / norm
    ax.add_patch(FancyArrowPatch((x0 + ux * r_start, y0 + uy * r_start),
                                 (x1 - ux * r_end, y1 - uy * r_end),
                                 arrowstyle="-|>", mutation_scale=14,
                                 color=EDGE, lw=1.3, shrinkA=0, shrinkB=0, zorder=1))


def plate(ax, xy, width, height, label, pad=.0):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                                boxstyle="round,pad=%.2f,rounding_size=0.22" % pad,
                                facecolor="none", edgecolor=EDGE, lw=1.2, zorder=0))
    ax.text(x + width - .12, y + .16, label, ha="right", va="bottom", fontsize=13)


def save(fig, ax, path, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    fig.tight_layout(pad=.1)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print("wrote", path)


def unrolled():
    """theta -> z_i -> x_i, drawn for three observations."""
    fig, ax = figure(5.2, 3.6)
    node(ax, (0, 2.6), r"$\theta$")
    for k, x in enumerate([-1.6, 0, 1.6]):
        node(ax, (x, 1.3), r"$\mathbf{z}_%d$" % (k + 1))
        node(ax, (x, 0), r"$\mathbf{x}_%d$" % (k + 1), observed=True)
        arrow(ax, (0, 2.6), (x, 1.3))
        arrow(ax, (x, 1.3), (x, 0))
    save(fig, ax, "figures/lec4/lvm-unrolled.png", (-2.3, 2.3), (-.6, 3.2))


def plated(hyper=False):
    """The same model in plate notation, optionally with hyperparameters."""
    fig, ax = figure(4.0 if not hyper else 5.0, 3.6)
    node(ax, (0, 2.6), r"$\theta$")
    plate(ax, (-.9, -.65), 1.8, 2.65, "$N$")
    node(ax, (0, 1.3), r"$\mathbf{z}_i$")
    node(ax, (0, 0), r"$\mathbf{x}_i$", observed=True)
    arrow(ax, (0, 2.6), (0, 1.3))
    arrow(ax, (0, 1.3), (0, 0))
    if hyper:
        square(ax, (1.7, 2.6), r"$\alpha$")
        arrow(ax, (1.7, 2.6), (0, 2.6), r_start=.18)
        square(ax, (1.7, 1.3), r"$\beta$")
        arrow(ax, (1.7, 1.3), (0, 1.3), r_start=.18)
    name = "lvm-plate-hyper" if hyper else "lvm-plate"
    save(fig, ax, f"figures/lec4/{name}.png", (-1.2, 2.6 if hyper else 1.2), (-.8, 3.2))


def ppca():
    """Probabilistic PCA: a latent z_i, parameters pointing at x_i."""
    fig, ax = figure(4.6, 3.2)
    plate(ax, (-.9, -.65), 1.8, 2.65, "$N$")
    node(ax, (0, 1.3), r"$\mathbf{z}_i$")
    node(ax, (0, 0), r"$\mathbf{x}_i$", observed=True)
    arrow(ax, (0, 1.3), (0, 0))
    square(ax, (2.0, 0), r"$\mathbf{B}, \boldsymbol{\mu}, \sigma^2$")
    arrow(ax, (2.0, 0), (0, 0), r_start=.18)
    ax.text(-1.25, 2.95, r"$\mathbf{z}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$", fontsize=14)
    ax.text(-1.25, 2.45, r"$\mathbf{x}_i \mid \mathbf{z}_i \sim \mathcal{N}(\mathbf{B}\mathbf{z}_i + \boldsymbol{\mu},\ \sigma^2 \mathbf{I})$", fontsize=14)
    save(fig, ax, "figures/lec4/ppca-model.png", (-1.35, 3.7), (-.85, 3.45))


def mixture():
    """Gaussian mixture: weights pi, components (mu_k, sigma_k^2)."""
    fig, ax = figure(5.6, 4.6)
    node(ax, (0, 3.2), r"$\boldsymbol{\pi}$")
    square(ax, (1.9, 3.2), r"$\alpha$")
    arrow(ax, (1.9, 3.2), (0, 3.2), r_start=.18)

    plate(ax, (-.9, .55), 1.8, 2.15, "$N$")
    node(ax, (0, 2.1), r"$z_i$")
    node(ax, (0, 1.0), r"$\mathbf{x}_i$", observed=True)
    arrow(ax, (0, 3.2), (0, 2.1))
    arrow(ax, (0, 2.1), (0, 1.0))

    plate(ax, (-1.6, -1.35), 3.2, 1.55, "$K$")
    node(ax, (-.75, -.55), r"$\boldsymbol{\mu}_k$")
    node(ax, (.75, -.55), r"$\sigma^2_k$")
    arrow(ax, (-.75, -.55), (0, 1.0))
    arrow(ax, (.75, -.55), (0, 1.0))
    square(ax, (-2.2, -.55), r"$\sigma^2_\mu$", ha="right")
    arrow(ax, (-2.2, -.55), (-.75, -.55), r_start=.18)
    square(ax, (2.2, -.55), r"$\sigma^2_\sigma$")
    arrow(ax, (2.2, -.55), (.75, -.55), r_start=.18)
    save(fig, ax, "figures/lec4/mixture-model.png", (-3.2, 3.4), (-1.6, 3.9))


if __name__ == "__main__":
    unrolled()
    plated()
    plated(hyper=True)
    ppca()
    mixture()
