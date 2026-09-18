"""Redraw the graphical models of lecture 4, in one consistent style.

Usage: uv run python scripts/figures_lec4.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

GREY = "#354046"
OBSERVED = "#cfe0f3"
R = .46           # node radius
R_SQ = .17        # half-side of a hyperparameter square
PAD = .42         # padding between a plate and the nodes it holds
FS = 17           # label size

plt.rcParams.update({"text.color": GREY, "mathtext.fontset": "cm"})


def figure(width, height):
    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    ax.set_axis_off()
    ax.set_aspect("equal")
    return fig, ax


def node(ax, xy, label, observed=False):
    ax.add_patch(Circle(xy, R, facecolor=OBSERVED if observed else "white",
                        edgecolor=GREY, lw=1.6, zorder=2))
    ax.text(*xy, label, ha="center", va="center", zorder=3, fontsize=FS)


def square(ax, xy, label, side=.34, gap=.28, side_of="right"):
    x, y = xy
    ax.add_patch(Rectangle((x - side / 2, y - side / 2), side, side,
                           facecolor=OBSERVED, edgecolor=GREY, lw=1.4, zorder=2))
    dx = gap if side_of == "right" else -gap
    ax.text(x + dx, y, label, ha="left" if side_of == "right" else "right",
            va="center", fontsize=FS)


def arrow(ax, start, end, r_start=R, r_end=R):
    (x0, y0), (x1, y1) = start, end
    dx, dy = x1 - x0, y1 - y0
    norm = (dx ** 2 + dy ** 2) ** .5
    ux, uy = dx / norm, dy / norm
    ax.add_patch(FancyArrowPatch((x0 + ux * r_start, y0 + uy * r_start),
                                 (x1 - ux * r_end, y1 - uy * r_end),
                                 arrowstyle="-|>", mutation_scale=16,
                                 color=GREY, lw=1.5, shrinkA=0, shrinkB=0, zorder=1))


def box(xy, r=R):
    """The bounding box of a node, as plates are sized from what they contain."""
    x, y = xy
    return (x - r, y - r, x + r, y + r)


def plate(ax, items, label, pad=PAD):
    """A plate sized from the boxes it contains, nested plates included."""
    x0 = min(b[0] for b in items) - pad
    y0 = min(b[1] for b in items) - pad
    x1 = max(b[2] for b in items) + pad
    y1 = max(b[3] for b in items) + pad
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=0.25",
                                facecolor="none", edgecolor=GREY, lw=1.4, zorder=0))
    ax.text(x1 - .18, y0 + .18, label, ha="right", va="bottom", fontsize=FS - 2)
    return (x0, y0, x1, y1)


def save(fig, ax, path, pad=.35):
    ax.relim()
    ax.autoscale_view()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    fig.tight_layout(pad=.1)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print("wrote", path)


def unrolled():
    """theta -> z_i -> x_i, drawn for three observations."""
    fig, ax = figure(5.4, 3.8)
    node(ax, (0, 2.7), r"$\theta$")
    for k, x in enumerate([-1.7, 0, 1.7]):
        node(ax, (x, 1.35), r"$\mathbf{z}_%d$" % (k + 1))
        node(ax, (x, 0), r"$\mathbf{x}_%d$" % (k + 1), observed=True)
        arrow(ax, (0, 2.7), (x, 1.35))
        arrow(ax, (x, 1.35), (x, 0))
    save(fig, ax, "figures/lec4/lvm-unrolled.png")


def plated(hyper=False):
    """The same model in plate notation, optionally with hyperparameters."""
    fig, ax = figure(3.6 if not hyper else 4.6, 3.8)
    node(ax, (0, 2.7), r"$\theta$")
    plate(ax, [box((0, 1.35)), box((0, 0))], "$N$")
    node(ax, (0, 1.35), r"$\mathbf{z}_i$")
    node(ax, (0, 0), r"$\mathbf{x}_i$", observed=True)
    arrow(ax, (0, 2.7), (0, 1.35))
    arrow(ax, (0, 1.35), (0, 0))
    if hyper:
        square(ax, (2.1, 2.7), r"$\alpha$")
        arrow(ax, (2.1, 2.7), (0, 2.7), r_start=R_SQ)
        square(ax, (2.1, 1.35), r"$\beta$")
        arrow(ax, (2.1, 1.35), (0, 1.35), r_start=R_SQ)
    save(fig, ax, "figures/lec4/lvm-plate%s.png" % ("-hyper" if hyper else ""))


def ppca():
    """Probabilistic PCA: a latent z_i, parameters pointing at x_i."""
    fig, ax = figure(4.4, 3.2)
    plate(ax, [box((0, 1.35)), box((0, 0))], "$N$")
    node(ax, (0, 1.35), r"$\mathbf{z}_i$")
    node(ax, (0, 0), r"$\mathbf{x}_i$", observed=True)
    arrow(ax, (0, 1.35), (0, 0))
    square(ax, (2.3, 0), r"$\mathbf{B}, \boldsymbol{\mu}, \sigma^2$")
    arrow(ax, (2.3, 0), (0, 0), r_start=R_SQ)
    save(fig, ax, "figures/lec4/ppca-model.png")


def mixture():
    """Gaussian mixture: weights pi, components (mu_k, sigma_k^2)."""
    fig, ax = figure(5.8, 4.6)
    node(ax, (0, 3.3), r"$\boldsymbol{\pi}$")
    square(ax, (2.3, 3.3), r"$\alpha$")
    arrow(ax, (2.3, 3.3), (0, 3.3), r_start=R_SQ)

    plate(ax, [box((0, 2.0)), box((0, .7))], "$N$")
    node(ax, (0, 2.0), r"$z_i$")
    node(ax, (0, .7), r"$\mathbf{x}_i$", observed=True)
    arrow(ax, (0, 3.3), (0, 2.0))
    arrow(ax, (0, 2.0), (0, .7))

    plate(ax, [box((-.9, -1.5)), box((.9, -1.5))], "$K$")
    node(ax, (-.9, -1.5), r"$\boldsymbol{\mu}_k$")
    node(ax, (.9, -1.5), r"$\sigma^2_k$")
    arrow(ax, (-.9, -1.5), (0, .7))
    arrow(ax, (.9, -1.5), (0, .7))
    square(ax, (-2.7, -1.5), r"$\sigma^2_\mu$", side_of="left")
    arrow(ax, (-2.7, -1.5), (-.9, -1.5), r_start=R_SQ)
    square(ax, (2.7, -1.5), r"$\sigma^2_\sigma$")
    arrow(ax, (2.7, -1.5), (.9, -1.5), r_start=R_SQ)
    save(fig, ax, "figures/lec4/mixture-model.png")


def lda():
    """Mixed membership: topics per document, one assignment per word."""
    fig, ax = figure(5.8, 4.8)
    square(ax, (2.6, 3.0), r"$\alpha$")
    node(ax, (0, 3.0), r"$\boldsymbol{\pi}_m$")
    arrow(ax, (2.6, 3.0), (0, 3.0), r_start=R_SQ)

    node(ax, (0, 1.7), r"$z_{mn}$")
    node(ax, (0, .4), r"$x_{mn}$", observed=True)
    arrow(ax, (0, 3.0), (0, 1.7))
    arrow(ax, (0, 1.7), (0, .4))
    words = plate(ax, [box((0, 1.7)), box((0, .4))], "$N$")
    plate(ax, [box((0, 3.0)), words], "$M$")

    node(ax, (0, -2.2), r"$\boldsymbol{\mu}_k$")
    plate(ax, [box((0, -2.2))], "$K$")
    arrow(ax, (0, -2.2), (0, .4))
    square(ax, (2.6, -2.2), r"$\eta$")
    arrow(ax, (2.6, -2.2), (0, -2.2), r_start=R_SQ)
    save(fig, ax, "figures/lec4/lda-model.png")


if __name__ == "__main__":
    unrolled()
    plated()
    plated(hyper=True)
    ppca()
    mixture()
    lda()
