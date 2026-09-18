"""Redraw the graphical models of lecture 4, in one consistent style.

Every coordinate is derived from the constants below, and all six figures are
saved on a canvas of the same width, at the same scale. They therefore share a
single .width-NN class in the deck, with no figure stretched relative to another.

Usage: uv run python scripts/figures_lec4.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

GREY = "#354046"
OBSERVED = "#cfe0f3"
R = .46           # node radius
R_SQ = .10        # half-side of a hyperparameter square
STEP = 1.35       # vertical distance between two nodes
COL = 1.70        # horizontal distance between two nodes
PAD = .42         # padding between a plate and the nodes it holds
GAP = .34         # clearance between a plate and what sits outside it
LABEL = .22       # distance from a square to its label
FS = 17           # label size, with the canvas sized so it scales up on the slide
UNIT = .53        # inches per drawing unit: the same in every figure, so that all
                  # six render at one scale and can share a single .width-NN class
MARGIN = .35      # white margin around the drawing

plt.rcParams.update({"text.color": GREY, "mathtext.fontset": "cm", "svg.fonttype": "path"})


def figure():
    fig, ax = plt.subplots(dpi=200)
    ax.set_axis_off()
    return fig, ax


def node(ax, xy, label, observed=False):
    ax.add_patch(Circle(xy, R, facecolor=OBSERVED if observed else "white",
                        edgecolor=GREY, lw=1.15, zorder=2))
    ax.text(*xy, label, ha="center", va="center", zorder=3, fontsize=FS)


def arrow(ax, start, end, r_start=R, r_end=R):
    """An edge between two nodes, stopping at their borders."""
    (x0, y0), (x1, y1) = start, end
    dx, dy = x1 - x0, y1 - y0
    norm = (dx ** 2 + dy ** 2) ** .5
    ux, uy = dx / norm, dy / norm
    ax.add_patch(FancyArrowPatch((x0 + ux * r_start, y0 + uy * r_start),
                                 (x1 - ux * r_end, y1 - uy * r_end),
                                 arrowstyle="-|>", mutation_scale=11,
                                 color=GREY, lw=1.05, shrinkA=0, shrinkB=0, zorder=1))


def hyper(ax, target, label, clear, side="right"):
    """A fixed hyperparameter: a small filled square, just outside the edge `clear`."""
    way = 1 if side == "right" else -1
    x, y = clear + way * (GAP + R_SQ), target[1]
    ax.add_patch(Rectangle((x - R_SQ, y - R_SQ), 2 * R_SQ, 2 * R_SQ,
                           facecolor=GREY, edgecolor=GREY, lw=.7, zorder=2))
    ax.text(x + way * LABEL, y, label, va="center", fontsize=FS,
            ha="left" if side == "right" else "right")
    arrow(ax, (x, y), target, r_start=R_SQ)


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
                                facecolor="none", edgecolor=GREY, lw=1.0, zorder=0))
    ax.text(x1 - .13, y0 + .10, label, ha="right", va="bottom", fontsize=FS - 4)
    return (x0, y0, x1, y1)


def above(plate_box):
    """Where to put a node sitting just above a plate."""
    return (0, plate_box[3] + GAP + R)


def below(plate_box):
    """Where to put a node whose own plate sits just below another one."""
    return plate_box[1] - GAP - PAD - R


def rescale(fig, ax, x0, x1, y0, y1):
    """Show [x0, x1] x [y0, y1] with one drawing unit exactly UNIT inches wide."""
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    fig.set_size_inches((x1 - x0) * UNIT, (y1 - y0) * UNIT)
    ax.set_position((0, 0, 1, 1))


def extent(fig, ax):
    """The drawing's bounding box, labels included, in drawing units."""
    fig.canvas.draw()
    bb = ax.get_tightbbox(fig.canvas.get_renderer())
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
    return x0, y0, x1, y1


def measure(fig, ax):
    """Set the final scale, then measure: text is sized in points, not in units."""
    ax.relim()
    ax.autoscale_view()
    rescale(fig, ax, *ax.get_xlim(), *ax.get_ylim())
    return extent(fig, ax)


def unrolled():
    """theta -> z_i -> x_i, drawn for three observations."""
    fig, ax = figure()
    theta = (0, 2 * STEP)
    node(ax, theta, r"$\theta$")
    for k, x in enumerate([-COL, 0, COL]):
        node(ax, (x, STEP), r"$\mathbf{z}_%d$" % (k + 1))
        node(ax, (x, 0), r"$\mathbf{x}_%d$" % (k + 1), observed=True)
        arrow(ax, theta, (x, STEP))
        arrow(ax, (x, STEP), (x, 0))
    return fig, ax, "figures/lec4/lvm-unrolled.svg"


def plated(with_hyper=False):
    """The same model in plate notation, optionally with hyperparameters."""
    fig, ax = figure()
    observations = plate(ax, [box((0, STEP)), box((0, 0))], "$N$")
    theta = above(observations)
    node(ax, theta, r"$\theta$")
    node(ax, (0, STEP), r"$\mathbf{z}_i$")
    node(ax, (0, 0), r"$\mathbf{x}_i$", observed=True)
    arrow(ax, theta, (0, STEP))
    arrow(ax, (0, STEP), (0, 0))
    if with_hyper:
        hyper(ax, theta, r"$\alpha$", observations[2])
        hyper(ax, (0, STEP), r"$\beta$", observations[2])
    return fig, ax, "figures/lec4/lvm-plate%s.svg" % ("-hyper" if with_hyper else "")


def ppca():
    """Probabilistic PCA: a latent z_i, parameters pointing at x_i."""
    fig, ax = figure()
    observations = plate(ax, [box((0, STEP)), box((0, 0))], "$N$")
    node(ax, (0, STEP), r"$\mathbf{z}_i$")
    node(ax, (0, 0), r"$\mathbf{x}_i$", observed=True)
    arrow(ax, (0, STEP), (0, 0))
    hyper(ax, (0, 0), r"$\mathbf{B}, \boldsymbol{\mu}, \sigma^2$", observations[2])
    return fig, ax, "figures/lec4/ppca-model.svg"


def mixture():
    """Gaussian mixture: weights pi, components (mu_k, sigma_k^2)."""
    fig, ax = figure()
    observations = plate(ax, [box((0, STEP)), box((0, 0))], "$N$")
    node(ax, (0, STEP), r"$z_i$")
    node(ax, (0, 0), r"$\mathbf{x}_i$", observed=True)
    arrow(ax, (0, STEP), (0, 0))

    pi = above(observations)
    node(ax, pi, r"$\boldsymbol{\pi}$")
    arrow(ax, pi, (0, STEP))

    y = below(observations)
    components = plate(ax, [box((-COL / 2, y)), box((COL / 2, y))], "$K$")
    node(ax, (-COL / 2, y), r"$\boldsymbol{\mu}_k$")
    node(ax, (COL / 2, y), r"$\sigma^2_k$")
    arrow(ax, (-COL / 2, y), (0, 0))
    arrow(ax, (COL / 2, y), (0, 0))

    right = max(observations[2], components[2])
    hyper(ax, pi, r"$\alpha$", right)
    hyper(ax, (-COL / 2, y), r"$\sigma^2_\mu$", components[0], side="left")
    hyper(ax, (COL / 2, y), r"$\sigma^2_\sigma$", right)
    return fig, ax, "figures/lec4/mixture-model.svg"


def lda():
    """Mixed membership: topics per document, one assignment per word."""
    fig, ax = figure()
    node(ax, (0, STEP), r"$z_{mn}$")
    node(ax, (0, 0), r"$x_{mn}$", observed=True)
    arrow(ax, (0, STEP), (0, 0))
    words = plate(ax, [box((0, STEP)), box((0, 0))], "$N$")

    pi = above(words)
    node(ax, pi, r"$\boldsymbol{\pi}_m$")
    arrow(ax, pi, (0, STEP))
    documents = plate(ax, [box(pi), words], "$M$")

    y = below(documents)
    plate(ax, [box((0, y))], "$K$")
    node(ax, (0, y), r"$\boldsymbol{\mu}_k$")
    arrow(ax, (0, y), (0, 0))

    hyper(ax, pi, r"$\alpha$", documents[2])
    hyper(ax, (0, y), r"$\eta$", documents[2])
    return fig, ax, "figures/lec4/lda-model.svg"


def gaia(learned=False):
    """Distances to stars: a noisy parallax per star, a prior from the Galaxy.

    With `learned`, the length scale of that prior becomes a parameter shared by
    all the stars, which is what makes the model hierarchical.
    """
    fig, ax = figure()
    node(ax, (0, STEP), r"$r_i$")
    node(ax, (0, 0), r"$\varpi_i$", observed=True)
    arrow(ax, (0, STEP), (0, 0))
    noise = R + GAP + R_SQ                       # the known uncertainty of star i
    stars = plate(ax, [box((0, STEP)), box((0, 0)), box((noise + .35, 0), r=.40)], "$N$")
    hyper(ax, (0, 0), r"$\sigma_i$", R)
    if learned:
        length = (0, stars[3] + GAP + R)
        node(ax, length, "$L$")
        arrow(ax, length, (0, STEP))
    else:
        hyper(ax, (0, STEP), "$L$", stars[2])
    return fig, ax, "figures/lec4/gaia-model%s.svg" % ("-learned" if learned else "")


if __name__ == "__main__":
    drawings = [unrolled(), plated(), plated(with_hyper=True), ppca(), mixture(), lda(),
                gaia(), gaia(learned=True)]
    boxes = [measure(fig, ax) for fig, ax, _ in drawings]
    width = max(x1 - x0 for x0, _, x1, _ in boxes) + 2 * MARGIN
    for (fig, ax, path), (x0, y0, x1, y1) in zip(drawings, boxes):
        centre = (x0 + x1) / 2
        rescale(fig, ax, centre - width / 2, centre + width / 2, y0 - MARGIN, y1 + MARGIN)
        fig.savefig(path, facecolor="white")
        print("wrote %-38s %.2f x %.2f in" % (path, *fig.get_size_inches()))
