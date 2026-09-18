"""Regenerate the figures of lecture 3 that are drawn for the course.

Usage: uv run python scripts/figures_lec3.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import hsv_to_rgb

GREY = "#354046"
RED = "#c0392b"
SPECIES = ["Adelie", "Chinstrap", "Gentoo"]
COLOR = {"Adelie": "#0173b2", "Chinstrap": "#de8f05", "Gentoo": "#029e73"}
LABEL_AT = {"Adelie": (178, 4750), "Chinstrap": (203.5, 3050), "Gentoo": (212, 6250)}

plt.rcParams.update({"font.size": 11, "text.color": GREY, "axes.labelcolor": GREY,
                     "xtick.color": GREY, "ytick.color": GREY, "axes.edgecolor": "#b8c0c6"})


def save(fig, path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print("wrote", path)


def hsv_model():
    """The HSV model: a hue wheel, then saturation and value ramps."""
    fig = plt.figure(figsize=(12.0, 4.4), dpi=200)
    gs = fig.add_gridspec(1, 3, width_ratios=[2.6, 1, 1], wspace=1.0)

    ax = fig.add_subplot(gs[0], projection="polar")
    theta = np.linspace(0, 2 * np.pi, 720)
    radius = np.linspace(0, 1, 120)
    T, R = np.meshgrid(theta, radius)
    rgb = hsv_to_rgb(np.dstack([T / (2 * np.pi), R, np.ones_like(R)]))
    ax.pcolormesh(theta, radius, np.zeros_like(R), color=rgb.reshape(-1, 3), shading="auto")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)
    for degrees, name in [(0, "Red\n0°"), (60, "Yellow\n60°"), (120, "Green\n120°"),
                          (180, "Cyan\n180°"), (240, "Blue\n240°"), (300, "Magenta\n300°")]:
        ax.text(np.deg2rad(degrees), 1.24, name, ha="center", va="center", fontsize=10, color=GREY)
    ax.set_title("Hue", fontsize=13, color=GREY, pad=26)

    def ramp(position, title, channels):
        ax = fig.add_subplot(gs[position])
        t = np.linspace(0, 1, 256)
        img = hsv_to_rgb(np.dstack(channels(t)).reshape(1, 256, 3))
        ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower", aspect="auto", extent=(0, 1, 0, 100))
        ax.set_xticks([])
        ax.set_yticks([0, 50, 100])
        ax.set_yticklabels(["0%", "50%", "100%"], fontsize=10, color=GREY)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=13, color=GREY, pad=26)

    ramp(1, "Saturation", lambda t: (np.zeros_like(t), t, np.ones_like(t)))
    ramp(2, "Value", lambda t: (np.zeros_like(t), np.ones_like(t), t))
    save(fig, "figures/lec3/hsv.png", tight=False)


def exploratory_explanatory(df):
    """The same scatter drawn for oneself, then for an audience."""
    fig, (raw, polished) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)
    raw.scatter(df.flipper_length_mm, df.body_mass_g)
    raw.set_title("Exploratory: ten seconds, for yourself", fontsize=12, loc="left")

    for species in SPECIES:
        d = df[df.species == species]
        polished.scatter(d.flipper_length_mm, d.body_mass_g, s=18, alpha=.75, color=COLOR[species])
        polished.annotate(species, LABEL_AT[species], color=COLOR[species], fontsize=12,
                          fontweight="bold", ha="center")
    polished.set_xlabel("Flipper length [mm]")
    polished.set_ylabel("Body mass [g]")
    polished.set_title("Explanatory: longer flippers, heavier birds", fontsize=12, loc="left")
    polished.grid(alpha=.25)
    polished.set_ylim(2500, 6700)

    for ax in (raw, polished):
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, "figures/lec3/exploratory-explanatory.png")


def four_questions(df):
    """Doumont's four questions, each answered on the same data."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), dpi=200)

    means = df.groupby("species").body_mass_g.mean().reindex(SPECIES)
    axes[0, 0].barh(SPECIES, means.values, color=[COLOR[s] for s in SPECIES])
    axes[0, 0].set_xlim(0, 5600)
    axes[0, 0].set_title("Comparison", loc="left", fontsize=12)
    axes[0, 0].set_xlabel("Mean body mass [g]")

    axes[0, 1].hist(df.body_mass_g, bins=25, color="#8c9196")
    axes[0, 1].set_title("Distribution", loc="left", fontsize=12)
    axes[0, 1].set_xlabel("Body mass [g]")

    for species in SPECIES:
        d = df[df.species == species]
        axes[1, 0].scatter(d.flipper_length_mm, d.body_mass_g, s=12, alpha=.7, color=COLOR[species])
    axes[1, 0].set_title("Correlation", loc="left", fontsize=12)
    axes[1, 0].set_xlabel("Flipper length [mm]")
    axes[1, 0].set_ylabel("Body mass [g]")

    evolution = df.groupby(["year", "species"]).body_mass_g.mean().unstack()
    for species in SPECIES:
        axes[1, 1].plot(evolution.index, evolution[species], marker="o", color=COLOR[species])
    axes[1, 1].set_xticks(evolution.index)
    axes[1, 1].set_title("Evolution", loc="left", fontsize=12)
    axes[1, 1].set_xlabel("Year")
    axes[1, 1].set_ylabel("Mean body mass [g]")

    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, "figures/lec3/four-questions.png")


def anatomy(df):
    """One scatter, annotated with the pieces of the grammar."""
    fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=200)
    for species in SPECIES:
        d = df[df.species == species]
        ax.scatter(d.flipper_length_mm, d.body_mass_g, s=22, alpha=.8, color=COLOR[species],
                   label=species)
    ax.set_xlabel("Flipper length [mm]")
    ax.set_ylabel("Body mass [g]")
    ax.legend(title="Species", frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(168, 236)
    ax.set_ylim(2400, 7400)

    arrow = dict(arrowstyle="->", color=RED, lw=1.3)
    ax.annotate("mark: one point per penguin", xy=(190.2, 4650), xytext=(176, 6100),
                color=RED, fontsize=11, arrowprops=arrow)
    ax.text(176, 6950, "channels: x position, y position, colour", color=RED, fontsize=11)
    ax.text(176, 6650, "scales: 170–235 mm, 2500–6500 g", color=RED, fontsize=11)
    ax.annotate("guides: axis labels, ticks, legend", xy=(222, 3050), xytext=(196, 2550),
                color=RED, fontsize=11, arrowprops=arrow)
    save(fig, "figures/lec3/anatomy.png")


def colormap_types(df):
    """Sequential, diverging and categorical colormaps, each on fitting data."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), dpi=200)
    grid = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(grid, grid)
    Z = np.exp(-(X ** 2 + Y ** 2) / 3) * np.sin(2 * X)

    axes[0].imshow(np.abs(Z), cmap="viridis", extent=(-3, 3, -3, 3))
    axes[0].set_title("Sequential: ordered data", fontsize=12, loc="left")
    axes[1].imshow(Z, cmap="RdBu_r", vmin=-.9, vmax=.9, extent=(-3, 3, -3, 3))
    axes[1].set_title("Diverging: a meaningful zero", fontsize=12, loc="left")

    for species in SPECIES:
        d = df[df.species == species]
        axes[2].scatter(d.flipper_length_mm, d.body_mass_g, s=12, alpha=.8, color=COLOR[species],
                        label=species)
    axes[2].set_title("Categorical: unordered groups", fontsize=12, loc="left")
    axes[2].legend(frameon=False, fontsize=9, loc="upper left")
    axes[2].spines[["top", "right"]].set_visible(False)

    for ax in axes[:2]:
        ax.set_xticks([])
        ax.set_yticks([])
    save(fig, "figures/lec3/colormap-types.png")


if __name__ == "__main__":
    penguins = pd.read_csv("data/penguins.csv").dropna(subset=["flipper_length_mm", "body_mass_g"])
    hsv_model()
    exploratory_explanatory(penguins)
    four_questions(penguins)
    anatomy(penguins)
    colormap_types(penguins)
