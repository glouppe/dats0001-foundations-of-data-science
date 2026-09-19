"""Illustrations for the measurement process section of lecture 2.

One figure per example that needs drawing: the last poll of the 2024 Belgian
election against its result, and an evening of listening in which only plays of
30 seconds or more become streams. The LHC slide uses an animation from CERN.

Usage: uv run python scripts/figures_lec2_measurement.py
"""

import matplotlib.pyplot as plt
import numpy as np

GREY = "#354046"
LIGHT = "#b8c0c6"
BLUE = "#0173b2"
RED = "#c0392b"
GREEN = "#029e73"

plt.rcParams.update({"font.size": 12, "text.color": GREY, "axes.labelcolor": GREY,
                     "xtick.color": GREY, "ytick.color": GREY, "axes.edgecolor": LIGHT,
                     "mathtext.fontset": "cm"})


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print("wrote", path)


def belgian_poll():
    """The last poll of the 2024 Belgian federal election, against the result."""
    fig, ax = plt.subplots(figsize=(5.8, 2.9), dpi=200)
    x = np.arange(2)
    ax.bar(x - .19, [19, 25.6], .36, color=LIGHT, label="N-VA")
    ax.bar(x + .19, [27, 21.9], .36, color=BLUE, label="Vlaams Belang")
    for xi, (a, b) in zip(x, [(19, 27), (25.6, 21.9)]):
        ax.text(xi - .19, a + .8, "%g%%" % a, ha="center", fontsize=11)
        ax.text(xi + .19, b + .8, "%g%%" % b, ha="center", fontsize=11, color=BLUE)
    ax.set_xticks(x, ["the last poll, 4 June 2024", "the election, 9 June 2024"])
    ax.set_ylim(0, 40)
    ax.set_yticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, ncols=2, loc="upper center", fontsize=11)
    ax.set_title("Votes in Flanders, Chamber of Representatives", loc="left",
                 fontsize=11, pad=8)
    save(fig, "figures/lec2/belgian-poll.png")


def streams():
    """One evening of listening: a play becomes a stream only after 30 seconds."""
    played = [212, 14, 187, 6, 31, 245, 22, 164]      # seconds, in order
    counted = [t >= 30 for t in played]

    fig, ax = plt.subplots(figsize=(5.8, 2.3), dpi=200)
    y = np.arange(len(played))[::-1]
    ax.barh(y, played, .62, color=[BLUE if c else LIGHT for c in counted])
    ax.axvline(30, color=RED, lw=1.2, ls=(0, (4, 3)))
    ax.text(33, len(played) - .35, "30 s", color=RED, fontsize=11, va="bottom")
    for yi, t, c in zip(y, played, counted):
        ax.text(max(t, 30) + 5, yi, "stream" if c else "not counted", va="center",
                fontsize=9, color=BLUE if c else GREY)
    ax.set_yticks([])
    ax.set_ylim(-.7, len(played) + .3)
    ax.set_xlim(0, 300)
    ax.set_xlabel("Seconds played, one track after another")
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    save(fig, "figures/lec2/streams.png")


if __name__ == "__main__":
    belgian_poll()
    streams()
