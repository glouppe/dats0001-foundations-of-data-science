"""Illustrations for the measurement process section of lecture 2.

One figure per example: the last poll of the 2024 Belgian election against its
result, and the timeout that cuts a stream of events into sessions. The LHC
slide uses an animation and a recorded event, both from CERN.

Usage: uv run python scripts/figures_lec2_measurement.py
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

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
    ax.set_ylim(0, 34)
    ax.set_yticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, ncols=2, loc="upper center", fontsize=11)
    ax.set_title("Votes in Flanders, Chamber of Representatives", loc="left",
                 fontsize=11, pad=8)
    save(fig, "figures/lec2/belgian-poll.png")


def sessions():
    """An evening on a streaming app, cut into sessions by a timeout."""
    fig, ax = plt.subplots(figsize=(5.8, 2.0), dpi=200)
    events = [(0, "open"), (3, "play"), (6, "pause"), (8, "play"), (14, "seek"),
              (17, "stop"), (62, "open"), (66, "play"), (73, "stop"), (120, "play")]
    ax.plot([e for e, _ in events], [0] * len(events), "o", ms=9, color=BLUE, zorder=3)
    ax.axhline(0, color=LIGHT, lw=1)
    for e, label in events:
        ax.text(e, -.28, label, rotation=45, ha="right", va="top", fontsize=9)
    for a, b, label in [(0, 17, "session 1"), (62, 73, "session 2"), (120, 120, "session 3")]:
        ax.add_patch(FancyBboxPatch((a - 3, -.18), b - a + 6, .36,
                                    boxstyle="round,pad=0,rounding_size=.15",
                                    facecolor=BLUE, alpha=.12, edgecolor="none"))
        ax.text((a + b) / 2, .3, label, ha="center", fontsize=11, color=BLUE)
    for a, b in [(17, 62), (73, 120)]:
        ax.annotate("", (a, -1.15), (b, -1.15),
                    arrowprops=dict(arrowstyle="<->", color=RED, lw=1))
        ax.text((a + b) / 2, -1.45, "%d min" % (b - a), ha="center", fontsize=10, color=RED)
    ax.set_ylim(-1.9, .7)
    ax.set_xlim(-8, 132)
    ax.set_yticks([])
    ax.set_xlabel("Minutes since the app was opened")
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    save(fig, "figures/lec2/sessions.png")


if __name__ == "__main__":
    belgian_poll()
    sessions()
