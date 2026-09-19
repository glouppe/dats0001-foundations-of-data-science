"""Illustrations for the measurement process section of lecture 2.

One figure per example: what the LHC trigger keeps, the 1936 Literary Digest poll
against the election, the timeout that cuts a log into sessions, and the
annotators behind a label.

Usage: uv run python scripts/figures_lec2_measurement.py
"""

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


def trigger():
    """Of 40 million crossings a second, a few thousand are written down.

    A funnel rather than bars: on a log scale, bars would start nowhere.
    """
    levels = [(1.00, "40 000 000 crossings per second"),
              (0.42, "100 000 kept by the hardware trigger"),
              (0.12, "3 000 kept by the software trigger, and written down")]

    fig, ax = plt.subplots(figsize=(5.8, 2.3), dpi=200)
    ax.set_axis_off()
    xs = [w / 2 for w, _ in levels] + [-w / 2 for w, _ in levels][::-1]
    ys = [0, -1, -2, -2, -1, 0]
    ax.fill(xs, ys, color=BLUE, alpha=.18, edgecolor="none")
    for k, (w, label) in enumerate(levels):
        ax.plot([-w / 2, w / 2], [-k, -k], color=BLUE, lw=1.6)
        ax.text(w / 2 + .06, -k, label, va="center", fontsize=11,
                color=BLUE if k == 2 else GREY)
    ax.set_xlim(-.62, 1.9)
    ax.set_ylim(-2.45, .45)
    save(fig, "figures/lec2/trigger.png")


def literary_digest():
    """Two and a half million answers, and the wrong winner."""
    fig, ax = plt.subplots(figsize=(5.8, 2.9), dpi=200)
    x = np.arange(2)
    ax.bar(x - .19, [57, 38], .36, color=LIGHT, label="Landon")
    ax.bar(x + .19, [43, 62], .36, color=BLUE, label="Roosevelt")
    for xi, (a, b) in zip(x, [(57, 43), (38, 62)]):
        ax.text(xi - .19, a + 1.5, f"{a}%", ha="center", fontsize=11)
        ax.text(xi + .19, b + 1.5, f"{b}%", ha="center", fontsize=11, color=BLUE)
    ax.set_xticks(x, ["the Literary Digest poll\n(2.4 million answers)",
                      "the election"])
    ax.set_ylim(0, 74)
    ax.set_yticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, ncols=2, loc="upper center", fontsize=11)
    save(fig, "figures/lec2/literary-digest.png")


def sessions():
    """A log is a list of events; a session is a rule applied to it."""
    fig, ax = plt.subplots(figsize=(5.8, 1.9), dpi=200)
    events = [0, 3, 6, 8, 14, 17, 62, 66, 73, 120]          # minutes
    gap = 30
    ax.plot(events, [0] * len(events), "o", ms=9, color=BLUE, zorder=3)
    ax.axhline(0, color=LIGHT, lw=1)
    start = 0
    for k, (a, b) in enumerate(zip(events, events[1:] + [events[-1]])):
        if b - a > gap or b == a:
            ax.add_patch(FancyBboxPatch((start - 3, -.35), a - start + 6, .7,
                                        boxstyle="round,pad=0,rounding_size=.2",
                                        facecolor=BLUE, alpha=.12, edgecolor="none"))
            ax.text((start + a) / 2, .55, "session %d" % (k and 1 or 1), ha="center",
                    fontsize=11, color=BLUE, visible=False)
            start = b
    for a, b, label in [(0, 17, "session 1"), (62, 73, "session 2"), (120, 120, "session 3")]:
        ax.text((a + b) / 2, .5, label, ha="center", fontsize=11, color=BLUE)
    for a, b in [(17, 62), (73, 120)]:
        ax.annotate("", (a, -.6), (b, -.6),
                    arrowprops=dict(arrowstyle="<->", color=RED, lw=1))
        ax.text((a + b) / 2, -.95, "%d min" % (b - a), ha="center", fontsize=10, color=RED)
    ax.set_ylim(-1.3, .95)
    ax.set_xlim(-8, 130)
    ax.set_yticks([])
    ax.set_xlabel("Minutes since the first event")
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    save(fig, "figures/lec2/sessions.png")


def annotation():
    """Three people, one label, and the disagreements a majority vote hides."""
    fig, ax = plt.subplots(figsize=(5.8, 2.2), dpi=200)
    ax.set_axis_off()
    items = ["a husky in the snow", "a cassette player", "a lakeside restaurant"]
    votes = [["dog", "dog", "wolf"], ["phone", "cassette", "phone"],
             ["boathouse", "restaurant", "restaurant"]]
    stored = ["dog", "phone", "restaurant"]
    columns = [6.4, 8.4, 10.4]
    last = 13.2

    for x, head in zip(columns, ["annotator 1", "annotator 2", "annotator 3"]):
        ax.text(x, 3.4, head, fontsize=10, style="italic", ha="center")
    ax.text(0, 3.4, "the image", fontsize=10, style="italic")
    ax.text(last, 3.4, "stored label", fontsize=10, style="italic", ha="center")

    for row, (item, vote, final) in enumerate(zip(items, votes, stored)):
        y = 2.4 - row
        ax.text(0, y, item, fontsize=10, va="center")
        for x, v in zip(columns, vote):
            ax.text(x, y, v, fontsize=10, va="center", ha="center",
                    color=GREY if v == final else RED)
        ax.text(last, y, final, fontsize=10, va="center", ha="center", color=GREEN)
    ax.plot([-.3, 14.6], [3.05, 3.05], color=LIGHT, lw=1)
    ax.plot([11.6, 11.6], [-.6, 3.3], color=LIGHT, lw=1)
    ax.set_xlim(-.4, 14.8)
    ax.set_ylim(-.8, 3.8)
    save(fig, "figures/lec2/annotation.png")


if __name__ == "__main__":
    trigger()
    literary_digest()
    sessions()
    annotation()
