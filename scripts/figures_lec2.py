"""Regenerate the figures of lecture 2 that are drawn from the penguins data.

Usage: uv run python scripts/figures_lec2.py
"""

import matplotlib.pyplot as plt
import pandas as pd

GREY = "#354046"
COLS = ["species", "island", "bill_length_mm", "body_mass_g", "sex"]
HEADER = ["", "species", "island", "bill length (mm)", "body mass (g)", "sex"]


def cell(value):
    if pd.isna(value):
        return "NaN"
    return f"{value:.1f}" if isinstance(value, float) else str(value)


def table(rows, header, missing, path, figsize):
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=header, cellLoc="right", loc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.35)

    for (r, c), box in t.get_celld().items():
        box.set_linewidth(0)
        box.set_text_props(color=GREY)
        if r == 0:
            box.set_facecolor("#e8e8e8")
            box.set_text_props(weight="bold")
        elif (r, c) in missing:
            box.set_facecolor("#fdecea")
            box.set_text_props(color="#c0392b")
        else:
            box.set_facecolor("#ffffff" if r % 2 else "#f5f5f5")

    t.auto_set_column_width(col=list(range(len(header))))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print("wrote", path)


def head_and_tail(df):
    """The first and last records, with an ellipsis row in between."""
    rows = []
    for idx in [0, 1, 2, 3, 4]:
        rows.append([str(idx)] + [cell(v) for v in df.loc[idx, COLS]])
    rows.append(["..."] * (len(COLS) + 1))
    for idx in [339, 340, 341, 342, 343]:
        rows.append([str(idx)] + [cell(v) for v in df.loc[idx, COLS]])
    return rows


def with_missing(df):
    """The records holding at least one missing entry, and the cells to highlight."""
    cols = ["species", "island", "bill_length_mm", "bill_depth_mm",
            "flipper_length_mm", "body_mass_g", "sex"]
    header = ["", "species", "island", "bill len.", "bill depth", "flipper len.",
              "body mass", "sex"]
    rows, missing = [], set()
    for r, (idx, record) in enumerate(df[df.isna().any(axis=1)][cols].iterrows(), start=1):
        row = [str(idx)]
        for c, value in enumerate(record, start=1):
            if pd.isna(value):
                missing.add((r, c))
            row.append(cell(value))
        rows.append(row)
    return rows, header, missing


if __name__ == "__main__":
    df = pd.read_csv("data/penguins.csv")
    table(head_and_tail(df), HEADER, set(), "figures/lec2/penguins-tabular.png", (7.4, 3.3))
    rows, header, missing = with_missing(df)
    table(rows, header, missing, "figures/lec2/penguins-missing.png", (8.6, 3.0))
