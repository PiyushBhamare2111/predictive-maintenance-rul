"""Shared dark visual theme (matplotlib + plotly) with the #ff8a3d accent."""
from __future__ import annotations

BG = "#0b0c0f"
PANEL = "#12141a"
GRID = "#23262e"
TEXT = "#ecebe8"
MUTED = "#9a9ea8"
ACCENT = "#ff8a3d"      # predictions / the thing we built
BLUE = "#3987e5"        # ground truth
AQUA = "#2bb58a"        # third series
RED = "#e66767"         # failure / critical
TRUE = "#ecebe8"        # ground truth lines (neutral, so models keep their colours)
MODEL_COLORS = {"gbm": AQUA, "cnn": ACCENT, "lstm": BLUE}


def use_matplotlib():
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": TEXT,
        "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlelocation": "left",
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
        "legend.frameon": False, "legend.labelcolor": TEXT, "font.size": 11,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "text.parse_math": False, "axes.axisbelow": True,
    })


def plotly_layout(**kw):
    base = dict(
        paper_bgcolor=BG, plot_bgcolor=PANEL, font=dict(color=TEXT, family="Segoe UI, sans-serif"),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, color=MUTED),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, color=MUTED),
        legend=dict(orientation="h", y=1.08, x=0), margin=dict(l=50, r=20, t=50, b=45),
        hovermode="x unified",
    )
    base.update(kw)
    return base
