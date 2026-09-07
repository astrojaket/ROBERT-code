"""Shared visual palette for maintained forward-model benchmarks."""

ROBERT_COLOR = "mediumpurple"
REFERENCE_COLOR = "#2f2733"
RESIDUAL_COLOR = "rebeccapurple"
PURPLE_DARK = "darkorchid"
PURPLE_LIGHT = "plum"
PURPLE_PALETTE = (
    "mediumpurple",
    "rebeccapurple",
    "darkorchid",
    "orchid",
    "plum",
    "thistle",
)

# The default Matplotlib settings for ROBERT science and benchmark figures.
# Keep this mapping free of optional plotting-package dependencies so all
# configured tasks can use it.
ROBERT_MATPLOTLIB_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 11.0,
    "axes.titlesize": 13.0,
    "axes.labelsize": 11.0,
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 5.0,
    "ytick.major.size": 5.0,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
}

__all__ = [
    "PURPLE_DARK",
    "PURPLE_LIGHT",
    "PURPLE_PALETTE",
    "REFERENCE_COLOR",
    "RESIDUAL_COLOR",
    "ROBERT_COLOR",
    "ROBERT_MATPLOTLIB_STYLE",
]
