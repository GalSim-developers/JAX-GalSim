"""Shared matplotlib settings for the JAX-GalSim paper figures."""

import matplotlib as mpl

# openjournal.cls (two-column, 10pt)
TEXT_WIDTH = 7.1  # inches, full page width (figure*)
COLUMN_WIDTH = (7.1 - 0.3125) / 2  # inches, one column

DPI = 300

# residual colormap (diverging, neutral midpoint) and image colormap (sequential)
RES_CMAP = "RdBu_r"
IMAGE_CMAP = "gray_r"


def set_style():
    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            # LaTeX for all text, with the paper's font (T1 Computer Modern, see macros.sty)
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsmath}",
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def clean_axes(axes):
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
