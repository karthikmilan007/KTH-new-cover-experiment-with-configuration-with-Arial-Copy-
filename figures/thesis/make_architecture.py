"""Layer-level architecture of the two-stage model (Chapter 4).
Usage: python make_architecture.py <out_dir> [--preview]"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                     "font.size": 7.3, "pdf.fonttype": 42})
GREY, BLUE, ORANGE = "#f2f2f2", "#dce8f6", "#fbe3d3"
EB, EO, EG = "#2f6db5", "#d0682a", "#555555"

fig, ax = plt.subplots(figsize=(6.3, 4.6))
ax.set_xlim(0, 104)
ax.set_ylim(0, 74)
ax.axis("off")


def region(x0, y0, x1, y1, fc, ec, label):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                 boxstyle="round,pad=0,rounding_size=1.5", fc=fc, ec=ec,
                 lw=0.8, alpha=0.35, zorder=0))
    ax.text(x0 + 1.5, y1 - 1, label, color=ec, fontweight="bold",
            fontsize=8.3, ha="left", va="top")


def box(x, y, w, h, text, fc="white", ec=EG, lw=0.8):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0.2,rounding_size=0.8", fc=fc, ec=ec,
                 lw=lw, zorder=2))
    ax.text(x, y, text, ha="center", va="center", linespacing=1.2, zorder=3)


def arrow(x0, y0, x1, y1, c="black", ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=c, lw=0.9, ls=ls,
                                mutation_scale=8))


def line(xs, ys, c="black"):
    ax.plot(xs, ys, color=c, lw=0.9, zorder=1)


def note(x, y, text, c="#333333", **kw):
    ax.text(x, y, text, fontsize=6.8, color=c, style="italic", va="center",
            **kw)


region(1, 4, 27, 72, GREY, EG, "Shared encoder")
region(30, 47, 97, 72, BLUE, EB, "Stage 1: dimension stage")
region(30, 2, 97, 44, ORANGE, EO, "Stage 2: MOS stage")

# ---------------- shared encoder (top to bottom)
box(14, 63, 21, 8, "XLS-R features\n(layer 12, 10 s)\n$1920 \\times 500$",
    fc=GREY)
conv = "Conv1d (32, kernel 5)\nLayerNorm, ReLU\nDropout 0.3\nMax-pool 5"
box(14, 47, 21, 11, conv)
box(14, 29, 21, 11, conv)
box(14, 12, 21, 7, "Flatten\n$\\mathbf{h} \\in \\mathbb{R}^{608}$")
arrow(14, 58.8, 14, 52.7)
arrow(14, 41.3, 14, 34.7)
note(15, 38, "$32 \\times 99$")
arrow(14, 23.3, 14, 15.7)
note(15, 19.5, "$32 \\times 19$")

# ---------------- stage 1 (left to right)
Y1 = 59
box(39, Y1, 10, 7, "Dense 64\nReLU")
box(51, Y1, 10, 7, "Dense 32\nReLU")
box(62.5, Y1, 9, 7, "Dense 14")
box(83, Y1, 20, 12,
    "$\\boldsymbol{\\mu}_d$ (4), $\\mathbf{L}_d$ (10)\n"
    "$\\mathcal{N}(2\\boldsymbol{\\mu}_d+3,\\,4\\mathbf{L}_d\\mathbf{L}_d^{\\top})$\n"
    "NOI, COL, DIS, LOUD", ec=EB, lw=1.2)
line([24.7, 29.5], [12, 12])
line([29.5, 29.5], [12, Y1])
arrow(29.5, Y1, 33.8, Y1)
arrow(44.2, Y1, 45.8, Y1)
arrow(56.2, Y1, 57.8, Y1)
arrow(67.2, Y1, 72.8, Y1)

# ---------------- stage 2: quality branch (right to left)
YQ = 33
box(83, YQ, 20, 9,
    "$\\mathbf{q} \\in \\mathbb{R}^4$\nlabelled: true values\n"
    "MOS-only, test: $\\boldsymbol{\\mu}_d$", ec=EO, lw=1.0)
box(64, YQ, 10, 7, "Dense 32\nReLU")
box(51, YQ, 10, 7, "Dense 16")
arrow(83, Y1 - 6.2, 83, YQ + 4.7, c=EB)
note(84, 45.5, "$\\boldsymbol{\\mu}_d$", c=EB)
arrow(72.8, YQ, 69.2, YQ)
arrow(58.8, YQ, 56.2, YQ)

# ---------------- stage 2: fusion head (left to right)
Y2 = 12
box(39, Y2, 10, 7, "Concatenate\n$608 + 16$")
box(51, Y2, 10, 7, "Dense 64\nReLU")
box(62, Y2, 10, 7, "Dense 32\nReLU")
box(72, Y2, 8, 7, "Dense 2")
box(85.5, Y2, 15, 10,
    "$\\mu_m$, $\\sigma_m^2$ (softplus)\n"
    "$\\mathcal{N}(2\\mu_m+3,\\,4\\sigma_m^2)$\nMOS", ec=EO, lw=1.2)
arrow(29.5, Y2, 33.8, Y2)
line([45.8, 39], [YQ, YQ])
arrow(39, YQ, 39, Y2 + 3.7)
arrow(44.2, Y2, 45.8, Y2)
arrow(56.2, Y2, 56.8, Y2)
arrow(67.2, Y2, 67.8, Y2)
arrow(76.2, Y2, 77.8, Y2)

# ---------------- MOS-only gradient path (dashed, outside the regions)
ax.annotate("", xy=(93.5, Y1 - 6.2), xytext=(93.5, Y2 + 5.2), zorder=1,
            arrowprops=dict(arrowstyle="-|>", color=EO, lw=0.9, ls="--",
                            mutation_scale=8))
note(99.5, 37, "error of MOS-only clips\nflows back to Stage 1",
     c=EO, rotation=90, ha="center")

fig.savefig(os.path.join(OUT, "architecture.pdf"), bbox_inches="tight",
            pad_inches=0.02)
if "--preview" in sys.argv:
    fig.savefig(os.path.join(OUT, "architecture_preview.png"), dpi=170,
                bbox_inches="tight")
