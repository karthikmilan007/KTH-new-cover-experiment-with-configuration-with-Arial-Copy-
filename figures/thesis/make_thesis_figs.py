"""Thesis figures for Ch2-Ch4, written as vector PDFs.

Usage: python make_thesis_figs.py <nisqa_corpus_dir> <out_dir>
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

NISQA, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
})

BLUE = "#2f6db5"      # fully labelled / SIM
ORANGE = "#e07b39"    # MOS-only / LIVE
GREY = "#555555"
LIGHT = "#f2f2f2"
QUANT = ["mos", "noi", "col", "dis", "loud"]
QLAB = ["MOS", "NOI", "COL", "DIS", "LOUD"]


def load(split):
    path = os.path.join(NISQA, split, f"{split}_file.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return np.array([[float(r[q]) for q in QUANT] for r in rows])


def box(ax, x, y, w, h, text, fc="white", ec="black", lw=1.0, ls="-",
        fontsize=9, weight="normal", color="black"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            weight=weight, color=color, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color="black", ls="-", lw=1.1, rad=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                shrinkA=0, shrinkB=0, mutation_scale=10,
                                connectionstyle=f"arc3,rad={rad}"))


def canvas(w_in, h_in, xmax, ymax):
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- Fig: two-stage model (Ch4)
def fig_two_stage():
    fig, ax = canvas(7.4, 3.2, 13.6, 5.8)
    bw, bh = 2.1, 1.05
    box(ax, 1.25, 2.9, 2.3, bh, "XLS-R features\n(layer 12)", fc=LIGHT)
    box(ax, 3.9, 2.9, bw, bh, "Shared\nconvolutional\nencoder")
    box(ax, 7.0, 4.55, bw, bh, "Stage 1\ndimension stage", fc="#e3ecf7", ec=BLUE)
    box(ax, 7.0, 1.25, bw, bh, "Stage 2\nMOS stage", fc="#fbe9dd", ec=ORANGE)
    box(ax, 11.7, 4.55, 3.6, bh, "4-D Gaussian over\nNOI, COL, DIS, LOUD", ls="--")
    box(ax, 11.7, 1.25, 3.6, bh, "1-D Gaussian\nover MOS", ls="--")

    arrow(ax, 2.4, 2.9, 2.85, 2.9)
    # encoder -> both stages
    ax.plot([4.95, 5.4], [2.9, 2.9], color="black", lw=1.1)
    ax.plot([5.4, 5.4], [1.25, 4.55], color="black", lw=1.1)
    arrow(ax, 5.4, 4.55, 5.95, 4.55)
    arrow(ax, 5.4, 1.25, 5.95, 1.25)
    ax.text(5.25, 1.75, "encoded\nfeatures $\\mathbf{h}$", fontsize=8, ha="right",
            va="center", color=GREY)
    arrow(ax, 8.05, 4.55, 9.9, 4.55)
    arrow(ax, 8.05, 1.25, 9.9, 1.25)

    # stage 1 -> stage 2 (forward) and the gradient path back (dashed)
    arrow(ax, 6.75, 4.02, 6.75, 1.78)
    ax.text(6.6, 2.9, "dimension\nvalues $\\mathbf{q}$", fontsize=8, ha="right",
            va="center")
    arrow(ax, 7.3, 1.78, 7.3, 4.02, color=ORANGE, ls="--")
    ax.text(7.45, 2.9, "MOS error of\nMOS-only clips", fontsize=8, ha="left",
            va="center", color=ORANGE)
    save(fig, "two_stage_model.pdf")


# ---------------------------------------------------------------- Fig: research process (Ch3)
def fig_process():
    steps = ["1\nReproduce\nthe baseline", "2\nBuild the\ntwo-stage model",
             "3\nDesign\nstudies", "4\nMain experiments\nA and B",
             "5\nStatistical\nevaluation"]
    fig, ax = canvas(7.0, 1.3, 13.4, 2.4)
    w, gap = 2.3, 0.45
    for i, s in enumerate(steps):
        x = 1.2 + i * (w + gap)
        box(ax, x, 1.2, w, 1.9, s, fc=LIGHT if i != 1 else "#e3ecf7",
            fontsize=7.5)
        if i < len(steps) - 1:
            arrow(ax, x + w / 2 + 0.05, 1.2, x + w / 2 + gap - 0.05, 1.2)
    save(fig, "research_process.pdf")


# ---------------------------------------------------------------- Fig: experiments A and B (Ch3)
def fig_experiments():
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 5.0),
                             gridspec_kw={"height_ratios": [1, 1.2]})
    panels = [
        ("Experiment A: separate MOS-only corpus",
         [("NISQA train (SIM + LIVE)\n11,020 clips", "all 5 labels", BLUE),
          ("PSTN_TRAIN12K\n12,000 clips", "MOS only", ORANGE)],
         [("Two-stage model", [1, 1]),
          ("MultiGauss\n(published)", [1, 0])],
         "Evaluated on: 2,000 NISQA val + 2,000 PSTN_VAL2K clips"),
        ("Experiment B: MOS-only part of the same corpus",
         [("NISQA_TRAIN_SIM\n10,000 clips", "all 5 labels", BLUE),
          ("NISQA_TRAIN_LIVE\n1,020 clips", "MOS only (dimension labels removed)", ORANGE)],
         [("Two-stage model", [1, 1]),
          ("MultiGauss\n(SIM only)", [1, 0]),
          ("MultiGauss\n(masked loss)", [1, 1])],
         "Evaluated on: NISQA val (SIM, LIVE) + 3 NISQA test sets"),
    ]
    for ax, (title, data, models, ev) in zip(axes, panels):
        n = len(models)
        ymax = 3.4 + 1.2 * n
        ax.set_xlim(0, 14)
        ax.set_ylim(0, ymax)
        ax.axis("off")
        ax.set_title(title, fontsize=9, loc="left", weight="bold")
        cols = [6.2, 11.0]
        for (name, lab, c), x in zip(data, cols):
            box(ax, x, ymax - 0.75, 4.2, 1.2, name, fc="white", ec=c, lw=1.4,
                fontsize=8)
            ax.text(x, ymax - 1.65, lab, ha="center", va="center",
                    fontsize=8, color=c, style="italic")
        top = ymax - 2.65
        for k, (mname, use) in enumerate(models):
            y = top - k * 1.2
            ax.text(0.1, y, mname, ha="left", va="center", fontsize=8.5,
                    linespacing=1.15)
            for u, x, (_, _, c) in zip(use, cols, data):
                if u:
                    ax.add_patch(FancyBboxPatch(
                        (x - 1.2, y - 0.45), 2.4, 0.9,
                        boxstyle="round,pad=0.02,rounding_size=0.15",
                        fc=c, ec=c, alpha=0.85))
                    ax.text(x, y, "trained on", ha="center", va="center",
                            fontsize=7.5, color="white")
                else:
                    ax.add_patch(FancyBboxPatch(
                        (x - 1.2, y - 0.45), 2.4, 0.9,
                        boxstyle="round,pad=0.02,rounding_size=0.15",
                        fc="white", ec="#bbbbbb", ls=":"))
                    ax.text(x, y, "not used", ha="center", va="center",
                            fontsize=7.5, color="#999999")
        ax.text(0.1, 0.3, ev, ha="left", va="center", fontsize=8,
                color=GREY)
    fig.subplots_adjust(hspace=0.25)
    save(fig, "experiments_ab.pdf")


# ---------------------------------------------------------------- Fig: correlation heatmaps (Ch2)
def fig_correlation():
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.75))
    for ax, split, title in zip(axes, ["NISQA_TRAIN_SIM", "NISQA_TRAIN_LIVE"],
                                ["NISQA_TRAIN_SIM (10,000 clips)",
                                 "NISQA_TRAIN_LIVE (1,020 clips)"]):
        corr = np.corrcoef(load(split).T)
        im = ax.imshow(corr, vmin=0, vmax=1, cmap="Blues")
        for i in range(5):
            for j in range(5):
                v = corr[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if v > 0.6 else "black")
        ax.set_xticks(range(5), QLAB)
        ax.set_yticks(range(5), QLAB)
        ax.set_title(title)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03)
    cb.set_label("Pearson correlation")
    cb.outline.set_visible(False)
    save(fig, "mos_dim_correlation.pdf")


# ---------------------------------------------------------------- Fig: SIM vs LIVE violins (Ch3)
def fig_violins():
    sim, live = load("NISQA_TRAIN_SIM"), load("NISQA_TRAIN_LIVE")
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    pos_s = np.arange(5) * 2.6
    pos_l = pos_s + 1.0
    for data, pos, c in [(sim, pos_s, BLUE), (live, pos_l, ORANGE)]:
        parts = ax.violinplot([data[:, k] for k in range(5)], positions=pos,
                              widths=0.9, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor(c)
            b.set_edgecolor(c)
            b.set_alpha(0.55)
        for k in range(5):
            q1, med, q3 = np.percentile(data[:, k], [25, 50, 75])
            ax.plot([pos[k], pos[k]], [q1, q3], color="black", lw=2.2)
            ax.plot(pos[k], med, "o", ms=3.5, mfc="white", mec="black",
                    mew=0.8)
    ax.set_xticks(pos_s + 0.5, QLAB)
    ax.set_ylabel("Label (1–5 scale)")
    ax.set_ylim(0.9, 5.1)
    ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=BLUE, alpha=0.55),
               plt.Rectangle((0, 0), 1, 1, fc=ORANGE, alpha=0.55)]
    ax.legend(handles, ["NISQA_TRAIN_SIM", "NISQA_TRAIN_LIVE"],
              loc="lower left", ncol=2, frameon=False, fontsize=8,
              bbox_to_anchor=(0.0, 1.0))
    save(fig, "sim_live_distributions.pdf")


fig_two_stage()
fig_process()
fig_experiments()
fig_correlation()
fig_violins()
