"""Chapter 5 result figures (vector PDFs).

All numbers are copied from the experiment summaries:
  full_data_batch_ratio_noseed/_summary/fdbr_summary.csv        (Fig. ratio, panel a)
  sim_live_batch_ratio_noseed/_summary/results.md               (Fig. ratio, panel b)
  multigauss_low_data_noseed/_summary/comparison_summary.csv    (Fig. low data)
  nan_mask_baseline_noseed/_summary/results.md                  (Fig. Experiment B)
Usage: python make_ch5_figs.py <out_dir>
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "pdf.fonttype": 42,
})
BLUE, ORANGE, GREEN, GREY = "#2f6db5", "#e07b39", "#3a9a5b", "#777777"


def style(ax):
    ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- batch ratio
def fig_ratio():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.2, 2.5))
    # (a) Experiment A setting: PSTN clips per 64-clip batch
    pstn = [11, 18, 24, 28, 32, 35, 37, 43, 48]
    mos = [0.916, 0.912, 0.909, 0.915, 0.913, 0.913, 0.910, 0.913, 0.913]
    avg = [0.828, 0.826, 0.828, 0.825, 0.828, 0.825, 0.827, 0.828, 0.828]
    a.plot(pstn, mos, "o-", color=BLUE, ms=3.5, label="MOS")
    a.plot(pstn, avg, "s-", color=GREEN, ms=3.5, label="5-quantity average")
    a.axhline(0.912, color=BLUE, ls="--", lw=0.9)
    a.axhline(0.833, color=GREEN, ls="--", lw=0.9)
    a.text(48.5, 0.9135, "MultiGauss", fontsize=7, color=BLUE, ha="right",
           va="bottom")
    a.text(48.5, 0.8345, "MultiGauss", fontsize=7, color=GREEN, ha="right",
           va="bottom")
    a.set_xlabel("PSTN clips per batch of 64")
    a.set_ylabel("PCC, pooled NISQA test sets")
    a.set_title("(a) NISQA + PSTN", loc="left")
    a.set_ylim(0.80, 0.93)
    style(a)
    a.legend(loc="center right", frameon=False)
    # (b) Experiment B setting: LIVE clips per 64-clip batch
    live = [1, 8, 16, 32, 48]
    vl_mos = [0.857, 0.856, 0.857, 0.868, 0.864]
    t_avg = [0.823, 0.824, 0.825, 0.821, 0.823]
    b.plot(live, vl_mos, "o-", color=ORANGE, ms=3.5,
           label="MOS on NISQA_VAL_LIVE")
    b.plot(live, t_avg, "s-", color=GREEN, ms=3.5,
           label="5-quantity avg., test sets")
    b.axhline(0.794, color=ORANGE, ls="--", lw=0.9)
    b.axhline(0.831, color=GREEN, ls="--", lw=0.9)
    b.text(48.5, 0.7955, "MultiGauss (SIM only)", fontsize=7, color=ORANGE,
           ha="right", va="bottom")
    b.text(48.5, 0.8325, "MultiGauss (SIM only)", fontsize=7, color=GREEN,
           ha="right", va="bottom")
    b.set_xlabel("LIVE (MOS-only) clips per batch of 64")
    b.set_ylabel("PCC")
    b.set_title("(b) SIM + LIVE", loc="left")
    b.set_ylim(0.77, 0.89)
    style(b)
    b.legend(loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(w_pad=1.5)
    save(fig, "batch_ratio.pdf")


# ---------------------------------------------------------------- low data
def fig_low_data():
    nisqa = [100, 150, 200, 250, 300, 350, 400]
    ours_mos = [0.855, 0.857, 0.859, 0.867, 0.853, 0.858, 0.871]
    ours_mos_sd = [0.014, 0.012, 0.011, 0.007, 0.026, 0.020, 0.014]
    mg_mos = [0.828, 0.832, 0.838, 0.850, 0.853, 0.844, 0.856]
    mg_mos_sd = [0.020, 0.024, 0.038, 0.027, 0.017, 0.016, 0.012]
    ours_avg = [0.711, 0.713, 0.742, 0.748, 0.747, 0.752, 0.764]
    mg_avg = [0.741, 0.750, 0.754, 0.773, 0.772, 0.761, 0.773]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.2, 2.4))
    a.errorbar(nisqa, ours_mos, yerr=ours_mos_sd, fmt="o-", color=BLUE,
               ms=3.5, capsize=2, lw=1.1, label="two-stage (+ PSTN)")
    a.errorbar(nisqa, mg_mos, yerr=mg_mos_sd, fmt="s--", color=GREY, ms=3.5,
               capsize=2, lw=1.1, label="MultiGauss")
    a.set_title("(a) MOS", loc="left")
    b.plot(nisqa, ours_avg, "o-", color=BLUE, ms=3.5, lw=1.1,
           label="two-stage (+ PSTN)")
    b.plot(nisqa, mg_avg, "s--", color=GREY, ms=3.5, lw=1.1,
           label="MultiGauss")
    b.set_title("(b) Average over the five quantities", loc="left")
    for ax in (a, b):
        ax.set_xlabel("Fully labelled NISQA clips (of 500 in total)")
        ax.set_ylabel("PCC, pooled NISQA test sets")
        style(ax)
    a.legend(loc="lower right", frameon=False)
    fig.tight_layout(w_pad=1.5)
    save(fig, "low_data.pdf")


# ---------------------------------------------------------------- Experiment B deltas
def fig_exp_b():
    dims = ["MOS", "NOI", "COL", "DIS", "LOUD"]
    # (delta, significant) vs MultiGauss trained on SIM only
    data = {
        "NISQA_VAL_LIVE (200 clips)": {
            "two-stage": [(.073, 1), (-.028, 0), (.013, 0), (-.034, 0), (.102, 1)],
            "masked": [(.054, 1), (.003, 0), (.036, 0), (-.025, 0), (.089, 1)],
        },
        "NISQA test sets, pooled (712 clips)": {
            "two-stage": [(-.006, 0), (.003, 0), (-.001, 0), (-.012, 0), (-.034, 1)],
            "masked": [(.002, 0), (.003, 0), (.005, 0), (.012, 0), (-.000, 0)],
        },
    }
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.5), sharey=True)
    x = np.arange(5)
    w = 0.36
    for ax, (title, d) in zip(axes, data.items()):
        for off, key, c, lab in [(-w / 2, "two-stage", BLUE, "two-stage model"),
                                 (w / 2, "masked", ORANGE,
                                  "MultiGauss, masked loss")]:
            vals = [v for v, _ in d[key]]
            ax.bar(x + off, vals, w, color=c, label=lab)
            for xi, (v, s) in zip(x + off, d[key]):
                if s:
                    ax.text(xi, v + (0.004 if v >= 0 else -0.012), "*",
                            ha="center", va="bottom", fontsize=10)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, dims)
        ax.set_title(title, loc="left")
        ax.tick_params(axis="x", length=0)
        style(ax)
    axes[0].set_ylabel("PCC difference from\nMultiGauss (SIM only)")
    axes[0].set_ylim(-0.06, 0.125)
    axes[0].legend(loc="upper center", frameon=False, ncol=1)
    fig.tight_layout(w_pad=1.0)
    save(fig, "exp_b_deltas.pdf")


fig_ratio()
fig_low_data()
fig_exp_b()
