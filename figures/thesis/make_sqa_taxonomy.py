"""SQA taxonomy figure for Chapter 2. Usage: python make_sqa_taxonomy.py <out_dir>"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                     "font.size": 8.5, "pdf.fonttype": 42})
BLUE, LIGHT, GREY = "#2f6db5", "#dce8f6", "#f2f2f2"

fig, ax = plt.subplots(figsize=(6.2, 3.3))
ax.set_xlim(-1, 103); ax.set_ylim(0, 56); ax.axis("off")

def box(x, y, w, h, text, fc=GREY, ec="#555555", bold=False, lw=0.8):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec, lw=lw))
    ax.text(x, y, text, ha="center", va="center",
            fontweight="bold" if bold else "normal", linespacing=1.15)

def link(x0, y0, x1, y1):
    ym = (y0 + y1) / 2
    ax.plot([x0, x0, x1, x1], [y0, ym, ym, y1], color="#555555", lw=0.8)

H = 7.0
# level 0
box(50, 51, 34, H, "Speech quality assessment", bold=True)
# level 1
box(22, 38, 30, H, "Subjective\n(listening tests)")
box(70, 38, 30, H, "Objective\n(computational models)")
link(50, 51 - H / 2, 22, 38 + H / 2); link(50, 51 - H / 2, 70, 38 + H / 2)
# level 2 subjective
box(11, 24, 19, H + 1, "Laboratory\nP.800 ACR,\nP.804 dimensions")
box(33, 24, 19, H + 1, "Crowdsourced\nP.808")
link(22, 38 - H / 2, 11, 24 + (H + 1) / 2); link(22, 38 - H / 2, 33, 24 + (H + 1) / 2)
# level 2 objective
box(56, 24, 22, H + 1, "Intrusive\n(needs reference)\nPESQ, POLQA")
box(84, 24, 24, H + 1, "Non-intrusive\n(degraded signal only)")
link(70, 38 - H / 2, 56, 24 + (H + 1) / 2); link(70, 38 - H / 2, 84, 24 + (H + 1) / 2)
# level 3 non-intrusive
box(68, 8, 18, H + 2, "Signal-based\nP.563")
box(91, 8, 22, H + 3, "Data-driven\nMOSNet, DNSMOS,\nNISQA, MultiGauss,\nthis thesis",
    fc=LIGHT, ec=BLUE, lw=1.3)
link(84, 24 - (H + 1) / 2, 68, 8 + (H + 2) / 2); link(84, 24 - (H + 1) / 2, 91, 8 + (H + 3) / 2)
# dashed arrow: subjective labels train data-driven models
ax.annotate("", xy=(79.4, 3.2), xytext=(33, 19.4),
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.9, ls="--",
                            connectionstyle="arc3,rad=0.35"))
ax.text(30, 6.5, "rated corpora used\nas training labels", color=BLUE,
        ha="center", va="center", fontsize=7.5, style="italic")
fig.savefig(os.path.join(OUT, "sqa_taxonomy.pdf"), bbox_inches="tight", pad_inches=0.02)

