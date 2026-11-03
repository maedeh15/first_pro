"""fig_style.py -- one publication style for every figure of the thesis (Okabe-Ito colour-blind-safe palette, 300 dpi PNG + vector PDF, English labels, 8 pt text,
single-column 3.4 in / double-column 7.1 in widths). Import: from fig_style import apply, save, C, METHOD_COLOR, panel, strip, mean_ci, stars."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00", sky="#56B4E9", purple="#CC79A7", yellow="#F0E442", black="#000000", grey="#8C8C8C")
METHOD_COLOR = {"attention": C["grey"], "input_gradient": C["sky"], "grad_x_input": C["purple"], "gradient_shap": C["orange"], "grad_cam": C["green"], "occlusion": C["red"],
                "integrated_gradients": C["yellow"]}
METHOD_LABEL = {"attention": "Attention", "input_gradient": "Input gradient", "grad_x_input": "Gradient x input", "gradient_shap": "GradientShap", "grad_cam": "Grad-CAM",
                "occlusion": "Occlusion", "integrated_gradients": "Integrated gradients"}
W1, W2 = 3.4, 7.1


def apply():
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "figure.dpi": 100,
                         "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
                         "lines.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans"})


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    return os.path.join(out_dir, name + ".png")


def panel(ax, letter, dx=-0.14, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def strip(ax, x, values, color="k", width=0.16, size=7, alpha=0.45, seed=0, horizontal=False):
    rng = np.random.RandomState(seed); j = rng.uniform(-width, width, len(values))
    (ax.scatter(values, x + j, s=size, color=color, alpha=alpha, zorder=3, linewidths=0) if horizontal else ax.scatter(x + j, values, s=size, color=color, alpha=alpha, zorder=3, linewidths=0))


def boot_ci(v, n=5000, seed=0):
    v = np.asarray(v, float); v = v[~np.isnan(v)]; rng = np.random.RandomState(seed)
    m = v[rng.randint(0, len(v), (n, len(v)))].mean(1); return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def mean_ci(ax, x, values, color="k", capsize=2.5, horizontal=False, lw=1.2, marker="o", ms=4):
    v = np.asarray(values, float); v = v[~np.isnan(v)]; lo, hi = boot_ci(v); m = v.mean()
    if horizontal:
        ax.errorbar(m, x, xerr=[[m - lo], [hi - m]], fmt=marker, color=color, capsize=capsize, lw=lw, ms=ms, zorder=4)
    else:
        ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt=marker, color=color, capsize=capsize, lw=lw, ms=ms, zorder=4)
    return m, lo, hi


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
