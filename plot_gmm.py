"""Plot a GMM sweep and the pooled feature space for one membrane.

Writes two PNGs next to the sweep: sweep.png (BIC, AMI vs surface and smallest cluster
against k, one line per smoothing mode) and pca_k<K>.png (PCA axes 1-2, one small panel
per cluster and one per surface, each highlighted over all points in gray).

Usage: python plot_gmm.py BASELINE_DIR [--mem IMM] [--k 4] [--modes sigma5 fill_only]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Reference palette, light mode (categorical slots 1-2), surface and text tokens.
SERIES = ("#2a78d6", "#eb6834")
SURFACE, TEXT, TEXT_2, GRID, BACKDROP = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#c9c8c2"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


def plot_sweep(base, mem, modes, out):
    """Three small multiples against k; one line per mode, legend plus direct end labels."""
    panels = [("bic", "BIC (lower = better fit)", 1e-6, "×10⁶"),
              ("ami_surface", "AMI, cluster vs surface (0 = mixed)", 1, ""),
              ("smallest_cluster", "Smallest cluster, % of vertices", 100, "%")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for (col, title, scale, unit), ax in zip(panels, axes):
        ends = {}
        for mode, color in zip(modes, SERIES):
            t = pd.read_csv(base / mode / f"gmm_{mem}" / "gmm_sweep.csv")
            y = t[col] * scale
            ax.plot(t["k"], y, color=color, lw=2, marker="o", ms=5, label=mode)
            ends[mode] = y.iloc[-1]
        # Direct end labels, nudged apart vertically so close lines don't collide.
        for rank, mode in enumerate(sorted(ends, key=ends.get, reverse=True)):
            ax.annotate(mode, (10, ends[mode]), xytext=(6, 6 if rank == 0 else -6),
                        textcoords="offset points", va="center", color=TEXT_2, fontsize=9)
        ax.set_title(title, loc="left", fontsize=10, color=TEXT)
        ax.set_xlabel("k (number of clusters)")
        ax.set_xticks(range(2, 11))
        ax.set_xlim(1.7, 11.6)
        if unit == "×10⁶":
            ax.set_ylabel("BIC ×10⁶")
    axes[1].set_ylim(0, 0.1)
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle(f"{mem} GMM sweep: fit, surface dependence and smallest cluster by k",
                 x=0.01, ha="left", fontsize=12)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_select_k(summary, mem, out):
    """Two panels against k: seed stability, and mesh coherence next to its baselines."""
    t = pd.read_csv(summary)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    a1.plot(t["k"], t["ari_mean"], color=SERIES[0], lw=2, marker="o", ms=5, label="mean")
    a1.plot(t["k"], t["ari_min"], color=SERIES[1], lw=2, marker="o", ms=5, label="worst pair")
    a1.set_title("Stability across 5 seeds and fit subsets (ARI, 1 = identical)", loc="left", fontsize=10)
    a1.set_ylim(0, 1)
    a1.legend(frameon=False, loc="lower right")
    for col, color, name in (("coherence", SERIES[0], "clusters"), ("null", SERIES[1], "smoothing-only null")):
        a2.plot(t["k"], t[col], color=color, lw=2, marker="o", ms=5, label=name)
    a2.plot(t["k"], t["chance"], color=TEXT_2, lw=1.5, ls="--", label="chance (cluster sizes)")
    a2.set_title("Mesh coherence: edges with matching labels (median, 24 surfaces)", loc="left", fontsize=10)
    a2.set_ylim(0, 1)
    a2.legend(frameon=False, loc="lower left")
    for ax in (a1, a2):
        ax.set_xlabel("k (number of clusters)")
        ax.set_xticks(t["k"])
    fig.suptitle(f"{mem} choosing k", x=0.01, ha="left", fontsize=12)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def highlight_grid(Z, groups, names, title, out, ncols, n_show, axis="PC"):
    """Small multiples: each panel shows one group in blue over every point in gray."""
    n = len(names)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.5 * nrows),
                             sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    lim = np.percentile(Z, [0.5, 99.5], axis=0)
    for i, ax in enumerate(axes):
        if i >= n:
            ax.set_visible(False)
            continue
        ax.scatter(Z[:, 0], Z[:, 1], s=1, c=BACKDROP, linewidths=0, rasterized=True)
        m = groups == i
        pick = np.flatnonzero(m)[:n_show]
        ax.scatter(Z[pick, 0], Z[pick, 1], s=1.5, c=SERIES[0], linewidths=0, rasterized=True)
        ax.set_title(f"{names[i]}  ({m.mean():.1%})", fontsize=8.5, loc="left", color=TEXT)
        ax.set_xlim(lim[:, 0]); ax.set_ylim(lim[:, 1])
        ax.tick_params(labelsize=7)
    fig.supxlabel(f"{axis} 1", color=TEXT_2, fontsize=9)
    fig.supylabel(f"{axis} 2", color=TEXT_2, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base", type=Path, help="e.g. data/baseline/YTC041_1")
    parser.add_argument("--mem", default="IMM", type=str.upper)
    parser.add_argument("--k", type=int, default=4, help="k for the PCA scatter (default: 4)")
    parser.add_argument("--modes", nargs="+", default=["sigma5", "fill_only"])
    parser.add_argument("--n-plot", type=int, default=40000, help="points drawn per panel")
    args = parser.parse_args()

    out_dir = args.base / "plots"
    out_dir.mkdir(exist_ok=True)
    plot_sweep(args.base, args.mem, args.modes, out_dir / f"{args.mem}_sweep.png")

    mode = args.modes[0]
    summary = args.base / mode / f"select_k_{args.mem}" / "summary.csv"
    if summary.exists():
        plot_select_k(summary, args.mem, out_dir / f"{args.mem}_{mode}_select_k.png")
    d = np.load(args.base / mode / f"{args.mem}_pooled.npz")
    lab = np.load(args.base / mode / f"gmm_{args.mem}" / "labels.npz")[f"k{args.k}"]
    Z = PCA(n_components=2, random_state=0).fit_transform(d["X"])
    rows = np.random.default_rng(0).permutation(len(Z))[:args.n_plot]
    Z, lab, surf = Z[rows], lab[rows], d["index"][rows, 0]

    highlight_grid(Z, lab, [f"cluster {c}" for c in range(args.k)],
                   f"{args.mem} {mode}, GMM k={args.k}: each cluster (blue) over all vertices, PCA 1-2",
                   out_dir / f"{args.mem}_{mode}_pca_k{args.k}_clusters.png", ncols=min(args.k, 5), n_show=args.n_plot)
    short = [s.split("_labels")[0].replace("YTC041_1_", "") for s in d["surfaces"]]
    highlight_grid(Z, surf, short,
                   f"{args.mem} {mode}: each surface (blue) over all vertices, PCA 1-2",
                   out_dir / f"{args.mem}_{mode}_pca_surfaces.png", ncols=6, n_show=args.n_plot)
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
