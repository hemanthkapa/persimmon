"""Cluster a pooled matrix with PCA + full-covariance GMM and check surface dependence.

For each k, fits the mixture on a random subset of rows, labels every row, and reports
BIC plus how much the clusters follow surface identity: AMI between cluster and surface
(0 = clusters mix surfaces, 1 = clusters are surfaces) and, per cluster, the effective
number of surfaces it draws from (exp of the entropy of its surface mix; max = n surfaces).

Usage: python gmm_baseline.py POOLED_NPZ [--k-min 2] [--k-max 10] [--n-fit N] [-o OUT_DIR]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.mixture import GaussianMixture


def effective_surfaces(labels, surface):
    """Per cluster: exp(entropy) of its surface distribution, and the largest surface share."""
    out = {}
    for c in np.unique(labels):
        p = np.bincount(surface[labels == c]) / (labels == c).sum()
        p = p[p > 0]
        out[int(c)] = (float(np.exp(-(p * np.log(p)).sum())), float(p.max()))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pooled", type=Path, help="pool_matrix.py output .npz")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--n-fit", type=int, default=100000, help="rows used to fit each mixture")
    parser.add_argument("--var", type=float, default=0.95, help="PCA variance kept (default: 0.95)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-o", "--output", type=Path, help="output folder (default: next to POOLED_NPZ)")
    args = parser.parse_args()

    d = np.load(args.pooled)
    X, surface = d["X"], d["index"][:, 0]
    n_surf = len(d["surfaces"])
    out = args.output or args.pooled.parent / f"gmm_{args.pooled.stem.split('_')[0]}"
    out.mkdir(parents=True, exist_ok=True)

    pca = PCA(n_components=args.var, random_state=args.seed).fit(X)
    Z = pca.transform(X)
    print(f"{args.pooled}: X={X.shape}  PCA {args.var:.0%} -> {Z.shape[1]} components")

    rng = np.random.default_rng(args.seed)
    fit_rows = rng.choice(len(Z), size=min(args.n_fit, len(Z)), replace=False)
    rows, labels_by_k = [], {}
    for k in range(args.k_min, args.k_max + 1):
        gmm = GaussianMixture(k, covariance_type="full", random_state=args.seed).fit(Z[fit_rows])
        lab = gmm.predict(Z)
        eff = effective_surfaces(lab, surface)
        effs = np.array([e for e, _ in eff.values()])
        rows.append({
            "k": k, "bic": gmm.bic(Z[fit_rows]), "ami_surface": adjusted_mutual_info_score(surface, lab),
            "min_eff_surfaces": effs.min(), "median_eff_surfaces": float(np.median(effs)),
            "max_single_surface_share": max(m for _, m in eff.values()),
            "smallest_cluster": np.bincount(lab).min() / len(lab),
        })
        labels_by_k[f"k{k}"] = lab.astype(np.int8)
        r = rows[-1]
        print(f"k={k:2d}  BIC={r['bic']:.4g}  AMI(surface)={r['ami_surface']:.3f}  "
              f"eff. surfaces min/median={r['min_eff_surfaces']:.1f}/{r['median_eff_surfaces']:.1f} of {n_surf}  "
              f"max one-surface share={r['max_single_surface_share']:.0%}  smallest={r['smallest_cluster']:.1%}")

    table = pd.DataFrame(rows)
    table.to_csv(out / "gmm_sweep.csv", index=False)
    np.savez_compressed(out / "labels.npz", index=d["index"], **labels_by_k)
    (out / "params.json").write_text(json.dumps(
        {"pooled": str(args.pooled), "n_fit": len(fit_rows), "var": args.var,
         "n_components": int(Z.shape[1]), "seed": args.seed}, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
