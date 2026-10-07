"""Choose k for the pooled PCA + GMM by stability, mesh coherence and cluster profiles.

For each k:
  stability  fits the mixture with several seeds, each on its own random subset of rows,
             and reports the pairwise ARI between their labelings of the pooled rows.
  coherence  labels every triangle of every surface with the seed-0 model and reports the
             fraction of mesh edges whose two triangles share a label, next to two
             baselines: chance (sum of squared cluster shares) and a smoothing-only null
             (a random field smoothed like features.py, cut into classes of the same
             shares). Coherence above the null is structure beyond the smoothing itself.
  profiles   median raw-unit features per cluster (seed-0 model, pooled rows).

Usage: python select_k.py POOLED_NPZ RESULTS_DIR [--k-min 3] [--k-max 10] [--seeds 5]
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from features import diffusion_operator, load_mesh, n_steps
from pool_matrix import features_dir_of, surface_matrix

PROFILE = ("shape_index_VV", "curvedness_VV", "mean_curvature_VV", "thickness",
           "self_dist_min", "self_dist_min_valid", "OMM_dist", "IMM_dist", "OMM_orientation",
           "IMM_orientation", "verticality")


def fit_gmm(Z, k, seed):
    """Full-covariance GMM; retries with more regularisation if a covariance is singular."""
    for reg in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
        try:
            return GaussianMixture(k, covariance_type="full", random_state=seed, reg_covar=reg).fit(Z)
        except ValueError as e:
            print(f"  k={k} seed={seed}: reg_covar={reg:g} failed ({str(e)[:60]}...), retrying", flush=True)
    raise RuntimeError(f"GMM k={k} seed={seed} failed at every reg_covar")


def edge_agreement(labels, edges):
    return float((labels[edges[:, 0]] == labels[edges[:, 1]]).mean())


def smoothing_null(field, shares, edges):
    """Cut a smoothed random field into classes with the given shares; edge agreement."""
    cuts = np.quantile(field, np.clip(np.cumsum(shares)[:-1], 0, 1))  # cumsum can round past 1
    return edge_agreement(np.searchsorted(cuts, field), edges)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pooled", type=Path, help="pool_matrix.py output .npz")
    parser.add_argument("results_dir", type=Path, help="folder with the surfaces' .AVV_rh9.gt")
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n-fit", type=int, default=100000)
    parser.add_argument("--var", type=float, default=0.95)
    args = parser.parse_args()

    d = np.load(args.pooled)
    mem = args.pooled.stem.split("_")[0]
    mode_dir = args.pooled.resolve().parent
    feat_dir = features_dir_of(args.pooled, d)
    sigma = json.loads((feat_dir / "params.json").read_text()).get("sigma", 0.0)
    out = mode_dir / f"select_k_{mem}"
    out.mkdir(exist_ok=True)
    ks = range(args.k_min, args.k_max + 1)

    pca = PCA(n_components=args.var, random_state=0).fit(d["X"])
    Z = pca.transform(d["X"])
    print(f"{args.pooled}: {Z.shape[1]} PCA components; k={list(ks)}; {args.seeds} seeds", flush=True)

    # Stability across seeds and fit subsets.
    models, stab = {}, []
    for k in ks:
        labs = []
        for s in range(args.seeds):
            rows = np.random.default_rng(s).choice(len(Z), size=min(args.n_fit, len(Z)), replace=False)
            gmm = fit_gmm(Z[rows], k, s)
            labs.append(gmm.predict(Z))
            if s == 0:
                models[k] = gmm
        ari = [adjusted_rand_score(a, b) for a, b in combinations(labs, 2)]
        stab.append({"k": k, "ari_mean": np.mean(ari), "ari_min": np.min(ari)})
        print(f"k={k:2d}  stability ARI mean={stab[-1]['ari_mean']:.3f} min={stab[-1]['ari_min']:.3f}", flush=True)
    joblib.dump({"pca": pca, "gmm": models}, out / "models.joblib")

    # Mesh coherence on every triangle of every surface, with both baselines.
    coh, full = [], {}
    rng = np.random.default_rng(0)
    for name in d["surfaces"]:
        stem = name.split("YTC041_1_")[-1].split("_labels")[0]
        edges, n, xyz = load_mesh(str(args.results_dir / name.replace("_AVV_rh9_vec.parquet", ".AVV_rh9.gt")))
        Zs = pca.transform(surface_matrix(pd.read_parquet(feat_dir / name), d))
        steps = n_steps(edges, xyz, sigma)
        P = diffusion_operator(edges, n)
        field = rng.standard_normal(n)
        for _ in range(steps):
            field = P @ field
        for k in ks:
            lab = models[k].predict(Zs).astype(np.int8)
            full[f"{stem}_k{k}"] = lab
            shares = np.bincount(lab, minlength=k) / n
            coh.append({"surface": stem, "k": k, "coherence": edge_agreement(lab, edges),
                        "chance": float((shares ** 2).sum()), "null": smoothing_null(field, shares, edges)})
        print(f"  {stem}: n={n}, smoothing steps={steps}", flush=True)
    coh = pd.DataFrame(coh)
    coh.to_csv(out / "coherence_per_surface.csv", index=False)
    np.savez_compressed(out / "labels_full.npz", **full)

    # Profiles in raw units (pooled rows, seed-0 model).
    names = list(d["feature_names"])
    raw = pd.DataFrame(d["X"] / d["weights"] * d["stds"] + d["means"], columns=names)
    cols = [c for c in PROFILE if c in names]
    dens = [c for c in names if c.startswith("density_")]
    for k in ks:
        lab = models[k].predict(Z)
        prof = raw[cols].groupby(lab).median()
        prof.insert(0, "share", np.bincount(lab) / len(lab))
        prof["density_mean"] = raw[dens].mean(axis=1).groupby(lab).median()
        prof.round(4).to_csv(out / f"profiles_k{k}.csv", index_label="cluster")

    summary = (pd.DataFrame(stab).merge(
        coh.assign(excess=coh.coherence - coh.null).groupby("k")
           .agg(coherence=("coherence", "median"), null=("null", "median"), chance=("chance", "median"),
                excess_median=("excess", "median"), surfaces_above_null=("excess", lambda e: int((e > 0).sum()))),
        on="k"))
    summary.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 160)
    print("\n" + summary.round(3).to_string(index=False))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
