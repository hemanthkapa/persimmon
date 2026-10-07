"""Pool one membrane's feature vectors across surfaces into a clustering matrix.

Samples vertices from every features.py surface, z-scores each column with one scaler
fit on the pooled sample, and weights each feature group by 1/sqrt(its column count) so
the 81 density bins count as much as one group, not 81 features.

--no-flags drops every <col>_valid column; --no-self-dist also drops self_dist_min/far,
which still say whether a normal ray hit another membrane (no hit = --dist-max). Either
writes to <mode>_<tag>/ so the default matrix is kept.

Usage: python pool_matrix.py FEATURES_DIR [--mem IMM|OMM] [--no-flags] [--no-self-dist] [-o OUT]
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# Not membrane properties: triangle size follows the mesh, and orientation_class is a
# categorical copy of verticality.
EXCLUDE = ("area", "orientation_class")
CURVATURE = ("kappa_1", "kappa_2", "gauss_curvature_VV", "mean_curvature_VV",
             "shape_index_VV", "curvedness_VV")
BILAYER = ("thickness", "offset", "bilayer_resolution")
# A flag that is 1 on more than this fraction of rows is near-constant; z-scoring it
# turns its few zeros into extreme outliers.
MAX_FLAG_ONES = 0.99


def group_of(col):
    """Feature group for a column; a <col>_valid flag joins its column's group."""
    base = col.removesuffix("_valid")
    if base.startswith("density_"):
        return "density"
    if base.startswith("emb_"):
        return "embedding"
    if base in CURVATURE:
        return "curvature"
    if base in BILAYER:
        return "bilayer"
    if base.startswith("self_dist") or re.fullmatch(r"[IO]MM_dist", base):
        return "spacing"
    if base == "verticality" or re.fullmatch(r"[IO]MM_orientation", base):
        return "orientation"
    return "other"


def load_sample(paths, n_per_surface, rng):
    """Stack a random vertex sample from each surface; missing flags mean fully valid."""
    frames, index = [], []
    for i, p in enumerate(paths):
        v = pd.read_parquet(p)
        rows = np.sort(rng.choice(len(v), size=min(n_per_surface, len(v)), replace=False))
        frames.append(v.iloc[rows].reset_index(drop=True))
        index.append(np.column_stack([np.full(len(rows), i), rows]))
    df = pd.concat(frames, ignore_index=True)
    flags = [c for c in df.columns if c.endswith("_valid")]
    df[flags] = df[flags].fillna(1.0)
    return df, np.concatenate(index)


def features_dir_of(pooled_path, d):
    """features.py folder a pooled matrix was built from (older files: from its path)."""
    if "features_dir" in d:
        return Path(str(d["features_dir"]))
    mode_dir = Path(pooled_path).resolve().parent
    return mode_dir.parents[2] / "features" / mode_dir.parent.name / mode_dir.name


def surface_matrix(vec, d):
    """Apply the pooled column selection, scaler and weights to one surface's vector."""
    names = list(d["feature_names"])
    for c in names:
        if c.endswith("_valid") and c not in vec.columns:
            vec[c] = 1.0
    return (vec[names].to_numpy(float) - d["means"]) / d["stds"] * d["weights"]


def neighbor_mix(X, surface, k, n_query, rng):
    """Fraction of each point's k nearest feature-space neighbors from its own surface.

    Near 1 means kNN graphs (UMAP) would mostly link points of the same surface, the
    geography risk of smoothed features; the chance level is about 1/n_surfaces.
    """
    q = rng.choice(len(X), size=min(n_query, len(X)), replace=False)
    _, nb = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X[q])
    return (surface[nb[:, 1:]] == surface[q, None]).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("features_dir", type=Path, help="features.py output, e.g. data/features/YTC041_1/sigma5")
    parser.add_argument("--mem", default="IMM", choices=["IMM", "OMM"], type=str.upper)
    parser.add_argument("--n-per-surface", type=int, default=20000,
                        help="vertices sampled per surface (default: 20000)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-flags", action="store_true", help="drop all <col>_valid columns")
    parser.add_argument("--no-self-dist", action="store_true", help="also drop self_dist_min/far")
    parser.add_argument("-o", "--output", type=Path,
                        help="output .npz (default: <repo>/data/baseline/<dataset>/<mode>[_<tag>]/<MEM>_pooled.npz)")
    args = parser.parse_args()

    paths = sorted(args.features_dir.glob(f"*_{args.mem}_AVV_rh9_vec.parquet"))
    if not paths:
        raise SystemExit(f"no {args.mem} *_vec.parquet under {args.features_dir}")
    rng = np.random.default_rng(args.seed)
    df, index = load_sample(paths, args.n_per_surface, rng)

    drop = [c for c in EXCLUDE if c in df.columns]
    flags = [c for c in df.columns if c.endswith("_valid")]
    drop += flags if args.no_flags else [c for c in flags if (df[c] == 1).mean() > MAX_FLAG_ONES]
    if args.no_self_dist:
        drop += [c for c in df.columns if c.startswith("self_dist") and c not in drop]
    df = df.drop(columns=drop)
    assert not df.isna().any().any(), "NaNs left in features; rerun features.py"

    groups = np.array([group_of(c) for c in df.columns])
    if (groups == "other").any():
        print("WARNING ungrouped columns:", df.columns[groups == "other"].tolist())
    means, stds = df.mean().to_numpy(), df.std(ddof=0).to_numpy().copy()
    stds[stds == 0] = 1.0
    sizes = pd.Series(groups).value_counts()
    weights = np.array([1 / np.sqrt(sizes[g]) for g in groups])
    X = ((df.to_numpy() - means) / stds * weights).astype(np.float32)

    dataset, mode = args.features_dir.resolve().parent.name, args.features_dir.resolve().name
    tag = "geometry" if args.no_self_dist else "noflags" if args.no_flags else ""
    mode = f"{mode}_{tag}" if tag else mode
    out = args.output or Path(__file__).resolve().parent / "data" / "baseline" / dataset / mode / f"{args.mem}_pooled.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, X=X, index=index, surfaces=np.array([p.name for p in paths]),
        feature_names=np.array(df.columns, dtype=str), groups=groups.astype(str),
        means=means, stds=stds, weights=weights, features_dir=str(args.features_dir.resolve()),
    )

    print(f"{args.mem}: {len(paths)} surfaces, X={X.shape} -> {out}")
    print("dropped:", drop)
    print("columns per group:", sizes.to_dict())
    same = neighbor_mix(X, index[:, 0], k=15, n_query=5000, rng=rng)
    print(f"15-NN from same surface: {same:.1%} (chance ~{1 / len(paths):.1%})")


if __name__ == "__main__":
    main()
