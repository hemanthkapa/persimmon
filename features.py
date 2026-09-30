"""Build per-triangle feature vectors from explore_gt.py tables.

Default: smooths every feature by diffusion over the mesh graph (triangle adjacency, so it
never crosses to a separate component or a facing membrane); gaps wider than the smoothing
reach are then filled from the nearest smoothed values. With --fill-only, real values
are kept as-is and only NaNs are filled from the nearest real values on the mesh, for models
that learn their own smoothing (e.g. DiffusionNet).

Every column with NaNs on a surface gets a <col>_valid flag; a column without a flag had no
NaNs there (treat it as all valid when pooling surfaces). Writes one feature and one
metadata parquet per surface.

Usage: python features.py RESULTS_DIR [--processed DIR] [--sigma NM | --fill-only] [-o OUT]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from graph_tool import load_graph
from scipy import sparse


# NaN self-distance means no hit within the search range, i.e. far away, not unmeasured.
SELF_DIST = ("self_dist_min", "self_dist_far")


def load_mesh(path):
    """Return (edges, n_vertices, xyz) of the triangle adjacency graph in a .gt file."""
    g = load_graph(path)
    return g.get_edges()[:, :2], g.num_vertices(), g.vp["xyz"].get_2d_array([0, 1, 2]).T


def diffusion_operator(edges, n):
    """Row-normalized adjacency with self loops: one random-walk step over the mesh."""
    idx = np.arange(n)
    i = np.concatenate([edges[:, 0], edges[:, 1], idx])
    j = np.concatenate([edges[:, 1], edges[:, 0], idx])
    A = sparse.csr_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    return sparse.diags(1.0 / np.asarray(A.sum(axis=1)).ravel()) @ A


def n_steps(edges, xyz, sigma):
    """Diffusion steps whose RMS spread matches sigma (nm).

    A walk step moves to a neighbor with probability p = deg/(deg+1), so after k steps
    the mean squared displacement is about k * p * spacing^2.
    """
    if sigma <= 0:
        return 0
    spacing = np.median(np.linalg.norm(xyz[edges[:, 0]] - xyz[edges[:, 1]], axis=1))
    deg = np.median(np.bincount(edges.ravel()))
    return max(1, int(round(sigma**2 / (deg / (deg + 1) * spacing**2))))


def smooth(P, X, steps):
    """Diffuse columns of X over the mesh ignoring NaNs.

    Returns (smoothed, valid): valid is the diffused fraction of non-NaN neighbors,
    and smoothed is NaN where no valid neighbor was reached.
    """
    mask = ~np.isnan(X)
    S = np.where(mask, X, 0.0)
    C = mask.astype(float)
    for _ in range(steps):
        S, C = P @ S, P @ C
    with np.errstate(invalid="ignore", divide="ignore"):
        out = S / C
    out[C == 0] = np.nan
    return out, C


def fill_nearest(P, X, max_steps):
    """Fill NaNs in columns of X from the nearest real values on the mesh; keep real values.

    Diffuses like smooth() but freezes each NaN at the first step a real value reaches it,
    so it takes the average of the closest real values. NaNs no real value can reach (a
    component with none) stay NaN.
    """
    mask = ~np.isnan(X)
    out = X.copy()
    todo = ~mask
    S = np.where(mask, X, 0.0)
    C = mask.astype(float)
    for _ in range(max_steps):
        if not todo.any():
            break
        S, C = P @ S, P @ C
        reached = todo & (C > 0)
        if not reached.any():
            break
        out[reached] = S[reached] / C[reached]
        todo &= ~reached
    return out


def process_one(gt_path, processed_dir, output_dir, sigma, dist_max, fill_only, max_fill_steps):
    """Build and save the feature vector for one surface."""
    safe = gt_path.stem.replace(".", "_")
    feat_path = processed_dir / f"{safe}_features.parquet"
    if not feat_path.exists():
        print(f"SKIP (no explore_gt output): {safe}")
        return

    features = pd.read_parquet(feat_path)
    meta = pd.read_parquet(processed_dir / f"{safe}_metadata.parquet")
    edges, n, xyz = load_mesh(str(gt_path))
    assert len(features) == n, f"row mismatch: features={len(features)} gt={n}"

    raw = features.to_numpy(float)
    P = diffusion_operator(edges, n)
    nan_cols = features.columns[features.isna().any()]
    # No hit is information, so self-distances go straight to dist_max, not neighbor values.
    dist_cols = [c for c in SELF_DIST if c in features.columns]
    rest = [features.columns.get_loc(c) for c in features.columns if c not in dist_cols]
    if fill_only:
        X = raw.copy()
        X[:, rest] = fill_nearest(P, raw[:, rest], max_fill_steps)
        valid = (~np.isnan(raw)).astype(float)  # 1 = measured, 0 = filled
        steps = "fill"
    else:
        steps = n_steps(edges, xyz, sigma)
        X, valid = smooth(P, raw, steps)  # valid = fraction of the value from real data
        # Gaps wider than the smoothing reach: fill from the nearest smoothed values (valid stays 0).
        X[:, rest] = fill_nearest(P, X[:, rest], max_fill_steps)
    vec = pd.DataFrame(X, columns=features.columns)

    for c in nan_cols:
        vec[f"{c}_valid"] = valid[:, features.columns.get_loc(c)]
    # No hit within the search range: record it as "beyond range".
    for c in SELF_DIST:
        if c in vec.columns:
            vec[c] = vec[c].fillna(dist_max)

    meta = meta[[c for c in ("cc_id", "flipped") if c in meta.columns]]
    vec.astype(np.float32).to_parquet(output_dir / f"{safe}_vec.parquet", index=False)
    meta.to_parquet(output_dir / f"{safe}_meta.parquet", index=False)
    left = vec.columns[vec.isna().any()].tolist()
    print(f"{safe}: n={n} steps={steps} NaN cols={left}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_dir", type=Path, help="folder of *.AVV_rh9.gt used by explore_gt.py")
    parser.add_argument("--processed", type=Path,
                        help="explore_gt.py output (default: <repo>/data/processed/<dataset>)")
    parser.add_argument("--sigma", type=float, default=5.0,
                        help="RMS smoothing radius in nm, 0 = no smoothing (default: 5)")
    parser.add_argument("--fill-only", action="store_true",
                        help="keep real values, only fill NaNs from nearest mesh neighbors (ignores --sigma)")
    parser.add_argument("--max-fill-steps", type=int, default=2000,
                        help="cap on diffusion steps when filling NaNs (default: 2000)")
    parser.add_argument("--dist-max", type=float, default=400.0,
                        help="self-distance search range in nm, used for no-hit values (default: 400)")
    parser.add_argument("-o", "--output", type=Path,
                        help="output folder (default: <repo>/data/features/<dataset>/sigma<N> or fill_only)")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent
    dataset = args.results_dir.resolve().parent.name  # e.g. YTC041_1
    processed_dir = args.processed or repo / "data" / "processed" / dataset
    mode = "fill_only" if args.fill_only else f"sigma{args.sigma:g}"
    output_dir = args.output or repo / "data" / "features" / dataset / mode
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_files = [p for p in sorted(args.results_dir.glob("*.AVV_rh9.gt")) if "_cc_flipped" not in p.name]
    if not gt_files:
        raise SystemExit(f"no *.AVV_rh9.gt under {args.results_dir}")

    keep = ("fill_only", "max_fill_steps", "dist_max") + (() if args.fill_only else ("sigma",))
    params = {k: v for k, v in vars(args).items() if k in keep}
    (output_dir / "params.json").write_text(json.dumps(params, indent=2))
    print(f"graphs: {len(gt_files)}  processed: {processed_dir}  output: {output_dir}  {params}")
    for gt_path in gt_files:
        process_one(gt_path, processed_dir, output_dir, args.sigma, args.dist_max,
                    args.fill_only, args.max_fill_steps)
    print(f"\ndone -> {output_dir}")


if __name__ == "__main__":
    main()
