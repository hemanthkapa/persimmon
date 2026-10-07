"""Locate one GMM cluster on the mesh relative to the boundary group and mesh borders.

For every surface, measures mesh-graph hop distance (triangle adjacency) from each
triangle of the target cluster and of the comparison clusters to the nearest triangle
of the reference group, and to the nearest open-edge triangle (a triangle with an edge
no other triangle shares, from the .vtp). Distances are only over triangles whose mesh
piece contains a triangle of that kind; the share that can reach one is reported too.
Also counts the target cluster's connected components and how many touch the
reference group, and reports raw-feature medians per cluster.

Usage: python inspect_c3.py LABELS_NPZ RESULTS_DIR FEATURES_DIR [--k 5] [--target 3]
                            [--compare 1 2] [--reference 0 4] [-o OUT_DIR]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra

from features import load_mesh

FEATURES = ("curvedness_VV", "shape_index_VV", "OMM_dist", "self_dist_min", "thickness")


def border_triangles(vtp):
    """Triangles with at least one edge used by no other triangle (open mesh edges)."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    r = vtk.vtkXMLPolyDataReader()
    r.SetFileName(str(vtp))
    r.Update()
    tri = vtk_to_numpy(r.GetOutput().GetPolys().GetConnectivityArray()).reshape(-1, 3)
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [0, 2]]]), axis=1)
    _, inv, cnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    return (cnt[inv.ravel()] == 1).reshape(3, -1).any(axis=0)


def hops_to(A, sources):
    """Hop distance from every triangle to the nearest triangle in `sources`."""
    if not sources.any():
        return np.full(A.shape[0], np.inf)
    return dijkstra(A, indices=np.flatnonzero(sources), min_only=True, unweighted=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("labels", type=Path, help="select_k.py labels_full.npz")
    parser.add_argument("results_dir", type=Path, help="folder with the surfaces' .AVV_rh9.gt")
    parser.add_argument("features_dir", type=Path, help="features.py output for the same mode")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--target", type=int, default=3)
    parser.add_argument("--compare", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--reference", type=int, nargs="+", default=[0, 4])
    parser.add_argument("--mem", default="IMM")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    labs = np.load(args.labels)
    stems = sorted(k[: -len(f"_k{args.k}")] for k in labs.files if k.endswith(f"_k{args.k}"))
    clusters = [args.target] + args.compare
    rows, comps, feats = [], [], []
    for stem in stems:
        lab = labs[f"{stem}_k{args.k}"]
        gt = next(args.results_dir.glob(f"*_{stem}_labels_{args.mem}.AVV_rh9.gt"))
        edges, n, _ = load_mesh(str(gt))
        A = sparse.coo_matrix((np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]],
                                                         np.r_[edges[:, 1], edges[:, 0]])), shape=(n, n)).tocsr()
        d_ref = hops_to(A, np.isin(lab, args.reference))
        d_border = hops_to(A, border_triangles(gt.with_suffix(".vtp")))
        for c in clusters:
            m = lab == c
            if m.any():
                rows.append({"surface": stem, "cluster": c, "n": int(m.sum()),
                             "hops_to_ref": d_ref[m], "hops_to_border": d_border[m]})
        # Connected components of the target cluster and whether they touch the reference group.
        t = lab == args.target
        if t.any():
            idx = np.flatnonzero(t)
            nc, cc = connected_components(A[idx][:, idx], directed=False)
            touch = np.zeros(nc, bool)
            nb_ref = (A[idx] @ np.isin(lab, args.reference).astype(float)) > 0
            touch[np.unique(cc[nb_ref])] = True
            sizes = np.bincount(cc)
            comps.append({"surface": stem, "n_components": nc, "median_size": float(np.median(sizes)),
                          "largest": int(sizes.max()), "touch_ref_frac": float(touch.mean()),
                          "touch_ref_frac_by_area": float(sizes[touch].sum() / sizes.sum())})
        vec = pd.read_parquet(next(args.features_dir.glob(f"*_{stem}_labels_{args.mem}_AVV_rh9_vec.parquet")))
        cols = [c for c in FEATURES if c in vec.columns]
        for c in clusters:
            m = lab == c
            if m.any():
                feats.append({"surface": stem, "cluster": c, **vec.loc[m, cols].median().to_dict()})
        print(f"{stem}: n={n}", flush=True)

    # Pool per-triangle distances across surfaces.
    summary = []
    for c in clusters:
        r = [x for x in rows if x["cluster"] == c]
        ref = np.concatenate([x["hops_to_ref"] for x in r])
        bor = np.concatenate([x["hops_to_border"] for x in r])
        rf, bf = ref[np.isfinite(ref)], bor[np.isfinite(bor)]
        summary.append({"cluster": c, "triangles": int(sum(x["n"] for x in r)),
                        "frac_connected_to_ref": float(np.isfinite(ref).mean()),
                        "median_hops_to_ref": float(np.median(rf)), "frac_within_5_hops_of_ref": float((rf <= 5).mean()),
                        "median_hops_to_border": float(np.median(bf)), "frac_within_5_hops_of_border": float((bf <= 5).mean())})
    summary = pd.DataFrame(summary)
    comps = pd.DataFrame(comps)
    feats = pd.DataFrame(feats).groupby("cluster").median(numeric_only=True)
    summary.to_csv(args.output / "distances.csv", index=False)
    comps.to_csv(args.output / "components.csv", index=False)
    feats.to_csv(args.output / "features.csv")
    pd.set_option("display.width", 160)
    print("\n" + summary.round(3).to_string(index=False))
    print("\ntarget components per surface (median over surfaces):")
    print(comps.drop(columns="surface").median().round(3).to_string())
    print("\nfeature medians (median over surfaces):")
    print(feats.round(4).to_string())


if __name__ == "__main__":
    main()
