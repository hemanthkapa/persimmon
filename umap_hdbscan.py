"""Cluster a pooled matrix with UMAP + HDBSCAN and compare with the GMM labels.

Embeds a random subset of pooled rows with UMAP (on the weighted features, no PCA),
clusters the embedding with HDBSCAN, and reports the number of clusters, noise share,
dependence on surface (AMI; effective number of surfaces per cluster) and agreement with
the GMM labels of the same rows (ARI / AMI on non-noise rows, plus a contingency table).

With --surfaces, every triangle of those surfaces is projected into the same UMAP and
takes the majority HDBSCAN label of its 15 nearest embedded rows, then written for
ChimeraX on the mesh project_mesh.py already exported.

Usage: python umap_hdbscan.py POOLED_NPZ [--gmm-k 5] [--n 100000] [--surfaces STEM ...]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from gmm_baseline import effective_surfaces


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pooled", type=Path, help="pool_matrix.py output .npz")
    parser.add_argument("--gmm-k", type=int, default=5, help="GMM labels to compare against")
    parser.add_argument("--n", type=int, default=100000, help="rows embedded (default: 100000)")
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.0)
    parser.add_argument("--min-cluster-size", type=int, default=1000)
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--surfaces", nargs="*", default=[], help="short surface names to label on the mesh")
    args = parser.parse_args()

    d = np.load(args.pooled)
    mem = args.pooled.stem.split("_")[0]
    gmm = np.load(args.pooled.parent / f"gmm_{mem}" / "labels.npz")[f"k{args.gmm_k}"]
    rows = np.sort(np.random.default_rng(args.seed).choice(len(d["X"]), size=min(args.n, len(d["X"])), replace=False))
    X, surface, gmm = d["X"][rows], d["index"][rows, 0], gmm[rows]

    reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist, n_components=2,
                        random_state=args.seed)
    emb = reducer.fit_transform(X)
    lab = HDBSCAN(min_cluster_size=args.min_cluster_size, min_samples=args.min_samples).fit_predict(emb)

    ok = lab >= 0
    eff = effective_surfaces(lab[ok], surface[ok])
    res = {
        "n_rows": len(rows), "n_clusters": int(lab.max() + 1), "noise": float((~ok).mean()),
        "ami_surface": float(adjusted_mutual_info_score(surface[ok], lab[ok])),
        "min_eff_surfaces": min(e for e, _ in eff.values()),
        "max_single_surface_share": max(m for _, m in eff.values()),
        "ari_vs_gmm": float(adjusted_rand_score(gmm[ok], lab[ok])),
        "ami_vs_gmm": float(adjusted_mutual_info_score(gmm[ok], lab[ok])),
    }
    table = pd.crosstab(pd.Series(lab, name="hdbscan"), pd.Series(gmm, name=f"gmm_k{args.gmm_k}"), normalize="index")

    out = args.pooled.parent / f"umap_hdbscan_{mem}"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / "embedding.npz", rows=rows, embedding=emb, hdbscan=lab, gmm=gmm, surface=surface)
    (out / "summary.json").write_text(json.dumps({**res, "params": {k: v for k, v in vars(args).items() if k != "pooled"}},
                                                 indent=2, default=str))
    table.round(3).to_csv(out / "contingency.csv")

    print(json.dumps(res, indent=2))
    print("\nHDBSCAN cluster (rows) -> share of its points in each GMM cluster (columns):")
    print((table * 100).round(0).astype(int).to_string())
    print(f"cluster sizes: {np.bincount(lab[ok]).tolist()}  noise: {(~ok).sum()}")
    print(f"saved -> {out}")

    if args.surfaces:
        label_meshes(args, d, mem, reducer, emb[ok], lab[ok], out)


def label_meshes(args, d, mem, reducer, emb, lab, out):
    """Label every triangle of each surface by kNN in the embedding; write ChimeraX files."""
    from sklearn.neighbors import KNeighborsClassifier
    from pool_matrix import features_dir_of, surface_matrix
    from project_mesh import CHIMERAX_SCRIPT, COLORS

    knn = KNeighborsClassifier(n_neighbors=15).fit(emb, lab)
    mode_dir = args.pooled.resolve().parent
    feat_dir = features_dir_of(args.pooled, d)
    for stem in args.surfaces:
        name = next(s for s in d["surfaces"] if f"_{stem}_labels_{mem}_" in s)
        full = knn.predict(reducer.transform(surface_matrix(pd.read_parquet(feat_dir / name), d))).astype(np.int8)
        # Mesh points and triangles as exported by project_mesh.py (same triangle order).
        mesh = np.load(mode_dir / f"mesh_{mem}_k{args.gmm_k}" / stem / "mesh_labels.npz")
        assert len(mesh["labels"]) == len(full), f"{stem}: mesh {len(mesh['labels'])} vs {len(full)} triangles"
        sd = out / "mesh" / stem
        sd.mkdir(parents=True, exist_ok=True)
        npz = sd / "mesh_labels.npz"
        np.savez_compressed(npz, points=mesh["points"], triangles=mesh["triangles"], labels=full)
        k = int(lab.max()) + 1
        (sd / "view_chimerax.py").write_text(CHIMERAX_SCRIPT.format(npz=str(npz.resolve()), colors=COLORS[:k]))
        names = [f'rename #{c + 1} "hdbscan {c} ({(full == c).mean():.0%})"' for c in range(k)]
        (sd / "names.cxc").write_text("\n".join(names + [f"2dlabels text '{stem} {mem}, UMAP + HDBSCAN' "
                                                          "xpos 0.02 ypos 0.95 size 20 color black", "view"]) + "\n")
        print(f"  {stem}: {len(full)} triangles, shares {np.round(np.bincount(full, minlength=k) / len(full), 3).tolist()}")


if __name__ == "__main__":
    main()
