"""Label every triangle of one surface with the pooled PCA + GMM, on its existing mesh.

Uses the seed-0 PCA and mixture select_k.py saved (or, without them, refits as
gmm_baseline.py does and checks against its saved labels), transforms the surface's full features.py vector with the
pooled scaler and group weights, and writes:
  - <stem>_clusters.vtp: a copy of the surface's .AVV_rh9.vtp with a per-triangle
    "cluster" array (open in ParaView and color by it)
  - view_chimerax.py: opens that same mesh in ChimeraX, colored by cluster

Usage: python project_mesh.py POOLED_NPZ RESULTS_DIR STEM [--k 4] [-o OUT_DIR]
    STEM: short surface name, e.g. lam11_2_ts_003
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from pool_matrix import features_dir_of, surface_matrix

# Reference palette, categorical slots 1-8 in fixed order.
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")

# Runs inside ChimeraX: `runscript view_chimerax.py`, or `chimerax view_chimerax.py`.
# The .vtp's triangles share corners, so each triangle gets its own 3 corners here to
# color it flat by its cluster instead of blending colors across cluster borders.
CHIMERAX_SCRIPT = '''\
import numpy as np
from chimerax.core.models import Surface
from chimerax.surface import calculate_vertex_normals
from chimerax.core.commands import run

d = np.load({npz!r})
k = int(d["labels"].max()) + 1
colors = {colors!r}
run(session, "close session; set bgColor white; lighting soft; graphics silhouettes true")
for c in range(k):
    tri = d["triangles"][d["labels"] == c]
    v = d["points"][tri].reshape(-1, 3).astype(np.float32)
    t = np.arange(len(v), dtype=np.int32).reshape(-1, 3)
    s = Surface(f"cluster {{c}} ({{(d['labels'] == c).mean():.0%}})", session)
    s.set_geometry(v, calculate_vertex_normals(v, t), t)
    h = colors[c].lstrip("#")
    s.color = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (255,)
    session.models.add([s])
run(session, "view")
'''


def fit_like_sweep(d, k, n_fit=100000, var=0.95, seed=0):
    """PCA + GMM exactly as gmm_baseline.py fits them for this k."""
    X = d["X"]
    pca = PCA(n_components=var, random_state=seed).fit(X)
    Z = pca.transform(X)
    fit_rows = np.random.default_rng(seed).choice(len(Z), size=min(n_fit, len(Z)), replace=False)
    gmm = GaussianMixture(k, covariance_type="full", random_state=seed).fit(Z[fit_rows])
    return pca, gmm, gmm.predict(Z)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pooled", type=Path, help="pool_matrix.py output .npz")
    parser.add_argument("results_dir", type=Path, help="folder with the surface's .AVV_rh9.vtp")
    parser.add_argument("stem", help="short surface name, e.g. lam11_2_ts_003")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("-o", "--output", type=Path, help="output folder (default: next to POOLED_NPZ)")
    args = parser.parse_args()
    import vtk  # only here, so other envs can import this module's helpers
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    d = np.load(args.pooled)
    mem = args.pooled.stem.split("_")[0]
    hits = [s for s in d["surfaces"] if f"_{args.stem}_labels_{mem}_" in s]
    if len(hits) != 1:
        raise SystemExit(f"stem {args.stem!r} matches {len(hits)} {mem} surfaces")
    vec_path = features_dir_of(args.pooled, d) / hits[0]
    vtp_path = args.results_dir / vec_path.name.replace("_AVV_rh9_vec.parquet", ".AVV_rh9.vtp")

    models = args.pooled.parent / f"select_k_{mem}" / "models.joblib"
    if models.exists():
        # The seed-0 fits select_k.py saved: reusing them avoids refits that differ with
        # thread count and keeps mesh labels identical to its labels_full.npz.
        import joblib
        m = joblib.load(models)
        pca, gmm = m["pca"], m["gmm"][args.k]
    else:
        pca, gmm, pooled_lab = fit_like_sweep(d, args.k)
        saved = np.load(args.pooled.parent / f"gmm_{mem}" / "labels.npz")[f"k{args.k}"]
        assert (pooled_lab == saved).all(), "refit differs from gmm_baseline.py labels"
    lab = gmm.predict(pca.transform(surface_matrix(pd.read_parquet(vec_path), d)))

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp_path))
    reader.Update()
    mesh = reader.GetOutput()
    assert mesh.GetNumberOfCells() == len(lab), f"mesh has {mesh.GetNumberOfCells()} triangles, labels {len(lab)}"
    arr = numpy_to_vtk(lab.astype(np.int32), deep=True)
    arr.SetName("cluster")
    mesh.GetCellData().AddArray(arr)

    out = args.output or args.pooled.parent / f"mesh_{mem}_k{args.k}" / args.stem
    out.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(out / f"{args.stem}_{mem}_clusters.vtp"))
    writer.SetInputData(mesh)
    writer.Write()

    npz = out / "mesh_labels.npz"
    np.savez_compressed(npz, points=vtk_to_numpy(mesh.GetPoints().GetData()),
                        triangles=vtk_to_numpy(mesh.GetPolys().GetConnectivityArray()).reshape(-1, 3),
                        labels=lab.astype(np.int8))
    (out / "view_chimerax.py").write_text(
        CHIMERAX_SCRIPT.format(npz=str(npz.resolve()), colors=COLORS[:args.k]))

    shares = ", ".join(f"{c}: {(lab == c).mean():.1%}" for c in range(args.k))
    print(f"{args.stem} {mem}: {len(lab)} triangles labeled (k={args.k}; {shares}) -> {out}")


if __name__ == "__main__":
    main()
