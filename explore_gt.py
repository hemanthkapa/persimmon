"""Build per-vertex feature tables from surface morphometrics outputs.

Joins the scalar properties of each ``*.AVV_rh9.gt`` with its density
``*_sampling.csv`` and writes feature/metadata parquets. NaNs are kept;
features.py fills them from mesh neighbors.

Usage: python explore_gt.py RESULTS_DIR [--mem IMM|OMM|ALL] [-o OUTPUT_DIR]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from graph_tool import load_graph


# graph-tool value types stored as one number per vertex.
SCALARS = {"bool", "int16_t", "int32_t", "int64_t", "unsigned long", "double", "long double"}


def load_gt(path):
    """Load scalar vertex properties and a mask of vertices whose normal was flipped.

    path (str): graph-tool .gt file
    Returns (DataFrame of scalars, bool array). The flipper reverses n_v but never
    the triangle normal, so n_v . normal < 0 marks a flipped vertex.
    """
    g = load_graph(path)
    cols = {k: g.vp[k].get_array() for k in g.vp.keys() if g.vp[k].value_type() in SCALARS}
    n_v = g.vp["n_v"].get_2d_array([0, 1, 2])
    normal = g.vp["normal"].get_2d_array([0, 1, 2])
    flipped = np.sum(n_v * normal, axis=0) < 0
    return pd.DataFrame(cols), flipped


def orient_curvatures(df, flipped):
    """Re-sign curvatures on flipped vertices to match n_v.

    The flipper leaves pycurv's curvatures untouched, but density is sampled along
    the flipped n_v. Reversing the normal maps (k1, k2) to (-k2, -k1) and negates
    mean curvature and shape index; Gaussian curvature and curvedness are unchanged.

    df (DataFrame): scalars from load_gt
    flipped (bool array): mask from load_gt
    """
    out = df.copy()
    k1 = df.loc[flipped, "kappa_1"].to_numpy()
    k2 = df.loc[flipped, "kappa_2"].to_numpy()
    out.loc[flipped, "kappa_1"] = -k2
    out.loc[flipped, "kappa_2"] = -k1
    for c in ("mean_curvature_VV", "shape_index_VV"):
        out.loc[flipped, c] = -df.loc[flipped, c]
    return out


def load_density_csv(path):
    """Load a density sampling CSV (rows aligned with .gt vertices), prefixing columns with density_."""
    dens = pd.read_csv(path)
    dens.columns = [f"density_{c}" for c in dens.columns]
    return dens


def split_features_metadata(df):
    """Split off index-like columns (neighbor/self ids, cc_id) and per-surface values as metadata.

    average_width is one whole-surface bilayer fit copied onto every vertex, so it carries
    no per-vertex information.
    """
    metadata_cols = [
        c for c in df.columns
        if c.endswith("_neighbor_index")
        or c in {"self_id_min", "self_id_far", "cc_id", "average_width"}
    ]
    return df.drop(columns=metadata_cols), df[metadata_cols].copy()


def process_one(gt_path, output_dir):
    """Build and save the feature table for one .gt + sampling pair."""
    samp_path = gt_path.with_name(gt_path.stem + "_sampling.csv")
    if not samp_path.exists():
        print(f"SKIP (no sampling csv): {gt_path.name}")
        return

    stem = gt_path.stem  # e.g. YTC041_1_lam11_2_ts_003_labels_IMM.AVV_rh9
    print(f"\n=== {stem} ===")

    raw, flipped = load_gt(str(gt_path))
    df = orient_curvatures(raw, flipped)
    print(f"flipped vertices: {flipped.mean():.1%}")

    dens = load_density_csv(str(samp_path))
    assert len(df) == len(dens), f"row mismatch: gt={len(df)} dens={len(dens)}"
    combined = pd.concat([df.reset_index(drop=True), dens.reset_index(drop=True)], axis=1)

    features, meta = split_features_metadata(combined)
    meta["flipped"] = flipped
    print("features:", features.shape, "meta:", meta.shape)

    nan_counts = features.isna().sum()
    if nan_counts.any():
        print("NaNs per column (nonzero only):")
        print(nan_counts[nan_counts > 0])

    safe = stem.replace(".", "_")
    features.to_parquet(output_dir / f"{safe}_features.parquet", index=False)
    meta.to_parquet(output_dir / f"{safe}_metadata.parquet", index=False)
    print(f"saved {safe}_features.parquet + _metadata.parquet")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_dir", type=Path, help="folder of *.AVV_rh9.gt + *_sampling.csv pairs")
    parser.add_argument("--mem", default="ALL", choices=["IMM", "OMM", "ALL"], type=str.upper,
                        help="membrane to process (default: ALL)")
    parser.add_argument("-o", "--output", type=Path,
                        help="output folder (default: <repo>/data/processed/<dataset>)")
    args = parser.parse_args()

    results_dir, mem = args.results_dir, args.mem
    if not results_dir.is_dir():
        raise SystemExit(f"results dir not found: {results_dir}")

    gt_files = [p for p in sorted(results_dir.glob("*.AVV_rh9.gt")) if "_cc_flipped" not in p.name]
    if mem != "ALL":
        gt_files = [p for p in gt_files if f"_{mem}." in p.name or f"_{mem}_" in p.name]
    if not gt_files:
        raise SystemExit(f"no matching *.AVV_rh9.gt under {results_dir} (MEM={mem})")

    dataset = results_dir.resolve().parent.name  # e.g. YTC041_1
    output_dir = args.output or Path(__file__).resolve().parent / "data" / "processed" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"results: {results_dir}  MEM: {mem}  graphs: {len(gt_files)}  output: {output_dir}")
    for gt_path in gt_files:
        process_one(gt_path, output_dir)
    print(f"\ndone: processed {len(gt_files)} graph(s) -> {output_dir}")


if __name__ == "__main__":
    main()
