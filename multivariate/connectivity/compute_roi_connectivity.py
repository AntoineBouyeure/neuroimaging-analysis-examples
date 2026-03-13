#!/usr/bin/env python3
"""Compute condition-wise ROI connectivity matrices from trialwise beta maps."""
from __future__ import annotations

import argparse
from pathlib import Path
import logging
import pickle
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import concat_imgs
from nilearn.connectome import ConnectivityMeasure

LOGGER = logging.getLogger("compute_roi_connectivity")


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def build_atlas(roi_paths: list[Path], output_path: Path) -> Path:
    ref = nib.load(str(roi_paths[0]))
    atlas = np.zeros(ref.shape, dtype=np.int16)
    for idx, roi_path in enumerate(roi_paths, start=1):
        atlas[nib.load(str(roi_path)).get_fdata() > 0] = idx
    atlas_img = nib.Nifti1Image(atlas, ref.affine, ref.header)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(atlas_img, str(output_path))
    return output_path


def compute_condition_connectivity(atlas_path: Path, beta_maps: list[Path], metric: str) -> np.ndarray:
    imgs = concat_imgs([str(p) for p in beta_maps])
    masker = NiftiLabelsMasker(labels_img=str(atlas_path), standardize=False)
    ts = masker.fit_transform(imgs)
    conn = ConnectivityMeasure(kind=metric)
    return conn.fit_transform([ts])[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-output", type=Path, required=True)
    parser.add_argument("--roi-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--beta-table", type=Path, required=True, help="CSV with columns: condition,path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", nargs="+", default=["correlation", "partial correlation"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)

    atlas_path = build_atlas(args.roi_paths, args.atlas_output)
    beta_df = pd.read_csv(args.beta_table)
    results = {metric: {} for metric in args.metrics}
    for condition, cond_df in beta_df.groupby("condition"):
        maps = [Path(p) for p in cond_df["path"].tolist()]
        for metric in args.metrics:
            results[metric][condition] = compute_condition_connectivity(atlas_path, maps, metric)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    LOGGER.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
