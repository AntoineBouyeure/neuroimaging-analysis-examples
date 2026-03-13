#!/usr/bin/env python3
"""Extract ROI- and layer-specific summary values from NIfTI maps.

This script is designed for high-resolution / laminar analyses in which each ROI
contains a layer label map (e.g., deep/middle/superficial). It computes the mean
signal within each ROI × layer mask for one or more statistical maps.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import logging
import re
import nibabel as nib
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("extract_layer_profiles")
DEFAULT_LABELS = {1: "deep", 2: "middle", 3: "superficial"}


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def load_mask_data(mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(mask_path))
    return img.get_fdata(), img.affine


def sample_image(image_path: Path, layer_mask: np.ndarray, labels: dict[int, str], roi_name: str, subject: str) -> list[dict]:
    img = nib.load(str(image_path))
    data = img.get_fdata()
    rows = []
    contrast = image_path.stem.replace('.nii','')
    for value, layer_name in labels.items():
        vox = data[layer_mask == value]
        rows.append(
            {
                "subject": subject,
                "roi": roi_name,
                "layer": layer_name,
                "contrast": contrast,
                "n_voxels": int(np.sum(layer_mask == value)),
                "mean": float(np.nanmean(vox)) if vox.size else np.nan,
                "std": float(np.nanstd(vox)) if vox.size else np.nan,
            }
        )
    return rows


def infer_subject(path: Path) -> str:
    match = re.search(r"sub\d+", str(path))
    return match.group(0) if match else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-dir", type=Path, required=True, help="Directory with ROI layer-label maps")
    parser.add_argument("--images", type=Path, nargs="+", required=True, help="Input NIfTI map(s)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-pattern", default="*.nii.gz")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)

    rows = []
    for mask_path in sorted(args.mask_dir.glob(args.mask_pattern)):
        roi_name = mask_path.name.replace(".nii.gz", "")
        subject = infer_subject(mask_path)
        mask, _ = load_mask_data(mask_path)
        for image_path in args.images:
            rows.extend(sample_image(image_path, mask, DEFAULT_LABELS, roi_name, subject))

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    LOGGER.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
