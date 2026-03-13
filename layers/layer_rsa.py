#!/usr/bin/env python3
"""Summarise ROI- and layer-specific representational similarity matrices.

Input pickles are expected to contain items of the form:
    [subject, roi, layer, similarity_matrix]
The script exports a long-format table with within- and between-condition
similarity summaries.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import logging
import pickle
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("layer_rsa")


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def mean_upper_triangle(mat: np.ndarray) -> float:
    iu = np.triu_indices_from(mat, k=1)
    vals = mat[iu]
    return float(np.nanmean(vals))


def summarise_matrix(subject: str, roi: str, layer: str, mat: np.ndarray, split_index: int | None = None) -> list[dict]:
    rows = [{"subject": subject, "roi": roi, "layer": layer, "comparison": "all", "similarity": mean_upper_triangle(mat)}]
    if split_index is not None and 0 < split_index < mat.shape[0]:
        a = mat[:split_index, :split_index]
        b = mat[split_index:, split_index:]
        between = mat[:split_index, split_index:]
        rows.extend(
            [
                {"subject": subject, "roi": roi, "layer": layer, "comparison": "within_a", "similarity": mean_upper_triangle(a)},
                {"subject": subject, "roi": roi, "layer": layer, "comparison": "within_b", "similarity": mean_upper_triangle(b)},
                {"subject": subject, "roi": roi, "layer": layer, "comparison": "between", "similarity": float(np.nanmean(between))},
            ]
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-index", type=int)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)

    with open(args.input, "rb") as f:
        items = pickle.load(f)
    rows = []
    for subject, roi, layer, mat in items:
        rows.extend(summarise_matrix(subject, roi, layer, np.asarray(mat), args.split_index))
    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    LOGGER.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
