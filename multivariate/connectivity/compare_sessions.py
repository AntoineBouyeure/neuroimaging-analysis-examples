#!/usr/bin/env python3
"""Compare ROI connectivity matrices between two sessions."""
from __future__ import annotations

import argparse
from pathlib import Path
import logging
import pickle
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import fdrcorrection

LOGGER = logging.getLogger("compare_sessions_connectivity")


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def vectorize_upper(mat: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(mat, k=1)
    return mat[iu]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-a", type=Path, required=True)
    parser.add_argument("--session-b", type=Path, required=True)
    parser.add_argument("--metric", default="partial correlation")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)

    with open(args.session_a, "rb") as f:
        a = pickle.load(f)
    with open(args.session_b, "rb") as f:
        b = pickle.load(f)

    mats_a = a[args.metric][args.condition]
    mats_b = b[args.metric][args.condition]
    rows = []
    for idx, (mat_a, mat_b) in enumerate(zip(mats_a, mats_b)):
        diff = vectorize_upper(np.asarray(mat_a)) - vectorize_upper(np.asarray(mat_b))
        rows.append(pd.DataFrame({"subject_index": idx, "edge": np.arange(diff.size), "delta": diff}))
    df = pd.concat(rows, ignore_index=True)
    stats_rows = []
    for edge, edge_df in df.groupby("edge"):
        t_stat, p_val = ttest_rel(edge_df["delta"], np.zeros(len(edge_df)))
        stats_rows.append({"edge": edge, "t": t_stat, "p": p_val})
    stats_df = pd.DataFrame(stats_rows)
    _, stats_df["p_fdr"] = fdrcorrection(stats_df["p"].fillna(1.0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(args.output, index=False)
    LOGGER.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
