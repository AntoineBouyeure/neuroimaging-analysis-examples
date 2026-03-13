#!/usr/bin/env python3
"""Compare layer-wise RSA summaries across acquisition and extinction sessions."""
from __future__ import annotations

import argparse
from pathlib import Path
import logging
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

LOGGER = logging.getLogger("compare_layer_rsa_between_sessions")


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def paired_tests(df_acq: pd.DataFrame, df_ext: pd.DataFrame) -> pd.DataFrame:
    results = []
    merge_cols = ["subject", "roi", "layer", "comparison"]
    merged = df_acq.merge(df_ext, on=merge_cols, suffixes=("_acq", "_ext"))
    for (roi, layer, comparison), subdf in merged.groupby(["roi", "layer", "comparison"]):
        t_stat, p_val = stats.ttest_rel(subdf["similarity_acq"], subdf["similarity_ext"])
        results.append({"roi": roi, "layer": layer, "comparison": comparison, "t": t_stat, "p": p_val, "n": len(subdf)})
    out = pd.DataFrame(results)
    if not out.empty:
        _, out["p_fdr"] = fdrcorrection(out["p"].fillna(1.0).values)
    return out.sort_values(["roi", "layer", "comparison"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acq", type=Path, required=True)
    parser.add_argument("--ext", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison-filter")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)

    df_acq = pd.read_csv(args.acq)
    df_ext = pd.read_csv(args.ext)
    if args.comparison_filter:
        df_acq = df_acq[df_acq["comparison"] == args.comparison_filter]
        df_ext = df_ext[df_ext["comparison"] == args.comparison_filter]
    out = paired_tests(df_acq, df_ext)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    LOGGER.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
