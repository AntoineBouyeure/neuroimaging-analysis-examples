#!/usr/bin/env python3
"""Layer-wise statistical analysis and plotting.

Performs linear mixed-effects modelling and one-sample tests on long-format
ROI × layer summary tables, with optional FDR correction and figure export.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import logging
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

LOGGER = logging.getLogger("layer_stats")


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def fit_models(df: pd.DataFrame, value_col: str, group_col: str) -> pd.DataFrame:
    results = []
    for roi, roi_df in df.groupby("roi"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm(f"{value_col} ~ C(layer)", roi_df, groups=roi_df[group_col]).fit(reml=False)
            pval = model.pvalues.drop("Intercept", errors="ignore").min()
        except Exception as exc:
            LOGGER.warning("Model failed for %s: %s", roi, exc)
            pval = np.nan
        for layer, layer_df in roi_df.groupby("layer"):
            t_stat, t_p = stats.ttest_1samp(layer_df[value_col].dropna(), 0.0)
            results.append({"roi": roi, "layer": layer, "model_p": pval, "onesample_t": t_stat, "onesample_p": t_p})
    out = pd.DataFrame(results)
    if not out.empty:
        _, out["onesample_p_fdr"] = fdrcorrection(out["onesample_p"].fillna(1.0).values)
    return out


def plot_layers(df: pd.DataFrame, value_col: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for roi, roi_df in df.groupby("roi"):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=roi_df, x="layer", y=value_col, ax=ax)
        sns.stripplot(data=roi_df, x="layer", y=value_col, ax=ax, color="black", alpha=0.5)
        ax.set_title(roi)
        fig.tight_layout()
        fig.savefig(output_dir / f"{roi}_layer_summary.png", dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-stats", type=Path, required=True)
    parser.add_argument("--output-figures", type=Path)
    parser.add_argument("--value-col", default="mean")
    parser.add_argument("--group-col", default="subject")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)

    df = pd.read_csv(args.input)
    stats_df = fit_models(df, args.value_col, args.group_col)
    args.output_stats.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(args.output_stats, index=False)
    LOGGER.info("Wrote %s", args.output_stats)
    if args.output_figures is not None:
        plot_layers(df, args.value_col, args.output_figures)


if __name__ == "__main__":
    main()
