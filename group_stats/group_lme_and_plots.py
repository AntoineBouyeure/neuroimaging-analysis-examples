#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(see top of previous attempt for full docstring)
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

def sem(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size <= 1: return np.nan
    return np.std(v, ddof=1) / np.sqrt(v.size)

def stars(p):
    if not np.isfinite(p): return ""
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 0.05: return "*"
    return "n.s."

def fit_mixedlm(formula, df, groups):
    try:
        model = smf.mixedlm(formula, data=df, groups=df[groups])
        res = model.fit(method="lbfgs", reml=True, disp=False)
        return res
    except Exception as e:
        return None

def fdr_by_family(rows, contrast_key):
    df = pd.DataFrame(rows)
    if df.empty: return df
    out = []
    for contrast in df[contrast_key].unique():
        sub = df[df[contrast_key]==contrast]
        for fam in sub['term_family'].unique():
            ss = sub[sub['term_family']==fam].copy()
            if ss.empty: continue
            rej, q, _, _ = multipletests(ss['pval'].values, method='fdr_bh')
            ss['qval'] = q
            ss['signif'] = rej
            out.append(ss)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--layers", nargs="+", default=["sup","mid","deep"])
    ap.add_argument("--rois", nargs="+", required=True)
    ap.add_argument("--contrasts", nargs="+", default=["CS","US"])
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.metrics)
    df = df[df["roi"].isin(args.rois) & df["contrast"].isin(args.contrasts)]
    needed = {"subject_id","roi","layer","phase","contrast","distance_early","distance_late","slope_bins"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Long format for early/late
    long_rows = []
    for _, r in df.iterrows():
        for time, val in [("early", r["distance_early"]), ("late", r["distance_late"])]:
            long_rows.append(dict(
                subject_id=r["subject_id"], roi=r["roi"], layer=r["layer"],
                phase=r["phase"], contrast=r["contrast"], time=time, value=val
            ))
    dfl = pd.DataFrame(long_rows)
    dfl = dfl[np.isfinite(dfl["value"])]

    fam_rows = []
    per_roi_summary = []

    for roi in args.rois:
        for contrast in args.contrasts:
            # Early-Late model
            dsub = dfl[(dfl["roi"]==roi) & (dfl["contrast"]==contrast)].copy()
            dsub["time"] = pd.Categorical(dsub["time"], categories=["early","late"])
            dsub["layer"] = pd.Categorical(dsub["layer"], categories=args.layers)
            res_el = None
            if not dsub.empty and dsub["subject_id"].nunique() >= 3:
                res_el = fit_mixedlm("value ~ time * layer", dsub, groups="subject_id")
            if res_el is not None:
                pv = res_el.pvalues.to_dict()
                p_time = pv.get("time[T.late]", np.nan)
                fam_rows.append(dict(roi=roi, contrast=contrast, term="time", term_family="EL_time", pval=p_time))
                p_layer_terms = [v for k,v in pv.items() if k.startswith("layer[T.")]
                p_layer = np.nanmin(p_layer_terms) if p_layer_terms else np.nan
                fam_rows.append(dict(roi=roi, contrast=contrast, term="layer", term_family="EL_layer", pval=p_layer))
                p_int_terms = [v for k,v in pv.items() if k.startswith("time[T.late]:layer[T.")]
                p_int = np.nanmin(p_int_terms) if p_int_terms else np.nan
                fam_rows.append(dict(roi=roi, contrast=contrast, term="time:layer", term_family="EL_interaction", pval=p_int))
                ann_time, ann_layerEL, ann_int = p_time, p_layer, p_int
            else:
                fam_rows.extend([
                    dict(roi=roi, contrast=contrast, term="time", term_family="EL_time", pval=np.nan),
                    dict(roi=roi, contrast=contrast, term="layer", term_family="EL_layer", pval=np.nan),
                    dict(roi=roi, contrast=contrast, term="time:layer", term_family="EL_interaction", pval=np.nan),
                ])
                ann_time = ann_layerEL = ann_int = np.nan

            # Slope model
            ds = df[(df["roi"]==roi)&(df["contrast"]==contrast)].copy()
            ds = ds[np.isfinite(ds["slope_bins"])]
            ds["layer"] = pd.Categorical(ds["layer"], categories=args.layers)
            res_sl = None
            if not ds.empty and ds["subject_id"].nunique() >= 3:
                res_sl = fit_mixedlm("slope_bins ~ layer", ds, groups="subject_id")
            if res_sl is not None:
                pv = res_sl.pvalues.to_dict()
                p_layer_terms = [v for k,v in pv.items() if k.startswith("layer[T.")]
                p_layer = np.nanmin(p_layer_terms) if p_layer_terms else np.nan
                fam_rows.append(dict(roi=roi, contrast=contrast, term="layer", term_family="SL_layer", pval=p_layer))
                ann_layerSL = p_layer
            else:
                fam_rows.append(dict(roi=roi, contrast=contrast, term="layer", term_family="SL_layer", pval=np.nan))
                ann_layerSL = np.nan

            # Plot per-ROI×contrast
            figdir = outdir / f"fig_{roi}_{contrast}"
            figdir.mkdir(parents=True, exist_ok=True)

            # early/late per layer
            means_e, means_l, se_e, se_l, p_el = [], [], [], [], []
            for L in args.layers:
                seL = dsub[(dsub["layer"]==L)&(dsub["time"]=="early")]
                slL = dsub[(dsub["layer"]==L)&(dsub["time"]=="late")]
                idx = sorted(set(seL["subject_id"]).intersection(set(slL["subject_id"])))
                if len(idx) >= 2:
                    idx = list(idx)  # ensure list for .loc
                    ve = seL.set_index("subject_id").loc[idx, "value"].astype(float).values
                    vl = slL.set_index("subject_id").loc[idx, "value"].astype(float).values

                    means_e.append(np.mean(ve)); means_l.append(np.mean(vl))
                    se_e.append(sem(ve)); se_l.append(sem(vl))
                    t,p = stats.ttest_rel(ve, vl, nan_policy='omit')
                    p_el.append(p)
                else:
                    means_e.append(np.nan); means_l.append(np.nan)
                    se_e.append(np.nan); se_l.append(np.nan)
                    p_el.append(np.nan)
            if np.sum(np.isfinite(p_el)) >= 2:
                _, q_el, _, _ = multipletests([p for p in p_el if np.isfinite(p)], method='fdr_bh')
                q_iter = iter(q_el)
                q_el_full = [next(q_iter) if np.isfinite(p) else np.nan for p in p_el]
            else:
                q_el_full = [np.nan]*len(p_el)

            # slopes
            slopes, se_s, p_s = [], [], []
            for L in args.layers:
                v = ds[ds["layer"]==L].groupby("subject_id")["slope_bins"].mean()
                vv = v.values
                if vv.size >= 2:
                    slopes.append(np.mean(vv)); se_s.append(sem(vv))
                    t,p = stats.ttest_1samp(vv, 0.0, nan_policy='omit')
                    p_s.append(p)
                else:
                    slopes.append(np.nan); se_s.append(np.nan); p_s.append(np.nan)
            if np.sum(np.isfinite(p_s)) >= 2:
                _, q_s, _, _ = multipletests([p for p in p_s if np.isfinite(p)], method='fdr_bh')
                q_iter = iter(q_s)
                q_s_full = [next(q_iter) if np.isfinite(p) else np.nan for p in p_s]
            else:
                q_s_full = [np.nan]*len(p_s)

            x = np.arange(len(args.layers))
            width = 0.35
            fig = plt.figure(figsize=(7,5))
            plt.bar(x - width/2, means_e, width, yerr=se_e, label="Early")
            plt.bar(x + width/2, means_l, width, yerr=se_l, label="Late")
            plt.plot(x, slopes, marker="o", linestyle="-", label="Slope (per layer)")
            plt.xticks(x, args.layers)
            plt.xlabel("Layer"); plt.ylabel("Crossnobis distance / slope")
            subtitle = f"EL: time p={ann_time:.3g}, layer p≈{ann_layerEL:.3g}, time×layer p≈{ann_int:.3g} | Slope: layer p≈{ann_layerSL:.3g}"
            plt.title(f"{roi} — {contrast}\n"+subtitle)
            plt.legend()
            # layer-level asterisks
            for i,(mE,mL,pe,qe) in enumerate(zip(means_e, means_l, p_el, q_el_full)):
                y = np.nanmax([mE, mL])
                if np.isfinite(y) and np.isfinite(pe):
                    plt.text(i-0.1, y*1.05 if y!=0 else 0.05, ("*" if (qe if np.isfinite(qe) else pe)<0.05 else "n.s."),
                             ha="center", va="bottom", fontsize=10)
            for i,(s,ps,qs) in enumerate(zip(slopes, p_s, q_s_full)):
                if np.isfinite(s) and np.isfinite(ps):
                    plt.text(i+0.1, s*1.05 if s!=0 else 0.05, ("*" if (qs if np.isfinite(qs) else ps)<0.05 else "n.s."),
                             ha="center", va="bottom", fontsize=10, color="C2")
            fig.tight_layout()
            fig.savefig(figdir / f"{roi}_{contrast}_bars_slopes.png", dpi=200)
            plt.close(fig)

            per_roi_summary.append(dict(roi=roi, contrast=contrast,
                                        n_subjects_el=dsub["subject_id"].nunique(),
                                        n_subjects_sl=ds["subject_id"].nunique()))

    fam_df = fdr_by_family(fam_rows, contrast_key="contrast")
    fam_df.to_csv(outdir/"group_LME_FDR_results.csv", index=False)
    pd.DataFrame(per_roi_summary).to_csv(outdir/"per_roi_raw_summary.csv", index=False)
    print("Wrote:", outdir/"group_LME_FDR_results.csv")

if __name__ == "__main__":
    main()

