#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSA Crossnobis Pipeline (v10.2 - Run-Aware Splitting)
-----------------------------------------------------
Key Updates:
1. EXPLICIT EARLY/LATE SPLIT: Uses 'run' column to define Early vs Late.
   (e.g., Run 1 = Early, Run 2 = Late) instead of just trial count.
2. ADAPTIVE CV:
   - For 'All' (Multiple runs): Uses Leave-One-Run-Out (Train R1, Test R2).
   - For 'Early/Late' (Single run): Uses Stratified Split-Half with Buffer.
3. NORMALIZATION: Output distances are per-voxel (Distance / n_vox).

Dependencies: numpy, pandas, scipy, sklearn, nibabel, joblib, matplotlib
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LinearRegression
import nibabel as nib
import warnings
import json

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    warnings.warn("joblib not found; running single-threaded.")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------- Constants -----------------------

CONDITION_ORDER = ["CS-", "CS+", "US-", "US+"]
PAIR_MAP = {"CS": ("CS-", "CS+"), "US": ("US-", "US+")}

# ----------------------- Utilities -----------------------

def zscore(A, axis=0):
    A = np.asarray(A, dtype=float)
    mu = np.mean(A, axis=axis, keepdims=True)
    sd = np.std(A, axis=axis, keepdims=True)
    sd[sd == 0] = 1.0
    return (A - mu) / sd

def resample_mask_to_img(mask_img, ref_img):
    from nilearn.image import resample_to_img
    res = resample_to_img(mask_img, ref_img, interpolation='nearest',
                          force_resample=True, copy_header=True)
    data = (res.get_fdata() > 0).astype(np.uint8)
    return nib.Nifti1Image(data, ref_img.affine, ref_img.header)

def load_masked_data_from_volumes(trial_specs, mask_img, resample=False, dtype=np.float32):
    first_beta = nib.load(trial_specs[0]["beta_path"])
    m_img = resample_mask_to_img(mask_img, first_beta) if resample else mask_img
    mask = np.asanyarray(m_img.get_fdata(), dtype=bool)
    idx = np.where(mask.ravel())[0]
    
    if idx.size == 0:
        raise ValueError("Mask has zero voxels after (re)sampling.")
        
    n_trials = len(trial_specs)
    raw_data = np.zeros((n_trials, idx.size), dtype=dtype)
    
    for i, spec in enumerate(trial_specs):
        img = nib.load(spec["beta_path"])
        data = img.get_fdata()
        if data.ndim == 4:
            vi = int(spec["volume_index"])
            vol = data[..., vi]
        else:
            vol = data
        vec = vol.ravel()[idx].astype(dtype, copy=False)
        raw_data[i, :] = vec

    valid_voxels = np.isfinite(raw_data).all(axis=0)
    if not valid_voxels.any():
        raise ValueError("No valid finite voxels found in ROI.")
        
    X_clean = raw_data[:, valid_voxels]
    return X_clean, first_beta, idx[valid_voxels]

def build_run_folds(df_subset, buffer_percent=0.10):
    """
    Constructs cross-validation folds intelligently.
    1. If multiple runs exist: Use Run ID (Strict Independence).
    2. If single run exists: Use Stratified Split-Half (Temporal Split with Buffer).
    """
    # 1. Use Run column if available and we have >1 run
    if "run" in df_subset.columns:
        runs = df_subset["run"].to_numpy()
        if len(np.unique(runs)) > 1:
            return runs

    # 2. Fallback: Stratified Split-Half (for Single Run / Early / Late)
    folds = np.zeros(len(df_subset), dtype=int)
    conditions = df_subset["condition"].to_numpy()
    orders = df_subset["order_in_phase"].to_numpy()
    
    unique_conds = np.unique(conditions)
    
    for cond in unique_conds:
        idx_cond = np.where(conditions == cond)[0]
        
        if len(idx_cond) < 2:
            folds[idx_cond] = -1 
            continue
            
        cond_orders = orders[idx_cond]
        sort_args = np.argsort(cond_orders)
        
        n_c = len(idx_cond)
        ranks = np.linspace(0, 1, n_c)
        
        mask0 = ranks < (0.5 - buffer_percent/2) # First half
        mask1 = ranks > (0.5 + buffer_percent/2) # Second half
        
        folds[idx_cond[sort_args[mask0]]] = 0
        folds[idx_cond[sort_args[mask1]]] = 1
        folds[idx_cond[sort_args[(~mask0) & (~mask1)]]] = -1 # Buffer
        
    return folds

def compute_precision(X, shrinkage=True, ridge=1e-3, var_floor=1e-6):
    try:
        if shrinkage:
            lw = LedoitWolf(store_precision=False).fit(X)
            Prec = np.atleast_2d(lw.precision_)
        else:
            v = np.maximum(np.var(X, axis=0), var_floor)
            Prec = np.diag(1.0 / (v + ridge))
    except Exception:
        v = np.maximum(np.var(X, axis=0), var_floor)
        Prec = np.diag(1.0 / (v + ridge))
    return Prec

def mean_class(A, lab, val):
    sel = (lab == val)
    return None if not np.any(sel) else A[sel].mean(axis=0)

def compute_crossnobis_normalized(X, y, folds, cov_mode='train', shrinkage=True, ridge=1e-3):
    uniq = np.unique(folds)
    uniq = uniq[uniq != -1]
    
    if len(uniq) < 2:
        return np.nan, [], "NA"
        
    Xz = X.copy()
    
    # Z-score within fold if possible
    for f in uniq:
        sel = folds == f
        if np.sum(sel) > 2:
            Xz[sel] = zscore(X[sel], axis=0)
    
    Prec_global = None
    if cov_mode == 'global':
        Prec_global = compute_precision(Xz, shrinkage=shrinkage, ridge=ridge)

    per_fold = []
    used_mode = cov_mode
    n_vox = X.shape[1]

    for f in uniq:
        test_mask = folds == f
        train_mask = (folds != f) & (folds != -1)
        
        if np.sum(test_mask) == 0 or np.sum(train_mask) == 0:
            continue
            
        X_tr, X_te = Xz[train_mask], Xz[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]
        
        m0_tr = mean_class(X_tr, y_tr, 0); m1_tr = mean_class(X_tr, y_tr, 1)
        m0_te = mean_class(X_te, y_te, 0); m1_te = mean_class(X_te, y_te, 1)
        
        if any(v is None for v in [m0_tr, m1_tr, m0_te, m1_te]):
            continue 

        diff_tr = m1_tr - m0_tr
        diff_te = m1_te - m0_te
        
        if cov_mode == 'global':
            Prec = Prec_global
        elif cov_mode == 'diag':
            vtr = np.var(X_tr, axis=0)
            Prec = np.diag(1.0 / (vtr + ridge))
        else: # train
            Prec = compute_precision(X_tr, shrinkage=shrinkage, ridge=ridge)

        val = 0.0
        try:
            if Prec.ndim == 2:
                 val = float(diff_tr @ Prec @ diff_te.T)
            else:
                 val = float(np.sum(diff_tr * np.diag(Prec) * diff_te))
        except Exception:
            val = float(np.sum(diff_tr * diff_te))
            used_mode = "identity(fallback)"

        per_fold.append(val)

    if not per_fold:
        return np.nan, [], "NA"

    raw_dist = np.mean(per_fold)
    norm_dist = raw_dist / n_vox 
    
    return raw_dist, per_fold, used_mode

def jackknife_se_from_folds_normalized(per_fold_vals, n_vox):
    vals = np.array(per_fold_vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2: return np.nan
    vals_norm = vals / n_vox
    n = len(vals_norm)
    jk = []
    for i in range(n):
        jk.append(np.mean(np.delete(vals_norm, i)))
    jk = np.array(jk)
    se = np.sqrt((n - 1) * np.var(jk, ddof=1))
    return se

# ----------------------- Per-subject processing -----------------------

def _process_combo(meta_df, mask_df, subject_id, roi, layer, phases, params):
    rows, diags = [], []
    m_mask = mask_df.query("subject_id == @subject_id and roi == @roi and layer == @layer")
    if m_mask.empty:
        return rows, diags

    mask_img = nib.load(m_mask.iloc[0]["mask_path"])
    sub_meta = meta_df.query("subject_id == @subject_id and phase in @phases").copy()

    for phase in phases:
        m = sub_meta.query("phase == @phase").copy()
        if m.empty: continue

        trial_specs = [{"beta_path": bp, "volume_index": int(vi)} 
                       for bp, vi in zip(m["beta_path"], m["volume_index"])]
        
        try:
            X_all, _, idx = load_masked_data_from_volumes(
                trial_specs, mask_img, resample=params['resample_masks']
            )
            n_vox = X_all.shape[1]
        except Exception as e:
            diags.append(dict(subject_id=subject_id, roi=roi, layer=layer, phase=phase,
                              reason=f"load_error:{e}", used=False))
            continue

        if n_vox < params['min_vox']:
            diags.append(dict(subject_id=subject_id, roi=roi, layer=layer, phase=phase,
                              reason=f"min_vox<{params['min_vox']}", used=False))
            continue

        def compute_pair_contrast(tag, contrast_name, sel_mask):
            m_sub = m.loc[sel_mask].copy()
            if m_sub.shape[0] < 4: 
                return np.nan, np.nan, "NA"

            a, b = PAIR_MAP[contrast_name]
            pair_mask = m_sub["condition"].isin([a, b]).to_numpy()
            
            X_sub = X_all[sel_mask][pair_mask, :]
            y_sub = (m_sub.loc[pair_mask, "condition"] == b).astype(int)
            
            # --- ADAPTIVE FOLD GENERATION ---
            # 1. If 'All' (Runs > 1): Uses Run IDs (Train R1, Test R2)
            # 2. If 'Early' (Run 1): Uses Stratified Split-Half
            folds = build_run_folds(m_sub.loc[pair_mask])
            
            if len(np.unique(folds[folds != -1])) < 2:
                return np.nan, np.nan, "Insufficient_Folds"

            d, per_fold, mode_used = compute_crossnobis_normalized(
                X_sub, y_sub, folds, 
                cov_mode=params['cov_mode'], 
                shrinkage=(params['cov_mode']=='train')
            )
            se = jackknife_se_from_folds_normalized(per_fold, n_vox)
            return d, se, mode_used

        # --- EXPLICIT RUN-BASED EARLY/LATE DEFINITION ---
        orders = m["order_in_phase"].to_numpy()
        sel_all = np.ones(len(orders), dtype=bool)

        if "run" in m.columns and m["run"].nunique() > 1:
            # We have run info. Let's sort runs by time and split in half.
            unique_runs = m["run"].unique()
            # Sort runs by their first order_in_phase
            run_starts = [m[m["run"] == r]["order_in_phase"].min() for r in unique_runs]
            sorted_runs = [r for _, r in sorted(zip(run_starts, unique_runs))]
            
            # First half of runs = Early, Second half = Late
            mid = len(sorted_runs) // 2
            early_runs = set(sorted_runs[:mid])
            late_runs = set(sorted_runs[mid:])
            
            early_mask = m["run"].isin(early_runs).to_numpy()
            late_mask = m["run"].isin(late_runs).to_numpy()
        else:
            # Fallback for single run or no metadata: Median split
            early_mask = orders <= np.median(orders)
            late_mask = ~early_mask

        bins_idx = pd.qcut(pd.Series(orders).rank(method="first"), 
                           q=params['bins'], labels=False).to_numpy()

        for contrast_name in ["CS", "US"]:
            d_all, se_all, _ = compute_pair_contrast("all", contrast_name, sel_all)
            d_early, se_early, _ = compute_pair_contrast("early", contrast_name, early_mask)
            d_late, se_late, _ = compute_pair_contrast("late", contrast_name, late_mask) # Using new late_mask
            
            bin_d = []
            valid_x = []
            for b in sorted(np.unique(bins_idx)):
                sel = bins_idx == b
                db, _, _ = compute_pair_contrast(f"bin{int(b)}", contrast_name, sel)
                if np.isfinite(db):
                    bin_d.append(db)
                    valid_x.append(b)
            
            slope = np.nan
            if len(bin_d) >= 2:
                lr = LinearRegression().fit(np.array(valid_x).reshape(-1,1), bin_d)
                slope = float(lr.coef_[0])

            rows.append(dict(
                subject_id=subject_id, roi=roi, layer=layer, phase=phase, contrast=contrast_name,
                n_trials=m.shape[0], n_vox=n_vox,
                distance_all=d_all, se_all=se_all,
                distance_early=d_early, se_early=se_early,
                distance_late=d_late, se_late=se_late,
                slope_bins=slope
            ))

    return rows, diags

def main():
    ap = argparse.ArgumentParser(description="RSA Crossnobis (v10.2 - Run-Aware)")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--mask_table", required=True)
    ap.add_argument("--subjects", nargs="+", default=["all"])
    ap.add_argument("--phases", nargs="+", default=["acquisition"])
    ap.add_argument("--rois", nargs="+", required=True)
    ap.add_argument("--layers", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cov_mode", choices=["train", "diag", "global"], default="train")
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--min_vox", type=int, default=10)
    ap.add_argument("--bins", type=int, default=3)
    ap.add_argument("--resample_masks", action="store_true")

    args = ap.parse_args()

    meta = pd.read_csv(args.metadata)
    masks = pd.read_csv(args.mask_table)
    
    meta = meta[meta["phase"].isin(args.phases)].copy()
    meta = meta[meta["condition"].isin(CONDITION_ORDER)].copy()
    subjects = sorted(meta["subject_id"].unique()) if args.subjects == ["all"] else args.subjects
    
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    
    params = dict(
        resample_masks=args.resample_masks,
        min_vox=args.min_vox,
        cov_mode=args.cov_mode,
        bins=args.bins
    )

    all_rows = []
    
    for sid in subjects:
        print(f"Processing {sid}...", flush=True)
        combos = [(roi, layer) for roi in args.rois for layer in args.layers]
        
        if HAS_JOBLIB and args.n_jobs != 1:
            res = Parallel(n_jobs=args.n_jobs)(delayed(_process_combo)(
                meta, masks, sid, roi, layer, args.phases, params) for roi, layer in combos)
        else:
            res = [_process_combo(meta, masks, sid, roi, layer, args.phases, params) for roi, layer in combos]
            
        for r, d in res:
            all_rows.extend(r)

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(outdir / "all_subject_metrics.csv", index=False)
        print(f"Success. Computed metrics for {len(df)} conditions.")
        print(f"NOTE: 'distance_*' columns are normalized (Distance / n_vox).")
    else:
        print("No valid results found. Check masks/metadata.")

if __name__ == "__main__":
    main()
