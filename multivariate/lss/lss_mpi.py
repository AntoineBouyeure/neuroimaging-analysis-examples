#!/usr/bin/env python
"""
Layer-fMRI LSS per-trial GLM with MPI (CPU-friendly, robust, low-memory)

- Default noise_model='ols' (override with env LSS_NOISE_MODEL=ar1)
- standardize=False (avoid extra arrays)
- Confounds cast to float32 (numeric cols only) and passed as NumPy array
- Use file paths (strings) for fmri & mask so nilearn handles masking efficiently
- Nilearn caching enabled (env LSS_CACHE, default /tmp/nilearn_cache)
- Periodically recreate GLM to free caches (RESET_EVERY)
- Save trial betas as float32
- Combine via streaming mean (constant memory)
- Includes both CS and US events

LSS MODES:
- 'cs': only CS trials (trial_type starting with 'CS') get per-trial LSS regressors;
        US events are modeled as nuisance regressors.
- 'us': only US trials (trial_type starting with 'US') get per-trial LSS regressors;
        CS events are modeled as nuisance regressors.
"""

from nilearn.glm.first_level import FirstLevelModel
import nibabel as nib
import pandas as pd
import numpy as np
import os
import sys
import gc
import glob
import re
import traceback
import urllib.parse as ul
from mpi4py import MPI

# ---------------- MPI ----------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def log0(msg: str):
    if rank == 0:
        print(msg, flush=True)

# Noise model: default 'ols' for CPU partition; allow override via env
NOISE_MODEL = os.environ.get("LSS_NOISE_MODEL", "ols").lower()
if NOISE_MODEL not in {"ols", "ar1"}:
    NOISE_MODEL = "ols"

# How often to recreate the GLM to drop internal caches
RESET_EVERY = int(os.environ.get("LSS_RESET_EVERY", "8"))

# Nilearn cache directory
NILEARN_CACHE_DIR = os.environ.get("LSS_CACHE", "/tmp/nilearn_cache")

# ---------------- I/O helpers ----------------
def path_for_run(input_root: str, subject_id: str, run_num: int):
    """
    Filenames as defined in your pipeline.
    """
    fmri_path = os.path.join(
        input_root, f"func_run{run_num}_POCS_bbrreg_MotDistCor_anatomySpace.nii.gz"
    )
    confounds_path = os.path.join(input_root, f"{subject_id}_confounds_fearext_run{run_num}.csv")
    onsets_path = os.path.join(input_root, f"{subject_id}_onsets_fearext_part{run_num}.csv")
    return fmri_path, confounds_path, onsets_path

def load_run_data(input_root, subject_id, run_num):
    """
    Load events & confounds for the given run on rank 0; broadcast to all ranks.
    - Downcast confounds (numeric only) to float32 NumPy
    - Combine CS and US events and sort by onset
    """
    events, confounds_np = None, None

    if rank == 0:
        try:
            fmri_path, confounds_path, onsets_path = path_for_run(input_root, subject_id, run_num)
            for pth, label in [(fmri_path, "fMRI"), (confounds_path, "confounds"), (onsets_path, "onsets")]:
                if not os.path.exists(pth):
                    raise FileNotFoundError(f"{label} file not found for run{run_num}: {pth}")

            # Confounds -> numeric columns -> float32 NumPy
            confounds_df = pd.read_csv(confounds_path)
            numeric_cols = confounds_df.select_dtypes(include=[np.number])
            confounds_np = numeric_cols.to_numpy(dtype=np.float32)

            onsets = pd.read_csv(onsets_path)
            required_cs = {'CS_onset', 'CS_duration', 'CStype_real'}
            required_us = {'US_onset', 'US_duration', 'UStype_new'}
            if not required_cs.issubset(onsets.columns):
                missing = required_cs - set(onsets.columns)
                raise ValueError(f"Onsets missing CS columns {missing} in {onsets_path}")
            if not required_us.issubset(onsets.columns):
                missing = required_us - set(onsets.columns)
                raise ValueError(f"Onsets missing US columns {missing} in {onsets_path}")

            cs_events = onsets[['CS_onset', 'CS_duration', 'CStype_real']].rename(
                columns={'CS_onset': 'onset', 'CS_duration': 'duration', 'CStype_real': 'trial_type'}
            )
            us_events = onsets[['US_onset', 'US_duration', 'UStype_new']].rename(
                columns={'US_onset': 'onset', 'US_duration': 'duration', 'UStype_new': 'trial_type'}
            )
            events = pd.concat([cs_events, us_events], ignore_index=True).sort_values('onset').reset_index(drop=True)

            if events['onset'].isna().any() or events['duration'].isna().any():
                raise ValueError("NaNs in events onset/duration.")

            log0(f"[run{run_num}] Loaded events {len(events)} | confounds {confounds_np.shape}")

            # Cleanup intermediates early
            del confounds_df, numeric_cols, onsets
            gc.collect()

        except Exception as e:
            log0(f"Error loading data for run{run_num}: {e}")
            traceback.print_exc()
            comm.Abort(1)

    # Broadcast to all ranks
    events = comm.bcast(events, root=0)
    confounds_np = comm.bcast(confounds_np, root=0)
    return events, confounds_np

# ---------------- LSS transformer ----------------
def lss_transformer(events_run: pd.DataFrame, row_number: int):
    """
    LSS per trial: duplicate events; rename only target row's trial_type to unique 'cond__NNN'.
    """
    df = events_run.copy()
    trial_condition = df.loc[row_number, 'trial_type']
    mask = df['trial_type'].eq(trial_condition)
    trial_number = df[mask].index.get_loc(row_number)  # 0-based within condition
    unique_label = f"{trial_condition}__{trial_number:03d}"
    df.loc[row_number, 'trial_type'] = unique_label
    return df, unique_label, trial_condition, trial_number

# ---------------- Main per-subject processing ----------------
def process_subject(subject_id, input_root, output_root, lss_mode):
    """
    Run per-trial LSS for a subject in a given mode:
    - lss_mode='cs': only CS trials (trial_type starting with 'CS') get LSS regressors.
    - lss_mode='us': only US trials (trial_type starting with 'US') get LSS regressors.

    CS and US events are always in the design matrix:
    - In CS mode, US act as nuisance regressors.
    - In US mode, CS act as nuisance regressors.
    """
    # Ensure output dir
    if rank == 0:
        os.makedirs(output_root, exist_ok=True)
    comm.Barrier()

    # Mask path (pass string to nilearn)
    mask_path = os.path.join(input_root, "gm_cropped.nii.gz")
    if rank == 0 and not os.path.exists(mask_path):
        log0(f"ERROR: Mask not found: {mask_path}")
        comm.Abort(1)
    mask_path = comm.bcast(mask_path, root=0)

    for run_num in [3,4]:
        fmri_path, _, _ = path_for_run(input_root, subject_id, run_num)
        if rank == 0:
            log0(f"[run{run_num}] fMRI path: {fmri_path}")

        # Get n_scans with a lightweight header read, then drop object
        img_hdr = nib.load(fmri_path, mmap=True)
        n_scans = img_hdr.shape[-1]
        del img_hdr
        gc.collect()

        # Load per-run events/confounds
        events, confounds_np = load_run_data(input_root, subject_id, run_num)
        if confounds_np.shape[0] != n_scans and rank == 0:
            log0(f"[run{run_num}] WARNING: confounds rows ({confounds_np.shape[0]}) != n_scans ({n_scans})")

        # Select trials of interest according to LSS mode
        trial_types = events["trial_type"].astype(str)

        if lss_mode == "cs":
            # Only CS trials (e.g., 'CS+', 'CS-', 'CS-u') get an LSS regressor
            mask_target = trial_types.str.startswith("CS")
        elif lss_mode == "us":
            # Only US trials (e.g., 'US+', 'US-', 'US-u') get an LSS regressor
            mask_target = trial_types.str.startswith("US")
        else:
            if rank == 0:
                log0(f"ERROR: Unknown lss_mode '{lss_mode}' in process_subject")
            comm.Abort(1)

        idx_all = np.where(mask_target.values)[0]
        n_trials = len(idx_all)

        # Distribute these target trials across ranks
        trials_for_rank = idx_all[rank::size]

        if rank == 0:
            log0(
                f"[run{run_num}] LSS mode='{lss_mode}' | "
                f"total trials of interest: {n_trials} | "
                f"Rank {rank} runs {len(trials_for_rank)}"
            )

        if n_trials == 0 and rank == 0:
            log0(f"[run{run_num}] WARNING: No trials of interest found for mode '{lss_mode}'")

        # GLM factory so we can recreate it periodically
        def make_glm():
            return FirstLevelModel(
                t_r=3.1,
                noise_model=NOISE_MODEL,       # 'ols' on CPU unless overridden
                hrf_model='glover',
                drift_model='cosine',
                drift_order=128,
                mask_img=mask_path,            # pass path (string), nilearn handles masking efficiently
                standardize=False,             # avoid extra copies
                smoothing_fwhm=None,
                memory=NILEARN_CACHE_DIR,      # enable caching
                memory_level=2,                # cache calculated intermediates
                minimize_memory=True,          # memory optimization
                verbose=1 if rank == 0 else 0,
            )

        glm = make_glm()

        for tcount, trial_idx in enumerate(trials_for_rank, start=1):
            try:
                lss_events, unique_label, base_condition, trial_number = lss_transformer(events, trial_idx)
                onset_time = lss_events.loc[trial_idx, 'onset']

                # Fit GLM for this trial using file path (string) and compact NumPy confounds
                glm.fit(fmri_path, lss_events, confounds=confounds_np)

                # Vector contrast is robust to '+' in condition names
                design = glm.design_matrices_[0]
                col_idx = design.columns.get_loc(unique_label)
                c = np.zeros(design.shape[1], dtype=np.float32)
                c[col_idx] = 1.0

                beta_map = glm.compute_contrast(c, output_type='effect_size')

                # Downcast to float32; ensure clean header scaling
                beta_data = beta_map.get_fdata(dtype=np.float32)
                hdr = beta_map.header.copy()
                hdr.set_data_dtype(np.float32)
                try:
                    hdr.set_slope_inter(1.0, 0.0)
                except Exception:
                    pass
                beta_img = nib.Nifti1Image(beta_data, beta_map.affine, hdr)

                # Save with key=value scheme (URL-encode condition)
                cond_enc = ul.quote(base_condition, safe='')
                temp_path = os.path.join(
                    output_root,
                    f"temp_cond={cond_enc}_trial={trial_number:03d}_onset={onset_time:.6f}_rank={rank}_run={run_num}.nii.gz"
                )
                nib.save(beta_img, temp_path)

                # Cleanup per trial
                del lss_events, beta_map, beta_data, beta_img, design, c
                gc.collect()

                if trial_idx == trials_for_rank[0] and rank == 0:
                    log0(
                        f"[run{run_num}] First processed: "
                        f"cond={base_condition}, trial={trial_number:03d}, onset={onset_time:.3f}"
                    )

                # Periodically recreate GLM to drop internal caches
                if RESET_EVERY > 0 and (tcount % RESET_EVERY) == 0:
                    del glm
                    gc.collect()
                    glm = make_glm()

            except Exception as e:
                print(f"[run{run_num}] Rank {rank} failed trial {trial_idx}: {e}")
                traceback.print_exc()
                comm.Abort(1)

        # Per-run cleanup
        del glm
        gc.collect()
        comm.Barrier()

# ---------------- Combine step ----------------
_TAG_RE_CACHE = {}

def _compiled_tag_re(key: str):
    r = _TAG_RE_CACHE.get(key)
    if r is None:
        r = re.compile(rf"{key}=([^_]+)")
        _TAG_RE_CACHE[key] = r
    return r

def parse_tag(fname: str, key: str):
    r = _compiled_tag_re(key)
    m = r.search(os.path.basename(fname))
    return m.group(1) if m else None

def combine_maps(output_root, subject_id):
    if rank != 0:
        return

    log0(f"Combining maps for subject {subject_id}…")

    for run_num in [3,4]:
        patt = os.path.join(output_root, f"temp_*_run={run_num}.nii.gz")
        all_temp_files = glob.glob(patt)
        if not all_temp_files:
            log0(f"[run{run_num}] No temp files found; skipping.")
            continue

        # Group by decoded base condition
        groups = {}
        for f in all_temp_files:
            cond_enc = parse_tag(f, "cond")
            if not cond_enc:
                continue
            cond = ul.unquote(cond_enc)
            groups.setdefault(cond, []).append(f)

        if not groups:
            log0(f"[run{run_num}] No valid groups; skipping.")
            continue

        for condition, files in groups.items():
            # Sort by within-condition trial index (stable, independent of rank/node)
            def trial_index(file_path):
                t = parse_tag(file_path, "trial")
                return int(t) if t is not None else 0

            files_sorted = sorted(files, key=trial_index)

            log0(f"[run{run_num}] Combining {len(files_sorted)} files for condition '{condition}'")
            for i, f in enumerate(files_sorted[:5]):
                on = parse_tag(f, "onset")
                log0(f"  {i+1:02d}: {os.path.basename(f)} (onset={on})")

            # Streaming mean (constant memory)
            first_img = nib.load(files_sorted[0])
            affine = first_img.affine
            header = first_img.header.copy()
            header.set_data_dtype(np.float32)
            try:
                header.set_slope_inter(1.0, 0.0)
            except Exception:
                pass

            mean_data = np.zeros(first_img.shape, dtype=np.float32)
            k = 0
            for i, fp in enumerate(files_sorted, 1):
                img = nib.load(fp)
                x = img.get_fdata(dtype=np.float32)
                if k == 0:
                    mean_data[...] = x
                    k = 1
                else:
                    k += 1
                    mean_data += (x - mean_data) / k
                del img, x
                if (i % 10) == 0:
                    log0(f"  averaged {i}/{len(files_sorted)}")
                gc.collect()

            combined_img = nib.Nifti1Image(mean_data, affine, header)
            safe_cond = ul.quote(condition, safe='')
            final_path = os.path.join(output_root, f"{subject_id}_run{run_num}_{safe_cond}_lssmap.nii.gz")
            nib.save(combined_img, final_path)
            log0(f"  Saved {final_path}")

            # Cleanup
            del mean_data, combined_img, first_img
            gc.collect()

# ---------------- Entrypoint ----------------
if __name__ == "__main__":
    if len(sys.argv) != 5:
        if rank == 0:
            print("Usage: python script.py <input_path> <output_path> <subject_id> <lss_mode>")
            print("  <lss_mode> should be 'cs' or 'us'")
        comm.Abort(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    subject_id = sys.argv[3]
    lss_mode = sys.argv[4].lower()

    if lss_mode not in {"cs", "us"}:
        if rank == 0:
            print(f"ERROR: lss_mode must be 'cs' or 'us', got '{lss_mode}'")
        comm.Abort(1)

    if rank == 0:
        log0(f"Starting LSS processing for subject {subject_id}")
        log0(f"Input path:  {input_path}")
        log0(f"Output path: {output_path}")
        log0(f"Noise model: {NOISE_MODEL}")
        log0(f"RESET_EVERY: {RESET_EVERY}")
        log0(f"Nilearn cache: {NILEARN_CACHE_DIR}")
        log0(f"LSS mode: {lss_mode}")

    try:
        process_subject(subject_id, input_path, output_path, lss_mode)
        comm.Barrier()

        # Exit all non-root ranks BEFORE combine to avoid OOM
        if rank != 0:
            print(f"Processing complete for subject {subject_id} on rank {rank}", flush=True)
            gc.collect()
            sys.exit(0)

        # Rank 0 alone does the combine
        combine_maps(output_path, subject_id)
        print(f"Processing complete for subject {subject_id} on rank {rank}", flush=True)

    except Exception as e:
        print(f"Fatal error for subject {subject_id} on rank {rank}: {e}", flush=True)
        traceback.print_exc()
        comm.Abort(1)

