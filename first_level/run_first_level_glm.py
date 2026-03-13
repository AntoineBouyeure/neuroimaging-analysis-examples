
#!/usr/bin/env python3
"""First-level GLM for high-resolution fear-learning data.

Supports run-wise subject-level modelling for acquisition or extinction sessions
using event tables derived from the onset-preparation utilities in this folder.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import gc
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix

TR = 0
N_SCANS = 0



def build_events(onsets_df: pd.DataFrame) -> pd.DataFrame:
    cs = onsets_df.copy()
    us = onsets_df.copy()

    events = pd.concat([
        cs[['onset', 'duration', 'trial_type']],
        us[['onset', 'duration', 'trial_type']],
    ], ignore_index=True).sort_values('onset').reset_index(drop=True)
    return events


def fit_run(
    subject: str,
    run: str,
    func_root: Path,
    confounds_dir: Path,
    onsets_dir: Path,
    output_dir: Path,
    cache_dir: Path | None,
    drift_model: str,
    drift_order: int,
    session_label: str,
    confounds_pattern: str,
    onsets_pattern: str,
    func_pattern: str,
    mask_img: str | None = None,
    tr: float = TR,
    n_scans: int = N_SCANS,
) -> None:
    fmri_img_path = Path(str(func_pattern).format(func_root=func_root, subject=subject, session=session_label, run=run))
    confounds_path = Path(str(confounds_pattern).format(confounds_dir=confounds_dir, subject=subject, run=run, session=session_label))
    onsets_path = Path(str(onsets_pattern).format(onsets_dir=onsets_dir, subject=subject, run=run, session=session_label, part=run[-1]))

    confounds = pd.read_csv(confounds_path)
    onsets = pd.read_csv(onsets_path)
    events = build_events(onsets)
    frame_times = np.arange(n_scans) * tr
    design_matrix = make_first_level_design_matrix(
        frame_times,
        events=events,
        hrf_model='glover',
        drift_model=drift_model,
        drift_order=drift_order if drift_model == 'polynomial' else None,
        high_pass=0.008 if drift_model == 'cosine' else None,
        add_regs=confounds,
    )

    glm = FirstLevelModel(
        t_r=tr,
        noise_model='ar1',
        standardize=True,
        hrf_model='glover',
        drift_model=drift_model,
        drift_order=drift_order if drift_model == 'polynomial' else None,
        high_pass=0.008 if drift_model == 'cosine' else None,
        signal_scaling=0,
        subject_label=subject,
        n_jobs=1,
        mask_img=mask_img,
        smoothing_fwhm=None,
        memory=str(cache_dir) if cache_dir else None,
    )
    glm = glm.fit(str(fmri_img_path), design_matrices=design_matrix)

    contrast_matrix = np.eye(design_matrix.shape[1])
    contrasts = {column: contrast_matrix[i] for i, column in enumerate(design_matrix.columns)}
    requested = {
        'csplus_v_csminus': contrasts['CS+'] - contrasts['CS-'],
        'usplus_v_usminus': contrasts['US+'] - contrasts['US-'],
    }

    suffix = 'cosine' if drift_model == 'cosine' else 'polynomial'
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, contrast in requested.items():
        result = glm.compute_contrast(contrast, output_type='all')
        nib.save(result['z_score'], output_dir / f'{subject}_{run}_{name}_zscore_{suffix}.nii.gz')
        nib.save(result['effect_size'], output_dir / f'{subject}_{run}_{name}_effectsize_{suffix}.nii.gz')
        nib.save(result['effect_variance'], output_dir / f'{subject}_{run}_{name}_effectvariance_{suffix}.nii.gz')
    del glm, design_matrix
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description='Run first-level GLM for 7T fear-learning data.')
    parser.add_argument('--subjects', nargs='+', required=True)
    parser.add_argument('--runs', nargs='+', default=['run3', 'run4'])
    parser.add_argument('--func-root', required=True)
    parser.add_argument('--confounds-dir', required=True)
    parser.add_argument('--onsets-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--cache-dir', default=None)
    parser.add_argument('--drift-model', choices=['cosine', 'polynomial'], default='polynomial')
    parser.add_argument('--drift-order', type=int, default=2)
    parser.add_argument('--mask-img', default=None)
    parser.add_argument('--session-label', default='ses02')
    parser.add_argument('--func-pattern', default='{func_root}/{subject}/{session}/func_{run}_POCS_bbrreg_MotDistCor_anatomySpace.nii.gz')
    parser.add_argument('--confounds-pattern', default='{confounds_dir}/{subject}_confounds_fearext_{run}.csv')
    parser.add_argument('--onsets-pattern', default='{onsets_dir}/{subject}_onsets_fearext_part{part}.csv')
    parser.add_argument('--tr', type=float, default=TR)
    parser.add_argument('--n-scans', type=int, default=N_SCANS)
    args = parser.parse_args()

    for subject in args.subjects:
        for run in args.runs:
            fit_run(
                subject=subject,
                run=run,
                func_root=Path(args.func_root),
                confounds_dir=Path(args.confounds_dir),
                onsets_dir=Path(args.onsets_dir),
                output_dir=Path(args.output_dir),
                cache_dir=Path(args.cache_dir) if args.cache_dir else None,
                drift_model=args.drift_model,
                drift_order=args.drift_order,
                session_label=args.session_label,
                confounds_pattern=args.confounds_pattern,
                onsets_pattern=args.onsets_pattern,
                func_pattern=args.func_pattern,
                mask_img=args.mask_img,
                tr=args.tr,
                n_scans=args.n_scans,
            )


if __name__ == '__main__':
    main()
