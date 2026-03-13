
#!/usr/bin/env python3
"""First-level GLM example for 7T fear-extinction data."""
from __future__ import annotations
import argparse
from pathlib import Path
import gc
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix

TR = 3.1
N_SCANS = 316


def build_events(onsets_df: pd.DataFrame) -> pd.DataFrame:
    cs = onsets_df.copy()
    cs['onset'] = cs['CS_onset']
    cs['duration'] = cs['CS_duration']
    cs['trial_type'] = cs['CStype_new']

    us = onsets_df.copy()
    us['onset'] = us['US_onset']
    us['duration'] = us['US_duration']
    us['trial_type'] = us['UStype_new']

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
    mask_img: str | None = None,
) -> None:
    fmri_img_path = func_root / subject / 'ses02' / f'func_{run}_POCS_bbrreg_MotDistCor_anatomySpace.nii.gz'
    confounds_path = confounds_dir / f'{subject}_confounds_fearext_{run}.csv'
    onsets_path = onsets_dir / f'{subject}_onsets_fearext_part{run[-1]}.csv'

    confounds = pd.read_csv(confounds_path)
    onsets = pd.read_csv(onsets_path)
    events = build_events(onsets)
    frame_times = np.arange(N_SCANS) * TR
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
        t_r=TR,
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
    parser = argparse.ArgumentParser(description='Run first-level GLM for 7T fear-extinction data.')
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
                mask_img=args.mask_img,
            )


if __name__ == '__main__':
    main()
