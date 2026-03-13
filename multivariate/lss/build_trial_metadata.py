
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build trial-level metadata CSV compatible with rsa_crossnobis_pipeline_v8/v8_1.
- Scans 4D LSS maps per subject/phase/run/condition.
- Creates one row per trial (volume) with: subject_id, phase, condition, run, beta_path, volume_index, order_in_phase.
- Conditions limited to: CS-, CS+, US-, US+  (others like US-u are ignored).
- Order logic: within each (subject, phase, condition), trials are ordered chronologically as
  run1 volumes (0..n1-1) then run2 volumes (0..n2-1), producing a continuous 'order_in_phase' starting at 1.
Usage:
  python make_metadata_v8.py \
    --root /media/abouyeure/Elements/MRI_7T/extinction7T/derivatives/RSA \
    --phase_dirs fear_acq fear_ext \
    --subjects sub06 sub07 ... \
    --out ./metadata_tables/rsa_trial_metadata.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import nibabel as nib
import re

COND_KEEP = {"CS-","CS+","US-","US+"}

def find_lss_file(root: Path, phase_dir: str, subject: str, run: int, condition: str):
    # Expected pattern:
    # <root>/<phase_dir>/LSS_maps/<subject>/<subject>_run<run>_<cond>_lssmap.nii.gz
    subdir = root / phase_dir / "fear_acq_cosine_category" / "fear_acq_cosine" / subject 
    # Try exact expected name
    fname = f"{subject}_run{run}_{condition}_lssmap.nii.gz"
    pth = subdir / fname
    if pth.exists():
        return pth
    # Fallback: glob by pattern to be tolerant of minor naming differences
    matches = list(subdir.glob(f"{subject}_run{run}_*{condition}*_lssmap.nii*"))
    return matches[0] if matches else None

def volumes_in_nii(nii_path: Path) -> int:
    img = nib.load(str(nii_path))
    data = img.shape
    return data[3] if len(data) == 4 else 1

def build_metadata(root, phase_dirs, subjects, runs=(1,2)):
    rows = []
    for sid in subjects:
        for phase_dir in phase_dirs:
            # map to human-readable phase name
            phase = "acquisition" if "acq" in phase_dir else ("extinction" if "ext" in phase_dir else phase_dir)
            for cond in ["CS-","CS+","US-","US+"]:
                # collect per-run info
                per_run = []
                for r in runs:
                    f = find_lss_file(root, phase_dir, sid, r, cond)
                    if f is None:
                        per_run.append((r, None, 0))
                        continue
                    try:
                        nvol = volumes_in_nii(f)
                    except Exception:
                        nvol = 0
                    per_run.append((r, f, nvol))
                # Skip condition if no volumes in any run
                if sum(n for _,_,n in per_run) == 0:
                    continue
                # Build rows: run1 vols then run2 vols, order_in_phase starting at 1
                order_ctr = 1
                for r, f, nvol in per_run:
                    if f is None or nvol == 0: 
                        continue
                    for vi in range(nvol):
                        rows.append(dict(
                            subject_id=sid, phase=phase, condition=cond, run=f"run{r}",
                            beta_path=str(f), volume_index=vi, order_in_phase=order_ctr
                        ))
                        order_ctr += 1
    df = pd.DataFrame(rows)
    # Sort for readability
    if not df.empty:
        df = df.sort_values(["subject_id","phase","condition","run","volume_index"]).reset_index(drop=True)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory that contains phase dirs (e.g., .../derivatives/RSA)")
    ap.add_argument("--phase_dirs", nargs="+", default=["fear_acq"], help="Directory names for phases under root")
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    root = Path(args.root)
    df = build_metadata(root, args.phase_dirs, args.subjects)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out} with {len(df)} rows.")

if __name__ == "__main__":
    main()

