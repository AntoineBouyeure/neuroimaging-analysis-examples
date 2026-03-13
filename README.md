
# neuroimaging-analysis-examples

Example scripts for ultra-high-field (7T) MRI analysis, reorganized into a cleaner repository structure.

## Layout

- `preprocessing/`: ANTs/FSL-based preprocessing orchestration.
- `registration/`: anatomical registration and FreeSurfer segmentation.
- `layers/`: layer-specific utilities (e.g. LAYNII deveining).
- `first_level/`: subject-level GLM examples.
- `second_level/`: group-level GLM and permutation testing.
- `multivariate/lss/`: trial-wise LSS estimation with MPI/HPC examples.
- `multivariate/searchlight/`: MPI searchlight RSA examples.
- `multivariate/rsa/`: ROI-based RSA / crossnobis utilities.
- `group_stats/`: group-level mixed-effects summaries and plotting.

These scripts are designed as reusable examples rather than a drop-in replacement for any specific dataset.
