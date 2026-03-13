# neuroimaging-analysis-examples

Example workflows for ultra-high-field (7T) neuroimaging analysis.

## Layout

- `preprocessing/`: ANTs/FSL-based preprocessing orchestration and anatomical tissue segmentation.
- `registration/`: anatomical registration and FreeSurfer segmentation.
- `layers/`: layer-specific extraction, modelling, and laminar RSA utilities.
- `first_level/`: subject-level GLM and event/onset preparation.
- `second_level/`: fixed-effects combination and group-level GLM/permutation testing.
- `multivariate/lss/`: trial-wise LSS estimation with MPI/HPC examples.
- `multivariate/searchlight/`: MPI searchlight analyses.
- `multivariate/rsa/`: ROI-based RSA and LSS-map preparation.
- `multivariate/connectivity/`: ROI-based beta-series connectivity and between-session comparisons.
- `group_stats/`: group-level mixed-effects summaries and plotting.

## Scope

The repository is organized around common stages of a high-resolution fMRI workflow, from preprocessing and registration through univariate, multivariate, connectivity, and layer-specific analyses. The scripts are intended as reusable examples for methodological organization, statistical modelling, and HPC-friendly execution.
