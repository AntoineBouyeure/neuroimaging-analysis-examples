# First-level GLM and event preparation

This folder contains subject-level modelling utilities for high-resolution fear-learning fMRI data.

## Scripts

- `run_first_level_glm.py` fits run-wise first-level GLMs using Nilearn. The script builds event tables from CS and US onset columns, incorporates nuisance regressors from confound files, and exports contrast-specific effect-size, effect-variance, and z-score maps for downstream fixed-effects or second-level analyses.
- `prepare_onset_files.py` prepares event/onset tables used by the first-level GLM. It can annotate existing onset files with stimulus-type labels derived from outcome codes, or build compact events tables from behavioral exports.

## Typical use cases

- run-wise modelling of acquisition or extinction sessions;
- generation of subject-level contrast maps for fixed-effects and group-level analysis;
- harmonization of onset/event files prior to univariate modelling.
