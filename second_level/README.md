# Second-level GLM and fixed-effects combination

This folder contains group-level and subject-level summary utilities for univariate analyses.

## Scripts

- `run_second_level_glm.py` performs second-level modelling and non-parametric inference from subject-specific contrast maps.
- `fixed_effects_from_runs.py` combines run-wise effect-size and effect-variance maps into subject-level fixed-effects maps, which can then be entered into second-level analyses.
- `run_second_level_glm.example.sh` provides a simple example launcher for cluster or terminal use.

## Typical use cases

- combine multiple runs into subject-level summaries;
- perform acquisition/extinction group analyses;
- export statistical maps for ROI, layer, or visualization workflows.
