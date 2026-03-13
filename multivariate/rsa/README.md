# ROI-based RSA

This folder contains ROI-based representational similarity analysis utilities.

## Scripts

- `rsa_crossnobis.py` performs cross-run, crossnobis-based ROI RSA and supports model-level summaries suitable for inference.
- `prepare_lss_maps.py` prepares inputs for RSA by concatenating trialwise LSS maps and exporting simple mask-size / trial-count summaries.

## Typical use cases

- trialwise or condition-wise ROI RSA;
- preparation of LSS-derived beta series for multivariate analyses;
- metadata and QC summaries prior to RSA modelling.
