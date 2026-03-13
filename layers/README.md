# Layer analyses

This folder contains utilities for layer-specific analyses in high-resolution / laminar fMRI datasets.

## Scripts

- `laynii_devein.sh` runs a simple LAYNII-based deveining workflow with accompanying tSNR quality-control outputs.
- `extract_layer_profiles.py` extracts ROI × layer summary values from NIfTI maps using layer-label masks (e.g., deep, middle, superficial).
- `layer_stats.py` performs mixed-effects modelling and one-sample testing on long-format layer tables and can export summary figures.
- `layer_rsa.py` summarizes ROI- and layer-specific similarity matrices into long-format tables suitable for downstream statistics.
- `compare_layer_rsa_between_sessions.py` compares layer-wise RSA summaries across sessions (e.g., acquisition vs extinction) with paired statistics and FDR correction.

## Typical use cases

- extraction of layer-specific beta or contrast summaries;
- ROI-wise laminar modelling;
- within-session and between-session RSA comparisons across cortical depth.
