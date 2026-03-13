# ROI connectivity

This folder contains ROI-based connectivity utilities for trialwise or condition-wise beta-series analyses.

## Scripts

- `compute_roi_connectivity.py` builds a subject atlas from ROI masks and computes condition-wise connectivity matrices (correlation or partial correlation) from concatenated beta maps.
- `compare_sessions.py` compares connectivity matrices between two sessions and exports edge-wise paired statistics with FDR correction.

## Typical use cases

- beta-series connectivity from LSS-derived trial maps;
- comparison of acquisition vs extinction connectivity patterns;
- ROI-network summaries for downstream visualization or statistics.
