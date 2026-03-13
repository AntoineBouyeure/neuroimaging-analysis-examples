# Preprocessing

This folder contains orchestration scripts for a high-resolution 7T fMRI preprocessing workflow based on the ANTs/FSL helper scripts distributed with this repository.

## Files

- `run_7t_preprocessing.sh`: main shell entry point for subject-level preprocessing.
- `config.example.sh`: example configuration with paths and dataset-specific settings.
- `runs.example.tsv`: example run manifest.
- `../sk_ants*.sh`: helper scripts used by the orchestration layer.

## Workflow summary

The wrapper script coordinates:

1. motion and distortion estimation for each run;
2. selection of session-level reference runs;
3. inter-run alignment;
4. functional-to-anatomical registration;
5. final one-shot reslicing.

## Usage

```bash
cp config.example.sh config.sh
bash run_7t_preprocessing.sh config.sh runs.tsv
```

The wrapper is intentionally lightweight and keeps the original helper scripts unchanged.
