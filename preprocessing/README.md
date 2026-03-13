# Preprocessing

This folder contains orchestration and anatomical tissue-segmentation utilities for a high-resolution 7T fMRI workflow based on ANTs, FSL, and custom helper scripts.

## Files

- `run_7t_preprocessing.sh`: main shell entry point for subject-level preprocessing.
- `config.example.sh`: example configuration with paths and dataset-specific settings.
- `runs.example.tsv`: example run manifest.
- `run_fast_tissue_segmentation.py`: runs FSL FAST on skull-stripped anatomical images to generate tissue-segmentation outputs used in downstream masking or QC.
- `../sk_ants*.sh`: helper scripts used by the orchestration layer.

## Workflow summary

The wrapper script coordinates:

1. motion and distortion estimation for each run;
2. selection of session-level reference runs;
3. inter-run alignment;
4. functional-to-anatomical registration;
5. final one-shot reslicing.

The FAST utility adds a simple anatomical segmentation stage for extracting tissue classes from high-resolution T1 images.

## Usage

```bash
cp config.example.sh config.sh
bash run_7t_preprocessing.sh config.sh runs.tsv
python run_fast_tissue_segmentation.py --t1-brain /path/to/UNI_MPRAGEised_brain.nii.gz --output-dir /path/to/fast
```
