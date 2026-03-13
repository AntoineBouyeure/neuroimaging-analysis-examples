
# Neuroimaging Analysis Examples (7T MRI)

Example workflows for **ultra‑high‑field (7T) neuroimaging analysis**, covering preprocessing, laminar analysis, univariate GLM modeling, multivariate pattern analysis, representational similarity analysis (RSA), connectivity analyses, and group‑level statistics.

The repository illustrates modular strategies commonly used in **high‑resolution cognitive and affective neuroscience studies**.  
Scripts are organized according to the major stages of a typical neuroimaging analysis pipeline.

---

![Layer‑fMRI pipeline](docs/figures/layer_pipeline.png)

---

# Quickstart

Clone the repository:

```bash
git clone https://github.com/AntoineBouyeure/neuroimaging-analysis-examples.git
cd neuroimaging-analysis-examples
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Some modules require commonly used neuroimaging software:

- ANTs  
- FreeSurfer  
- FSL  
- LAYNII  

These tools should be installed separately and available in your system environment.

---

# Overview of the Analysis Workflow

The repository demonstrates a complete neuroimaging workflow spanning:

1. Preprocessing of high‑resolution MRI data  
2. Anatomical segmentation and spatial registration  
3. Layer‑resolved ROI analyses  
4. First‑ and second‑level GLM modeling  
5. Multivariate analyses (searchlight, LSS)  
6. Representational similarity analysis (RSA)  
7. Functional connectivity analyses  
8. Group‑level statistical modeling  

---

# Conceptual Framework

Many analyses in this repository focus on **changes in neural representations across experimental conditions**.

These representational differences are commonly studied using multivariate methods such as:

- searchlight decoding
- representational similarity analysis (RSA)
- cross‑validated distance metrics (e.g., crossnobis)

Such approaches allow researchers to characterize how neural population codes evolve across stimuli, conditions, or experimental phases.

---

# Layer‑Resolved fMRI Analyses

The repository also includes workflows for **layer‑resolved analysis of high‑resolution fMRI data**.

Typical steps include:

1. Upsampling anatomical images to improve segmentation precision  
2. Generating **equivolume cortical layers** within the gray‑matter ribbon  
3. Downsampling layer maps to functional resolution  
4. Intersecting layer maps with region‑of‑interest (ROI) masks  
5. Extracting layer‑specific signals within each ROI  

These analyses allow investigation of **laminar‑specific neural computations**.

---

# Repository Structure

```
neuroimaging-analysis-examples
│
├── preprocessing/
├── registration/
│   ├── template/
│   └── freesurfer/
├── layers/
├── first_level/
├── second_level/
├── multivariate/
│   ├── rsa/
│   ├── searchlight/
│   ├── lss/
│   └── connectivity/
└── group_stats/
```

Each module corresponds to one stage of the analysis pipeline.

---

# Preprocessing

`preprocessing/`

Scripts for preprocessing high‑resolution fMRI data.

Typical operations:

- motion estimation
- run‑to‑run alignment
- anatomical registration
- brain extraction
- reslicing and interpolation

Example outputs:

```
preprocessed_func.nii.gz
motion_parameters.tsv
aligned_run_func.nii.gz
```

The preprocessing workflow aims to **minimize interpolation and preserve spatial resolution**, which is particularly important for ultra‑high‑field imaging.

---

# Registration

`registration/`

Tools for anatomical segmentation and spatial normalization.

### Template Registration

`registration/template/`

High‑precision nonlinear registration using **ANTs**.

Example outputs:

```
T1_to_template_Warped.nii.gz
T1_to_template_0GenericAffine.mat
```

### FreeSurfer Segmentation

`registration/freesurfer/`

Cortical reconstruction and volumetric segmentation using **FreeSurfer**.

Example outputs:

```
aparc+aseg.mgz
ribbon.mgz
lh.white
rh.pial
```

These outputs form the basis for **ROI and layer analyses**.

---

# Layer Analysis

`layers/`

Scripts for extracting and analyzing signals across cortical layers.

Operations include:

- generating equivolume cortical layers
- intersecting layers with ROI masks
- extracting layer‑specific signals
- performing layer‑specific RSA
- comparing signals across layers

Example outputs:

```
layer_signal_roi_vmPFC.csv
layer_signal_roi_dlPFC.csv
layer_rsa_results.csv
```

These analyses allow investigation of **laminar differences in neural processing**.

---

# First‑Level Analysis

`first_level/`

Subject‑level GLM analyses.

Typical steps:

- creation of event/onset files
- design matrix construction
- GLM estimation
- computation of condition contrasts

Example outputs:

```
sub01_contrast_conditionA.nii.gz
sub01_contrast_conditionB.nii.gz
design_matrix.png
```

---

# Second‑Level Analysis

`second_level/`

Group‑level statistical modeling and inference.

Typical analyses:

- combining subject contrast maps
- fixed‑effects or mixed‑effects modeling
- permutation‑based inference

Example outputs:

```
group_stat_map.nii.gz
thresholded_cluster_map.nii.gz
```

---

# Multivariate Analyses

`multivariate/`

Scripts implementing pattern‑based analyses.

### RSA

`multivariate/rsa/`

Computes representational similarity matrices or cross‑validated distance measures.

Example outputs:

```
rsa_matrix.npy
crossnobis_distances.csv
```

### Searchlight

`multivariate/searchlight/`

Voxel‑wise representational analyses.

Example outputs:

```
searchlight_accuracy_map.nii.gz
searchlight_rsa_map.nii.gz
```

### LSS Modeling

`multivariate/lss/`

Least‑Squares‑Separate models for estimating **trial‑level beta maps**.

Example outputs:

```
trial_beta_maps/
beta_trial001.nii.gz
beta_trial002.nii.gz
```

---

# Connectivity Analyses

`multivariate/connectivity/`

Scripts for computing functional connectivity between ROIs.

Example outputs:

```
roi_connectivity_matrix.csv
roi_connectivity_session1.csv
roi_connectivity_session2.csv
```

---

# Group‑Level Statistics

`group_stats/`

Group‑level statistical analyses and visualization.

Example outputs:

```
roi_layer_lme_results.csv
group_summary_plot.png
```

---

# Dependencies

Python packages commonly used:

- numpy
- scipy
- pandas
- nilearn
- scikit‑learn
- matplotlib
- seaborn

External neuroimaging software:

- ANTs
- FreeSurfer
- FSL
- LAYNII

---

# Citation

If you use code from this repository in academic work, please cite the repository.

**Suggested citation:**

Bouyeure, A. (2026). *Neuroimaging Analysis Examples: Workflows for ultra‑high‑field (7T) MRI analysis*.  
GitHub repository:  
https://github.com/AntoineBouyeure/neuroimaging-analysis-examples

BibTeX:

```bibtex
@software{bouyeure_7t_neuroimaging_repo,
  author = {Bouyeure, Antoine},
  title = {Neuroimaging Analysis Examples: Workflows for ultra-high-field MRI analysis},
  year = {2026},
  url = {https://github.com/AntoineBouyeure/neuroimaging-analysis-examples}
}
```

---

# License

This project is distributed under the **MIT License**.
