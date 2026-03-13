# Example config for run_7t_preprocessing.sh

# Output / working directory for copied files and derived products
WORK_DIR="/path/to/derivatives/sub-01/preprocessing"

# Anatomy inputs
UNI_FILE="/path/to/sub-01/anat/UNI.nii.gz"
INV2_FILE="/path/to/sub-01/anat/INV2.nii.gz"

# Study-level master reference (must match a session_ref=1 run in the manifest)
MASTER_REF_DAY="day1"
MASTER_REF_RUN="run1"

# Acquisition / processing options
TR="3.1"
THREADS="24"
TRIM_VOLS="0"            # set >0 only for quick testing
METHOD="bbrreg"          # bbrreg or customreg
CUSTOM_REG=""            # required if METHOD=customreg
CUSTOM_MASK=""           # optional manual mask for session reference

# Helper scripts directory (existing scripts left untouched)
HELPER_DIR="/path/to/preprocessing"
REALIGN_ESTIMATE_SH="$HELPER_DIR/sk_ants_Realign_Estimate.sh"
REALIGN_RESLICE_SH="$HELPER_DIR/sk_ants_Realign_Reslice.sh"
FINE_REG_SH="$HELPER_DIR/sk_antsFineReg.sh"
BBR_SH="$HELPER_DIR/sk_ants_fsl_BBReg.sh"

# MATLAB / MPRAGEise (required if running with --run-mprageise)
MATLAB_BIN="matlab"
MPRAGEISE_DIR="/path/to/matlab_mprageise_scripts"
