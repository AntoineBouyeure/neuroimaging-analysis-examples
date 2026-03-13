
# Paths
ANATOMY_ROOT="/path/to/anat"
TEMPLATE_IMAGE="/path/to/mni_075mm.nii.gz"
TEMPLATE_MASK="/path/to/mni_075mm_mask.nii.gz"
OUTPUT_ROOT="/path/to/output"

# Optional
ANTS_MODULE_CMD="spack load ants"
THREADS=7
REGISTRATION_TYPE="s"   # passed to antsRegistrationSyN.sh -t
PRECISION_TYPE="d"      # passed to antsRegistrationSyN.sh -p
