
#!/usr/bin/env bash
#SBATCH --job-name=ants_t1_template
#SBATCH --output=logs/ants_template_%x_%j.out
#SBATCH --error=logs/ants_template_%x_%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: sbatch $0 <config.sh> <subject_id>"
  exit 1
fi

CONFIG="$1"
SUBJECT_ID="$2"
source "$CONFIG"

mkdir -p logs

if [[ -n "${ANTS_MODULE_CMD:-}" ]]; then
  eval "$ANTS_MODULE_CMD"
fi
command -v antsRegistrationSyN.sh >/dev/null 2>&1 || { echo "antsRegistrationSyN.sh not found"; exit 1; }

export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${THREADS:-$((SLURM_CPUS_PER_TASK-1))}"

T1_NATIVE="${ANATOMY_ROOT}/${SUBJECT_ID}/UNI_MPRAGEised.nii.gz"
SUBJECT_OUTPUT_DIR="${OUTPUT_ROOT}/${SUBJECT_ID}"
OUTPUT_PREFIX="${SUBJECT_OUTPUT_DIR}/UNI_MPRAGEised_MNI_"

mkdir -p "$SUBJECT_OUTPUT_DIR"
[[ -f "$T1_NATIVE" ]] || { echo "Missing input: $T1_NATIVE"; exit 1; }
[[ -f "$TEMPLATE_IMAGE" ]] || { echo "Missing template: $TEMPLATE_IMAGE"; exit 1; }
[[ -f "$TEMPLATE_MASK" ]] || { echo "Missing template mask: $TEMPLATE_MASK"; exit 1; }

antsRegistrationSyN.sh   -d 3   -f "$TEMPLATE_IMAGE"   -m "$T1_NATIVE"   -o "$OUTPUT_PREFIX"   -t "${REGISTRATION_TYPE:-s}"   -p "${PRECISION_TYPE:-d}"   -x "$TEMPLATE_MASK"

echo "Finished template registration for ${SUBJECT_ID}"
