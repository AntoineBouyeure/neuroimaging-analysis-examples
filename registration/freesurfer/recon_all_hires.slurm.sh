
#!/usr/bin/env bash
#SBATCH --job-name=freesurfer_recon
#SBATCH --output=logs/freesurfer_%x_%j.out
#SBATCH --error=logs/freesurfer_%x_%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=30:00:00

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: sbatch $0 <config.sh> <subject_id> [subject_id ...]"
  exit 1
fi

CONFIG="$1"
shift
source "$CONFIG"
mkdir -p logs "$SUBJECTS_DIR"

if [[ -n "${FREESURFER_SETUP_CMD:-}" ]]; then
  eval "$FREESURFER_SETUP_CMD"
fi
export FREESURFER_HOME="${FREESURFER_HOME:-$(dirname "$(dirname "$(command -v recon-all)")")}" || true
if [[ -n "${FREESURFER_HOME:-}" && -f "${FREESURFER_HOME}/SetUpFreeSurfer.sh" ]]; then
  source "${FREESURFER_HOME}/SetUpFreeSurfer.sh"
fi
export SUBJECTS_DIR

for sid in "$@"; do
  input_file="${ANATOMY_ROOT}/${sid}/UNI_MPRAGEised.nii.gz"
  [[ -f "$input_file" ]] || { echo "Missing input for ${sid}: ${input_file}"; continue; }
  recon-all -subjid "$sid" -all -hires ${EXPERT_OPTS:+-expert "$EXPERT_OPTS"} -i "$input_file"
done
