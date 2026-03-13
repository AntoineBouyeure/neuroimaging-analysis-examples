
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=fat_cpu
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --job-name=searchlight
#SBATCH --exclusive
#SBATCH --output=logs/searchlight_%j.out
#SBATCH --error=logs/searchlight_%j.err

set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: sbatch $0 <subject_id> <rsa_dir> <anat_root> <func_root> <output_root> <mask_csv>"
  exit 1
fi

SUBJECT_ID="$1"
RSA_DIR="$2"
ANAT_ROOT="$3"
FUNC_ROOT="$4"
OUTPUT_ROOT="$5"
MASK_CSV="$6"

module purge
module load intel-oneapi-mpi/2021.12.1
module load python/3.11.7
mkdir -p logs

TMP_BASE="/tmp/searchlight_${SUBJECT_ID}_${SLURM_JOB_ID:-local}"
TMP_RSA="${TMP_BASE}/rsa"
TMP_ANAT="${TMP_BASE}/anat"
TMP_FUNC="${TMP_BASE}/func"
TMP_OUTPUT="${TMP_BASE}/output"
TMP_CSV="${TMP_BASE}/csv"

srun --ntasks-per-node=1 --nodes=${SLURM_JOB_NUM_NODES} mkdir -p "$TMP_RSA" "$TMP_ANAT" "$TMP_FUNC" "$TMP_OUTPUT" "$TMP_CSV"
srun --ntasks-per-node=1 --nodes=${SLURM_JOB_NUM_NODES} bash -c "mkdir -p ${TMP_RSA}/fear_acq/LSS_maps/concatenated && rsync -a --include='*/' --include='${SUBJECT_ID}_lssmap.nii.gz' --exclude='*' ${RSA_DIR}/fear_acq/LSS_maps/concatenated/ ${TMP_RSA}/fear_acq/LSS_maps/concatenated/"
srun --ntasks-per-node=1 --nodes=${SLURM_JOB_NUM_NODES} bash -c "mkdir -p ${TMP_ANAT}/${SUBJECT_ID} ${TMP_FUNC}/${SUBJECT_ID} && rsync -a --include='*/' --include='gm_cropped.nii.gz' --exclude='*' ${ANAT_ROOT}/${SUBJECT_ID}/ ${TMP_ANAT}/${SUBJECT_ID}/ && rsync -a --include='*/' --include='func_mean_anatomySpace.nii.gz' --exclude='*' ${FUNC_ROOT}/${SUBJECT_ID}/ ${TMP_FUNC}/${SUBJECT_ID}/ && cp ${MASK_CSV} ${TMP_CSV}/mask_sizes.csv"

srun --mpi=pmi2 python3 searchlight_between_item_mpi.py "$SUBJECT_ID" "$TMP_RSA" "$TMP_ANAT" "$TMP_FUNC" "$TMP_OUTPUT" "${TMP_CSV}/mask_sizes.csv"
mkdir -p "$OUTPUT_ROOT"
rsync -a "$TMP_OUTPUT/" "$OUTPUT_ROOT/"
