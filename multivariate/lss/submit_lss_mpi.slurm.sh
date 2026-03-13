
#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=10:00:00
#SBATCH --job-name=lss_mpi
#SBATCH --exclusive
#SBATCH --output=logs/lss_%j.out
#SBATCH --error=logs/lss_%j.err

set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: sbatch $0 <subject_id> <input_root> <output_root> <mode>"
  exit 1
fi

SUBJECT_ID="$1"
INPUT_ROOT="$2"
OUTPUT_ROOT="$3"
MODE="$4"

module purge
module load intel-oneapi-mpi/2021.12.1
module load python/3.11.7

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LSS_RESET_EVERY="${LSS_RESET_EVERY:-4}"
export LSS_NOISE_MODEL="${LSS_NOISE_MODEL:-ols}"
JOB_ID="${SLURM_JOB_ID:-local}"
export LSS_CACHE="/tmp/nilearn_cache_${JOB_ID}"
mkdir -p logs "$LSS_CACHE"

mpirun -np "${SLURM_NTASKS}" python3 lss_mpi.py "$INPUT_ROOT" "$OUTPUT_ROOT" "$SUBJECT_ID" "$MODE"
