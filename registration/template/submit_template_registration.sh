
#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <config.sh> <subject1> [subject2 ...]"
  exit 1
fi

CONFIG="$1"
shift
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/register_t1_to_template.slurm.sh"

for sid in "$@"; do
  sbatch --job-name="antsTpl_${sid}" "$JOB_SCRIPT" "$CONFIG" "$sid"
done
