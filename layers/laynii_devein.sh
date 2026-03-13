
#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage:
  $0 <subject_id> <input_func> <layers_file> <output_dir> [method] [alf_file] [column_file]

Methods:
  linear   Simple linear detrending
  CBV      CBV-based normalization (requires ALF and column files)
  leakage  Leakage-model correction (requires ALF and column files)
EOF
}

[[ $# -ge 4 ]] || { usage; exit 1; }
SUBJECT_ID="$1"
FUNC_DATA="$2"
LAYERS_FILE="$3"
OUTPUT_DIR="$4"
METHOD="${5:-linear}"
ALF_FILE="${6:-}"
COLUMN_FILE="${7:-}"

[[ -r "$FUNC_DATA" ]] || { echo "Cannot read functional data: $FUNC_DATA"; exit 1; }
[[ -r "$LAYERS_FILE" ]] || { echo "Cannot read layers file: $LAYERS_FILE"; exit 1; }
command -v LN2_DEVEIN >/dev/null 2>&1 || { echo "LN2_DEVEIN not found"; exit 1; }
command -v fslmaths >/dev/null 2>&1 || { echo "fslmaths not found"; exit 1; }
command -v fslstats >/dev/null 2>&1 || { echo "fslstats not found"; exit 1; }
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

case "$METHOD" in
  linear) cmd=(LN2_DEVEIN -linear -layer_file "$LAYERS_FILE" -input "$FUNC_DATA") ;;
  CBV|leakage)
    [[ -r "$ALF_FILE" ]] || { echo "${METHOD} requires readable ALF file"; exit 1; }
    [[ -r "$COLUMN_FILE" ]] || { echo "${METHOD} requires readable column file"; exit 1; }
    cmd=(LN2_DEVEIN -layer_file "$LAYERS_FILE" -column_file "$COLUMN_FILE" -input "$FUNC_DATA" -ALF "$ALF_FILE")
    [[ "$METHOD" == "CBV" ]] && cmd=(LN2_DEVEIN -CBV -layer_file "$LAYERS_FILE" -column_file "$COLUMN_FILE" -input "$FUNC_DATA" -ALF "$ALF_FILE")
    ;;
  *) echo "Unknown method: $METHOD"; exit 1 ;;
 esac

"${cmd[@]}"

FINAL_OUTPUT=$(find . -maxdepth 1 -type f -name '*devein*.nii.gz' | head -n 1)
[[ -n "$FINAL_OUTPUT" ]] || { echo "No deveined output found"; exit 1; }
OUTPUT_PREFIX="${SUBJECT_ID}_deveined_${METHOD}"

fslmaths "$FUNC_DATA" -Tmean mean_before.nii.gz
fslmaths "$FUNC_DATA" -Tstd std_before.nii.gz
fslmaths mean_before.nii.gz -div std_before.nii.gz "${OUTPUT_PREFIX}_tsnr_before.nii.gz"

fslmaths "$FINAL_OUTPUT" -Tmean mean_after.nii.gz
fslmaths "$FINAL_OUTPUT" -Tstd std_after.nii.gz
fslmaths mean_after.nii.gz -div std_after.nii.gz "${OUTPUT_PREFIX}_tsnr_after.nii.gz"
fslmaths "${OUTPUT_PREFIX}_tsnr_after.nii.gz" -sub "${OUTPUT_PREFIX}_tsnr_before.nii.gz" "${OUTPUT_PREFIX}_tsnr_improvement.nii.gz"

cat > "${OUTPUT_PREFIX}_summary.txt" <<EOF
Subject: ${SUBJECT_ID}
Method: ${METHOD}
Input functional: ${FUNC_DATA}
Layers file: ${LAYERS_FILE}
ALF file: ${ALF_FILE}
Column file: ${COLUMN_FILE}
Main output: ${FINAL_OUTPUT}
EOF
