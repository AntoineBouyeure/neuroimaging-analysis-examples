#!/usr/bin/env bash
set -euo pipefail

# 7T / UHF fMRI preprocessing orchestrator
#
# This wrapper keeps the original low-level preprocessing scripts untouched and
# replaces the old subject-specific pipeline text/Python file with a reusable
# shell entry point driven by a config file + run manifest.
#
# Expected helper scripts in the same directory (or provided via config):
#   sk_ants_Realign_Estimate.sh
#   sk_ants_Realign_Reslice.sh
#   sk_antsFineReg.sh
#   sk_ants_fsl_BBReg.sh
#
# The run manifest must be a TSV with columns:
#   day    run    func    opp    session_ref
# where session_ref is 1 for the reference run of that session, 0 otherwise.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") --config config.sh --manifest runs.tsv [--run-mprageise]

Required:
  --config    Shell config file defining subject/anatomy paths and options
  --manifest  TSV file listing runs (day, run, func, opp, session_ref)

Optional:
  --run-mprageise   Run MATLAB MPRAGEise before BBR
  --dry-run         Print commands without executing
  -h, --help        Show this help

Notes:
  - The first session/day reference and the study master reference are defined in
    the config file.
  - The existing helper scripts are not modified.
  - All subject-specific paths are externalized into the config + manifest.
USAGE
}

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

run_cmd() {
  log "$*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    eval "$@"
  fi
}

abspath() {
  python3 - <<'PY' "$1"
import os,sys
print(os.path.abspath(sys.argv[1]))
PY
}

base_noext() {
  local x
  x="$(basename "$1")"
  x="${x%.nii.gz}"
  x="${x%.nii}"
  x="${x%.txt}"
  x="${x%.mat}"
  echo "$x"
}

require_file() {
  [[ -f "$1" ]] || die "File not found: $1"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

CONFIG=""
MANIFEST=""
RUN_MPRAGEISE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --run-mprageise) RUN_MPRAGEISE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[[ -n "$CONFIG" ]] || die "--config is required"
[[ -n "$MANIFEST" ]] || die "--manifest is required"

CONFIG="$(abspath "$CONFIG")"
MANIFEST="$(abspath "$MANIFEST")"
require_file "$CONFIG"
require_file "$MANIFEST"

# shellcheck disable=SC1090
source "$CONFIG"

: "${WORK_DIR:?Set WORK_DIR in config}"
: "${UNI_FILE:?Set UNI_FILE in config}"
: "${INV2_FILE:?Set INV2_FILE in config}"
: "${MASTER_REF_DAY:?Set MASTER_REF_DAY in config}"
: "${MASTER_REF_RUN:?Set MASTER_REF_RUN in config}"
: "${TR:?Set TR in config}"

THREADS="${THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"
TRIM_VOLS="${TRIM_VOLS:-0}"
METHOD="${METHOD:-bbrreg}"
HELPER_DIR="${HELPER_DIR:-$SCRIPT_DIR}"
MPRAGEISE_DIR="${MPRAGEISE_DIR:-}"
MATLAB_BIN="${MATLAB_BIN:-matlab}"
CUSTOM_MASK="${CUSTOM_MASK:-}"
CUSTOM_REG="${CUSTOM_REG:-}"

REALIGN_ESTIMATE_SH="${REALIGN_ESTIMATE_SH:-$HELPER_DIR/sk_ants_Realign_Estimate.sh}"
REALIGN_RESLICE_SH="${REALIGN_RESLICE_SH:-$HELPER_DIR/sk_ants_Realign_Reslice.sh}"
FINE_REG_SH="${FINE_REG_SH:-$HELPER_DIR/sk_antsFineReg.sh}"
BBR_SH="${BBR_SH:-$HELPER_DIR/sk_ants_fsl_BBReg.sh}"

require_file "$UNI_FILE"
require_file "$INV2_FILE"
require_file "$REALIGN_ESTIMATE_SH"
require_file "$REALIGN_RESLICE_SH"
require_file "$FINE_REG_SH"
require_file "$BBR_SH"

mkdir -p "$WORK_DIR"
CMD_LOG="$WORK_DIR/CMD.txt"
STATE_TSV="$WORK_DIR/preproc_state.tsv"
: > "$CMD_LOG"
: > "$STATE_TSV"

# Associative arrays describing the dataset
# Keys are DAY::RUN

declare -A FUNC_PATH OPP_PATH SESSION_REF

declare -A FIXED_IM FIXED_MASK TEMP0 DIST_AFF

declare -A INTERRUN_MAT INTERRUN_WARP INTERRUN_1WARP

declare -A ANAT_ITK ANAT_WARP

declare -A FUNC_ANAT FUNC_EPI

declare -A DAY_REF_RUN

trim_if_needed() {
  local input="$1"
  local vols="$2"
  local out
  if [[ "$vols" -le 0 ]]; then
    echo "$input"
    return 0
  fi
  out="$(dirname "$input")/$(base_noext "$input")_${vols}vols.nii.gz"
  if [[ ! -f "$out" ]]; then
    run_cmd "fslroi '$input' '$out' 0 '$vols'"
  fi
  echo "$out"
}

read_manifest() {
  local line day run func opp session_ref key
  while IFS=$'\t' read -r day run func opp session_ref; do
    [[ -z "${day:-}" ]] && continue
    [[ "$day" =~ ^# ]] && continue
    [[ "$day" == "day" ]] && continue
    require_file "$func"
    require_file "$opp"
    func="$(trim_if_needed "$(abspath "$func")" "$TRIM_VOLS")"
    opp="$(abspath "$opp")"
    key="$day::$run"
    FUNC_PATH["$key"]="$func"
    OPP_PATH["$key"]="$opp"
    SESSION_REF["$key"]="$session_ref"
    if [[ "$session_ref" == "1" ]]; then
      DAY_REF_RUN["$day"]="$run"
    fi
  done < "$MANIFEST"
}

get_single_match() {
  local pattern="$1"
  shopt -s nullglob
  local matches=( $pattern )
  shopt -u nullglob
  [[ ${#matches[@]} -gt 0 ]] || die "No file matched: $pattern"
  echo "${matches[0]}"
}

run_realign_estimate() {
  local day="$1" run="$2" key="$day::$run"
  local func="${FUNC_PATH[$key]}"
  local opp="${OPP_PATH[$key]}"
  local args
  args=("'$REALIGN_ESTIMATE_SH'" -n "$THREADS" -t "$TR" -a "'$func'" -b "'$opp'")

  if [[ -n "${FIXED_IM[$key]:-}" && -n "${FIXED_MASK[$key]:-}" ]]; then
    args+=(-f "'${FIXED_IM[$key]}'" -x "'${FIXED_MASK[$key]}'")
  fi
  if [[ "${SESSION_REF[$key]}" == "1" && -z "${FIXED_IM[$key]:-}" ]]; then
    args+=(-s 1)
  fi

  run_cmd "${args[*]}" | tee -a "$CMD_LOG" >/dev/null || true

  local fbase
  fbase="$(base_noext "$func")"
  if [[ "${SESSION_REF[$key]}" == "1" && -z "${FIXED_IM[$key]:-}" ]]; then
    FIXED_IM["$key"]="$(get_single_match "$WORK_DIR/${fbase}*_fixed.nii.gz")"
    if [[ -n "$CUSTOM_MASK" ]]; then
      FIXED_MASK["$key"]="$(abspath "$CUSTOM_MASK")"
    else
      FIXED_MASK["$key"]="$(get_single_match "$WORK_DIR/${fbase}*fixedMask.nii.gz")"
    fi
  else
    TEMP0["$key"]="$(get_single_match "$WORK_DIR/${fbase}*DistCorr_template0.nii.gz")"
    DIST_AFF["$key"]="$(get_single_match "$WORK_DIR/${fbase}*DistCorr_00GenericAffine.mat")"
  fi
}

run_session_reference_workflow() {
  local day="$1"
  local run="${DAY_REF_RUN[$day]:-}"
  local key="$day::$run"
  [[ -n "$run" ]] || die "No session reference run marked for day: $day"

  log "Session reference: $key"
  run_realign_estimate "$day" "$run"
  run_realign_estimate "$day" "$run"

  # after second pass, temp0/distortion transform should exist
  local fbase
  fbase="$(base_noext "${FUNC_PATH[$key]}")"
  TEMP0["$key"]="$(get_single_match "$WORK_DIR/${fbase}*DistCorr_template0.nii.gz")"
  DIST_AFF["$key"]="$(get_single_match "$WORK_DIR/${fbase}*DistCorr_00GenericAffine.mat")"

  # propagate fixed image + mask to other runs of same day
  local other_key
  for other_key in "${!FUNC_PATH[@]}"; do
    [[ "$other_key" == "$key" ]] && continue
    [[ "${other_key%%::*}" != "$day" ]] && continue
    FIXED_IM["$other_key"]="${FIXED_IM[$key]}"
    FIXED_MASK["$other_key"]="${FIXED_MASK[$key]}"
    run_realign_estimate "${other_key%%::*}" "${other_key##*::}"
  done
}

run_interrun_registration() {
  local mov_day="$1" mov_run="$2" ref_day="$3" ref_run="$4" use_syn="${5:-0}"
  local mov_key="$mov_day::$mov_run"
  local ref_key="$ref_day::$ref_run"
  local mov_temp="${TEMP0[$mov_key]:-}"
  local ref_temp="${TEMP0[$ref_key]:-}"
  [[ -n "$mov_temp" ]] || die "Missing moving temp0 for $mov_key"
  [[ -n "$ref_temp" ]] || die "Missing reference temp0 for $ref_key"

  local addflags=""
  if [[ -n "${FIXED_MASK[$mov_key]:-}" && -n "${FIXED_MASK[$ref_key]:-}" ]]; then
    addflags="-g '${FIXED_MASK[$ref_key]}' -n '${FIXED_MASK[$mov_key]}' -x 1"
  fi

  local synflag=""
  [[ "$use_syn" == "1" ]] && synflag="-s 1"

  run_cmd "'$FINE_REG_SH' -f '$ref_temp' -m '$mov_temp' $synflag $addflags" | tee -a "$CMD_LOG" >/dev/null || true

  local fbase
  fbase="$(base_noext "${FUNC_PATH[$mov_key]}")"
  INTERRUN_MAT["$mov_key"]="$(get_single_match "$WORK_DIR/${fbase}*antsFineReg*GenericAffine.mat")"
  INTERRUN_WARP["$mov_key"]="$(get_single_match "$WORK_DIR/${fbase}*antsFineReg_Warped.nii.gz")"
  shopt -s nullglob
  local warp_matches=( "$WORK_DIR/${fbase}"*antsFineReg_1Warp.nii.gz )
  shopt -u nullglob
  if [[ ${#warp_matches[@]} -gt 0 ]]; then
    INTERRUN_1WARP["$mov_key"]="${warp_matches[0]}"
  fi
}

run_mprageise() {
  [[ "$RUN_MPRAGEISE" == "1" ]] || return 0
  [[ -n "$MPRAGEISE_DIR" ]] || die "MPRAGEISE_DIR must be set in config when using --run-mprageise"
  require_cmd "$MATLAB_BIN"

  local matlab_cmd
  matlab_cmd="addpath('$MPRAGEISE_DIR'); [mprageised_im,wmseg_im,collected]=MPRAGEise('$UNI_FILE','$INV2_FILE','$WORK_DIR'); fid=fopen(fullfile('$WORK_DIR','mprageise_outputs.txt'),'w'); fprintf(fid,'%s\n%s\n', char(mprageised_im), char(wmseg_im)); fclose(fid); exit;"
  run_cmd "$MATLAB_BIN -batch \"$matlab_cmd\"" | tee -a "$CMD_LOG" >/dev/null || true
}

load_mprageise_outputs() {
  local f="$WORK_DIR/mprageise_outputs.txt"
  require_file "$f"
  MPRAGEISED_IM="$(sed -n '1p' "$f")"
  WMSEG_IM="$(sed -n '2p' "$f")"
  require_file "$MPRAGEISED_IM"
  require_file "$WMSEG_IM"
}

run_bbr_for_master_reference() {
  local ref_key="$MASTER_REF_DAY::$MASTER_REF_RUN"
  local temp0="${TEMP0[$ref_key]:-}"
  [[ -n "$temp0" ]] || die "Missing master reference temp0 for $ref_key"

  case "$METHOD" in
    bbrreg)
      run_cmd "'$BBR_SH' -f '$MPRAGEISED_IM' -m '$temp0' -s '$WMSEG_IM'" | tee -a "$CMD_LOG" >/dev/null || true
      local fbase
      fbase="$(base_noext "$temp0")"
      ANAT_ITK["$ref_key"]="$(get_single_match "$WORK_DIR/${fbase}_reg2anat_bbr_itk.mat")"
      ANAT_WARP["$ref_key"]="$(get_single_match "$WORK_DIR/${fbase}_reg2anat_bbr.nii.gz")"
      ;;
    customreg)
      [[ -n "$CUSTOM_REG" ]] || die "METHOD=customreg but CUSTOM_REG is not set"
      ANAT_ITK["$ref_key"]="$(abspath "$CUSTOM_REG")"
      ;;
    *)
      die "Unsupported METHOD: $METHOD"
      ;;
  esac
}

run_reslice_all() {
  local key day run cmd addflags mat
  local master_key="$MASTER_REF_DAY::$MASTER_REF_RUN"
  mat="${ANAT_ITK[$master_key]:-}"
  [[ -n "$mat" ]] || die "Missing anatomy transform for master reference"

  for key in "${!FUNC_PATH[@]}"; do
    day="${key%%::*}"
    run="${key##*::}"
    addflags=""

    if [[ "$day" != "$MASTER_REF_DAY" ]]; then
      local ref_run_this_day="${DAY_REF_RUN[$day]:-}"
      [[ -n "$ref_run_this_day" ]] || die "No session reference for day $day"
      local ref_key_this_day="$day::$ref_run_this_day"
      addflags="-u '${INTERRUN_MAT[$ref_key_this_day]}' -r '${TEMP0[$master_key]}'"
      if [[ -n "${INTERRUN_1WARP[$ref_key_this_day]:-}" ]]; then
        addflags="$addflags -v '${INTERRUN_1WARP[$ref_key_this_day]}'"
      fi
    fi

    cmd="'$REALIGN_RESLICE_SH' -x '$mat' -f '$MPRAGEISED_IM' -t '$TR' -n '$THREADS' -a '${FUNC_PATH[$key]}' $addflags"
    run_cmd "$cmd" | tee -a "$CMD_LOG" >/dev/null || true

    local fbase
    fbase="$(base_noext "${FUNC_PATH[$key]}")"
    FUNC_ANAT["$key"]="$(get_single_match "$WORK_DIR/${fbase}*_anatomySpaceAligned.nii.gz")"
    FUNC_EPI["$key"]="$(get_single_match "$WORK_DIR/${fbase}*_nativeEPISpace*Aligned.nii.gz")"
  done
}

write_state() {
  {
    printf 'day\trun\tfunc\topp\tfixed_im\tfixed_mask\ttemp0\tinterrun_mat\tanat_itk\tfunc_anat\tfunc_epi\n'
    local key
    for key in "${!FUNC_PATH[@]}"; do
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${key%%::*}" "${key##*::}" \
        "${FUNC_PATH[$key]:-}" "${OPP_PATH[$key]:-}" \
        "${FIXED_IM[$key]:-}" "${FIXED_MASK[$key]:-}" \
        "${TEMP0[$key]:-}" "${INTERRUN_MAT[$key]:-}" \
        "${ANAT_ITK[$key]:-${ANAT_ITK[$MASTER_REF_DAY::$MASTER_REF_RUN]:-}}" \
        "${FUNC_ANAT[$key]:-}" "${FUNC_EPI[$key]:-}"
    done | sort
  } > "$STATE_TSV"
  log "Wrote state table: $STATE_TSV"
}

main() {
  read_manifest

  [[ -n "${DAY_REF_RUN[$MASTER_REF_DAY]:-}" ]] || die "MASTER_REF_DAY has no session_ref=1 run in manifest"
  [[ "${DAY_REF_RUN[$MASTER_REF_DAY]}" == "$MASTER_REF_RUN" ]] || die "MASTER_REF_RUN must be the session reference for MASTER_REF_DAY"

  # 1) per-session estimation
  local day
  for day in "${!DAY_REF_RUN[@]}"; do
    run_session_reference_workflow "$day"
  done

  # 2) inter-session registration: align each non-master session reference to master reference
  for day in "${!DAY_REF_RUN[@]}"; do
    [[ "$day" == "$MASTER_REF_DAY" ]] && continue
    run_interrun_registration "$day" "${DAY_REF_RUN[$day]}" "$MASTER_REF_DAY" "$MASTER_REF_RUN" 1
  done

  # 3) anatomy preparation and BBR on master reference
  run_mprageise
  load_mprageise_outputs
  run_bbr_for_master_reference

  # 4) final one-shot reslicing for all runs
  run_reslice_all

  write_state
  log "Preprocessing orchestration completed."
}

main "$@"
