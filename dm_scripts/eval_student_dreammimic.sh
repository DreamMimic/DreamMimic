#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

DEFAULT_CKPT="ckpts/student_ref/mimic_bestv4.pth"
LEGACY_CKPT="checkpoints/ref_vis_student_wmdaggerrlx_fullheads_input_multistep_transformer_pcg/smplx_student_ref_vis_studentwm_fullheads_input_multistep_transformer_pcg/nn/mimic_best.pth"
CHECKPOINT_PATH="${1:-$DEFAULT_CKPT}"
NUM_ENVS="${2:-1024}"
OUTPUT_DIR="${3:-/tmp/dreammimic_eval/student_ref}"
RECORD_VIDEO="${4:-false}"
RECORD_ENV_IDX="${5:-0}"
RECORD_DIR="${6:-${OUTPUT_DIR}/record_video}"
RECORD_EVERY="${7:-1}"
RECORD_MAX_FRAMES="${8:-300}"
RECORD_FPS="${9:-30}"

if [ "${RECORD_VIDEO}" != "true" ] && [ "${RECORD_VIDEO}" != "false" ]; then
  echo "Error: RECORD_VIDEO must be true or false"
  exit 1
fi

if [ "$CHECKPOINT_PATH" = "$DEFAULT_CKPT" ]; then
  ensure_default_ckpt "$DEFAULT_CKPT" "$LEGACY_CKPT"
fi
require_ckpt_file "$CHECKPOINT_PATH"
mkdir -p "$OUTPUT_DIR"

RECORD_ARGS=()
if [ "${RECORD_VIDEO}" = "true" ]; then
  RECORD_ARGS=(
    --record_video
    --record_env_idx "${RECORD_ENV_IDX}"
    --record_dir "${RECORD_DIR}"
    --record_every "${RECORD_EVERY}"
    --record_max_frames "${RECORD_MAX_FRAMES}"
    --record_fps "${RECORD_FPS}"
  )
fi

python dreammimic/run_distill_student_wm.py \
  --task InterMimic_All_NoRef_Vis_StudentObs_WM \
  --cfg_env dreammimic/data/cfg/dm_ref_eval.yaml \
  --cfg_train dreammimic/data/cfg/train/rlg/dm_ref_pcg.yaml \
  --test \
  --headless \
  --num_envs "${NUM_ENVS}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --resume 1 \
  --output "${OUTPUT_DIR}" \
  "${RECORD_ARGS[@]}"
