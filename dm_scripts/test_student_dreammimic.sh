#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

DEFAULT_CKPT="ckpts/student_ref/mimic_best.pth"
LEGACY_CKPT="checkpoints/ref_vis_student_wmdaggerrlx_fullheads_input_multistep_transformer_pcg/smplx_student_ref_vis_studentwm_fullheads_input_multistep_transformer_pcg/nn/mimic_best.pth"
CHECKPOINT_PATH="${1:-$DEFAULT_CKPT}"
NUM_ENVS="${2:-4}"
OUTPUT_DIR="${3:-/tmp/dreammimic_test/student_ref}"

if [ "$CHECKPOINT_PATH" = "$DEFAULT_CKPT" ]; then
  ensure_default_ckpt "$DEFAULT_CKPT" "$LEGACY_CKPT"
fi
require_ckpt_file "$CHECKPOINT_PATH"
mkdir -p "$OUTPUT_DIR"

python dreammimic/run_distill_student_wm.py \
  --task InterMimic_All_NoRef_Vis_StudentObs_WM \
  --cfg_env dreammimic/data/cfg/dm_ref_test.yaml \
  --cfg_train dreammimic/data/cfg/train/rlg/dm_ref_pcg.yaml \
  --test \
  --num_envs "${NUM_ENVS}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --resume 1 \
  --output "${OUTPUT_DIR}"
