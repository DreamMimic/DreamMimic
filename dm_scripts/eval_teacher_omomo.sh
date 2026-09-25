#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

DEFAULT_CKPT="ckpts/teacher_omomo/sub8.pth"
LEGACY_CKPT="checkpoints/smplx_teachers/sub8.pth"
CHECKPOINT_PATH="${1:-$DEFAULT_CKPT}"
NUM_ENVS="${2:-1024}"
OUTPUT_DIR="${3:-/tmp/dreammimic_eval/teacher_omomo}"

if [ "$CHECKPOINT_PATH" = "$DEFAULT_CKPT" ]; then
  ensure_default_ckpt "$DEFAULT_CKPT" "$LEGACY_CKPT"
fi
require_ckpt_file "$CHECKPOINT_PATH"
mkdir -p "$OUTPUT_DIR"

python dreammimic/run.py \
  --task InterMimic \
  --cfg_env dreammimic/data/cfg/omomo_test.yaml \
  --cfg_train dreammimic/data/cfg/train/rlg/omomo.yaml \
  --test \
  --headless \
  --checkpoint "${CHECKPOINT_PATH}" \
  --num_envs "${NUM_ENVS}" \
  --output "${OUTPUT_DIR}"
