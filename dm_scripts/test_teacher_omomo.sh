#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

DEFAULT_CKPT="ckpts/teacher_omomo/sub2.pth"
LEGACY_CKPT="checkpoints/smplx_teachers_new/sub2.pth"
CHECKPOINT_PATH="${1:-$DEFAULT_CKPT}"
NUM_ENVS="${2:-16}"

if [ "$CHECKPOINT_PATH" = "$DEFAULT_CKPT" ]; then
  ensure_default_ckpt "$DEFAULT_CKPT" "$LEGACY_CKPT"
fi
require_ckpt_file "$CHECKPOINT_PATH"

python dreammimic/run.py \
  --task InterMimic \
  --cfg_env dreammimic/data/cfg/omomo_test_new.yaml \
  --cfg_train dreammimic/data/cfg/train/rlg/omomo.yaml \
  --test \
  --checkpoint "${CHECKPOINT_PATH}" \
  --num_envs "${NUM_ENVS}"
