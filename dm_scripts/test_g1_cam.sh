#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

DEFAULT_CKPT="ckpts/teacher_g1/sub8.pth"
LEGACY_CKPT="checkpoints/g1/sub8.pth"
CHECKPOINT_PATH="${1:-$DEFAULT_CKPT}"
NUM_ENVS="${2:-4}"

if [ "$CHECKPOINT_PATH" = "$DEFAULT_CKPT" ]; then
  if [ -f "$REPO_ROOT/$LEGACY_CKPT" ]; then
    ensure_default_ckpt "$DEFAULT_CKPT" "$LEGACY_CKPT"
  elif [ -f "$REPO_ROOT/checkpoints/checkpoints/g1/sub8.pth" ]; then
    ensure_default_ckpt "$DEFAULT_CKPT" "checkpoints/checkpoints/g1/sub8.pth"
  fi
fi
require_ckpt_file "$CHECKPOINT_PATH"

python dreammimic/run.py \
  --task InterMimicG1 \
  --cfg_env dreammimic/data/cfg/omomo_g1_29dof_with_hand_cam.yaml \
  --cfg_train dreammimic/data/cfg/train/rlg/omomo_g1_29dof_with_hand.yaml \
  --checkpoint "${CHECKPOINT_PATH}" \
  --test \
  --num_envs "${NUM_ENVS}"
