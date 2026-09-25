#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

NUM_ENVS="${1:-1024}"
OUTPUT_DIR="${2:-checkpoints/teacher_omomo}"

python dreammimic/run.py \
  --task InterMimic \
  --cfg_env dreammimic/data/cfg/omomo_train.yaml \
  --cfg_train dreammimic/data/cfg/train/rlg/omomo.yaml \
  --headless \
  --num_envs "${NUM_ENVS}" \
  --output "${OUTPUT_DIR}"
