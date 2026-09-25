#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

NUM_ENVS="${1:-16}"

python dreammimic/run.py \
  --task InterMimic \
  --cfg_env dreammimic/data/cfg/omomo_test.yaml \
  --cfg_train dreammimic/data/cfg/train/rlg/omomo.yaml \
  --test \
  --play_dataset \
  --num_envs "${NUM_ENVS}"
