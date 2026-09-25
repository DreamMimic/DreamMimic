#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

NUM_ENVS="${1:-1024}"
OUTPUT_DIR="${2:-checkpoints/dreammimic_student_ref_pcg}"

python dreammimic/run_distill_student_wm.py \
  --task InterMimic_All_NoRef_Vis_StudentObs_WM \
  --cfg_env dreammimic/data/cfg/dm_ref_train.yaml \
  --cfg_train dreammimic/data/cfg/train/rlg/dm_ref_pcg.yaml \
  --headless \
  --num_envs "${NUM_ENVS}" \
  --minibatch_size 4096 \
  --horizon_length 16 \
  --output "${OUTPUT_DIR}"
