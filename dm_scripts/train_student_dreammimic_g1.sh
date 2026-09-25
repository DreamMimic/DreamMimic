#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

NUM_ENVS="${1:-1024}"
OUTPUT_DIR="${2:-checkpoints/dreammimic_g1_student_ref_wm}"
DISTILL_MODE="${3:-dagger_rl_studentact}"
HORIZON_LENGTH="${4:-16}"
MINIBATCH_SIZE="${5:-2048}"

case "${DISTILL_MODE}" in
  dagger|dagger_only)
    DISTILL_MODE_CFG="dagger_only"
    ;;
  dagger_rl_studentact)
    DISTILL_MODE_CFG="dagger_rl_studentact"
    ;;
  *)
    echo "Error: DISTILL_MODE must be one of: dagger, dagger_only, dagger_rl_studentact"
    exit 1
    ;;
esac

BASE_TRAIN_CFG="dreammimic/data/cfg/train/rlg/dm_g1_ref_wm.yaml"
TMP_TRAIN_CFG="$(mktemp /tmp/dm_g1_ref_wm.XXXXXX.yaml)"
trap 'rm -f "$TMP_TRAIN_CFG"' EXIT

python - "$BASE_TRAIN_CFG" "$TMP_TRAIN_CFG" "$DISTILL_MODE_CFG" <<'PY'
import sys
import yaml

src, dst, mode = sys.argv[1:4]
with open(src, "r") as f:
    cfg = yaml.safe_load(f)
cfg["params"]["config"]["distillation_mode"] = mode
cfg["params"]["config"]["full_experiment_name"] = f"g1_student_ref_vis_studentwm_{mode}"
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

python dreammimic/run_distill_student_wm_g1.py \
  --task InterMimicG1_Vis_StudentObs_WM \
  --cfg_env dreammimic/data/cfg/dm_g1_ref_train.yaml \
  --cfg_train "${TMP_TRAIN_CFG}" \
  --headless \
  --num_envs "${NUM_ENVS}" \
  --minibatch_size "$(( MINIBATCH_SIZE < NUM_ENVS * HORIZON_LENGTH ? MINIBATCH_SIZE : NUM_ENVS * HORIZON_LENGTH ))" \
  --horizon_length "${HORIZON_LENGTH}" \
  --output "${OUTPUT_DIR}"
