#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/dreammimic:${PYTHONPATH:-}"
if [ -n "${CONDA_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

if [ "${DREAMMIMIC_DEBUG_CUDA:-0}" = "1" ]; then
  export CUDA_LAUNCH_BLOCKING=1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64,garbage_collection_threshold:0.9}"

CKPT_ROOT="$REPO_ROOT/ckpts"
mkdir -p "$CKPT_ROOT"

repo_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf "%s\n" "$path"
  else
    printf "%s\n" "$REPO_ROOT/$path"
  fi
}

ensure_default_ckpt() {
  local default_ckpt="$1"
  local legacy_ckpt="$2"

  local default_abs
  local legacy_abs
  default_abs="$(repo_path "$default_ckpt")"
  legacy_abs="$(repo_path "$legacy_ckpt")"

  if [ -f "$default_abs" ]; then
    return 0
  fi

  if [ -f "$legacy_abs" ]; then
    mkdir -p "$(dirname "$default_abs")"
    cp "$legacy_abs" "$default_abs"
    echo "[DreamMimic] Bootstrapped checkpoint: $default_ckpt (copied from $legacy_ckpt)"
    return 0
  fi

  echo "[DreamMimic] ERROR: Missing default checkpoint."
  echo "  expected: $default_ckpt"
  echo "  fallback: $legacy_ckpt"
  echo "Please place the checkpoint under ckpts/ or pass an explicit path."
  exit 1
}

require_ckpt_file() {
  local ckpt_path="$1"
  local ckpt_abs
  ckpt_abs="$(repo_path "$ckpt_path")"

  if [ ! -f "$ckpt_abs" ]; then
    echo "[DreamMimic] ERROR: Checkpoint not found: $ckpt_path"
    exit 1
  fi
}
