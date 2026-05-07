#!/usr/bin/env bash
# Launch Qwen3.6-27B-MTP. Designed to be invoked directly by systemd or by
# hand. All paths come from environment variables (see ~/.config/qwen36-mtp/env
# for the canonical config; the installer writes that file).
#
# Required env:
#   LLAMA_BIN        Absolute path to llama-server binary
#   MODEL_PATH       Absolute path to the GGUF
# Optional env (with defaults):
#   CTX_SIZE         196608
#   PORT             8080
#   HOST             0.0.0.0
#   ALIAS            qwen3.6-27b
#   SLOT_CACHE_DIR   <empty = disabled on-disk slot save/restore>
#   CACHE_REUSE      256
#   SPEC_DRAFT_N_MAX 4
#   PARALLEL         1   (NOTE: the patched MTP fork rejects PARALLEL>1 when
#                        --spec-type mtp is set; to run multi-slot you must
#                        also set DISABLE_MTP=1 and accept ~30-50% lower
#                        single-stream tok/s)
#   DISABLE_MTP      <empty = MTP enabled>
#   SPS              0.5 (only used when PARALLEL>1)
#
# Forward any extra CLI args to llama-server via "$@".

set -e

: "${LLAMA_BIN:?LLAMA_BIN env var is required}"
: "${MODEL_PATH:?MODEL_PATH env var is required}"

CTX_SIZE="${CTX_SIZE:-196608}"
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
ALIAS="${ALIAS:-qwen3.6-27b}"
CACHE_REUSE="${CACHE_REUSE:-256}"
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-4}"
PARALLEL="${PARALLEL:-1}"
SPS="${SPS:-0.5}"

slot_args=()
if [[ -n "${SLOT_CACHE_DIR:-}" ]]; then
  mkdir -p "$SLOT_CACHE_DIR"
  slot_args=(--slot-save-path "$SLOT_CACHE_DIR")
fi

# MTP and multi-slot are mutually exclusive in the crucible-mtp fork.
spec_args=(--spec-type mtp --spec-draft-n-max "$SPEC_DRAFT_N_MAX")
parallel_args=(--parallel 1)
if [[ "$PARALLEL" -gt 1 || -n "${DISABLE_MTP:-}" ]]; then
  spec_args=()
  parallel_args=(--parallel "$PARALLEL" --kv-unified --slot-prompt-similarity "$SPS")
fi

exec "$LLAMA_BIN" \
  -m "$MODEL_PATH" \
  --alias "$ALIAS" \
  -ngl 999 -fa on \
  "${spec_args[@]}" \
  --no-mmap \
  --ctx-size "$CTX_SIZE" \
  --batch-size 1024 --ubatch-size 512 \
  -ctk q4_0 -ctv q4_0 \
  "${parallel_args[@]}" \
  --ctx-checkpoints 8 --checkpoint-every-n-tokens 2048 \
  --cache-ram -1 --cache-idle-slots \
  --cache-reuse "$CACHE_REUSE" \
  "${slot_args[@]}" \
  --reasoning-format deepseek \
  --metrics --jinja \
  --host "$HOST" --port "$PORT" \
  "$@"
