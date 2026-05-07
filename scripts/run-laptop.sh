#!/usr/bin/env bash
# Launch Qwen3.6-35B-A3B (MoE, 40 layers, 3B active) on a single
# RTX 5070 Ti laptop (12 GB VRAM, sm_120 / Blackwell). Designed to be
# invoked directly by systemd or by hand. All paths come from environment
# variables (see ~/.config/qwen36-mtp/laptop.env which install-laptop.sh
# writes).
#
# Required env:
#   LLAMA_BIN        Absolute path to llama-server binary
#   MODEL_PATH       Absolute path to the GGUF
# Optional env (with defaults):
#   CTX_SIZE         131072        (128 K — Qwen3.6 native is 262144 if you have RAM)
#   PORT             8080
#   HOST             127.0.0.1
#   ALIAS            qwen3.6-35b-a3b
#   SLOT_CACHE_DIR   <empty = disabled on-disk slot save/restore>
#   CACHE_REUSE      256
#   N_CPU_MOE        40            (move all 40 layers' MoE expert tensors to CPU;
#                                   keep attention/router/embeddings on GPU.
#                                   Lower this if you have spare VRAM.)
#   PARALLEL         1
#
# Forward any extra CLI args to llama-server via "$@".
#
# Why these flags:
#   -ngl 999          offload as many layers as possible (attention/router/etc)
#   --n-cpu-moe N     keep MoE expert tensors of the first N layers on CPU.
#                     This is the key trick for 12 GB cards: on Qwen3.6-35B-A3B
#                     the experts are ~17 GB, attention/router/embeddings are
#                     ~3 GB. Set N to the model's layer count to push *all*
#                     experts to CPU (only 3B params active per token, so RAM
#                     bandwidth — not compute — is the bottleneck).
#   -fa on            flash attention (saves VRAM, faster prefill)
#   -ctk q8_0 -ctv q8_0   Q8_0 KV cache (halves KV size vs fp16, near-lossless)
#   --no-mmap         pre-allocate cleanly on first request
#   --cache-reuse 256 enable longest-common-prefix prefix caching across requests
#   --slot-save-path  persist slot KV to disk so restarts/context switches don't
#                     re-prefill the system prompt (huge win for agentic loops)
#   --jinja           use the GGUF-embedded chat template (Qwen3 tool-calling)
#   --reasoning-format deepseek   parse <think> blocks into reasoning_content

set -e

: "${LLAMA_BIN:?LLAMA_BIN env var is required}"
: "${MODEL_PATH:?MODEL_PATH env var is required}"

CTX_SIZE="${CTX_SIZE:-131072}"
PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"
ALIAS="${ALIAS:-qwen3.6-35b-a3b}"
CACHE_REUSE="${CACHE_REUSE:-256}"
N_CPU_MOE="${N_CPU_MOE:-40}"
PARALLEL="${PARALLEL:-1}"

slot_args=()
if [[ -n "${SLOT_CACHE_DIR:-}" ]]; then
  mkdir -p "$SLOT_CACHE_DIR"
  slot_args=(--slot-save-path "$SLOT_CACHE_DIR")
fi

exec "$LLAMA_BIN" \
  -m "$MODEL_PATH" \
  --alias "$ALIAS" \
  -ngl 999 \
  --n-cpu-moe "$N_CPU_MOE" \
  -fa on \
  --no-mmap \
  --ctx-size "$CTX_SIZE" \
  --batch-size 1024 --ubatch-size 512 \
  -ctk q8_0 -ctv q8_0 \
  --parallel "$PARALLEL" \
  --ctx-checkpoints 8 --checkpoint-every-n-tokens 2048 \
  --cache-ram -1 --cache-idle-slots \
  --cache-reuse "$CACHE_REUSE" \
  "${slot_args[@]}" \
  --reasoning-format deepseek \
  --metrics --jinja \
  --host "$HOST" --port "$PORT" \
  "$@"
