#!/usr/bin/env bash
# One-shot installer: Qwen3.6-27B-MTP IQ4_XS on a single RTX 4090 (Linux).
#
# Stack:
#   * upstream llama.cpp (MTP support merged in ggml-org/llama.cpp#22673)
#   * MTP-preserving GGUF (localweights/Qwen3.6-27B-MTP-IQ4_XS-GGUF, ~14 GiB)
#   * conda env qwen36-hf for huggingface-hub
#   * conda env qwen36-build for the CUDA 12.4 toolkit (driver-only systems)
#
# Reference numbers (HF model card, 3090 Ti, same recipe):
#   100.3 tok/s short-ctx decode, 70-73 tok/s mean over 4K..256K, 86.6% MTP accept.
# Measured on a stock RTX 4090 with this script:
#   196 608 ctx -> 141 tok/s decode, 196 tok/s prefill, 4/4 MTP accept, 22.2 GB VRAM.
#   262 144 ctx -> 159 tok/s decode, 342 tok/s prefill, 4/4 MTP accept, 23.8 GB VRAM.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------- tunables --------------------------------------------------------
PREFIX="${PREFIX:-$HOME/Dev/qwen36}"
LLAMA_DIR="$PREFIX/llama.cpp"
MODEL_DIR="$PREFIX/models/qwen36-27b-mtp"
MODEL_REPO="localweights/Qwen3.6-27B-MTP-IQ4_XS-GGUF"
MODEL_FILE="Qwen3.6-27B-MTP-IQ4_XS.gguf"
HF_ENV="${HF_ENV:-qwen36-hf}"
BUILD_ENV="${BUILD_ENV:-qwen36-build}"
CUDA_LABEL="${CUDA_LABEL:-cuda-12.4.1}"   # conda nvidia channel label
CUDA_ARCH="${CUDA_ARCH:-89}"               # 89 = Ada / RTX 4090; 86 = Ampere / 3090
CTX_SIZE="${CTX_SIZE:-196608}"             # safe on 24 GB (~22.2 GB used)
PARALLEL="${PARALLEL:-4}"
PORT="${PORT:-8080}"
JOBS="$(nproc)"

LLAMA_UPSTREAM_URL="${LLAMA_UPSTREAM_URL:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_REF="${LLAMA_REF:-master}"
LLAMA_REMOTE_NAME="${LLAMA_REMOTE_NAME:-upstream}"

# ---------- helpers ---------------------------------------------------------
log()  { printf '\033[1;32m[+] %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[!] %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[x] %s\033[0m\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
conda_env_prefix() { conda env list | awk -v name="$1" '$1 == name {print $NF; exit}'; }

# ---------- 0. sanity -------------------------------------------------------
log "Sanity checks"
have git || die "git not found"
have cmake || die "cmake not found"
have nvidia-smi || die "nvidia-smi not found (NVIDIA driver missing?)"
have conda || die "conda not found. Install Miniconda/Mamba first."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
mkdir -p "$PREFIX" "$MODEL_DIR"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/lib/conda-strict.sh"

# ---------- 1. HF download env ---------------------------------------------
log "Conda env for huggingface-hub: $HF_ENV"
conda env list | awk '{print $1}' | grep -qx "$HF_ENV" || conda create -y -n "$HF_ENV" python=3.11
conda_activate "$HF_ENV"
pip install -q --upgrade "huggingface-hub[hf_transfer]"

# ---------- 2. download MTP-preserving GGUF --------------------------------
log "Downloading $MODEL_REPO -> $MODEL_DIR"
if [[ -f "$MODEL_DIR/$MODEL_FILE" ]]; then
  log "  already present, skipping"
else
  HF_HUB_ENABLE_HF_TRANSFER=1 \
    hf download "$MODEL_REPO" "$MODEL_FILE" --local-dir "$MODEL_DIR"
fi
SIZE_BYTES=$(stat -c%s "$MODEL_DIR/$MODEL_FILE")
(( SIZE_BYTES > 10*1024*1024*1024 )) \
  || die "GGUF only $SIZE_BYTES bytes — likely corrupt (don't use aria2c for multi-GB models)"
log "  GGUF OK ($((SIZE_BYTES/1024/1024/1024)) GiB)"

HF_HUB_ENABLE_HF_TRANSFER=1 \
  hf download "$MODEL_REPO" --local-dir "$MODEL_DIR" \
    --include "*.json" --include "tokenizer*" 2>/dev/null || true

conda_deactivate

# ---------- 3. CUDA toolkit env --------------------------------------------
# Why an env? Driver-only Linux installs (Ubuntu 22/24, Debian) have no nvcc on
# PATH, so cmake fails to find CUDAToolkit. Conda's nvidia channel ships a
# self-contained toolkit that builds against any in-range driver.
log "Conda env for CUDA toolkit: $BUILD_ENV ($CUDA_LABEL)"
if ! conda env list | awk '{print $1}' | grep -qx "$BUILD_ENV"; then
  conda create -y -n "$BUILD_ENV" -c "nvidia/label/$CUDA_LABEL" cuda-toolkit
elif [[ ! -x "$(conda_env_prefix "$BUILD_ENV")/bin/nvcc" ]]; then
  log "  nvcc missing; installing cuda-toolkit into existing env"
  conda install -y -n "$BUILD_ENV" -c "nvidia/label/$CUDA_LABEL" cuda-toolkit
fi
conda_activate "$BUILD_ENV"
NVCC="$CONDA_PREFIX/bin/nvcc"
[[ -x "$NVCC" ]] || die "nvcc not found in $BUILD_ENV after installing cuda-toolkit"
"$NVCC" --version | tail -1

# ---------- 4. upstream llama.cpp ------------------------------------------
log "Preparing upstream llama.cpp ($LLAMA_REF)"
if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  git clone "$LLAMA_UPSTREAM_URL" "$LLAMA_DIR"
fi

cd "$LLAMA_DIR"

if ! git diff --quiet || ! git diff --cached --quiet; then
  die "Existing llama.cpp checkout at $LLAMA_DIR has local changes. Commit/stash them or set PREFIX to a fresh path."
fi

# Keep any existing fork remote intact; fetch upstream into a dedicated remote
# and build from a detached upstream ref. This migrates old crucible-mtp clones
# without rewriting their local branches.
if git remote get-url "$LLAMA_REMOTE_NAME" >/dev/null 2>&1; then
  git remote set-url "$LLAMA_REMOTE_NAME" "$LLAMA_UPSTREAM_URL"
else
  git remote add "$LLAMA_REMOTE_NAME" "$LLAMA_UPSTREAM_URL"
fi
git fetch --prune "$LLAMA_REMOTE_NAME"

if git rev-parse --verify --quiet "refs/remotes/$LLAMA_REMOTE_NAME/$LLAMA_REF" >/dev/null; then
  git checkout --detach "$LLAMA_REMOTE_NAME/$LLAMA_REF"
else
  git fetch "$LLAMA_REMOTE_NAME" "$LLAMA_REF"
  git checkout --detach FETCH_HEAD
fi

# Fail early if LLAMA_REF is too old for the upstream-MTP launcher flags.
grep -Rqs '"draft-mtp"' common/ \
  || die "llama.cpp checkout lacks draft-mtp support. Use LLAMA_REF=master or a post-#22673 commit."
grep -Rqs -- '--slot-save-path' common/ tools/ \
  || die "llama.cpp checkout lacks --slot-save-path support. Use a newer LLAMA_REF."
grep -Rqs -- '--checkpoint-min-step' common/ tools/ \
  || die "llama.cpp checkout lacks --checkpoint-min-step support. Use a newer LLAMA_REF."
grep -Rqs -- '--kv-unified' common/ tools/ \
  || die "llama.cpp checkout lacks --kv-unified support. Use a newer LLAMA_REF."
grep -Rqs -- '--cache-idle-slots' common/ tools/ \
  || die "llama.cpp checkout lacks --cache-idle-slots support. Use a newer LLAMA_REF."
grep -Rqs -- '--slot-prompt-similarity' common/ tools/ \
  || die "llama.cpp checkout lacks --slot-prompt-similarity support. Use a newer LLAMA_REF."
log "  using llama.cpp commit $(git rev-parse --short HEAD)"

# ---------- 5. build llama.cpp with CUDA -----------------------------------
# Two non-obvious flags below:
#   * CMAKE_CUDA_RUNTIME_LIBRARY=Shared so we link libcudart.so (not the static
#     archive) — conda's cudart package is shared-only.
#   * -Wl,-rpath,$CONDA_PREFIX/lib so the resulting binary finds libcudart.so.12
#     at runtime without requiring LD_LIBRARY_PATH or activating the build env.
log "Building llama.cpp (CUDA arch $CUDA_ARCH, $JOBS jobs)"
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF \
  -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared \
  -DCUDAToolkit_ROOT="$CONDA_PREFIX" \
  -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,$CONDA_PREFIX/lib -L$CONDA_PREFIX/lib" \
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,$CONDA_PREFIX/lib -L$CONDA_PREFIX/lib"
cmake --build build --config Release -j "$JOBS" \
  --target llama-server llama-cli llama-quantize
[[ -x build/bin/llama-server ]] || die "build failed: build/bin/llama-server missing"
log "  build OK -> $LLAMA_DIR/build/bin/"

conda_deactivate

# ---------- 6. config + launcher (single source of truth: this repo) -------
LAUNCH="$REPO_DIR/scripts/run.sh"
[[ -x "$LAUNCH" ]] || die "missing repo launcher: $LAUNCH (re-clone the repo)"

CONF_DIR="$HOME/.config/qwen36-mtp"
CONF="$CONF_DIR/env"
mkdir -p "$CONF_DIR"
log "Writing config: $CONF"
cat > "$CONF" <<EOF
# Generated by scripts/install.sh — edit and \`systemctl --user restart qwen36\`
LLAMA_BIN=$LLAMA_DIR/build/bin/llama-server
MODEL_PATH=$MODEL_DIR/$MODEL_FILE
CTX_SIZE=$CTX_SIZE
PORT=$PORT
HOST=0.0.0.0
ALIAS=qwen3.6-27b
SLOT_CACHE_DIR=$PREFIX/slot-cache
CACHE_REUSE=256
SPEC_DRAFT_N_MAX=4
CHECKPOINT_MIN_STEP=2048
PARALLEL=$PARALLEL
EOF

# ---------- 7. systemd user unit -------------------------------------------
SVC="$HOME/.config/systemd/user/qwen36.service"
mkdir -p "$(dirname "$SVC")"
cat > "$SVC" <<EOF
[Unit]
Description=Qwen3.6-27B MTP llama-server (RTX 4090, parallel=$PARALLEL)
After=network-online.target
Conflicts=qwen36-multi.service

[Service]
Type=simple
EnvironmentFile=$CONF
Environment=CUDA_VISIBLE_DEVICES=0
Environment=GGML_CUDA_ENABLE_UNIFIED_MEMORY=0
ExecStart=$LAUNCH
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
log "Wrote systemd user unit: $SVC"

# qwen36-multi.service is deprecated. Existing units continue to work through
# scripts/run-multi.sh as a compatibility shim, but fresh installs should use
# qwen36.service. PARALLEL=4 is now the default in $CONF.

# ---------- 8. summary ------------------------------------------------------
cat <<EOF

========================================================================
DONE.

Model:    $MODEL_DIR/$MODEL_FILE
Binary:   $LLAMA_DIR/build/bin/llama-server
Launcher: $LAUNCH   (edit $CONF for tuning, no need to touch this file)
Endpoint: http://localhost:$PORT/v1   (OpenAI-compatible, model alias "qwen3.6-27b")

One-shot:
  systemctl --user start qwen36   # or: $LAUNCH

As a systemd user service (auto-start on login):
  systemctl --user daemon-reload
  systemctl --user enable --now qwen36
  # to keep running across logout: sudo loginctl enable-linger \$USER

Parallel requests:
  # upstream MTP supports parallel slots; PARALLEL=$PARALLEL is configured.
  # Edit PARALLEL in $CONF if you want fewer/more slots, then restart:
  systemctl --user restart qwen36

Deprecated:
  * qwen36-multi.service is no longer installed by this script. If an older
    install already has it, it now runs scripts/run-multi.sh as a compatibility
    shim, uses the same MTP parallel path, and prints a deprecation warning.

Smoke test:
  curl -s http://localhost:$PORT/v1/chat/completions \\
    -H 'Content-Type: application/json' \\
    -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"write a python prime sieve"}]}' | jq .

Caveats:
  * --ctx-size $CTX_SIZE is tuned for 24 GB. Raise toward 262144 only if no
    other workload uses the GPU; drop to 65536 for quickest prefill.
  * Thinking is on by default. With --reasoning-format deepseek the <think>
    block lands in response.reasoning_content; content stays clean. To disable
    per-request: chat_template_kwargs={"enable_thinking": false}.
  * If you delete or rename the $BUILD_ENV conda env, the binary will lose its
    rpath target. Either keep the env, or set LD_LIBRARY_PATH to a CUDA 12 lib
    dir before launching.
========================================================================
EOF
