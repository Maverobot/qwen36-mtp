#!/usr/bin/env bash
# One-shot installer: Qwen3.6-35B-A3B (MoE, 40 layers, 3B active) on a single
# RTX 5070 Ti laptop (12 GB VRAM, sm_120 / Blackwell).
#
# Why a separate installer from install.sh:
#   * Different GPU arch (Blackwell sm_120 vs Ada sm_89). CUDA 12.4 doesn't ship
#     PTX/SASS for sm_120; we pull a newer toolkit (12.8+).
#   * Different model (35B-A3B MoE) and different quant (mradermacher i1-Q4_K_S).
#   * The MoE-A3B model needs --n-cpu-moe at runtime (set in run-laptop.sh) so
#     the 20 GB of experts live in RAM while attention/router/embeddings stay
#     on the 12 GB GPU. We do NOT need the crucible-mtp fork here: at the time
#     of writing the official Qwen3.6-35B-A3B has no MTP head, so plain
#     upstream llama.cpp is the right choice (and gets us native --n-cpu-moe).
#
# Reference numbers (community reports, RTX 5070 Ti laptop, i1-Q4_K_S, 128 K
# ctx, Q8_0 KV, flash attention, --n-cpu-moe 40):
#   ~25-45 tok/s decode depending on RAM speed (DDR5-5600 vs 6400) and prefix
#   cache hits. Prefill ~250-400 tok/s.
#
# Required system: 32 GB RAM minimum (48 GB+ comfortable for 128 K ctx).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------- tunables --------------------------------------------------------
PREFIX="${PREFIX:-$HOME/Dev/qwen36-laptop}"
LLAMA_DIR="$PREFIX/llama.cpp"
MODEL_DIR="$PREFIX/models/qwen36-35b-a3b"
MODEL_REPO="${MODEL_REPO:-mradermacher/Qwen3.6-35B-A3B-i1-GGUF}"
MODEL_FILE="${MODEL_FILE:-Qwen3.6-35B-A3B.i1-Q4_K_S.gguf}"
HF_ENV="${HF_ENV:-qwen36-hf}"
BUILD_ENV="${BUILD_ENV:-qwen36-build-blackwell}"
CUDA_LABEL="${CUDA_LABEL:-cuda-12.8.1}"   # Blackwell support landed in 12.8
CUDA_ARCH="${CUDA_ARCH:-120}"             # 120 = Blackwell consumer (5070/5080/5090)
PORT="${PORT:-8080}"
JOBS="$(nproc)"

LLAMA_UPSTREAM_URL="https://github.com/ggml-org/llama.cpp.git"

log()  { printf '\033[1;32m[+] %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[!] %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[x] %s\033[0m\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# ---------- 0. sanity -------------------------------------------------------
log "Sanity checks"
have git || die "git not found"
have cmake || die "cmake not found"
have nvidia-smi || die "nvidia-smi not found (NVIDIA driver missing?)"
have conda || die "conda not found. Install Miniconda/Mamba first."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
mkdir -p "$PREFIX" "$MODEL_DIR"

# RAM check (35B-A3B i1-Q4_K_S is ~20 GB; with 128K Q8_0 KV expect ~24-26 GB
# resident in CPU/RAM after MoE offload).
TOTAL_RAM_GB=$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
if (( TOTAL_RAM_GB < 30 )); then
  warn "Only ${TOTAL_RAM_GB} GiB RAM detected. 32 GiB is the realistic floor for"
  warn "this profile; you may need to reduce CTX_SIZE in run-laptop.sh."
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/lib/conda-strict.sh"

# ---------- 1. HF download env ---------------------------------------------
log "Conda env for huggingface-hub: $HF_ENV"
conda env list | awk '{print $1}' | grep -qx "$HF_ENV" || conda create -y -n "$HF_ENV" python=3.11
conda_activate "$HF_ENV"
pip install -q --upgrade "huggingface-hub[hf_transfer]"

# ---------- 2. download i1-Q4_K_S GGUF -------------------------------------
log "Downloading $MODEL_REPO/$MODEL_FILE -> $MODEL_DIR"
if [[ -f "$MODEL_DIR/$MODEL_FILE" ]]; then
  log "  already present, skipping"
else
  HF_HUB_ENABLE_HF_TRANSFER=1 \
    hf download "$MODEL_REPO" "$MODEL_FILE" --local-dir "$MODEL_DIR"
fi
SIZE_BYTES=$(stat -c%s "$MODEL_DIR/$MODEL_FILE")
(( SIZE_BYTES > 15*1024*1024*1024 )) \
  || die "GGUF only $SIZE_BYTES bytes — likely corrupt"
log "  GGUF OK ($((SIZE_BYTES/1024/1024/1024)) GiB)"

# Tokenizer / config files (best-effort)
HF_HUB_ENABLE_HF_TRANSFER=1 \
  hf download "$MODEL_REPO" --local-dir "$MODEL_DIR" \
    --include "*.json" --include "tokenizer*" 2>/dev/null || true

conda_deactivate

# ---------- 3. CUDA toolkit env (Blackwell) --------------------------------
log "Conda env for CUDA toolkit: $BUILD_ENV ($CUDA_LABEL)"
if ! conda env list | awk '{print $1}' | grep -qx "$BUILD_ENV"; then
  conda create -y -n "$BUILD_ENV" -c "nvidia/label/$CUDA_LABEL" cuda-toolkit
fi
conda_activate "$BUILD_ENV"
NVCC="$CONDA_PREFIX/bin/nvcc"
[[ -x "$NVCC" ]] || die "nvcc not found in $BUILD_ENV"
"$NVCC" --version | tail -1

# ---------- 4. upstream llama.cpp ------------------------------------------
log "Cloning llama.cpp (upstream master; --n-cpu-moe is supported there)"
if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  git clone "$LLAMA_UPSTREAM_URL" "$LLAMA_DIR"
fi
cd "$LLAMA_DIR"
git fetch origin
git checkout master
git pull --ff-only

# Verify --n-cpu-moe support is present (landed Sept 2025).
if ! grep -rq "n-cpu-moe" common/ tools/ 2>/dev/null; then
  die "--n-cpu-moe not found in this llama.cpp checkout. Update the repo."
fi

# ---------- 5. build llama.cpp with CUDA -----------------------------------
# CMAKE_CUDA_RUNTIME_LIBRARY=Shared: conda's cudart package is shared-only;
# without this flag CMake defaults to static linking which fails with
# undefined references to @libcudart.so.12 symbols.
#
# CUDA 12.8+ conda packages may install libraries under
# targets/x86_64-linux/lib/ rather than directly under lib/; discover the
# actual path and export LIBRARY_PATH so the GCC linker subprocess finds
# libcudart.so / libcublas.so regardless of cmake's own path detection.
#
# BUILD_SHARED_LIBS=OFF: avoids transitive shared-library dependency failures
# where libggml-cuda.so's unresolved CUDA symbols cannot be satisfied when
# linking llama-server.  Everything is statically bundled into one binary.
log "Building llama.cpp (CUDA arch $CUDA_ARCH, $JOBS jobs)"
rm -rf build

# Locate the CUDA shared-library directory inside the conda env.
_CUDA_LIB_DIR=$(find "$CONDA_PREFIX" -maxdepth 6 \
  -name "libcudart.so*" -not -path "*/stubs/*" -not -name "*.a" \
  2>/dev/null | head -1 | xargs -r dirname)
[[ -d "$_CUDA_LIB_DIR" ]] || _CUDA_LIB_DIR="$CONDA_PREFIX/lib"
log "  CUDA lib dir: $_CUDA_LIB_DIR"

# LIBRARY_PATH is used by GCC/ld as an extra library search path and is
# inherited by every linker invocation spawned by cmake/make.
export LIBRARY_PATH="${_CUDA_LIB_DIR}:${CONDA_PREFIX}/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"

cmake -S . -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DCUDAToolkit_ROOT="$CONDA_PREFIX" \
  -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,$_CUDA_LIB_DIR -Wl,-rpath,$CONDA_PREFIX/lib -L$_CUDA_LIB_DIR -L$CONDA_PREFIX/lib" \
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,$_CUDA_LIB_DIR -Wl,-rpath,$CONDA_PREFIX/lib -L$_CUDA_LIB_DIR -L$CONDA_PREFIX/lib"
cmake --build build --config Release -j "$JOBS" --target llama-server

LLAMA_BIN="$LLAMA_DIR/build/bin/llama-server"
[[ -x "$LLAMA_BIN" ]] || die "llama-server not built"

# ---------- 6. write env file ----------------------------------------------
ENV_DIR="$HOME/.config/qwen36-mtp"
ENV_FILE="$ENV_DIR/laptop.env"
mkdir -p "$ENV_DIR"
SLOT_CACHE_DIR="$PREFIX/slot-cache"
mkdir -p "$SLOT_CACHE_DIR"
cat > "$ENV_FILE" <<EOF
# Generated by install-laptop.sh on $(date -Iseconds)
LLAMA_BIN=$LLAMA_BIN
MODEL_PATH=$MODEL_DIR/$MODEL_FILE
SLOT_CACHE_DIR=$SLOT_CACHE_DIR
PORT=$PORT
HOST=127.0.0.1
ALIAS=qwen3.6-35b-a3b
CTX_SIZE=131072
N_CPU_MOE=40
EOF
log "Wrote $ENV_FILE"

# ---------- 7. symlink copilot-wrappers into ~/.local/bin ------------------
LOCAL_BIN="$HOME/.local/bin"
mkdir -p "$LOCAL_BIN"
log "Symlinking copilot-wrappers -> $LOCAL_BIN"
for _src in "$REPO_DIR/copilot-wrappers"/*; do
  _dst="$LOCAL_BIN/$(basename "$_src")"
  ln -sf "$_src" "$_dst"
  log "  $_dst"
done

log "Done. Start the server with:"
echo "  set -a; source $ENV_FILE; set +a; $(dirname "$0")/run-laptop.sh"
