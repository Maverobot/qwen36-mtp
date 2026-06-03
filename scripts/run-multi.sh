#!/usr/bin/env bash
# Deprecated compatibility wrapper for the old qwen36-multi profile.
#
# Upstream llama.cpp now supports MTP with PARALLEL>1, so keep one service
# (`qwen36.service`) with the default PARALLEL=4. This wrapper exists only so
# old qwen36-multi.service units keep working during migration.
#
# Compatibility defaults:
#   PARALLEL=4
#   PORT / ALIAS are inherited from the environment, exactly like scripts/run.sh.
#   SPS=0.5             (--slot-prompt-similarity)
#
# Everything else (LLAMA_BIN, MODEL_PATH, CTX_SIZE, CACHE_REUSE, SLOT_CACHE_DIR,
# HOST) is inherited from the environment, exactly like scripts/run.sh.

set -e

if [[ -z "${QWEN36_SUPPRESS_DEPRECATION:-}" ]]; then
    echo "DEPRECATED: scripts/run-multi.sh and qwen36-multi.service are deprecated. Use qwen36.service; PARALLEL=4 is now the default." >&2
fi

# Force old multi units onto the new upstream-MTP parallel path, but preserve
# host-specific PORT / ALIAS overrides from the shared EnvironmentFile.
unset DISABLE_MTP
export PARALLEL=4
export SPS="${SPS:-0.5}"

exec "$(dirname "$0")/run.sh" "$@"
