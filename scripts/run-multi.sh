#!/usr/bin/env bash
# Convenience wrapper: launch the multi-slot, no-MTP profile.
#
# Trade-off vs the default MTP+single-slot profile:
#   - Concurrent clients (e.g. parallel Copilot CLI subagents) no longer queue
#   - Single-stream tok/s drops ~30-50% because MTP speculative decoding is off
#
# Defaults (override by exporting before invocation, or via an EnvironmentFile):
#   PARALLEL=4
#   PORT=8080           (shared with the MTP profile; only one can run at a time
#                        — see Conflicts= in qwen36{,-multi}.service)
#   ALIAS=qwen3.6-27b-multi
#   SPS=0.5             (--slot-prompt-similarity)
#
# Everything else (LLAMA_BIN, MODEL_PATH, CTX_SIZE, CACHE_REUSE, SLOT_CACHE_DIR,
# HOST) is inherited from the environment, exactly like scripts/run.sh.

set -e

# These four values *define* the multi profile, so we force-set them regardless
# of what was inherited from EnvironmentFile= or the parent shell. To customize
# (different port, different parallel count), edit this script or invoke
# scripts/run.sh directly with your own overrides.
export DISABLE_MTP=1
export PARALLEL=4
export PORT=8080
export ALIAS=qwen3.6-27b-multi
export SPS="${SPS:-0.5}"

exec "$(dirname "$0")/run.sh" "$@"
