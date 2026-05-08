#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "$REPO_DIR/scripts/lib/conda-strict.sh"

activation_called=0
deactivation_called=0

conda() {
  case "$1" in
    activate)
      activation_called=1
      export CONDA_PREFIX="/tmp/fake-conda-env"
      export CXX="/usr/bin/g++"
      NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS} -ccbin=${CXX}"
      export NVCC_PREPEND_FLAGS
      ;;
    deactivate)
      deactivation_called=1
      unset CONDA_PREFIX
      ;;
    *)
      return 2
      ;;
  esac
}

assert_nounset_enabled() {
  if (: "${QWEN36_TEST_UNSET_SENTINEL}") 2>/dev/null; then
    echo "nounset was not restored" >&2
    return 1
  fi
}

conda_activate fake-env
[[ "$activation_called" == 1 ]]
[[ "$NVCC_PREPEND_FLAGS" == " -ccbin=/usr/bin/g++" ]]
assert_nounset_enabled

conda_deactivate
[[ "$deactivation_called" == 1 ]]
assert_nounset_enabled

echo "conda strict-mode wrapper OK"
