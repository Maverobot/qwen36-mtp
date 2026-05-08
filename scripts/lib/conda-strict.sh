#!/usr/bin/env bash

_qwen36_conda_with_nounset_disabled() {
  local restore_nounset=0
  local status=0

  if [[ $- == *u* ]]; then
    restore_nounset=1
    set +u
  fi

  conda "$@" || status=$?

  if (( restore_nounset )); then
    set -u
  fi

  return "$status"
}

conda_activate() {
  _qwen36_conda_with_nounset_disabled activate "$@"
}

conda_deactivate() {
  _qwen36_conda_with_nounset_disabled deactivate "$@"
}
