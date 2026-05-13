#!/usr/bin/env bash

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "env.sh: activate a conda environment before sourcing this file" >&2
  return 1 2>/dev/null || exit 1
fi

CONDA_ENV_NAME="$(basename "${CONDA_PREFIX}")"

_snapshot_choose_writable_dir() {
  local current="$1"
  local preferred="$2"
  local fallback="$3"
  local label="$4"
  local candidate="${current:-${preferred}}"

  if mkdir -p "${candidate}" 2>/dev/null && [[ -w "${candidate}" ]]; then
    printf '%s\n' "${candidate}"
    return 0
  fi

  if [[ "${candidate}" != "${fallback}" ]]; then
    echo "env.sh: ${label} '${candidate}' is not writable; using '${fallback}'" >&2
  fi
  mkdir -p "${fallback}"
  printf '%s\n' "${fallback}"
}

export SNAPSHOT_CACHE_DIR="${HOME}/.cache/snapshots"
mkdir -p "${SNAPSHOT_CACHE_DIR}"

export SNAPSHOT_OCI_LAYOUT_PATH="$(
  _snapshot_choose_writable_dir \
    "${SNAPSHOT_OCI_LAYOUT_PATH:-}" \
    "${CONDA_PREFIX}/snapshot-oci" \
    "${HOME}/.cache/snapshots/oci-layouts/${CONDA_ENV_NAME}" \
    "SNAPSHOT_OCI_LAYOUT_PATH"
)"
export SNAPSHOT_OCI_TAG="${SNAPSHOT_OCI_TAG:-${CONDA_ENV_NAME}}"

unset -f _snapshot_choose_writable_dir
