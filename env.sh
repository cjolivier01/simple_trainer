#!/usr/bin/env bash

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "env.sh: activate a conda environment before sourcing this file" >&2
  return 1 2>/dev/null || exit 1
fi

CONDA_ENV_NAME="$(basename "${CONDA_PREFIX}")"
export SNAPSHOT_CACHE_DIR="${CONDA_PREFIX}/snapshot-cache"
export SNAPSHOT_OCI_LAYOUT_PATH="${CONDA_PREFIX}/snapshot-oci"
export SNAPSHOT_OCI_TAG="${CONDA_ENV_NAME}"
mkdir -p "${SNAPSHOT_CACHE_DIR}"
mkdir -p "${SNAPSHOT_OCI_LAYOUT_PATH}"
