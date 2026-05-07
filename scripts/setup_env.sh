#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SNAPSHOT_REPO_URL="${SNAPSHOT_REPO_URL:-ssh://git@github-ap.tesla.com/AI/snapshot}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "setup_env.sh: activate the target conda environment first" >&2
  exit 1
fi

export PYTHONNOUSERSITE=1
export PIP_USER=0

python - <<'PY'
import site
import sys

if not sys.prefix:
    raise SystemExit("Python sys.prefix is empty")
if site.ENABLE_USER_SITE:
    raise SystemExit("User site-packages is enabled; refusing to install into this environment")
PY

python -m pip install --disable-pip-version-check \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu130

python -m pip install --disable-pip-version-check -r "${REPO_ROOT}/requirements.txt"

if [[ -d "${REPO_ROOT}/../snapshot" ]]; then
  python -m pip install --disable-pip-version-check -e "${REPO_ROOT}/../snapshot"
else
  python -m pip install --disable-pip-version-check "git+${SNAPSHOT_REPO_URL}"
fi

python -m pip install --disable-pip-version-check pytest
