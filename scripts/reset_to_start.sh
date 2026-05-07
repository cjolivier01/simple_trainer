#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${SNAPSHOT_OCI_LAYOUT_PATH:-}" ]]; then
  echo "reset_to_start.sh: source env.sh first so SNAPSHOT_OCI_LAYOUT_PATH is set" >&2
  exit 1
fi
if [[ -z "${SNAPSHOT_CACHE_DIR:-}" ]]; then
  echo "reset_to_start.sh: source env.sh first so SNAPSHOT_CACHE_DIR is set" >&2
  exit 1
fi

case "${SNAPSHOT_OCI_LAYOUT_PATH}" in
  ""|"/"|"/tmp"|"/tmp/"*|"/var"|"/var/"*|"/home"|"/home/"*)
    if [[ "${SNAPSHOT_OCI_LAYOUT_PATH}" != */snapshot-oci ]]; then
      echo "reset_to_start.sh: refusing unsafe OCI layout path: ${SNAPSHOT_OCI_LAYOUT_PATH}" >&2
      exit 1
    fi
    ;;
esac
case "${SNAPSHOT_CACHE_DIR}" in
  ""|"/"|"/tmp"|"/tmp/"*|"/var"|"/var/"*|"/home"|"/home/"*)
    if [[ "${SNAPSHOT_CACHE_DIR}" != */snapshot-cache ]]; then
      echo "reset_to_start.sh: refusing unsafe cache path: ${SNAPSHOT_CACHE_DIR}" >&2
      exit 1
    fi
    ;;
esac

python - <<'PY'
import json
import subprocess
import sys

proc = subprocess.run(
    ["snapshot", "image", "ls", "--json"],
    text=True,
    capture_output=True,
    check=False,
)
if proc.returncode != 0:
    sys.stderr.write(proc.stderr)
    raise SystemExit(proc.returncode)

for entry in json.loads(proc.stdout or "[]"):
    name = str(entry.get("name", "")).strip()
    tag = str(entry.get("tag", "")).strip()
    snapshot_id = str(entry.get("snapshot_id", "")).strip()
    if name and tag and tag != "<none>":
        ref = f"{name}:{tag}"
    elif name and name != "<none>":
        ref = name
    else:
        ref = snapshot_id
    if ref:
        subprocess.run(["snapshot", "image", "rm", ref], check=False)
PY

rm -rf "${SNAPSHOT_OCI_LAYOUT_PATH}"
mkdir -p "${SNAPSHOT_OCI_LAYOUT_PATH}"

rm -f \
  "${REPO_ROOT}/bootstrap_train.py" \
  "${REPO_ROOT}/bootstrap_train.py.manifest.json" \
  "${REPO_ROOT}/stable_modules.json" \
  "${REPO_ROOT}/stable_modules_filtered.json" \
  "${REPO_ROOT}/tools/stable_modules.json" \
  "${REPO_ROOT}/tools/stable_modules_filtered.json"

find "${REPO_ROOT}/__pycache__" -maxdepth 1 -type f -name 'bootstrap_train.*' -delete 2>/dev/null || true

snapshot cache clean --yes

python - <<'PY'
import json
import subprocess
import sys

checks = [
    ("snapshot image ls", ["snapshot", "image", "ls", "--json"]),
    ("snapshot oci ls", ["snapshot", "oci", "ls", "--json"]),
]
for label, cmd in checks:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    entries = json.loads(proc.stdout or "[]")
    if entries:
        sys.stderr.write(f"{label} is not empty after reset: {entries!r}\n")
        raise SystemExit(1)

from snapshot.cache import iter_cache_entries

cache_entries = [entry.repo_key for entry in iter_cache_entries()]
if cache_entries:
    sys.stderr.write(f"snapshot cache is not empty after reset: {cache_entries!r}\n")
    raise SystemExit(1)
PY

echo "Stable start state ready."
