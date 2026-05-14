#!/bin/bash

# LeNet/CIFAR-10 training launcher, modeled on ai/cubic.sh.
# Wraps tools/xtrain.py (which itself wraps tools/train.py).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${CONDA_PREFIX:-}" && -f "${SCRIPT_DIR}/env.sh" ]]; then
  # shellcheck source=env.sh
  source "${SCRIPT_DIR}/env.sh"
fi

# TRAIN_SCRIPT="${TRAIN_SCRIPT:-tools/xtrain.py}"
# TRAIN_SCRIPT="tools/xtrain.py --xtrain-external-only=0"
# TRAIN_SCRIPT="tools/xtrain.py --xtrain-restore=ai/train_lenet"
# TRAIN_SCRIPT="tools/xtrain.py --xtrain-build-snapshot=1 --xtrain-snapshot-tag=ai/train_lenet:$(basename ${CONDA_PREFIX}) --xtrain-snapshot-push=1"
TRAIN_SCRIPT="tools/xtrain.py"

CLEAN=0
CLEAN_NOEXIT=0
DDP_GPUS=1
FAST=1
DATA_WORKERS=2
CREATE_ARGS=""
RESTORE_REF=""
PASSTHROUGH=()

# Snapshot ref published by --create and (optionally) restored via --restore.
SNAPSHOT_TAG="ai/train_lenet:$(basename ${CONDA_PREFIX:-default})"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)            CLEAN=1; shift ;;
    --create)           CREATE_ARGS="--xtrain-build-snapshot=1 --xtrain-snapshot-tag=${SNAPSHOT_TAG} --xtrain-snapshot-push=1"; CLEAN=1; CLEAN_NOEXIT=1; shift ;;
    --restore)          RESTORE_REF="${SNAPSHOT_TAG}"; shift ;;
    --restore=*)        RESTORE_REF="${1#*=}"; shift ;;
    --ddp)              DDP_GPUS=2; shift ;;
    --ddp=*)            DDP_GPUS="${1#*=}"; shift ;;
    --no-ddp|--single)  DDP_GPUS=1; shift ;;
    --data-workers=*)   DATA_WORKERS="${1#*=}"; shift ;;
    --data-workers)     DATA_WORKERS="$2"; shift 2 ;;
    --fast)             FAST=1; shift ;;
    --fast=*)           FAST="${1#*=}"; shift ;;
    --no-fast)          FAST=0; shift ;;
    *)                  PASSTHROUGH+=("$1"); shift ;;
  esac
done

# --fast=0 / --no-fast: bypass xtrain entirely; run tools/train.py directly.
# CREATE_ARGS / RESTORE_REF are xtrain-only, so silently drop them — `--fast=0`
# wins (matches xtrain.py's own --xtrain-fast=0 semantics).
if [[ "${FAST}" == "0" ]]; then
  TRAIN_SCRIPT="tools/train.py"
  if [[ -n "${CREATE_ARGS}" || -n "${RESTORE_REF}" ]]; then
    echo "[lenet.sh] --fast=0: ignoring --create/--restore (xtrain-only)" >&2
    CREATE_ARGS=""
    RESTORE_REF=""
  fi
fi

if [[ "${CLEAN}" == "1" ]]; then
  IDS="$(snapshot image ls --plain | awk '{print$1}')"
  if [[ ! -z "${IDS}" ]]; then
    snapshot image rm ${IDS}
  fi
  snapshot cache clean --yes
  IDS="$(snapshot oci ls --plain | awk '{print$1}')"
  if [[ ! -z "${IDS}" ]]; then
    while read -r ID; do
      if [[ ! -z "${ID}" ]]; then
        snapshot oci rm "${ID}" --yes
      fi
    done <<< "${IDS}"
  fi
  echo "Resultant state"
  snapshot image ls
  snapshot cache ls
  snapshot oci ls
  if [[ "${CLEAN_NOEXIT}" != "1" ]]; then
    exit 0
  fi
fi

if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  LOCAL_CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((DDP_GPUS - 1)))
else
  LOCAL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
fi

if [[ $DDP_GPUS -gt 1 ]]; then
  LAUNCHER=(torchrun --nproc-per-node="$DDP_GPUS")
  if [[ -n "${MASTER_PORT:-}" ]]; then
    LAUNCHER+=(--master-port="$MASTER_PORT")
  fi
  LAUNCHER+=(scripts/distributed_launcher.py)
else
  LAUNCHER=(python)
fi

RESTORE_ARGS=()
if [[ -n "${RESTORE_REF}" ]]; then
  RESTORE_ARGS=(--xt-restore="${RESTORE_REF}")
fi

ENV_ARGS=(CUDA_VISIBLE_DEVICES="${LOCAL_CUDA_VISIBLE_DEVICES}")
if [[ -z "${XTRAIN_DEFAULT_RESTORE_REF+x}" ]]; then
  ENV_ARGS+=(XTRAIN_DEFAULT_RESTORE_REF="${SNAPSHOT_TAG}")
fi

set -x
env "${ENV_ARGS[@]}" \
  "${LAUNCHER[@]}" \
  ${TRAIN_SCRIPT} \
  ${CREATE_ARGS} \
  "${RESTORE_ARGS[@]}" \
  --epochs 1 \
  --batch-size 64 \
  --max-steps 10 \
  --data-workers "${DATA_WORKERS}" \
  "${PASSTHROUGH[@]}"
