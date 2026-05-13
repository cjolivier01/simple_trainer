#!/bin/bash
# LeNet/CIFAR-10 training launcher, modeled on ai/cubic.sh.
# Wraps tools/xtrain.py (which itself wraps tools/train.py).

# TRAIN_SCRIPT="${TRAIN_SCRIPT:-tools/xtrain.py}"
# TRAIN_SCRIPT="tools/xtrain.py --xtrain-external-only=0"
# TRAIN_SCRIPT="tools/xtrain.py --xtrain-restore=ai/train_lenet"
# TRAIN_SCRIPT="tools/xtrain.py --xtrain-build-snapshot=1 --xtrain-snapshot-tag=ai/train_lenet:$(basename ${CONDA_PREFIX}) --xtrain-snapshot-push=1"
TRAIN_SCRIPT="tools/xtrain.py"

CLEAN=0
CLEAN_NOEXIT=0
DDP_GPUS=1
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
    *)                  PASSTHROUGH+=("$1"); shift ;;
  esac
done

if [[ "${CLEAN}" == "1" ]]; then
  IDS="$(snapshot image ls --plain | awk '{print$1}')"
  if [[ ! -z "${IDS}" ]]; then
    snapshot image rm ${IDS}
  fi
  snapshot cache clean --yes
  IDS="$(snapshot oci ls --plain | awk '{print$1}')"
  if [[ ! -z "${IDS}" ]]; then
    snapshot oci rm ${IDS} --yes
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
  LAUNCHER=(torchrun --nproc-per-node="$DDP_GPUS" scripts/distributed_launcher.py)
else
  LAUNCHER=(python)
fi

RESTORE_ARGS=()
if [[ -n "${RESTORE_REF}" ]]; then
  RESTORE_ARGS=(--xt-restore="${RESTORE_REF}")
fi

set -x
CUDA_VISIBLE_DEVICES="${LOCAL_CUDA_VISIBLE_DEVICES}" \
  "${LAUNCHER[@]}" \
  ${TRAIN_SCRIPT} \
  ${CREATE_ARGS} \
  "${RESTORE_ARGS[@]}" \
  --epochs 1 \
  --batch-size 64 \
  --max-steps 5 \
  "${PASSTHROUGH[@]}"
