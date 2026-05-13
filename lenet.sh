#!/bin/bash
# LeNet/CIFAR-10 training launcher, modeled on ai/cubic.sh.
# Wraps tools/xtrain.py (which itself wraps tools/train.py or tools/train_ddp.py).

# TRAIN_SCRIPT="${TRAIN_SCRIPT:-tools/xtrain.py}"
# TRAIN_SCRIPT="tools/xtrain.py --xtrain-external-only=0"
TRAIN_SCRIPT="tools/xtrain.py"

CLEAN=0
CLEAN_NOEXIT=0
DDP_GPUS=1
CREATE=0
RESTORE_REF=""
PASSTHROUGH=()

# Snapshot ref published by --create and (optionally) restored via --restore.
SNAPSHOT_TAG="lenet/cifar10:$(basename ${CONDA_PREFIX:-default})"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)            CLEAN=1; shift ;;
    --create)           CREATE=1; CLEAN=1; CLEAN_NOEXIT=1; shift ;;
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

# DDP routes through tools/train_ddp.py via xtrain's XTRAIN_SCRIPT override;
# single-GPU uses xtrain's default tools/train.py.
if [[ $DDP_GPUS -gt 1 ]]; then
  export XTRAIN_SCRIPT="tools/train_ddp.py"
  LAUNCHER=(torchrun --nproc-per-node="$DDP_GPUS")
else
  LAUNCHER=(python)
fi

# --create: build & tag a fresh snapshot via xtrain's --xt-restore=auto-equivalent.
# xtrain doesn't have --xtrain-build-snapshot the way xdtrain does, so we drive
# snapshot generate + snapshot snapshot + snapshot tag manually.
if [[ "${CREATE}" == "1" ]]; then
  REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  RUNTIME_DIR="${REPO_ROOT}/.xtrain_runtime"
  BOOTSTRAP="${REPO_ROOT}/bootstrap_train.py"
  MANIFEST="${BOOTSTRAP}.manifest.json"
  TRAIN_TARGET="${REPO_ROOT}/${XTRAIN_SCRIPT:-tools/train.py}"
  rm -rf "${RUNTIME_DIR}"
  mkdir -p "${RUNTIME_DIR}"

  set -x
  python -m snapshot generate \
    --repo-root "${REPO_ROOT}" \
    --pythonpath "${REPO_ROOT}" \
    --script "${TRAIN_TARGET}" \
    --output-script "${BOOTSTRAP}" \
    --manifest "${MANIFEST}" \
    --runtime-dir "${RUNTIME_DIR}" \
    --allow-missing-stable-modules

  CUDA_VISIBLE_DEVICES="${LOCAL_CUDA_VISIBLE_DEVICES}" \
    python "${BOOTSTRAP}" snapshot \
      --runtime-dir "${RUNTIME_DIR}" \
      --worker-timeout 180 \
      --cuda-warmup \
      --snapshotd

  python -m snapshot tag --runtime-dir "${RUNTIME_DIR}" "${SNAPSHOT_TAG}"
  set +x
  echo "Snapshot built + tagged: ${SNAPSHOT_TAG}"
  echo "Re-run without --create (add --restore to restore from this tag)."
  exit 0
fi

RESTORE_ARGS=()
if [[ -n "${RESTORE_REF}" ]]; then
  RESTORE_ARGS=(--xt-restore="${RESTORE_REF}")
fi

# --standalone --nnodes=1
set -x
CUDA_VISIBLE_DEVICES="${LOCAL_CUDA_VISIBLE_DEVICES}" \
  "${LAUNCHER[@]}" \
  ${TRAIN_SCRIPT} \
  "${RESTORE_ARGS[@]}" \
  --epochs 1 \
  --batch-size 64 \
  --max-steps 5 \
  "${PASSTHROUGH[@]}"
