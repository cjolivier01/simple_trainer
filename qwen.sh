#!/bin/bash
set -euo pipefail

# Qwen-style synthetic training launcher with SIGUSR1 pause/resume support.
# Normal launch runs through tools/xtrain.py. Use --pause from another shell to
# send SIGUSR1 to rank 0, wait for all rank snapshots, and stop the parked
# source ranks. Use --resume to restore those per-rank runtime snapshots, or
# fall back to the PyTorch checkpoint when no runtime snapshot is available.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${CONDA_PREFIX:-}" && -f "${SCRIPT_DIR}/env.sh" ]]; then
  # shellcheck source=env.sh
  source "${SCRIPT_DIR}/env.sh"
fi

MODE="run"
DDP_GPUS=2
FAST=1
DATA_WORKERS=2
SEED=1234
RUNTIME_DIR="${SCRIPT_DIR}/.qwen_pause"
SNAPSHOT_NAME="qwen-sigusr1"
SAVE_PATH="${RUNTIME_DIR}/qwen_hard_resume.pt"
PAUSE_TIMEOUT=180
RESUME_TIMEOUT=120
USE_SNAPSHOTD=1
HOST_PID_RESTORE=1
PAUSE_ALL=0
PASSTHROUGH=()

usage() {
  cat <<'EOF'
Usage: ./qwen.sh [options] [train args...]

Modes:
  --pause              Send SIGUSR1 to the running job and stop parked ranks
                       after all per-rank runtime snapshots are complete.
  --resume             Restore per-rank snapshots when present; otherwise
                       resume from the PyTorch checkpoint with loader position.

Launch options:
  --ddp[=N]            Launch N local ranks (default: 2).
  --single, --no-ddp   Launch one process.
  --data-workers=N     DataLoader workers per rank.
  --runtime-dir=DIR    Runtime directory for pause metadata and snapshots.
  --snapshot-name=NAME Named restore point under each rank runtime dir.
  --save-path=PATH     PyTorch checkpoint used for hard resume fallback.
  --seed=N             Deterministic seed.
  --qwen-dataset llava-instruct
                       Use LLaVA-Instruct-150K instead of synthetic token data.
  --fast / --no-fast   Use tools/xtrain.py / tools/train.py directly.
  --snapshotd          Use snapshotd for snapshot runtime restore (default).
  --unprivileged       Use direct CRIU instead of snapshotd.
  --host-pid-restore   Restore into the host PID namespace (default).
  --pid-namespace-restore
                       Restore into a fresh PID namespace.
  --pause-all          Send SIGUSR1 to every rank instead of rank 0 only.
EOF
}

die() {
  echo "[qwen.sh] error: $*" >&2
  exit 1
}

read_json_field() {
  local path="$1"
  local field="$2"
  python - "$path" "$field" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    payload = json.load(stream)
value = payload.get(sys.argv[2], "")
print(value)
PY
}

read_process_start_time() {
  local pid="$1"
  local stat_path="/proc/$pid/stat"
  [[ -r "$stat_path" ]] || return 1
  awk '{print $22}' "$stat_path"
}

process_matches_identity() {
  local pid="$1"
  local expected_start_time="$2"
  [[ -n "$expected_start_time" ]] || return 1
  local current_start_time=""
  current_start_time="$(read_process_start_time "$pid" 2>/dev/null || true)"
  [[ -n "$current_start_time" && "$current_start_time" == "$expected_start_time" ]]
}

stop_process_if_running() {
  local pid="$1"
  local expected_start_time="$2"
  local deadline=$((SECONDS + 10))
  local kill_deadline=$((SECONDS + 2))

  if ! process_matches_identity "$pid" "$expected_start_time"; then
    return 0
  fi

  echo "[qwen.sh] stopping parked checkpoint source pid $pid" >&2
  kill "$pid" 2>/dev/null || true
  while (( SECONDS < deadline )); do
    if ! process_matches_identity "$pid" "$expected_start_time"; then
      return 0
    fi
    if (( SECONDS >= kill_deadline )); then
      kill -KILL "$pid" 2>/dev/null || true
    fi
    sleep 1
  done
  die "timed out stopping pid $pid"
}

rank_runtime_dir() {
  local rank="$1"
  printf '%s/rank-%s\n' "$RUNTIME_DIR" "$rank"
}

snapshot_done_file() {
  local rank="$1"
  printf '%s/snapshots/%s/checkpoint.done\n' "$(rank_runtime_dir "$rank")" "$SNAPSHOT_NAME"
}

snapshot_available() {
  local rank
  for ((rank = 0; rank < DDP_GPUS; rank++)); do
    [[ -s "$(snapshot_done_file "$rank")" ]] || return 1
  done
  return 0
}

default_cuda_visible_devices() {
  local count="$1"
  local value=""
  local index
  for ((index = 0; index < count; index++)); do
    [[ -z "$value" ]] || value+=","
    value+="$index"
  done
  printf '%s\n' "$value"
}

pause_job() {
  shopt -s nullglob
  local pid_files=("${RUNTIME_DIR}"/pids/rank-*.json)
  shopt -u nullglob
  [[ "${#pid_files[@]}" -gt 0 ]] || die "no rank pid metadata under ${RUNTIME_DIR}/pids"

  local targets=()
  if [[ "$PAUSE_ALL" == "1" ]]; then
    targets=("${pid_files[@]}")
  else
    targets=("${RUNTIME_DIR}/pids/rank-0.json")
  fi

  local path pid start_time
  for path in "${targets[@]}"; do
    [[ -s "$path" ]] || die "missing pid metadata: $path"
    pid="$(read_json_field "$path" pid)"
    start_time="$(read_json_field "$path" pid_start_time)"
    if process_matches_identity "$pid" "$start_time"; then
      echo "[qwen.sh] sending SIGUSR1 to pid $pid from $path" >&2
      kill -USR1 "$pid"
    else
      die "pid $pid from $path is not the original running rank"
    fi
  done

  local deadline=$((SECONDS + PAUSE_TIMEOUT))
  local rank
  while (( SECONDS < deadline )); do
    local complete=1
    for ((rank = 0; rank < DDP_GPUS; rank++)); do
      if [[ ! -s "$(snapshot_done_file "$rank")" ]]; then
        complete=0
        break
      fi
    done
    [[ "$complete" == "1" ]] && break
    sleep 1
  done
  snapshot_available || die "timed out waiting for rank snapshots in $RUNTIME_DIR"

  for path in "${pid_files[@]}"; do
    pid="$(read_json_field "$path" pid)"
    start_time="$(read_json_field "$path" pid_start_time)"
    stop_process_if_running "$pid" "$start_time"
  done
  echo "[qwen.sh] pause snapshots are ready under $RUNTIME_DIR" >&2
}

restore_snapshots() {
  snapshot_available || return 1
  local rank cmd=()
  for ((rank = 1; rank < DDP_GPUS; rank++)); do
    cmd=(python -m snapshot.cli runtime restore
      --runtime-dir "$(rank_runtime_dir "$rank")"
      --snapshot-name "$SNAPSHOT_NAME"
      --resume-timeout "$RESUME_TIMEOUT"
      --background)
    if [[ "$HOST_PID_RESTORE" == "1" ]]; then
      cmd+=(--host-pid-restore)
    fi
    if [[ "$USE_SNAPSHOTD" == "1" ]]; then
      cmd+=(--snapshotd)
    else
      cmd+=(--unprivileged)
    fi
    "${cmd[@]}" &
  done

  cmd=(python -m snapshot.cli runtime restore
    --runtime-dir "$(rank_runtime_dir 0)"
    --snapshot-name "$SNAPSHOT_NAME"
    --resume-timeout "$RESUME_TIMEOUT")
  if [[ "$HOST_PID_RESTORE" == "1" ]]; then
    cmd+=(--host-pid-restore)
  fi
  if [[ "$USE_SNAPSHOTD" == "1" ]]; then
    cmd+=(--snapshotd)
  else
    cmd+=(--unprivileged)
  fi
  "${cmd[@]}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pause) MODE="pause"; shift ;;
    --resume) MODE="resume"; shift ;;
    --ddp) DDP_GPUS=2; shift ;;
    --ddp=*) DDP_GPUS="${1#*=}"; shift ;;
    --single|--no-ddp) DDP_GPUS=1; shift ;;
    --data-workers=*) DATA_WORKERS="${1#*=}"; shift ;;
    --data-workers) DATA_WORKERS="$2"; shift 2 ;;
    --runtime-dir=*) RUNTIME_DIR="${1#*=}"; shift ;;
    --runtime-dir) RUNTIME_DIR="$2"; shift 2 ;;
    --snapshot-name=*) SNAPSHOT_NAME="${1#*=}"; shift ;;
    --snapshot-name) SNAPSHOT_NAME="$2"; shift 2 ;;
    --save-path=*) SAVE_PATH="${1#*=}"; shift ;;
    --save-path) SAVE_PATH="$2"; shift 2 ;;
    --seed=*) SEED="${1#*=}"; shift ;;
    --seed) SEED="$2"; shift 2 ;;
    --pause-timeout=*) PAUSE_TIMEOUT="${1#*=}"; shift ;;
    --pause-timeout) PAUSE_TIMEOUT="$2"; shift 2 ;;
    --resume-timeout=*) RESUME_TIMEOUT="${1#*=}"; shift ;;
    --resume-timeout) RESUME_TIMEOUT="$2"; shift 2 ;;
    --snapshotd) USE_SNAPSHOTD=1; shift ;;
    --unprivileged|--no-snapshotd) USE_SNAPSHOTD=0; shift ;;
    --host-pid-restore) HOST_PID_RESTORE=1; shift ;;
    --pid-namespace-restore) HOST_PID_RESTORE=0; shift ;;
    --pause-all) PAUSE_ALL=1; shift ;;
    --fast) FAST=1; shift ;;
    --fast=*) FAST="${1#*=}"; shift ;;
    --no-fast) FAST=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done

mkdir -p "$RUNTIME_DIR"

if [[ "$MODE" == "pause" ]]; then
  pause_job
  exit 0
fi

if [[ "$MODE" == "resume" ]]; then
  if restore_snapshots; then
    exit 0
  fi
  [[ -s "$SAVE_PATH" ]] || die "no snapshots found and no hard checkpoint at $SAVE_PATH"
  PASSTHROUGH=(--init-from "$SAVE_PATH" "${PASSTHROUGH[@]}")
fi

if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  LOCAL_CUDA_VISIBLE_DEVICES="$(default_cuda_visible_devices "$DDP_GPUS")"
else
  LOCAL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
fi

if [[ "$DDP_GPUS" -gt 1 ]]; then
  LAUNCHER=(torchrun --nproc-per-node="$DDP_GPUS")
  if [[ -n "${MASTER_PORT:-}" ]]; then
    LAUNCHER+=(--master-port="$MASTER_PORT")
  fi
  LAUNCHER+=(scripts/distributed_launcher.py)
else
  LAUNCHER=(python)
fi

if [[ "$FAST" == "0" ]]; then
  TRAIN_SCRIPT=(tools/train.py)
else
  TRAIN_SCRIPT=(tools/xtrain.py)
fi

PAUSE_BACKEND_ARG=(--pause-snapshotd)
if [[ "$USE_SNAPSHOTD" != "1" ]]; then
  PAUSE_BACKEND_ARG=(--no-pause-snapshotd)
fi

set -x
env CUDA_VISIBLE_DEVICES="${LOCAL_CUDA_VISIBLE_DEVICES}" \
  "${LAUNCHER[@]}" \
  "${TRAIN_SCRIPT[@]}" \
  --model qwen \
  --epochs 1 \
  --batch-size 8 \
  --max-steps 100 \
  --lr 1e-3 \
  --deterministic \
  --seed "$SEED" \
  --data-workers "$DATA_WORKERS" \
  --save-path "$SAVE_PATH" \
  --pause-runtime-dir "$RUNTIME_DIR" \
  --pause-snapshot-name "$SNAPSHOT_NAME" \
  "${PAUSE_BACKEND_ARG[@]}" \
  "${PASSTHROUGH[@]}"
