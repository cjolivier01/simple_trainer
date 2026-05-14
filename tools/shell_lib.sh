#!/usr/bin/env bash

ST_SHELL_LIB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ST_REPO_ROOT="$(cd -- "${ST_SHELL_LIB_DIR}/.." && pwd)"

st_log() {
  echo "[${ST_LOG_NAME}] $*" >&2
}

st_die() {
  st_log "error: $*"
  exit 1
}

st_source_env() {
  if [[ -n "${CONDA_PREFIX:-}" && -f "${ST_REPO_ROOT}/env.sh" ]]; then
    # shellcheck source=../env.sh
    source "${ST_REPO_ROOT}/env.sh"
  fi
}

st_usage() {
  local default_model="$1"
  cat <<EOF
Usage: ./${ST_LOG_NAME} [options] [train args...]

Modes:
  --clean              Remove snapshot/xtrain runtime state and model pause data.
  --create             Clean, run, build an xtrain snapshot, tag it, and push it.
  --restore            Restore the default xtrain snapshot for the selected model.
  --restore=REF        Restore a specific xtrain snapshot ref.
  --pause              Send SIGUSR1 to the running job and stop parked ranks
                       after all per-rank runtime snapshots are complete.
  --resume             Restore per-rank runtime snapshots when present; otherwise
                       resume from the PyTorch checkpoint with loader position.

Launch options:
  --model NAME         Model to pass to tools/train.py (default: ${default_model}).
  --ddp[=N]            Launch N local ranks.
  --single, --no-ddp   Launch one process.
  --data-workers=N     DataLoader workers per rank.
  --runtime-dir=DIR    Runtime directory for pause metadata and snapshots.
  --snapshot-name=NAME Named restore point under each rank runtime dir.
  --save-path=PATH     PyTorch checkpoint used for hard resume fallback.
  --seed=N             Deterministic seed.
  --fast / --no-fast   Use tools/xtrain.py / tools/train.py directly.
  --snapshotd          Use snapshotd for snapshot runtime restore (default).
  --unprivileged       Use direct CRIU instead of snapshotd.
  --host-pid-restore   Restore into the host PID namespace (default).
  --pid-namespace-restore
                       Restore into a fresh PID namespace.
  --pause-all          Send SIGUSR1 to every rank instead of rank 0 only.

Model-specific train args are forwarded, for example:
  --qwen-dataset llava-instruct
EOF
}

st_find_requested_model() {
  local model="$1"
  shift
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model=*)
        model="${1#*=}"
        ;;
      --model)
        if [[ $# -gt 1 ]]; then
          model="$2"
          shift
        fi
        ;;
    esac
    shift
  done
  printf '%s\n' "$model"
}

st_set_model_defaults() {
  local model="$1"
  local env_name
  env_name="$(basename "${CONDA_PREFIX:-default}")"

  DDP_GPUS=1
  FAST=1
  DATA_WORKERS=2
  SEED=1234
  BATCH_SIZE=64
  MAX_STEPS=10
  LR=""
  DETERMINISTIC=0
  RUNTIME_DIR="${ST_REPO_ROOT}/.${model}_pause"
  SNAPSHOT_NAME="${model}-sigusr1"
  SAVE_PATH="${RUNTIME_DIR}/${model}_hard_resume.pt"
  SNAPSHOT_TAG="ai/train_${model}:${env_name}"

  case "$model" in
    lenet)
      SAVE_PATH="./lenet_cifar10.pt"
      SNAPSHOT_TAG="ai/train_lenet:${env_name}"
      ;;
    qwen)
      DDP_GPUS=2
      BATCH_SIZE=1024
      MAX_STEPS=100000
      LR="1e-3"
      DETERMINISTIC=1
      SAVE_PATH="${RUNTIME_DIR}/qwen_hard_resume.pt"
      SNAPSHOT_TAG="ai/train_qwen:${env_name}"
      ;;
  esac
}

st_read_json_field() {
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

st_read_process_start_time() {
  local pid="$1"
  local stat_path="/proc/$pid/stat"
  [[ -r "$stat_path" ]] || return 1
  awk '{print $22}' "$stat_path"
}

st_process_matches_identity() {
  local pid="$1"
  local expected_start_time="$2"
  [[ -n "$expected_start_time" ]] || return 1
  local current_start_time=""
  current_start_time="$(st_read_process_start_time "$pid" 2>/dev/null || true)"
  [[ -n "$current_start_time" && "$current_start_time" == "$expected_start_time" ]]
}

st_stop_process_if_running() {
  local pid="$1"
  local expected_start_time="$2"
  local deadline=$((SECONDS + 10))
  local kill_deadline=$((SECONDS + 2))

  if ! st_process_matches_identity "$pid" "$expected_start_time"; then
    return 0
  fi

  st_log "stopping parked checkpoint source pid $pid"
  kill "$pid" 2>/dev/null || true
  while ((SECONDS < deadline)); do
    if ! st_process_matches_identity "$pid" "$expected_start_time"; then
      return 0
    fi
    if ((SECONDS >= kill_deadline)); then
      kill -KILL "$pid" 2>/dev/null || true
    fi
    sleep 1
  done
  st_die "timed out stopping pid $pid"
}

st_rank_runtime_dir() {
  local rank="$1"
  printf '%s/rank-%s\n' "$RUNTIME_DIR" "$rank"
}

st_snapshot_done_file() {
  local rank="$1"
  printf '%s/snapshots/%s/checkpoint.done\n' "$(st_rank_runtime_dir "$rank")" "$SNAPSHOT_NAME"
}

st_snapshot_available() {
  local rank
  for ((rank = 0; rank < DDP_GPUS; rank++)); do
    [[ -s "$(st_snapshot_done_file "$rank")" ]] || return 1
  done
  return 0
}

st_default_cuda_visible_devices() {
  local count="$1"
  local value=""
  local index
  for ((index = 0; index < count; index++)); do
    [[ -z "$value" ]] || value+=","
    value+="$index"
  done
  printf '%s\n' "$value"
}

st_pause_job() {
  shopt -s nullglob
  local pid_files=("${RUNTIME_DIR}"/pids/rank-*.json)
  shopt -u nullglob
  [[ "${#pid_files[@]}" -gt 0 ]] || st_die "no rank pid metadata under ${RUNTIME_DIR}/pids"

  local targets=()
  if [[ "$PAUSE_ALL" == "1" ]]; then
    targets=("${pid_files[@]}")
  else
    targets=("${RUNTIME_DIR}/pids/rank-0.json")
  fi

  local path pid start_time
  for path in "${targets[@]}"; do
    [[ -s "$path" ]] || st_die "missing pid metadata: $path"
    pid="$(st_read_json_field "$path" pid)"
    start_time="$(st_read_json_field "$path" pid_start_time)"
    if st_process_matches_identity "$pid" "$start_time"; then
      st_log "sending SIGUSR1 to pid $pid from $path"
      kill -USR1 "$pid"
    else
      st_die "pid $pid from $path is not the original running rank"
    fi
  done

  local deadline=$((SECONDS + PAUSE_TIMEOUT))
  local rank
  while ((SECONDS < deadline)); do
    local complete=1
    for ((rank = 0; rank < DDP_GPUS; rank++)); do
      if [[ ! -s "$(st_snapshot_done_file "$rank")" ]]; then
        complete=0
        break
      fi
    done
    [[ "$complete" == "1" ]] && break
    sleep 1
  done
  st_snapshot_available || st_die "timed out waiting for rank snapshots in $RUNTIME_DIR"

  for path in "${pid_files[@]}"; do
    pid="$(st_read_json_field "$path" pid)"
    start_time="$(st_read_json_field "$path" pid_start_time)"
    st_stop_process_if_running "$pid" "$start_time"
  done
  st_log "pause snapshots are ready under $RUNTIME_DIR"
}

st_restore_snapshots() {
  st_snapshot_available || return 1
  local rank cmd=()
  for ((rank = 1; rank < DDP_GPUS; rank++)); do
    cmd=(python -m snapshot.cli runtime restore
      --runtime-dir "$(st_rank_runtime_dir "$rank")"
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
    --runtime-dir "$(st_rank_runtime_dir 0)"
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

st_validate_positive_int() {
  local value="$1"
  local label="$2"
  case "$value" in
    ""|*[!0-9]*)
      st_die "${label} must be a positive integer"
      ;;
  esac
  ((value > 0)) || st_die "${label} must be a positive integer"
}

st_remove_path_if_safe() {
  local target="$1"
  [[ -e "$target" || -L "$target" ]] || return 0

  local resolved repo data_dir
  resolved="$(realpath -m -- "$target")"
  repo="$(realpath -m -- "$ST_REPO_ROOT")"
  data_dir="$(realpath -m -- "${ST_REPO_ROOT}/data")"

  [[ "$resolved" != "/" ]] || st_die "refusing to remove /"
  [[ "$resolved" != "$repo" ]] || st_die "refusing to remove repo root $repo"
  if [[ -n "${HOME:-}" ]]; then
    local home_dir
    home_dir="$(realpath -m -- "$HOME")"
    [[ "$resolved" != "$home_dir" ]] || st_die "refusing to remove home directory $home_dir"
  fi
  if [[ "$resolved" == "$data_dir" || "$resolved" == "$data_dir/"* ]]; then
    st_log "skipping dataset path $resolved"
    return 0
  fi

  st_log "removing $resolved"
  rm -rf -- "$resolved"
}

st_clean_repo_artifacts() {
  shopt -s nullglob globstar
  local targets=(
    "${ST_REPO_ROOT}/bootstrap_train.py"
    "${ST_REPO_ROOT}/bootstrap_train.py.manifest.json"
    "${ST_REPO_ROOT}"/bootstrap_train_*.py
    "${ST_REPO_ROOT}"/bootstrap_train_*.py.manifest.json
    "${ST_REPO_ROOT}/.xdtrain_runtime"
    "${ST_REPO_ROOT}/.pytest_cache"
    "${ST_REPO_ROOT}/.ruff_cache"
    "${ST_REPO_ROOT}"/**/__pycache__
  )
  shopt -u nullglob globstar

  local target
  for target in "${targets[@]}"; do
    st_remove_path_if_safe "$target"
  done
}

st_clean_runtime_dirs() {
  shopt -s nullglob
  local candidates=(
    "$RUNTIME_DIR"
    "${ST_REPO_ROOT}/.${MODEL}_pause"
    "${ST_REPO_ROOT}/${MODEL}_pause"
    "${ST_REPO_ROOT}"/.*_pause
    "${ST_REPO_ROOT}"/*_pause
  )
  shopt -u nullglob
  local seen=()
  local candidate resolved existing
  for candidate in "${candidates[@]}"; do
    [[ -n "$candidate" ]] || continue
    resolved="$(realpath -m -- "$candidate")"
    for existing in "${seen[@]}"; do
      [[ "$resolved" != "$existing" ]] || continue 2
    done
    seen+=("$resolved")
    st_remove_path_if_safe "$resolved"
  done
}

st_clean_snapshot_state() {
  if ! command -v snapshot >/dev/null 2>&1; then
    st_log "snapshot command not found; skipping snapshot image/cache cleanup"
    return 0
  fi

  local image_ids
  image_ids="$(snapshot image ls --plain 2>/dev/null | awk '{print $1}' || true)"
  if [[ -n "$image_ids" ]]; then
    # Intentional word splitting: snapshot ids are newline-separated tokens.
    snapshot image rm ${image_ids} || st_log "warning: failed to remove one or more snapshot images"
  fi

  snapshot cache clean --yes || st_log "warning: snapshot cache clean failed"

  local oci_ids id
  oci_ids="$(snapshot oci ls --plain 2>/dev/null | awk '{print $1}' || true)"
  if [[ -n "$oci_ids" ]]; then
    while read -r id; do
      if [[ -n "$id" ]]; then
        snapshot oci rm "$id" --yes || st_log "warning: failed to remove OCI layout $id"
      fi
    done <<< "$oci_ids"
  fi

  st_log "resultant snapshot state"
  snapshot image ls || true
  snapshot cache ls || true
  snapshot oci ls || true
}

st_clean_all() {
  st_clean_snapshot_state
  st_clean_repo_artifacts
  st_clean_runtime_dirs
}

simple_trainer_launch() {
  local default_model="$1"
  shift

  ST_LOG_NAME="$(basename -- "$0")"
  st_source_env

  MODEL="$(st_find_requested_model "$default_model" "$@")"
  st_set_model_defaults "$MODEL"

  MODE="run"
  CLEAN=0
  CLEAN_NOEXIT=0
  CREATE=0
  RESTORE_REF=""
  RESTORE_DEFAULT=0
  PAUSE_TIMEOUT=180
  RESUME_TIMEOUT=120
  USE_SNAPSHOTD=1
  HOST_PID_RESTORE=1
  PAUSE_ALL=0
  PASSTHROUGH=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --clean) CLEAN=1; shift ;;
      --create) CREATE=1; CLEAN=1; CLEAN_NOEXIT=1; shift ;;
      --restore) RESTORE_DEFAULT=1; shift ;;
      --restore=*) RESTORE_REF="${1#*=}"; shift ;;
      --pause) MODE="pause"; shift ;;
      --resume) MODE="resume"; shift ;;
      --model=*) MODEL="${1#*=}"; shift ;;
      --model)
        [[ $# -gt 1 ]] || st_die "--model expects a value"
        MODEL="$2"
        shift 2
        ;;
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
      -h|--help) st_usage "$default_model"; exit 0 ;;
      *) PASSTHROUGH+=("$1"); shift ;;
    esac
  done

  if [[ "$RESTORE_DEFAULT" == "1" && -z "$RESTORE_REF" ]]; then
    RESTORE_REF="$SNAPSHOT_TAG"
  fi

  CREATE_ARGS=()
  if [[ "$CREATE" == "1" ]]; then
    CREATE_ARGS=(--xtrain-build-snapshot=1 --xtrain-snapshot-tag="$SNAPSHOT_TAG" --xtrain-snapshot-push=1)
  fi

  RESTORE_ARGS=()
  if [[ -n "$RESTORE_REF" ]]; then
    RESTORE_ARGS=(--xt-restore="$RESTORE_REF")
  fi

  TRAIN_SCRIPT=(tools/xtrain.py)
  if [[ "$FAST" == "0" ]]; then
    TRAIN_SCRIPT=(tools/train.py)
    if [[ "${#CREATE_ARGS[@]}" -gt 0 || "${#RESTORE_ARGS[@]}" -gt 0 ]]; then
      st_log "--fast=0: ignoring --create/--restore (xtrain-only)"
      CREATE_ARGS=()
      RESTORE_ARGS=()
    fi
  fi

  if [[ "$CLEAN" == "1" ]]; then
    st_clean_all
    if [[ "$CLEAN_NOEXIT" != "1" ]]; then
      exit 0
    fi
  fi

  st_validate_positive_int "$DDP_GPUS" "--ddp"

  if [[ "$MODE" == "pause" ]]; then
    st_pause_job
    exit 0
  fi

  mkdir -p "$RUNTIME_DIR"

  if [[ "$MODE" == "resume" ]]; then
    if st_restore_snapshots; then
      exit 0
    fi
    [[ -s "$SAVE_PATH" ]] || st_die "no snapshots found and no hard checkpoint at $SAVE_PATH"
    PASSTHROUGH=(--init-from "$SAVE_PATH" "${PASSTHROUGH[@]}")
  fi

  local local_cuda_visible_devices
  if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
    local_cuda_visible_devices="$(st_default_cuda_visible_devices "$DDP_GPUS")"
  else
    local_cuda_visible_devices="${CUDA_VISIBLE_DEVICES}"
  fi

  local launcher=()
  if [[ "$DDP_GPUS" -gt 1 ]]; then
    launcher=(torchrun --nproc-per-node="$DDP_GPUS")
    if [[ -n "${MASTER_PORT:-}" ]]; then
      launcher+=(--master-port="$MASTER_PORT")
    fi
    launcher+=(scripts/distributed_launcher.py)
  else
    launcher=(python)
  fi

  local env_args=(CUDA_VISIBLE_DEVICES="${local_cuda_visible_devices}")
  if [[ -z "${XTRAIN_DEFAULT_RESTORE_REF+x}" ]]; then
    env_args+=(XTRAIN_DEFAULT_RESTORE_REF="${SNAPSHOT_TAG}")
  fi

  local pause_backend_arg=(--pause-snapshotd)
  if [[ "$USE_SNAPSHOTD" != "1" ]]; then
    pause_backend_arg=(--no-pause-snapshotd)
  fi

  local train_args=(
    --model "$MODEL"
    --batch-size "$BATCH_SIZE"
    --max-steps "$MAX_STEPS"
    --data-workers "$DATA_WORKERS"
    --save-path "$SAVE_PATH"
    --pause-runtime-dir "$RUNTIME_DIR"
    --pause-snapshot-name "$SNAPSHOT_NAME"
    "${pause_backend_arg[@]}"
  )
  if [[ -n "$LR" ]]; then
    train_args+=(--lr "$LR")
  fi
  if [[ "$DETERMINISTIC" == "1" ]]; then
    train_args+=(--deterministic)
  fi
  train_args+=(--seed "$SEED")

  set -x
  env "${env_args[@]}" \
    "${launcher[@]}" \
    "${TRAIN_SCRIPT[@]}" \
    "${CREATE_ARGS[@]}" \
    "${RESTORE_ARGS[@]}" \
    "${train_args[@]}" \
    "${PASSTHROUGH[@]}"
}
