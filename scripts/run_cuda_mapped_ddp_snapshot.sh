#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: run_cuda_mapped_ddp_snapshot.sh [options] [-- <train_ddp.py args...>]

Create a CUDA-warmed DDP bootstrap snapshot, then launch restored ranks through
tools/xdtrain.py. xdtrain.py owns rank env, one-GPU CUDA_VISIBLE_DEVICES views,
and cuda-checkpoint device maps for each rank.

Options:
  --runtime-dir DIR                  Runtime state directory.
  --nproc-per-node N                 Local ranks to launch (default: 2).
  --snapshot-cuda-visible-devices L  GPU list used for snapshot warmup (default: 0).
  --cuda-visible-devices L           Restore target GPU list (default: CUDA_VISIBLE_DEVICES or 0..N-1).
  --python-bin PATH                  Python interpreter (default: python3).
  --master-addr HOST                 DDP master address (default: 127.0.0.1).
  --master-port PORT                 DDP master port (default: 29500).
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

require_cmd() {
  local target="$1"
  if [[ "$target" == */* ]]; then
    [[ -x "$target" ]] || die "missing executable: $target"
    return
  fi
  command -v "$target" >/dev/null 2>&1 || die "command not found: $target"
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

runtime_dir="$repo_root/.xdtrain_runtime"
nproc_per_node=2
snapshot_cuda_visible_devices="0"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES-}"
python_bin="python3"
master_addr="127.0.0.1"
master_port="29500"

while (($# > 0)); do
  case "$1" in
    --runtime-dir)
      [[ $# -ge 2 ]] || die "missing value for --runtime-dir"
      runtime_dir="$2"
      shift 2
      ;;
    --nproc-per-node)
      [[ $# -ge 2 ]] || die "missing value for --nproc-per-node"
      nproc_per_node="$2"
      shift 2
      ;;
    --snapshot-cuda-visible-devices)
      [[ $# -ge 2 ]] || die "missing value for --snapshot-cuda-visible-devices"
      snapshot_cuda_visible_devices="$2"
      shift 2
      ;;
    --cuda-visible-devices)
      [[ $# -ge 2 ]] || die "missing value for --cuda-visible-devices"
      cuda_visible_devices="$2"
      shift 2
      ;;
    --python-bin)
      [[ $# -ge 2 ]] || die "missing value for --python-bin"
      python_bin="$2"
      shift 2
      ;;
    --master-addr)
      [[ $# -ge 2 ]] || die "missing value for --master-addr"
      master_addr="$2"
      shift 2
      ;;
    --master-port)
      [[ $# -ge 2 ]] || die "missing value for --master-port"
      master_port="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

require_cmd "$python_bin"
require_cmd nvidia-smi

if [[ -z "$cuda_visible_devices" ]]; then
  cuda_visible_devices="$(default_cuda_visible_devices "$nproc_per_node")"
fi

bootstrap="$repo_root/bootstrap_train_ddp.py"
manifest="$bootstrap.manifest.json"
train_script="$repo_root/tools/train_ddp.py"
save_path="$repo_root/lenet_cifar10_ddp.pt"

if (($# > 0)); then
  train_args=("$@")
else
  train_args=(--epochs 1 --batch-size 64 --max-steps 2 --save-path "$save_path")
fi

rm -rf "$runtime_dir"
mkdir -p "$runtime_dir"

generate_cmd=(
  "$python_bin"
  -m
  snapshot
  generate
  --repo-root "$repo_root"
  --pythonpath "$repo_root"
  --script "$train_script"
  --output-script "$bootstrap"
  --manifest "$manifest"
  --runtime-dir "$runtime_dir"
  --allow-missing-stable-modules
)

snapshot_cmd=(
  "$python_bin"
  "$bootstrap"
  snapshot
  --runtime-dir "$runtime_dir"
  --worker-timeout 180
  --cuda-warmup
  --snapshotd
)

xdtrain_cmd=(
  "$python_bin"
  "$repo_root/tools/xdtrain.py"
  --runtime-dir "$runtime_dir"
  --nproc-per-node "$nproc_per_node"
  --snapshot-cuda-visible-devices "$snapshot_cuda_visible_devices"
  --cuda-visible-devices "$cuda_visible_devices"
  --master-addr "$master_addr"
  --master-port "$master_port"
)

export PYTHONPATH="$repo_root:$repo_root/../snapshot/src${PYTHONPATH:+:$PYTHONPATH}"

"${generate_cmd[@]}"
CUDA_VISIBLE_DEVICES="$snapshot_cuda_visible_devices" "${snapshot_cmd[@]}"
CUDA_VISIBLE_DEVICES="$cuda_visible_devices" "${xdtrain_cmd[@]}" -- "${train_args[@]}"
