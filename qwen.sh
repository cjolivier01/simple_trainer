#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=tools/shell_lib.sh
source "${SCRIPT_DIR}/tools/shell_lib.sh"

simple_trainer_launch qwen "$@"
