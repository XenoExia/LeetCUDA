#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT_DIR"

# Ensure CUTLASS and the repo submodules are available before the optional
# official flash-attn build starts.
git submodule update --init --recursive --force
export MAX_JOBS="${MAX_JOBS:-1}"

python3 -m pip install flash-attn --no-build-isolation || {
  echo "flash-attn install failed; custom MMA kernels can still run without official FA comparison"
}
