#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT_DIR"

# Initialize CUTLASS before attempting to build the wheel.
git submodule update --init --recursive --force
export MAX_JOBS="${MAX_JOBS:-1}"

# Rebuild the local wheel from scratch so the active environment picks up the
# latest extension sources.
cd kernels/hgemm
python3 -m pip uninstall toy-hgemm -y || true
python3 setup.py bdist_wheel
cd dist
python3 -m pip install *.whl
cd ..
# Remove build leftovers that only matter during packaging.
rm -rf toy_hgemm.egg-info __pycache__
