#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT_DIR"

git submodule update --init --recursive --force
export MAX_JOBS="${MAX_JOBS:-1}"

cd kernels/hgemm
python3 -m pip uninstall toy-hgemm -y || true
python3 setup.py bdist_wheel
cd dist
python3 -m pip install *.whl
cd ..
rm -rf toy_hgemm.egg-info __pycache__
