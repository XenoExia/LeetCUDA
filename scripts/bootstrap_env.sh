#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

echo "[1/4] Initializing submodules"
git submodule update --init --recursive --force

echo "[2/4] Installing base Python dependencies"
python3 -m pip install --upgrade packaging ninja matplotlib

echo "[3/4] Ensuring flash-attn comparison dependency"
if [[ "${SKIP_FLASH_ATTN_INSTALL:-0}" == "1" ]]; then
  echo "Skipping flash-attn install because SKIP_FLASH_ATTN_INSTALL=1"
elif python3 -c "import flash_attn" >/dev/null 2>&1; then
  echo "flash-attn already installed"
else
  export MAX_JOBS="${MAX_JOBS:-1}"
  if python3 -m pip install flash-attn --no-build-isolation; then
    echo "flash-attn installed successfully"
  else
    echo "flash-attn install failed; LeetCUDA custom kernels still work, but official comparison will be skipped"
  fi
fi

echo "[4/4] Running environment doctor"
python3 scripts/doctor.py
