#!/usr/bin/env python3
"""Inspect the local GPU/CUDA/PyTorch environment for LeetCUDA."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

from common import (
    configure_build_environment,
    cutlass_ready,
    package_available,
    read_total_memory_gib,
    repo_root,
    run_command,
)


def status_line(ok: bool, label: str, detail: str) -> None:
    """Render one aligned diagnostic line."""
    prefix = "[OK]" if ok else "[WARN]"
    print(f"{prefix:<6} {label:<18} {detail}")


def main() -> int:
    """Print the environment status and the recommended first commands."""
    root = repo_root()
    print(f"Repo root: {root}")
    print(f"Python: {sys.executable}")
    print(f"Platform: {platform.platform()}")

    env_info = configure_build_environment(verbose=False)
    capability = env_info["capability"] or "unknown"
    device_name = env_info["device_name"] or "not detected"
    status_line(
        env_info["device_name"] is not None,
        "CUDA device",
        f"{device_name} / sm_{str(capability).replace('.', '')}",
    )
    status_line(
        env_info["torch_version"] is not None,
        "PyTorch",
        f"{env_info['torch_version']} (CUDA {env_info['torch_cuda']})",
    )
    status_line(
        True,
        "Build defaults",
        f"TORCH_CUDA_ARCH_LIST={capability}, MAX_JOBS={env_info['max_jobs']}",
    )

    mem_gib = read_total_memory_gib()
    if mem_gib is not None:
        status_line(True, "Host memory", f"{mem_gib:.1f} GiB")

    nvcc_code, nvcc_out = run_command(["nvcc", "--version"])
    status_line(nvcc_code == 0, "nvcc", nvcc_out.splitlines()[-1] if nvcc_out else "not found")

    smi_code, smi_out = run_command(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    status_line(smi_code == 0, "nvidia-smi", smi_out.splitlines()[0] if smi_out else "not found")

    status_line(package_available("ninja"), "Python pkg", "ninja")
    status_line(package_available("matplotlib"), "Python pkg", "matplotlib")
    status_line(package_available("flash_attn"), "Python pkg", "flash_attn")
    status_line(cutlass_ready(), "CUTLASS", str(root / "third-party" / "cutlass"))

    submodule_code, submodule_out = run_command(
        ["git", "submodule", "status", "--recursive"], cwd=root
    )
    if submodule_code == 0 and submodule_out:
        # A leading '-' means the submodule is registered but not initialized.
        initialized = all(not line.startswith("-") for line in submodule_out.splitlines())
        status_line(initialized, "Submodules", submodule_out.replace("\n", " | "))

    print()
    print("Recommended workflow:")
    print("  1. bash scripts/bootstrap_env.sh")
    print("  2. python3 scripts/doctor.py")
    print("  3. python3 scripts/run_example.py kernels/elementwise/elementwise.py")
    print("  4. python3 scripts/run_example.py kernels/hgemm/hgemm.py --M 256 --N 256 --K 256 --mma --no-default --iters 1 --warmup 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
