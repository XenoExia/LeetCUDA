"""Shared helpers for bootstrap, diagnostics, and example launchers."""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    """Return the repository root resolved from this file."""
    return REPO_ROOT


def package_available(module_name: str) -> bool:
    """Check whether a Python package can be imported."""
    return importlib.util.find_spec(module_name) is not None


def read_total_memory_gib() -> float | None:
    """Read total host memory from /proc/meminfo when available."""
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) / (1024 * 1024)
    return None


def recommend_max_jobs(
    total_memory_gib: float | None = None, cpu_count: int | None = None
) -> int:
    """Choose a conservative build parallelism level for CUDA extensions."""
    if cpu_count is None:
        cpu_count = os.cpu_count() or 1
    if total_memory_gib is None:
        total_memory_gib = read_total_memory_gib()
    if total_memory_gib is None or total_memory_gib < 32:
        return 1
    if total_memory_gib < 64:
        return min(cpu_count, 2)
    return min(cpu_count, 4)


def detect_torch_device() -> dict[str, object]:
    """Collect the torch/CUDA facts used by the repo entrypoints."""
    info: dict[str, object] = {
        "torch_available": False,
        "torch_version": None,
        "torch_cuda": None,
        "cuda_available": False,
        "device_name": None,
        "capability": None,
        "import_error": None,
    }
    try:
        import torch
    except Exception as exc:  # pragma: no cover - diagnostic only
        info["import_error"] = str(exc)
        return info

    info["torch_available"] = True
    info["torch_version"] = torch.__version__
    info["torch_cuda"] = torch.version.cuda
    info["cuda_available"] = torch.cuda.is_available()
    if info["cuda_available"]:
        major, minor = torch.cuda.get_device_capability(0)
        info["capability"] = f"{major}.{minor}"
        info["device_name"] = torch.cuda.get_device_name(0)
    return info


def capability_to_sm(capability: str | None) -> str | None:
    """Convert a torch capability string like '12.0' to '120'."""
    if not capability:
        return None
    parts = capability.split(".")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        return None
    return "".join(parts)


def gencode_flags_for_capability(capability: str | None) -> str | None:
    """Build one explicit nvcc gencode flag for a capability string."""
    sm = capability_to_sm(capability)
    if sm is None:
        return None
    return f"-gencode arch=compute_{sm},code=sm_{sm}"


def configure_build_environment(verbose: bool = False) -> dict[str, object]:
    """Populate arch and build-parallelism defaults for direct JIT builds."""
    info = detect_torch_device()
    if info["capability"] and "TORCH_CUDA_ARCH_LIST" not in os.environ:
        os.environ["TORCH_CUDA_ARCH_LIST"] = str(info["capability"])
    if "MAX_JOBS" not in os.environ:
        os.environ["MAX_JOBS"] = str(recommend_max_jobs())
    if verbose:
        print(
            "Configured build environment: "
            f"TORCH_CUDA_ARCH_LIST={os.environ.get('TORCH_CUDA_ARCH_LIST')}, "
            f"MAX_JOBS={os.environ.get('MAX_JOBS')}"
        )
    return {
        "device_name": info["device_name"],
        "capability": info["capability"],
        "torch_version": info["torch_version"],
        "torch_cuda": info["torch_cuda"],
        "max_jobs": os.environ.get("MAX_JOBS"),
    }


def run_command(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    """Run a command and capture combined stdout/stderr for diagnostics."""
    proc = subprocess.run(
        cmd,
        cwd=cwd or REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout.strip()


def cutlass_header() -> Path:
    """Return the canonical CUTLASS header path used as a readiness check."""
    return REPO_ROOT / "third-party" / "cutlass" / "include" / "cute" / "tensor.hpp"


def cutlass_ready() -> bool:
    """Report whether the required CUTLASS header is present."""
    return cutlass_header().exists()
