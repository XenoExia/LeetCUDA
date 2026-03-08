"""Build configuration for the optional toy_hgemm wheel."""

import os
import subprocess
from pathlib import Path

import torch
from setuptools import find_packages, setup
from tools.utils import get_build_cuda_cflags, get_build_sources
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

# Package name managed by pip, so `pip uninstall toy-hgemm` removes it cleanly.
PACKAGE_NAME = "toy-hgemm"


def parse_version(version: str) -> tuple[int, ...]:
    """Extract numeric version components from nvcc or torch version strings."""
    numbers: list[int] = []
    for part in version.replace("+", ".").split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if digits:
            numbers.append(int(digits))
    return tuple(numbers)


def version_gte(found: str, expected: str) -> bool:
    """Compare loose numeric version tuples."""
    return parse_version(found) >= parse_version(expected)


def add_codegen_flag(flags: list[str], arch_code: str) -> None:
    """Append one nvcc gencode pair for a concrete SM architecture."""
    flags.append("-gencode")
    flags.append(f"arch=compute_{arch_code},code=sm_{arch_code}")


def get_cuda_bare_metal_version(cuda_dir: str) -> tuple[str, str]:
    """Read the installed nvcc version from the CUDA toolkit."""
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = output[release_idx].split(",")[0]
    return raw_output, bare_metal_version


def get_local_arch_code() -> str | None:
    """Return the local GPU arch like '120' when CUDA is available."""
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(0)
    return f"{major}{minor}"


def get_default_arch_codes() -> list[str]:
    """Return the conservative multi-arch wheel defaults used for distribution."""
    arch_codes: list[str] = ["80", "89"]
    if CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        if version_gte(bare_metal_version, "11.8"):
            arch_codes.append("90")
    return arch_codes


def parse_arch_codes(raw_value: str) -> list[str]:
    """Parse a comma- or semicolon-separated arch list like '89,90,120'."""
    arch_codes: list[str] = []
    for part in raw_value.replace(";", ",").split(","):
        digits = "".join(ch for ch in part if ch.isdigit())
        if digits:
            arch_codes.append(digits)
    return arch_codes


def get_codegen_flags() -> list[str]:
    """Select native or explicit wheel architectures for the current build."""
    requested_arches = os.getenv("LEETCUDA_WHEEL_ARCHES", "").strip().lower()
    local_arch_code = get_local_arch_code()

    if requested_arches == "default":
        # Distribution builds keep the historical default set and add the local
        # GPU when one is present.
        arch_codes = get_default_arch_codes()
        if local_arch_code is not None:
            arch_codes.append(local_arch_code)
    elif requested_arches:
        arch_codes = parse_arch_codes(requested_arches)
    elif local_arch_code is not None:
        # Native-only wheels are much cheaper to build on memory-constrained
        # development machines.
        arch_codes = [local_arch_code]
    else:
        arch_codes = get_default_arch_codes()

    if not arch_codes:
        raise RuntimeError(
            "No CUDA arch codes selected. Set LEETCUDA_WHEEL_ARCHES "
            "to a list like '120' or '89,90,120'."
        )

    deduped_flags: list[str] = []
    seen: set[str] = set()
    for arch_code in arch_codes:
        if arch_code in seen:
            continue
        seen.add(arch_code)
        add_codegen_flag(deduped_flags, arch_code)
    return deduped_flags


ext_modules = []
generator_flag: list[str] = []
cc_flag = get_codegen_flags()

# Ninja expects absolute include paths for reliable extension builds.
this_dir = os.path.dirname(os.path.abspath(__file__))

# The wheel exposes only the Python extension. Benchmark binaries stay out of
# the package build to keep installation predictable.
ext_modules.append(
    CUDAExtension(
        # package name for import
        name="toy_hgemm",
        sources=get_build_sources(),
        extra_compile_args={
            # add c compile flags
            "cxx": ["-O3", "-std=c++17"] + generator_flag,
            # add nvcc compile flags
            "nvcc": get_build_cuda_cflags(build_pkg=True)
            + generator_flag
            + cc_flag,
        },
        include_dirs=[
            Path(this_dir) / "naive",
            Path(this_dir) / "utils",
            Path(this_dir) / "wmma",
            Path(this_dir) / "mma",
            Path(this_dir) / "cutlass",
            Path(this_dir) / "cublas",
            Path(this_dir) / "pybind",
        ],
    )
)

setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(
        exclude=(
            "build",
            "naive",
            "wmma",
            "mma",
            "cutlass",
            "cublas",
            "utils",
            "bench",
            "pybind",
            "tmp",
        )
    ),
    description="My Toy HGEMM implement by CUDA",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "ninja",
    ],
)
