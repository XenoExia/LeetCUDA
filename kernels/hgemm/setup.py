import os
import subprocess
from pathlib import Path

import torch
from setuptools import find_packages, setup
from tools.utils import get_build_cuda_cflags, get_build_sources
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

# package name managed by pip, which can be remove by `pip uninstall toy-hgemm`
PACKAGE_NAME = "toy-hgemm"


def parse_version(version: str) -> tuple[int, ...]:
    numbers: list[int] = []
    for part in version.replace("+", ".").split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if digits:
            numbers.append(int(digits))
    return tuple(numbers)


def version_gte(found: str, expected: str) -> bool:
    return parse_version(found) >= parse_version(expected)


def add_codegen_flag(flags: list[str], arch_code: str) -> None:
    flags.append("-gencode")
    flags.append(f"arch=compute_{arch_code},code=sm_{arch_code}")


def get_cuda_bare_metal_version(cuda_dir: str) -> tuple[str, str]:
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = output[release_idx].split(",")[0]
    return raw_output, bare_metal_version


def get_local_arch_code() -> str | None:
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(0)
    return f"{major}{minor}"


def get_default_arch_codes() -> list[str]:
    arch_codes: list[str] = ["80", "89"]
    if CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        if version_gte(bare_metal_version, "11.8"):
            arch_codes.append("90")
    return arch_codes


def parse_arch_codes(raw_value: str) -> list[str]:
    arch_codes: list[str] = []
    for part in raw_value.replace(";", ",").split(","):
        digits = "".join(ch for ch in part if ch.isdigit())
        if digits:
            arch_codes.append(digits)
    return arch_codes


def get_codegen_flags() -> list[str]:
    requested_arches = os.getenv("LEETCUDA_WHEEL_ARCHES", "").strip().lower()
    local_arch_code = get_local_arch_code()

    if requested_arches == "default":
        arch_codes = get_default_arch_codes()
        if local_arch_code is not None:
            arch_codes.append(local_arch_code)
    elif requested_arches:
        arch_codes = parse_arch_codes(requested_arches)
    elif local_arch_code is not None:
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

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

# cuda module
# may need export LD_LIBRARY_PATH=PATH-TO/torch/lib:$LD_LIBRARY_PATH
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
