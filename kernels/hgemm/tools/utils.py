"""Runtime helpers for the HGEMM teaching scripts and wheel build."""

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def get_device_name():
    """Return the active CUDA device name with the repo's WSL naming tweak."""
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    # since we will run GPU on WSL2, so add WSL2 tag.
    if "Laptop" in device_name:
        device_name += " WSL2"
    return device_name


def get_device_capability():
    """Return the active CUDA capability tuple."""
    return torch.cuda.get_device_capability(torch.cuda.current_device())


def get_build_sources():
    """Return the source set used by both JIT and package builds."""
    build_sources = []
    build_sources.append("naive/hgemm.cu")
    build_sources.append("naive/hgemm_async.cu")
    build_sources.append("cublas/hgemm_cublas.cu")
    build_sources.append("wmma/hgemm_wmma.cu")
    build_sources.append("wmma/hgemm_wmma_stage.cu")
    build_sources.append("mma/basic/hgemm_mma.cu")
    build_sources.append("mma/basic/hgemm_mma_stage.cu")
    build_sources.append("mma/basic/hgemm_mma_stage_tn.cu")
    build_sources.append("mma/swizzle/hgemm_mma_stage_swizzle.cu")
    build_sources.append("mma/swizzle/hgemm_mma_stage_tn_swizzle_x4.cu")
    build_sources.append("cutlass/hgemm_mma_stage_tn_cute.cu")
    build_sources.append("pybind/hgemm.cc")
    return build_sources


def get_project_dir():
    """Return the repository root from the nested helper directory."""
    return os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    )


def get_total_memory_gib():
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


def configure_build_environment():
    """Populate arch and MAX_JOBS defaults for local CUDA extension builds."""
    if torch.cuda.is_available() and "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    if "MAX_JOBS" not in os.environ:
        total_memory_gib = get_total_memory_gib()
        if total_memory_gib is None or total_memory_gib < 32:
            max_jobs = 1
        elif total_memory_gib < 64:
            max_jobs = 2
        else:
            max_jobs = min(os.cpu_count() or 1, 4)
        os.environ["MAX_JOBS"] = str(max_jobs)
    return os.environ.get("TORCH_CUDA_ARCH_LIST"), os.environ.get("MAX_JOBS")


def ensure_cutlass_submodule():
    """Fail early when the CUTLASS submodule has not been initialized."""
    header = (
        Path(get_project_dir())
        / "third-party"
        / "cutlass"
        / "include"
        / "cute"
        / "tensor.hpp"
    )
    if not header.exists():
        raise FileNotFoundError(
            "CUTLASS submodule is missing. Run "
            "`git submodule update --init --recursive --force` "
            "from the repository root first."
        )


def get_build_cuda_cflags(build_pkg: bool = False):
    # Keep ptxas verbosity enabled for local builds so benchmark runs expose
    # register usage, shared-memory usage, constant-memory usage, and spills.
    extra_cuda_cflags = []
    extra_cuda_cflags.append("-O3")
    extra_cuda_cflags.append("-std=c++17")
    extra_cuda_cflags.append("-U__CUDA_NO_HALF_OPERATORS__")
    extra_cuda_cflags.append("-U__CUDA_NO_HALF_CONVERSIONS__")
    extra_cuda_cflags.append("-U__CUDA_NO_HALF2_OPERATORS__")
    extra_cuda_cflags.append("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
    extra_cuda_cflags.append("--expt-relaxed-constexpr")
    extra_cuda_cflags.append("--expt-extended-lambda")
    extra_cuda_cflags.append("--use_fast_math")
    if not build_pkg:
        extra_cuda_cflags.append("-diag-suppress 177")
        extra_cuda_cflags.append("-Xptxas -v")
    else:
        extra_cuda_cflags.append("--ptxas-options=-v")
        extra_cuda_cflags.append("--ptxas-options=-O3")
    # Package builds export only the pybind extension, not the standalone
    # benchmark binaries used by the teaching scripts.
    project_dir = get_project_dir()
    extra_cuda_cflags.append("-DNO_MMA_HGEMM_BIN")
    extra_cuda_cflags.append("-DNO_WMMA_HGEMM_BIN")
    extra_cuda_cflags.append("-DNO_CUTE_HGEMM_BIN")
    extra_cuda_cflags.append("-DNO_CUBLAS_HGEMM_BIN")
    # add cutlass headers and link cublas.
    extra_cuda_cflags.append(f"-I {project_dir}/kernels/hgemm")
    extra_cuda_cflags.append(f"-I {project_dir}/kernels/hgemm/utils")
    extra_cuda_cflags.append(f"-I {project_dir}/kernels/hgemm/naive")
    extra_cuda_cflags.append(f"-I {project_dir}/kernels/hgemm/wmma")
    extra_cuda_cflags.append(f"-I {project_dir}/kernels/hgemm/mma/basic")
    extra_cuda_cflags.append(f"-I {project_dir}/kernels/hgemm/mma/swizzle")
    extra_cuda_cflags.append(f"-I {project_dir}/kernels/hgemm/cutlass")
    extra_cuda_cflags.append(f"-I {project_dir}/kernels/hgemm/cublas")
    extra_cuda_cflags.append(f"-I {project_dir}/kernels/hgemm/pybind")
    extra_cuda_cflags.append(f"-I {project_dir}/third-party/cutlass/include")
    extra_cuda_cflags.append(
        f"-I {project_dir}/third-party/cutlass/tools/util/include"
    )
    extra_cuda_cflags.append("-lcublas")
    return extra_cuda_cflags


def pretty_print_line(m: str = "", sep: str = "-", width: int = 150):
    """Render the wide separators used by the benchmark scripts."""
    res_len = width - len(m)
    left_len = int(res_len / 2)
    right_len = res_len - left_len
    pretty_line = sep * left_len + m + sep * right_len
    print(pretty_line)


def build_from_sources(verbose: bool = False):
    """JIT-build the HGEMM extension directly from source."""
    ensure_cutlass_submodule()
    torch_arch_list_env, max_jobs_env = configure_build_environment()
    pretty_print_line(
        f"Loading hgemm lib on device: {get_device_name()}, "
        f"capability: {get_device_capability()}, "
        f"Arch ENV: {torch_arch_list_env}, "
        f"MAX_JOBS: {max_jobs_env}"
    )
    return load(
        name="hgemm_lib",
        sources=get_build_sources(),
        extra_cuda_cflags=get_build_cuda_cflags(),
        extra_cflags=["-std=c++17"],
        verbose=verbose,
    )


def try_load_hgemm_library(force_build: bool = False, verbose: bool = False):
    """Reuse an installed wheel when possible, otherwise fall back to JIT."""
    if not force_build:
        try:
            import toy_hgemm as hgemm

            pretty_print_line("Import toy-hgemm library done, use it!")
        except Exception:
            pretty_print_line(
                "Can't import toy-hgemm, force build "
                "from source or run <bash tools/install.sh>"
            )
            pretty_print_line(
                "Also may need export LD_LIBRARY_PATH="
                "PATH-TO/torch/lib:$LD_LIBRARY_PATH"
            )
            hgemm = build_from_sources(verbose=verbose)
    else:
        pretty_print_line("Force hgemm lib build from sources")
        hgemm = build_from_sources(verbose=verbose)

    return hgemm


@torch.no_grad
def as_col_major(x: torch.Tensor):
    # Reinterpret the tensor through a transpose/reshape pair so the final
    # contiguous storage follows column-major order.
    x_trans = x.t()
    x_col_major = x_trans.reshape(x.shape)
    return x_col_major.contiguous()  # must be a contiguous tensor
