# 00. Getting Started

This repository now has one recommended startup flow for modern NVIDIA GPUs,
including Blackwell-class devices such as RTX 5090.

## One-Time Setup

```bash
bash scripts/bootstrap_env.sh
python3 scripts/doctor.py
```

If you do not want bootstrap to spend time compiling the official
`flash-attn` package on a low-memory machine:

```bash
SKIP_FLASH_ATTN_INSTALL=1 bash scripts/bootstrap_env.sh
```

What the bootstrap script does:

- Initializes git submodules such as `third-party/cutlass`.
- Installs baseline Python packages used by the examples.
- Tries to install `flash-attn` for comparison benchmarks.
- Prints a diagnostic summary at the end.

## Official Run Entry

Prefer the wrapper for Python examples:

```bash
python3 scripts/run_example.py kernels/elementwise/elementwise.py
python3 scripts/run_example.py kernels/reduce/block_all_reduce.py
python3 scripts/run_example.py kernels/hgemm/hgemm.py --M 256 --N 256 --K 256 --mma --no-default --iters 1 --warmup 0
```

The wrapper does two useful things for you:

- Detects the current GPU and sets `TORCH_CUDA_ARCH_LIST` if it is missing.
- Sets a safer `MAX_JOBS` for large CUDA extension builds.

This matters on machines with limited host RAM, because compiling many heavy
Tensor Core kernels in parallel can cause the build to be killed.

## Suggested First Verification

```bash
make smoke
```

If this passes, your basic CUDA extension toolchain, JIT path, and teaching
subset builds are healthy.

## Native Packaging Check

If you want to verify the local wheel packaging path for HGEMM:

```bash
make hgemm-wheel
```

This now builds only for the current GPU architecture by default. If you need
multi-arch packaging for distribution, use `make hgemm-wheel-multi`.

## Common Failure Modes

- `cute/tensor.hpp: No such file or directory`
  Cause: CUTLASS submodule is missing.
  Fix: `git submodule update --init --recursive --force`

- `ModuleNotFoundError: No module named 'flash_attn'`
  Cause: official comparison package is missing.
  Fix: `python3 -m pip install flash-attn --no-build-isolation`

- `Killed` or exit code `137` during build
  Cause: too many parallel `nvcc` jobs for available host memory.
  Fix: run with `MAX_JOBS=1`.

## Learning Advice

Do not start from HGEMM or FlashAttention.
You will move faster if you first internalize:

1. thread indexing
2. memory coalescing
3. reductions
4. normalization and softmax
5. matrix multiply tiling

The rest of the repository becomes much easier once these patterns are familiar.
