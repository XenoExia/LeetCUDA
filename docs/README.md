# LeetCUDA Learning Path

LeetCUDA can now be used as a structured AI infra learning repo instead of only a kernel catalog.

## Language

- English tutorial: current page
- 中文教程: [zh/README.md](./zh/README.md)

## Quick Start

```bash
make bootstrap
make doctor
make elementwise
make smoke
```

Use the wrapper when you want modern GPU defaults such as auto-detected
`TORCH_CUDA_ARCH_LIST` and a conservative `MAX_JOBS` for heavy CUDA builds:

```bash
python3 scripts/run_example.py kernels/relu/relu.py
python3 scripts/run_example.py kernels/hgemm/hgemm.py --M 256 --N 256 --K 256 --mma --no-default --iters 1 --warmup 0
```

## Study Order

1. [00-getting-started.md](./00-getting-started.md)
2. [01-cpu-gpu-architecture.md](./01-cpu-gpu-architecture.md)
3. [02-cuda-programming-model.md](./02-cuda-programming-model.md)
4. [03-kernel-roadmap.md](./03-kernel-roadmap.md)
5. [04-gemm-and-attention.md](./04-gemm-and-attention.md)
6. [05-debugging-and-profiling.md](./05-debugging-and-profiling.md)
7. [06-exercises.md](./06-exercises.md)
8. [07-glossary-and-cheatsheet.md](./07-glossary-and-cheatsheet.md)
9. [08-beyond-kernels.md](./08-beyond-kernels.md)
10. [09-repo-atlas.md](./09-repo-atlas.md)
11. [10-reading-list.md](./10-reading-list.md)

## What You Will Learn

- CPU and GPU hardware organization, and why GPUs win on throughput workloads.
- CUDA's execution model: grid, block, warp, lane, occupancy, memory hierarchy.
- How to read and write progressively harder CUDA kernels.
- How GEMV/GEMM and attention kernels evolve from naive code to Tensor Core code.
- How to validate correctness, benchmark performance, and debug bottlenecks.

## Recommended Rhythm

- Start each chapter by reading the markdown file.
- Run the matching example with `scripts/run_example.py`.
- Write down one correctness observation and one performance observation.
- Do the exercise before moving to the next level.

## Core Repo Areas

- `kernels/`: hands-on labs from elementwise ops to FlashAttention.
- `others/`: PyTorch distributed and TensorRT side topics.
- `slides/`: supplementary study material.
- `scripts/`: environment bootstrap, doctor, and example runner.

## Executable Lab Map

- `00 Getting Started`: `make bootstrap`, `make doctor`, `make smoke`
- `01 CPU and GPU Architecture`: read first, then revisit one kernel from `kernels/elementwise/`
- `02 CUDA Programming Model`: run `make elementwise` and `make relu`
- `03 Kernel Roadmap`: move into `kernels/reduce/`, `kernels/softmax/`, and `kernels/mat-transpose/`
- `04 GEMM and Attention`: run `make hgemm-smoke`, then `make flash-attn-smoke`
- `05 Debugging and Profiling`: profile one simple kernel and one GEMM kernel
- `06 Exercises`: turn one chapter into a written performance review
- `07 Glossary and Cheatsheet`: use as your lookup page while reading or profiling
- `08 Beyond Kernels`: connect raw CUDA ideas to Triton, distributed, and TensorRT workflows
