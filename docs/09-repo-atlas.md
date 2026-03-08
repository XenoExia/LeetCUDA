# 09. Repo Atlas

This page exists so that the whole repository is part of the tutorial, not only
the files already mentioned in the early chapters.

## Core Kernel Labs

These are the main hands-on directories under `kernels/`:

- `elementwise`, `relu`, `sigmoid`, `swish`, `elu`
  Focus: indexing, vectorization, and memory coalescing.
- `dot-product`, `reduce`, `softmax`, `layer-norm`, `rms-norm`
  Focus: reduction patterns, stability, and accumulation.
- `mat-transpose`, `embedding`, `rope`, `swizzle`, `histogram`, `nms`
  Focus: data layout, memory traffic, and irregular access.
- `sgemv`, `hgemv`, `sgemm`, `hgemm`, `ws-hgemm`
  Focus: GEMV/GEMM, Tensor Cores, pipelining, warp specialization.
- `flash-attn`
  Focus: transformer attention kernels and SRAM tradeoffs.

## Advanced CUDA Side Tracks

These directories are not the first stop, but they are part of the course once
you understand the main kernel ladder.

- `kernels/cutlass/cute/`
  Small CuTe-focused experiments such as `vector_add.cu` and `mma_tile_tex.cc`.
- `kernels/openai-triton/`
  Higher-level GPU programming track via Triton.
- `kernels/nvidia-nsight/`
  Profiling and PTX/SASS inspection sandbox.
- `kernels/ws-hgemm/`
  Warp-specialization HGEMM experiments.

## System-Level Labs

These live outside `kernels/`, but they matter for infra work.

- `others/pytorch/distributed/`
  Collective communication and multi-GPU coordination.
- `others/tensorrt/`
  Deployment and TensorRT integration experiments.
- `others/pytorch/custom_ops/`
  Currently a placeholder area for future custom-op examples.

## Companion Repos

These are checked out in the same workspace as submodules.

- `HGEMM/`
  Standalone HGEMM-focused repo derived from this project.
- `ffpa-attn/`
  Follow-up repo for large-headdim Flash Prefill Attention.
- `third-party/cutlass/`
  Upstream CUTLASS source used by multiple learning tracks.

## Reference Material

These are not executable labs, but they are part of the learning path.

- `slides/cuda-slides/`
  CUDA, CUTLASS, TensorRT, NCCL, PTX ISA, architecture whitepapers.
- `slides/vllm-slides/`
  vLLM, inference serving, and system design references.
- `others/pytorch/slides/`
  supplementary PyTorch slides.

## Reserved Or Sparse Areas

Some directories are intentionally light right now, but still part of the repo map.

- `kernels/transformer/`
  reserved transformer-oriented space for future kernel or system examples.
- `kernels/openai-triton/fused-attention/`
  currently lighter than the other Triton subdirectories.
- `others/pytorch/custom_ops/`
  reserved space for future operator-extension material.

## Suggested Tour After Finishing The Main Course

1. Revisit `kernels/nvidia-nsight/`
2. Study `kernels/ws-hgemm/`
3. Study `kernels/cutlass/cute/`
4. Study `kernels/openai-triton/`
5. Study `others/pytorch/distributed/`
6. Study `others/tensorrt/`
7. Compare with `HGEMM/` and `ffpa-attn/`
