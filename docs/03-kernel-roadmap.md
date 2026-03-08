# 03. Kernel Roadmap

This is the recommended order for learning the repository.

## Stage 1: Vector Add and Unary Ops

Goal: understand indexing, contiguous memory access, and vectorized loads.

- `kernels/elementwise/`
- `kernels/relu/`
- `kernels/sigmoid/`
- `kernels/swish/`
- `kernels/elu/`

Suggested commands:

```bash
python3 scripts/run_example.py kernels/elementwise/elementwise.py
python3 scripts/run_example.py kernels/relu/relu.py
```

## Stage 2: Reductions and Normalization

Goal: learn warp collaboration, block collaboration, and accumulation patterns.

- `kernels/dot-product/`
- `kernels/reduce/`
- `kernels/layer-norm/`
- `kernels/rms-norm/`
- `kernels/softmax/`

What to focus on:

- warp shuffle versus shared memory
- reduction tree shape
- numerical stability
- accumulation precision

## Stage 3: Layout and Data Movement

Goal: learn why memory layout dominates performance.

- `kernels/mat-transpose/`
- `kernels/embedding/`
- `kernels/rope/`
- `kernels/swizzle/`

What to focus on:

- row-major versus column-major
- transpose cost
- shared-memory bank conflicts
- swizzle as a layout trick, not magic

## Stage 4: GEMV and GEMM

Goal: move from CUDA core kernels to Tensor Core kernels.

- `kernels/sgemv/`
- `kernels/hgemv/`
- `kernels/sgemm/`
- `kernels/hgemm/`
- `kernels/ws-hgemm/`
- `kernels/cutlass/`

This is where you learn:

- tiling
- double buffering
- cp.async
- WMMA and MMA
- Tensor Core-friendly data layout
- warp specialization
- CuTe layout algebra

## Stage 5: Attention and AI Infra Kernels

Goal: connect the hardware model to real transformer kernels.

- `kernels/flash-attn/`
- `kernels/transformer/`
- `kernels/openai-triton/`
- `others/tensorrt/`
- `others/pytorch/distributed/`

By this point, you should be able to answer:

- which part of attention is bandwidth-bound
- where SRAM pressure comes from
- why tiled Q/K/V movement matters
- why GEMM understanding is a prerequisite for attention understanding

## Rule For Advancing

Do not move to the next stage until you can explain:

- the thread-to-data mapping
- the dominant memory movement
- the likely bottleneck
- the reason the optimized version beats the naive one

## Stage 6: Profiling and Inspection Labs

Goal: learn how to inspect kernels instead of only running them.

- `kernels/nvidia-nsight/`
- `slides/cuda-slides/`

What to focus on:

- PTX versus SASS
- line info and source correlation
- bank conflicts
- timeline versus kernel-counter profiling
