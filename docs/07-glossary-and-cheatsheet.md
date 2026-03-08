# 07. Glossary and Cheatsheet

Use this page when you know a result is slow or wrong, but you do not yet know
which concept explains it.

## Hardware Terms

- `SM`: streaming multiprocessor, the main compute block on an NVIDIA GPU.
- `warp`: 32 threads that execute in lockstep under SIMT.
- `lane`: one thread inside a warp.
- `register`: the fastest private storage, but limited in quantity.
- `shared memory`: on-chip scratchpad shared by threads in a block.
- `global memory`: large off-chip DRAM with high latency.
- `Tensor Core`: specialized matrix-math unit for MMA-style instructions.

## Performance Terms

- `occupancy`: how many warps can be resident on an SM relative to the limit.
- `coalescing`: whether nearby threads access nearby global-memory addresses.
- `bank conflict`: multiple threads contending for the same shared-memory bank.
- `register pressure`: high register usage that reduces occupancy or causes spills.
- `latency hiding`: keeping enough independent warps ready so stalls are masked.
- `arithmetic intensity`: compute work divided by data movement.
- `roofline`: the idea that kernels are limited by either bandwidth or compute.

## Kernel Design Terms

- `tiling`: split a large problem into block-, warp-, and instruction-sized chunks.
- `double buffering`: load the next tile while computing on the current tile.
- `cp.async`: async copy path used to pipeline global-to-shared movement.
- `WMMA`: higher-level Tensor Core API.
- `MMA`: lower-level Tensor Core instruction interface with more control.
- `CuTe/CUTLASS`: template libraries for expressing tiled layouts and copies.
- `swizzle`: a layout trick to reduce shared-memory bank conflicts or improve locality.

## First Questions To Ask

When reading a kernel, answer these in order:

1. Which data does one thread own?
2. Which data does one warp own?
3. What gets reused from shared memory or registers?
4. Where are the synchronizations?
5. Is the kernel likely bandwidth-bound or compute-bound?

## Repo Command Cheatsheet

Environment and validation:

```bash
make bootstrap
make doctor
make smoke
```

Early learning labs:

```bash
make elementwise
make relu
python3 scripts/run_example.py kernels/reduce/block_all_reduce.py
python3 scripts/run_example.py kernels/softmax/softmax.py
```

GEMM and attention:

```bash
make hgemm-smoke
make flash-attn-smoke
make hgemm-wheel
```

Optional multi-arch packaging:

```bash
make hgemm-wheel-multi
```

## Profiling Cheatsheet

If Nsight tools are installed:

```bash
nsys profile --trace=cuda,nvtx python3 scripts/run_example.py kernels/hgemm/hgemm.py --M 256 --N 256 --K 256 --mma --no-default --iters 1 --warmup 0
ncu --set full python3 scripts/run_example.py kernels/relu/relu.py
```

Look for:

- register count
- shared-memory usage
- spill stores and spill loads
- DRAM throughput
- tensor core utilization
- warp stall reasons

## Study Milestones

- Beginner: explain indexing, coalescing, and reduction shape.
- Intermediate: explain shared-memory reuse, normalization, and softmax stability.
- Advanced: explain tiling, staging, Tensor Cores, and why attention inherits GEMM ideas.
